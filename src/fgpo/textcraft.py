"""Public adapter for the bundled BiPACE TextCraft simulator and split files."""
from __future__ import annotations

import json
from pathlib import Path
import random
import re
import sys


TEXTCRAFT_SOURCE_COMMIT = "f19f6fe1d140dd342d81b8e797f09c37c620b42f"
TEXTCRAFT_COUNTS = {"train": 318, "val": 79}
TEXTCRAFT_HORIZON = 30
TEXTCRAFT_OUTPUT_TOKENS = 128
TEXTCRAFT_CONTEXT_TOKENS = 8192
TEXTCRAFT_TRAIN_ROUNDS = 50
TEXTCRAFT_TRAIN_GROUPS = 16
TEXTCRAFT_1P5B_FORMAT_STUDY = "textcraft-1p5b-joint-format-from-base-v1"
TEXTCRAFT_FORMAT_EXAMPLE = (
    "Example response format: <think>I will check the inventory.</think><action>inventory</action>"
)
TEXTCRAFT_SYSTEM_PROMPT = (
    "You are an expert agent in TextCraft. Craft the goal item using the listed recipes. "
    "Use one command each step: get <count> <raw item>, "
    "craft <out count> <out item> using <count> <ingredient>, ..., or inventory. "
    "Spell item names in lowercase with spaces and match the listed recipe and quantities exactly. "
    "Use a specific variant for a generic ingredient, such as oak planks for planks. "
    "Fetch raw ingredients and craft intermediates as needed. "
    "Read the latest simulator feedback and inventory. "
    "Reason briefly in <think>...</think> and give one command in <action>...</action>."
)
_ACTION_TAG = re.compile(r"</?action\b")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def textcraft_action(text: str) -> tuple[str, bool]:
    """BiPACE command extraction and validity, with native rejection of ambiguous tags."""
    lower = text.lower()
    tags = tuple(_ACTION_TAG.finditer(lower))
    if len(tags) != 2:
        return "", False
    start, end = (tag.start() for tag in tags)
    if not lower.startswith("<action>", start) or not lower.startswith("</action>", end):
        return "", False
    command = lower[start + len("<action>"):end].strip()
    if not command or "\n" in command or "\r" in command:
        return "", False
    valid = "<think>" in text and "</think>" in text and _CJK.search(text) is None
    return command, valid


def _observation(feedback: str, inventory: str) -> str:
    return json.dumps({"feedback": feedback, "inventory": inventory}, ensure_ascii=False, separators=(",", ":"))


def textcraft_observation(raw: str) -> str:
    """Serialize the exact two public inputs identically for every study arm."""
    fields = json.loads(raw)
    return _observation(fields["feedback"], fields["inventory"])


class TextCraftEnv:
    benchmark = "textcraft"

    def __init__(self, data_root, *, run_dir, format_example=False):
        root = Path(data_root).resolve()
        sys.path.insert(0, str(root / "vendor/agent_system/environments/env_package"))
        from textcraft.envs import TextCraftWorker, _load_goal_index, _load_split

        self.worker = TextCraftWorker
        index = _load_goal_index()
        self.catalog = {}
        self.tasks = {}
        for split, count in TEXTCRAFT_COUNTS.items():
            goals = _load_split(split == "train")
            rows = tuple({"task_id": goal, "goal_item": goal, "depth": index[goal], "split": split}
                         for goal in goals)
            if len(rows) != count or len(set(goals)) != count or set(goals) & self.tasks.keys():
                raise ValueError(f"Invalid official TextCraft {split} catalog")
            self.catalog[split] = rows
            self.tasks.update((row["task_id"], row) for row in rows)
        self.world = self.task = self.info = None
        self.done = False
        prompt = TEXTCRAFT_SYSTEM_PROMPT
        if format_example:
            prompt += "\n" + TEXTCRAFT_FORMAT_EXAMPLE
        self.action_schema = {"benchmark": self.benchmark, "native_system_prompt": prompt}

    def list_tasks(self, split):
        return self.catalog[split]

    def reset(self, task_id, initial_seed):
        selected = self.tasks[task_id]
        random.seed(initial_seed)
        if self.world is None:
            self.world = self.worker(initial_seed, {"max_steps": TEXTCRAFT_HORIZON, "reward_scale": 10.})
        # The worker uses both Python's global RNG and its private recipe RNG.
        # Every lease restarts those streams for the original group seed.
        self.world._rng.seed(initial_seed)
        feedback, self.info = self.world.reset(selected["goal_item"], selected["depth"])
        self.task, self.done = selected, False
        return {**selected, "benchmark": self.benchmark, "instruction": feedback.rsplit("\n\n", 1)[-1],
                "observation": _observation(feedback, self.info["inventory"]),
                "action_schema": dict(self.action_schema)}

    def step_native(self, action):
        from .envs import StepResult

        if self.task is None or self.done:
            raise RuntimeError("Native TextCraft step requires a live reset episode")
        if action.get("kind") not in ("native_text", "invalid"):
            raise ValueError("TextCraft requires the native_text action mapping")
        feedback, reward, self.done, self.info = self.world.step(action["text"])
        won = self.info["won"]
        return StepResult(_observation(feedback, self.info["inventory"]), self.done, won,
                          {**self.info, "native_executed": True}, native_reward=float(reward))

    def finalize_evaluation(self):
        if self.task is None:
            raise RuntimeError("Native TextCraft scoring requires a reset episode")
        return {**self.task, "benchmark": self.benchmark, **self.info, "success": self.info["won"],
                "steps": self.info["step_count"], "native_evaluator": "BiPACE.TextCraftWorker.info.won"}

    def close(self):
        if self.world is not None:
            # The native close leaves the immutable recipe tree reusable.
            self.world.close()
        self.task = self.info = None
        self.done = False
