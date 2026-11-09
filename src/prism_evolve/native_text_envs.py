"""Official text-game bridges; private goals never enter actor observations.

The registered I/E partition is an information-intent partition. Navigation
and opening containers retain their real state changes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from functools import partial
import hashlib
import importlib
from importlib import metadata as package_metadata
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from .native_core import BudgetExhausted, BudgetLimits, BudgetMeter, NativeStep, monotonic_seconds


_PUBLIC_STATE = "classic_text_action_state"
_OBSERVATION_TYPE = "native_text_transition"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(_plain(value)).encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(child) for child in value), key=_json)
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def _version(package: str) -> str | None:
    try:
        return package_metadata.version(package)
    except package_metadata.PackageNotFoundError:
        return None


def _visible_state(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("feedback_type") == _OBSERVATION_TYPE:
        state = value.get(_PUBLIC_STATE)
        return state if isinstance(state, Mapping) else None
    # initial_messages wraps the reset observation without changing its content.
    if value.get("feedback_type") == "task":
        observation = value.get("observation")
        if isinstance(observation, str):
            try:
                observation = json.loads(observation)
            except ValueError:
                return None
        return _visible_state(observation)
    return None


def public_text_action_schema(
    action_schema: Mapping[str, Any], messages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild action support from visible feedback, including masked histories."""
    if action_schema.get("static_native_tools") is True:
        # ClinDiag's public JSON request vocabulary is static across observations.
        return deepcopy(dict(action_schema))
    schema = {key: deepcopy(value) for key, value in action_schema.items()
              if key not in {"mode_actions", "search_available", "available_actions",
                             "probe_actions", "feasible_modes", "public_state_id",
                             "native_action_schemas_by_mode", "open_text_modes", "native_tools",
                             "probe_tool_names"}}
    state: Mapping[str, Any] | None = None
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            candidate = _visible_state(json.loads(content))
        except ValueError:
            continue
        if candidate is not None:
            if candidate.get("benchmark") != schema.get("benchmark"):
                raise ValueError("Visible action state belongs to another benchmark.")
            state = candidate
    if state is None:
        schema.update(mode_actions={"I": [], "E": []}, search_available=False,
                      available_actions=[], probe_actions=[], feasible_modes=[],
                      probe_tool_names=[], open_text_modes={"I": [], "E": []},
                      native_action_schemas_by_mode={}, native_tools=[])
    else:
        schema.update(deepcopy(dict(state)))
    return schema


def classify_alfworld_action(action: str) -> str:
    """Registered scheme A, applied before execution and without goal labels."""
    text = action.strip().lower()
    if (text in {"look", "inventory", "help"}
            or text.startswith(("look at ", "examine ", "go to ", "open "))):
        return "I"
    if (text in {"pass", "wait"}
            or text.startswith(("take ", "pick up ", "move ", "put ", "clean ",
                                "heat ", "cool ", "use ", "slice ", "close ", "toggle "))):
        return "E"
    raise ValueError(f"Unregistered official ALFWorld command: {action!r}")


class _NativeTextEnv:
    benchmark: str

    def __init__(self, *, output_dir: str | Path | None, max_steps: int,
                 meter: BudgetMeter | None, interface: str) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self.max_steps = int(max_steps)
        self.meter = meter
        self.replayed_env_steps = 0
        self.snapshot_restore_seconds = 0.0
        self.last_native_result: dict[str, Any] | None = None
        self.action_schema: dict[str, Any] = {
            "benchmark": self.benchmark, "classic_text": True,
            "native_interface": interface, "native_system_prompt": interface,
            "native_format": "One native text command; no JSON program or command rewriting.",
            "information_semantics": "information_intent_with_real_state_changes",
            "mode_actions": {"I": [], "E": []}, "search_available": False,
            "available_actions": [], "probe_actions": [], "feasible_modes": [],
        }
        self._task_id = ""
        self._seed = 0
        self._attempts = 0
        self._done = False
        self._won = False
        self._submitted = False
        self._native_score = 0.0
        self._actions: list[dict[str, Any]] = []
        self._observations: list[str] = []
        self._last_observation = ""
        self._python_rng = random.Random(0).getstate()
        self._numpy_rng: Any = None

    def _seed_rng(self, seed: int) -> None:
        import numpy as np
        self._python_rng = random.Random(int(seed)).getstate()
        self._numpy_rng = np.random.RandomState(int(seed) % (2 ** 32)).get_state()

    @contextmanager
    def _rng(self) -> Iterator[None]:
        """Keep native randomness independent of actor sampling and other cells."""
        import numpy as np
        outer_py, outer_np = random.getstate(), np.random.get_state()
        torch = sys.modules.get("torch")
        cpu_rng = torch.random.get_rng_state() if torch is not None else None
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch is not None and torch.cuda.is_initialized() else None)
        random.setstate(self._python_rng)
        np.random.set_state(self._numpy_rng)
        try:
            yield
        finally:
            self._python_rng, self._numpy_rng = random.getstate(), np.random.get_state()
            random.setstate(outer_py)
            np.random.set_state(outer_np)
            if cpu_rng is not None:
                torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def schema_for_history(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return public_text_action_schema(self.action_schema, messages)

    def _publish(self, observation: str, modes: Mapping[str, Sequence[str]], *,
                 search_available: bool = False, probe_actions: Sequence[str] = ()) -> str:
        grouped = {mode: list(dict.fromkeys(modes.get(mode, ()))) for mode in ("I", "E")}
        if set(grouped["I"]) & set(grouped["E"]):
            raise ValueError("Native action modes must be disjoint.")
        if self._done:
            grouped, search_available, probe_actions = {"I": [], "E": []}, False, ()
        state = {
            "benchmark": self.benchmark, "mode_actions": grouped,
            "search_available": bool(search_available),
            "available_actions": grouped["I"] + grouped["E"],
            "probe_actions": list(probe_actions),
            "feasible_modes": [mode for mode in ("I", "E")
                               if grouped[mode] or (mode == "I" and search_available)],
        }
        state["public_state_id"] = _digest({"observation": observation, **state})
        self.action_schema.update(deepcopy(state))
        self.action_schema["exact_snapshot_available"] = True
        self._last_observation = _json({"feedback_type": _OBSERVATION_TYPE,
                                       "native_observation": observation,
                                       _PUBLIC_STATE: state})
        return self._last_observation

    def _context(self, instruction: str) -> dict[str, Any]:
        row = self._task_row(self._task_id)
        observation = json.loads(self._last_observation)
        return {"benchmark": self.benchmark, "task_id": self._task_id,
                "scenario_id": row["scenario_id"], "instruction": instruction,
                "system_prompt": self.action_schema["native_system_prompt"],
                "observation": self._last_observation,
                "action_schema": deepcopy(self.action_schema),
                "initial_information_signature": self._information_signature(observation)}

    @staticmethod
    def _information_signature(observation: Mapping[str, Any]) -> str:
        state = observation[_PUBLIC_STATE]
        return _json({"observation": observation["native_observation"],
                      "available_actions": state["available_actions"],
                      "search_available": state["search_available"]})

    def _begin(self, task_id: str, seed: int) -> None:
        self._task_row(task_id)
        self._task_id, self._seed = str(task_id), int(seed)
        self._attempts = 0
        self._done = self._won = self._submitted = False
        self._native_score = 0.0
        self._actions, self._observations = [], []
        self.last_native_result = None
        self._seed_rng(seed)

    def _action_text(self, action: Mapping[str, Any]) -> str:
        if action.get("kind") == "native_text" and isinstance(action.get("text"), str):
            return str(action["text"])
        if action.get("kind") == "invalid":
            return str(action.get("raw_text", action.get("text", "")))
        raise ValueError("The native text interface requires kind=native_text and a text string.")

    def _mode(self, text: str) -> str | None:
        for mode, actions in self.action_schema["mode_actions"].items():
            if text in actions:
                return mode
        return None

    def _comparison_action(self, text: str) -> str:
        return text

    def _step_unmetered(self, action: Mapping[str, Any], *, probe: bool = False) -> NativeStep:
        if not self._task_id or self._done:
            raise RuntimeError("Native text actions require a live reset episode.")
        text = self._action_text(action)
        mode = self._mode(text)
        requested = action.get("mode")
        if requested is not None and requested not in {"I", "E"}:
            raise ValueError("Unknown requested action mode.")
        mode_blocked = requested is not None and mode is not None and requested != mode
        probe_blocked = probe and self._comparison_action(text) not in self.action_schema["probe_actions"]
        self._attempts += 1
        before = self._world_state()
        if mode_blocked or probe_blocked:
            failed, executed = True, False
            reason = "mode_guard" if mode_blocked else "probe_guard"
            observation = f"Native interface rejected this action ({reason}). No environment command was executed."
            modes = self.action_schema["mode_actions"]
            search = self.action_schema["search_available"]
            probes = self.action_schema["probe_actions"]
        else:
            with self._rng():
                observation, native_done, won, score, failed = self._native_step(text)
            self._won = bool(won)
            self._done = bool(native_done)
            self._native_score = float(score)
            if not math.isfinite(self._native_score) or not 0 <= self._native_score <= 1:
                raise ValueError("The official normalized score must be finite and in [0, 1].")
            self._submitted = bool(native_done)
            executed = True
            modes, search, probes = self._action_state() if not self._done else ({"I": [], "E": []}, False, [])
        if self._attempts >= self.max_steps:
            self._done = True
        changed = before != self._world_state()
        public = self._publish(observation, modes, search_available=search, probe_actions=probes)
        payload = json.loads(public)
        information = bool(executed and not failed and mode == "I")
        metadata = {
            "classic_text": True, "native_action": text, "action_mode": mode,
            "native_executed": executed, "native_execution_succeeded": executed and not failed,
            "information_action_executed": executed and mode == "I",
            "step_failed": bool(failed), "mode_guard_blocked": bool(mode_blocked),
            "probe_guard_blocked": bool(probe_blocked), "submitted": self._submitted,
            "native_score": self._native_score if self._submitted else None,
            "native_success": self._won if self._done else None,
            "termination": ("native_terminal" if self._submitted else
                            "environment_step_limit" if self._done else "running"),
            "receipts": [],
            "native_transition": {
                "state_changed": changed, "task_progress": self._won,
                "task_progress_source": "official_terminal_success",
                "new_information_candidate": information,
                "public_information_evidence": observation if information else None,
                "public_information_signature": self._information_signature(payload),
                "hard_bad": bool(failed), "final_success": self._won,
            },
        }
        self._actions.append({"action": deepcopy(dict(action)), "probe": bool(probe)})
        self._observations.append(public)
        return NativeStep(public, self._done, self._won, metadata)

    def step(self, action: Mapping[str, Any]) -> NativeStep:
        return self.step_native(action)

    def step_native(self, action: Mapping[str, Any], *, probe: bool = False) -> NativeStep:
        if self.meter is not None:
            self.meter.ensure_env_steps(1)
        started = monotonic_seconds()
        try:
            result = self._step_unmetered(action, probe=probe)
            return NativeStep(result.observation, result.done, result.success,
                              {**result.metadata, "environment_seconds": monotonic_seconds() - started})
        finally:
            if self.meter is not None:
                self.meter.record_env_step(monotonic_seconds() - started)

    def _snapshot_state(self) -> dict[str, Any]:
        return _plain({"state": self._private_state(), "python_rng": self._python_rng,
                       "numpy_rng": self._numpy_rng, "attempts": self._attempts,
                       "native_score": self._native_score, "won": self._won,
                       "submitted": self._submitted, "done": self._done})

    def save_snapshot(self) -> dict[str, Any]:
        if not self._task_id or self._done:
            raise RuntimeError("Snapshots require a live nonterminal native episode.")
        started = monotonic_seconds()
        snapshot = {"benchmark": self.benchmark, "task_id": self._task_id, "seed": self._seed,
                    "actions": deepcopy(self._actions), "observations": list(self._observations),
                    "native_state": self._snapshot_state(),
                    "public_observation": self._last_observation}
        if self.meter is not None:
            self.meter.record_snapshot_save(monotonic_seconds() - started)
        return snapshot

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("benchmark") != self.benchmark:
            raise ValueError("Snapshot belongs to another native benchmark.")
        actions, observations = list(snapshot["actions"]), list(snapshot["observations"])
        if len(actions) != len(observations):
            raise ValueError("Incomplete native replay snapshot.")
        if self.meter is not None:
            self.meter.ensure_env_steps(len(actions))
        started, replayed = monotonic_seconds(), 0
        try:
            self.reset(str(snapshot["task_id"]), int(snapshot["seed"]))
            for packet, expected in zip(actions, observations):
                replayed += 1
                result = self._step_unmetered(packet["action"], probe=packet["probe"])
                if result.observation != expected or result.done:
                    raise RuntimeError("Native text prefix replay diverged; branch pair rejected.")
            if (self._snapshot_state() != snapshot["native_state"]
                    or self._last_observation != snapshot["public_observation"]):
                raise RuntimeError("Native replay changed physical state, randomness, or public action support.")
        finally:
            elapsed = monotonic_seconds() - started
            self.replayed_env_steps += replayed
            self.snapshot_restore_seconds += elapsed
            if self.meter is not None:
                self.meter.record_restore(elapsed, replayed)


class NativeALFWorldEnv(_NativeTextEnv):
    """Official ALFWorld TextWorld games with PoS's 134-game unseen evaluation."""

    benchmark = "alfworld"
    _SPLITS = {"train": "train", "dev": "valid_seen", "test": "valid_unseen"}

    def __init__(self, data_root: str | Path, *, output_dir: str | Path | None = None,
                 max_steps: int = 50, meter: BudgetMeter | None = None) -> None:
        super().__init__(output_dir=output_dir, max_steps=max_steps, meter=meter, interface=(
            "You act in the official ALFWorld text environment. Emit one native command using the "
            "currently disclosed admissible commands. Each command consumes one environment step. "
            "The environment ends the task when its goal is satisfied; do not invent a submit API. "
            "I means observation, searching, navigation or opening a container; E means manipulation, "
            "use, closing or waiting. Information actions can change real state."
        ))
        self.data_root = Path(data_root).resolve()
        self.dataset_root = (self.data_root if self.data_root.name == "json_2.1.1"
                             else self.data_root / "json_2.1.1").resolve()
        self._runtime = importlib.import_module("alfworld.agents.environment.alfred_tw_env")
        self._textworld = importlib.import_module("textworld")
        self._env: Any = None
        self._state: Any = None
        self._rows: dict[str, dict[str, str]] = {}
        self._games: dict[str, Path] = {}
        self.split_manifest: dict[str, Any] = {
            "benchmark": self.benchmark, "scheme": "official_train_seen_unseen",
            "data_source": str(self.dataset_root), "package_version": _version("alfworld"),
            "evaluation_reference": "PoS valid_unseen, one trajectory, 50 text actions",
            "splits": {},
        }

    def list_tasks(self, split: str) -> list[dict[str, str]]:
        aliases = {"eval_in_distribution": "dev", "valid_seen": "dev",
                   "eval_out_of_distribution": "test", "valid_unseen": "test"}
        split = aliases.get(split, split)
        if split not in self._SPLITS:
            raise ValueError(f"Unknown ALFWorld split: {split}")
        if split not in self.split_manifest["splits"]:
            cache_path = self.dataset_root / ".prism_alfworld_splits_cache_v1.json"
            cached_splits: dict[str, Any] = {}
            if cache_path.is_file():
                try:
                    cached_splits = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    cached_splits = {}
            cached_entry = cached_splits.get(split)
            if (isinstance(cached_entry, Mapping)
                    and isinstance(cached_entry.get("tasks"), list)
                    and cached_entry["tasks"]
                    and (split != "test" or len(cached_entry["tasks"]) == 134)):
                rows = [dict(item) for item in cached_entry["tasks"]]
                for row in rows:
                    task_id = str(row["task_id"])
                    self._rows[task_id] = row
                    self._games[task_id] = (self.dataset_root / task_id).resolve()
                self.split_manifest["splits"][split] = {
                    "count": len(rows), "tasks": rows,
                    "source": str(self.dataset_root / self._SPLITS[split]),
                }
            else:
                config = {"dataset": {"data_path": str(self.dataset_root / "train"),
                                       "eval_id_data_path": str(self.dataset_root / "valid_seen"),
                                       "eval_ood_data_path": str(self.dataset_root / "valid_unseen"),
                                       "num_train_games": -1, "num_eval_games": -1},
                          "env": {"goal_desc_human_anns_prob": 0.0, "task_types": [1, 2, 3, 4, 5, 6]}}
                native_split = {"train": "train", "dev": "eval_in_distribution", "test": "eval_out_of_distribution"}[split]
                loader = self._runtime.AlfredTWEnv(config, train_eval=native_split)
                games = sorted(Path(path).resolve() for path in loader.game_files)
                if not games or (split == "test" and len(games) != 134):
                    raise ValueError(f"ALFWorld {split} contains {len(games)} games; PoS test requires all 134 unseen games.")
                rows = []
                for game in games:
                    task_id = game.relative_to(self.dataset_root).as_posix()
                    task_type = json.loads(game.with_name("traj_data.json").read_text())["task_type"]
                    category = ("Inspect" if task_type == "look_at_obj_in_light" else
                                "Transform" if task_type in {"pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep"}
                                else "Place")
                    row = {"task_id": task_id, "scenario_id": game.parent.parent.name,
                           "split": split, "native_split": self._SPLITS[split],
                           "task_type": task_type, "category": category}
                    self._rows[task_id], self._games[task_id] = row, game
                    rows.append(row)
                self.split_manifest["splits"][split] = {"count": len(rows), "tasks": rows,
                                                        "source": str(self.dataset_root / self._SPLITS[split])}
                try:
                    cached_splits[split] = self.split_manifest["splits"][split]
                    tmp_cache = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
                    tmp_cache.write_text(_json(cached_splits), encoding="utf-8")
                    os.replace(tmp_cache, cache_path)
                except OSError:
                    pass
        return deepcopy(self.split_manifest["splits"][split]["tasks"])

    def _task_row(self, task_id: str) -> dict[str, str]:
        if task_id not in self._rows:
            native_split = str(task_id).split("/", 1)[0]
            split = next((key for key, value in self._SPLITS.items() if value == native_split), None)
            if split is not None:
                self.list_tasks(split)
        if task_id not in self._rows:
            raise ValueError("Task is not a registered official ALFWorld game.")
        return self._rows[task_id]

    def _action_state(self):
        actions = list(self._state["admissible_commands"])
        modes: dict[str, list[str]] = {"I": [], "E": []}
        for action in actions:
            modes[classify_alfworld_action(action)].append(action)
        probes = [action for action in actions if action in {"look", "inventory"}
                  or action.startswith("examine ")]
        return modes, False, probes

    @staticmethod
    def _information_signature(observation: Mapping[str, Any]) -> str:
        public = deepcopy(dict(observation))
        public["native_observation"] = re.sub(
            r"(?:^|\n)Your task is to:.*", "", str(public["native_observation"])
        ).strip()
        return _NativeTextEnv._information_signature(public)

    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        self.close()
        self._begin(task_id, seed)
        # Facts stay private and are used only to check physical replay identity.
        infos = self._textworld.EnvInfos(won=True, facts=True, admissible_commands=True, extras=["gamefile"])
        with self._rng():
            self._env = self._textworld.start(str(self._games[task_id]), request_infos=infos,
                                            wrappers=[partial(self._runtime.AlfredDemangler, shuffle=False),
                                                      self._runtime.AlfredInfos])
            self._env.seed(int(seed))
            self._state = self._env.reset()
        observation = str(self._state["feedback"])
        match = re.search(r"Your task is to:\s*(.*)", observation)
        if match is None:
            raise ValueError("Official ALFWorld reset observation has no public task instruction.")
        modes, search, probes = self._action_state()
        self._publish(observation, modes, search_available=search, probe_actions=probes)
        return self._context(match.group(1).strip())

    def _mode(self, text: str) -> str | None:
        try:
            return classify_alfworld_action(text)
        except ValueError:
            return None

    def _world_state(self):
        if self._state is None:
            return None
        return tuple(sorted(str(fact) for fact in self._state["facts"]))

    def _private_state(self):
        return {"facts": self._world_state(), "game": self._task_id,
                "admissible_commands": list(self._state["admissible_commands"]),
                "feedback": str(self._state["feedback"])}

    def _native_step(self, text: str):
        admissible = set(self._state["admissible_commands"])
        self._state, _, native_done = self._env.step(text)
        observation = str(self._state["feedback"])
        won = bool(self._state["won"])
        return observation, bool(native_done or won), won, float(won), text not in admissible

    def training_reference(self, task_id: str | None = None) -> dict[str, Any]:
        task_id = str(task_id if task_id is not None else self._task_id)
        row = self._task_row(task_id)
        if row["split"] != "train":
            raise ValueError("Official ALFWorld references are restricted to registered training tasks.")
        source = self._games[task_id].with_name("traj_data.json")
        data = json.loads(source.read_text())
        reference = {"task_type": data["task_type"], "pddl_params": data.get("pddl_params"),
                     "high_level_plan": data.get("plan", {}).get("high_pddl", []),
                     "reference_kind": "official_goal_and_high_level_plan_not_native_text_walkthrough"}
        if not reference["pddl_params"] and not reference["high_level_plan"]:
            raise ValueError("Official training trajectory contains no reference goal or plan.")
        return {"benchmark": self.benchmark, "task_id": task_id, "source": str(source),
                "reference_id": f"alfworld/train/{task_id}", "reference": reference}

    def finalize_evaluation(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("ALFWorld scoring requires a reset episode.")
        row = self._task_row(self._task_id)
        self.last_native_result = {"benchmark": self.benchmark, **row,
                                  "success": bool(self._state["won"]), "submitted": self._submitted,
                                  "native_evaluator": "ALFWorld.TextWorld.won",
                                  "package_version": _version("alfworld"), "steps": self._attempts,
                                  "native_score": float(bool(self._state["won"]))}
        return deepcopy(self.last_native_result)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


class NativeWebShopEnv(_NativeTextEnv):
    """Official WebShop text environment with GiGPO/verl-agent split and 15-step horizon."""

    benchmark = "webshop"
    _SEARCH_RE = re.compile(r"^search\[[^\[\]\r\n]+\]$")

    def __init__(self, repo_root: str | Path, *, output_dir: str | Path | None = None,
                 max_steps: int = 15, use_small: bool = True, human_goals: bool = False,
                 meter: BudgetMeter | None = None) -> None:
        super().__init__(output_dir=output_dir, max_steps=max_steps, meter=meter, interface=(
            "You act in the official WebShop text environment. Emit one native command: either "
            "search[keywords] when a search bar is available, or click[value] for one of the "
            "currently disclosed clickable buttons/links/options. Each command consumes one "
            "environment step. Clicking click[buy now] purchases the current product and ends "
            "the episode. I means search, navigation, pagination or inspecting product details; "
            "E means selecting a purchase option or clicking buy now."
        ))
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.data_root = (self.repo_root / "data").resolve()
        self.use_small = bool(use_small)
        self.human_goals = bool(human_goals)
        self.file_path = (self.data_root / (
            "items_shuffle_1000.json" if self.use_small else "items_shuffle.json")).resolve()
        self.attr_path = (self.data_root / (
            "items_ins_v2_1000.json" if self.use_small else "items_ins_v2.json")).resolve()
        if not self.file_path.is_file() or not self.attr_path.is_file():
            raise FileNotFoundError(f"Missing official WebShop data files in {self.data_root}")
        self._env: Any = None
        self._rows: dict[str, dict[str, Any]] = {}
        self._goals: dict[str, dict[str, Any]] = {}
        self.split_manifest: dict[str, Any] = {
            "benchmark": self.benchmark,
            "scheme": "official_webshop_test500_dev1000_train_rest",
            "data_source": str(self.data_root),
            "file_path": str(self.file_path),
            "attr_path": str(self.attr_path),
            "use_small": self.use_small,
            "human_goals": self.human_goals,
            "evaluation_reference": "Official WebShop / GiGPO 500 test instructions, 15 text actions",
            "splits": {},
        }

    @contextmanager
    def _in_repo_root(self) -> Iterator[None]:
        previous = Path.cwd()
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        os.chdir(self.repo_root)
        try:
            yield
        finally:
            os.chdir(previous)

    def _splits_cache_path(self) -> Path:
        digest = hashlib.sha256(str(self.file_path).encode("utf-8")).hexdigest()[:12]
        return Path(tempfile.gettempdir()) / (
            f"prism_webshop_splits_{digest}_small{int(self.use_small)}_human{int(self.human_goals)}_v1.json"
        )

    def _populate_splits_from_goals(self, goals: Sequence[Mapping[str, Any]]) -> None:
        if len(goals) <= 1500:
            raise ValueError(f"Official WebShop loaded only {len(goals)} goals; expected > 1500.")
        ranges = {
            "test": range(0, 500),
            "dev": range(500, 1500),
            "train": range(1500, len(goals)),
        }
        for split_name, indices in ranges.items():
            rows = []
            for idx in indices:
                goal = goals[idx]
                task_id = f"webshop/{idx}"
                asin = str(goal.get("asin", ""))
                row = {
                    "task_id": task_id,
                    "scenario_id": asin or task_id,
                    "split": split_name,
                    "session_index": int(idx),
                    "asin": asin,
                    "category": str(goal.get("category", "product")),
                    "task_type": "webshop_purchase",
                }
                self._rows[task_id] = row
                self._goals[task_id] = deepcopy(_plain(goal))
                rows.append(row)
            self.split_manifest["splits"][split_name] = {
                "count": len(rows),
                "tasks": rows,
                "source": str(self.file_path),
            }

    def _ensure_splits(self) -> None:
        if self.split_manifest["splits"] and self._rows and self._goals:
            return
        cache_path = self._splits_cache_path()
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                goals = cached.get("goals")
                if isinstance(goals, list) and len(goals) > 1500:
                    self._populate_splits_from_goals(goals)
                    return
            except Exception:
                pass
        self._ensure_env()

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        self._seed_rng(42)
        with self._in_repo_root(), self._rng():
            module = importlib.import_module("web_agent_site.envs.web_agent_text_env")
            self._env = module.WebAgentTextEnv(
                observation_mode="text",
                file_path=str(self.file_path),
                attr_path=str(self.attr_path),
                human_goals=self.human_goals,
                seed=42,
            )
        goals = [_plain(g) for g in self._env.server.goals]
        self._populate_splits_from_goals(goals)
        cache_path = self._splits_cache_path()
        if not cache_path.is_file():
            try:
                tmp_cache = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
                tmp_cache.write_text(_json({"goals": goals}), encoding="utf-8")
                os.replace(tmp_cache, cache_path)
            except OSError:
                pass

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown WebShop split: {split}")
        self._ensure_splits()
        return deepcopy(self.split_manifest["splits"][split]["tasks"])

    def _task_row(self, task_id: str) -> dict[str, Any]:
        self._ensure_splits()
        if task_id not in self._rows:
            raise ValueError(f"Task is not a registered official WebShop goal: {task_id!r}")
        return self._rows[task_id]

    def _action_state(self):
        available = self._env.get_available_actions()
        has_search = bool(available.get("has_search_bar", False))
        clickables = [str(item) for item in available.get("clickables", ())]
        html_obj = self._env._parse_html(self._env.browser.page_source)
        buying_options = {
            str(opt.get("value", "")).lower()
            for opt in html_obj.select('input[type="radio"]')
            if opt.get("value")
        }
        modes: dict[str, list[str]] = {"I": [], "E": []}
        for item in clickables:
            command = f"click[{item}]"
            if item == "buy now" or item in buying_options:
                modes["E"].append(command)
            else:
                modes["I"].append(command)
        probes = [
            f"click[{item}]"
            for item in clickables
            if item in {"description", "features", "reviews", "attributes",
                        "back to search", "< prev", "next >"}
        ]
        if has_search:
            probes.append("search[<your query>]")
        return modes, has_search, probes

    @staticmethod
    def _information_signature(observation: Mapping[str, Any]) -> str:
        public = deepcopy(dict(observation))
        public["native_observation"] = re.sub(
            r"^Instruction:\s*\[SEP\]\s*.*?\[SEP\]\s*",
            "",
            str(public["native_observation"]),
        ).strip()
        return _NativeTextEnv._information_signature(public)

    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        self._ensure_env()
        row = self._task_row(task_id)
        self._begin(task_id, seed)
        with self._in_repo_root(), self._rng():
            # Official reset reuses done/reward fields from an existing session.
            session_id = str(self._env.session_prefix or "") + str(int(row["session_index"]))
            self._env.server.user_sessions.pop(session_id, None)
            self._env.reset(session=int(row["session_index"]))
            observation = str(self._env.observation)
            instruction = str(self._env.get_instruction_text()).strip()
            modes, search, probes = self._action_state()
        if not instruction:
            raise ValueError("Official WebShop reset observation has no public task instruction.")
        self._publish(observation, modes, search_available=search, probe_actions=probes)
        return self._context(instruction)

    def _comparison_action(self, text: str) -> str:
        if (bool(self.action_schema.get("search_available", False))
                and bool(self._SEARCH_RE.fullmatch(text.strip()))):
            return "search[<your query>]"
        return text

    def _mode(self, text: str) -> str | None:
        direct = super()._mode(text)
        if direct is not None:
            return direct
        if (bool(self.action_schema.get("search_available", False))
                and bool(self._SEARCH_RE.fullmatch(text.strip()))):
            return "I"
        return None

    def _world_state(self):
        if self._env is None or not self._task_id:
            return None
        session = self._env.server.user_sessions.get(self._env.session, {})
        return _plain({
            "url": self._env.browser.current_url,
            "session": deepcopy(session),
        })

    def _private_state(self):
        return {
            "world": self._world_state(),
            "task_id": self._task_id,
            "observation": str(self._env.observation),
            "available_actions": _plain(self._env.get_available_actions()),
        }

    def _native_step(self, text: str):
        available = self._env.get_available_actions()
        admissible_clicks = {f"click[{item}]" for item in available.get("clickables", ())}
        valid_search = bool(
            available.get("has_search_bar", False) and self._SEARCH_RE.fullmatch(text)
        )
        admissible = (text in admissible_clicks) or valid_search
        with self._in_repo_root():
            observation, reward, native_done, _ = self._env.step(text)
        score = float(reward or 0.0)
        won = bool(native_done and score == 1.0)
        return str(observation), bool(native_done), won, score, not admissible

    def training_reference(self, task_id: str | None = None) -> dict[str, Any]:
        selected = str(task_id if task_id is not None else self._task_id)
        row = self._task_row(selected)
        if row["split"] != "train":
            raise ValueError("Official WebShop references are restricted to registered training tasks.")
        goal = deepcopy(self._goals[selected])
        asin = str(goal.get("asin", "")).lower()
        options = goal.get("goal_options", {})
        option_values = (
            [str(v).lower() for v in options.values()]
            if isinstance(options, Mapping)
            else [str(v).lower() for v in options]
        )
        walkthrough = [
            f"search[{goal.get('instruction_text', row['asin'])}]",
            f"click[{asin}]",
            *[f"click[{opt}]" for opt in option_values if opt],
            "click[buy now]",
        ]
        reference = {
            "asin": goal.get("asin"),
            "attributes": goal.get("attributes", []),
            "goal_options": goal.get("goal_options", {}),
            "price_upper": goal.get("price_upper"),
            "walkthrough": walkthrough,
            "reference_kind": "official_goal_attributes_options_and_purchase_sequence",
        }
        return {
            "benchmark": self.benchmark,
            "task_id": selected,
            "source": str(self.file_path),
            "reference_id": f"webshop/train/{selected}",
            "reference": reference,
        }

    def finalize_evaluation(self) -> dict[str, Any]:
        if self._env is None or not self._task_id:
            raise RuntimeError("WebShop scoring requires a reset episode.")
        row = self._task_row(self._task_id)
        self.last_native_result = {
            "benchmark": self.benchmark,
            **row,
            "success": bool(self._won),
            "submitted": bool(self._submitted),
            "native_evaluator": "WebShop.WebAgentTextEnv.get_reward",
            "steps": self._attempts,
            "native_score": float(self._native_score),
            "task_score": float(self._native_score),
        }
        return deepcopy(self.last_native_result)

    def close(self) -> None:
        return None


class NativeEnvironmentProcess:
    """JSON transport for an official environment's independent Python runtime."""

    def __init__(self, python_executable: str | Path, *, environment_class: str,
                 constructor_kwargs: Mapping[str, Any], meter: BudgetMeter | None = None) -> None:
        self.meter = meter
        self.action_schema: dict[str, Any] = {}
        self.split_manifest: dict[str, Any] = {}
        self.last_native_result: dict[str, Any] | None = None
        self.replayed_env_steps = 0
        self.snapshot_restore_seconds = 0.0
        self._request_id = 0
        self._closed = False
        self._log = None
        output = constructor_kwargs.get("output_dir")
        if output is not None:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            self._log = (directory / "environment-worker.log").open("a", encoding="utf-8")
        environment = dict(os.environ)
        source = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        try:
            self._process = subprocess.Popen(
                [str(python_executable), "-u", "-m", "prism_evolve.native_text_envs", "--worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._log, text=True, encoding="utf-8", bufsize=1, env=environment,
            )
        except BaseException:
            if self._log is not None:
                self._log.close()
            raise
        try:
            self._rpc("initialize", environment_class=environment_class,
                      constructor_kwargs=_plain(dict(constructor_kwargs)))
        except BaseException:
            self._shutdown()
            raise

    def _record_usage(self, usage: Mapping[str, Any]) -> None:
        if self.meter is None:
            return
        replayed = int(usage.get("replayed_env_steps", 0))
        direct = int(usage.get("env_steps", 0)) - replayed
        seconds = float(usage.get("environment_seconds", 0.0))
        for _ in range(direct):
            self.meter.record_env_step(seconds / direct)
        if usage.get("snapshot_restores", 0):
            self.meter.record_restore(float(usage["restore_seconds"]), replayed)
        if usage.get("snapshot_saves", 0):
            self.meter.record_snapshot_save(float(usage["snapshot_save_seconds"]))
        calls = int(usage.get("native_api_calls", 0))
        if calls:
            self.meter.record_api_calls(calls, successful=int(usage.get("successful_native_api_calls", 0)))

    def _rpc(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("The isolated native environment is closed.")
        limits = None
        if self.meter is not None and method in {"step", "step_native", "restore_snapshot"}:
            count = len(args[0].get("actions", ())) if method == "restore_snapshot" else 1
            self.meter.ensure_env_steps(count)
            remaining_wall = math.inf
            current = self.meter
            while current is not None:
                remaining_wall = min(remaining_wall, current.limits.max_wall_seconds - current.elapsed_seconds)
                current = current.parent
            limits = {"max_env_steps": max(1, self.meter.remaining_env_steps),
                      "max_model_tokens": max(1, self.meter.remaining_model_tokens),
                      "max_wall_seconds": max(1e-9, remaining_wall)}
        self._request_id += 1
        packet = {"id": self._request_id, "method": method, "args": _plain(args),
                  "kwargs": _plain(kwargs), "limits": limits}
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(_json(packet) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"Native environment worker exited unexpectedly (code={self._process.poll()}).")
        reply = json.loads(line)
        if reply.get("id") != self._request_id:
            raise RuntimeError("Out-of-order native environment response.")
        self._record_usage(reply.get("usage", {}))
        state = reply.get("state", {})
        for name in ("benchmark", "action_schema", "split_manifest", "last_native_result",
                     "replayed_env_steps", "snapshot_restore_seconds"):
            if name in state:
                setattr(self, name, state[name])
        if not reply.get("ok"):
            kind, message = reply.get("error_type"), str(reply.get("error", "Native worker failed."))
            if kind == "BudgetExhausted":
                raise BudgetExhausted(message)
            if kind == "ValueError":
                raise ValueError(message)
            raise RuntimeError(f"{kind}: {message}")
        return reply.get("result")

    def list_tasks(self, split: str):
        return self._rpc("list_tasks", split)

    def reset(self, task_id: str, seed: int):
        return self._rpc("reset", task_id, seed)

    def schema_for_history(self, messages):
        return public_text_action_schema(self.action_schema, messages)

    def step(self, action):
        return NativeStep(**self._rpc("step", action))

    def step_native(self, action, *, probe=False):
        return NativeStep(**self._rpc("step_native", action, probe=probe))

    def save_snapshot(self):
        return self._rpc("save_snapshot")

    def restore_snapshot(self, snapshot):
        return self._rpc("restore_snapshot", snapshot)

    def finalize_evaluation(self):
        return self._rpc("finalize_evaluation")

    def training_reference(self, task_id=None):
        return self._rpc("training_reference", task_id)

    def _shutdown(self):
        self._closed = True
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        for stream in (self._process.stdin, self._process.stdout, self._log):
            if stream is not None:
                stream.close()

    def close(self):
        if self._closed:
            return
        try:
            self._rpc("close")
        finally:
            self._shutdown()


def _environment_worker() -> None:
    channel = sys.stdout
    env = None
    manifest_key = None
    permitted = {"list_tasks", "reset", "step", "step_native", "save_snapshot",
                 "restore_snapshot", "finalize_evaluation", "training_reference", "close"}
    for line in sys.stdin:
        packet = json.loads(line)
        method = packet["method"]
        # This meter records only this RPC; the parent retains the cumulative ledger.
        limits = packet.get("limits") or {"max_env_steps": 2 ** 31,
                                           "max_model_tokens": 2 ** 31,
                                           "max_wall_seconds": 365 * 24 * 3600}
        meter = BudgetMeter(BudgetLimits(**limits))
        reply: dict[str, Any] = {"id": packet["id"]}
        try:
            with redirect_stdout(sys.stderr):
                if method == "initialize":
                    if env is not None:
                        raise RuntimeError("The native worker was already initialized.")
                    module, name = packet["kwargs"]["environment_class"].split(":", 1)
                    if not module.startswith("prism_evolve.") or name.startswith("_"):
                        raise ValueError("Environment worker requires a registered project adapter class.")
                    adapter = getattr(importlib.import_module(module), name)
                    env = adapter(**packet["kwargs"]["constructor_kwargs"], meter=meter)
                    result = None
                else:
                    if method not in permitted or env is None:
                        raise ValueError("Unsupported native environment request.")
                    env.meter = meter
                    result = getattr(env, method)(*packet["args"], **packet["kwargs"])
                if isinstance(result, NativeStep):
                    result = asdict(result)
            reply.update(ok=True, result=result)
        except Exception as error:
            reply.update(ok=False, error_type=type(error).__name__, error=str(error))
        state = {}
        if env is not None:
            state = {name: getattr(env, name) for name in (
                "benchmark", "action_schema", "last_native_result", "replayed_env_steps",
                "snapshot_restore_seconds") if hasattr(env, name)}
            current_key = tuple((name, split.get("count")) for name, split in
                                env.split_manifest.get("splits", {}).items())
            if manifest_key != current_key:
                state["split_manifest"] = env.split_manifest
                manifest_key = current_key
        reply.update(usage=meter.snapshot(), state=state)
        channel.write(_json(_plain(reply)) + "\n")
        channel.flush()
        if method == "close":
            return
    if env is not None:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    _environment_worker()


