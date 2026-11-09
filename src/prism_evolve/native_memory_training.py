"""WorldEvolver's train-only memory exposure under the shared native budget."""

from __future__ import annotations

from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .native_baselines import run_inference_method
from .native_core import BudgetExhausted, BudgetLimits, BudgetMeter
from .native_training import _global_budget_limited, public_trace


def _memory_content(checkpoint: Mapping[str, Any] | None) -> dict[str, Any]:
    checkpoint = checkpoint or {}
    episodic = checkpoint.get("episodic", {})
    semantic = checkpoint.get("semantic", {})
    return {"episodic": episodic.get("records", []),
            "semantic": semantic.get("registry", {}).get("entries", {}),
            "pending": semantic.get("pending", [])}


def _content_hash(content: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def warm_start_worldevolver(
    policy: Any,
    env: Any,
    train_tasks: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    meter: BudgetMeter,
    output_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Accumulate official memory over one seeded pass of the native train split.

    The initial reference model remains frozen through run_inference_method.
    Each episode consumes the common training meter, including every auxiliary
    model call. The returned checkpoint contains only the upstream method's
    actual memory; held-out sequences start from separate copies of this file.
    """
    import torch

    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in train_tasks:
        if item["split"] != "train":
            raise ValueError("WorldEvolver memory exposure accepts only native train tasks.")
        task_id, scenario_id = str(item["task_id"]), str(item["scenario_id"])
        if not task_id or not scenario_id or task_id in seen:
            raise ValueError("Memory training requires unique task IDs and nonempty scenario IDs.")
        seen.add(task_id)
        tasks.append({"task_id": task_id, "scenario_id": scenario_id, "split": "train"})
    if not tasks:
        raise ValueError("WorldEvolver memory training requires a nonempty train split.")
    tasks.sort(key=lambda item: item["task_id"])
    random.Random(seed).shuffle(tasks)
    for index, task in enumerate(tasks):
        task["environment_seed"] = int(seed) * 1000003 + index

    benchmark, backbone = str(config["benchmark"]), str(config["model_id"])
    limits = BudgetLimits(
        max_env_steps=int(config["episode_max_steps"]),
        max_model_tokens=int(config["episode_max_model_tokens"]),
        max_wall_seconds=float(config["episode_max_wall_seconds"]),
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "memory_training_protocol.json", {
        "protocol_version": "native-v1",
        "method": "worldevolver",
        "method_identity": "shared-backbone-native-task-transfer",
        "benchmark": benchmark,
        "backbone": backbone,
        "seed": seed,
        "split": "train",
        "parameter_updates": False,
        "task_order": tasks,
        "sampling": "seeded_without_replacement_single_pass",
        "training_budget": asdict(meter.limits),
        "episode_budget": asdict(limits),
        "worldevolver": dict(config["worldevolver"]),
    })

    memory: dict[str, Any] = {}
    num_tasks = 0
    num_completed = 0
    changed_episodes = 0
    stop_reason = "train_split_exhausted"
    try:
        with (output / "records.jsonl").open("x", encoding="utf-8") as records:
            for index, task in enumerate(tasks):
                try:
                    meter.ensure_env_steps(1)
                    meter.ensure_model_call(0, 1, role="worldevolver_memory_training")
                except BudgetExhausted as error:
                    stop_reason = f"budget:{error}"
                    break

                episode = BudgetMeter(limits, parent=meter,
                                      event_path=output / "costs" / f"task-{index:05d}.jsonl")
                torch.manual_seed(task["environment_seed"])
                before_content = _memory_content(memory.get("checkpoint"))
                trajectory = run_inference_method(
                    "worldevolver", policy, env, task, seed, config, episode,
                    output / "official", memory_state=memory,
                )
                _save_json(output / "traces" / f"task-{index:05d}.json", public_trace(trajectory))
                if trajectory.termination in {"method_error", "method_generation_failure"}:
                    raise RuntimeError(
                        "WorldEvolver memory training failed: "
                        + str(trajectory.metadata.get(trajectory.termination))
                    )

                record = trajectory.result(benchmark=benchmark, backbone=backbone,
                                           seed=seed, method="worldevolver")
                after_content = _memory_content(memory.get("checkpoint"))
                content_changed = before_content != after_content
                changed_episodes += int(content_changed)
                record.update(split="train", task_order_index=index,
                              environment_seed=task["environment_seed"],
                              memory_content_changed=content_changed,
                              memory_content_before_sha256=_content_hash(before_content),
                              memory_content_after_sha256=_content_hash(after_content),
                              episodic_records=len(after_content["episodic"]),
                              semantic_rules=len(after_content["semantic"]),
                              resource_usage=episode.snapshot())
                records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                records.flush()
                num_tasks += 1
                num_completed += int(trajectory.termination in {"success", "native_failure"})

                reason = trajectory.metadata.get("budget_exhausted")
                if reason is not None and _global_budget_limited(episode, str(reason)):
                    stop_reason = f"budget:{reason}"
                    break

        checkpoint = memory.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or not checkpoint:
            raise RuntimeError("WorldEvolver returned no actual official memory checkpoint.")
        checkpoint_path = output / "worldevolver_initial_memory.json"
        _save_json(checkpoint_path, checkpoint)
        summary = {
            "protocol_version": "native-v1",
            "method": "worldevolver",
            "method_identity": "shared-backbone-native-task-transfer",
            "parameter_updates": False,
            "parameter_training": "not_applicable",
            "memory_changed_episodes": changed_episodes,
            "memory_update_status": "changed" if changed_episodes else "no_content_change",
            "memory_content_sha256": _content_hash(_memory_content(checkpoint)),
            "episodic_records": len(_memory_content(checkpoint)["episodic"]),
            "semantic_rules": len(_memory_content(checkpoint)["semantic"]),
            "benchmark": benchmark,
            "backbone": backbone,
            "seed": seed,
            "memory_checkpoint": str(checkpoint_path),
            "training_usage": meter.snapshot(),
            "num_train_tasks": num_tasks,
            "num_completed_train_tasks": num_completed,
            "num_registered_train_tasks": len(tasks),
            "stop_reason": stop_reason,
            "records": str(output / "records.jsonl"),
            "task_order": str(output / "memory_training_protocol.json"),
        }
        _save_json(output / "memory_training_summary.json", summary)
        return summary
    finally:
        if "session" in memory:
            memory["session"].close()
