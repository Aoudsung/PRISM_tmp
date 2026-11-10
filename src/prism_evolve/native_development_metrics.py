"""Formal development-set feedback utilization under design §6.2.

The physical post-information state, paid prefix costs and actor random stream
are shared by the visible and masked continuations. This measurement never
updates policy parameters or serves as an experiment approval condition.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch

from .native_core import BudgetExhausted, BudgetLimits, BudgetMeter, NativeTrajectory, monotonic_seconds
from .native_training import (
    TRAINING_METHODS, _continuation_meter, _global_budget_limited,
    _limits, _meters, _preserve_actor_rng, _restore_environment, public_trace, rollout,
    actor_contract, executed_information_event,
)


_FRAMEWORK_METHODS = frozenset({"pos", "worldevolver", "envprobe_simple", "envprobe_judge"})
_MASK = "[MASKED_FEEDBACK]"


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _restore_actor_rng(packet: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(packet["rng_cpu"])
    if packet["rng_cuda"] is not None:
        torch.cuda.set_rng_state_all(packet["rng_cuda"])


def _feedback_continuation(
    policy: Any, env: Any, task: Mapping[str, Any], config: Mapping[str, Any],
    total_meter: BudgetMeter, packet: Mapping[str, Any], *, method: str, environment_seed: int,
    masked: bool,
) -> NativeTrajectory:
    prefix = deepcopy(packet["prefix"])
    messages = deepcopy(packet["messages"])
    if not prefix or not executed_information_event(prefix[-1]):
        raise ValueError("Feedback measurement requires a complete program with an actual native information call.")
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("Captured feedback must be the final public observation message.")
    if messages[-1].get("content") != prefix[-1].observation:
        raise ValueError("Captured information observation and public history disagree.")
    if masked:
        prefix[-1].observation = _MASK
        messages[-1]["content"] = _MASK
    prefix[-1].metadata.pop("belief_generation", None)
    prefix[-1].metadata.pop("belief_messages", None)
    prefix_seconds = float(packet["prefix_seconds"])

    # Replay restores physical state only. Its work remains on the total meter.
    with _meters(policy, env, total_meter):
        _restore_environment(env, packet["snapshot"])
    _restore_actor_rng(packet)
    prefix_usage = packet.get("prefix_usage")
    if not isinstance(prefix_usage, Mapping):
        raise ValueError("Feedback continuation requires the captured prefix resource ledger.")
    episode_meter = _continuation_meter(_limits(config), total_meter, prefix_usage)
    episode_meter.started = monotonic_seconds() - prefix_seconds

    trajectory = rollout(
        policy, env, task, config, episode_meter, seed=environment_seed, method=method, training=False,
        context=packet["context"], prefix=prefix, prefix_messages=messages,
        prefix_seconds=prefix_seconds,
    )
    continuation = trajectory.events[len(prefix):]
    trajectory.metadata["feedback_intervention"] = {
        "masked": masked, "catalog_visibility": "actor_visible_history",
        "first_continuation_context_id": (
            continuation[0].generation.metadata.get("program_contract", {}).get("context_id")
            if continuation else None),
        "physical_state_and_namespace": "restored_original_post_information_snapshot",
    }
    return trajectory


def evaluate_feedback_use(
    policy: Any, env: Any, tasks: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    meter: BudgetMeter, output_dir: str | Path, method: str, seed: int, backbone: str,
) -> dict[str, Any]:
    """Collect one visible and necessary masked continuation per registered dev task.

    A metric-wide budget interruption leaves the interrupted pair unestimated.
    Normal episode limits remain part of the fixed-budget deployment outcome.
    An episode with no nonterminal information action contributes exactly zero
    by the registered §6.2 convention and requires no extra model invocation.
    The caller owns env; each rollout resets the episode and the caller closes
    the environment after the complete measurement.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "feedback_use_records.jsonl"
    summary_path = output / "feedback_use_summary.json"
    if record_path.exists() or summary_path.exists():
        raise FileExistsError("Feedback-use evidence already exists; completed data are not rerun.")
    if method in _FRAMEWORK_METHODS:
        summary = {
            "metric": "delta_use", "status": "not_applicable", "method": method,
            "reason": "The registered operator applies to one policy's public-history state.",
            "records": None, "mean_delta_use": None, "complete": True,
            "resource_usage": meter.snapshot(),
        }
        _save(summary_path, summary)
        return summary
    if method not in TRAINING_METHODS and method != "frozen_base":
        raise ValueError(f"Unregistered single-policy method: {method}")
    planned = [dict(task) for task in tasks]
    if not planned or any(task.get("split") != "dev" for task in planned):
        raise ValueError("Feedback utilization uses the nonempty registered development split only.")
    if any(not task.get("task_id") or not task.get("scenario_id") for task in planned):
        raise ValueError("Every development task needs its native task and scenario IDs.")
    if len({str(task["task_id"]) for task in planned}) != len(planned):
        raise ValueError("Development task IDs must be unique.")
    benchmark = str(config.get("benchmark", env.benchmark))
    limits = _limits(config)
    _save(output / "feedback_use_protocol.json", {
        "protocol_version": "native-v1", "metric": "delta_use", "design_section": "6.2",
        "method": method, "benchmark": benchmark, "backbone": backbone, "seed": seed,
        "actor_contract": actor_contract(method, classic_text=bool(env.action_schema.get("classic_text"))),
        "tasks": planned, "mask": _MASK, "scope": "formal_development_measurement",
        "selection": "first_complete_nonterminal_program_with_executed_information_call",
        "catalog_visibility": "rebuild_from_each_visible_or_masked_history",
        "anchor_admits_native_information_errors": True,
        "paired_state": "same_post_information_physical_state_and_paid_prefix",
        "paired_randomness": "restore_actor_rng_after_information_before_belief_update",
        "no_information_contribution": 0.0,
    })
    completed_deltas: list[float] = []
    stop_reason = "completed"
    old_training = bool(policy.training)
    policy.eval()
    try:
        with record_path.open("x", encoding="utf-8") as records:
            for index, task in enumerate(planned):
                environment_seed = int(task.get("environment_seed", seed * 1000003 + index))
                actor_seed = seed * 1000003 + index
                visible: NativeTrajectory | None = None
                masked: NativeTrajectory | None = None
                packet: dict[str, Any] = {}
                row: dict[str, Any] = {
                    "benchmark": benchmark, "backbone": backbone, "method": method,
                    "seed": seed, "environment_seed": environment_seed, "actor_seed": actor_seed,
                    "task_id": str(task["task_id"]), "scenario_id": str(task["scenario_id"]),
                    "split": "dev", "status": "partial", "visible_utility": None,
                    "masked_utility": None, "visible_success": None, "masked_success": None,
                    "delta_use": None,
                }
                try:
                    meter.ensure_env_steps(1)
                    with _preserve_actor_rng():
                        torch.manual_seed(actor_seed)
                        visible = rollout(
                            policy, env, task, config, BudgetMeter(limits, parent=meter),
                            seed=environment_seed, method=method, training=False,
                            capture_information=packet,
                        )
                        row.update(visible_utility=visible.utility, visible_success=int(visible.success),
                                   visible_termination=visible.termination)
                        if visible.metadata.get("global_budget_limited"):
                            raise BudgetExhausted(f"visible_pair:{visible.termination}")
                        if packet:
                            masked = _feedback_continuation(
                                policy, env, task, config, meter, packet,
                                method=method, environment_seed=environment_seed, masked=True,
                            )
                            row.update(information_step=len(packet["prefix"]) - 1,
                                       anchor_evidence=packet["anchor_evidence"],
                                       masked_utility=masked.utility, masked_success=int(masked.success),
                                       masked_termination=masked.termination)
                            if masked.metadata.get("global_budget_limited"):
                                raise BudgetExhausted(f"masked_pair:{masked.termination}")
                        else:
                            row.update(information_step=None, masked_utility=visible.utility,
                                       masked_success=int(visible.success),
                                       masked_termination="no_nonterminal_information_action")
                        row.update(status="complete", delta_use=row["visible_utility"] - row["masked_utility"])
                        completed_deltas.append(float(row["delta_use"]))
                except BudgetExhausted as error:
                    stop_reason = f"metric_budget:{error}"
                    row["stop_reason"] = stop_reason
                finally:
                    if visible is not None:
                        _save(output / "visible" / f"task-{index:05d}.json", public_trace(visible))
                    if masked is not None:
                        _save(output / "masked" / f"task-{index:05d}.json", public_trace(masked))
                    row["cumulative_resource_usage"] = meter.snapshot()
                    records.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    records.flush()
                if row["status"] != "complete":
                    break
    finally:
        policy.train(old_training)
    complete = len(completed_deltas) == len(planned)
    summary = {
        "metric": "delta_use", "status": "complete" if complete else "partial",
        "method": method, "benchmark": benchmark, "backbone": backbone, "seed": seed,
        "records": str(record_path), "num_tasks": len(planned),
        "num_complete_pairs": len(completed_deltas), "complete": complete,
        "mean_delta_use": math.fsum(completed_deltas) / len(planned) if complete else None,
        "stop_reason": stop_reason, "resource_usage": meter.snapshot(),
    }
    _save(summary_path, summary)
    return summary


def _registered_feedback_tasks(
    dev_tasks: Sequence[Mapping[str, Any]], selection_seed: int, num_tasks: int,
) -> list[tuple[int, dict[str, Any]]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    seen: set[str] = set()
    for original_index, source in enumerate(dev_tasks):
        task = dict(source)
        if task.get("split") != "dev" or not task.get("task_id") or not task.get("scenario_id"):
            raise ValueError("Feedback change requires native development task and scenario IDs.")
        task_id = str(task["task_id"])
        if task_id in seen:
            raise ValueError("Development task IDs must be unique.")
        seen.add(task_id)
        groups.setdefault(str(task["scenario_id"]), []).append((original_index, task))
    if len(seen) < num_tasks:
        raise ValueError(f"Feedback change requires {num_tasks} distinct development tasks.")
    rng = random.Random(selection_seed)
    scenarios = sorted(groups)
    rng.shuffle(scenarios)
    for scenario in scenarios:
        rng.shuffle(groups[scenario])
    selected: list[tuple[int, dict[str, Any]]] = []
    for depth in range(max(len(group) for group in groups.values())):
        for scenario in scenarios:
            if depth < len(groups[scenario]):
                selected.append(groups[scenario][depth])
                if len(selected) == num_tasks:
                    return selected
    raise RuntimeError("Registered development selection did not reach its configured size.")


def evaluate_feedback_change(
    policy: Any, env: Any, dev_tasks: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    meter: BudgetMeter, output_dir: str | Path, initial_checkpoint: str | Path,
    final_checkpoint: str | Path, seed: int, backbone: str,
) -> dict[str, Any]:
    """Measure checkpoint feedback-use change on a fixed initial-policy anchor distribution.

    Initial visible trajectories supply the common first nonterminal I anchor.
    The other three endpoints restore that physical state, paid prefix and RNG.
    Tasks without an anchor remain registered and have no estimated delta.
    The caller owns env and closes it after all registered tasks are processed.
    """
    diagnostic = config.get("feedback_diagnostic", {})
    num_tasks = int(diagnostic.get("num_tasks", 24))
    if num_tasks != 24:
        raise ValueError("The registered feedback-change experiment contains 24 development tasks.")
    selection_seed = int(diagnostic.get("selection_seed", 20261105))
    selected = _registered_feedback_tasks(dev_tasks, selection_seed, num_tasks)
    benchmark = str(config.get("benchmark", env.benchmark))
    method = "prism_evolve"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "feedback_change_records.jsonl"
    summary_path = output / "feedback_change_summary.json"
    protocol_path = output / "feedback_change_protocol.json"
    if any(path.exists() for path in (record_path, summary_path, protocol_path)):
        raise FileExistsError("Feedback-change evidence already exists; use a new experiment directory.")
    episode_limits = _limits(config)
    checkpoint_limits = BudgetLimits(
        max_env_steps=meter.limits.max_env_steps // 2,
        max_model_tokens=meter.limits.max_model_tokens // 2,
        max_wall_seconds=meter.limits.max_wall_seconds / 2,
    )
    checkpoint_meters = {
        label: BudgetMeter(checkpoint_limits, parent=meter) for label in ("initial", "final")
    }
    last_active_end = {label: child.started for label, child in checkpoint_meters.items()}

    @contextmanager
    def active_checkpoint(label: str):
        child = checkpoint_meters[label]
        child.started += monotonic_seconds() - last_active_end[label]
        try:
            yield child
        finally:
            last_active_end[label] = monotonic_seconds()

    def checkpoint_usage() -> dict[str, Any]:
        usage = {}
        for label, child in checkpoint_meters.items():
            now = monotonic_seconds()
            child.started += now - last_active_end[label]
            last_active_end[label] = now
            usage[label] = child.snapshot()
        return usage

    def require_endpoint(trajectory: NativeTrajectory, child: BudgetMeter) -> None:
        limited = bool(trajectory.metadata.get("global_budget_limited"))
        # Nested total meters also account for time spent in the other checkpoint.
        if trajectory.termination == "budget:wall_seconds":
            ancestor: BudgetMeter | None = child
            while ancestor is not None:
                limited |= ancestor.elapsed_seconds >= ancestor.limits.max_wall_seconds
                ancestor = ancestor.parent
        if limited:
            raise BudgetExhausted(trajectory.termination)

    registered_tasks = [
        {**task, "task_id": str(task["task_id"]), "scenario_id": str(task["scenario_id"]),
         "split": "dev", "original_index": original_index,
         "environment_seed": int(task.get("environment_seed", seed * 1000003 + original_index)),
         "actor_seed": seed * 1000003 + original_index}
        for original_index, task in selected
    ]
    anchor_distribution = "initial_policy_first_nonterminal_information_action_on_registered_dev_tasks"
    _save(protocol_path, {
        "protocol_version": "native-feedback-change-v1", "metric": "delta_change",
        "design_section": "6.2", "scope": "formal_development_measurement",
        "benchmark": benchmark, "backbone": backbone, "method": method, "seed": seed,
        "actor_contract": actor_contract(method, classic_text=bool(env.action_schema.get("classic_text"))),
        "selection_seed": selection_seed, "num_tasks": num_tasks, "tasks": registered_tasks,
        "selection": "seeded_scenario_and_within_scenario_shuffle_then_round_robin",
        "initial_checkpoint": str(initial_checkpoint), "final_checkpoint": str(final_checkpoint),
        "anchor_distribution": anchor_distribution, "anchor_policy": "initial",
        "mask": _MASK, "no_anchor_contribution": None,
        "delta_definition": "utility_visible_minus_utility_masked",
        "delta_change_definition": "delta_final_minus_delta_initial",
        "paired_state": "same_initial_policy_post_information_state_and_paid_prefix",
        "paired_randomness": "same_initial_policy_actor_rng_after_information",
        "catalog_visibility": "rebuild_from_each_visible_or_masked_history; retain_prior_public_documentation",
        "anchor_admits_native_information_errors": True,
        "episode_limits": asdict(episode_limits), "checkpoint_limits": asdict(checkpoint_limits),
        "total_limits": asdict(meter.limits),
        "checkpoint_wall_clock": "active_checkpoint_work_only",
        "total_wall_clock": "all_measurement_work_including_replay_and_checkpoint_switching",
        "claim_scope": "conditional_on_registered_initial_policy_anchor_distribution",
    })
    rows: list[dict[str, Any]] = []
    old_training = bool(policy.training)
    loaded_checkpoint: str | None = None
    stop_reason = "completed"
    policy.eval()
    try:
        with record_path.open("x", encoding="utf-8") as records:
            for position, ((original_index, task), registration) in enumerate(zip(selected, registered_tasks)):
                environment_seed = int(registration["environment_seed"])
                row: dict[str, Any] = {
                    **registration, "benchmark": benchmark, "backbone": backbone,
                    "method": method, "seed": seed, "selection_position": position,
                    "status": "partial", "anchor_found": False, "anchor_search_complete": False,
                    "information_step": None, "prefix_seconds": None, "traces": {},
                    "delta_initial": None, "delta_final": None, "delta_change": None,
                }
                for label in ("initial_visible", "initial_masked", "final_visible", "final_masked"):
                    row[f"{label}_utility"] = None
                    row[f"{label}_success"] = None
                    row[f"{label}_termination"] = None
                trajectories: dict[str, NativeTrajectory] = {}
                packet: dict[str, Any] = {}
                try:
                    meter.ensure_env_steps(1)
                    with _preserve_actor_rng():
                        if loaded_checkpoint != "initial":
                            policy.load_adapter(initial_checkpoint)
                            loaded_checkpoint = "initial"
                        with active_checkpoint("initial") as initial_meter:
                            torch.manual_seed(registration["actor_seed"])
                            visible = rollout(
                                policy, env, task, config, BudgetMeter(episode_limits, parent=initial_meter),
                                seed=environment_seed, method=method, training=False,
                                capture_information=packet,
                            )
                            natural_label = "initial_visible" if packet else "initial_natural"
                            trajectories[natural_label] = visible
                            row["anchor_found"] = bool(packet)
                            if packet:
                                row.update(information_step=len(packet["prefix"]) - 1,
                                           anchor_evidence=packet["anchor_evidence"],
                                           prefix_seconds=float(packet["prefix_seconds"]))
                            row[f"{natural_label}_termination"] = visible.termination
                            require_endpoint(visible, initial_meter)
                            row["anchor_search_complete"] = True
                            row[f"{natural_label}_utility"] = visible.utility
                            row[f"{natural_label}_success"] = int(visible.success)
                            if packet:
                                masked = _feedback_continuation(
                                    policy, env, task, config, initial_meter, packet,
                                    method=method, environment_seed=environment_seed, masked=True,
                                )
                                trajectories["initial_masked"] = masked
                                row["initial_masked_termination"] = masked.termination
                                require_endpoint(masked, initial_meter)
                                row.update(initial_masked_utility=masked.utility,
                                           initial_masked_success=int(masked.success))
                        if packet:
                            policy.load_adapter(final_checkpoint)
                            loaded_checkpoint = "final"
                            with active_checkpoint("final") as final_meter:
                                for masked_feedback, label in ((False, "final_visible"), (True, "final_masked")):
                                    trajectory = _feedback_continuation(
                                        policy, env, task, config, final_meter, packet,
                                        method=method, environment_seed=environment_seed,
                                        masked=masked_feedback,
                                    )
                                    trajectories[label] = trajectory
                                    row[f"{label}_termination"] = trajectory.termination
                                    require_endpoint(trajectory, final_meter)
                                    row[f"{label}_utility"] = trajectory.utility
                                    row[f"{label}_success"] = int(trajectory.success)
                            row["delta_initial"] = row["initial_visible_utility"] - row["initial_masked_utility"]
                            row["delta_final"] = row["final_visible_utility"] - row["final_masked_utility"]
                            row["delta_change"] = row["delta_final"] - row["delta_initial"]
                            row["status"] = "complete_pair"
                        else:
                            row.update(status="no_anchor", no_anchor_reason=visible.termination)
                except BudgetExhausted as error:
                    stop_reason = f"measurement_budget:{error}"
                    row["stop_reason"] = stop_reason
                finally:
                    for label, trajectory in trajectories.items():
                        path = output / label / f"task-{original_index:05d}.json"
                        _save(path, public_trace(trajectory))
                        row["traces"][label] = str(path)
                    row["checkpoint_resource_usage"] = checkpoint_usage()
                    row["cumulative_resource_usage"] = meter.snapshot()
                    records.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    records.flush()
                    rows.append(row)
                if row["status"] == "partial":
                    break
        if loaded_checkpoint != "final":
            policy.load_adapter(final_checkpoint)
    finally:
        policy.train(old_training)
    complete = len(rows) == num_tasks and all(row["status"] != "partial" for row in rows)
    paired = [row for row in rows if row["status"] == "complete_pair"]
    num_anchors = sum(int(row["anchor_found"]) for row in rows)
    summary = {
        "metric": "delta_change", "status": "complete" if complete else "partial",
        "complete": complete, "benchmark": benchmark, "backbone": backbone,
        "method": method, "seed": seed, "selection_seed": selection_seed,
        "initial_checkpoint": str(initial_checkpoint), "final_checkpoint": str(final_checkpoint),
        "records": str(record_path), "protocol": str(protocol_path),
        "num_registered_tasks": num_tasks, "num_recorded_tasks": len(rows),
        "num_anchor_search_complete": sum(int(row["anchor_search_complete"]) for row in rows),
        "num_anchors": num_anchors, "num_no_anchor": sum(row["status"] == "no_anchor" for row in rows),
        "paired_eligible_n": len(paired), "anchor_coverage": num_anchors / num_tasks if complete else None,
        "anchor_distribution": anchor_distribution,
        "claim_scope": "conditional_on_registered_initial_policy_anchor_distribution",
        "stop_reason": stop_reason, "checkpoint_resource_usage": checkpoint_usage(),
        "resource_usage": meter.snapshot(),
    }
    for field in ("delta_initial", "delta_final", "delta_change"):
        summary[f"mean_{field}"] = math.fsum(row[field] for row in paired) / len(paired) if complete and paired else None
    _save(summary_path, summary)
    return summary
