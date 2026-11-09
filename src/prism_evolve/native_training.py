"""Native-r3 learning and evaluation on the unmodified native benchmarks.

The actor sees public state only. A train-only demonstration supervisor records
bounded, task-related progress; the task return remains the submitted-success
time utility. Paired continuations begin at a public frontier
with both discovery and execution actions available. There are no intermediate
success gates and no corrected-action or alternate-policy execution paths.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import count
import io
import json
import logging
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid

import numpy as np
import torch

from .native_core import (
    BudgetExhausted, BudgetLimits, BudgetMeter, NativeEvent, NativeGeneration,
    NativeTrajectory, append_interaction, initial_messages, monotonic_seconds,
)
from .native_program import PROGRAM_VERSION, available_modes, build_frontier

TRAINING_METHODS = frozenset({
    "prism_evolve", "ablation_unweighted", "unstratified_2branch",
    "gigpo", "infopo", "arew", "t3", "step_dpo", "frozen_base",
})
PRISM_ACTOR_METHODS = frozenset({"prism_evolve", "ablation_unweighted", "unstratified_2branch", "frozen_base"})
GROUP_SIZES = {"gigpo": 8, "infopo": 5, "arew": 3, "t3": 5, "frozen_base": 2}
PUBLIC_PROXY_VERSION = "native-transition-feedback-v2"

FIRST_ERROR_INSTRUCTION = (
    "You are an offline training-data annotator. Use the supplied correct reference solution "
    "and actual public observations to locate the first erroneous action in this failed trajectory. "
    "Return JSON: first_error_index (zero-based integer or null), preceding_steps_valid (boolean), "
    "verifiable (boolean), and evidence (a nonempty explanation citing specific public observations). "
    "Only mark an index if the preceding actions form a valid task prefix and the error is justified "
    "by the observations. A terminal failure alone does not identify an erroneous earlier action. "
    "Use null and verifiable=false when an error cannot be localized."
)


def actor_contract(method: str, *, classic_text: bool = False) -> str:
    if method in PRISM_ACTOR_METHODS:
        if classic_text:
            from .native_text_actions import TEXT_ACTOR_CONTRACT
            return TEXT_ACTOR_CONTRACT
        return PROGRAM_VERSION
    if method in TRAINING_METHODS:
        return "official-native-action-v1"
    if method in {"pos", "worldevolver", "envprobe_simple", "envprobe_judge"}:
        return "official-controller-native-v1"
    raise ValueError(f"Unsupported native-r3 method: {method}")


def available_action_modes(schema: Mapping[str, Any]) -> tuple[str, ...]:
    if schema.get("classic_text"):
        from .native_text_actions import text_available_modes
        return text_available_modes(schema)
    return tuple(available_modes(schema["program_context"]))


def _information_signature(observation: str, receipts: Sequence[Mapping[str, Any]] = ()) -> str:
    """Use business content; receipt allocation never constitutes new evidence."""
    if receipts:
        content = [{"tool": item.get("tool"), "output": item.get("output"),
                    "succeeded": item.get("succeeded")} for item in receipts]
    else:
        try:
            content = json.loads(observation)
        except (TypeError, ValueError):
            content = observation
    return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)


def _event_information_signature(event: NativeEvent) -> str:
    facts = event.metadata.get("native_step", {}).get("native_transition", {})
    visible = facts.get("public_information_signature")
    return visible if isinstance(visible, str) and visible else _information_signature(event.observation, event.receipts)


def _anchor_public_history(messages: Sequence[Mapping[str, Any]]) -> str:
    """Conservative public information-state identity, retaining prior observations/actions."""
    return json.dumps([dict(message) for message in messages[:-1]],
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_training_return(
    trajectory: NativeTrajectory, progress: float, *, prefix_seconds: float = 0.0,
) -> float:
    """Use the design's submitted-success time utility for every task return."""
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("Training progress must be finite and bounded by [0, 1].")
    value = trajectory.utility
    trajectory.training_return = value
    trajectory.metadata["training_objective"] = {
        "definition": "U_B=Y*max(0,1-C/B)", "progress": progress,
        "progress_in_reward": False, "prefix_seconds": prefix_seconds, "training_return": value,
        "native_terminal_success": bool(trajectory.success),
        "reference_visibility": "train_only",
    }
    return value


def _training_value(trajectory: NativeTrajectory) -> float:
    if trajectory.training_return is None or not math.isfinite(trajectory.training_return):
        raise ValueError("A training trajectory requires its explicit native-r3 training return.")
    return float(trajectory.training_return)


def terminal_scoring_seconds(env: Any, step: Any) -> float:
    """ClinDiag scores an already submitted diagnosis outside deployment C."""
    if getattr(env, "benchmark", None) != "clindiag" or not step.done:
        return 0.0
    seconds = float(step.metadata.get("terminal_judge_seconds", 0.0))
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Post-submission judge time must be finite and nonnegative.")
    return seconds


def executed_information_event(event: NativeEvent) -> bool:
    return information_anchor_evidence(event)["eligible"]


def information_anchor_evidence(event: NativeEvent) -> dict[str, Any]:
    native = event.metadata.get("native_step", {})
    action = event.generation.action or {}
    mode_ok = (event.generation.mode == "I") if event.generation.mode is not None else (
        native.get("information_action_executed") is True
    )
    complete_ok = (
        event.generation.metadata.get("program_complete") is not False
        and action.get("kind") != "invalid"
    )
    mode_contract_ok = (
        action.get("mode") in {"I", None}
        and not native.get("mode_guard_blocked", False)
    )
    classic_text = bool(event.generation.metadata.get("classic_text") or native.get("classic_text"))
    conditions = (
        ("not_information_mode", mode_ok),
        ("incomplete_program", complete_ok),
        ("unconfirmed_information_action" if classic_text else "unconfirmed_readonly_program", mode_contract_ok),
        ("not_native_executed", native.get("native_executed") is True),
        ("no_information_native_action" if classic_text else "no_information_api_call",
         native.get("information_action_executed") is True),
        ("submission_action", not native.get("submitted", False)),
    )
    reason = next((reason for reason, satisfied in conditions if not satisfied), None)
    return {"eligible": reason is None,
            "reason": reason or ("actual_native_information_action" if classic_text
                                 else "actual_readonly_information_call"),
            "native_feedback_error": native.get("step_failed") is True}


@dataclass
class BranchPair:
    prefix: list[NativeEvent]
    first: NativeTrajectory
    second: NativeTrajectory | None
    split_step: int
    gate_probability: float
    stratified: bool
    partial_reason: str | None = None


@dataclass
class TokenUnit:
    generation: NativeGeneration
    messages: list[dict[str, str]]
    event_index: int
    phase: str = "action"
    label: int = 0


@dataclass
class PreferencePair:
    prompt: list[dict[str, str]]
    chosen: NativeGeneration
    rejected: NativeGeneration
    task_id: str
    first_error_index: int
    annotation: dict[str, Any]
    verification: dict[str, Any]


def _parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


class OfficialKernels:
    """Execute named upstream functions without importing incompatible verl forks."""

    def __init__(self, external_root: str | Path, method: str) -> None:
        self.provenance: dict[str, Any] = {}
        self.functions: dict[str, Any] = {}
        root = Path(external_root)
        if method == "gigpo":
            self._load(root / "gigpo-official/gigpo/core_gigpo.py", (
                "to_hashable", "summarize_group_size", "are_similar", "episode_norm_reward",
                "build_step_group", "step_norm_reward", "compute_gigpo_outcome_advantage",
            ))
        elif method == "infopo":
            self._load(root / "infopo-official/verl/trainer/ppo/core_algos.py",
                       ("compute_turn_credits", "compute_info_grpo_advantage"))
        elif method == "arew":
            self._load(root / "arew-official/verl/verl/trainer/ppo/ray_trainer.py",
                       ("_algo_config_get", "get_arew_modifier"))
        elif method == "t3":
            path = root / "arew-official/verl/search_r1/tau2_adapter/standard_space.py"
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            definitions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                           and node.name == "_update_proxy_state"]
            if len(definitions) != 1:
                raise RuntimeError("Pinned T3 source must define one _update_proxy_state method.")
            namespace: dict[str, Any] = {}
            exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                         str(path), "exec"), namespace)
            self.functions["_update_proxy_state"] = namespace["_update_proxy_state"]
            self.provenance[str(path)] = {"source": str(path),
                                          "functions": ["_update_proxy_state"]}

    def _load(self, path: Path, names: Sequence[str]) -> None:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        missing = set(names) - definitions.keys()
        if missing:
            raise RuntimeError(f"Official source {path} is missing {sorted(missing)}")
        selected = []
        repairs = []
        for name in names:
            node = definitions[name]
            node.decorator_list = []
            if name == "compute_info_grpo_advantage":
                node.body.insert(0, ast.Import(names=[ast.alias(name="logging")]))
                repairs.append("unconditional_local_logging_import")
            selected.append(node)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                  *selected], type_ignores=[])
        namespace = {"__name__": "prism_evolve.official_kernel", "torch": torch, "np": np,
                     "math": math, "logging": logging, "defaultdict": defaultdict, "Counter": Counter,
                     "uuid": uuid, "SequenceMatcher": SequenceMatcher}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        self.functions.update({name: namespace[name] for name in names})
        self.provenance[str(path)] = {"source": str(path),
                                      "functions": list(names), "engineering_repairs": repairs}

    def __getitem__(self, name: str) -> Any:
        return self.functions[name]


@lru_cache(maxsize=4)
def _t3_kernel(external_root: str) -> OfficialKernels:
    return OfficialKernels(external_root, "t3")


def _config(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    merged = dict(config)
    merged.update(dict(config.get("method_overrides", {}).get(method, {})))
    objective = "U_B=Y*max(0,1-C/B)"
    if merged.get("training_objective", objective) != objective:
        raise ValueError("Native training requires the registered common task utility U_B.")
    merged["training_objective"] = objective
    return merged


def _limits(config: Mapping[str, Any]) -> BudgetLimits:
    return BudgetLimits(
        int(config.get("episode_max_steps", 100)),
        int(config.get("episode_max_model_tokens", 8000000)),
        float(config.get("episode_max_wall_seconds", 1800.0)),
    )


def _task(task: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(task, str):
        return {"task_id": task, "scenario_id": task}
    result = dict(task)
    if "task_id" not in result:
        raise ValueError("Every task needs its native task_id.")
    result.setdefault("scenario_id", result["task_id"])
    return result


@contextmanager
def _meters(policy: Any, env: Any, meter: BudgetMeter) -> Iterator[None]:
    old_policy, old_env = getattr(policy, "meter", None), getattr(env, "meter", None)
    policy.meter = meter
    env.meter = meter
    try:
        yield
    finally:
        policy.meter, env.meter = old_policy, old_env


@contextmanager
def _preserve_actor_rng() -> Iterator[None]:
    """Reset simulator randomness without resetting independent model sampling."""
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _restore_environment(env: Any, snapshot: Any) -> None:
    with _preserve_actor_rng():
        env.restore_snapshot(snapshot)


def _global_budget_limited(meter: BudgetMeter, reason: str) -> bool:
    if meter.parent is None:
        return False
    parent = meter.parent
    if reason == "model_tokens":
        return parent.remaining_model_tokens <= meter.limits.max_model_tokens - meter.total_model_tokens
    if reason == "environment_steps":
        return parent.remaining_env_steps <= meter.limits.max_env_steps - meter.env_steps
    if reason == "wall_seconds":
        return parent.elapsed_seconds >= parent.limits.max_wall_seconds
    return False


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def public_trace(trajectory: NativeTrajectory, *, method: str | None = None) -> dict[str, Any]:
    """Save actual actor inputs plus separate native execution evidence and costs."""
    events = []
    for index, event in enumerate(trajectory.events):
        record = {
            "index": index, "messages_before": event.messages_before,
            "action": event.generation.action, "response": event.generation.text,
            "mode": event.generation.mode, "observation": event.observation,
            "prompt_token_ids": list(event.generation.prompt_ids),
            "response_token_ids": list(event.generation.response_ids),
            "mode_prompt_token_ids": list(event.generation.mode_prompt_ids),
            "mode_token_id": event.generation.mode_token_id,
            "sampling": {key: event.generation.metadata[key] for key in (
                "prompt_version", "sampling_log_probs", "mode_log_probability", "gate_probability",
                "actor_contract", "native_action_schema",
                "mode_temperature", "temperature", "native_prefix_token_ids", "forced_mode",
                "max_new_tokens", "output_limit_reached",
                "eos_token_ids", "eos_reached", "wall_deadline_reached", "generation_stop_reason",
                "generation_limit_sources", "requested_max_new_tokens",
                "program_complete", "program_eos_accepted", "program_compile_seconds",
                "counterfactual_support",
            ) if key in event.generation.metadata},
            "native_step": dict(event.metadata.get("native_step", {})),
            "receipts": deepcopy(event.receipts),
            "input_tokens": event.generation.input_tokens,
            "output_tokens": event.generation.output_tokens,
            "model_seconds": event.generation.elapsed_seconds,
            "environment_seconds": event.environment_seconds,
        }
        if event.generation.metadata.get("classic_text"):
            record["response_representation"] = "model_sampled_native_action"
            record["action_representation"] = "native_text"
            record["text_action_contract"] = {
                key: value for key, value in event.generation.metadata.get("text_action_contract", {}).items()
                if key != "schema_json"
            }
            record["information_action_executed"] = executed_information_event(event)
            record["information_anchor_evidence"] = information_anchor_evidence(event)
        elif "program_complete" in event.generation.metadata:
            program_context = event.generation.metadata.get("program_context", {})
            appworld = program_context.get("benchmark") == "appworld"
            record["response_representation"] = "model_sampled_program_json"
            record["action_representation"] = "compiled_native_python" if appworld else "compiled_native_action"
            record["compiler_output_model_tokens"] = 0
            record["program_contract"] = {
                key: value for key, value in event.generation.metadata.get("program_contract", {}).items()
                if key != "schema_json"
            }
            record["program_context"] = deepcopy(event.generation.metadata.get("program_context"))
            record["information_action_executed"] = executed_information_event(event)
            record["information_anchor_evidence"] = information_anchor_evidence(event)
        for key in ("belief_generation", "belief_messages", "critique", "elapsed_after", "public_error",
                    "training_progress", "counterfactual_information"):
            if key in event.metadata:
                value = event.metadata[key]
                record[key] = asdict(value) if isinstance(value, NativeGeneration) else value
        events.append(record)
    return {"task_id": trajectory.task_id, "scenario_id": trajectory.scenario_id,
            "success": trajectory.success, "utility": trajectory.utility,
            "training_return": trajectory.training_return,
            "termination": trajectory.termination, "elapsed_seconds": trajectory.elapsed_seconds,
            "budget_seconds": trajectory.budget_seconds, "events": events,
            "metadata": trajectory.metadata}


def _public_error(step_metadata: Mapping[str, Any], observation: str) -> bool:
    if isinstance(step_metadata.get("step_failed"), bool):
        return step_metadata["step_failed"]
    if step_metadata.get("step_failed") is True or step_metadata.get("public_error") is True:
        return True
    try:
        parsed = json.loads(observation)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed.get("success") is False or bool(parsed.get("error"))
    return observation.startswith("Traceback (most recent call last):")



def _distribution(values: Sequence[int]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "total": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"count": len(ordered), "total": sum(ordered), "mean": sum(ordered) / len(ordered),
            "p50": percentile(0.5), "p95": percentile(0.95), "max": ordered[-1]}


def program_execution_summary(events: Sequence[NativeEvent]) -> dict[str, Any]:
    """Report sampled tokens, actual native returns and terminal outcomes separately."""
    total = len(events)
    native = [event.metadata.get("native_step", {}) for event in events]
    complete = sum(
        event.generation.metadata.get("program_complete") is True
        or (event.generation.metadata.get("program_complete") is None
            and event.generation.metadata.get("action_boundary_reached") is True)
        for event in events
    )
    truncated = sum(
        event.generation.metadata.get("program_complete") is False
        or (event.generation.metadata.get("program_complete") is None
            and event.generation.metadata.get("action_boundary_reached") is False)
        for event in events
    )
    parsed = sum(
        isinstance(event.generation.action, Mapping)
        and event.generation.action.get("kind") not in {None, "invalid"}
        and step.get("validation_stage") != "parse_error"
        for event, step in zip(events, native)
    )
    finite_samples = 0
    finite_joint_samples = 0
    probability_steps = 0
    aligned_samples = 0
    for event in events:
        values = event.generation.metadata.get("sampling_log_probs")
        if isinstance(values, (list, tuple)):
            probability_steps += 1
            aligned_samples += int(len(values) == len(event.generation.response_ids))
            finite_body = bool(values) and all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in values
            )
            finite_samples += int(finite_body)
            mode_log_probability = event.generation.metadata.get("mode_log_probability")
            finite_mode = event.generation.mode is None or (
                isinstance(mode_log_probability, (int, float)) and math.isfinite(mode_log_probability)
            )
            finite_joint_samples += int(finite_body and finite_mode
                                        and len(values) == len(event.generation.response_ids))
    metric_keys = (
        "attempted_api_calls", "successful_api_calls", "successful_business_calls",
        "successful_documentation_calls", "successful_submission_calls",
    )
    tool = {key: sum(int(step.get(key, 0)) for step in native) for key in metric_keys}
    tool["metric_observed_steps"] = {
        key: sum(key in step for step in native) for key in metric_keys
    }
    tool.update(
        native_execution_successes=sum(step.get("native_execution_succeeded") is True for step in native),
        native_execution_failures=sum(step.get("native_executed") is True and step.get("step_failed") is True
                                      for step in native),
        successful_retrieval_actions=sum(
            (event.generation.action or {}).get("kind") == "retrieve_tools"
            and step.get("native_execution_succeeded") is True
            for event, step in zip(events, native)
        ),
        executed_information_actions=sum(executed_information_event(event) for event in events),
    )
    mode_events = [event for event in events if event.generation.mode is not None]
    mode_ok = sum(
        event.generation.mode in {"I", "E"}
        and isinstance(event.generation.metadata.get("gate_probability"), (int, float))
        and math.isfinite(event.generation.metadata["gate_probability"])
        for event in mode_events
    )
    guard_blocks = sum(step.get("mode_guard_blocked") is True for step in native)
    progress_events = sum(event.metadata.get("native_progress", {}).get("progress") is True for event in events)
    token_usage = {
        "action_prompt_tokens": [len(event.generation.prompt_ids) for event in events],
        "action_output_tokens": [len(event.generation.response_ids) for event in events],
        "mode_prompt_tokens": [len(event.generation.mode_prompt_ids) for event in events],
        "mode_output_tokens": [int(event.generation.mode_token_id is not None) for event in events],
        "finite_log_probability_steps": finite_joint_samples,
        "probability_steps": probability_steps,
        "prompt_tokens": _distribution([len(event.generation.prompt_ids) for event in events]),
        "billed_input_tokens": _distribution([event.generation.input_tokens for event in events]),
        "output_tokens": _distribution([len(event.generation.response_ids) for event in events]),
        "response_tokens": _distribution([len(event.generation.response_ids) for event in events]),
        "empty_response_steps": sum(not event.generation.response_ids for event in events),
        "complete_program_steps": complete,
        "length_capped_steps": sum(event.generation.metadata.get("output_limit_reached") is True for event in events),
        "eos_stopped_steps": sum(event.generation.metadata.get("eos_reached") is True for event in events),
        "finite_sampling_logprob_steps": finite_samples,
        "aligned_sampling_logprob_steps": aligned_samples,
        "stop_reasons": dict(Counter(str(event.generation.metadata.get("generation_stop_reason", "unrecorded"))
                                     for event in events)),
    }
    return {
        "counting_unit": "trajectory_events; shared training prefixes repeat in trajectory summaries",
        "total_events": total, "complete_programs": complete,
        "action_parse_successes": parsed, "action_parse_success_rate": parsed / total if total else 0.0,
        "truncated_programs": truncated,
        "sampled_information_labels": sum(event.generation.mode == "I" for event in events),
        "sampled_execution_labels": sum(event.generation.mode == "E" for event in events),
        "mode_probe_success_rate": mode_ok / len(mode_events) if mode_events else None,
        "mode_guard_interceptions": guard_blocks,
        "mode_guard_interception_rate": guard_blocks / total if total else 0.0,
        "progress_events": progress_events,
        "progress_event_rate": progress_events / total if total else 0.0,
        "native_terminal_successes": sum(event.metadata.get("native_terminal_success") is True for event in events),
        "token_usage": token_usage, "tool_execution": tool,
        **{key: tool[key] for key in metric_keys},
    }


def _decision_messages(
    messages: Sequence[Mapping[str, str]], trajectory: NativeTrajectory, limits: BudgetLimits,
    meter: BudgetMeter,
) -> list[dict[str, str]]:
    result = [dict(message) for message in messages]
    result.append({"role": "user", "content": json.dumps({
        "remaining_environment_steps": max(0, limits.max_env_steps - len(trajectory.events)),
        "remaining_wall_seconds": max(0.0, trajectory.budget_seconds - trajectory.elapsed_seconds),
        "remaining_model_tokens": max(0, limits.max_model_tokens - meter.total_model_tokens),
    }, sort_keys=True, separators=(",", ":"))})
    return result


def execution_result_summary(env: Any, trajectory: NativeTrajectory) -> dict[str, Any]:
    """Summarize existing public execution evidence without additional calls."""
    first_break = None
    for index, event in enumerate(trajectory.events):
        native = event.metadata.get("native_step", {})
        if not native.get("step_failed"):
            continue
        first_break = {"step": index + 1, "category": native.get("validation_stage", "native"),
                       "reason": native.get("validation_error") or native.get("native_error_kind")
                                 or native.get("failure_reason") or "native_call_failed"}
        break
    if first_break is None and not trajectory.success and trajectory.termination != "prefix_ready":
        first_break = {"step": len(trajectory.events), "category": trajectory.termination,
                       "reason": "The recorded public execution did not reach native terminal success."}
    if env.action_schema.get("classic_text"):
        return {"first_recorded_failure": first_break,
                "native_action_count": len(trajectory.events),
                "feasible_modes_at_end": list(available_action_modes(env.action_schema)),
                "native_success": bool(trajectory.success)}
    context = env.action_schema["program_context"]
    frontier = build_frontier(context)
    public = env.execution_state.public_context()
    receipts = public["receipts"]
    writes = [receipt for receipt in receipts if receipt.get("succeeded")
              and receipt.get("kind") == "call" and receipt.get("method") in {"POST", "PUT", "PATCH", "DELETE"}
              and not receipt["tool"].endswith((".login", ".logout"))]
    return {"first_recorded_failure": first_break,
            "successful_business_write_calls": len(writes),
            "public_target_binding_obtained": bool(frontier.get("target_answer_references")),
            "public_catalog_missing_required_inputs": len(frontier["missing_inputs"]),
            "feasible_modes_at_end": frontier["feasible_modes"],
            "native_success": bool(trajectory.success)}


def rollout(
    policy: Any, env: Any, task: str | Mapping[str, Any], config: Mapping[str, Any],
    meter: BudgetMeter, *, seed: int, method: str, training: bool,
    context: Mapping[str, Any] | None = None, prefix: Sequence[NativeEvent] = (),
    prefix_messages: Sequence[Mapping[str, str]] | None = None, prefix_seconds: float = 0.0,
    forced_mode: str | None = None, stop_after: int | None = None,
    first_decision_messages: Sequence[Mapping[str, str]] | None = None,
    snapshots: list[Any] | None = None,
    stop_at_frontier: bool = False, progress_supervisor: Any | None = None,
    capture_information: dict[str, Any] | None = None,
) -> NativeTrajectory:
    """Run the actual actor until native termination or an explicit experiment cap.

    Runtime errors, including CUDA OOM, propagate to the formal experiment.
    A branch prefix stops at the first public state where both modes are feasible.
    The train-only supervisor never contributes to actor messages or schemas.
    """
    actor_contract(method)
    gated_actor = method in PRISM_ACTOR_METHODS
    if forced_mode is not None and not gated_actor:
        raise ValueError("Only PRISM, its initial frozen control and its own ablations may force an I/E mode.")
    task_info = _task(task)
    if capture_information is not None:
        if training or task_info.get("split") != "dev":
            raise ValueError("Feedback-anchor capture is restricted to development evaluation.")
        if capture_information:
            raise ValueError("Feedback-anchor capture requires an empty output packet.")
    if progress_supervisor is not None and (not training or task_info.get("split") != "train"):
        raise ValueError("Progress supervision belongs only to registered training rollouts.")
    limits = _limits(config)
    with _meters(policy, env, meter):
        if context is None:
            with _preserve_actor_rng():
                context = env.reset(str(task_info["task_id"]), seed=seed)
        trajectory = NativeTrajectory(
            str(task_info["task_id"]), str(context.get("scenario_id", task_info["scenario_id"])),
            events=list(prefix), elapsed_seconds=float(prefix_seconds),
            budget_seconds=limits.max_wall_seconds,
            metadata={"environment_seed": seed,
                      "actor_contract": actor_contract(method, classic_text=bool(env.action_schema.get("classic_text")))},
        )
        messages = ([dict(item) for item in prefix_messages] if prefix_messages is not None
                    else initial_messages(context))
        started = monotonic_seconds()
        post_submission_seconds = 0.0
        meter.started = started - float(prefix_seconds)
        first_action = True
        progress_value = float(progress_supervisor.score(env)) if progress_supervisor is not None else 0.0
        seen_information = {_event_information_signature(event)
                            for event in prefix if not event.metadata.get("public_error")}
        if isinstance(context.get("initial_information_signature"), str):
            seen_information.add(context["initial_information_signature"])
        t3_state = SimpleNamespace(
            proxy_reason_history=[], hard_bad_streak=0, soft_no_progress_streak=0,
            max_soft_no_progress_streak=0, query_counts=0, cut=0.0, stalling_counts=0,
            must_stop=0.0, hard_truncation=0, soft_truncation=0, t3_truncation_turn=0,
            progress_mode="legacy", progress_min_turn=int(config.get("t3_min_turn", 6)),
            hard_tolerate_num=int(config.get("t3_hard_tolerance", 2)),
            tolerate_num=int(config.get("t3_no_progress_window", 3)),
        ) if method == "t3" else None
        try:
            while len(trajectory.events) < limits.max_env_steps:
                trajectory.elapsed_seconds = (float(prefix_seconds) if first_action
                                              else prefix_seconds + monotonic_seconds() - started)
                if trajectory.elapsed_seconds >= trajectory.budget_seconds:
                    trajectory.termination = "budget:episode_wall_seconds"
                    break
                if stop_after is not None and len(trajectory.events) >= stop_after:
                    trajectory.termination = "prefix_ready"
                    break
                meter.ensure_env_steps(1)
                if first_action and first_decision_messages is not None:
                    before = [dict(item) for item in first_decision_messages]
                else:
                    before = _decision_messages(messages, trajectory, limits, meter)
                # Derive the live grammar from the complete public history. The
                # policy may compact its model prompt after extracting this state.
                schema = env.schema_for_history(before) if gated_actor else env.action_schema
                modes = available_action_modes(schema) if gated_actor else ()
                if gated_actor and not modes:
                    trajectory.termination = "no_public_feasible_action"
                    break
                if stop_at_frontier and set(modes) == {"I", "E"}:
                    trajectory.termination = "prefix_ready"
                    break
                if snapshots is not None:
                    snapshot_started = monotonic_seconds()
                    snapshots.append(env.save_snapshot())
                    snapshot_seconds = monotonic_seconds() - snapshot_started
                    started += snapshot_seconds
                    meter.started += snapshot_seconds
                greedy = (not training) and bool(config.get("evaluation_greedy", False))
                if gated_actor:
                    generation = policy.act(before, schema, forced_mode=forced_mode,
                                            greedy=greedy, role="actor")
                else:
                    generation = policy.act_native(before, schema, greedy=greedy, role="actor")
                first_action = False
                forced_mode = None
                if not isinstance(generation.action, Mapping):
                    raise RuntimeError("The actor returned no compiled native action.")
                env_started = monotonic_seconds()
                step = env.step(generation.action) if gated_actor else env.step_native(generation.action)
                post_submission_seconds += terminal_scoring_seconds(env, step)
                event = NativeEvent(
                    messages_before=before, generation=generation, observation=step.observation,
                    environment_seconds=monotonic_seconds() - env_started,
                    metadata={"public_error": _public_error(step.metadata, step.observation),
                              "native_step": dict(step.metadata), "native_done": bool(step.done),
                              "native_terminal_success": bool(step.done and step.success)},
                    receipts=deepcopy(step.metadata.get("receipts", [])),
                )
                trajectory.events.append(event)
                append_interaction(messages, generation, step)
                facts = step.metadata.get("native_transition", {})
                information_signature = _event_information_signature(event)
                retrieval_receipts = [item for item in event.receipts if item.get("kind") == "discover"]
                if step.metadata.get("native_action") == "retrieve_tools" and "new_tool_count" in step.metadata:
                    has_new_content = int(step.metadata["new_tool_count"]) > 0
                elif retrieval_receipts and all("new_tool_count" in item for item in retrieval_receipts):
                    has_new_content = any(item["new_tool_count"] > 0 for item in retrieval_receipts)
                else:
                    has_new_content = information_signature not in seen_information
                new_information = bool(facts.get("new_information_candidate") and not facts.get("hard_bad")
                                       and has_new_content)
                if new_information:
                    seen_information.add(information_signature)
                # Navigation and option changes are physical transitions, but
                # they must not turn every exploration cycle into task progress.
                task_progress = (bool(facts.get("task_progress")) if schema.get("classic_text")
                                 else bool(facts.get("state_changed")))
                progress = bool(task_progress or new_information or facts.get("final_success"))
                event.metadata["native_progress"] = {
                    "progress": progress,
                    "new_information": new_information,
                    "state_changed": bool(facts.get("state_changed")),
                    "task_progress": task_progress,
                }
                if progress_supervisor is not None:
                    next_progress = float(progress_supervisor.score(env))
                    event.metadata["training_progress"] = {
                        "before": progress_value, "after": next_progress,
                        "delta": next_progress - progress_value, "visibility": "train_supervision_only",
                    }
                    progress_value = next_progress
                t3_stop = False
                if method in {"arew", "t3"}:
                    action_label = (1 if task_progress or facts.get("final_success") or new_information
                                    else -1 if facts.get("hard_bad") else 0)
                    event.metadata["critique"] = {"action_label": action_label,
                                                  "source": "native_information_progress_as_only", "facts": facts}
                    if t3_state is not None:
                        t3_state.query_counts = len(trajectory.events)
                        t3_stop = _t3_kernel(str(config["external_root"]))["_update_proxy_state"](
                            t3_state, progress=progress, hard_bad=bool(facts.get("hard_bad")),
                            reason="native_error" if facts.get("hard_bad") else "no_progress" if not progress else None,
                            validate=not training,
                        )
                trajectory.elapsed_seconds = max(0.0, prefix_seconds + monotonic_seconds() - started
                                                 - post_submission_seconds)
                event.metadata["elapsed_after"] = trajectory.elapsed_seconds
                if capture_information is not None and not capture_information and not step.done:
                    evidence = information_anchor_evidence(event)
                    if evidence["eligible"]:
                        # Capture after the real information action and before
                        # any next model/belief update. Diagnostic snapshot work
                        # stays on the physical ledger, outside the paid prefix C.
                        capture_started = monotonic_seconds()
                        try:
                            paid_seconds = trajectory.elapsed_seconds
                            paid_usage = meter.snapshot()
                            rng_cpu = torch.random.get_rng_state().clone()
                            rng_cuda = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                                        if torch.cuda.is_initialized() else None)
                            with _preserve_actor_rng():
                                snapshot = env.save_snapshot()
                            capture_information.update(
                                snapshot=snapshot,
                                prefix=deepcopy(trajectory.events), messages=deepcopy(messages),
                                prefix_seconds=paid_seconds, prefix_usage=paid_usage,
                                rng_cpu=rng_cpu, rng_cuda=rng_cuda,
                                anchor_evidence=deepcopy(evidence), context=deepcopy(dict(context)),
                            )
                        finally:
                            capture_seconds = monotonic_seconds() - capture_started
                            started += capture_seconds
                            meter.started += capture_seconds
                if step.done:
                    trajectory.metadata["native_terminal_success"] = bool(step.success)
                    trajectory.success = bool(step.success) and trajectory.elapsed_seconds <= trajectory.budget_seconds
                    trajectory.termination = "native_terminal" if trajectory.success else "native_failure"
                    break
                if t3_stop:
                    trajectory.termination = "t3_native_no_progress"
                    trajectory.metadata["truncation"] = vars(t3_state).copy()
                    break
            else:
                trajectory.termination = "budget:episode_environment_steps"
        except BudgetExhausted as error:
            trajectory.termination = f"budget:{error}"
            trajectory.metadata["global_budget_limited"] = _global_budget_limited(meter, str(error))
        trajectory.elapsed_seconds = max(0.0, prefix_seconds + monotonic_seconds() - started
                                         - post_submission_seconds)
        if post_submission_seconds:
            trajectory.metadata["terminal_judge_seconds"] = post_submission_seconds
        if trajectory.termination != "prefix_ready":
            official = dict(env.finalize_evaluation())
            trajectory.metadata["official_evaluation"] = official
            trajectory.metadata["official_evaluation_scope"] = "final_state_at_trajectory_stop"
            # Cutoff-state scoring is separate from the submitted, budget-compliant reward.
            trajectory.metadata.setdefault("native_terminal_success", False)
            trajectory.metadata["budget_compliant_success"] = bool(trajectory.success)
        trajectory.metadata["final_messages"] = messages
        trajectory.metadata["program_outcomes"] = program_execution_summary(trajectory.events)
        trajectory.metadata["execution_result"] = execution_result_summary(env, trajectory)
        trajectory.metadata["resource_usage"] = meter.snapshot()
        if training and trajectory.termination != "prefix_ready":
            compute_training_return(trajectory, progress_value, prefix_seconds=prefix_seconds)
        return trajectory


def _continuation_meter(
    limits: BudgetLimits, parent: BudgetMeter, prefix_usage: Mapping[str, Any],
) -> BudgetMeter:
    meter = BudgetMeter(limits, parent=parent)
    # Logical trajectory accounting includes the shared prefix. Physical parent
    # accounting was already charged when that prefix was actually generated.
    for key in ("env_steps", "input_tokens", "output_tokens", "forward_tokens"):
        setattr(meter, key, int(prefix_usage[key]))
    for key in ("native_api_calls", "successful_native_api_calls", "replayed_api_calls",
                "model_seconds", "environment_seconds"):
        if key in prefix_usage:
            setattr(meter, key, prefix_usage[key])
    if "calls_by_role" in prefix_usage:
        meter.calls_by_role = dict(prefix_usage["calls_by_role"])
    return meter


def collect_branch_pair(
    policy: Any, env: Any, task: str | Mapping[str, Any], config: Mapping[str, Any],
    meter: BudgetMeter, *, seed: int, stratified: bool,
    progress_supervisor: Any | None = None,
) -> BranchPair:
    """Use identical logical limits, native state and root messages for both branches."""
    task_info = _task(task)
    limits = _limits(config)
    prefix_meter = BudgetMeter(limits, parent=meter)
    with _meters(policy, env, prefix_meter), _preserve_actor_rng():
        context = env.reset(str(task_info["task_id"]), seed=seed)
    prefix_trajectory = rollout(policy, env, task_info, config, prefix_meter, seed=seed,
                                method="prism_evolve", training=True, context=context,
                                stop_at_frontier=True, progress_supervisor=progress_supervisor)
    prefix = prefix_trajectory.events
    split_step = len(prefix)
    if prefix_trajectory.termination != "prefix_ready":
        reason = "global_budget_during_prefix" if prefix_trajectory.metadata.get("global_budget_limited") else None
        return BranchPair(prefix, prefix_trajectory, None, len(prefix), 0.0, stratified, reason)
    with _meters(policy, env, meter):
        snapshot = env.save_snapshot()
    messages = prefix_trajectory.metadata["final_messages"]
    root_messages = _decision_messages(messages, prefix_trajectory, limits, prefix_meter)
    prefix_usage = prefix_trajectory.metadata["resource_usage"]
    first_meter = _continuation_meter(limits, meter, prefix_usage)
    first_seed = (2 * seed + 1) % (2**63 - 1)
    second_seed = (2 * seed + 2) % (2**63 - 1)
    with _preserve_actor_rng():
        torch.manual_seed(first_seed)
        first = rollout(policy, env, task_info, config, first_meter, seed=seed, method="prism_evolve", training=True,
                        context=context, prefix=prefix, prefix_messages=messages,
                        prefix_seconds=prefix_trajectory.elapsed_seconds,
                        forced_mode="I" if stratified else None, first_decision_messages=root_messages,
                        progress_supervisor=progress_supervisor)
    first.metadata["continuation_seed"] = first_seed
    if first.metadata.get("global_budget_limited"):
        return BranchPair(prefix, first, None, split_step, 0.0, stratified, "global_budget_during_first_branch")
    try:
        with _meters(policy, env, meter):
            _restore_environment(env, snapshot)
    except BudgetExhausted as error:
        return BranchPair(prefix, first, None, split_step, 0.0, stratified, f"restore_budget:{error}")
    second_meter = _continuation_meter(limits, meter, prefix_usage)
    with _preserve_actor_rng():
        torch.manual_seed(second_seed)
        second = rollout(policy, env, task_info, config, second_meter, seed=seed, method="prism_evolve", training=True,
                         context=context, prefix=prefix, prefix_messages=messages,
                         prefix_seconds=prefix_trajectory.elapsed_seconds,
                         forced_mode="E" if stratified else None, first_decision_messages=root_messages,
                         progress_supervisor=progress_supervisor)
    second.metadata["continuation_seed"] = second_seed
    if (len(first.events) <= split_step or len(second.events) <= split_step
            or second.metadata.get("global_budget_limited")):
        return BranchPair(prefix, first, second, split_step, 0.0, stratified, "incomplete_branch_pair")
    probability = float(first.events[split_step].generation.metadata["gate_probability"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise FloatingPointError("The recorded gate probability is invalid.")
    return BranchPair(prefix, first, second, split_step, probability, stratified)


@dataclass(frozen=True)
class ScoreTerm:
    kind: str
    generation: NativeGeneration
    coefficient: float

    @property
    def forward_tokens(self) -> int:
        if self.kind in {"mode", "gate_probability"}:
            return len(self.generation.mode_prompt_ids)
        return len(self.generation.prompt_ids) + len(self.generation.response_ids) - 1


def _event_terms(generation: NativeGeneration, coefficient: float, include_mode: bool = True) -> Iterator[ScoreTerm]:
    if coefficient == 0.0:
        return
    if not math.isfinite(coefficient):
        raise FloatingPointError("Nonfinite terminal-utility coefficient.")
    if include_mode and generation.mode is not None and generation.metadata.get("gate_active", False):
        yield ScoreTerm("mode", generation, coefficient)
    yield ScoreTerm("body", generation, coefficient)


def _loss_specs(sample: BranchPair, baseline: float, unweighted: bool) -> list[ScoreTerm]:
    if sample.partial_reason is not None:
        raise ValueError("An incomplete branch pair cannot enter the policy objective.")
    result: list[ScoreTerm] = []
    if sample.second is None:
        for event in sample.first.events:
            result.extend(_event_terms(event.generation, -(_training_value(sample.first) - baseline)))
        return result
    first, second, split = sample.first, sample.second, sample.split_step
    difference = _training_value(first) - _training_value(second)
    first_weight = sample.gate_probability if sample.stratified and not unweighted else 0.5
    second_weight = 1.0 - first_weight
    prefix_return = first_weight * _training_value(first) + second_weight * _training_value(second)
    for event in sample.prefix:
        result.extend(_event_terms(event.generation, -(prefix_return - baseline)))
    if sample.stratified and difference != 0.0:
        result.append(ScoreTerm("gate_probability", first.events[split].generation, -difference))
    for index, event in enumerate(first.events[split:]):
        result.extend(_event_terms(event.generation, -first_weight * difference,
                                   include_mode=not sample.stratified or index != 0))
    for index, event in enumerate(second.events[split:]):
        result.extend(_event_terms(event.generation, second_weight * difference,
                                   include_mode=not sample.stratified or index != 0))
    return result


def stratified_loss_terms(
    policy: Any, sample: BranchPair, *, baseline: float = 0.0, unweighted: bool = False,
) -> Iterator[torch.Tensor]:
    for term in _loss_specs(sample, baseline, unweighted):
        if term.kind == "mode":
            score = policy.mode_score(term.generation)
        elif term.kind == "gate_probability":
            score = policy.gate_probability(term.generation)
        else:
            score = policy.score(term.generation, include_mode=False)
        yield term.coefficient * score


def _update_pair(
    policy: Any, optimizer: Any, sample: BranchPair, meter: BudgetMeter,
    config: Mapping[str, Any], baseline: float, unweighted: bool,
) -> dict[str, Any]:
    specs = _loss_specs(sample, baseline, unweighted)
    optimizer.zero_grad(set_to_none=True)
    if not specs:
        return {"updated": False, "learning_outcome": "zero_training_signal", "loss": 0.0,
                "score_terms": 0, "scoring_forward_tokens": 0, "nonzero_gradient": False,
                "parameters_changed": False, "parameter_delta_l2": 0.0}
    required_tokens = sum(term.forward_tokens for term in specs)
    # Reserve the exact recorded scoring inputs before accumulating any gradient.
    # No guessed rollout multiplier or token floor reduces a native trajectory.
    meter.ensure_forward(required_tokens, role="complete_branch_update")
    started = monotonic_seconds()
    total_loss = 0.0
    try:
        for loss in stratified_loss_terms(policy, sample, baseline=baseline, unweighted=unweighted):
            if not torch.isfinite(loss.detach()).all():
                raise FloatingPointError("Nonfinite native training loss.")
            total_loss += float(loss.detach().item())
            if loss.requires_grad:
                loss.backward()
            del loss
        parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        clip = float(config.get("grad_clip_norm", 1.0))
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip if clip > 0 else float("inf"), error_if_nonfinite=True)
        gradient_norm = float(norm.detach().item())
        if gradient_norm == 0.0:
            optimizer.zero_grad(set_to_none=True)
            return {"updated": False, "learning_outcome": "zero_gradient", "loss": total_loss,
                    "score_terms": len(specs), "scoring_forward_tokens": required_tokens,
                    "gradient_l2_norm_before_clip": 0.0, "nonzero_gradient": False,
                    "parameters_changed": False, "parameter_delta_l2": 0.0}
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer.step()
        with torch.no_grad():
            parameter_delta = math.sqrt(sum(float((parameter.detach() - previous).float().square().sum().item())
                                            for parameter, previous in zip(parameters, before)))
        del before
        meter.record_update(monotonic_seconds() - started)
        optimizer.zero_grad(set_to_none=True)
        return {"updated": True, "learning_outcome": "parameters_updated" if parameter_delta > 0 else "no_parameter_change",
                "loss": total_loss, "score_terms": len(specs), "scoring_forward_tokens": required_tokens,
                "gradient_l2_norm_before_clip": gradient_norm, "nonzero_gradient": True,
                "parameters_changed": parameter_delta > 0, "parameter_delta_l2": parameter_delta,
                "optimizer_step": meter.gradient_steps}
    except BaseException:
        # A failed update is never completed with a partial set of loss terms.
        optimizer.zero_grad(set_to_none=True)
        raise


def _token_units(trajectory: NativeTrajectory) -> list[TokenUnit]:
    units = []
    for index, event in enumerate(trajectory.events):
        critique = event.metadata.get("critique", {})
        units.append(TokenUnit(event.generation, event.messages_before, index,
                               label=int(critique.get("action_label", 0))))
        if "belief_generation" in event.metadata:
            units.append(TokenUnit(event.metadata["belief_generation"], event.metadata["belief_messages"],
                                   index, phase="belief", label=int(critique.get("belief_label", 0))))
    return units


def _sampled_log_probs(policy: Any, generation: NativeGeneration) -> torch.Tensor:
    body = generation.metadata.get("sampling_log_probs")
    if body is not None and len(body) == len(generation.response_ids):
        values = list(body)
        if generation.mode is not None and generation.metadata.get("gate_active", False):
            mode = generation.metadata.get("mode_log_probability")
            if mode is None:
                raise ValueError("Sampled mode lacks its behavior-policy log probability.")
            values.insert(0, float(mode))
        return torch.tensor(values, dtype=torch.float32)
    raise ValueError("Sampled response lacks aligned behavior-policy token log probabilities.")


def _group_normalized(values: Sequence[float], *, standardize: bool = True) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.numel() < 2:
        raise ValueError("Group-relative algorithms require a complete multi-rollout group.")
    centered = tensor - tensor.mean()
    return centered / (tensor.std(unbiased=True) + 1e-6) if standardize else centered


def group_advantages(
    method: str, policy: Any, trajectories: Sequence[NativeTrajectory],
    units: Sequence[Sequence[TokenUnit]], old_log_probs: Sequence[Sequence[torch.Tensor]],
    kernels: OfficialKernels | None, config: Mapping[str, Any], epoch: int,
) -> list[list[torch.Tensor]]:
    """Return assistant-token advantages; groups contain one native task/initial state."""
    if len({(trajectory.task_id, trajectory.metadata.get("environment_seed")) for trajectory in trajectories}) != 1:
        raise ValueError("A rollout group must share a native task and initial environment seed.")
    rewards = [trajectory.utility for trajectory in trajectories]
    if method == "gigpo":
        assert kernels is not None
        rows, references, observations, discounted, indices, traj_indices = [], [], [], [], [], []
        gamma = float(config.get("gigpo_gamma", 0.95))
        for i, (trajectory, row_units) in enumerate(zip(trajectories, units)):
            for j, unit in enumerate(row_units):
                rows.append(old_log_probs[i][j])
                references.append((i, j))
                observations.append(_anchor_public_history(trajectory.events[unit.event_index].messages_before))
                discounted.append(rewards[i] * gamma ** (len(trajectory.events) - 1 - unit.event_index))
                indices.append("same_task_initial_state")
                traj_indices.append(str(i))
        width = max(row.numel() for row in rows)
        mask = torch.zeros((len(rows), width))
        reward_matrix = torch.zeros_like(mask)
        for index, (row, (i, _)) in enumerate(zip(rows, references)):
            mask[index, :row.numel()] = 1
            reward_matrix[index, row.numel() - 1] = rewards[i]
        with redirect_stdout(io.StringIO()), torch.no_grad():
            normalization = str(config.get("gigpo_episode_normalization", "official_step_weighted"))
            if normalization not in {"official_step_weighted", "paper_trajectory_equal"}:
                raise ValueError("Unknown GiGPO episode normalization.")
            mode = str(config.get("gigpo_mode", "mean_norm"))
            if mode not in {"mean_norm", "mean_std_norm"}:
                raise ValueError("Unknown GiGPO advantage mode.")
            indices_array = np.array(indices, dtype=object)
            episode_advantage = kernels["episode_norm_reward"](
                reward_matrix, mask, indices_array, np.array(traj_indices, dtype=object),
                remove_std=mode == "mean_norm",
                compute_mean_std_cross_steps=normalization == "official_step_weighted")
            anchors = kernels["build_step_group"](np.array(observations, dtype=object), indices_array,
                                                  enable_similarity=False)
            step_advantage = kernels["step_norm_reward"](
                torch.tensor(discounted), mask, anchors, remove_std=mode == "mean_norm")
            advantage = episode_advantage + float(config.get("gigpo_step_weight", 1.0)) * step_advantage
        result = [[torch.empty(0) for _ in row] for row in units]
        for index, (i, j) in enumerate(references):
            result[i][j] = advantage[index, :rows[index].numel()].clone()
        return result
    flat_lengths = [sum(value.numel() for value in row) for row in old_log_probs]
    width = max(flat_lengths)
    mask = torch.zeros((len(trajectories), width))
    reward_matrix = torch.zeros_like(mask)
    for i, length in enumerate(flat_lengths):
        mask[i, :length] = 1
        reward_matrix[i, length - 1] = rewards[i]
    if method == "infopo":
        assert kernels is not None
        if config.get("infopo_advantage_variant", "official_token_nonzero") != "official_token_nonzero":
            raise ValueError("InfoPO currently implements the pinned official token/nonzero normalization.")
        intrinsic = torch.zeros_like(mask)
        boundaries = torch.zeros_like(mask)
        histories = []
        for i, trajectory in enumerate(trajectories):
            offsets = np.cumsum([0] + [value.numel() for value in old_log_probs[i]]).tolist()
            histories.append([{"reward": rewards[i] if j == len(units[i]) - 1 else 0.0}
                              for j in range(len(units[i]))])
            for j, unit in enumerate(units[i]):
                boundaries[i, offsets[j]] = 1
                if j + 1 >= len(units[i]):
                    continue
                following = units[i][j + 1]
                masked = [dict(message) for message in following.messages]
                observation_index = len(masked) - 2
                if observation_index < 0 or masked[observation_index]["role"] != "user":
                    raise ValueError("InfoPO requires an explicit public feedback message before the next action.")
                masked[observation_index]["content"] = "No information found."
                with torch.no_grad():
                    visible_score = policy.score_tokens(following.generation, role="infopo_visible")
                    if following.generation.metadata.get("actor_contract") != "official-native-action-v1":
                        raise ValueError("InfoPO requires the same unrestricted token support in both histories.")
                    masked_score = policy.score_tokens(following.generation, masked, role="infopo_masked")
                    info_gain = float((visible_score - masked_score).mean().item())
                    intrinsic[i, offsets[j]:offsets[j + 1]] = info_gain / (offsets[j + 1] - offsets[j])
                trajectory.events[unit.event_index].metadata["counterfactual_information"] = {
                    "status": "supported", "support": "full_vocabulary",
                    "information_gain": info_gain, "reward_eligible": True,
                }
        with torch.no_grad():
            advantage, _, _ = kernels["compute_info_grpo_advantage"](
                reward_matrix, intrinsic, mask, np.array(["same_task_initial_state"] * len(trajectories), dtype=object),
                intrinsic_weight=float(config.get("intrinsic_weight", 0.1)),
                intrinsic_decay_rate=float(config.get("intrinsic_decay_rate", 0.0)), current_epoch=epoch,
                turn_boundaries=boundaries, conversation_histories=histories,
                turn_level_method="Equalized", trajectory_score_method="Sum",
                intrinsic_gate_temperature=float(config.get("intrinsic_gate_temperature", 0.05)),
            )
    else:
        advantage = _group_normalized(rewards).unsqueeze(1) * mask
        if method == "arew":
            assert kernels is not None
            if config.get("arew_variant", "official_tau2_as_only") != "official_tau2_as_only":
                raise ValueError("Native tools use the official AS-only AREW variant; BT requires a registered belief readout.")
            labels = torch.zeros_like(mask)
            step_ids = torch.zeros_like(mask, dtype=torch.long)
            for i, row_units in enumerate(units):
                offset = 0
                for j, unit in enumerate(row_units):
                    length = old_log_probs[i][j].numel()
                    labels[i, offset:offset + length] = unit.label
                    step_ids[i, offset:offset + length] = j + 1
                    offset += length
            bonus_mode = str(config.get("arew_bonus_mode", "minority_fixed"))
            if bonus_mode == "minority_fixed":
                modifier = kernels["get_arew_modifier"](
                    labels, advantage, {"arew_bonus_mode": bonus_mode,
                                        "arew_bonus_scale": float(config.get("arew_bonus_scale", 0.1)),
                                        "arew_negative_step_mode": "full"}, step_ids=step_ids,
                )[0]
            elif bonus_mode == "paper_step_mean":
                # Eq. (1) is a separate, unclipped likelihood-margin objective.
                modifier = torch.zeros_like(advantage)
            else:
                raise ValueError("AREW bonus mode must be minority_fixed or paper_step_mean.")
            advantage = advantage + modifier
    result = []
    for i, row in enumerate(old_log_probs):
        offset = 0
        slices = []
        for value in row:
            slices.append(advantage[i, offset:offset + value.numel()].detach().clone())
            offset += value.numel()
        result.append(slices)
    return result


def _backward_terms(terms: Iterator[torch.Tensor], scale: float = 1.0) -> tuple[float, int]:
    total, count = 0.0, 0
    it = iter(terms)
    while True:
        try:
            term = next(it)
        except StopIteration:
            break
        loss = term * scale
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError("Non-finite native training loss.")
        total += float(loss.detach().item())
        if loss.requires_grad:
            loss.backward()
            count += 1
        del term, loss
    return total, count


def _step_optimizer(policy: Any, optimizer: Any, meter: BudgetMeter, config: Mapping[str, Any], started: float) -> dict[str, Any]:
    parameters = [p for p in policy.parameters() if p.requires_grad]
    clip = float(config.get("grad_clip_norm", 1.0))
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, clip if clip > 0 else float("inf"),
        error_if_nonfinite=True,
    )
    gradient_norm = float(norm.detach().item())
    if gradient_norm == 0.0:
        optimizer.zero_grad(set_to_none=True)
        return {"optimizer_step": meter.gradient_steps, "gradient_l2_norm_before_clip": 0.0,
                "nonzero_gradient": False, "parameter_delta_l2": 0.0, "parameters_changed": False}
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    with torch.no_grad():
        squared_delta = torch.stack([
            (parameter.detach() - previous).float().square().sum()
            for parameter, previous in zip(parameters, before)
        ]).sum()
        parameter_delta_l2 = float(squared_delta.sqrt().item())
    del before
    meter.record_update(monotonic_seconds() - started)
    optimizer.zero_grad(set_to_none=True)
    return {"optimizer_step": meter.gradient_steps, "gradient_l2_norm_before_clip": gradient_norm,
            "nonzero_gradient": True,
            "parameter_delta_l2": parameter_delta_l2, "parameters_changed": parameter_delta_l2 > 0.0}


def _single_ppo_unit_loss(
    policy: Any, unit: TokenUnit, old_t: torch.Tensor, ref_t: torch.Tensor, adv_t: torch.Tensor,
    *, clip: float, dual_clip: float, kl_coef: float, denominator: int,
    diagnostics: dict[str, Any] | None, margin_coefficient: float = 0.0,
) -> torch.Tensor:
    current = policy.score_tokens(unit.generation, role="ppo_update")
    old = old_t.to(device=current.device, dtype=current.dtype)
    reference = ref_t.to(device=current.device, dtype=current.dtype)
    advantage = adv_t.to(device=current.device, dtype=current.dtype)
    if not (current.shape == old.shape == reference.shape == advantage.shape):
        raise ValueError("PPO assistant-token masks/log-probabilities are misaligned.")
    ratio = torch.exp(current - old)
    clipped_loss = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1 - clip, 1 + clip))
    dual_loss = torch.minimum(-advantage * dual_clip, clipped_loss)
    policy_loss = torch.where(advantage < 0, dual_loss, clipped_loss)
    difference = reference - current
    reference_kl = torch.exp(difference) - difference - 1.0
    margin_loss = margin_coefficient * current.sum()
    if diagnostics is not None:
        diagnostics["policy_loss"] += float(policy_loss.detach().sum().item()) / denominator
        diagnostics["reference_kl"] += float(reference_kl.detach().sum().item()) / denominator
        diagnostics["weighted_reference_kl_loss"] += kl_coef * float(reference_kl.detach().sum().item()) / denominator
        diagnostics["arew_likelihood_margin_loss"] += float(margin_loss.detach().item())
    return (policy_loss + kl_coef * reference_kl).sum() / denominator + margin_loss


def _arew_margin_coefficients(units: Sequence[Sequence[TokenUnit]], scale: float) -> list[list[float]]:
    """AREW Eq. (1): -lambda/G times positive-step mean minus negative-step mean."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("AREW likelihood-margin scale must be finite and nonnegative.")
    coefficients = []
    for row in units:
        positive = sum(unit.label > 0 for unit in row)
        negative = sum(unit.label < 0 for unit in row)
        coefficients.append([
            (-scale / (len(units) * positive) if unit.label > 0
             else scale / (len(units) * negative) if unit.label < 0 else 0.0)
            if positive and negative else 0.0 for unit in row
        ])
    return coefficients


def _ppo_terms(
    policy: Any, units: Sequence[Sequence[TokenUnit]], old_scores: Sequence[Sequence[torch.Tensor]],
    reference_scores: Sequence[Sequence[torch.Tensor]], advantages: Sequence[Sequence[torch.Tensor]],
    config: Mapping[str, Any], diagnostics: dict[str, Any] | None = None,
    margin_coefficients: Sequence[Sequence[float]] | None = None,
) -> Iterator[torch.Tensor]:
    denominator = sum(score.numel() for row in old_scores for score in row)
    if denominator == 0:
        raise ValueError("PPO batch contains no assistant response tokens.")
    clip = float(config.get("ppo_clip", 0.2))
    dual_clip = float(config.get("ppo_dual_clip", 3.0))
    kl_coef = float(config.get("reference_kl_coef", 0.01))
    if not 0 < clip < 1 or dual_clip <= 1 or kl_coef < 0:
        raise ValueError("Invalid PPO clipping or reference-KL configuration.")
    for i, row in enumerate(units):
        for j, unit in enumerate(row):
            yield _single_ppo_unit_loss(
                policy, unit, old_scores[i][j], reference_scores[i][j], advantages[i][j],
                clip=clip, dual_clip=dual_clip, kl_coef=kl_coef, denominator=denominator,
                diagnostics=diagnostics,
                margin_coefficient=margin_coefficients[i][j] if margin_coefficients is not None else 0.0,
            )


def update_rollout_group(
    method: str, policy: Any, trajectories: Sequence[NativeTrajectory], optimizer: Any,
    meter: BudgetMeter, config: Mapping[str, Any], *, baseline: float,
    kernels: OfficialKernels | None = None, epoch: int = 0,
) -> dict[str, Any]:
    """One actual same-task group; all autograd graphs are released per action."""
    if not trajectories or any(not trajectory.events for trajectory in trajectories):
        return {"updated": False, "learning_outcome": "empty_trajectory", "loss": 0.0,
                "nonzero_gradient": False, "parameters_changed": False, "parameter_delta_l2": 0.0}
    optimizer.zero_grad(set_to_none=True)
    try:
        units = [_token_units(trajectory) for trajectory in trajectories]
        old_scores = [[_sampled_log_probs(policy, unit.generation) for unit in row] for row in units]
        if any(score.numel() == 0 for row in old_scores for score in row):
            raise ValueError("A sampled assistant step has no scored response tokens.")
        advantages = group_advantages(method, policy, trajectories, units, old_scores, kernels, config, epoch)
        margins = (_arew_margin_coefficients(units, float(config.get("arew_bonus_scale", 0.1)))
                   if method == "arew" and config.get("arew_bonus_mode", "minority_fixed") == "paper_step_mean"
                   else None)
        has_margin_signal = margins is not None and any(value != 0.0 for row in margins for value in row)
        adv_nonzero = sum(int(torch.count_nonzero(value).item()) for row in advantages for value in row)
        if adv_nonzero == 0 and not has_margin_signal and float(config.get("reference_kl_coef", 0.01)) == 0:
            return {"updated": False, "learning_outcome": "zero_training_signal", "loss": 0.0,
                    "nonzero_gradient": False, "parameters_changed": False, "parameter_delta_l2": 0.0,
                    "advantage_nonzero_count": 0}
        references = []
        with torch.no_grad():
            for row in units:
                references.append([policy.score_tokens(unit.generation, reference=True,
                                                       role="ppo_reference").detach().cpu() for unit in row])
        losses, diagnostics, objective_parts = [], [], []
        for _ in range(int(config.get("ppo_epochs", 1))):
            optimizer.zero_grad(set_to_none=True)
            iteration_started = monotonic_seconds()
            parts = {"policy_loss": 0.0, "reference_kl": 0.0, "weighted_reference_kl_loss": 0.0,
                     "arew_likelihood_margin_loss": 0.0}
            loss, terms = _backward_terms(_ppo_terms(policy, units, old_scores, references, advantages,
                                                   config, parts, margin_coefficients=margins))
            objective_parts.append(parts)
            if terms:
                diagnostics.append(_step_optimizer(policy, optimizer, meter, config, iteration_started))
            losses.append(loss)
        nonzero_grad = any(d["nonzero_gradient"] for d in diagnostics)
        params_changed = any(d["parameters_changed"] for d in diagnostics)
        delta_l2 = sum(float(d.get("parameter_delta_l2", 0.0)) for d in diagnostics)
        return {"updated": params_changed,
                "learning_outcome": "parameters_updated" if params_changed else ("zero_gradient" if not nonzero_grad else "no_parameter_change"),
                "loss": sum(losses) / max(1, len(losses)),
                "ppo_epochs": len(losses), "assistant_tokens": sum(v.numel() for row in old_scores for v in row),
                "optimizer_step_diagnostics": diagnostics,
                "ppo_objective_parts": objective_parts,
                "nonzero_gradient": nonzero_grad, "parameters_changed": params_changed, "parameter_delta_l2": delta_l2,
                "task_reward_nonzero_count": sum(trajectory.utility != 0.0 for trajectory in trajectories),
                "advantage_nonzero_count": adv_nonzero,
                "advantage_l1": sum(float(value.abs().sum().item()) for row in advantages for value in row),
                "advantage_kind": "combined_official_method_advantage",
                "arew_bonus_mode": str(config.get("arew_bonus_mode", "minority_fixed")) if method == "arew" else None,
                "arew_label_counts": dict(Counter(unit.label for row in units for unit in row)) if method == "arew" else None,
                "kernel_provenance": kernels.provenance if kernels is not None else {}}
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise


def parse_first_error_annotation(text: str, num_events: int) -> dict[str, Any] | None:
    annotation = _parse_json(text)
    if annotation is None:
        return None
    index = annotation.get("first_error_index")
    if type(index) is not int or not 0 <= index < num_events:
        return None
    if annotation.get("preceding_steps_valid") is not True or annotation.get("verifiable") is not True:
        return None
    if not isinstance(annotation.get("evidence"), str) or not annotation["evidence"].strip():
        return None
    return annotation


def _preference_step(policy: Any, generation: NativeGeneration) -> NativeGeneration:
    eos_ids = generation.metadata["eos_token_ids"]
    response = tuple(generation.response_ids)
    if response and response[-1] in eos_ids:
        response = response[:-1]
        metadata = dict(generation.metadata)
        if "sampling_log_probs" in metadata:
            metadata["sampling_log_probs"] = metadata["sampling_log_probs"][:-1]
        metadata["dpo_terminal_eos_excluded"] = True
        return replace(generation, response_ids=response, metadata=metadata)
    return generation


def collect_step_preference(
    policy: Any, env: Any, task: str | Mapping[str, Any], config: Mapping[str, Any], meter: BudgetMeter,
    *, seed: int, output_dir: Path, group_index: int,
) -> tuple[PreferencePair | None, dict[str, Any], list[NativeTrajectory]]:
    """Localize a first error, sample replacement step(s) and verify native continuation."""
    task_info = _task(task)
    if task_info.get("split", "train") != "train":
        raise ValueError("Step-DPO reference supervision is restricted to training tasks.")
    snapshots: list[Any] = []
    limits = _limits(config)
    num_candidates = max(1, int(config.get("step_dpo_candidates", 1)))
    episode_meter = BudgetMeter(limits, parent=meter)
    # LOCA generates the private reference with the selected task instance.
    # Read it only after reset and keep that same public instance for rollout.
    with _meters(policy, env, episode_meter), _preserve_actor_rng():
        context = env.reset(str(task_info["task_id"]), seed=seed)
        reference = env.training_reference(str(task_info["task_id"]))
    with policy.frozen_reference():
        rejected_trajectory = rollout(
            policy, env, task_info, config, episode_meter, seed=seed, method="step_dpo", training=True,
            snapshots=snapshots, context=context,
        )
    executed_trajectories: list[NativeTrajectory] = [rejected_trajectory]
    trace_path = output_dir / "step_dpo_data" / f"group-{group_index:06d}.json"
    record: dict[str, Any] = {"group": group_index, "task_id": str(task_info["task_id"]),
                              "trajectory": public_trace(rejected_trajectory), "status": "unpaired", "candidates": [],
                              "reference_provenance": {key: reference[key] for key in ("benchmark", "task_id", "source", "reference_id")},
                              "visibility": "training_supervision_only",
                              "teacher": "initial_frozen_shared_backbone"}
    _write_json(trace_path, record)
    if rejected_trajectory.metadata.get("global_budget_limited"):
        record.update(reason="incomplete_global_budget", global_budget_limited=True)
        _write_json(trace_path, record)
        return None, record, executed_trajectories
    if rejected_trajectory.success or not rejected_trajectory.events:
        record["reason"] = "reference_succeeded" if rejected_trajectory.success else "no_actions"
        _write_json(trace_path, record)
        return None, record, executed_trajectories
    annotation_messages = [
        {"role": "system", "content": FIRST_ERROR_INSTRUCTION},
        {"role": "user", "content": json.dumps({
            "task": rejected_trajectory.events[0].messages_before,
            "public_trajectory": [{"index": i, "action": event.generation.text,
                                    "observation": event.observation}
                                   for i, event in enumerate(rejected_trajectory.events)],
            "native_terminal_success": False,
            "correct_training_reference": reference["reference"],
        }, ensure_ascii=False)},
    ]
    with _meters(policy, env, meter), policy.frozen_reference():
        try:
            generation = policy.complete(annotation_messages, role="step_dpo_annotation", greedy=True,
                                         max_new_tokens=int(config.get("step_dpo_annotation_max_tokens", 512)))
        except BudgetExhausted as error:
            record.update(reason=f"annotation_budget:{error}",
                          global_budget_limited=str(error) in {"wall_seconds", "model_tokens", "environment_steps"})
            _write_json(trace_path, record)
            return None, record, executed_trajectories
    record["annotation_usage"] = {"input_tokens": generation.input_tokens,
                                   "output_tokens": generation.output_tokens}
    annotation = parse_first_error_annotation(generation.text, len(rejected_trajectory.events))
    if annotation is None:
        record.update(reason="no_supported_first_error_annotation", annotation_text=generation.text)
        _write_json(trace_path, record)
        return None, record, executed_trajectories
    index = annotation["first_error_index"]
    rejected = rejected_trajectory.events[index].generation
    prompt = rejected_trajectory.events[index].messages_before
    prefix = rejected_trajectory.events[:index]
    prefix_usage = {
        "env_steps": len(prefix),
        "input_tokens": sum(event.generation.input_tokens for event in prefix),
        "output_tokens": sum(event.generation.output_tokens for event in prefix),
        "forward_tokens": 0,
    }
    prefix_seconds = float(prefix[-1].metadata["elapsed_after"]) if prefix else 0.0
    context = {"task_id": rejected_trajectory.task_id, "scenario_id": rejected_trajectory.scenario_id}
    for candidate_index in range(num_candidates):
        try:
            with _meters(policy, env, meter):
                _restore_environment(env, snapshots[index])
        except BudgetExhausted as error:
            record.update(reason=f"candidate_restore_budget:{error}", global_budget_limited=True)
            _write_json(trace_path, record)
            return None, record, executed_trajectories
        candidate_meter = _continuation_meter(limits, meter, prefix_usage)
        with _meters(policy, env, candidate_meter), policy.frozen_reference():
            public_schema = env.action_schema
            try:
                # Step-DPO Eq. (3): sample from the reference on the unchanged correct prefix.
                public_target = policy.act_native(prompt, public_schema, greedy=False,
                                                  role="step_dpo_correction_teacher")
            except BudgetExhausted as error:
                record.update(reason=f"correction_teacher_budget:{error}",
                              global_budget_limited=_global_budget_limited(candidate_meter, str(error)))
                _write_json(trace_path, record)
                return None, record, executed_trajectories
            target_metadata = dict(public_target.metadata)
            target_metadata["supervision_reference_id"] = reference["reference_id"]
            chosen = replace(public_target, metadata=target_metadata, role="step_dpo_supervised_target")
            started = monotonic_seconds()
            try:
                step = env.step_native(chosen.action or {"kind": "invalid", "raw_text": chosen.text})
            except BudgetExhausted as error:
                record["candidates"].append({"index": candidate_index, "native_success": False,
                                             "termination": f"budget:{error}",
                                             "resource_usage": candidate_meter.snapshot()})
                record.update(reason=f"candidate_action_budget:{error}",
                              global_budget_limited=_global_budget_limited(candidate_meter, str(error)))
                _write_json(trace_path, record)
                return None, record, executed_trajectories
            elapsed = monotonic_seconds() - started
            judge_seconds = terminal_scoring_seconds(env, step)
            submission_elapsed = max(0.0, prefix_seconds + chosen.elapsed_seconds + elapsed - judge_seconds)
            event = NativeEvent(prompt, chosen, str(step.observation), elapsed,
                                {"public_error": _public_error(step.metadata, str(step.observation)),
                                 "native_step": dict(step.metadata),
                                 "native_done": bool(step.done),
                                 "native_terminal_success": bool(step.done and step.success),
                                 "elapsed_after": submission_elapsed})
            if step.done:
                success = bool(step.success) and submission_elapsed <= limits.max_wall_seconds
                candidate = NativeTrajectory(
                    rejected_trajectory.task_id, rejected_trajectory.scenario_id, [*prefix, event],
                    success=success, elapsed_seconds=submission_elapsed,
                    budget_seconds=limits.max_wall_seconds,
                    termination="native_terminal" if success else "native_failure",
                    metadata={"environment_seed": seed, "actor_contract": actor_contract("step_dpo"),
                              "native_terminal_success": bool(step.success),
                              "terminal_judge_seconds": judge_seconds},
                )
                candidate.metadata["official_evaluation"] = dict(env.finalize_evaluation())
                candidate.metadata["official_evaluation_scope"] = "native_terminal_state"
                candidate.metadata["budget_compliant_success"] = (
                    bool(candidate.success) and candidate.elapsed_seconds <= candidate.budget_seconds)
                candidate.metadata["program_outcomes"] = program_execution_summary(candidate.events)
                candidate.metadata["resource_usage"] = candidate_meter.snapshot()
            else:
                next_msgs = [dict(m) for m in prompt[:-1]]
                append_interaction(next_msgs, chosen, step)
                candidate = rollout(
                    policy, env, task_info, config, candidate_meter, seed=seed, method="step_dpo", training=True,
                    context=context, prefix=[*prefix, event],
                    prefix_messages=next_msgs,
                    prefix_seconds=event.metadata["elapsed_after"],
                )
            executed_trajectories.append(candidate)
        record["candidates"].append({"index": candidate_index, "native_success": candidate.success,
                                      "termination": candidate.termination,
                                      "verification_steps": len(candidate.events) - index,
                                      "resource_usage": candidate_meter.snapshot()})
        _write_json(trace_path, record)
        if candidate.metadata.get("global_budget_limited"):
            record.update(reason="candidate_global_budget", global_budget_limited=True)
            _write_json(trace_path, record)
            return None, record, executed_trajectories
        if (candidate.success and candidate.elapsed_seconds <= candidate.budget_seconds
                and not (chosen.text == rejected.text and chosen.mode == rejected.mode)):
            verification = {"native_success": True, "environment_seed": seed,
                            "continuation_steps": len(candidate.events) - index,
                            "candidate_index": candidate_index}
            chosen_step, rejected_step = _preference_step(policy, chosen), _preference_step(policy, rejected)
            if not chosen_step.response_ids or not rejected_step.response_ids:
                continue
            pair = PreferencePair(prompt, chosen_step, rejected_step, rejected_trajectory.task_id, index, annotation, verification)
            record.update(status="paired", annotation={key: annotation[key] for key in
                          ("first_error_index", "preceding_steps_valid", "verifiable")}, verification=verification,
                          prompt=prompt, chosen=chosen.text, rejected=rejected.text,
                          chosen_mode=chosen.mode, rejected_mode=rejected.mode)
            _write_json(trace_path, record)
            return pair, record, executed_trajectories
    record["reason"] = "no_native_successful_correction"
    _write_json(trace_path, record)
    return None, record, executed_trajectories


def step_dpo_loss_terms(policy: Any, pair: PreferencePair, beta: float = 0.4,
                        diagnostics: dict[str, Any] | None = None) -> Iterator[torch.Tensor]:
    if beta <= 0:
        raise ValueError("DPO beta must be positive.")
    with torch.no_grad():
        current_chosen = policy.score(pair.chosen, role="dpo_current_coefficient")
        current_rejected = policy.score(pair.rejected, role="dpo_current_coefficient")
        ref_chosen = policy.score(pair.chosen, reference=True, role="dpo_reference")
        ref_rejected = policy.score(pair.rejected, reference=True, role="dpo_reference")
        logit = beta * ((current_chosen - ref_chosen) - (current_rejected - ref_rejected))
        coefficient = beta * float(torch.sigmoid(-logit).item())
        if diagnostics is not None:
            diagnostics.update(dpo_loss=float(torch.nn.functional.softplus(-logit).item()),
                               preference_logit=float(logit.item()))
    yield -coefficient * policy.score(pair.chosen, role="dpo_update_chosen")
    yield coefficient * policy.score(pair.rejected, role="dpo_update_rejected")


def evaluate_policy(
    policy: Any, env: Any, tasks: Sequence[str | Mapping[str, Any]], config: Mapping[str, Any],
    meter_factory: Callable[[], BudgetMeter], method: str, seed: int, backbone: str,
) -> list[dict[str, Any]]:
    """Evaluate the complete fixed task set with the configured sampling policy."""
    benchmark = str(config.get("benchmark", getattr(env, "benchmark", "")))
    if not benchmark:
        raise ValueError("Evaluation requires its native benchmark identity.")
    output = Path(config["output_dir"]) if config.get("output_dir") else None
    policy.eval()
    records = []
    for index, task in enumerate(tasks):
        task_info = _task(task)
        meter = meter_factory()
        environment_seed = int(task_info.get("environment_seed", seed * 1000003 + index))
        with _preserve_actor_rng():
            torch.manual_seed(environment_seed)
            trajectory = rollout(policy, env, task_info, config, meter,
                                 seed=environment_seed, method=method, training=False)
        result = trajectory.result(benchmark=benchmark, backbone=backbone, seed=seed, method=method)
        result.update(resource_usage=meter.snapshot(), split=task_info.get("split", config.get("split", "dev")),
                      environment_seed=environment_seed, arm=config.get("arm"))
        records.append(result)
        if output is not None:
            _append_json(output / "records.jsonl", result)
            _write_json(output / "traces" / f"task-{index:05d}.json", public_trace(trajectory, method=method))
    return records


def run_training(
    method: str, policy: Any, env: Any, train_tasks: Sequence[str | Mapping[str, Any]],
    dev_tasks: Sequence[str | Mapping[str, Any]], config: Mapping[str, Any], meter: BudgetMeter,
    output_dir: str | Path, seed: int, *, progress_supervisors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the fixed PRISM design and save one final checkpoint for formal evaluation."""
    if method not in TRAINING_METHODS:
        raise ValueError(f"Unsupported native-r3 training method: {method}")
    config = _config(config, method)
    tasks = [_task(task) for task in train_tasks]
    if not tasks or any(task.get("split", "train") != "train" for task in tasks):
        raise ValueError("Native training requires a nonempty native train split.")
    if progress_supervisors is not None:
        missing_supervisors = sorted({task["task_id"] for task in tasks} - set(progress_supervisors))
        if missing_supervisors:
            raise ValueError("The fixed native-r3 progress dataset is missing complete successful references for: "
                             + ", ".join(missing_supervisors))
    if {task["task_id"] for task in tasks} & {_task(task)["task_id"] for task in dev_tasks}:
        raise ValueError("Train and development task IDs must be disjoint.")
    configured_groups = config.get("max_train_groups")
    max_groups = None if configured_groups is None else int(configured_groups)
    if (max_groups is not None and max_groups <= 0) or int(config.get("groups_per_update", 1)) != 1:
        raise ValueError("Configure a positive group cap when present and one group per update.")
    if config.get("branch_anchor", "first_public_feasible_frontier") != "first_public_feasible_frontier":
        raise ValueError("Native-r3 branches use the first public feasible frontier.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training.jsonl").exists() or (output / "training_summary.json").exists():
        raise FileExistsError("Use an unused training output directory.")
    policy.train()
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The training actor has no trainable parameters.")
    initial = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=float(config.get("learning_rate", 1e-6)),
                                  weight_decay=float(config.get("weight_decay", 0.0)))
    kernels = OfficialKernels(config["external_root"], method) if method in {"gigpo", "infopo", "arew", "t3"} else None
    randomizer = random.Random(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _write_json(output / "training_manifest.json", {
        "protocol_version": "native-r3", "method": method, "seed": seed,
        "task_outcome_reward": "U_B=Y*max(0,1-C/B)",
        "objective": ("verified_single_step_preference" if method == "step_dpo"
                      else "native_utility_plus_registered_method_specific_advantage"),
        "reference_visibility": "train_only", "branch_anchor": "first_public_feasible_frontier",
        "actor_contract": actor_contract(method, classic_text=bool(env.action_schema.get("classic_text"))),
        "configuration": dict(config),
        "implementation_variants": {
            "gigpo_episode_normalization": config.get("gigpo_episode_normalization", "official_step_weighted"),
            "gigpo_anchor": "complete_public_history_without_budget_message",
            "infopo_advantage": config.get("infopo_advantage_variant", "official_token_nonzero"),
            "infopo_scoring": "full_vocabulary_same_next_action",
            "arew": config.get("arew_variant", "official_tau2_as_only"),
            "arew_channels": ["AS"],
            "arew_labels": "positive=state_change_or_new_public_business_evidence_or_success;negative=hard_error;otherwise=abstain",
            "arew_bonus": config.get("arew_bonus_mode", "minority_fixed"),
            "arew_paper_margin_reduction": "trajectory_mean_of_signed_step_means_of_summed_token_logp_unclipped",
            "t3_progress": "public_business_content_increment",
            "step_dpo_sampling": "frozen_reference_public_prefix_only",
            "ppo_reduction": "token_mean", "zero_advantage_reference_kl": "retained",
        },
        "training_budget": asdict(meter.limits), "checkpoint_selection": "final_after_fixed_training_run",
        "intermediate_evaluations": False, "intermediate_success_gates": False,
        "kernel_provenance": kernels.provenance if kernels is not None else {},
    })
    baseline = 0.0
    baseline_initialized = False
    complete_groups = 0
    branch_pairs = 0
    nonzero_gradient_steps = 0
    parameter_change_steps = 0
    cumulative_delta = 0.0
    outcomes: Counter[str] = Counter()
    physical_events: list[NativeEvent] = []
    trajectories: list[NativeTrajectory] = []
    stop_reason = "configured_groups_completed" if max_groups is not None else "budget_exhausted"
    attempted_groups = 0
    no_work_groups = 0
    groups = count() if max_groups is None else range(max_groups)
    for group in groups:
        task = tasks[group % len(tasks)]
        work_before = (meter.env_steps, meter.total_model_tokens)
        environment_seed = int(task.get("environment_seed", randomizer.randrange(1, 2**31 - 1)))
        supervisor = progress_supervisors.get(task["task_id"]) if progress_supervisors is not None else None
        entry: dict[str, Any] = {"group": group, "task_id": task["task_id"], "method": method,
                                 "environment_seed": environment_seed,
                                 "branch_anchor": "first_public_feasible_frontier"}
        sample = None
        group_trajectories: list[NativeTrajectory] = []
        partial = False
        try:
            meter.ensure_env_steps(1)
            meter.ensure_forward(1)
            with _meters(policy, env, meter):
                if method in {"prism_evolve", "ablation_unweighted", "unstratified_2branch"}:
                    sample = collect_branch_pair(policy, env, task, config, meter, seed=environment_seed,
                                                 stratified=method != "unstratified_2branch",
                                                 progress_supervisor=supervisor)
                    group_trajectories = [sample.first] + ([sample.second] if sample.second is not None else [])
                    if sample.partial_reason:
                        partial = True
                        stop_reason = sample.partial_reason
                        entry.update(updated=False, learning_outcome="incomplete_pair_budget", reason=stop_reason)
                    else:
                        complete_groups += 1
                        branch_pairs += int(sample.second is not None)
                        entry.update(_update_pair(policy, optimizer, sample, meter, config, baseline,
                                                  unweighted=method == "ablation_unweighted"))
                        mean_return = sum(_training_value(item) for item in group_trajectories) / len(group_trajectories)
                        baseline = (0.9 * baseline + 0.1 * mean_return) if baseline_initialized else mean_return
                        baseline_initialized = True
                elif method == "step_dpo":
                    pair, construction, group_trajectories = collect_step_preference(
                        policy, env, task, config, meter,
                        seed=environment_seed, output_dir=output, group_index=group,
                    )
                    entry["preference_status"] = construction["status"]
                    partial = bool(construction.get("global_budget_limited"))
                    if partial:
                        stop_reason = str(construction.get("reason", "incomplete_global_budget"))
                    else:
                        complete_groups += 1
                    branch_pairs += int(len(group_trajectories) >= 2)
                    if pair is None:
                        entry.update(updated=False, learning_outcome=("incomplete_group_budget" if partial
                                                                      else "no_valid_preference_pair"),
                                     reason=construction.get("reason"),
                                     nonzero_gradient=False, parameters_changed=False, parameter_delta_l2=0.0)
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        started = monotonic_seconds()
                        dpo_diagnostics: dict[str, Any] = {}
                        loss, terms = _backward_terms(step_dpo_loss_terms(
                            policy, pair, beta=float(config.get("step_dpo_beta", 0.4)),
                            diagnostics=dpo_diagnostics))
                        diag = _step_optimizer(policy, optimizer, meter, config, started) if terms else {
                            "nonzero_gradient": False, "parameters_changed": False, "parameter_delta_l2": 0.0
                        }
                        entry.update(updated=bool(diag["parameters_changed"]),
                                     learning_outcome="parameters_updated" if diag["parameters_changed"] else "zero_gradient",
                                     loss=dpo_diagnostics.get("dpo_loss", 0.0),
                                     gradient_surrogate_loss=loss, **dpo_diagnostics, **diag,
                                     first_error_index=pair.first_error_index)
                    idx_err = construction.get("annotation", {}).get("first_error_index", 0) if isinstance(construction.get("annotation"), Mapping) else 0
                    if group_trajectories:
                        physical_events.extend(group_trajectories[0].events)
                        for cand_traj in group_trajectories[1:]:
                            physical_events.extend(cand_traj.events[idx_err:])
                elif method == "frozen_base":
                    group_size = int(config.get("frozen_base_group_size", GROUP_SIZES["frozen_base"]))
                    limits = _limits(config)
                    with policy.frozen_reference():
                        for _ in range(group_size):
                            ep_meter = BudgetMeter(limits, parent=meter)
                            traj = rollout(policy, env, task, config, ep_meter, seed=environment_seed,
                                           method="frozen_base", training=True, progress_supervisor=supervisor)
                            group_trajectories.append(traj)
                            physical_events.extend(traj.events)
                            if traj.metadata.get("global_budget_limited"):
                                partial = True
                                stop_reason = "global_budget_during_frozen_rollout"
                                break
                    if not partial:
                        complete_groups += 1
                        branch_pairs += int(len(group_trajectories) >= 2)
                    entry.update(updated=False, learning_outcome="frozen_initial_control",
                                 nonzero_gradient=False, parameters_changed=False, parameter_delta_l2=0.0)
                else:
                    group_size = int(config.get(f"{method}_group_size", GROUP_SIZES.get(method, 2)))
                    limits = _limits(config)
                    for _ in range(group_size):
                        ep_meter = BudgetMeter(limits, parent=meter)
                        traj = rollout(policy, env, task, config, ep_meter, seed=environment_seed,
                                       method=method, training=True, progress_supervisor=supervisor)
                        group_trajectories.append(traj)
                        physical_events.extend(traj.events)
                        if traj.metadata.get("global_budget_limited"):
                            partial = True
                            stop_reason = "global_budget_during_rollout_group"
                            break
                    if partial:
                        entry.update(updated=False, learning_outcome="incomplete_group_budget", reason=stop_reason,
                                     nonzero_gradient=False, parameters_changed=False, parameter_delta_l2=0.0)
                    else:
                        complete_groups += 1
                        branch_pairs += int(len(group_trajectories) >= 2)
                        stats = update_rollout_group(method, policy, group_trajectories, optimizer, meter, config,
                                                     baseline=baseline, kernels=kernels, epoch=group)
                        entry.update(stats, group_size=group_size)
                        mean_utility = sum(item.utility for item in group_trajectories) / len(group_trajectories)
                        baseline = (0.9 * baseline + 0.1 * mean_utility) if baseline_initialized else mean_utility
                        baseline_initialized = True
        except BudgetExhausted as error:
            optimizer.zero_grad(set_to_none=True)
            partial = True
            stop_reason = f"budget:{error}"
            entry.update(updated=False, learning_outcome="budget_exhausted", reason=stop_reason)
        attempted_groups += 1
        nonzero_gradient_steps += int(entry.get("nonzero_gradient", False))
        parameter_change_steps += int(entry.get("parameters_changed", False))
        cumulative_delta += float(entry.get("parameter_delta_l2", 0.0))
        outcomes[str(entry["learning_outcome"])] += 1
        if sample is not None:
            # Shared prefixes are counted once in this physical-work summary.
            if sample.second is None:
                physical_events.extend(sample.first.events)
            else:
                physical_events.extend(sample.prefix)
                physical_events.extend(sample.first.events[sample.split_step:])
                physical_events.extend(sample.second.events[sample.split_step:])
            entry.update(gate_probability=sample.gate_probability, split_step=sample.split_step,
                         branch_utilities=[item.utility for item in group_trajectories],
                         branch_training_returns=[item.training_return for item in group_trajectories],
                         branch_successes=[int(item.success) for item in group_trajectories],
                         branch_terminations=[item.termination for item in group_trajectories])
        elif group_trajectories:
            entry.update(branch_utilities=[item.utility for item in group_trajectories],
                         branch_training_returns=[item.training_return for item in group_trajectories],
                         branch_successes=[int(item.success) for item in group_trajectories],
                         branch_terminations=[item.termination for item in group_trajectories])
        trajectories.extend(group_trajectories)
        for index, trajectory in enumerate(group_trajectories):
            _write_json(output / "traces" / f"group-{group:05d}-trajectory-{index}.json", public_trace(trajectory, method=method))
        entry.update(resource_usage=meter.snapshot(), complete_groups=complete_groups)
        _append_json(output / "training.jsonl", entry)
        if partial:
            break
        work_after = (meter.env_steps, meter.total_model_tokens)
        no_work_groups = no_work_groups + 1 if work_after == work_before else 0
        if max_groups is None and no_work_groups >= len(tasks):
            stop_reason = "no_actor_or_environment_work_across_training_tasks"
            break
    with torch.no_grad():
        net_delta = math.sqrt(sum(float((parameter.detach() - previous).float().square().sum().item())
                                 for parameter, previous in zip(parameters, initial)))
    del initial
    checkpoint = output / "final_adapter"
    policy.save_adapter(checkpoint)
    policy.eval()
    summary = {
        "protocol_version": "native-r3", "method": method, "seed": seed,
        "selected_checkpoint": str(checkpoint), "checkpoint_selection": "final_after_fixed_training_run",
        "stop_reason": stop_reason, "configured_groups": max_groups, "attempted_groups": attempted_groups,
        "complete_groups": complete_groups, "complete_branch_pairs": branch_pairs,
        "gradient_steps": meter.gradient_steps, "nonzero_gradient_steps": nonzero_gradient_steps,
        "parameter_change_steps": parameter_change_steps, "parameter_delta_l2": net_delta,
        "cumulative_parameter_delta_l2": cumulative_delta, "learning_outcomes": dict(outcomes),
        "learning_status": "parameters_updated" if parameter_change_steps else "no_parameter_change",
        "training_trajectories": len(trajectories),
        "native_terminal_successes": sum(int(trajectory.success) for trajectory in trajectories),
        "native_train_success_rate": sum(int(trajectory.success) for trajectory in trajectories) / len(trajectories) if trajectories else 0.0,
        "program_outcomes": program_execution_summary(physical_events), "usage": meter.snapshot(),
        "reward_uses_only_native_terminal_utility": method in {
            "prism_evolve", "ablation_unweighted", "unstratified_2branch", "gigpo", "t3", "frozen_base"},
        "task_outcome_reward": "U_B=Y*max(0,1-C/B)",
        "training_objective": ("verified_single_step_preference" if method == "step_dpo"
                               else "native_utility_plus_registered_method_specific_advantage"),
        "reference_visibility": "train_only",
        "actor_contract": actor_contract(method, classic_text=bool(env.action_schema.get("classic_text"))),
    }
    summary["program_outcomes"]["counting_unit"] = "physically_executed_actor_events; shared_prefix_counted_once"
    _write_json(output / "training_summary.json", summary)
    return summary


def collect_native_demonstrations(
    policy: Any, environment_factory: Callable[..., Any], tasks: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any], meter: BudgetMeter, output_dir: str | Path,
) -> Any:
    """Load the reference collector only in a training call, never on evaluation import."""
    from .native_curriculum import collect_native_demonstrations as collect
    return collect(policy, environment_factory, tasks, config, meter, output_dir)


def train_warmstart(
    policy: Any, demonstrations: Sequence[Any], config: Mapping[str, Any],
    meter: BudgetMeter, output_dir: str | Path, *, seed: int,
) -> dict[str, Any]:
    """Fit full public-action traces and freeze the resulting shared reference adapter."""
    from .native_curriculum import train_warmstart as train
    return train(policy, demonstrations, config, meter, output_dir, seed=seed)
