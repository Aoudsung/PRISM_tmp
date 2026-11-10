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

DCC_METHODS = frozenset({
    "prism_evolve", "dcc_label_shuffled", "dcc_no_program_growth",
    "dcc_reset_parameters", "dcc_pre_first_success", "dcc_k0_planner",
    "dcc_fixed_d_shuffled",
})
TRAINING_METHODS = frozenset({
    "prism_evolve", "dcc_label_shuffled", "dcc_no_program_growth",
    "dcc_reset_parameters", "dcc_pre_first_success", "dcc_k0_planner",
    "dcc_fixed_d_shuffled",
    "ablation_unweighted", "unstratified_2branch",
    "gigpo", "infopo", "arew", "t3", "step_dpo", "frozen_base",
})
PRISM_ACTOR_METHODS = frozenset({
    "prism_evolve", "dcc_label_shuffled", "dcc_no_program_growth",
    "dcc_reset_parameters", "dcc_pre_first_success", "dcc_k0_planner",
    "dcc_fixed_d_shuffled",
    "ablation_unweighted", "unstratified_2branch", "frozen_base",
})
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
    stop_after_termination: str = "prefix_ready",
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
    benchmark_name = str(config.get("benchmark", getattr(env, "benchmark", "alfworld"))).split(":", 1)[0]
    if hasattr(policy, "dcc_mode_enabled"):
        policy.dcc_mode_enabled = method in DCC_METHODS
        policy.dcc_eval_variant = (
            "no_program_growth" if method == "dcc_no_program_growth"
            else "reset_parameters" if method == "dcc_reset_parameters"
            else "k0_planner" if method == "dcc_k0_planner"
            else "full"
        )
        if policy.dcc_mode_enabled and not prefix:
            init_task_instruction = str(
                (context or {}).get("instruction", task_info.get("instruction", ""))
            )
            policy.reset_dcc_episode(benchmark_name, task_instruction=init_task_instruction)
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
                    trajectory.termination = stop_after_termination
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
                task_instruction = str(context.get("instruction", task_info.get("instruction", "")))
                rem_step_ratio = max(0.0, 1.0 - (len(trajectory.events) / max(1, limits.max_env_steps)))
                if gated_actor:
                    generation = policy.act(
                        before,
                        schema,
                        forced_mode=forced_mode,
                        greedy=greedy,
                        role="actor",
                        events=trajectory.events,
                        task_instruction=task_instruction,
                        remaining_budget_ratio=rem_step_ratio,
                    )
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
                if method in DCC_METHODS and hasattr(policy, "dcc_ensemble"):
                    from .native_dcc import PublicBenchmarkBinder
                    task_instruction = str(context.get("instruction", task_info.get("instruction", "")))
                    rem_step_ratio = max(0.0, 1.0 - (len(trajectory.events) / max(1, limits.max_env_steps)))
                    obs_update = policy.observe_dcc_transition(
                        benchmark_name,
                        task_instruction,
                        trajectory.events,
                        schema,
                        actual_action=generation.text.strip(),
                        remaining_budget_ratio=rem_step_ratio,
                        had_error=bool(facts.get("hard_bad") or event.metadata.get("public_error")),
                        terminal_done=bool(step.done),
                        terminal_success=bool(step.done and step.success),
                    )
                    pub_goal_val = PublicBenchmarkBinder.public_goal_readout(
                        benchmark_name,
                        task_instruction,
                        trajectory.events,
                        schema,
                        terminal_success=bool(step.done and step.success),
                        within_budget=True,
                        cost_used=float(len(trajectory.events)),
                        cost_budget=float(max(1, limits.max_env_steps)),
                    )
                    event.metadata["dcc_pre_public_fields"] = obs_update["pre_public_fields"]
                    event.metadata["dcc_public_fields"] = obs_update["post_public_fields"]
                    event.metadata["dcc_outcome_bin"] = obs_update["outcome_bin"]
                    event.metadata["dcc_public_goal_readout"] = pub_goal_val
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
            if method in DCC_METHODS:
                from .native_dcc import PublicBenchmarkBinder
                task_instruction = str(context.get("instruction", task_info.get("instruction", "")))
                trajectory.metadata["dcc_public_goal_readout"] = PublicBenchmarkBinder.public_goal_readout(
                    benchmark_name,
                    task_instruction,
                    trajectory.events,
                    env.action_schema,
                    terminal_success=bool(trajectory.success),
                    within_budget=trajectory.elapsed_seconds <= trajectory.budget_seconds,
                )
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


def _extract_public_transitions(
    events: Sequence[NativeEvent],
    *,
    task_id: str,
    episode_id: str,
    cluster_id: str,
    benchmark: str,
    goal_text: str,
    max_env_steps: int,
    terminal_outcome: bool | None,
    global_budget_limited: bool,
    start_index: int = 0,
) -> list[Any]:
    """Convert physically executed NativeEvents into supervised PublicTransition records."""
    from .native_dcc import PublicBenchmarkBinder, PublicTransition, _short_hash

    transitions = []
    total_events = len(events)
    for idx in range(start_index, total_events):
        ev = events[idx]
        is_last = idx == (total_events - 1)
        pre_fields = dict(
            ev.metadata.get("dcc_pre_public_fields")
            or PublicBenchmarkBinder.extract_public_fields(benchmark, goal_text, events[:idx], {})
        )
        post_fields = dict(
            ev.metadata.get("dcc_public_fields")
            or PublicBenchmarkBinder.extract_public_fields(benchmark, goal_text, events[: idx + 1], {})
        )
        role_bindings = PublicBenchmarkBinder.extract_role_bindings(benchmark, goal_text, events[: idx + 1], {})
        act_text = ev.generation.text.strip()
        act_tmpl = PublicBenchmarkBinder.abstract_action_to_template(benchmark, act_text, role_bindings)

        if is_last and global_budget_limited:
            term_cat = "live"
            nat_out = None
        elif bool(ev.metadata.get("native_done")):
            term_cat = "native_terminal"
            nat_out = bool(ev.metadata.get("native_terminal_success")) if terminal_outcome is None else bool(terminal_outcome)
        elif is_last and (idx + 1) >= max_env_steps:
            term_cat = "task_budget_terminal"
            nat_out = False if terminal_outcome is None else bool(terminal_outcome)
        else:
            term_cat = "live"
            nat_out = terminal_outcome if (is_last and terminal_outcome is not None and not global_budget_limited) else None

        rem_ratio = max(0.0, 1.0 - (float(idx + 1) / float(max(1, max_env_steps))))
        transitions.append(
            PublicTransition(
                transition_id=_short_hash((cluster_id, episode_id, idx, act_text)),
                task_id=task_id,
                episode_id=episode_id,
                cluster_id=cluster_id,
                benchmark=benchmark,
                step_index=idx,
                goal_text=goal_text,
                pre_history_ref=f"{episode_id}:step:{idx}",
                pre_public_fields=pre_fields,
                action_text=act_text,
                action_template=act_tmpl,
                role_bindings=role_bindings,
                raw_response=str(ev.observation)[:512],
                typed_response=post_fields,
                resource_delta={
                    "env_steps": 1.0,
                    "cost_ratio": 1.0 / float(max(1, max_env_steps)),
                    "input_tokens": float(ev.generation.input_tokens),
                    "output_tokens": float(ev.generation.output_tokens),
                },
                remaining_budget={"ratio": rem_ratio, "steps_left": float(max(0, max_env_steps - idx - 1))},
                termination=term_cat,
                native_outcome=nat_out,
            )
        )
    return transitions


def collect_branch_pair(
    policy: Any, env: Any, task: str | Mapping[str, Any], config: Mapping[str, Any],
    meter: BudgetMeter, *, seed: int, stratified: bool,
    progress_supervisor: Any | None = None,
    method: str = "prism_evolve",
) -> BranchPair:
    """DCC v2 unified measurement block xi = (h, q, z, u -> pi_0, v -> pi_0).

    1. Fix one bootstrap member m_ep for the episode and reach an anchor h along the main trajectory.
    2. Select executable program q in Xi using m_ep's unified long-horizon value Q_{m_ep}^q (no EVSI gate)
       and execute q's AST templates step-by-step via `step_program_interpreter`.
    3. If q does not end the native task, save the complete post-q checkpoint:
       (environment snapshot + public history + controller state + bindings + RNG state).
       Never revert to the pre-q snapshot while retaining post-q readout z.
    4. Run factual continuation u -> pi_0 to native/task-budget terminal as the main trajectory suffix,
       then restore the exact post-q checkpoint and run counterfactual continuation v -> pi_0 to
       native/task-budget terminal.
    5. Store all unique physical transitions in one measurement cluster and record observed terminal
       difference d_real separately from model-predicted difference d_model.
    """
    from .native_dcc import (
        TAIL_POLICY_VERSION,
        MeasurementRecord,
        ProgramExecutionState,
        PublicBenchmarkBinder,
        _short_hash,
    )

    task_info = _task(task)
    limits = _limits(config)
    benchmark_name = str(config.get("benchmark", getattr(env, "benchmark", "alfworld"))).split(":", 1)[0]
    prefix_meter = BudgetMeter(limits, parent=meter)
    with _meters(policy, env, prefix_meter), _preserve_actor_rng():
        context = env.reset(str(task_info["task_id"]), seed=seed)

    task_instruction = str(context.get("instruction", task_info.get("instruction", "")))
    is_dcc = method in DCC_METHODS and hasattr(policy, "dcc_ensemble")
    if is_dcc:
        ep_member = int(seed) % int(policy.dcc_ensemble.ensemble_size)
        policy.reset_dcc_episode(
            benchmark_name, task_instruction=task_instruction, member_index=ep_member
        )

    anchor_trajectory = rollout(policy, env, task_info, config, prefix_meter, seed=seed,
                                method=method, training=True, context=context,
                                stop_at_frontier=True, progress_supervisor=progress_supervisor)
    if anchor_trajectory.termination != "prefix_ready":
        reason = "global_budget_during_prefix" if anchor_trajectory.metadata.get("global_budget_limited") else None
        if is_dcc:
            cluster_id = _short_hash((task_info["task_id"], seed, "prefix_only"))
            term_out = (
                None if anchor_trajectory.metadata.get("global_budget_limited")
                else bool(anchor_trajectory.success)
            )
            anchor_trajectory.metadata["dcc_cluster_transitions"] = [
                t.to_dict()
                for t in _extract_public_transitions(
                    anchor_trajectory.events,
                    task_id=str(task_info["task_id"]),
                    episode_id=f"{task_info['task_id']}:prefix",
                    cluster_id=cluster_id,
                    benchmark=benchmark_name,
                    goal_text=task_instruction,
                    max_env_steps=limits.max_env_steps,
                    terminal_outcome=term_out,
                    global_budget_limited=bool(anchor_trajectory.metadata.get("global_budget_limited")),
                )
            ]
        return BranchPair(anchor_trajectory.events, anchor_trajectory, None,
                          len(anchor_trajectory.events), 0.0, stratified, reason)

    library = policy.get_dcc_library(benchmark_name)
    if getattr(policy, "dcc_active_belief", None) is None:
        policy.reset_dcc_episode(benchmark_name, task_instruction=task_instruction)

    anchor_fields = PublicBenchmarkBinder.extract_public_fields(
        benchmark_name, task_instruction, anchor_trajectory.events, env.action_schema
    )
    anchor_bindings = PublicBenchmarkBinder.extract_role_bindings(
        benchmark_name, task_instruction, anchor_trajectory.events, env.action_schema
    )
    candidate_progs = library.active_programs(
        starter_only=(method in ("dcc_no_program_growth", "dcc_k0_planner"))
    )
    use_init_params = method in ("dcc_reset_parameters", "dcc_k0_planner")

    # Select executable probe program q using the episode's bootstrap member Q_{m_ep}^q
    selected_prog = candidate_progs[0]
    best_q_score = -1e9
    for prog in candidate_progs:
        temp_state = ProgramExecutionState(program=prog, bound_variables=dict(anchor_bindings))
        first_act, _ = PublicBenchmarkBinder.step_program_interpreter(
            temp_state, env.action_schema, task_instruction, anchor_trajectory.events
        )
        cand_act = first_act or (prog.step_templates[0] if prog.step_templates else "look")
        q_info = policy.dcc_ensemble.evaluate_candidate_q(
            goal_text=task_instruction,
            public_fields=anchor_fields,
            belief=policy.dcc_active_belief,
            candidate_action=cand_act,
            program=prog,
            remaining_budget_ratio=max(0.0, 1.0 - len(anchor_trajectory.events) / max(1, limits.max_env_steps)),
            member_index=getattr(policy, "dcc_active_member_index", None),
            use_initial_params=use_init_params,
        )
        if q_info["q_selected"] > best_q_score:
            best_q_score = q_info["q_selected"]
            selected_prog = prog

    # Execute bounded probe program q (1..3 steps) by stepping the program interpreter (fixes I06)
    max_probe_steps = int(config.get("dcc_max_probe_steps", 3))
    probe_steps = min(max_probe_steps, max(1, len(selected_prog.step_templates)))
    probe_events = list(anchor_trajectory.events)
    probe_messages = [dict(item) for item in anchor_trajectory.metadata["final_messages"]]
    probe_started = monotonic_seconds()
    probe_elapsed = float(anchor_trajectory.elapsed_seconds)
    probe_prog_state = ProgramExecutionState(
        program=selected_prog, bound_variables=dict(anchor_bindings)
    )
    probe_done = False
    probe_success = False
    probe_global_limited = False

    with _meters(policy, env, prefix_meter):
        for _ in range(probe_steps):
            if len(probe_events) >= limits.max_env_steps:
                break
            try:
                prefix_meter.ensure_env_steps(1)
                before_msgs = _decision_messages(probe_messages, anchor_trajectory, limits, prefix_meter)
                schema = env.schema_for_history(before_msgs)
                modes = available_action_modes(schema)
                if not modes:
                    break
                rem_ratio = max(0.0, 1.0 - (len(probe_events) / max(1, limits.max_env_steps)))
                generation = policy.act(
                    before_msgs,
                    schema,
                    forced_mode=None,
                    greedy=False,
                    role="actor",
                    events=probe_events,
                    task_instruction=task_instruction,
                    remaining_budget_ratio=rem_ratio,
                    forced_program=probe_prog_state,
                )
                if getattr(policy, "dcc_active_program_state", None) is not None:
                    probe_prog_state = policy.dcc_active_program_state
                env_t0 = monotonic_seconds()
                step = env.step(generation.action)
                ev = NativeEvent(
                    messages_before=before_msgs,
                    generation=generation,
                    observation=step.observation,
                    environment_seconds=monotonic_seconds() - env_t0,
                    metadata={
                        "public_error": _public_error(step.metadata, step.observation),
                        "native_step": dict(step.metadata),
                        "native_done": bool(step.done),
                        "native_terminal_success": bool(step.done and step.success),
                        "dcc_probe_step": True,
                        "dcc_program_id": selected_prog.program_id,
                    },
                    receipts=deepcopy(step.metadata.get("receipts", [])),
                )
                probe_events.append(ev)
                append_interaction(probe_messages, generation, step)
                facts = step.metadata.get("native_transition", {})
                obs_upd = policy.observe_dcc_transition(
                    benchmark_name,
                    task_instruction,
                    probe_events,
                    schema,
                    actual_action=generation.text.strip(),
                    remaining_budget_ratio=rem_ratio,
                    had_error=bool(facts.get("hard_bad") or ev.metadata.get("public_error")),
                    terminal_done=bool(step.done),
                    terminal_success=bool(step.done and step.success),
                )
                ev.metadata["dcc_pre_public_fields"] = obs_upd["pre_public_fields"]
                ev.metadata["dcc_public_fields"] = obs_upd["post_public_fields"]
                ev.metadata["dcc_outcome_bin"] = obs_upd["outcome_bin"]
                if step.done:
                    probe_done = True
                    probe_success = bool(step.success)
                    break
            except BudgetExhausted as error:
                probe_global_limited = _global_budget_limited(prefix_meter, str(error))
                break

    probe_elapsed = max(0.0, float(anchor_trajectory.elapsed_seconds) + (monotonic_seconds() - probe_started))
    probe_trajectory = NativeTrajectory(
        str(task_info["task_id"]),
        str(context.get("scenario_id", task_info["scenario_id"])),
        events=probe_events,
        elapsed_seconds=probe_elapsed,
        budget_seconds=limits.max_wall_seconds,
        success=probe_success and probe_elapsed <= limits.max_wall_seconds,
        termination=(
            "budget:global" if probe_global_limited
            else ("native_terminal" if probe_Success else "native_failure") if probe_done
            else "prefix_ready"
        ) if False else (
            "budget:global" if probe_global_limited
            else ("native_terminal" if probe_success else "native_failure") if probe_done
            else "prefix_ready"
        ),
        metadata={
            "environment_seed": seed,
            "actor_contract": actor_contract(method, classic_text=bool(env.action_schema.get("classic_text"))),
            "final_messages": probe_messages,
            "resource_usage": prefix_meter.snapshot(),
            "global_budget_limited": probe_global_limited,
        },
    )

    cluster_id = _short_hash((task_info["task_id"], seed, len(probe_events), selected_prog.program_id))
    if probe_global_limited:
        probe_trajectory.metadata["dcc_cluster_transitions"] = [
            t.to_dict()
            for t in _extract_public_transitions(
                probe_events,
                task_id=str(task_info["task_id"]),
                episode_id=f"{task_info['task_id']}:probe",
                cluster_id=cluster_id,
                benchmark=benchmark_name,
                goal_text=task_instruction,
                max_env_steps=limits.max_env_steps,
                terminal_outcome=None,
                global_budget_limited=True,
            )
        ]
        return BranchPair(probe_events, probe_trajectory, None,
                          len(probe_events), 0.0, stratified, "global_budget_during_prefix")

    public_readout_z = PublicBenchmarkBinder.extract_public_fields(
        benchmark_name, task_instruction, probe_events, env.action_schema
    )
    post_q_bindings = PublicBenchmarkBinder.extract_role_bindings(
        benchmark_name, task_instruction, probe_events, env.action_schema
    )
    z_parity_bit = int(_short_hash(public_readout_z), 16) & 1

    # If q already ended the native task, retain the real trajectory for response learning without reverting pre-q (fixes I07)
    if probe_done or len(probe_events) >= limits.max_env_steps:
        compute_training_return(probe_trajectory, 0.0)
        probe_trajectory.metadata["dcc_cluster_transitions"] = [
            t.to_dict()
            for t in _extract_public_transitions(
                probe_events,
                task_id=str(task_info["task_id"]),
                episode_id=f"{task_info['task_id']}:probe_terminal",
                cluster_id=cluster_id,
                benchmark=benchmark_name,
                goal_text=task_instruction,
                max_env_steps=limits.max_env_steps,
                terminal_outcome=bool(probe_trajectory.success),
                global_budget_limited=False,
            )
        ]
        return BranchPair(probe_events, probe_trajectory, None, len(probe_events), 1.0, stratified, None)

    # Save complete post-q checkpoint: environment + public history + controller state + bindings + RNG state (fixes I07, I08)
    with _meters(policy, env, meter):
        post_q_env_snapshot = env.save_snapshot()
    post_q_controller_state = policy.snapshot_dcc_controller_state() if hasattr(policy, "snapshot_dcc_controller_state") else {}
    post_q_checkpoint_id = _short_hash((cluster_id, "post_q", len(probe_events)))

    prefix = list(probe_events)
    split_step = len(prefix)
    messages = [dict(item) for item in probe_messages]
    root_messages = _decision_messages(messages, probe_trajectory, limits, prefix_meter)
    branch_modes = available_action_modes(env.schema_for_history(root_messages))
    forced_first = "I" if (stratified and "I" in branch_modes) else None
    forced_second = "E" if (stratified and "E" in branch_modes) else None
    prefix_usage = probe_trajectory.metadata["resource_usage"]

    # Determine continuation horizon: run to native/task-budget terminal (or configured horizon when explicitly overridden)
    cont_horizon_cfg = config.get("dcc_full_tail_rollout", True)
    continuation_steps = (
        limits.max_env_steps
        if cont_horizon_cfg
        else int(config.get("dcc_continuation_steps", limits.max_env_steps))
    )
    cont_stop_after = min(limits.max_env_steps, split_step + continuation_steps)

    shared_cont_seed = (2 * seed + 1) % (2**63 - 1)
    second_cont_seed = shared_cont_seed if forced_first != forced_second else (2 * seed + 2) % (2**63 - 1)

    # Factual branch u -> pi_0 (retained as the main trajectory suffix)
    first_meter = _continuation_meter(limits, meter, prefix_usage)
    with _preserve_actor_rng():
        torch.manual_seed(shared_cont_seed)
        first = rollout(
            policy, env, task_info, config, first_meter, seed=seed, method=method, training=True,
            context=context, prefix=prefix, prefix_messages=messages,
            prefix_seconds=probe_trajectory.elapsed_seconds,
            forced_mode=forced_first,
            stop_after=cont_stop_after,
            stop_after_termination="budget:episode_environment_steps",
            first_decision_messages=root_messages,
            progress_supervisor=progress_supervisor,
        )
    first.metadata["continuation_seed"] = shared_cont_seed

    # Restore complete post-q checkpoint before counterfactual branch v -> pi_0 (fixes I08)
    if hasattr(policy, "restore_dcc_controller_state"):
        policy.restore_dcc_controller_state(post_q_controller_state)

    if first.metadata.get("global_budget_limited"):
        first.metadata["dcc_cluster_transitions"] = [
            t.to_dict()
            for t in _extract_public_transitions(
                first.events,
                task_id=str(task_info["task_id"]),
                episode_id=f"{task_info['task_id']}:u_partial",
                cluster_id=cluster_id,
                benchmark=benchmark_name,
                goal_text=task_instruction,
                max_env_steps=limits.max_env_steps,
                terminal_outcome=None,
                global_budget_limited=True,
            )
        ]
        return BranchPair(prefix, first, None, split_step, 0.0, stratified, "global_budget_during_first_branch")

    try:
        with _meters(policy, env, meter):
            _restore_environment(env, post_q_env_snapshot)
    except BudgetExhausted as error:
        return BranchPair(prefix, first, None, split_step, 0.0, stratified, f"restore_budget:{error}")

    second_meter = _continuation_meter(limits, meter, prefix_usage)
    with _preserve_actor_rng():
        torch.manual_seed(second_cont_seed)
        second = rollout(
            policy, env, task_info, config, second_meter, seed=seed, method=method, training=True,
            context=context, prefix=prefix, prefix_messages=messages,
            prefix_seconds=probe_trajectory.elapsed_seconds,
            forced_mode=forced_second,
            stop_after=cont_stop_after,
            stop_after_termination="budget:episode_environment_steps",
            first_decision_messages=root_messages,
            progress_supervisor=progress_supervisor,
        )
    second.metadata["continuation_seed"] = second_cont_seed

    # Collect unique physical transitions across shared prefix + factual suffix + counterfactual suffix
    u_global_lim = bool(first.metadata.get("global_budget_limited"))
    v_global_lim = bool(second.metadata.get("global_budget_limited"))
    u_nat_outcome: bool | None = None if u_global_lim else bool(first.success)
    v_nat_outcome: bool | None = None if v_global_lim else bool(second.success)

    cluster_transitions = _extract_public_transitions(
        first.events,
        task_id=str(task_info["task_id"]),
        episode_id=f"{task_info['task_id']}:factual_u",
        cluster_id=cluster_id,
        benchmark=benchmark_name,
        goal_text=task_instruction,
        max_env_steps=limits.max_env_steps,
        terminal_outcome=u_nat_outcome,
        global_budget_limited=u_global_lim,
        start_index=0,
    )
    if len(second.events) > split_step:
        cluster_transitions.extend(
            _extract_public_transitions(
                second.events,
                task_id=str(task_info["task_id"]),
                episode_id=f"{task_info['task_id']}:counterfactual_v",
                cluster_id=cluster_id,
                benchmark=benchmark_name,
                goal_text=task_instruction,
                max_env_steps=limits.max_env_steps,
                terminal_outcome=v_nat_outcome,
                global_budget_limited=v_global_lim,
                start_index=split_step,
            )
        )
    first.metadata["dcc_cluster_transitions"] = [t.to_dict() for t in cluster_transitions]

    if (len(first.events) <= split_step or len(second.events) <= split_step
            or second.metadata.get("global_budget_limited")):
        return BranchPair(prefix, first, second, split_step, 0.0, stratified, "incomplete_branch_pair")

    probability = float(first.events[split_step].generation.metadata.get("gate_probability", 0.5))
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        probability = 0.5

    # Sole utility objective U_B(Y, C): never substitute hand-crafted progress integral for terminal utility
    u_score = float(first.utility)
    v_score = float(second.utility)
    d_real: float | None = (
        float(u_score - v_score)
        if (u_nat_outcome is not None and v_nat_outcome is not None)
        else None
    )
    action_diff_d = float(d_real if d_real is not None else 0.0)

    # Separate model-estimated continuation difference d_model (never written into d_real)
    u_first_act = first.events[split_step].generation.text.strip()
    v_first_act = second.events[split_step].generation.text.strip()
    post_q_belief = post_q_controller_state.get("belief", policy.dcc_active_belief)
    q_u_est = policy.dcc_ensemble.evaluate_candidate_q(
        goal_text=task_instruction,
        public_fields=public_readout_z,
        belief=post_q_belief,
        candidate_action=u_first_act,
        program=selected_prog,
    )["q_mean"]
    q_v_est = policy.dcc_ensemble.evaluate_candidate_q(
        goal_text=task_instruction,
        public_fields=public_readout_z,
        belief=post_q_belief,
        candidate_action=v_first_act,
        program=selected_prog,
    )["q_mean"]
    d_model = float(q_u_est - q_v_est)

    outcome_bin = PublicBenchmarkBinder.encode_response_outcome_bin(
        anchor_fields,
        public_readout_z,
        had_error=bool(prefix[-1].metadata.get("public_error")) if prefix else False,
        terminal_done=bool(first.success or second.success),
        terminal_success=bool(first.success or second.success),
    )
    rebound_templates = tuple(
        PublicBenchmarkBinder.abstract_action_to_template(
            benchmark_name, ev.generation.text, post_q_bindings
        )
        for ev in prefix[-max_probe_steps:]
    ) or selected_prog.step_templates
    rebound_fields = tuple(
        sorted(k for k, v in public_readout_z.items() if v != "unknown")
    )[:4] or selected_prog.readout_fields

    u_tmpl = PublicBenchmarkBinder.abstract_action_to_template(benchmark_name, u_first_act, post_q_bindings)
    v_tmpl = PublicBenchmarkBinder.abstract_action_to_template(benchmark_name, v_first_act, post_q_bindings)
    record = MeasurementRecord(
        record_id=_short_hash((task_info["task_id"], seed, split_step, selected_prog.program_id)),
        benchmark=benchmark_name,
        task_id=str(task_info["task_id"]),
        goal_category=str(task_info.get("category", task_info.get("task_type", "general"))),
        anchor_signature=_short_hash(anchor_fields),
        program_id=selected_prog.program_id,
        step_templates=selected_prog.step_templates,
        readout_fields=selected_prog.readout_fields,
        public_readout_z=public_readout_z,
        z_parity_bit=z_parity_bit,
        continuation_pair_key=f"{u_tmpl}__vs__{v_tmpl}",
        u_score=u_score,
        v_score=v_score,
        u_cost=len(first.events) - split_step,
        v_cost=len(second.events) - split_step,
        u_done=bool(first.success),
        v_done=bool(second.success),
        action_diff_d=action_diff_d,
        outcome_bin=outcome_bin,
        surprise=0.0,
        rebound_trace_templates=rebound_templates,
        rebound_trace_fields=rebound_fields,
        post_q_checkpoint_id=post_q_checkpoint_id,
        u_policy_id=u_tmpl,
        v_policy_id=v_tmpl,
        tail_policy_version=TAIL_POLICY_VERSION,
        factual_tail_ref=f"{task_info['task_id']}:factual_u:{split_step}:{len(first.events)}",
        counterfactual_tail_ref=f"{task_info['task_id']}:counterfactual_v:{split_step}:{len(second.events)}",
        u_native_outcome=u_nat_outcome,
        v_native_outcome=v_nat_outcome,
        d_real=d_real,
        d_model=d_model,
        d_source="observed_native_terminal" if d_real is not None else "missing_unobserved",
        role_bindings=post_q_bindings,
    )
    first.metadata["dcc_measurement_record"] = record.to_dict()
    first.metadata["dcc_selected_program"] = selected_prog.to_dict()
    first.metadata["dcc_anchor_fields"] = dict(anchor_fields)
    first.metadata["dcc_post_q_checkpoint_id"] = post_q_checkpoint_id
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


def _update_dcc_block(
    policy: Any,
    optimizer: Any,
    sample: BranchPair,
    proposer: Any,
    library: Any,
    meter: BudgetMeter,
    config: Mapping[str, Any],
    *,
    allow_program_growth: bool = True,
    fixed_d_proposer: Any | None = None,
    fixed_d_library: Any | None = None,
) -> dict[str, Any]:
    """Update shared controlled response ensemble K_phi on real physical transitions and grow library Xi."""
    from .native_dcc import ExecutableProgram, MeasurementRecord, PublicTransition

    raw_transitions = sample.first.metadata.get("dcc_cluster_transitions", ())
    transitions = [
        PublicTransition(
            transition_id=str(t["transition_id"]),
            task_id=str(t["task_id"]),
            episode_id=str(t["episode_id"]),
            cluster_id=str(t["cluster_id"]),
            benchmark=str(t["benchmark"]),
            step_index=int(t["step_index"]),
            goal_text=str(t["goal_text"]),
            pre_history_ref=str(t["pre_history_ref"]),
            pre_public_fields=dict(t["pre_public_fields"]),
            action_text=str(t["action_text"]),
            action_template=str(t["action_template"]),
            role_bindings=dict(t["role_bindings"]),
            raw_response=str(t["raw_response"]),
            typed_response=dict(t["typed_response"]),
            resource_delta={str(k): float(v) for k, v in dict(t["resource_delta"]).items()},
            remaining_budget={str(k): float(v) for k, v in dict(t["remaining_budget"]).items()},
            termination=str(t["termination"]),
            native_outcome=t.get("native_outcome"),
        )
        for t in raw_transitions
        if isinstance(t, Mapping)
    ]

    if not transitions and (sample.partial_reason is not None or sample.second is None):
        return {
            "updated": False,
            "learning_outcome": "incomplete_pair_budget",
            "loss": 0.0,
            "nonzero_gradient": False,
            "parameters_changed": False,
            "parameter_delta_l2": 0.0,
        }

    # Charge exact forward tokens proportional to unique physical transitions (never a fixed 32-token stub)
    forward_tokens_to_charge = max(
        1,
        sum(
            max(1, int(len(tr.action_text.split()) + len(tr.raw_response.split())))
            for tr in transitions
        ) if transitions else 16,
    )
    meter.ensure_forward(forward_tokens_to_charge, role="dcc_response_ensemble_update")
    started = monotonic_seconds()
    optimizer.zero_grad(set_to_none=True)

    # Cluster-level Poisson(1) / Dirichlet bootstrap weights over the measurement cluster
    M = int(policy.dcc_ensemble.ensemble_size)
    cluster_seed = int(hash(transitions[0].cluster_id) & 0x7FFFFFFF) if transitions else 42
    gen_rng = torch.Generator(device="cpu").manual_seed(cluster_seed)
    poisson_counts = torch.poisson(torch.ones(M, dtype=torch.float32), generator=gen_rng)
    if float(poisson_counts.sum().item()) <= 0.0:
        poisson_counts = torch.ones(M, dtype=torch.float32)
    bootstrap_w = poisson_counts.to(policy.device)

    rec_dict = sample.first.metadata.get("dcc_measurement_record")
    prog_dict = sample.first.metadata.get("dcc_selected_program")
    anchor_fields = dict(sample.first.metadata.get("dcc_anchor_fields", {}))

    if transitions:
        loss_tensor, next_belief, trans_stats = policy.dcc_ensemble.transition_negative_log_likelihood(
            transitions, bootstrap_weights=bootstrap_w
        )
    elif isinstance(rec_dict, Mapping) and isinstance(prog_dict, Mapping):
        program_tmp = ExecutableProgram.from_dict(prog_dict)
        belief_init = policy.dcc_ensemble.reset_intra_task_belief(device=policy.device)
        loss_tensor, next_belief = policy.dcc_ensemble.block_negative_log_likelihood(
            program_tmp,
            anchor_fields or dict(rec_dict.get("public_readout_z", {})),
            belief_init,
            int(rec_dict.get("outcome_bin", 0)),
            float(rec_dict.get("u_score", 0.0)),
            float(rec_dict.get("v_score", 0.0)),
            u_native_outcome=rec_dict.get("u_native_outcome"),
            v_native_outcome=rec_dict.get("v_native_outcome"),
            bootstrap_weights=bootstrap_w,
        )
        trans_stats = {"num_transitions": 1.0, "observed_terminal_outcomes": 0.0}
    else:
        return {
            "updated": False,
            "learning_outcome": "missing_dcc_measurement",
            "loss": 0.0,
            "nonzero_gradient": False,
            "parameters_changed": False,
            "parameter_delta_l2": 0.0,
        }

    policy.dcc_active_belief = next_belief
    loss_val = float(loss_tensor.detach().item())

    parameters = [p for p in policy.dcc_ensemble.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in parameters]
    loss_tensor.backward()
    clip = float(config.get("grad_clip_norm", 1.0))
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(
            parameters, clip if clip > 0 else float("inf"), error_if_nonfinite=True
        ).detach().item()
    )
    optimizer.step()
    with torch.no_grad():
        param_delta = math.sqrt(
            sum(float((p.detach() - prev).float().square().sum().item()) for p, prev in zip(parameters, before))
        )
    del before
    meter.record_update(monotonic_seconds() - started)
    optimizer.zero_grad(set_to_none=True)

    if not isinstance(rec_dict, Mapping) or not isinstance(prog_dict, Mapping):
        return {
            "updated": param_delta > 0,
            "learning_outcome": "parameters_updated_partial_trajectory" if param_delta > 0 else "no_parameter_change",
            "loss": loss_val,
            "score_terms": len(transitions),
            "scoring_forward_tokens": forward_tokens_to_charge,
            "gradient_l2_norm_before_clip": grad_norm,
            "nonzero_gradient": grad_norm > 0,
            "parameters_changed": param_delta > 0,
            "parameter_delta_l2": param_delta,
            "optimizer_step": meter.gradient_steps,
            "dcc_transitions_updated": len(transitions),
        }

    program = ExecutableProgram.from_dict(prog_dict)
    record = MeasurementRecord(
        record_id=str(rec_dict["record_id"]),
        benchmark=str(rec_dict["benchmark"]),
        task_id=str(rec_dict["task_id"]),
        goal_category=str(rec_dict["goal_category"]),
        anchor_signature=str(rec_dict["anchor_signature"]),
        program_id=str(rec_dict["program_id"]),
        step_templates=tuple(rec_dict["step_templates"]),
        readout_fields=tuple(rec_dict["readout_fields"]),
        public_readout_z=dict(rec_dict["public_readout_z"]),
        z_parity_bit=int(rec_dict["z_parity_bit"]),
        continuation_pair_key=str(rec_dict["continuation_pair_key"]),
        u_score=float(rec_dict["u_score"]),
        v_score=float(rec_dict["v_score"]),
        u_cost=int(rec_dict["u_cost"]),
        v_cost=int(rec_dict["v_cost"]),
        u_done=bool(rec_dict["u_done"]),
        v_done=bool(rec_dict["v_done"]),
        action_diff_d=float(rec_dict["action_diff_d"]),
        outcome_bin=int(rec_dict["outcome_bin"]),
        surprise=max(0.0, loss_val),
        rebound_trace_templates=tuple(rec_dict["rebound_trace_templates"]),
        rebound_trace_fields=tuple(rec_dict["rebound_trace_fields"]),
        post_q_checkpoint_id=str(rec_dict.get("post_q_checkpoint_id", "")),
        u_policy_id=str(rec_dict.get("u_policy_id", "u_continuation")),
        v_policy_id=str(rec_dict.get("v_policy_id", "v_continuation")),
        tail_policy_version=str(rec_dict.get("tail_policy_version", "frozen-prior-tail-v1")),
        factual_tail_ref=str(rec_dict.get("factual_tail_ref", "")),
        counterfactual_tail_ref=str(rec_dict.get("counterfactual_tail_ref", "")),
        u_native_outcome=rec_dict.get("u_native_outcome"),
        v_native_outcome=rec_dict.get("v_native_outcome"),
        d_real=rec_dict.get("d_real"),
        d_model=rec_dict.get("d_model"),
        d_source=str(rec_dict.get("d_source", "observed_native_terminal")),
        role_bindings=dict(rec_dict.get("role_bindings", {})),
    )

    proposer.record_measurement(record)
    grown = (
        proposer.grow_library(
            library,
            dataset_version=f"step_{meter.gradient_steps}",
            model_version=f"step_{meter.gradient_steps}",
        )
        if allow_program_growth else []
    )
    # Synchronously advance the matched fixed-D shuffled proposer with the exact same call count and record prefix (fixes E01)
    if fixed_d_proposer is not None and fixed_d_library is not None:
        fixed_d_proposer.record_measurement(record)
        if allow_program_growth:
            fixed_d_proposer.grow_library(
                fixed_d_library,
                dataset_version=f"step_{meter.gradient_steps}",
                model_version=f"step_{meter.gradient_steps}",
            )

    return {
        "updated": param_delta > 0,
        "learning_outcome": "parameters_updated" if param_delta > 0 else "no_parameter_change",
        "loss": loss_val,
        "score_terms": max(1, len(transitions)),
        "scoring_forward_tokens": forward_tokens_to_charge,
        "gradient_l2_norm_before_clip": grad_norm,
        "nonzero_gradient": grad_norm > 0,
        "parameters_changed": param_delta > 0,
        "parameter_delta_l2": param_delta,
        "optimizer_step": meter.gradient_steps,
        "dcc_d_real": record.d_real,
        "dcc_d_model": record.d_model,
        "dcc_action_diff_d": record.action_diff_d,
        "dcc_u_utility": record.u_score,
        "dcc_v_utility": record.v_score,
        "dcc_selected_program_id": program.program_id,
        "dcc_selected_program_origin": program.origin,
        "dcc_grown_programs": [p.program_id for p in grown],
        "dcc_library_size": len(library.active_programs()),
        "dcc_unique_witness_pairs": proposer.witness_pairs_found,
        "dcc_cumulative_witness_visits": proposer.cumulative_witness_visits,
        "dcc_transitions_updated": int(trans_stats.get("num_transitions", len(transitions))),
    }


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
    benchmark_name = str(config.get("benchmark", getattr(env, "benchmark", "alfworld"))).split(":", 1)[0]
    is_dcc = method in DCC_METHODS and hasattr(policy, "configure_dcc_trainable_mode")
    dcc_proposer = None
    dcc_library = None
    fixed_d_shuffled_proposer = None
    fixed_d_shuffled_library = None
    if is_dcc:
        from .native_dcc import ActionContrastProposer, ProgramLibrary
        policy.configure_dcc_trainable_mode(True)
        dcc_library = policy.get_dcc_library(benchmark_name)
        dcc_proposer = ActionContrastProposer(
            benchmark_name,
            max_probe_steps=int(config.get("dcc_max_probe_steps", 3)),
            proposal_budget=int(config.get("dcc_proposal_budget", 3)),
            shuffle_witness_labels=(method in ("dcc_label_shuffled", "dcc_fixed_d_shuffled")),
            seed=seed,
        )
        # Matched fixed-D shuffled proposer that grows at the exact same call frequency on the exact same record prefix
        fixed_d_shuffled_library = ProgramLibrary(benchmark_name)
        fixed_d_shuffled_proposer = ActionContrastProposer(
            benchmark_name,
            max_probe_steps=int(config.get("dcc_max_probe_steps", 3)),
            proposal_budget=int(config.get("dcc_proposal_budget", 3)),
            shuffle_witness_labels=True,
            seed=seed + 999,
        )
    elif hasattr(policy, "configure_dcc_trainable_mode"):
        policy.configure_dcc_trainable_mode(False)
    policy.train()
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The training actor has no trainable parameters.")
    initial = [parameter.detach().clone() for parameter in parameters]
    lr_val = float(config.get("dcc_learning_rate", 5e-3)) if is_dcc else float(config.get("learning_rate", 1e-6))
    optimizer = torch.optim.AdamW(parameters, lr=lr_val,
                                  weight_decay=float(config.get("weight_decay", 0.0)))
    pre_first_success_dir = output / "pre_first_success_adapter"
    first_terminal_success_seen = False
    if is_dcc and not pre_first_success_dir.exists():
        policy.save_adapter(pre_first_success_dir)
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
                if method in (*DCC_METHODS, "ablation_unweighted", "unstratified_2branch"):
                    sample = collect_branch_pair(policy, env, task, config, meter, seed=environment_seed,
                                                 stratified=method != "unstratified_2branch",
                                                 progress_supervisor=supervisor,
                                                 method=method)
                    group_trajectories = [sample.first] + ([sample.second] if sample.second is not None else [])
                    if sample.partial_reason:
                        partial = True
                        stop_reason = sample.partial_reason
                        entry.update(updated=False, learning_outcome="incomplete_pair_budget", reason=stop_reason)
                    else:
                        complete_groups += 1
                        branch_pairs += int(sample.second is not None)
                        if is_dcc and dcc_proposer is not None and dcc_library is not None:
                            # Save pre-first-success checkpoint BEFORE updating on the first successful block
                            if any(bool(t.success) for t in group_trajectories):
                                first_terminal_success_seen = True
                            entry.update(_update_dcc_block(
                                policy, optimizer, sample, dcc_proposer, dcc_library, meter, config,
                                allow_program_growth=(method not in ("dcc_no_program_growth", "dcc_k0_planner")),
                                fixed_d_proposer=fixed_d_shuffled_proposer,
                                fixed_d_library=fixed_d_shuffled_library,
                            ))
                            if not first_terminal_success_seen:
                                import shutil as _shutil
                                if pre_first_success_dir.exists():
                                    _shutil.rmtree(pre_first_success_dir)
                                policy.save_adapter(pre_first_success_dir)
                        else:
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
    if is_dcc:
        policy.configure_dcc_trainable_mode(False)
    policy.eval()
    selected_ckpt = pre_first_success_dir if (is_dcc and method == "dcc_pre_first_success" and pre_first_success_dir.exists()) else checkpoint
    summary = {
        "protocol_version": "native-r3", "method": method, "seed": seed,
        "selected_checkpoint": str(selected_ckpt), "checkpoint_selection": "final_after_fixed_training_run",
        "pre_first_success_checkpoint": str(pre_first_success_dir) if is_dcc and pre_first_success_dir.exists() else None,
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
            *DCC_METHODS, "ablation_unweighted", "unstratified_2branch", "gigpo", "t3", "frozen_base"},
        "task_outcome_reward": "U_B=Y*max(0,1-C/B)",
        "training_objective": ("verified_single_step_preference" if method == "step_dpo"
                               else "dcc_joint_response_likelihood_and_action_contrast_program_growth" if is_dcc
                               else "native_utility_plus_registered_method_specific_advantage"),
        "reference_visibility": "train_only",
        "actor_contract": actor_contract(method, classic_text=bool(env.action_schema.get("classic_text"))),
    }
    if is_dcc and dcc_library is not None and dcc_proposer is not None:
        from .native_dcc import DCC_CONTRACT_VERSION, ActionContrastProposer, ProgramLibrary

        belief_eval = policy.dcc_ensemble.reset_intra_task_belief(device=policy.device)
        # Use observed anchor fields from the latest recorded measurement when available
        ref_fields = (
            dict(dcc_proposer.records[-1].public_readout_z)
            if dcc_proposer.records else {}
        )
        full_progs = dcc_library.active_programs(starter_only=False)
        starter_progs = dcc_library.active_programs(starter_only=True)
        full_scores = [
            policy.dcc_ensemble.score_program_evsi(p, ref_fields, belief_eval, use_Temporary_Initial_Params=False)
            for p in full_progs
        ]
        no_growth_scores = [
            policy.dcc_ensemble.score_program_evsi(p, ref_fields, belief_eval, use_Temporary_Initial_Params=False)
            for p in starter_progs
        ]
        reset_param_scores = [
            policy.dcc_ensemble.score_program_evsi(p, ref_fields, belief_eval, use_Temporary_Initial_Params=True)
            for p in full_progs
        ]
        k0_starter_scores = [
            policy.dcc_ensemble.score_program_evsi(p, ref_fields, belief_eval, use_Temporary_Initial_Params=True)
            for p in starter_progs
        ]

        if fixed_d_shuffled_library is None or fixed_d_shuffled_proposer is None:
            fixed_d_shuffled_library = ProgramLibrary(benchmark_name)
            fixed_d_shuffled_proposer = ActionContrastProposer(
                benchmark_name,
                max_probe_steps=int(config.get("dcc_max_probe_steps", 3)),
                proposal_budget=int(config.get("dcc_proposal_budget", 3)),
                shuffle_witness_labels=True,
                seed=seed + 999,
            )
            for idx_rec, rec in enumerate(dcc_proposer.records, start=1):
                fixed_d_shuffled_proposer.record_measurement(rec)
                fixed_d_shuffled_proposer.grow_library(
                    fixed_d_shuffled_library,
                    dataset_version=f"replay_{idx_rec}",
                    model_version=f"replay_{idx_rec}",
                )

        shuffled_progs = fixed_d_shuffled_library.active_programs(starter_only=False)
        shuffled_scores = [
            policy.dcc_ensemble.score_program_evsi(p, ref_fields, belief_eval, use_Temporary_Initial_Params=False)
            for p in shuffled_progs
        ]

        fixed_d_ckpt_dir = output / "fixed_d_shuffled_adapter"
        if not fixed_d_ckpt_dir.exists():
            orig_lib = policy.dcc_libraries.get(benchmark_name)
            try:
                policy.dcc_libraries[benchmark_name] = fixed_d_shuffled_library
                policy.save_adapter(fixed_d_ckpt_dir)
            finally:
                if orig_lib is not None:
                    policy.dcc_libraries[benchmark_name] = orig_lib

        ablations_payload = {
            "diagnostic_scope": "internal_model_scoring_diagnostic_only; native_trajectory_cells_required_for_performance_claims",
            "benchmark": benchmark_name,
            "seed": seed,
            "complete_dcc_v2": {
                "cell": "phi_T_Xi_T",
                "library_size": len(full_progs),
                "grow_calls": int(dcc_proposer.call_index),
                "unique_witness_pairs": int(dcc_proposer.witness_pairs_found),
                "cumulative_witness_visits": int(dcc_proposer.cumulative_witness_visits),
                "max_unified_q": max((s.get("unified_q", 0.0) for s in full_scores), default=0.0),
                "max_evsi_per_cost": max((s["evsi_per_cost"] for s in full_scores), default=0.0),
                "max_dual_control_j": max((s["dual_control_j"] for s in full_scores), default=0.0),
                "max_feedback_use_value_i": max((s["feedback_use_value_i"] for s in full_scores), default=0.0),
            },
            "dcc_no_program_growth": {
                "cell": "phi_T_Xi_0",
                "library_size": len(starter_progs),
                "max_unified_q": max((s.get("unified_q", 0.0) for s in no_growth_scores), default=0.0),
                "max_evsi_per_cost": max((s["evsi_per_cost"] for s in no_growth_scores), default=0.0),
                "max_dual_control_j": max((s["dual_control_j"] for s in no_growth_scores), default=0.0),
                "max_feedback_use_value_i": max((s["feedback_use_value_i"] for s in no_growth_scores), default=0.0),
            },
            "dcc_reset_parameters": {
                "cell": "phi_0_Xi_T",
                "library_size": len(full_progs),
                "max_unified_q": max((s.get("unified_q", 0.0) for s in reset_param_scores), default=0.0),
                "max_evsi_per_cost": max((s["evsi_per_cost"] for s in reset_param_scores), default=0.0),
                "max_dual_control_j": max((s["dual_control_j"] for s in reset_param_scores), default=0.0),
                "max_feedback_use_value_i": max((s["feedback_use_value_i"] for s in reset_param_scores), default=0.0),
            },
            "dcc_k0_planner": {
                "cell": "phi_0_Xi_0",
                "library_size": len(starter_progs),
                "max_unified_q": max((s.get("unified_q", 0.0) for s in k0_starter_scores), default=0.0),
                "max_evsi_per_cost": max((s["evsi_per_cost"] for s in k0_starter_scores), default=0.0),
                "max_dual_control_j": max((s["dual_control_j"] for s in k0_starter_scores), default=0.0),
                "max_feedback_use_value_i": max((s["feedback_use_value_i"] for s in k0_starter_scores), default=0.0),
            },
            "dcc_fixed_d_shuffled": {
                "library_size": len(shuffled_progs),
                "grow_calls": int(fixed_d_shuffled_proposer.call_index),
                "unique_witness_pairs": int(fixed_d_shuffled_proposer.witness_pairs_found),
                "cumulative_witness_visits": int(fixed_d_shuffled_proposer.cumulative_witness_visits),
                "witness_pairs_found": int(fixed_d_shuffled_proposer.witness_pairs_found),
                "max_unified_q": max((s.get("unified_q", 0.0) for s in shuffled_scores), default=0.0),
                "max_evsi_per_cost": max((s["evsi_per_cost"] for s in shuffled_scores), default=0.0),
                "max_dual_control_j": max((s["dual_control_j"] for s in shuffled_scores), default=0.0),
                "max_feedback_use_value_i": max((s["feedback_use_value_i"] for s in shuffled_scores), default=0.0),
                "checkpoint": str(fixed_d_ckpt_dir),
            },
            "dcc_label_shuffled": {
                "library_size": len(shuffled_progs),
                "grow_calls": int(fixed_d_shuffled_proposer.call_index),
                "unique_witness_pairs": int(fixed_d_shuffled_proposer.witness_pairs_found),
                "cumulative_witness_visits": int(fixed_d_shuffled_proposer.cumulative_witness_visits),
                "witness_pairs_found": int(fixed_d_shuffled_proposer.witness_pairs_found),
                "max_evsi_per_cost": max((s["evsi_per_cost"] for s in shuffled_scores), default=0.0),
                "max_dual_control_j": max((s["dual_control_j"] for s in shuffled_scores), default=0.0),
                "max_feedback_use_value_i": max((s["feedback_use_value_i"] for s in shuffled_scores), default=0.0),
            },
            "pre_first_success_checkpoint": str(pre_first_success_dir) if pre_first_success_dir.exists() else None,
            "zero_success_throughout_training": not bool(first_terminal_success_seen),
        }
        summary["fixed_d_shuffled_checkpoint"] = str(fixed_d_ckpt_dir)
        summary["zero_success_throughout_training"] = not bool(first_terminal_success_seen)
        summary["dcc_summary"] = {
            "contract_version": DCC_CONTRACT_VERSION,
            "frozen_language_backbone": True,
            "ensemble_size": int(policy.dcc_ensemble.ensemble_size),
            "final_mechanism_weights": [float(x) for x in policy.dcc_ensemble.mechanism_weights.detach().cpu().tolist()],
            "library_size": len(dcc_library.active_programs()),
            "grow_calls": int(dcc_proposer.call_index),
            "unique_witness_pairs": int(dcc_proposer.witness_pairs_found),
            "cumulative_witness_visits": int(dcc_proposer.cumulative_witness_visits),
            "witness_pairs_found": int(dcc_proposer.witness_pairs_found),
            "proposed_counts": dict(dcc_proposer.proposed_counts),
            "shuffle_witness_labels": bool(dcc_proposer.shuffle_witness_labels),
            "zero_success_throughout_training": not bool(first_terminal_success_seen),
            "mechanism_ablations": ablations_payload,
        }
        _write_json(output / "dcc_program_library.json", dcc_library.to_dict())
        _write_json(output / "dcc_fixed_d_shuffled_library.json", fixed_d_shuffled_library.to_dict())
        _write_json(
            output / "dcc_proposal_runs.json",
            {
                "full": [r.to_dict() for r in dcc_proposer.proposal_runs],
                "fixed_d_shuffled": [r.to_dict() for r in fixed_d_shuffled_proposer.proposal_runs],
            },
        )
        _write_json(output / "dcc_mechanism_ablations.json", ablations_payload)
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
