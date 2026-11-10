"""Shared records and measured resource limits for protocol ``native-r3``.

This module has no model or benchmark imports. Public observations and private
terminal evaluation records stay separate at the environment boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


class BudgetExhausted(RuntimeError):
    """An explicitly configured resource limit prevents the next operation."""


def monotonic_seconds() -> float:
    """Use the OS clock; AppWorld freezes Python's perf_counter/monotonic."""
    return time.clock_gettime(time.CLOCK_MONOTONIC)


@dataclass(frozen=True)
class NativeStep:
    observation: str
    done: bool = False
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NativeGeneration:
    text: str
    prompt_ids: tuple[int, ...] = ()
    response_ids: tuple[int, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0
    role: str = "actor"
    action: dict[str, Any] | None = None
    mode: str | None = None
    mode_prompt_ids: tuple[int, ...] = ()
    mode_token_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetLimits:
    max_env_steps: int
    max_model_tokens: int
    max_wall_seconds: float

    def __post_init__(self) -> None:
        if self.max_env_steps <= 0 or self.max_model_tokens <= 0:
            raise ValueError("Environment and model-token limits must be positive.")
        if not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0:
            raise ValueError("Wall-time limit must be finite and positive.")


class BudgetMeter:
    """One budget shared by actor, judges, auxiliary models and replay.

    ``max_model_tokens`` counts input, generated output and teacher-forced
    forward tokens. The three quantities are also retained independently.
    Child episode meters can share a cumulative training meter. Restoring an
    environment never restores the resource ledger.
    """

    def __init__(
        self,
        limits: BudgetLimits,
        *,
        parent: BudgetMeter | None = None,
        event_path: str | Path | None = None,
    ) -> None:
        self.limits = limits
        self.parent = parent
        self.event_path = Path(event_path) if event_path is not None else None
        self.started = monotonic_seconds()
        self.env_steps = 0
        self.replayed_env_steps = 0
        self.native_api_calls = 0
        self.successful_native_api_calls = 0
        self.replayed_api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.forward_tokens = 0
        self.model_seconds = 0.0
        self.environment_seconds = 0.0
        self.restore_seconds = 0.0
        self.snapshot_saves = 0
        self.snapshot_restores = 0
        self.snapshot_save_seconds = 0.0
        self.update_seconds = 0.0
        self.gradient_steps = 0
        self.calls_by_role: dict[str, int] = {}

    @property
    def total_model_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.forward_tokens

    @property
    def remaining_model_tokens(self) -> int:
        remaining = self.limits.max_model_tokens - self.total_model_tokens
        if self.parent is not None:
            remaining = min(remaining, self.parent.remaining_model_tokens)
        return max(0, remaining)

    @property
    def remaining_env_steps(self) -> int:
        remaining = self.limits.max_env_steps - self.env_steps
        if self.parent is not None:
            remaining = min(remaining, self.parent.remaining_env_steps)
        return max(0, remaining)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, monotonic_seconds() - self.started)

    def _ensure(self, env_steps: int = 0, model_tokens: int = 0) -> None:
        if env_steps < 0 or model_tokens < 0:
            raise ValueError("Reserved resources cannot be negative.")
        if self.elapsed_seconds >= self.limits.max_wall_seconds:
            raise BudgetExhausted("wall_seconds")
        if self.env_steps + env_steps > self.limits.max_env_steps:
            raise BudgetExhausted("environment_steps")
        if self.total_model_tokens + model_tokens > self.limits.max_model_tokens:
            raise BudgetExhausted("model_tokens")
        if self.parent is not None:
            self.parent._ensure(env_steps, model_tokens)

    def ensure_env_steps(self, count: int = 1) -> None:
        self._ensure(env_steps=count)

    def ensure_model_call(self, input_tokens: int, max_output_tokens: int, role: str = "actor") -> None:
        self._ensure(model_tokens=int(input_tokens) + int(max_output_tokens))

    def ensure_forward(self, input_tokens: int, role: str = "score") -> None:
        self._ensure(model_tokens=int(input_tokens))

    def _append(self, record: Mapping[str, Any]) -> None:
        if self.event_path is not None:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n")

    def record_model_call(
        self,
        input_tokens: int,
        output_tokens: int,
        elapsed_seconds: float,
        role: str = "actor",
        forward_tokens: int = 0,
    ) -> None:
        if min(input_tokens, output_tokens, forward_tokens) < 0:
            raise ValueError("Token counts cannot be negative.")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Measured model time must be finite and nonnegative.")
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        self.forward_tokens += int(forward_tokens)
        self.model_seconds += float(elapsed_seconds)
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1
        self._append({"kind": "model_call", "role": role, "input_tokens": input_tokens,
                      "output_tokens": output_tokens, "forward_tokens": forward_tokens,
                      "elapsed_seconds": elapsed_seconds})
        if self.parent is not None:
            self.parent.record_model_call(input_tokens, output_tokens, elapsed_seconds, role, forward_tokens)

    def record_env_step(self, elapsed_seconds: float = 0.0, *, replay: bool = False) -> None:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Measured environment time must be finite and nonnegative.")
        self.env_steps += 1
        self.replayed_env_steps += int(replay)
        self.environment_seconds += float(elapsed_seconds)
        self._append({"kind": "environment_step", "replay": replay, "elapsed_seconds": elapsed_seconds})
        if self.parent is not None:
            self.parent.record_env_step(elapsed_seconds, replay=replay)

    def record_restore(self, elapsed_seconds: float, replay_steps: int = 0) -> None:
        if replay_steps < 0 or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Invalid restore accounting.")
        self.restore_seconds += float(elapsed_seconds)
        self.snapshot_restores += 1
        for _ in range(replay_steps):
            self.record_env_step(replay=True)
        self._append({"kind": "snapshot_restore", "elapsed_seconds": elapsed_seconds,
                      "replayed_env_steps": replay_steps})
        if self.parent is not None:
            self.parent.record_restore(elapsed_seconds, replay_steps=0)

    def record_api_calls(self, count: int, *, successful: int = 0, replay: bool = False) -> None:
        """Count a completed native request batch separately from policy turns.

        This method writes the physical ledger and its parent ledger. Native
        adapters must call it after leaving their execution sandbox; request
        callbacks inside AppWorld's SafetyGuard collect counts only in memory.
        """
        if count < 0 or successful < 0 or successful > count:
            raise ValueError("Invalid native API call accounting.")
        self.native_api_calls += int(count)
        self.successful_native_api_calls += int(successful)
        self.replayed_api_calls += int(count) if replay else 0
        self._append({"kind": "native_api_calls", "attempted": int(count),
                      "successful": int(successful), "replay": bool(replay)})
        if self.parent is not None:
            self.parent.record_api_calls(count, successful=successful, replay=replay)

    def record_snapshot_save(self, elapsed_seconds: float) -> None:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Invalid snapshot-save time.")
        self.snapshot_saves += 1
        self.snapshot_save_seconds += float(elapsed_seconds)
        self._append({"kind": "snapshot_save", "elapsed_seconds": elapsed_seconds})
        if self.parent is not None:
            self.parent.record_snapshot_save(elapsed_seconds)

    def record_update(self, elapsed_seconds: float) -> None:
        self.gradient_steps += 1
        self.update_seconds += float(elapsed_seconds)
        self._append({"kind": "gradient_update", "elapsed_seconds": elapsed_seconds})
        if self.parent is not None:
            self.parent.record_update(elapsed_seconds)

    def snapshot(self) -> dict[str, Any]:
        return {"limits": asdict(self.limits), "env_steps": self.env_steps,
                "replayed_env_steps": self.replayed_env_steps, "input_tokens": self.input_tokens,
                "native_api_calls": self.native_api_calls,
                "successful_native_api_calls": self.successful_native_api_calls,
                "replayed_api_calls": self.replayed_api_calls,
                "output_tokens": self.output_tokens, "forward_tokens": self.forward_tokens,
                "total_model_tokens": self.total_model_tokens, "model_seconds": self.model_seconds,
                "environment_seconds": self.environment_seconds, "restore_seconds": self.restore_seconds,
                "snapshot_saves": self.snapshot_saves, "snapshot_restores": self.snapshot_restores,
                "snapshot_save_seconds": self.snapshot_save_seconds,
                "gradient_steps": self.gradient_steps, "update_seconds": self.update_seconds,
                "wall_seconds": self.elapsed_seconds, "calls_by_role": dict(self.calls_by_role)}


@dataclass
class NativeEvent:
    messages_before: list[dict[str, str]]
    generation: NativeGeneration
    observation: str
    environment_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    receipts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class NativeTrajectory:
    task_id: str
    scenario_id: str
    events: list[NativeEvent] = field(default_factory=list)
    success: bool = False
    elapsed_seconds: float = 0.0
    budget_seconds: float = 1800.0
    termination: str = "unfinished"
    metadata: dict[str, Any] = field(default_factory=dict)
    training_return: float | None = None

    @property
    def utility(self) -> float:
        return float(self.success) * max(0.0, 1.0 - self.elapsed_seconds / self.budget_seconds)

    def result(self, *, benchmark: str, backbone: str, seed: int, method: str) -> dict[str, Any]:
        return {"protocol_version": "native-r3", "benchmark": benchmark, "backbone": backbone,
                "seed": seed, "method": method, "task_id": self.task_id,
                "scenario_id": self.scenario_id, "success": int(self.success),
                "utility": self.utility, "elapsed_seconds": self.elapsed_seconds,
                "budget_seconds": self.budget_seconds, "environment_steps": len(self.events),
                "termination": self.termination, "training_return": self.training_return,
                "metadata": self.metadata}


def initial_messages(context: Mapping[str, Any]) -> list[dict[str, str]]:
    """Serialize only the environment's explicitly public reset context."""
    system = context.get("system_prompt", "")
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    instruction = str(context.get("instruction", "")).strip()
    observation = str(context.get("observation", "")).strip()
    if observation:
        try:
            obs_obj = json.loads(observation)
            public_observation = obs_obj
        except (TypeError, ValueError):
            public_observation = observation
    else:
        public_observation = ""
    content = json.dumps({"feedback_type": "task", "instruction": instruction,
                          "observation": public_observation},
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    messages.append({"role": "user", "content": content})
    return messages


def append_interaction(
    messages: list[dict[str, str]], generation: NativeGeneration,
    observation: str | NativeStep,
) -> list[dict[str, str]]:
    """Append the actual model response and native observation in place.

    Return the same list so both ``append_interaction(messages, ...)`` and
    ``messages = append_interaction(messages, ...)`` retain the transition.
    A compiled executor action never replaces the text sampled by the model.
    """
    if not isinstance(messages, list):
        raise TypeError("Interaction history must be a mutable list.")
    text = observation.observation if isinstance(observation, NativeStep) else observation
    if not isinstance(text, str):
        raise TypeError("An interaction requires a native observation string or NativeStep.")
    messages.extend([
        {"role": "assistant", "content": generation.text},
        {"role": "user", "content": text},
    ])
    return messages
