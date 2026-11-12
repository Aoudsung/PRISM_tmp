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
import uuid
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


@dataclass
class ModelCall:
    """One physical inference attempt; unknown usage is never represented as zero."""
    call_id: str
    role: str
    model: dict[str, Any]
    prompt_version: str
    charged_round_id: str
    started_at: float
    finished_at: float | None = None
    status: str = "STARTED"
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    usage_complete: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    generation_started: bool = False
    generation_started_at: float | None = None
    lease_status: str = "WITHIN_LEASE"
    error: str | None = None
    prompt_messages: list[dict[str, Any]] = field(default_factory=list)
    prompt_token_ids: list[int] = field(default_factory=list)
    response_token_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ActionExecution:
    origin: str
    action: dict[str, Any] | None
    model_call_id: str | None
    program_version: str | None = None
    node_version: str | None = None
    cost_event_ids: tuple[str, ...] = ()


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
        charged_round_id: str | None = None,
    ) -> None:
        self.limits = limits
        self.parent = parent
        self.event_path = Path(event_path) if event_path is not None else None
        self.started = monotonic_seconds()
        self.charged_round_id = charged_round_id or (parent.charged_round_id if parent else uuid.uuid4().hex)
        if parent is not None and self.charged_round_id != parent.charged_round_id:
            raise ValueError("A shared physical ledger must have one charged round")
        self.model_calls: list[ModelCall] = []
        self.cost_events: list[dict[str, Any]] = []
        self.usage_complete = True
        self.exploration_stopped = False
        self.last_cost_event_id: str | None = None
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
        record = dict(record)
        record.setdefault("charged_round_id", self.charged_round_id)
        if record.get("kind") not in {"call_started", "call_finished"}:
            record.setdefault("cost_event_id", uuid.uuid4().hex)
            self.last_cost_event_id = record["cost_event_id"]
            record.setdefault("status", "OK")
            self.cost_events.append(dict(record))
        if self.event_path is not None:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n")

    def start_model_call(self, role: str, model: Mapping[str, Any], prompt_version: str,
                         messages: Sequence[Mapping[str, Any]] = ()) -> ModelCall:
        record = ModelCall(call_id=uuid.uuid4().hex, role=role, model=dict(model),
                           prompt_version=prompt_version, charged_round_id=self.charged_round_id,
                           started_at=monotonic_seconds(), prompt_messages=[dict(message) for message in messages])
        self.model_calls.append(record)
        writer = self
        while writer.event_path is None and writer.parent is not None:
            writer = writer.parent
        writer._append({"kind": "call_started", **asdict(record)})
        return record

    def finish_model_call(self, record: ModelCall) -> str:
        if record.finished_at is not None:
            raise ValueError("A physical model call can only finish once")
        record.finished_at = monotonic_seconds()
        elapsed = max(0.0, record.finished_at - record.started_at)
        record.timings["total_wall"] = elapsed
        if record.generation_started_at is not None and not record.usage_complete:
            record.timings["generate"] = max(0.0, record.finished_at - record.generation_started_at)
        cursor = self
        deadline = self.started + self.limits.max_wall_seconds
        while cursor.parent is not None:
            cursor = cursor.parent
            deadline = min(deadline, cursor.started + cursor.limits.max_wall_seconds)
        if record.finished_at > deadline:
            record.lease_status = "LEASE_OVERRUN"
            self.exploration_stopped = True
        self.usage_complete = self.usage_complete and record.usage_complete
        self.exploration_stopped = self.exploration_stopped or not record.usage_complete
        writer = self
        while writer.event_path is None and writer.parent is not None:
            writer = writer.parent
        writer._append({"kind": "call_finished", **asdict(record)})
        cost_id = uuid.uuid4().hex
        self.record_model_call(record.input_tokens or 0, record.output_tokens or 0, elapsed,
                               record.role, call_id=record.call_id, cost_event_id=cost_id,
                               usage_complete=record.usage_complete, status=record.status,
                               lease_status=record.lease_status)
        return cost_id

    def record_model_call(
        self,
        input_tokens: int,
        output_tokens: int,
        elapsed_seconds: float,
        role: str = "actor",
        forward_tokens: int = 0,
        *, call_id: str | None = None, cost_event_id: str | None = None,
        usage_complete: bool = True, status: str = "OK", lease_status: str = "WITHIN_LEASE",
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
        self.usage_complete = self.usage_complete and usage_complete
        self.exploration_stopped = self.exploration_stopped or not usage_complete or lease_status == "LEASE_OVERRUN"
        cost_id = cost_event_id or uuid.uuid4().hex
        self._append({"kind": "model_call", "role": role, "call_id": call_id,
                      "cost_event_id": cost_id, "charged_round_id": self.charged_round_id,
                      "input_tokens": input_tokens if usage_complete else None,
                      "output_tokens": output_tokens if usage_complete else None,
                      "known_input_tokens": input_tokens, "known_output_tokens": output_tokens,
                      "forward_tokens": forward_tokens, "usage_complete": usage_complete,
                      "status": status, "lease_status": lease_status,
                      "elapsed_seconds": elapsed_seconds})
        if self.parent is not None:
            self.parent.record_model_call(input_tokens, output_tokens, elapsed_seconds, role, forward_tokens,
                                          call_id=call_id, cost_event_id=cost_id, usage_complete=usage_complete,
                                          status=status, lease_status=lease_status)

    def record_env_step(self, elapsed_seconds: float = 0.0, *, replay: bool = False,
                        cost_event_id: str | None = None) -> None:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Measured environment time must be finite and nonnegative.")
        self.env_steps += 1
        self.replayed_env_steps += int(replay)
        self.environment_seconds += float(elapsed_seconds)
        cost_id = cost_event_id or uuid.uuid4().hex
        self._append({"kind": "environment_step", "replay": replay, "elapsed_seconds": elapsed_seconds,
                      "cost_event_id": cost_id})
        if self.parent is not None:
            self.parent.record_env_step(elapsed_seconds, replay=replay, cost_event_id=cost_id)

    def record_restore(self, elapsed_seconds: float, replay_steps: int = 0, *,
                       cost_event_id: str | None = None) -> None:
        if replay_steps < 0 or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Invalid restore accounting.")
        self.restore_seconds += float(elapsed_seconds)
        self.snapshot_restores += 1
        for _ in range(replay_steps):
            self.record_env_step(replay=True)
        cost_id = cost_event_id or uuid.uuid4().hex
        self._append({"kind": "snapshot_restore", "elapsed_seconds": elapsed_seconds,
                      "replayed_env_steps": replay_steps, "cost_event_id": cost_id})
        if self.parent is not None:
            self.parent.record_restore(elapsed_seconds, replay_steps=0, cost_event_id=cost_id)

    def record_api_calls(self, count: int, *, successful: int = 0, replay: bool = False,
                         cost_event_id: str | None = None) -> None:
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
        cost_id = cost_event_id or uuid.uuid4().hex
        self._append({"kind": "native_api_calls", "attempted": int(count),
                      "successful": int(successful), "replay": bool(replay), "cost_event_id": cost_id})
        if self.parent is not None:
            self.parent.record_api_calls(count, successful=successful, replay=replay, cost_event_id=cost_id)

    def record_snapshot_save(self, elapsed_seconds: float, *, cost_event_id: str | None = None) -> None:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Invalid snapshot-save time.")
        self.snapshot_saves += 1
        self.snapshot_save_seconds += float(elapsed_seconds)
        cost_id = cost_event_id or uuid.uuid4().hex
        self._append({"kind": "snapshot_save", "elapsed_seconds": elapsed_seconds, "cost_event_id": cost_id})
        if self.parent is not None:
            self.parent.record_snapshot_save(elapsed_seconds, cost_event_id=cost_id)

    def record_update(self, elapsed_seconds: float, *, cost_event_id: str | None = None) -> None:
        self.gradient_steps += 1
        self.update_seconds += float(elapsed_seconds)
        cost_id = cost_event_id or uuid.uuid4().hex
        self._append({"kind": "gradient_update", "elapsed_seconds": elapsed_seconds, "cost_event_id": cost_id})
        if self.parent is not None:
            self.parent.record_update(elapsed_seconds, cost_event_id=cost_id)

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
                "wall_seconds": self.elapsed_seconds, "calls_by_role": dict(self.calls_by_role),
                "charged_round_id": self.charged_round_id, "usage_complete": self.usage_complete,
                "exploration_stopped": self.exploration_stopped}


@dataclass(frozen=True)
class Resources:
    wall_seconds: float
    model_tokens: int
    env_steps: int

    def __post_init__(self) -> None:
        if (not math.isfinite(self.wall_seconds) or self.wall_seconds < 0
                or type(self.model_tokens) is not int or type(self.env_steps) is not int
                or self.model_tokens < 0 or self.env_steps < 0):
            raise ValueError("Resources must be finite and nonnegative")

    def __add__(self, other: Resources) -> Resources:
        return Resources(self.wall_seconds + other.wall_seconds,
                         self.model_tokens + other.model_tokens, self.env_steps + other.env_steps)

    def fits_fraction(self, other: Resources, fraction: float) -> bool:
        return (self.wall_seconds <= fraction * other.wall_seconds
                and self.model_tokens <= fraction * other.model_tokens
                and self.env_steps <= fraction * other.env_steps)


class ExplorationLedger:
    """Contract §4/5.8: one rho, independent dimensions, prepaid joint leases.

    P contains only actual paid base work. Global reservations and per-round
    reservations are simultaneous constraints; a later cheaper execution never
    retroactively changes a previously denied reservation.
    """
    def __init__(self, *, rho: float = 0.2, event_path: Path | None = None) -> None:
        if not 0 < rho < 1:
            raise ValueError("rho must be in (0,1)")
        self.rho = rho
        self.event_path = event_path
        self.paid = Resources(0, 0, 0)
        self.spent = Resources(0, 0, 0)
        self.rounds: dict[str, dict[str, Any]] = {}
        self.leases: dict[str, dict[str, Any]] = {}
        self.paid_episode_ids: set[str] = set()
        self.stopped = False

    def _event(self, kind: str, **payload: Any) -> None:
        if self.event_path is not None:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": kind, "time": monotonic_seconds(), **payload},
                                        ensure_ascii=False, allow_nan=False) + "\n")

    def credit_paid_base(self, episode_id: str, actual: Resources, *, usage_complete: bool) -> None:
        if episode_id in self.paid_episode_ids or not usage_complete:
            raise ValueError("Startup credit requires a unique, completely billed base episode")
        self.paid_episode_ids.add(episode_id)
        self.paid = self.paid + actual
        self._event("paid_base_credit", episode_id=episode_id, actual=asdict(actual), paid=asdict(self.paid))

    def open_round(self, round_id: str, budget: Resources) -> None:
        if round_id in self.rounds:
            raise ValueError("A charged round can only be preregistered once")
        self.rounds[round_id] = {"budget": budget, "spent": Resources(0, 0, 0)}
        self._event("round_preregistered", charged_round_id=round_id, budget=asdict(budget))

    def _reserved(self, round_id: str | None = None) -> Resources:
        total = Resources(0, 0, 0)
        for lease in self.leases.values():
            if lease["status"] == "RESERVED" and (round_id is None or lease["round_id"] == round_id):
                total = total + lease["resources"]
        return total

    def reserve_compile_and_trial(self, case_id: str, round_id: str,
                                  compile_lease: Resources, trial_lease: Resources) -> tuple[str, str] | None:
        if self.stopped or round_id not in self.rounds:
            raise ValueError("No exploration is permitted in this round")
        if min(trial_lease.wall_seconds, trial_lease.model_tokens, trial_lease.env_steps) <= 0:
            raise ValueError("The complete first trial needs a nonzero lease in every resource dimension")
        joint = compile_lease + trial_lease
        round_state = self.rounds[round_id]
        global_ok = (self.spent + self._reserved() + joint).fits_fraction(self.paid, self.rho / (1 - self.rho))
        round_ok = (round_state["spent"] + self._reserved(round_id) + joint).fits_fraction(round_state["budget"], self.rho)
        self._event("joint_lease_check", case_id=case_id, charged_round_id=round_id,
                    compile_lease=asdict(compile_lease), trial_lease=asdict(trial_lease),
                    paid=asdict(self.paid), spent=asdict(self.spent), reserved=asdict(self._reserved()),
                    global_ok=global_ok, round_ok=round_ok)
        if not global_ok or not round_ok:
            return None
        ids = []
        for role, resources in (("compile", compile_lease), ("trial", trial_lease)):
            lease_id = uuid.uuid4().hex
            self.leases[lease_id] = {"case_id": case_id, "round_id": round_id,
                                     "role": role, "resources": resources, "status": "RESERVED"}
            self._event("lease_reserved", lease_id=lease_id, case_id=case_id,
                        charged_round_id=round_id, role=role, resources=asdict(resources))
            ids.append(lease_id)
        return ids[0], ids[1]

    def settle(self, lease_id: str, actual: Resources, *, usage_complete: bool) -> None:
        lease = self.leases[lease_id]
        if lease["status"] != "RESERVED":
            raise ValueError("A lease can only be settled once")
        overrun = not actual.fits_fraction(lease["resources"], 1)
        lease["status"] = "LEASE_OVERRUN" if overrun else "SETTLED"
        lease["actual"] = actual
        self.spent = self.spent + actual
        round_state = self.rounds[lease["round_id"]]
        round_state["spent"] = round_state["spent"] + actual
        self.stopped = self.stopped or overrun or not usage_complete
        self._event("lease_settled", lease_id=lease_id, charged_round_id=lease["round_id"],
                    actual=asdict(actual), usage_complete=usage_complete, status=lease["status"],
                    exploration_stopped=self.stopped)


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
