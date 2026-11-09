"""Module 4: Unified Task Utility and 6-Dimensional Full Cost Ledger.

Implements §3.1 and §7.3.3 of the PRISM-Evolve design specification:
- Area Under Success-Over-Time utility:
    U_B(tau) = Y(tau) * max(0.0, 1.0 - C(tau) / B)
  where Y(tau) in {0, 1} is the native environment terminal success indicator,
  C(tau) >= 0 is cumulative cost up to formal submission, and B > 0 is budget.
- Six-dimensional CostLedger tracking:
  (1) environment interaction steps,
  (2) input/output tokens,
  (3) model inference time/compute,
  (4) gradient update time/compute,
  (5) environment snapshot save/restore overhead,
  (6) end-to-end wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .event_log import PublicHistory


def compute_utility(native_success: int | bool, cumulative_cost: float, budget: float) -> float:
    """Compute U_B(tau) = Y(tau) * (1 - C(tau) / B)_+ in [0, 1] (§3.1)."""
    if budget <= 0.0:
        raise ValueError(f"Task budget B must be > 0, got {budget}")
    if cumulative_cost < 0.0:
        raise ValueError(f"cumulative_cost C(tau) must be >= 0, got {cumulative_cost}")
    y = 1 if bool(native_success) else 0
    if y == 0:
        return 0.0
    slack = max(0.0, 1.0 - (float(cumulative_cost) / float(budget)))
    return float(y) * slack


@dataclass(frozen=True)
class TrajectoryOutcome:
    """Complete trajectory record tau with native terminal judgment Y(tau) and utility U_B(tau)."""

    history: PublicHistory
    native_success: int
    total_cost: float
    budget: float
    utility: float
    terminated_by_submit: bool

    @classmethod
    def from_history(
        cls,
        history: PublicHistory,
        native_success: int | bool,
        terminated_by_submit: bool | None = None,
        override_cost: float | None = None,
    ) -> TrajectoryOutcome:
        if terminated_by_submit is None:
            terminated_by_submit = bool(
                history.events and history.events[-1].action.is_terminal_submit
            )
        # Per §3.1: Y(tau) can be 1 only when formal submit is called and verified by native evaluator
        y = 1 if (bool(native_success) and terminated_by_submit) else 0
        c = float(history.elapsed_cost if override_cost is None else override_cost)
        u = compute_utility(y, c, history.total_budget)
        return cls(
            history=history,
            native_success=y,
            total_cost=c,
            budget=history.total_budget,
            utility=u,
            terminated_by_submit=bool(terminated_by_submit),
        )

    @property
    def terminal_success(self) -> int:
        return self.native_success

    @property
    def cumulative_cost(self) -> float:
        return self.total_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.history.task_id,
            "scenario_id": self.history.scenario_id,
            "native_success": self.native_success,
            "total_cost": self.total_cost,
            "budget": self.budget,
            "utility": self.utility,
            "terminated_by_submit": self.terminated_by_submit,
            "step_count": self.history.step_count,
            "information_steps": sum(1 for ev in self.history.events if ev.mode == "I"),
            "execution_steps": sum(1 for ev in self.history.events if ev.mode == "E"),
        }


@dataclass
class CostLedger:
    """Six-dimensional full cost ledger (§7.3.3)."""

    env_steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    inference_calls: int = 0
    inference_time_sec: float = 0.0
    gradient_steps: int = 0
    gradient_update_time_sec: float = 0.0
    snapshot_saves: int = 0
    snapshot_restores: int = 0
    snapshot_restore_time_sec: float = 0.0
    wall_clock_time_sec: float = 0.0
    _wall_start: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def start_wall_clock(self) -> None:
        if self._wall_start is None:
            self._wall_start = float(os.times().elapsed)

    def stop_wall_clock(self) -> None:
        if self._wall_start is not None:
            self.wall_clock_time_sec += max(0.0, float(os.times().elapsed) - self._wall_start)
            self._wall_start = None

    def record_env_step(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        inference_time_sec: float = 0.0,
    ) -> None:
        self.env_steps += 1
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        if inference_time_sec > 0.0 or input_tokens > 0 or output_tokens > 0:
            self.inference_calls += 1
            self.inference_time_sec += float(inference_time_sec)

    def record_inference(
        self,
        input_tokens: int,
        output_tokens: int,
        elapsed_sec: float,
    ) -> None:
        self.inference_calls += 1
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        self.inference_time_sec += max(0.0, float(elapsed_sec))

    def record_gradient_update(self, elapsed_sec: float, steps: int = 1) -> None:
        self.gradient_steps += int(steps)
        self.gradient_update_time_sec += max(0.0, float(elapsed_sec))

    def record_snapshot_save(self, elapsed_sec: float) -> None:
        self.snapshot_saves += 1
        self.snapshot_restore_time_sec += max(0.0, float(elapsed_sec))

    def record_snapshot_restore(self, elapsed_sec: float) -> None:
        self.snapshot_restores += 1
        self.snapshot_restore_time_sec += max(0.0, float(elapsed_sec))

    def merge(self, other: CostLedger) -> None:
        self.env_steps += other.env_steps
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.inference_calls += other.inference_calls
        self.inference_time_sec += other.inference_time_sec
        self.gradient_steps += other.gradient_steps
        self.gradient_update_time_sec += other.gradient_update_time_sec
        self.snapshot_saves += other.snapshot_saves
        self.snapshot_restores += other.snapshot_restores
        self.snapshot_restore_time_sec += other.snapshot_restore_time_sec
        self.wall_clock_time_sec += other.wall_clock_time_sec

    def copy_snapshot(self) -> dict[str, Any]:
        extra_wall = (
            max(0.0, float(os.times().elapsed) - self._wall_start)
            if self._wall_start is not None
            else 0.0
        )
        inf_sec = round(self.inference_time_sec, 6)
        grad_sec = round(self.gradient_update_time_sec, 6)
        snap_sec = round(self.snapshot_restore_time_sec, 6)
        wall_sec = round(self.wall_clock_time_sec + extra_wall, 6)
        return {
            "env_steps": self.env_steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "inference_calls": self.inference_calls,
            "inference_time_sec": inf_sec,
            "model_inference_sec": inf_sec,
            "gradient_steps": self.gradient_steps,
            "gradient_update_time_sec": grad_sec,
            "gradient_update_sec": grad_sec,
            "snapshot_saves": self.snapshot_saves,
            "snapshot_restores": self.snapshot_restores,
            "snapshot_restore_time_sec": snap_sec,
            "snapshot_restore_sec": snap_sec,
            "wall_clock_time_sec": wall_sec,
            "wall_clock_sec": wall_sec,
        }
