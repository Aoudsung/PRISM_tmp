"""Module 6: Cross-Task Evaluator & Masked Feedback Diagnostic (§2.1, §6.2, §7.2, §8).

Implements:
- Single-track zero-history-residue deployment evaluation on unseen tasks;
- Cross-task self-evolution gain:
      Delta_evo(k, B) = E_{P ~ D_eval}[ U_B(pi_{theta_k}; P) - U_B(pi_{theta_0}; P) ]
  with scenario-grouped paired confidence intervals;
- Dev-set masked feedback diagnostic (§6.2):
      Delta_use(pi) = E[ U_visible(pi) - U_masked(pi) ]
  alongside information action selection rate and post-failure recovery rate
  (strictly separated from the training reward).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .collector import SnapshotCapableEnv
from .event_log import EventRecord, PublicHistory
from .policy import ActionSpacePartition, DualModePolicy
from .utility import CostLedger, TrajectoryOutcome


@dataclass(frozen=True)
class TaskEvalRecord:
    """Per-task single-track evaluation record on a held-out task."""

    task_id: str
    scenario_id: str
    success: int
    total_cost: float
    budget: float
    utility: float
    num_steps: int
    info_steps: int
    exec_steps: int
    had_intermediate_failure: bool
    recovered_from_failure: bool
    cost_ledger: CostLedger

    @property
    def cumulative_cost(self) -> float:
        return self.total_cost


@dataclass(frozen=True)
class PairedGainSummary:
    """Scenario-grouped paired difference summary for Delta_evo(k, B)."""

    stage_k: int
    num_tasks: int
    num_scenarios: int
    mean_success_k: float
    mean_success_0: float
    delta_success: float
    mean_utility_k: float
    mean_utility_0: float
    delta_evo: float
    ci95_halfwidth_utility: float
    ci95_halfwidth_success: float

    @property
    def ci95_low(self) -> float:
        return self.delta_evo - self.ci95_halfwidth_utility

    @property
    def ci95_high(self) -> float:
        return self.delta_evo + self.ci95_halfwidth_utility

    @property
    def success_ci95_low(self) -> float:
        return self.delta_success - self.ci95_halfwidth_success

    @property
    def success_ci95_high(self) -> float:
        return self.delta_success + self.ci95_halfwidth_success


@dataclass(frozen=True)
class MaskedDiagnosticSummary:
    """Summary of §6.2 masked feedback diagnostic on dev tasks."""

    num_evaluated_prefixes: int
    mean_visible_utility: float
    mean_masked_utility: float
    delta_use: float
    ci95_halfwidth_delta_use: float
    info_selection_rate: float
    recovery_success_rate: float


class CrossTaskEvaluator:
    """Evaluates frozen policy checkpoints on unseen tasks without history residue."""

    def __init__(
        self,
        env: SnapshotCapableEnv,
        max_steps: int = 20,
        greedy: bool = True,
    ) -> None:
        self.env = env
        self.max_steps = int(max_steps)
        self.greedy = bool(greedy)

    @torch.no_grad()
    def evaluate_task(
        self,
        policy: DualModePolicy,
        task_id: str,
        scenario_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> TaskEvalRecord:
        """Runs a single-track deployment trajectory with fresh intra-task history."""
        rng = random.Random(seed if seed is not None else 0)
        ledger = CostLedger()
        # Intra-task history is strictly reset on every task (§1.1, §2.1)
        history = self.env.reset(task_id=task_id, seed=seed)

        info_steps = 0
        exec_steps = 0
        had_failure = False
        native_success = 0
        terminated_by_submit = False

        while history.step_count < self.max_steps and history.remaining_budget > 0.0:
            t_idx = history.step_count
            partition = self.env.legal_actions(history)
            sampled = policy.act(
                history=history,
                partition=partition,
                rng=rng,
                forced_mode=None,
                greedy=self.greedy,
            )
            if sampled.action.mode == "I":
                info_steps += 1
            else:
                exec_steps += 1

            step_res = self.env.step(sampled.action)
            ledger.record_env_step(
                input_tokens=sampled.input_tokens,
                output_tokens=sampled.output_tokens,
                inference_time_sec=sampled.inference_time_sec,
            )

            meta = step_res.metadata or {}
            if (
                bool(meta.get("step_failed", False))
                or "ERROR" in step_res.observation
                or "Exception" in step_res.observation
            ):
                had_failure = True

            ev = EventRecord(
                step_index=t_idx,
                action=sampled.action,
                observation=step_res.observation,
                step_cost=step_res.step_cost,
                occurrence_id=f"eval:{task_id}:s{t_idx}",
                artifact_version=step_res.artifact_version,
                input_tokens=sampled.input_tokens,
                output_tokens=sampled.output_tokens,
                metadata=meta,
            )
            history = history.append(ev)

            if sampled.action.is_terminal_submit:
                terminated_by_submit = True
                native_success = int(step_res.native_success)
                break
            if step_res.done:
                native_success = int(step_res.native_success)
                break

        outcome = TrajectoryOutcome.from_history(
            history=history,
            native_success=native_success,
            terminated_by_submit=terminated_by_submit,
        )
        recovered = bool(had_failure and outcome.terminal_success == 1)
        scen_id = (
            scenario_id
            if scenario_id is not None
            else (history.scenario_id or task_id.split("__")[0])
        )

        return TaskEvalRecord(
            task_id=task_id,
            scenario_id=scen_id,
            success=outcome.terminal_success,
            total_cost=outcome.total_cost,
            budget=history.total_budget,
            utility=outcome.utility,
            num_steps=history.step_count,
            info_steps=info_steps,
            exec_steps=exec_steps,
            had_intermediate_failure=had_failure,
            recovered_from_failure=recovered,
            cost_ledger=ledger,
        )

    @torch.no_grad()
    def evaluate_suite(
        self,
        policy: DualModePolicy,
        task_specs: Sequence[Tuple[str, str]],
        base_seed: int = 0,
    ) -> List[TaskEvalRecord]:
        """Evaluates a sequence of (task_id, scenario_id) pairs."""
        records: List[TaskEvalRecord] = []
        for idx, (task_id, scenario_id) in enumerate(task_specs):
            rec = self.evaluate_task(
                policy=policy,
                task_id=task_id,
                scenario_id=scenario_id,
                seed=base_seed + idx,
            )
            records.append(rec)
        return records

    evaluate_split = evaluate_suite

    @staticmethod
    def compute_paired_evolution_gain(
        records_k: Sequence[TaskEvalRecord],
        records_0: Sequence[TaskEvalRecord],
        stage_k: int,
    ) -> PairedGainSummary:
        """Computes Delta_evo(k, B) with scenario-grouped paired 95% CIs (§2.1, §7.2)."""
        if len(records_k) != len(records_0):
            raise ValueError("Paired evaluation requires matching task lists.")
        n_tasks = len(records_k)
        if n_tasks == 0:
            raise ValueError("Task record list cannot be empty.")

        scenario_util_diffs: Dict[str, List[float]] = {}
        scenario_succ_diffs: Dict[str, List[float]] = {}

        for rk, r0 in zip(records_k, records_0):
            if rk.task_id != r0.task_id:
                raise ValueError(f"Mismatched task order: {rk.task_id} vs {r0.task_id}")
            scen = rk.scenario_id
            scenario_util_diffs.setdefault(scen, []).append(rk.utility - r0.utility)
            scenario_succ_diffs.setdefault(scen, []).append(float(rk.success - r0.success))

        scen_means_util = [sum(v) / len(v) for v in scenario_util_diffs.values()]
        scen_means_succ = [sum(v) / len(v) for v in scenario_succ_diffs.values()]
        m = len(scen_means_util)

        mean_uk = sum(r.utility for r in records_k) / n_tasks
        mean_u0 = sum(r.utility for r in records_0) / n_tasks
        delta_evo = mean_uk - mean_u0

        mean_sk = sum(r.success for r in records_k) / n_tasks
        mean_s0 = sum(r.success for r in records_0) / n_tasks
        delta_succ = mean_sk - mean_s0

        if m > 1:
            mu_u = sum(scen_means_util) / m
            var_u = sum((x - mu_u) ** 2 for x in scen_means_util) / (m - 1)
            ci_u = 1.96 * math.sqrt(var_u / m)

            mu_s = sum(scen_means_succ) / m
            var_s = sum((x - mu_s) ** 2 for x in scen_means_succ) / (m - 1)
            ci_s = 1.96 * math.sqrt(var_s / m)
        else:
            ci_u = 0.0
            ci_s = 0.0

        return PairedGainSummary(
            stage_k=stage_k,
            num_tasks=n_tasks,
            num_scenarios=m,
            mean_success_k=mean_sk,
            mean_success_0=mean_s0,
            delta_success=delta_succ,
            mean_utility_k=mean_uk,
            mean_utility_0=mean_u0,
            delta_evo=delta_evo,
            ci95_halfwidth_utility=ci_u,
            ci95_halfwidth_success=ci_s,
        )


class MaskedFeedbackDiagnostic:
    """Dev-set masked feedback diagnostic Delta_use(pi) (§6.2).

    Evaluates whether the policy genuinely uses the observation returned by an
    information action q_t in I to adjust subsequent execution and improve
    terminal utility, rather than merely issuing dummy probes.
    """

    def __init__(
        self,
        env: SnapshotCapableEnv,
        max_steps: int = 20,
        greedy: bool = True,
    ) -> None:
        self.env = env
        self.max_steps = int(max_steps)
        self.greedy = bool(greedy)

    @torch.no_grad()
    def run_diagnostic(
        self,
        policy: DualModePolicy,
        task_specs: Sequence[Tuple[str, str]],
        base_seed: int = 1000,
    ) -> MaskedDiagnosticSummary:
        """Computes Delta_use(pi), info selection rate, and recovery rate on dev tasks."""
        visible_utilities: List[float] = []
        masked_utilities: List[float] = []
        diffs: List[float] = []

        total_steps = 0
        total_info_steps = 0
        tasks_with_failure = 0
        tasks_recovered = 0

        for idx, (task_id, _) in enumerate(task_specs):
            seed = base_seed + idx
            rng_vis = random.Random(seed)
            history = self.env.reset(task_id=task_id, seed=seed)

            had_failure = False
            native_success = 0
            terminated_by_submit = False
            forked_info_step: Optional[int] = None
            saved_post_info_snapshot: Any = None
            post_info_history: Optional[PublicHistory] = None

            while history.step_count < self.max_steps and history.remaining_budget > 0.0:
                t_idx = history.step_count
                partition = self.env.legal_actions(history)
                sampled = policy.act(
                    history=history,
                    partition=partition,
                    rng=rng_vis,
                    forced_mode=None,
                    greedy=self.greedy,
                )

                total_steps += 1
                if sampled.action.mode == "I":
                    total_info_steps += 1

                step_res = self.env.step(sampled.action)
                meta = step_res.metadata or {}
                if (
                    bool(meta.get("step_failed", False))
                    or "ERROR" in step_res.observation
                    or "Exception" in step_res.observation
                ):
                    had_failure = True

                ev = EventRecord(
                    step_index=t_idx,
                    action=sampled.action,
                    observation=step_res.observation,
                    step_cost=step_res.step_cost,
                    occurrence_id=f"diag:{task_id}:s{t_idx}",
                    artifact_version=step_res.artifact_version,
                    input_tokens=sampled.input_tokens,
                    output_tokens=sampled.output_tokens,
                    metadata=meta,
                )
                history = history.append(ev)

                if sampled.action.is_terminal_submit:
                    terminated_by_submit = True
                    native_success = int(step_res.native_success)
                    break
                if step_res.done:
                    native_success = int(step_res.native_success)
                    break

                # Capture the first information action step for counterfactual masking
                if sampled.action.mode == "I" and forked_info_step is None:
                    forked_info_step = t_idx
                    saved_post_info_snapshot = self.env.save_snapshot()
                    post_info_history = history

            vis_outcome = TrajectoryOutcome.from_history(
                history=history,
                native_success=native_success,
                terminated_by_submit=terminated_by_submit,
            )

            if had_failure:
                tasks_with_failure += 1
                if vis_outcome.terminal_success == 1:
                    tasks_recovered += 1

            # Counterfactual continuation with masked observation at forked_info_step
            if (
                forked_info_step is not None
                and saved_post_info_snapshot is not None
                and post_info_history is not None
            ):
                self.env.restore_snapshot(saved_post_info_snapshot)
                masked_history = post_info_history.with_masked_step(forked_info_step)
                rng_mask = random.Random(seed + 99999)
                m_native_success = 0
                m_submit = False

                while (
                    masked_history.step_count < self.max_steps
                    and masked_history.remaining_budget > 0.0
                ):
                    m_idx = masked_history.step_count
                    partition = self.env.legal_actions(masked_history)
                    m_sampled = policy.act(
                        history=masked_history,
                        partition=partition,
                        rng=rng_mask,
                        forced_mode=None,
                        greedy=self.greedy,
                    )
                    m_res = self.env.step(m_sampled.action)
                    m_ev = EventRecord(
                        step_index=m_idx,
                        action=m_sampled.action,
                        observation=m_res.observation,
                        step_cost=m_res.step_cost,
                        occurrence_id=f"diag_mask:{task_id}:s{m_idx}",
                        artifact_version=m_res.artifact_version,
                        input_tokens=m_sampled.input_tokens,
                        output_tokens=m_sampled.output_tokens,
                        metadata=m_res.metadata or {},
                    )
                    masked_history = masked_history.append(m_ev)
                    if m_sampled.action.is_terminal_submit:
                        m_submit = True
                        m_native_success = int(m_res.native_success)
                        break
                    if m_res.done:
                        m_native_success = int(m_res.native_success)
                        break

                mask_outcome = TrajectoryOutcome.from_history(
                    history=masked_history,
                    native_success=m_native_success,
                    terminated_by_submit=m_submit,
                )
                visible_utilities.append(vis_outcome.utility)
                masked_utilities.append(mask_outcome.utility)
                diffs.append(vis_outcome.utility - mask_outcome.utility)
            else:
                # When policy took no info action on task, visible and masked trajectories are identical
                visible_utilities.append(vis_outcome.utility)
                masked_utilities.append(vis_outcome.utility)
                diffs.append(0.0)

        n_pref = len(diffs)
        if n_pref > 0:
            mean_vis = sum(visible_utilities) / n_pref
            mean_mask = sum(masked_utilities) / n_pref
            delta_use = mean_vis - mean_mask
            if n_pref > 1:
                var_d = sum((d - delta_use) ** 2 for d in diffs) / (n_pref - 1)
                ci_d = 1.96 * math.sqrt(var_d / n_pref)
            else:
                ci_d = 0.0
        else:
            mean_vis = 0.0
            mean_mask = 0.0
            delta_use = 0.0
            ci_d = 0.0

        info_rate = (total_info_steps / total_steps) if total_steps > 0 else 0.0
        rec_rate = (tasks_recovered / tasks_with_failure) if tasks_with_failure > 0 else 0.0

        return MaskedDiagnosticSummary(
            num_evaluated_prefixes=n_pref,
            mean_visible_utility=mean_vis,
            mean_masked_utility=mean_mask,
            delta_use=delta_use,
            ci95_halfwidth_delta_use=ci_d,
            info_selection_rate=info_rate,
            recovery_success_rate=rec_rate,
        )

    evaluate_information_use = run_diagnostic
