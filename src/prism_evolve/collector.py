"""Module 3: Same-Prefix Stratified Two-Branch Rollout Collector.

Implements §4.1, §5.3, and §7.3.1 of the PRISM-Evolve design specification:
- Non-oracular split step T selection prior to root action execution.
- SnapshotCapableEnv protocol (save_snapshot / restore_snapshot held strictly by
  the collector and invisible to the policy).
- Stratified two-branch collection (Branch I: root q ~ pi_{theta, I}, continuation ~ pi_theta;
  Branch E: root a ~ pi_{theta, E}, continuation ~ pi_theta).
- Unstratified two-branch collection (both branches ~ pi_theta from s_T) and single-track
  collection for fair baseline comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Any, Mapping, Protocol

from .event_log import CandidateAction, EventRecord, PublicHistory
from .policy import ActionSpacePartition, DualModePolicy
from .utility import CostLedger, TrajectoryOutcome


@dataclass(frozen=True)
class EnvStepResult:
    """Result returned by an environment step."""

    observation: str
    step_cost: float
    done: bool
    native_success: int = 0
    artifact_version: str = "v0"
    metadata: Mapping[str, Any] | None = None


class SnapshotCapableEnv(Protocol):
    """Protocol for interactive environments supporting physical state snapshot save/restore."""

    def reset(self, task_id: str | None = None, seed: int | None = None) -> PublicHistory:
        ...

    def legal_actions(self, history: PublicHistory) -> ActionSpacePartition:
        ...

    def step(self, action: CandidateAction) -> EnvStepResult:
        ...

    def save_snapshot(self) -> Any:
        ...

    def restore_snapshot(self, snapshot: Any) -> None:
        ...


@dataclass(frozen=True)
class StratifiedBranchSample:
    """Collected episode data (stratified 2-branch, unstratified 2-branch, or single-track)."""

    collection_mode: str  # "stratified", "unstratified_2branch", or "single"
    split_step: int
    did_split: bool
    prefix_history: PublicHistory
    root_gate_prob_detached: float
    branch_I: TrajectoryOutcome | None
    branch_E: TrajectoryOutcome | None
    single_trajectory: TrajectoryOutcome | None
    step_partitions_I: tuple[ActionSpacePartition, ...] = ()
    step_partitions_E: tuple[ActionSpacePartition, ...] = ()
    step_partitions_single: tuple[ActionSpacePartition, ...] = ()

    @property
    def weighted_prefix_return(self) -> float:
        """Compute bar_G = p * G_I + (1 - p) * G_E (§4.2)."""
        if self.did_split and self.branch_I is not None and self.branch_E is not None:
            if self.collection_mode == "stratified":
                p = self.root_gate_prob_detached
                return p * self.branch_I.utility + (1.0 - p) * self.branch_E.utility
            return 0.5 * (self.branch_I.utility + self.branch_E.utility)
        if self.single_trajectory is not None:
            return self.single_trajectory.utility
        return 0.0


class BranchCollector:
    """Online trajectory collector supporting stratified 2-branch, unstratified 2-branch, and single-track."""

    def __init__(
        self,
        max_steps: int = 12,
        max_split_step: int = 3,
        fixed_split_step: int | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.max_split_step = max_split_step
        self.fixed_split_step = fixed_split_step

    def select_split_step(self, rng: random.Random) -> int:
        """Select split step T non-oracularly before the episode starts (§4.1)."""
        if self.fixed_split_step is not None:
            return self.fixed_split_step
        if self.max_split_step <= 0:
            return 0
        return rng.randint(0, self.max_split_step)

    def _rollout_from(
        self,
        env: SnapshotCapableEnv,
        policy: DualModePolicy,
        history: PublicHistory,
        prior_partitions: list[ActionSpacePartition],
        rng: random.Random,
        ledger: CostLedger,
        first_step_forced_mode: str | None = None,
        episode_tag: str = "ep",
    ) -> tuple[TrajectoryOutcome, tuple[ActionSpacePartition, ...]]:
        curr_history = history
        partitions = list(prior_partitions)
        first_step = True
        native_success = 0
        terminated_by_submit = False

        while curr_history.step_count < self.max_steps and curr_history.remaining_budget > 0.0:
            t_idx = curr_history.step_count
            partition = env.legal_actions(curr_history)
            partitions.append(partition)

            forced = first_step_forced_mode if first_step else None
            first_step = False

            sampled = policy.act(
                history=curr_history,
                partition=partition,
                rng=rng,
                forced_mode=forced,
                greedy=False,
            )
            step_res = env.step(sampled.action)
            ledger.record_env_step(
                input_tokens=sampled.input_tokens,
                output_tokens=sampled.output_tokens,
                inference_time_sec=sampled.inference_time_sec,
            )

            ev = EventRecord(
                step_index=t_idx,
                action=sampled.action,
                observation=step_res.observation,
                step_cost=step_res.step_cost,
                occurrence_id=f"{episode_tag}-s{t_idx}-{rng.getrandbits(32):08x}",
                artifact_version=step_res.artifact_version,
                input_tokens=sampled.input_tokens,
                output_tokens=sampled.output_tokens,
                metadata=step_res.metadata or {},
            )
            curr_history = curr_history.append(ev)

            if sampled.action.is_terminal_submit:
                terminated_by_submit = True
                native_success = int(step_res.native_success)
                break
            if step_res.done:
                native_success = int(step_res.native_success)
                break

        outcome = TrajectoryOutcome.from_history(
            history=curr_history,
            native_success=native_success,
            terminated_by_submit=terminated_by_submit,
        )
        return outcome, tuple(partitions)

    def collect_episode(
        self,
        env: SnapshotCapableEnv,
        policy: DualModePolicy,
        task_id: str,
        rng: random.Random,
        ledger: CostLedger,
        collection_mode: str = "stratified",
    ) -> StratifiedBranchSample:
        """Collect one training episode under `stratified`, `unstratified_2branch`, or `single` mode."""
        if collection_mode not in ("stratified", "unstratified_2branch", "single"):
            raise ValueError(f"Unsupported collection_mode: {collection_mode!r}")

        ep_seed = rng.randint(0, 2**31 - 1)
        history = env.reset(task_id=task_id, seed=ep_seed)

        if collection_mode == "single":
            outcome, parts = self._rollout_from(
                env=env,
                policy=policy,
                history=history,
                prior_partitions=[],
                rng=rng,
                ledger=ledger,
                first_step_forced_mode=None,
                episode_tag=f"{task_id}-single",
            )
            return StratifiedBranchSample(
                collection_mode="single",
                split_step=0,
                did_split=False,
                prefix_history=history,
                root_gate_prob_detached=0.0,
                branch_I=None,
                branch_E=None,
                single_trajectory=outcome,
                step_partitions_single=parts,
            )

        split_step = self.select_split_step(rng)
        prefix_partitions: list[ActionSpacePartition] = []
        done_early = False
        early_success = 0
        early_submit = False

        # Roll out shared prefix 0 <= t < T
        while history.step_count < split_step and history.step_count < self.max_steps and history.remaining_budget > 0.0:
            t_idx = history.step_count
            partition = env.legal_actions(history)
            prefix_partitions.append(partition)
            sampled = policy.act(
                history=history,
                partition=partition,
                rng=rng,
                forced_mode=None,
                greedy=False,
            )
            step_res = env.step(sampled.action)
            ledger.record_env_step(
                input_tokens=sampled.input_tokens,
                output_tokens=sampled.output_tokens,
                inference_time_sec=sampled.inference_time_sec,
            )
            ev = EventRecord(
                step_index=t_idx,
                action=sampled.action,
                observation=step_res.observation,
                step_cost=step_res.step_cost,
                occurrence_id=f"{task_id}-pre-s{t_idx}-{rng.getrandbits(32):08x}",
                artifact_version=step_res.artifact_version,
                input_tokens=sampled.input_tokens,
                output_tokens=sampled.output_tokens,
                metadata=step_res.metadata or {},
            )
            history = history.append(ev)
            if sampled.action.is_terminal_submit:
                done_early = True
                early_submit = True
                early_success = int(step_res.native_success)
                break
            if step_res.done:
                done_early = True
                early_success = int(step_res.native_success)
                break

        if done_early or history.step_count >= self.max_steps or history.remaining_budget <= 0.0:
            outcome = TrajectoryOutcome.from_history(
                history=history,
                native_success=early_success,
                terminated_by_submit=early_submit,
            )
            return StratifiedBranchSample(
                collection_mode=collection_mode,
                split_step=split_step,
                did_split=False,
                prefix_history=history,
                root_gate_prob_detached=0.0,
                branch_I=None,
                branch_E=None,
                single_trajectory=outcome,
                step_partitions_single=tuple(prefix_partitions),
            )

        # At t = T, check if both I and E are non-empty
        root_partition = env.legal_actions(history)
        if not root_partition.both_modes_available:
            outcome, parts = self._rollout_from(
                env=env,
                policy=policy,
                history=history,
                prior_partitions=prefix_partitions,
                rng=rng,
                ledger=ledger,
                first_step_forced_mode=None,
                episode_tag=f"{task_id}-nosplit",
            )
            return StratifiedBranchSample(
                collection_mode=collection_mode,
                split_step=split_step,
                did_split=False,
                prefix_history=history,
                root_gate_prob_detached=0.0,
                branch_I=None,
                branch_E=None,
                single_trajectory=outcome,
                step_partitions_single=parts,
            )

        # Evaluate root gate probability p = p_{theta_k}(h_T, B_T)
        was_training = policy.training
        policy.eval()
        try:
            root_out = policy.forward_step(history, root_partition)
        finally:
            if was_training:
                policy.train()
        root_p = float(root_out.gate_prob.detach().cpu().item())

        # Save physical snapshot s_T and public history h_T
        t_save0 = float(os.times().elapsed)
        snapshot_s_T = env.save_snapshot()
        ledger.record_snapshot_save(max(0.0, float(os.times().elapsed) - t_save0))
        prefix_h_T = history

        # Independent RNGs for the two branches
        rng_branch_1 = random.Random(rng.randint(0, 2**63 - 1))
        rng_branch_2 = random.Random(rng.randint(0, 2**63 - 1))

        forced_1 = "I" if collection_mode == "stratified" else None
        forced_2 = "E" if collection_mode == "stratified" else None

        # Branch 1 (Information branch I in stratified mode)
        outcome_I, parts_I = self._rollout_from(
            env=env,
            policy=policy,
            history=prefix_h_T,
            prior_partitions=prefix_partitions,
            rng=rng_branch_1,
            ledger=ledger,
            first_step_forced_mode=forced_1,
            episode_tag=f"{task_id}-bI",
        )

        # Restore physical snapshot s_T and run Branch 2 (Execution branch E in stratified mode)
        t_rest0 = float(os.times().elapsed)
        env.restore_snapshot(snapshot_s_T)
        ledger.record_snapshot_restore(max(0.0, float(os.times().elapsed) - t_rest0))

        outcome_E, parts_E = self._rollout_from(
            env=env,
            policy=policy,
            history=prefix_h_T,
            prior_partitions=prefix_partitions,
            rng=rng_branch_2,
            ledger=ledger,
            first_step_forced_mode=forced_2,
            episode_tag=f"{task_id}-bE",
        )

        return StratifiedBranchSample(
            collection_mode=collection_mode,
            split_step=split_step,
            did_split=True,
            prefix_history=prefix_h_T,
            root_gate_prob_detached=root_p,
            branch_I=outcome_I,
            branch_E=outcome_E,
            single_trajectory=None,
            step_partitions_I=parts_I,
            step_partitions_E=parts_E,
        )


class TwoStageToyEnv:
    """Finite two-stage environment from §5.3 for exact rational enumeration and Monte Carlo verification.

    - Hidden state s in {0, 1} with equal prior P(s=0) = P(s=1) = 1/2.
    - Budget B = 1.0.
    - At t = 0:
      - Information action q_probe in I: reveals true state ("state=0" or "state=1"), costs 0.1 * B;
        followed at t = 1 by terminal submit actions {submit_0, submit_1} in E costing 0.2 * B
        (so total cost on branch I is 0.3 * B, giving G_I = 0.7 * Y_I).
      - Execution actions {submit_0, submit_1} in E: directly submit guess at t = 0 without probing,
        costing 0.2 * B (giving G_E = 0.8 * Y_E).
    """

    def __init__(self, budget: float = 1.0) -> None:
        self.budget = budget
        self._hidden_state: int = 0
        self._probed: bool = False
        self._done: bool = False

    def reset(self, task_id: str | None = None, seed: int | None = None, forced_state: int | None = None) -> PublicHistory:
        if forced_state is not None:
            self._hidden_state = int(forced_state)
        else:
            r = random.Random(seed)
            self._hidden_state = r.randint(0, 1)
        self._probed = False
        self._done = False
        return PublicHistory(
            task_id=task_id or "toy_2stage",
            goal="Identify hidden binary state s in {0, 1} and submit matching action.",
            initial_observation="Hidden state s in {0, 1} is unrevealed.",
            total_budget=self.budget,
        )

    def legal_actions(self, history: PublicHistory) -> ActionSpacePartition:
        exec_cands = [
            CandidateAction(
                action_id="submit_0",
                intent="submit",
                tool_name="submit_guess",
                arguments={"guess": 0},
            ),
            CandidateAction(
                action_id="submit_1",
                intent="submit",
                tool_name="submit_guess",
                arguments={"guess": 1},
            ),
        ]
        if history.step_count == 0 and not self._probed:
            info_cands = [
                CandidateAction(
                    action_id="probe_state",
                    intent="probe",
                    tool_name="inspect_state",
                    arguments={},
                )
            ]
            return ActionSpacePartition.from_candidates(info_cands + exec_cands)
        return ActionSpacePartition.from_candidates(exec_cands)

    def step(self, action: CandidateAction) -> EnvStepResult:
        if action.mode == "I":
            self._probed = True
            return EnvStepResult(
                observation=f"Probe result: state={self._hidden_state}",
                step_cost=0.1 * self.budget,
                done=False,
                native_success=0,
                artifact_version="v1",
            )
        guess = int(action.arguments.get("guess", -1))
        success = 1 if guess == self._hidden_state else 0
        self._done = True
        return EnvStepResult(
            observation=f"Submitted guess={guess}, native_success={success}",
            step_cost=0.2 * self.budget,
            done=True,
            native_success=success,
            artifact_version="v2",
        )

    def save_snapshot(self) -> dict[str, Any]:
        return {
            "hidden_state": self._hidden_state,
            "probed": self._probed,
            "done": self._done,
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._hidden_state = int(snapshot["hidden_state"])
        self._probed = bool(snapshot["probed"])
        self._done = bool(snapshot["done"])
