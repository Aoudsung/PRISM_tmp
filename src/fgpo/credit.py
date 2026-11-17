"""History-conditioned peer value sharing for finite-horizon terminal success.

Each value target uses real outcomes from other complete trajectories at the
same decision depth. No estimated value is propagated through another estimate.
The leave-one-trajectory-out baseline is action independent; value sharing is
an approximate, biased estimator whose error depends on history matching.
See docs/fgpo_math_contract.md for the r3 objective and estimator contract.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from .public import HORIZONS
from .protocol import OWN_METHODS as METHODS, RETIRED_METHODS


ALGORITHM = "r3_history_conditioned_peer_value_sharing"


class CrossTrajectoryCredit:
    """One same-task rollout group, with time-indexed independent peer outcomes.

    For peer j at the same depth as x, w_j(x) = [1 - d(x,j)/radius]_+.
    The prior is the mean terminal outcome of all other trajectories, including
    completed peers as task-level evidence, never as synthetic current states.
    b(x) = (sum_j w_j(x) * Y_j + kappa * prior) / (sum_j w_j(x) + kappa).
    For a nonterminal transition, share the next-state peer mean with weight
    lambda_max * support / (support + kappa); use the target's own MC return
    for the remaining weight. Terminal transitions have no continuation term.
    Y denotes reward_scale times official success. Advantages divide by this
    fixed scale, with no random group RMS or hand-written syntax reward.
    """

    def __init__(self, trajectories, reference_data, *, method="fgpo", config=None):
        if method in RETIRED_METHODS:
            raise ValueError(
                f"{method!r} belongs to the retired TF-BEFP/DAG design; "
                "select an explicit r3 ablation instead"
            )
        if method not in METHODS:
            raise ValueError(f"Unknown r3 credit method {method!r}")
        self.method = method
        self.trajectories = tuple(trajectories)
        self.reference = reference_data
        options: Mapping = {} if config is None else config
        self.radius = float(options.get("credit_radius", .10))
        self.prior_weight = float(options.get("credit_prior_weight", 1.))
        self.transfer_max = float(options.get("credit_transfer_max", .5))
        self.reward_scale = float(options.get("reward_scale", 10.))
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("credit_radius must be finite and positive")
        if not math.isfinite(self.prior_weight) or self.prior_weight <= 0:
            raise ValueError("credit_prior_weight must be finite and positive")
        if not math.isfinite(self.transfer_max) or not 0 <= self.transfer_max <= 1:
            raise ValueError("credit_transfer_max must lie in [0, 1]")
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")

        self.count = len(self.trajectories)
        if self.count < 2:
            raise ValueError("Peer credit requires at least two independently sampled trajectories")
        identities = {
            (t.group_id, t.task_id, t.benchmark, t.initial_seed, t.goal)
            for t in self.trajectories
        }
        if len(identities) != 1:
            raise ValueError("Peer trajectories must share group, task, benchmark, initial seed and goal")
        if len({t.trajectory_id for t in self.trajectories}) != self.count:
            raise ValueError("Each peer trajectory must have a distinct trajectory_id")
        self.group_id = self.trajectories[0].group_id
        benchmark = self.trajectories[0].benchmark
        self.benchmark = benchmark
        self.horizon = int(options.get("horizons", HORIZONS)[benchmark])
        if self.horizon <= 0:
            raise ValueError("The official finite horizon must be positive")

        self.rows, times, owners, terminals = [], [], [], []
        self.lengths = [len(t.steps) for t in self.trajectories]
        for owner, trajectory in enumerate(self.trajectories):
            if len(trajectory.steps) > self.horizon:
                raise ValueError("A trajectory exceeds the configured benchmark horizon")
            for index, row in enumerate(trajectory.steps):
                if row.index != index or row.trajectory_id != trajectory.trajectory_id:
                    raise ValueError("Step indices must be contiguous within their owning trajectory")
                terminal = index + 1 == len(trajectory.steps)
                # StepRecord.done is the environment's flag; reaching the
                # official horizon also closes a finite-horizon trajectory.
                if bool(row.done or index + 1 == self.horizon) != terminal:
                    raise ValueError("Credit accepts complete trajectories with one final terminal transition")
                self.rows.append(row)
                times.append(index)
                owners.append(owner)
                terminals.append(terminal)
        if len({row.step_id for row in self.rows}) != len(self.rows):
            raise ValueError("Step identities must be unique within a rollout group")
        self.depth = max(self.lengths, default=0)
        self.times = torch.tensor(times, dtype=torch.long, device="cpu")
        self.owners = torch.tensor(owners, dtype=torch.long, device="cpu")
        self.terminals = torch.tensor(terminals, dtype=torch.bool, device="cpu")
        self.outcomes = torch.tensor(
            [self.reward_scale * float(t.official_success) for t in self.trajectories],
            dtype=torch.float64, device="cpu",
        )
        self.valid = torch.arange(self.depth, device="cpu")[:, None] < torch.tensor(
            self.lengths, dtype=torch.long, device="cpu"
        )[None, :]
        self.prior = (self.outcomes.sum() - self.outcomes) / (self.count - 1)
        self.success_count = sum(1 for t in self.trajectories if t.official_success)
        self.all_zero = self.success_count == 0
        self.all_one = self.success_count == self.count
        self.homogeneous = self.all_zero or self.all_one
        self.diagnostics = {}

    def _features(self, attribute):
        vectors = [getattr(self.reference.steps[row.step_id], attribute) for row in self.rows]
        if any(vector is None for vector in vectors):
            raise ValueError(f"The r3 estimator requires pre-action {attribute} for every actual step")
        # Reference features are normally already on the host. One compact
        # transfer also supports a caller retaining them on an accelerator.
        flat = torch.stack(vectors).detach().to(device="cpu", dtype=torch.float32)
        if flat.ndim != 2 or not bool(torch.isfinite(flat).all()):
            raise ValueError(f"{attribute} must contain finite one-dimensional feature vectors")
        flat = F.normalize(flat, dim=-1)
        features = flat.new_zeros((self.depth, self.count, flat.shape[-1]))
        features[self.times, self.owners] = flat
        return features

    def _alfworld_obs_match(self) -> torch.Tensor:
        codes = torch.zeros((self.depth, self.count), dtype=torch.long, device="cpu")
        vocab: dict[tuple[int, ...], int] = {}
        for owner, trajectory in enumerate(self.trajectories):
            if not trajectory.steps:
                continue
            previous = tuple(trajectory.initial_observation_ids)
            for depth_idx, row in enumerate(trajectory.steps):
                codes[depth_idx, owner] = vocab.setdefault(previous, len(vocab) + 1)
                previous = tuple(row.observation_repr_ids)
        return (codes[:, :, None] == codes[:, None, :]) & (codes[:, :, None] > 0)

    @torch.no_grad()
    def advantages(self):
        matching_mode = "exact_token_id" if self.benchmark == "alfworld" else "dual_view_cosine"
        if not self.rows:
            self.diagnostics = {
                "algorithm": ALGORITHM, "method": self.method,
                "trajectory_count": self.count, "step_count": 0,
                "successes": self.success_count,
                "all_zero_group": self.all_zero,
                "all_one_group": self.all_one,
                "homogeneous_group": self.homogeneous,
                "homogeneous_bypass": self.homogeneous,
                "local_observation_matching": matching_mode,
                "horizon": self.horizon,
                "positive_peer_gated_steps": 0,
                "loo_outcome_brier": None,
                "loo_task_prior_brier": None,
            }
            return {}
        if self.homogeneous:
            count = int(self.valid.sum())
            zero_transfer = self.all_zero or self.method == "fgpo_no_transfer"
            self.diagnostics = {
                "algorithm": ALGORITHM,
                "method": self.method,
                "trajectory_count": self.count,
                "step_count": count,
                "successes": self.success_count,
                "all_zero_group": self.all_zero,
                "all_one_group": self.all_one,
                "homogeneous_group": True,
                "homogeneous_bypass": True,
                "local_observation_matching": matching_mode,
                "horizon": self.horizon,
                "credit_radius": self.radius,
                "credit_prior_weight": self.prior_weight,
                "credit_transfer_max": self.transfer_max,
                "reward_scale": self.reward_scale,
                "matched_steps": None,
                "matched_step_fraction": None,
                "transferred_steps": 0 if zero_transfer else None,
                "transfer_step_fraction": 0.0 if zero_transfer else None,
                "mean_support_mass": None,
                "mean_effective_peers": None,
                "mean_transfer_weight": 0.0 if zero_transfer else None,
                "positive_peer_gated_steps": 0 if self.all_one else None,
                "loo_outcome_brier": 0.0,
                "loo_task_prior_brier": 0.0,
                "matched_loo_outcome_brier": None,
                "mean_abs_return_shift": 0.0,
                "positive_failed_steps": 0,
                "mean_advantage": 0.0,
                "advantage_min": 0.0,
                "advantage_max": 0.0,
            }
            return {row.step_id: 0.0 for row in self.rows}
        if self.reference is None:
            raise ValueError("Mixed-outcome r3 credit requires frozen-reference features")

        # Shapes are [depth, target trajectory, peer trajectory]. This computes
        # O(H*K^2) matches, rather than comparing every pair of temporal nodes.
        eligible = self.valid[:, :, None] & self.valid[:, None, :]
        eligible &= ~torch.eye(self.count, dtype=torch.bool, device="cpu")[None, :, :]
        if self.benchmark == "alfworld":
            obs_match = self._alfworld_obs_match()
            if self.method == "fgpo_obs_only":
                weights = (obs_match & eligible).to(torch.float64)
            else:
                hist = self._features("z_hist")
                hist_dist = (1.0 - torch.bmm(hist, hist.transpose(1, 2))).clamp_(0.0, 2.0)
                del hist
                weights = ((1.0 - hist_dist / self.radius).clamp_min_(0.0) * (obs_match & eligible)).to(torch.float64)
                del hist_dist
            del obs_match
        else:
            obs = self._features("z_obs")
            distance = (1.0 - torch.bmm(obs, obs.transpose(1, 2))).clamp_(0.0, 2.0)
            del obs
            if self.method != "fgpo_obs_only":
                hist = self._features("z_hist")
                history_distance = (1.0 - torch.bmm(hist, hist.transpose(1, 2))).clamp_(0.0, 2.0)
                torch.maximum(distance, history_distance, out=distance)
                del hist, history_distance
            weights = ((1.0 - distance / self.radius).clamp_min_(0.0) * eligible).to(torch.float64)
            del distance

        support = weights.sum(dim=-1)
        weighted_outcomes = torch.matmul(weights, self.outcomes)
        positive_support = support > 0
        denominator = torch.where(positive_support, support, torch.ones_like(support))
        peer_values = weighted_outcomes / denominator
        baseline = (weighted_outcomes + self.prior_weight * self.prior[None, :]) / (
            support + self.prior_weight
        )
        valid_won = self.valid.to(torch.long) * (self.outcomes > 0).to(torch.long)[None, :]
        has_positive_peer = (valid_won.sum(dim=-1, keepdim=True) - valid_won) >= 1
        if self.method == "fgpo_no_transfer":
            state_share = torch.zeros_like(support)
        elif self.method == "fgpo_no_support_weight":
            state_share = (has_positive_peer & positive_support).to(torch.float64)
        else:
            state_share = has_positive_peer.to(torch.float64) * (
                self.transfer_max * support / (support + self.prior_weight)
            )

        next_values = torch.cat((peer_values[1:], torch.zeros_like(peer_values[:1])), dim=0)
        next_share = torch.cat((state_share[1:], torch.zeros_like(state_share[:1])), dim=0)
        next_pos_sup = torch.cat((positive_support[1:], torch.zeros_like(positive_support[:1])), dim=0)
        next_pos_peer = torch.cat((has_positive_peer[1:], torch.zeros_like(has_positive_peer[:1])), dim=0)
        terminal = torch.zeros_like(self.valid)
        terminal[self.times, self.owners] = self.terminals
        continuation = self.valid & ~terminal
        next_share = next_share * continuation
        gated = continuation & next_pos_sup & ~next_pos_peer
        mc_return = self.outcomes[None, :].expand(self.depth, -1)
        reward = mc_return * terminal
        target = (1.0 - next_share) * mc_return + next_share * (reward + next_values)
        advantage = (target - baseline) / self.reward_scale

        # A fixed scale preserves task weighting. The syntax_valid field is a
        # diagnostic only; parsing outcomes do not replace the official reward.
        values = advantage[self.times, self.owners].tolist()
        self._record_diagnostics(
            weights, support, peer_values, baseline, next_share, target, advantage,
            matching_mode=matching_mode, gated_steps=int(gated.sum()),
        )
        return {row.step_id: value for row, value in zip(self.rows, values)}

    def _record_diagnostics(self, weights, support, peer_values, baseline, share, target, advantage,
                            *, matching_mode, gated_steps):
        valid = self.valid
        count = int(valid.sum())
        matched = valid & (support > 0)
        transferred = valid & (share > 0)
        success = (self.outcomes / self.reward_scale)[None, :].expand(self.depth, -1)
        predicted = baseline / self.reward_scale
        prior = (self.prior / self.reward_scale)[None, :].expand_as(success)
        squared_mass = weights.square().sum(dim=-1)
        effective_peers = support.square() / torch.where(
            squared_mass > 0, squared_mass, torch.ones_like(squared_mass)
        )
        # The target trajectory's outcome is held out from prediction. Its
        # Brier error includes Bernoulli outcome noise; it is not a true-Q error
        # measurement, confidence bound, or a weight used to train this target.
        matched_brier = (
            float(((peer_values / self.reward_scale - success).square())[matched].mean())
            if bool(matched.any()) else None
        )
        self.diagnostics = {
            "algorithm": ALGORITHM,
            "method": self.method,
            "trajectory_count": self.count,
            "step_count": count,
            "successes": self.success_count,
            "all_zero_group": self.all_zero,
            "all_one_group": self.all_one,
            "homogeneous_group": False,
            "homogeneous_bypass": False,
            "local_observation_matching": matching_mode,
            "horizon": self.horizon,
            "credit_radius": self.radius,
            "credit_prior_weight": self.prior_weight,
            "credit_transfer_max": self.transfer_max,
            "reward_scale": self.reward_scale,
            "matched_steps": int(matched.sum()),
            "matched_step_fraction": float(matched.sum()) / count,
            "transferred_steps": int(transferred.sum()),
            "transfer_step_fraction": float(transferred.sum()) / count,
            "mean_support_mass": float(support[valid].mean()),
            "mean_effective_peers": float(effective_peers[valid].mean()),
            "mean_transfer_weight": float(share[valid].mean()),
            "positive_peer_gated_steps": gated_steps,
            "loo_outcome_brier": float((predicted - success).square()[valid].mean()),
            "loo_task_prior_brier": float((prior - success).square()[valid].mean()),
            "matched_loo_outcome_brier": matched_brier,
            "mean_abs_return_shift": float((target / self.reward_scale - success).abs()[valid].mean()),
            "positive_failed_steps": int((valid & (success == 0) & (advantage > 0)).sum()),
            "mean_advantage": float(advantage[valid].mean()),
            "advantage_min": float(advantage[valid].min()),
            "advantage_max": float(advantage[valid].max()),
        }


@torch.no_grad()
def advantages(trajectories, reference_data, *, method="fgpo", config=None, diagnostics=None):
    """Return detached step advantages; optionally fill per-group diagnostics.

    Passing retired method IDs is an error even for an empty input. This avoids
    silently interpreting a v1 ablation/checkpoint as a different r3 method.
    """
    if method in RETIRED_METHODS or method not in METHODS:
        raise ValueError(f"Select an explicit r3 credit method; received {method!r}")
    groups = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.group_id, []).append(trajectory)
    result = {}
    for group_id, group in groups.items():
        estimator = CrossTrajectoryCredit(group, reference_data, method=method, config=config)
        group_advantages = estimator.advantages()
        if result.keys() & group_advantages.keys():
            raise ValueError("Step identities must be unique across the entire collected round")
        result.update(group_advantages)
        if diagnostics is not None:
            diagnostics[group_id] = estimator.diagnostics
    return result
