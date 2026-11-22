"""History-conditioned peer value sharing for finite-horizon terminal success.

Each value target uses real outcomes from other complete trajectories. V4
matches at the same depth; the exact first-hit variant allows adjacent depths.
No estimated value is propagated through another estimate.
The leave-one-trajectory-out baseline is action independent; value sharing is
an approximate, biased estimator whose error depends on history matching.
See docs/fgpo_math_contract.md for the r3 objective and estimator contract.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from .public import EXACT_OBSERVATION_BENCHMARKS, HORIZONS
from .protocol import EXACT_FIRST_HIT_METHOD, OWN_METHODS as METHODS, RETIRED_METHODS


ALGORITHM = "r5_symmetric_peer_value_sharing"


class CrossTrajectoryCredit:
    """One same-task rollout group, with time-indexed independent peer outcomes.

    For peer j at the same depth as x, w_j(x) = [1 - d(x,j)/radius]_+.
    The prior is the mean terminal outcome of all other trajectories, including
    completed peers as task-level evidence, never as synthetic current states.
    The r5 estimator pairs symmetric pre/post peer potential contraction with
    smoothed pending-peer scaling omega = (K - 1 - C^+_{-k,t} + kappa) / (K - 1 + kappa),
    preserving leave-one-out action independence and potential telescoping for any
    lambda_max in [0, 1] without positive-peer continuation gating.
    Y denotes reward_scale times official success. Advantages divide by this
    fixed scale, with no random group RMS or hand-written syntax reward.

    The WebShop first-hit variant retains its original strongest-peer definition.
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
        self.exact_first_hit = method == EXACT_FIRST_HIT_METHOD
        self.trajectories = tuple(trajectories)
        self.reference = reference_data
        options: Mapping = {} if config is None else config
        if self.exact_first_hit:
            self.radius, self.prior_weight, self.transfer_max = .10, 1., 1.
        else:
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
        if self.exact_first_hit and (benchmark != "webshop" or self.horizon != 15 or self.count != 8):
            raise ValueError("Exact first-hit credit requires WebShop with horizon 15 and eight trajectories")

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

    def _observation_codes(self, *, pure=False) -> torch.Tensor:
        codes = torch.zeros((self.depth, self.count), dtype=torch.long, device="cpu")
        vocab: dict[tuple[int, ...], int] = {}
        for owner, trajectory in enumerate(self.trajectories):
            if not trajectory.steps:
                continue
            initial_repr = trajectory.initial_representation_ids or trajectory.initial_observation_ids
            previous = tuple(trajectory.initial_observation_ids if pure else initial_repr)
            for depth_idx, row in enumerate(trajectory.steps):
                codes[depth_idx, owner] = vocab.setdefault(previous, len(vocab) + 1)
                previous = tuple(row.observation_ids if pure else row.observation_repr_ids)
        return codes

    def _first_hit_weights(self):
        hist = self._features("z_hist")
        codes = self._observation_codes(pure=True)
        other = ~torch.eye(self.count, dtype=torch.bool, device="cpu")

        def matches(left, right, time_distance):
            left_codes, right_codes = codes[left, :, None], codes[right, None, :]
            eligible = (left_codes == right_codes) & (left_codes > 0) & other
            distance = (1.0 - torch.bmm(hist[left], hist[right].transpose(1, 2))).clamp_(time_distance, 2.0)
            return (1.0 - distance / self.radius).clamp_min_(0.0) * eligible

        # At H=15, a .10 radius gives positive time support only at offsets 0 and ±1.
        # Adjacent matches are symmetric. Keep only the largest weight per peer.
        weights = matches(slice(None), slice(None), 0.0)
        if self.depth > 1:
            adjacent = matches(slice(None, -1), slice(1, None), 1.0 / self.horizon)
            weights[:-1] = torch.maximum(weights[:-1], adjacent)
            weights[1:] = torch.maximum(weights[1:], adjacent.transpose(1, 2))
        return weights.to(torch.float64)

    def _same_depth_weights(self):
        # Shapes are [depth, target trajectory, peer trajectory].
        eligible = self.valid[:, :, None] & self.valid[:, None, :]
        eligible &= ~torch.eye(self.count, dtype=torch.bool, device="cpu")[None, :, :]
        if self.benchmark in EXACT_OBSERVATION_BENCHMARKS:
            codes = self._observation_codes()
            obs_match = (codes[:, :, None] == codes[:, None, :]) & (codes[:, :, None] > 0)
            if self.method == "fgpo_obs_only":
                return (obs_match & eligible).to(torch.float64)
            hist = self._features("z_hist")
            hist_dist = (1.0 - torch.bmm(hist, hist.transpose(1, 2))).clamp_(0.0, 2.0)
            return ((1.0 - hist_dist / self.radius).clamp_min_(0.0) * (obs_match & eligible)).to(torch.float64)

        obs = self._features("z_obs")
        distance = (1.0 - torch.bmm(obs, obs.transpose(1, 2))).clamp_(0.0, 2.0)
        del obs
        if self.method != "fgpo_obs_only":
            hist = self._features("z_hist")
            history_distance = (1.0 - torch.bmm(hist, hist.transpose(1, 2))).clamp_(0.0, 2.0)
            torch.maximum(distance, history_distance, out=distance)
            del hist, history_distance
        return ((1.0 - distance / self.radius).clamp_min_(0.0) * eligible).to(torch.float64)

    @torch.no_grad()
    def advantages(self):
        matching_mode = ("exact_token_id" if self.exact_first_hit or self.benchmark in EXACT_OBSERVATION_BENCHMARKS
                         else "dual_view_cosine")
        first_hit_stats = ({"first_hit_steps": 0, "mean_first_hit_distance": None,
                            "mean_first_hit_transfer_distance": None} if self.exact_first_hit else {})
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
                "mean_pending_weight": None,
                "loo_outcome_brier": None,
                "loo_task_prior_brier": None,
                **first_hit_stats,
            }
            return {}
        if self.homogeneous:
            count = int(self.valid.sum())
            zero_transfer = self.all_zero or self.method == "fgpo_no_transfer" or self.exact_first_hit
            won = (self.outcomes > 0).to(torch.float64)
            completed_won = (~self.valid).to(torch.float64) * won[None, :]
            completed_won_peers = completed_won.sum(dim=-1, keepdim=True) - completed_won
            omega = ((self.count - 1.0 - completed_won_peers) + self.prior_weight) / (
                (self.count - 1.0) + self.prior_weight
            )
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
                "mean_pending_weight": float(omega[self.valid].mean()) if count else None,
                "positive_peer_gated_steps": 0 if self.all_one else None,
                "loo_outcome_brier": 0.0,
                "loo_task_prior_brier": 0.0,
                "matched_loo_outcome_brier": None,
                "mean_abs_return_shift": 0.0,
                "positive_failed_steps": 0,
                "mean_advantage": 0.0,
                "advantage_min": 0.0,
                "advantage_max": 0.0,
                **first_hit_stats,
            }
            return {row.step_id: 0.0 for row in self.rows}
        if self.reference is None:
            raise ValueError("Mixed-outcome r3 credit requires frozen-reference features")

        weights = self._first_hit_weights() if self.exact_first_hit else self._same_depth_weights()
        support = weights.sum(dim=-1)
        weighted_outcomes = torch.matmul(weights, self.outcomes)
        positive_support = support > 0
        denominator = torch.where(positive_support, support, torch.ones_like(support))
        peer_values = weighted_outcomes / denominator
        terminal = torch.zeros_like(self.valid)
        terminal[self.times, self.owners] = self.terminals
        continuation = self.valid & ~terminal
        mc_return = self.outcomes[None, :].expand(self.depth, -1)
        reward = mc_return * terminal

        if self.exact_first_hit:
            baseline = (weighted_outcomes + self.prior_weight * self.prior[None, :]) / (
                support + self.prior_weight
            )
            valid_won = self.valid.to(torch.long) * (self.outcomes > 0).to(torch.long)[None, :]
            # A peer alive at v+1 is already alive at v. Include v-1 once per peer.
            valid_won[1:] = torch.maximum(valid_won[1:], valid_won[:-1])
            has_positive_peer = (valid_won.sum(dim=-1, keepdim=True) - valid_won) >= 1
            state_share = has_positive_peer.to(torch.float64) * (
                self.transfer_max * support / (support + self.prior_weight)
            )
            depths = torch.arange(self.depth, device="cpu")[:, None]
            terminal_depth = self.valid.sum(dim=0, keepdim=True)
            supported_depths = torch.where(positive_support, depths, terminal_depth)
            # Shift the suffix minimum one step: the hit must be strictly later.
            # Only support controls the stopping point; the success gate acts there.
            first_supported = supported_depths.flip(0).cummin(dim=0).values.flip(0)
            next_hit = torch.cat((first_supported[1:], terminal_depth))

            def at_hit(values):
                padded = torch.cat((values, torch.zeros_like(values[:1])))
                return padded.gather(0, next_hit)

            next_values = at_hit(peer_values)
            next_share = at_hit(state_share) * continuation
            next_pos_sup = next_hit < terminal_depth
            next_pos_peer = at_hit(has_positive_peer)
            gated = continuation & next_pos_sup & ~next_pos_peer
            target = (1.0 - next_share) * mc_return + next_share * (reward + next_values)
            omega = torch.ones_like(support)
            advantage = (target - baseline) / self.reward_scale
        else:
            if self.method == "fgpo_no_support_weight":
                pre_share = positive_support.to(torch.float64)
            else:
                pre_share = self.transfer_max * support / (support + self.prior_weight)
            baseline = (1.0 - pre_share) * self.prior[None, :] + pre_share * peer_values
            post_share = torch.zeros_like(support) if self.method == "fgpo_no_transfer" else pre_share
            next_values = torch.cat((peer_values[1:], torch.zeros_like(peer_values[:1])), dim=0)
            next_share = torch.cat((post_share[1:], torch.zeros_like(post_share[:1])), dim=0) * continuation
            gated = torch.zeros_like(continuation)
            target = (1.0 - next_share) * mc_return + next_share * (reward + next_values)
            won = (self.outcomes > 0).to(torch.float64)
            completed_won = (~self.valid).to(torch.float64) * won[None, :]
            completed_won_peers = completed_won.sum(dim=-1, keepdim=True) - completed_won
            omega = ((self.count - 1.0 - completed_won_peers) + self.prior_weight) / (
                (self.count - 1.0) + self.prior_weight
            )
            advantage = omega * (target - baseline) / self.reward_scale

        # A fixed scale preserves task weighting. The syntax_valid field is a
        # diagnostic only; parsing outcomes do not replace the official reward.
        values = advantage[self.times, self.owners].tolist()
        self._record_diagnostics(
            weights, support, peer_values, baseline, next_share, target, advantage, omega,
            matching_mode=matching_mode, gated_steps=int(gated.sum()),
        )
        if self.exact_first_hit:
            distances = (next_hit - depths).to(torch.float64)
            hit_distances = distances[continuation & next_pos_sup]
            transfer_distances = distances[next_share > 0]
            first_hit_stats.update(
                first_hit_steps=hit_distances.numel(),
                mean_first_hit_distance=float(hit_distances.mean()) if hit_distances.numel() else None,
                mean_first_hit_transfer_distance=(float(transfer_distances.mean()) if transfer_distances.numel() else None),
            )
            self.diagnostics.update(first_hit_stats)
        return {row.step_id: value for row, value in zip(self.rows, values)}

    def _record_diagnostics(self, weights, support, peer_values, baseline, share, target, advantage, omega,
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
            "mean_pending_weight": float(omega[valid].mean()),
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
