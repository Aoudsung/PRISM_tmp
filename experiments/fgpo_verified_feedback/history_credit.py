"""Structural history matching for the V5 empirical credit comparison."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fgpo.credit import CrossTrajectoryCredit
from fgpo.protocol import OWN_METHODS
from fgpo.reference import ReferenceRound, ReferenceStep, sampled_logprobs


EXACT_OBSERVATION_BENCHMARKS = {"alfworld", "webshop"}
ALGORITHM = "v5_structural_history_conditioned_peer_value_sharing"


@dataclass
class HistoryReferenceStep(ReferenceStep):
    z_action: torch.Tensor | None = None


class HistoryReferenceRound(ReferenceRound):
    def release_features(self):
        super().release_features()
        for step in self.steps.values():
            step.z_action = None


@torch.no_grad()
def prepare_round(policy, trajectories, *, method="fgpo", config=None, round_index=None):
    """Score actual responses and capture pre-action history in the same forward."""
    config = {} if config is None else config
    own = method in OWN_METHODS
    needs_history = own and method != "fgpo_obs_only"
    scale = float(config.get("reward_scale", 10.))
    chunk = int(config.get("head_chunk_tokens", 64))
    ledger = getattr(policy, "ledger", None)
    result = HistoryReferenceRound()
    observation_users = {}
    trajectory_count = 0
    for trajectory in trajectories:
        trajectory_count += 1
        by_segment = {}
        executed_states = [None] * len(trajectory.steps)
        previous = trajectory.initial_observation_ids
        for row in trajectory.steps:
            by_segment.setdefault(row.response_span.segment_id, []).append(row)
            if own and trajectory.benchmark == "appworld":
                key = tuple(previous) or policy.encode("[Empty stdout]")
                observation_users.setdefault(key, []).append(row.step_id)
                previous = row.observation_repr_ids

        for segment in trajectory.segments:
            rows = by_segment.get(segment.segment_id, ())
            if not rows:
                continue
            if any(row.response_span.length != len(row.response_ids) for row in rows):
                raise ValueError("Reference head slices must match each actual response's tokens.")
            targets = tuple(token for row in rows for token in row.response_ids)
            positions = [position for row in rows
                         for position in range(row.response_span.start - 1, row.response_span.stop - 1)]
            closed = segment.token_ids[:segment.behavior_closed_length]
            with (ledger.context(group_id=trajectory.group_id, round_index=round_index)
                  if ledger is not None else nullcontext()), \
                 (ledger.aggregate_model_calls() if ledger is not None else nullcontext()):
                histories = None
                if needs_history:
                    executed = [row for row in rows if row.environment_info["native_executed"]]
                    capture_positions = [row.prompt_length - 1 for row in rows]
                    capture_positions.extend(row.response_span.stop - 1 for row in executed)
                    # The packed scoring helper drops the final token. Retain it
                    # here to capture the real closure at a window boundary.
                    hidden, layer, _ = policy.body(
                        closed, hidden_layer=-8, hidden_positions=capture_positions,
                        role="reference.packed_segment")
                    selected = layer[0].float()
                    selected[:len(rows)] = F.normalize(selected[:len(rows)], dim=-1)
                    captured = selected.cpu()
                    histories = captured[:len(rows)].clone()
                    for row, vector in zip(executed, captured[len(rows):], strict=True):
                        executed_states[row.index] = vector
                    del layer, selected, captured
                else:
                    hidden, _ = policy.packed_segment_hidden(
                        closed, hidden_layer=None, role="reference.packed_segment")
                scores = sampled_logprobs(policy, hidden[0], targets, positions=positions,
                                          role="reference.response_head", chunk_tokens=chunk)
                del hidden

            offset = 0
            for index, row in enumerate(rows):
                stop = offset + len(row.response_ids)
                reward = scale * float(trajectory.official_success) if row.index == len(trajectory.steps) - 1 else 0.
                result.steps[row.step_id] = HistoryReferenceStep(
                    scores[offset:stop], z_hist=histories[index] if needs_history else None, reward=reward)
                offset = stop

        if needs_history and trajectory.steps:
            zero = torch.zeros_like(result.steps[trajectory.steps[0].step_id].z_hist)
            # Shift before accumulating. Already-captured closures remain in
            # chronological order when prior turns roll out of the actor window.
            preceding = [None, *executed_states[:-1]]
            raw = torch.stack([vector if vector is not None else zero for vector in preceding])
            count = torch.tensor([vector is not None for vector in preceding], device="cpu").cumsum(0)
            means = F.normalize(raw.cumsum(dim=0) / count.clamp_min(1)[:, None], dim=-1)
            # An empty executed-action history is a unit vector orthogonal to
            # the action space. This also makes the depth-zero distance zero.
            action_history = torch.cat((means, (count == 0)[:, None].to(raw.dtype)), dim=-1)
            for row, vector in zip(trajectory.steps, action_history, strict=True):
                result.steps[row.step_id].z_action = vector

    if observation_users:
        with (ledger.context(group_id=None, round_index=round_index)
              if ledger is not None else nullcontext()), \
             (ledger.aggregate_model_calls() if ledger is not None else nullcontext()):
            keys = list(observation_users)
            vectors = policy.observation_embeddings(keys, role="reference.local_observations")
            for key, vector in zip(keys, vectors, strict=True):
                for step_id in observation_users[key]:
                    result.steps[step_id].z_obs = vector
    result.diagnostics = {
        "actual_actor_prefixes": True,
        "intrinsic_reward_enabled": False,
        "scored_trajectories": trajectory_count,
        "local_observation_embeddings_computed": bool(observation_users),
        "unique_local_observations": len(observation_users),
    }
    return result


class StructuralHistoryCredit(CrossTrajectoryCredit):
    def _pure_observation_matches(self):
        codes = torch.zeros((self.depth, self.count), dtype=torch.long, device="cpu")
        vocabulary = {}
        for owner, trajectory in enumerate(self.trajectories):
            previous = tuple(trajectory.initial_observation_ids)
            for depth, row in enumerate(trajectory.steps):
                codes[depth, owner] = vocabulary.setdefault(previous, len(vocabulary) + 1)
                previous = tuple(row.observation_ids)
        return codes[:, :, None] == codes[:, None, :]

    def _distance(self, attribute):
        features = self._features(attribute)
        return (1. - torch.bmm(features, features.transpose(1, 2))).clamp_(0., 2.)

    def _weights(self, exact_observations):
        eligible = self.valid[:, :, None] & self.valid[:, None, :]
        eligible &= ~torch.eye(self.count, dtype=torch.bool, device="cpu")[None, :, :]
        if exact_observations:
            eligible &= self._pure_observation_matches()
        if self.method == "fgpo_obs_only":
            if exact_observations:
                return eligible.to(torch.float64)
            distance = self._distance("z_obs")
        else:
            distance = self._distance("z_hist")
            torch.maximum(distance, self._distance("z_action"), out=distance)
            if not exact_observations:
                torch.maximum(distance, self._distance("z_obs"), out=distance)
        return ((1. - distance / self.radius).clamp_min_(0.) * eligible).to(torch.float64)

    @torch.no_grad()
    def advantages(self):
        exact_observations = self.benchmark in EXACT_OBSERVATION_BENCHMARKS
        matching_mode = "exact_token_id" if exact_observations else "dual_view_cosine"
        if not self.rows or self.homogeneous:
            result = super().advantages()
            self.diagnostics.update(algorithm=ALGORITHM, local_observation_matching=matching_mode)
            return result
        if self.reference is None:
            raise ValueError("Mixed-outcome V5 credit requires frozen-reference features")

        # V4's update and leave-one-trajectory-out diagnostics consume the new weights.
        weights = self._weights(exact_observations)
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
        target = (1. - next_share) * mc_return + next_share * (reward + next_values)
        advantage = (target - baseline) / self.reward_scale
        self._record_diagnostics(
            weights, support, peer_values, baseline, next_share, target, advantage,
            matching_mode=matching_mode, gated_steps=int(gated.sum()),
        )
        self.diagnostics["algorithm"] = ALGORITHM
        values = advantage[self.times, self.owners].tolist()
        return {row.step_id: value for row, value in zip(self.rows, values, strict=True)}


@torch.no_grad()
def advantages(trajectories, reference_data, *, method="fgpo", config=None, diagnostics=None):
    """Return V5 step advantages and optional diagnostics on each supplied group."""
    if method not in OWN_METHODS:
        raise ValueError(f"Unknown F-GPO history credit method {method!r}")
    groups = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.group_id, []).append(trajectory)
    result = {}
    for group_id, group in groups.items():
        estimator = StructuralHistoryCredit(group, reference_data, method=method, config=config)
        group_advantages = estimator.advantages()
        if result.keys() & group_advantages.keys():
            raise ValueError("Step identities must be unique across the entire collected round")
        result.update(group_advantages)
        if diagnostics is not None:
            diagnostics[group_id] = estimator.diagnostics
    return result
