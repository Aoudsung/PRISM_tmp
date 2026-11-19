"""Frozen-reference response laws and bounded public-history representations.

r3 scores the exact sampled actor prefixes. No instruction reconstruction,
observation surprise, feedback-only scoring or intrinsic reward is computed.
Each packed segment is released before the next one is evaluated.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .protocol import OWN_METHODS
from .public import EXACT_OBSERVATION_BENCHMARKS, HORIZONS


@dataclass
class ReferenceStep:
    token_logprobs: tuple[float, ...]
    z_hist: torch.Tensor | None = None
    z_obs: torch.Tensor | None = None
    reward: float = 0.


@dataclass
class ReferenceRound:
    steps: dict[str, ReferenceStep] = field(default_factory=dict)
    conditioning: str = "actual-bounded-actor-prefix-r3"
    diagnostics: dict = field(default_factory=dict)

    def release_features(self):
        """Retain the frozen response law after credit and diagnostics finish."""
        for step in self.steps.values():
            step.z_hist = step.z_obs = None
        self.diagnostics["features_released_after_credit"] = True


@torch.no_grad()
def sampled_logprobs(policy, hidden, target, *, role, chunk_tokens=64, positions=None):
    """Upload targets and download scores once per packed segment.

    Optional causal positions gather only one head chunk at a time. Full
    vocabulary logits and selected hidden states are released before the next
    projection, including when adjacent chunks belong to different responses.
    """
    if not target:
        return ()
    source_length = len(hidden) if positions is None else len(positions)
    if chunk_tokens < 1 or source_length != len(target):
        raise ValueError("Reference head slices must match the actual response tokens.")
    device = policy.model.get_output_embeddings().weight.device
    ids = torch.tensor(target, dtype=torch.long, device=device)
    if positions is not None:
        positions = torch.as_tensor(positions, dtype=torch.long, device=hidden.device)
    values = []
    for start in range(0, len(target), chunk_tokens):
        end = min(start + chunk_tokens, len(target))
        selected = hidden[start:end] if positions is None else hidden.index_select(0, positions[start:end])
        logits = policy.head(selected, role=role)
        values.append(-F.cross_entropy(logits, ids[start:end], reduction="none"))
        del logits, selected
    return tuple(torch.cat(values).detach().cpu().tolist())


@torch.no_grad()
def actor_history_embeddings(policy, trajectories):
    """BiPACE's current-actor prompt fingerprints, including retained history."""
    result = {}
    for trajectory in trajectories:
        by_segment = {}
        for row in trajectory.steps:
            by_segment.setdefault(row.response_span.segment_id, []).append(row)
        for segment in trajectory.segments:
            rows = by_segment.get(segment.segment_id, ())
            if not rows:
                continue
            hidden, layer = policy.packed_segment_hidden(
                segment.token_ids[:segment.behavior_closed_length], hidden_layer=-8,
                hidden_positions=[row.prompt_length - 1 for row in rows], stop_at_hidden_layer=True,
                role="bipace.current_actor_history")
            values = F.normalize(layer[0].float(), dim=-1).cpu()
            result.update((row.step_id, value) for row, value in zip(rows, values, strict=True))
            del hidden, layer, values
    return result


@torch.no_grad()
def prepare_round(policy, trajectories, *, method="fgpo", config=None, round_index=None):
    config = {} if config is None else config
    trajectories = tuple(trajectories)
    own = method in OWN_METHODS
    benchmark = trajectories[0].benchmark if trajectories else config.get("benchmark")
    needs_history = own and method != "fgpo_obs_only"
    needs_obs = own and benchmark in HORIZONS and benchmark not in EXACT_OBSERVATION_BENCHMARKS
    scale = float(config.get("reward_scale", 10.))
    chunk = int(config.get("head_chunk_tokens", 64))
    ledger = getattr(policy, "ledger", None)
    result = ReferenceRound()
    observation_users = {}
    for trajectory in trajectories:
        by_segment = {}
        previous = tuple(trajectory.initial_observation_ids)
        for row in trajectory.steps:
            by_segment.setdefault(row.response_span.segment_id, []).append(row)
            if needs_obs:
                key = previous or policy.encode("[Empty stdout]")
                observation_users.setdefault(key, []).append(row.step_id)
                previous = tuple(row.observation_repr_ids)
        for segment in trajectory.segments:
            rows = by_segment.get(segment.segment_id, ())
            if not rows:
                continue
            with (ledger.context(group_id=trajectory.group_id, round_index=round_index)
                  if ledger is not None else nullcontext()), \
                 (ledger.aggregate_model_calls() if ledger is not None else nullcontext()):
                hidden, layer = policy.packed_segment_hidden(
                    segment.token_ids[:segment.behavior_closed_length],
                    hidden_layer=-8 if needs_history else None,
                    hidden_positions=[row.prompt_length - 1 for row in rows] if needs_history else None,
                    role="reference.packed_segment")
                histories = None
                if needs_history:
                    histories = F.normalize(layer[0].float(), dim=-1).cpu()
                del layer
                if any(row.response_span.length != len(row.response_ids) for row in rows):
                    raise ValueError("Reference head slices must match each actual response's tokens.")
                targets = tuple(token for row in rows for token in row.response_ids)
                positions = [position for row in rows
                             for position in range(row.response_span.start - 1, row.response_span.stop - 1)]
                scores = sampled_logprobs(policy, hidden[0], targets, positions=positions,
                                          role="reference.response_head", chunk_tokens=chunk)
                del hidden
            offset = 0
            for index, row in enumerate(rows):
                stop = offset + len(row.response_ids)
                reward = scale * float(trajectory.official_success) if row.index == len(trajectory.steps) - 1 else 0.
                result.steps[row.step_id] = ReferenceStep(
                    scores[offset:stop], z_hist=histories[index] if needs_history else None, reward=reward)
                offset = stop
            del histories, scores, targets, positions
    unique_observations = len(observation_users)
    if needs_obs and observation_users:
        # The policy implements length-aware, token-budgeted microbatches. A
        # single padded batch of all observation tokens is deliberately avoided.
        with (ledger.context(group_id=None, round_index=round_index)
              if ledger is not None else nullcontext()), \
             (ledger.aggregate_model_calls() if ledger is not None else nullcontext()):
            keys = list(observation_users)
            vectors = policy.observation_embeddings(keys, role="reference.local_observations")
            for key, vector in zip(keys, vectors, strict=True):
                for step_id in observation_users[key]:
                    result.steps[step_id].z_obs = vector
            observation_users.clear()
            del keys, vectors
    result.diagnostics = {
        "actual_actor_prefixes": True,
        "intrinsic_reward_enabled": False,
        "scored_trajectories": len(trajectories),
        "local_observation_embeddings_computed": bool(needs_obs and unique_observations),
        "unique_local_observations": unique_observations,
    }
    return result
