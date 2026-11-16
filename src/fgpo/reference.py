"""Frozen-reference response laws and bounded public-history representations.

r3 scores the exact sampled actor prefixes. No instruction reconstruction,
observation surprise, feedback-only scoring or intrinsic reward is computed.
Each packed segment is released before the next one is evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .protocol import OWN_METHODS


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


def sampled_logprobs(policy, hidden, target, *, role, chunk_tokens=64):
    """One target upload and one result download for the complete response."""
    if not target:
        return ()
    if chunk_tokens < 1 or len(hidden) != len(target):
        raise ValueError("Reference head slices must match the actual response tokens.")
    device = policy.model.get_output_embeddings().weight.device
    ids = torch.tensor(target, dtype=torch.long, device=device)
    values = []
    for start in range(0, len(target), chunk_tokens):
        end = min(start + chunk_tokens, len(target))
        logits = policy.head(hidden[start:end], role=role)
        values.append(-F.cross_entropy(logits, ids[start:end], reduction="none"))
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
                role="bipace.current_actor_history")
            positions = torch.tensor([row.prompt_length - 1 for row in rows],
                                     dtype=torch.long, device=layer.device)
            values = F.normalize(layer[0].index_select(0, positions).float(), dim=-1).cpu()
            result.update((row.step_id, value) for row, value in zip(rows, values, strict=True))
            del hidden, layer, values
    return result


@torch.no_grad()
def prepare_round(policy, trajectories, *, method="fgpo", config=None):
    config = {} if config is None else config
    own = method in OWN_METHODS
    scale = float(config.get("reward_scale", 10.))
    chunk = int(config.get("head_chunk_tokens", 64))
    result = ReferenceRound()
    observation_users = {}
    for trajectory in trajectories:
        by_segment = {}
        previous = tuple(trajectory.initial_observation_ids)
        for row in trajectory.steps:
            by_segment.setdefault(row.response_span.segment_id, []).append(row)
            if own:
                key = previous or policy.encode("[Empty stdout]")
                observation_users.setdefault(key, []).append(row.step_id)
                previous = tuple(row.observation_repr_ids)
        for segment in trajectory.segments:
            rows = by_segment.get(segment.segment_id, ())
            if not rows:
                continue
            hidden, layer = policy.packed_segment_hidden(
                segment.token_ids[:segment.behavior_closed_length],
                hidden_layer=-8 if own else None, role="reference.packed_segment")
            histories = None
            if own:
                positions = torch.tensor([row.prompt_length - 1 for row in rows],
                                         dtype=torch.long, device=layer.device)
                histories = F.normalize(layer[0].index_select(0, positions).float(), dim=-1).cpu()
            for index, row in enumerate(rows):
                span = row.response_span
                scores = sampled_logprobs(policy, hidden[0, span.start - 1:span.stop - 1],
                                          row.response_ids, role="reference.response_head",
                                          chunk_tokens=chunk)
                reward = scale * float(trajectory.official_success) if row.index == len(trajectory.steps) - 1 else 0.
                result.steps[row.step_id] = ReferenceStep(
                    scores, z_hist=histories[index] if own else None, reward=reward)
            del hidden, layer, histories
    if own and observation_users:
        # The policy implements length-aware, token-budgeted microbatches. A
        # single padded batch of all observation tokens is deliberately avoided.
        keys = list(observation_users)
        vectors = policy.observation_embeddings(keys, role="reference.local_observations")
        for key, vector in zip(keys, vectors, strict=True):
            for step_id in observation_users[key]:
                result.steps[step_id].z_obs = vector
    result.diagnostics = {"actual_actor_prefixes": True, "intrinsic_reward_enabled": False,
                          "unique_local_observations": len(observation_users)}
    return result
