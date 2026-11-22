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

from .protocol import EXACT_FIRST_HIT_METHOD, OWN_METHODS
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
    needs_obs = (own and method != EXACT_FIRST_HIT_METHOD
                 and benchmark in HORIZONS and benchmark not in EXACT_OBSERVATION_BENCHMARKS)
    scale = float(config.get("reward_scale", 10.))
    chunk = int(config.get("head_chunk_tokens", 64))
    ledger = getattr(policy, "ledger", None)
    result = ReferenceRound(conditioning="canonical-pomdp-filtration-r6" if own else "actual-bounded-actor-prefix-r3")
    observation_users = {}
    segment_items = []
    for trajectory in trajectories:
        by_segment = {}
        previous = tuple(trajectory.initial_representation_ids or trajectory.initial_observation_ids)
        for row in trajectory.steps:
            by_segment.setdefault(row.response_span.segment_id, []).append(row)
            if needs_obs:
                key = previous or policy.encode("[Empty stdout]")
                observation_users.setdefault(key, []).append(row.step_id)
                previous = tuple(row.observation_repr_ids)
        for segment in trajectory.segments:
            rows = tuple(by_segment.get(segment.segment_id, ()))
            if not rows:
                continue
            if any(row.response_span.length != len(row.response_ids) for row in rows):
                raise ValueError("Reference head slices must match each actual response's tokens.")
            closed = segment.behavior_closed_length or len(segment.token_ids)
            if closed < 2:
                raise ValueError("Reference segment requires at least two tokens.")
            segment_items.append((trajectory, segment, rows, closed))

    batch_limit = max(1, int(getattr(policy, "representation_batch_size", 8)))
    token_budget = max(1, int(getattr(policy, "representation_token_budget", 8192)))
    segment_items.sort(key=lambda item: item[3])
    microbatches, current_mb, current_max = [], [], 0
    for item in segment_items:
        seq_len = item[3] - 1
        next_max = max(current_max, seq_len)
        if current_mb and (len(current_mb) >= batch_limit or next_max * (len(current_mb) + 1) > token_budget):
            microbatches.append(current_mb)
            current_mb, current_max = [item], seq_len
        else:
            current_mb.append(item)
            current_max = next_max
    if current_mb:
        microbatches.append(current_mb)

    for mb in microbatches:
        mb_group_ids = {traj.group_id for traj, _, _, _ in mb}
        ctx_group = next(iter(mb_group_ids)) if len(mb_group_ids) == 1 else None
        with (ledger.context(group_id=ctx_group, round_index=round_index)
              if ledger is not None else nullcontext()), \
             (ledger.aggregate_model_calls() if ledger is not None else nullcontext()):
            if len(mb) == 1:
                _, segment, rows, closed = mb[0]
                hidden, layer = policy.packed_segment_hidden(
                    segment.token_ids[:closed],
                    hidden_layer=-8 if needs_history else None,
                    hidden_positions=[row.prompt_length - 1 for row in rows] if needs_history else None,
                    role="reference.packed_segment",
                )
                histories = F.normalize(layer[0].float(), dim=-1).cpu() if needs_history else None
                del layer
                targets = tuple(token for row in rows for token in row.response_ids)
                positions = [position for row in rows
                             for position in range(row.response_span.start - 1, row.response_span.stop - 1)]
                scores = sampled_logprobs(policy, hidden[0], targets, positions=positions,
                                          role="reference.response_head", chunk_tokens=chunk)
                del hidden
                offset = 0
                for index, row in enumerate(rows):
                    stop = offset + len(row.response_ids)
                    reward = scale * float(mb[0][0].official_success) if row.index == len(mb[0][0].steps) - 1 else 0.
                    result.steps[row.step_id] = ReferenceStep(
                        scores[offset:stop], z_hist=histories[index] if needs_history else None, reward=reward)
                    offset = stop
                del histories, scores, targets, positions
            else:
                seqs = [tuple(segment.token_ids[:closed - 1]) for _, segment, _, closed in mb]
                lengths = [len(seq) for seq in seqs]
                width = max(lengths)
                pad_id = policy.tokenizer.pad_token_id
                ids = torch.tensor([seq + (pad_id,) * (width - len(seq)) for seq in seqs],
                                   dtype=torch.long, device=policy.device)
                valid = torch.arange(width, device=policy.device)[None, :] < torch.tensor(lengths, device=policy.device)[:, None]
                hidden, layer, _ = policy.body(
                    ids, attention_mask=valid, hidden_layer=-8 if needs_history else None,
                    actual_tokens=sum(lengths), logical_length=width, role="reference.packed_segment",
                )
                del ids, valid
                histories = None
                if needs_history:
                    hist_positions = [b * width + (row.prompt_length - 1)
                                      for b, (_, _, rows, _) in enumerate(mb) for row in rows]
                    hist_tensor = layer.reshape(len(mb) * width, -1).index_select(
                        0, torch.tensor(hist_positions, dtype=torch.long, device=layer.device))
                    histories = F.normalize(hist_tensor.float(), dim=-1).cpu()
                    del hist_tensor, hist_positions
                del layer
                targets = tuple(token for _, _, rows, _ in mb for row in rows for token in row.response_ids)
                positions = [b * width + position
                             for b, (_, _, rows, _) in enumerate(mb)
                             for row in rows
                             for position in range(row.response_span.start - 1, row.response_span.stop - 1)]
                scores = sampled_logprobs(policy, hidden.reshape(len(mb) * width, -1), targets,
                                          positions=positions, role="reference.response_head", chunk_tokens=chunk)
                del hidden
                hist_offset, score_offset = 0, 0
                for traj, _, rows, _ in mb:
                    for row in rows:
                        stop = score_offset + len(row.response_ids)
                        reward = scale * float(traj.official_success) if row.index == len(traj.steps) - 1 else 0.
                        result.steps[row.step_id] = ReferenceStep(
                            scores[score_offset:stop],
                            z_hist=histories[hist_offset] if needs_history else None,
                            reward=reward,
                        )
                        score_offset = stop
                        hist_offset += 1
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
