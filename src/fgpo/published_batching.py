# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Adapted from verl-agent under the Apache License, Version 2.0.
# See THIRD_PARTY_NOTICES.md and THIRD_PARTY_LICENSE_APACHE_2_0.txt.
# Changes: preserve released distributed batch mathematics using virtual ranks,
# immutable rollout identifiers, and merged duplicate coefficients per update.
"""Source-exact batch multiplicity and microbatch weights without physical DP.

The source's rank count defines a mathematical row layout, independently of
the GPUs used to execute the model. No rollouts or synthetic tokens are added.
The returned plan is ordinary checkpointable data and consumes the NumPy RNG
only when the source's adjust_batch(copy) operation requires duplicate rows.
"""
from __future__ import annotations

import heapq
import math

from .baseline_specs import UPSTREAM_COMMIT, published_recipe


def _balanced_partitions(lengths, world_size):
    """Released equal-size Karmarkar--Karp, including its tie ordering.

    The source sorts each final partition by input position after balancing;
    those sorted positions determine subsequent rank-local PPO minibatches.
    """
    class Partition:
        def __init__(self):
            self.total = 0
            self.items = []

        def add(self, index, length):
            self.items.append((index, length))
            self.total += length

        def merge(self, other):
            for index, length in other.items:
                self.items.append((index, length))
                self.total += length

        def __lt__(self, other):
            if self.total != other.total:
                return self.total < other.total
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items):
            self.parts = [Partition() for _ in range(world_size)]
            for index, (position, length) in enumerate(items):
                self.parts[index].add(position, length)
            self.parts.sort(reverse=True)

        @property
        def spread(self):
            return self.parts[0].total - self.parts[-1].total

        def merge(self, other):
            for index in range(world_size):
                self.parts[index].merge(other.parts[world_size - 1 - index])
            self.parts.sort(reverse=True)

        def __lt__(self, other):
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.parts[0] > other.parts[0]

    ordered = sorted((length, index) for index, length in enumerate(lengths))
    queue = []
    for start in range(0, len(ordered), world_size):
        items = [(index, length) for length, index in ordered[start:start + world_size]]
        heapq.heappush(queue, State(items))
    while len(queue) > 1:
        first, second = heapq.heappop(queue), heapq.heappop(queue)
        first.merge(second)
        heapq.heappush(queue, first)
    return [sorted(index for index, _ in part.items) for part in queue[0].parts]


def plan_published_batch(trajectories, method, *, config=None, rng=None):
    """Return saved credit-row order and per-token weights per optimizer step.

    ``minibatches`` is a tuple of tuples of ``(step_id, coefficient)``. Each
    coefficient multiplies every response token's full PPO + KL - entropy
    objective. Repeated occurrences of a step are merged only within the same
    optimizer update. ``credit_row_ids`` includes copies for GiGPO, whose
    trainer adjusts/balances before computing credit; the other recipes compute
    credit first, on the original rollout rows.
    """
    trajectories = tuple(trajectories)
    if not trajectories:
        raise ValueError("A published batch plan requires actual completed rollouts")
    supplied = {} if config is None else config
    settings = {**published_recipe(method, trajectories[0].benchmark,
                                   supplied.get("model_path", "Qwen/Qwen2.5-7B-Instruct")), **supplied}
    if settings["source_dynamic_batching"] or settings["source_adjust_mode"] != "copy":
        raise ValueError("The selected source recipes use static microbatches and adjust_batch(copy)")
    if not settings["source_balance_batch"] or settings["source_row_order"] != "group_trajectory_step":
        raise ValueError("The selected source recipes balance their trajectory-major rollout rows")
    rows = [row for trajectory in trajectories for row in trajectory.steps]
    if not rows or any(len(row.response_ids) == 0 for row in rows):
        raise ValueError("Published PPO batch plans require nonempty sampled response rows")
    if len({row.step_id for row in rows}) != len(rows):
        raise ValueError("Original rollout rows must be unique before source batch copying")
    world = int(settings["source_virtual_world_size"])
    micro = int(settings["source_actor_microbatch"])
    global_mini = int(settings["minibatch"]) * int(settings["source_rollout_n"])
    if global_mini % world or (global_mini // world) % micro:
        raise ValueError("Source global minibatch must divide into rank-local whole microbatches")
    local_mini = global_mini // world
    accumulation = local_mini // micro
    divisor = math.lcm(world * micro,
        world * int(settings["source_rollout_logprob_microbatch"]),
        world * int(settings["source_ref_logprob_microbatch"]))
    indices = list(range(len(rows)))
    copies = []
    remainder = len(rows) % divisor
    if remainder:
        if rng is None:
            import numpy as np
            rng = np.random
        copies = [int(index) for index in rng.choice(len(rows), divisor - remainder, replace=False)]
        indices.extend(copies)
    # This is exactly the source attention_mask.sum() for each real response
    # row: complete effective prompt and generated response, with padding out.
    lengths = [rows[index].prompt_length + len(rows[index].response_ids) for index in indices]
    partitions = _balanced_partitions(lengths, world)
    ranks = [[indices[position] for position in partition] for partition in partitions]
    ordered_indices = [index for rank in ranks for index in rank]
    credit_indices = ordered_indices if settings["source_copy_before_advantage"] else range(len(rows))
    minibatches, micro_token_counts, occurrence_counts = [], [], []
    for start in range(0, len(ranks[0]), local_mini):
        weights, token_counts, occurrences = {}, [], 0
        for rank in ranks:
            local = rank[start:start + local_mini]
            occurrences += len(local)
            for offset in range(0, len(local), micro):
                members = local[offset:offset + micro]
                tokens = sum(len(rows[index].response_ids) for index in members)
                token_counts.append(tokens)
                # Retain the source's fixed accumulation even for a partial
                # final minibatch. FSDP averages the accumulated rank gradients.
                coefficient = 1. / (world * accumulation * tokens)
                for index in members:
                    step_id = rows[index].step_id
                    weights[step_id] = weights.get(step_id, 0.) + coefficient
        minibatches.append(tuple(weights.items()))
        micro_token_counts.append(tuple(token_counts))
        occurrence_counts.append(occurrences)
    return {
        "format": "published-verl-batch-v1",
        "method": method,
        "source_commit": UPSTREAM_COMMIT,
        "credit_row_ids": tuple(rows[index].step_id for index in credit_indices),
        "minibatches": tuple(minibatches),
        "diagnostics": {
            "input_rows": len(rows), "adjusted_rows": len(indices), "copied_rows": len(copies),
            "copied_row_ids": tuple(rows[index].step_id for index in copies),
            "copy_before_advantage": bool(settings["source_copy_before_advantage"]),
            "source_row_order": settings["source_row_order"],
            "virtual_world_size": world, "actor_microbatch_rows": micro,
            "global_minibatch_rows": global_mini, "fixed_gradient_accumulation": accumulation,
            "adjust_divisor": divisor, "rank_rows": tuple(map(len, ranks)),
            "rank_token_totals": tuple(sum(lengths[position] for position in partition)
                                       for partition in partitions),
            "minibatch_occurrences": tuple(occurrence_counts),
            "microbatch_response_tokens": tuple(micro_token_counts),
            "optimizer_steps": len(minibatches), "additional_rollouts": 0,
        },
    }
