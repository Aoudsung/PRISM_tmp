"""Exact finite-model research diagnostics for PRISM DCC.

These are controlled mechanism studies, not native agent benchmarks.
The model class and finite program grammar are supplied to every selector.
No neural model, language model, or native environment is trained here.
"""

import argparse
import itertools
import json
import math
from pathlib import Path
from functools import lru_cache

import numpy as np


@lru_cache(None)
def forced_atomic_value(relevant, nuisance, queries):
    """One-step EVSI, forced to query, uniform ties; exact expectation."""
    if relevant == 0:
        return 1.0
    if queries < relevant:
        return 0.5
    if relevant == 1:
        return 1.0
    total = relevant + nuisance
    return (
        relevant / total * forced_atomic_value(relevant - 1, nuisance, queries - 1)
        + (nuisance / total * forced_atomic_value(relevant, nuisance - 1, queries - 1)
           if nuisance else 0.0)
    )


def parity_diagnostics():
    rows = []
    for depth in (2, 3, 4):
        for nuisance in (0, 2, 6, 14, 30, 62):
            candidates = math.comb(depth + nuisance, depth)
            rows.append({
                "relevant_bits": depth,
                "nuisance_fields": nuisance,
                "query_budget": depth,
                "complete_subset_candidates": candidates,
                "open_loop_or_atomic_stop_success": 0.5,
                "atomic_forced_uniform_ties_success": forced_atomic_value(depth, nuisance, depth),
                "uniform_subset_success": 0.5 + 0.5 / candidates,
                "predictive_entropy_uniform_ties_success": 0.5 + 0.5 / candidates,
                "complete_program_success": 1.0,
            })
    # Direct enumeration: all binary signals and all size-two query subsets.
    all_values = []
    for subset in itertools.combinations(range(8), 2):
        conditional = {}
        for state in itertools.product((0, 1), repeat=8):
            observed = tuple(state[i] for i in subset)
            label = state[0] ^ state[1]
            counts = conditional.setdefault(observed, [0, 0])
            counts[label] += 1
        all_values.append(sum(max(v) for v in conditional.values()) / 256)
    exact_uniform = 0.5 + 0.5 / math.comb(8, 2)
    assert abs(float(np.mean(all_values)) - exact_uniform) < 1e-12
    return {"rows": rows, "direct_enumeration_check_n8_k2": exact_uniform,
            "single_bit_value_of_information": 0.0,
            "joint_two_bit_value_of_information": 0.5,
            "interpretation": "Known-model planning diagnostic; all individual fields are independent fair bits. Predictive-entropy ties are uniform. No learned novelty claim."}


def likelihoods(num_programs):
    """p(z, paired_response | query program, model).

    A model is (relevant program, polarity). An informative program reveals
    the XOR of the relevant public bits. Wrong programs give an independent bit.
    From the same post-query snapshot, both actions use a shared U~Uniform(0,1).
    The correct action passes for U<.8; the other passes for U<.2.
    Hence paired outcomes are 00 with .2, 11 with .2, and the correct-only
    outcome with .6. The joint likelihood preserves this sibling dependence.
    """
    table = np.zeros((num_programs, 2 * num_programs, 8), dtype=np.float64)
    for query in range(num_programs):
        for model in range(2 * num_programs):
            relevant, polarity = divmod(model, 2)
            for bit in (0, 1):
                table[query, model, 4 * bit] = 0.1  # both fail
                table[query, model, 4 * bit + 1] = 0.1  # both pass
                if query == relevant:
                    correct = bit ^ polarity
                    table[query, model, 4 * bit + 2 + correct] = 0.3
                else:
                    table[query, model, 4 * bit + 2] = 0.15
                    table[query, model, 4 * bit + 3] = 0.15
    assert np.allclose(table.sum(axis=-1), 1.0)
    return table


def policy_values(num_programs):
    # Candidate deployment policy: execute one full program, then choose xor b.
    values = np.full((2 * num_programs, 2 * num_programs), 0.5)
    for model in range(2 * num_programs):
        values[model, model] = 0.8
        values[model, model ^ 1] = 0.2
    return values


def entropy(probabilities, axis=-1):
    terms = np.zeros_like(probabilities)
    np.log(probabilities, out=terms, where=probabilities > 0)
    return -(probabilities * terms).sum(axis=axis)


def scores_for(weight, table, method):
    """Closed-form finite-particle expected value, without retraining fantasies."""
    predictive = np.einsum("qmo,m->qo", table, weight)
    if method == "predictive_entropy":
        return entropy(predictive)
    if method == "model_information":
        return entropy(predictive) - np.einsum("qm,m->q", entropy(table), weight)
    if method == "complete_program_evsi":
        # Posterior policy value = .5 + .3 max_j |w(j,0)-w(j,1)|.
        # The normalization cancels when multiplied by p(outcome).
        joint = table * weight[None, :, None]
        signed = joint[:, 0::2, :] - joint[:, 1::2, :]
        after = 0.5 + 0.3 * np.max(np.abs(signed), axis=1).sum(axis=1)
        before = 0.5 + 0.3 * np.max(np.abs(weight[0::2] - weight[1::2]))
        result = after - before
        assert np.min(result) > -1e-12
        return result
    raise ValueError(method)


def check_likelihood_and_evsi():
    """Independently enumerate raw four-bit worlds and verify EVSI formulas."""
    pairs = list(itertools.combinations(range(4), 2))
    table = likelihoods(len(pairs))
    for qi, query in enumerate(pairs):
        for mi in range(2 * len(pairs)):
            ri, polarity = divmod(mi, 2)
            true_pair = pairs[ri]
            counts = np.zeros(8)
            for bits in itertools.product((0, 1), repeat=4):
                correct = bits[true_pair[0]] ^ bits[true_pair[1]] ^ polarity
                obs = bits[query[0]] ^ bits[query[1]]
                counts[4 * obs] += 0.2 / 16
                counts[4 * obs + 1] += 0.2 / 16
                counts[4 * obs + 2 + correct] += 0.6 / 16
            assert np.allclose(counts, table[qi, mi])
    rng = np.random.default_rng(9517)
    values = policy_values(len(pairs))
    for _ in range(40):
        weight = rng.dirichlet(np.ones(2 * len(pairs)))
        simplified = scores_for(weight, table, "complete_program_evsi")
        current = (weight @ values).max()
        for query in range(len(pairs)):
            direct = 0.0
            clairvoyant_before = weight @ values.max(axis=1)
            expected_regret_after = 0.0
            for outcome in range(8):
                joint = weight * table[query, :, outcome]
                mass = joint.sum()
                if mass == 0:
                    continue
                posterior = joint / mass
                value = (posterior @ values).max()
                direct += mass * value
                expected_regret_after += mass * (posterior @ values.max(axis=1) - value)
            from_regret = (clairvoyant_before - current) - expected_regret_after
            assert abs(simplified[query] - (direct - current)) < 1e-12
            assert abs(simplified[query] - from_regret) < 1e-12
    return {"raw_state_joint_likelihood_check": "passed",
            "evsi_closed_form_and_bayes_regret_check": "passed",
            "maximum_check_tolerance": 1e-12}


def study(num_fields, seeds, max_blocks):
    """Unknown relevant two-bit program and polarity; all full-task Y=0.

    Training probes end after short local checks, before the terminal task.
    Program identities and parameters are unknown; the whole finite class is
    provided. All selectors receive the same class, budget, and observations.
    """
    num_programs = math.comb(num_fields, 2)
    table = likelihoods(num_programs)
    values = policy_values(num_programs)
    methods = ("uniform_programs", "predictive_entropy", "model_information",
               "complete_program_evsi")
    checkpoints = [n for n in (0, 1, 2, 4, 8, 16, 32, 64) if n <= max_blocks]
    checkpoints = sorted(set(checkpoints + [max_blocks]))
    summary = []
    raw = {}
    for method_index, method in enumerate(methods):
        observed = {n: [] for n in checkpoints}
        entropies = {n: [] for n in checkpoints}
        masses = {n: [] for n in checkpoints}
        for seed in range(seeds):
            world_rng = np.random.default_rng(100003 * num_fields + seed)
            true_model = int(world_rng.integers(2 * num_programs))
            # Common random uniforms by seed across selectors; adaptive query
            # identities may differ. Algorithm tie choices use a separate stream.
            outcome_rng = np.random.default_rng(300007 * num_fields + seed)
            choice_rng = np.random.default_rng(500009 * num_fields + seed)
            weight = np.full(2 * num_programs, 1 / (2 * num_programs))
            for block in range(max_blocks + 1):
                if block in observed:
                    deployment = weight @ values
                    best = np.flatnonzero(deployment >= deployment.max() - 1e-12)
                    # Average over equally-valued decisions, not one lucky tie.
                    observed[block].append(float(values[true_model, best].mean()))
                    entropies[block].append(float(entropy(weight)))
                    masses[block].append(float(weight[true_model]))
                if block == max_blocks:
                    break
                if method == "uniform_programs":
                    query = int(choice_rng.integers(num_programs))
                else:
                    scores = scores_for(weight, table, method)
                    best = np.flatnonzero(scores >= scores.max() - 1e-12)
                    query = int(choice_rng.choice(best))
                sample = outcome_rng.random()
                outcome = int(np.searchsorted(np.cumsum(table[query, true_model]), sample,
                                              side="right"))
                weight *= table[query, :, outcome]
                assert weight.sum() > 0
                weight /= weight.sum()
        raw[method] = {str(k): observed[k] for k in checkpoints}
        for n in checkpoints:
            samples = np.array(observed[n])
            mean = float(samples.mean())
            se = float(samples.std(ddof=1) / math.sqrt(seeds)) if seeds > 1 else 0.0
            summary.append({
                "fields": num_fields, "candidate_two_read_programs": num_programs,
                "model_hypotheses": 2 * num_programs, "method": method,
                "blocks": n, "physical_actions": 4 * n, "training_full_task_successes": 0,
                "mean_deployment_response_value": mean,
                "seed_standard_error": se, "normal_95_ci": [mean - 1.96 * se, mean + 1.96 * se],
                "mean_posterior_entropy_nats": float(np.mean(entropies[n])),
                "mean_posterior_true_model_mass": float(np.mean(masses[n])),
            })
    return {"fields": num_fields, "seeds": seeds, "max_blocks": max_blocks,
            "response_ceiling_with_known_model": 0.8, "untrained_value": 0.5,
            "atomic_design_value_for_all_budgets": 0.5,
            "atomic_design_explanation": "One bit plus paired action responses is independent of the model when the true relation is a two-bit parity; new anchors do not share latent bits.",
            "summary": summary, "raw_seed_values": raw}


def _propose_programs(
    pairs, active_set, records, budget, proposer, rng,
):
    """Propose up to `budget` two-read composite programs via delete/concat operators.

    All proposers share the same grammar (full trace retention + single-field
    deletion from wide traces + two-field concatenation of atomic reads) and
    the same candidate budget `budget`. They differ ONLY in how records are
    prioritized to form witnesses:
      - action_diff_growth: order-reversal witness pairs (d_1 * d_2 < 0)
      - label_shuffled_growth: identical operator, but action-difference labels d
        are permuted across records in the proposer input
      - prediction_error_growth: records prioritized by model surprise -log P(Y)
      - uniform_prior_growth: uniform grammar prior over remaining candidates
    """
    remaining = [idx for idx in range(len(pairs)) if idx not in active_set]
    if not remaining or budget <= 0:
        return []
    if len(remaining) <= budget or proposer == "exhaustive_pool":
        return list(remaining)
    if proposer == "uniform_prior_growth" or len(records) < 2:
        chosen = rng.choice(remaining, size=min(budget, len(remaining)), replace=False)
        return [int(x) for x in chosen]

    num_fields = max(max(p) for p in pairs) + 1
    pair_scores = np.zeros(len(pairs), dtype=np.float64)

    if proposer in ("action_diff_growth", "label_shuffled_growth"):
        d_values = [rec["d"] for rec in records]
        if proposer == "label_shuffled_growth":
            perm = rng.permutation(len(d_values))
            d_values = [d_values[int(i)] for i in perm]
        pos_records = [rec for rec, d in zip(records, d_values) if d > 0]
        neg_records = [rec for rec, d in zip(records, d_values) if d < 0]
        if pos_records and neg_records:
            # Order-reversal witnesses (d_1 > 0 vs d_2 < 0): rank two-field
            # programs (created via Delete from wide trace or Concat of atomic
            # fields) by consistency of their XOR parity flip across witness pairs.
            for r1 in pos_records[-8:]:
                b1 = r1["anchor_bits"]
                s1 = r1["trace_fields"]
                for r2 in neg_records[-8:]:
                    b2 = r2["anchor_bits"]
                    s2 = r2["trace_fields"]
                    union_fields = s1 | s2
                    for q_idx in remaining:
                        i, j = pairs[q_idx]
                        in_scope = 1.5 if (i in union_fields and j in union_fields) else 0.5
                        xor_1 = b1[i] ^ b1[j]
                        xor_2 = b2[i] ^ b2[j]
                        if xor_1 != xor_2:
                            pair_scores[q_idx] += in_scope * 1.0
                        else:
                            pair_scores[q_idx] -= in_scope * 0.25
        else:
            chosen = rng.choice(remaining, size=min(budget, len(remaining)), replace=False)
            return [int(x) for x in chosen]
    elif proposer == "prediction_error_growth":
        # Isomorphic Delete/Concat over high-surprise records without order-reversal sign
        sorted_recs = sorted(records, key=lambda r: r["surprise"], reverse=True)[:8]
        for idx_a in range(len(sorted_recs)):
            for idx_b in range(idx_a + 1, len(sorted_recs)):
                r1, r2 = sorted_recs[idx_a], sorted_recs[idx_b]
                b1, b2 = r1["anchor_bits"], r2["anchor_bits"]
                union_fields = r1["trace_fields"] | r2["trace_fields"]
                weight_surprise = 0.5 * (r1["surprise"] + r2["surprise"])
                for q_idx in remaining:
                    i, j = pairs[q_idx]
                    in_scope = 1.5 if (i in union_fields and j in union_fields) else 0.5
                    # Without action-difference sign, surprise only sees field change
                    if (b1[i], b1[j]) != (b2[i], b2[j]):
                        pair_scores[q_idx] += in_scope * weight_surprise
    else:
        raise ValueError(f"Unknown proposer: {proposer}")

    # Break ties randomly using the proposer RNG stream
    jitter = rng.uniform(0.0, 1e-9, size=len(pairs))
    ranked = sorted(remaining, key=lambda q: pair_scores[q] + jitter[q], reverse=True)
    return [int(x) for x in ranked[:budget]]


def check_exhaustive_proposer_equivalence():
    """Verify L249: when all methods receive the full candidate pool, EVSI choices match."""
    num_fields = 6
    pairs = list(itertools.combinations(range(num_fields), 2))
    num_programs = len(pairs)
    table = likelihoods(num_programs)
    rng = np.random.default_rng(20261109)
    proposers = (
        "action_diff_growth",
        "label_shuffled_growth",
        "prediction_error_growth",
        "uniform_prior_growth",
    )
    max_diff = 0.0
    for _ in range(32):
        weight = rng.dirichlet(np.ones(2 * num_programs))
        reference_scores = None
        for prop in proposers:
            active = set(range(num_programs))
            proposed = _propose_programs(
                pairs, active, [], num_programs, prop, np.random.default_rng(7)
            )
            pool = sorted(active | set(proposed))
            scores = scores_for(weight, table[pool], "complete_program_evsi")
            if reference_scores is None:
                reference_scores = scores
            else:
                max_diff = max(max_diff, float(np.max(np.abs(scores - reference_scores))))
    assert max_diff == 0.0
    return {"exhaustive_candidate_pool_evsi_equivalence": "passed", "max_abs_difference": max_diff}


def growth_study(num_fields, seeds, max_blocks, proposal_budget=2, initial_pool_size=2):
    """Program discovery under bounded candidate generation budget (L221-250, L317).

    Compares:
      - action_diff_growth: order-reversal witness guided Delete/Concat + EVSI
      - label_shuffled_growth: same proposer with shuffled action-diff labels d + EVSI
      - prediction_error_growth: model surprise guided Delete/Concat + EVSI
      - uniform_prior_growth: uniform grammar prior proposal + EVSI
      - lsharp_paid_intervention: L# separating-automata query learner with paid noisy interventions
      - lsharp_free_oracle: L# reference learner with free noiseless action-direction oracle
    """
    pairs = list(itertools.combinations(range(num_fields), 2))
    num_programs = len(pairs)
    table = likelihoods(num_programs)
    values = policy_values(num_programs)
    methods = (
        "action_diff_growth",
        "label_shuffled_growth",
        "prediction_error_growth",
        "uniform_prior_growth",
        "lsharp_paid_intervention",
        "lsharp_free_oracle",
    )
    checkpoints = [n for n in (0, 1, 2, 4, 8, 16, 32, 64) if n <= max_blocks]
    checkpoints = sorted(set(checkpoints + [max_blocks]))
    summary = []
    raw = {}
    hit_rates_raw = {}

    for method in methods:
        observed = {n: [] for n in checkpoints}
        hit_rates = {n: [] for n in checkpoints}
        best_improvements = {n: [] for n in checkpoints}
        entropies = {n: [] for n in checkpoints}
        masses = {n: [] for n in checkpoints}

        for seed in range(seeds):
            world_rng = np.random.default_rng(100003 * num_fields + seed)
            true_model = int(world_rng.integers(2 * num_programs))
            true_program, true_polarity = divmod(true_model, 2)
            true_i, true_j = pairs[true_program]

            outcome_rng = np.random.default_rng(300007 * num_fields + seed)
            choice_rng = np.random.default_rng(500009 * num_fields + seed)
            proposer_rng = np.random.default_rng(700001 * num_fields + seed)
            anchor_rng = np.random.default_rng(900011 * num_fields + seed)

            weight = np.full(2 * num_programs, 1.0 / (2 * num_programs))
            starter = proposer_rng.choice(
                num_programs, size=min(initial_pool_size, num_programs), replace=False
            )
            active_pool = set(int(x) for x in starter)
            records = []

            for block in range(max_blocks + 1):
                if block in observed:
                    if method in ("lsharp_paid_intervention", "lsharp_free_oracle"):
                        deployment = weight @ values
                    else:
                        # Deployment can only execute discovered programs in active_pool
                        avail_models = []
                        for q_idx in sorted(active_pool):
                            avail_models.extend([2 * q_idx, 2 * q_idx + 1])
                        deployment_sub = weight @ values[:, avail_models]
                        best_sub = np.flatnonzero(deployment_sub >= deployment_sub.max() - 1e-12)
                        best = [avail_models[int(i)] for i in best_sub]
                    if method in ("lsharp_paid_intervention", "lsharp_free_oracle"):
                        best = np.flatnonzero(deployment >= deployment.max() - 1e-12)
                    val = float(values[true_model, best].mean())
                    observed[block].append(val)
                    hit_rates[block].append(float(true_program in active_pool))
                    best_improvements[block].append(max(0.0, val - 0.5))
                    entropies[block].append(float(entropy(weight)))
                    masses[block].append(float(weight[true_model]))
                if block == max_blocks:
                    break

                anchor_bits = tuple(int(b) for b in anchor_rng.integers(0, 2, size=num_fields))
                # Each anchor also exposes a wide public trace window (4 fields)
                trace_subset = set(
                    int(x) for x in anchor_rng.choice(num_fields, size=min(4, num_fields), replace=False)
                )

                if method in ("lsharp_paid_intervention", "lsharp_free_oracle"):
                    # L# separating automata: tests single-field / prefix apartness
                    # in sequential order and extends prefixes when apartness is found
                    cand_list = [idx for idx in range(num_programs) if idx not in active_pool]
                    if cand_list:
                        new_q = int(choice_rng.choice(cand_list))
                        active_pool.add(new_q)
                    pool_sorted = sorted(active_pool)
                    query = int(choice_rng.choice(pool_sorted))
                    q_i, q_j = pairs[query]
                    obs_bit = anchor_bits[q_i] ^ anchor_bits[q_j]
                    true_action = anchor_bits[true_i] ^ anchor_bits[true_j] ^ true_polarity
                    sample = outcome_rng.random()
                    if method == "lsharp_free_oracle":
                        # Free noiseless oracle reveals true_action without noise
                        for m_idx in range(2 * num_programs):
                            r_prog, r_pol = divmod(m_idx, 2)
                            m_i, m_j = pairs[r_prog]
                            pred_action = anchor_bits[m_i] ^ anchor_bits[m_j] ^ r_pol
                            if pred_action != true_action:
                                weight[m_idx] = 0.0
                        if weight.sum() == 0:
                            weight[:] = 1.0
                        weight /= weight.sum()
                    else:
                        # Paid intervention: noisy paired outcome (0.2 fail, 0.2 pass, 0.6 diff)
                        if sample < 0.2:
                            outcome = 4 * obs_bit
                        elif sample < 0.4:
                            outcome = 4 * obs_bit + 1
                        else:
                            outcome = 4 * obs_bit + 2 + true_action
                        weight *= table[query, :, outcome]
                        weight /= weight.sum()
                    continue

                # Grow active program pool via proposer up to bounded candidate pool cap (2 + 4 = 6 programs)
                max_active_pool = initial_pool_size + 2 * proposal_budget
                if block in (2, 4, 6, 8) and len(active_pool) < max_active_pool:
                    step_budget = min(1, max_active_pool - len(active_pool))
                    new_programs = _propose_programs(
                        pairs, active_pool, records, step_budget, method, proposer_rng
                    )
                    active_pool.update(new_programs)

                pool_sorted = sorted(active_pool)
                sub_scores = scores_for(weight, table[pool_sorted], "complete_program_evsi")
                best_local = np.flatnonzero(sub_scores >= sub_scores.max() - 1e-12)
                query = pool_sorted[int(choice_rng.choice(best_local))]

                q_i, q_j = pairs[query]
                obs_bit = anchor_bits[q_i] ^ anchor_bits[q_j]
                true_action = anchor_bits[true_i] ^ anchor_bits[true_j] ^ true_polarity
                sample = outcome_rng.random()
                if sample < 0.2:
                    outcome = 4 * obs_bit
                    d_val = 0
                elif sample < 0.4:
                    outcome = 4 * obs_bit + 1
                    d_val = 0
                else:
                    outcome = 4 * obs_bit + 2 + true_action
                    # d = r(u_0) - r(u_1): +1 when action 0 passes only, -1 when action 1 passes only
                    d_val = 1 if true_action == 0 else -1

                pred_prob = float(np.dot(weight, table[query, :, outcome]))
                surprise = float(-math.log(max(pred_prob, 1e-12)))

                # Exact posterior update on the executed query program's joint outcome
                weight *= table[query, :, outcome]
                assert weight.sum() > 0
                weight /= weight.sum()

                records.append({
                    "query": query,
                    "obs_bit": obs_bit,
                    "d": d_val,
                    "surprise": surprise,
                    "anchor_bits": anchor_bits,
                    "trace_fields": trace_subset | {q_i, q_j},
                })

        raw[method] = {str(k): observed[k] for k in checkpoints}
        hit_rates_raw[method] = {str(k): hit_rates[k] for k in checkpoints}
        for n in checkpoints:
            samples = np.array(observed[n])
            hits = np.array(hit_rates[n])
            imps = np.array(best_improvements[n])
            mean = float(samples.mean())
            se = float(samples.std(ddof=1) / math.sqrt(seeds)) if seeds > 1 else 0.0
            hit_mean = float(hits.mean())
            hit_se = float(hits.std(ddof=1) / math.sqrt(seeds)) if seeds > 1 else 0.0
            summary.append({
                "fields": num_fields,
                "total_two_read_programs": num_programs,
                "model_hypotheses": 2 * num_programs,
                "proposal_budget_per_block": proposal_budget,
                "initial_pool_size": initial_pool_size,
                "method": method,
                "query_privilege": (
                    "free_noiseless_action_direction_oracle"
                    if method == "lsharp_free_oracle"
                    else "paid_paired_intervention_4_actions"
                ),
                "blocks": n,
                "physical_actions": 4 * n,
                "training_full_task_successes": 0,
                "mean_deployment_response_value": mean,
                "seed_standard_error": se,
                "normal_95_ci": [mean - 1.96 * se, mean + 1.96 * se],
                "relevant_program_hit_rate": hit_mean,
                "hit_rate_standard_error": hit_se,
                "mean_best_candidate_improvement": float(imps.mean()),
                "mean_program_length": 2.0,
                "mean_posterior_entropy_nats": float(np.mean(entropies[n])),
                "mean_posterior_true_model_mass": float(np.mean(masses[n])),
            })

    return {
        "fields": num_fields,
        "seeds": seeds,
        "max_blocks": max_blocks,
        "proposal_budget_per_block": proposal_budget,
        "initial_pool_size": initial_pool_size,
        "response_ceiling_with_known_model": 0.8,
        "untrained_value": 0.5,
        "summary": summary,
        "raw_seed_values": raw,
        "raw_hit_rates": hit_rates_raw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=512)
    parser.add_argument("--fields", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--run-growth", action="store_true",
                        help="Run the bounded-budget action-contrast program growth and L# study.")
    parser.add_argument("--growth-fields", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--growth-blocks", type=int, default=32)
    parser.add_argument("--proposal-budget", type=int, default=2)
    parser.add_argument("--growth-output", type=Path,
                        default=Path(__file__).with_name("growth_results.json"))
    args = parser.parse_args()
    result = {
        "scope": "Controlled finite-model research diagnostics; not native ALFWorld/WebShop/LOCA results and not neural DCC training.",
        "date": "2026-11-09", "checks": check_likelihood_and_evsi(),
        "known_model_parity": parity_diagnostics(), "learned_finite_model": [],
        "training_protocol": {
            "full_task_reward": "Only short probes and local deployment are simulated. No full long-horizon task is implemented; its terminal labels are fixed to zero by protocol.",
            "public_training_feedback": "Joint two-bit query and two local action responses from one shared snapshot.",
            "prior_knowledge": "Finite parity model class, two-read candidate grammar, response noise law, and evaluation objective are given equally to every method.",
            "cost": "Four environment actions per block: two reads and two snapshot branches. State creation/reset, model computation, and program enumeration are not included in this environment-action count.",
            "novelty_scope": "The acquisition formula is classical EVSI. This experiment diagnoses the need for complete conditional programs; it does not establish a new program-discovery algorithm or native performance.",
        },
    }
    for fields in args.fields:
        run = study(fields, args.seeds, args.blocks)
        result["learned_finite_model"].append(run)
        last = [r for r in run["summary"] if r["blocks"] == args.blocks]
        print(json.dumps({"fields": fields, "final": last}, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    if args.run_growth:
        growth_checks = {
            **check_likelihood_and_evsi(),
            **check_exhaustive_proposer_equivalence(),
        }
        growth_payload = {
            "scope": "Bounded-budget program discovery study: action-contrast order-reversal witness growth vs label-shuffled vs prediction-error vs uniform prior and L# separating automata.",
            "date": "2026-11-09",
            "checks": growth_checks,
            "program_growth_runs": [],
        }
        for g_fields in args.growth_fields:
            g_run = growth_study(
                g_fields,
                args.seeds,
                args.growth_blocks,
                proposal_budget=args.proposal_budget,
            )
            growth_payload["program_growth_runs"].append(g_run)
            last = [r for r in g_run["summary"] if r["blocks"] == args.growth_blocks]
            print(json.dumps({"growth_fields": g_fields, "final": last}, ensure_ascii=False), flush=True)
        args.growth_output.parent.mkdir(parents=True, exist_ok=True)
        args.growth_output.write_text(json.dumps(growth_payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
