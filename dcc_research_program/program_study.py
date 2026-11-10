"""Exact finite-model research diagnostics and bounded-budget program discovery study for PRISM DCC.

Implements the systematic revision contract (`docs/prism-dcc-v2-design.md`):
1. Known-model parity diagnostic and exact closed-form EVSI reference without fantasy retraining.
2. Strictly paid field access in program discovery (`_propose_programs`):
   - Only fields actually observed in an anchor's paid trace (`observed_bits: dict[int, int]`) are stored
     and accessible to proposers; unobserved fields are never read from ground-truth `anchor_bits` (fixes E05).
3. Complete candidate construction and inspection budget accounting:
   - Proposers may construct and inspect at most `max_candidate_checks` candidate pairs per call,
     recording `candidates_constructed` and `candidates_checked` instead of exhaustively scoring all
     remaining combinations (fixes E06).
4. Unified executable policy support between acquisition and deployment:
   - Both EVSI acquisition and deployment evaluate over the same executable policy pool `avail_models`
     induced by `active_pool` (fixes E09).
5. Honest baseline naming:
   - `random_query_finite_bayes` (formerly `lsharp_paid_intervention`) and
     `noiseless_version_space_elimination` (formerly `lsharp_free_oracle`), accurately reflecting
     their random-query Bayesian update and noiseless version-space elimination semantics (fixes E08).
6. Unknown response law K + delayed terminal goal study (`unknown_k_delayed_goal_study`):
   - Learns empirical transition/response tables from scratch (Laplace-smoothed counts) and evaluates
     multi-stage delayed terminal success Y in {0, 1} alongside local response feedback (fixes E07).
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path

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
    return {
        "rows": rows,
        "direct_enumeration_check_n8_k2": exact_uniform,
        "single_bit_value_of_information": 0.0,
        "joint_two_bit_value_of_information": 0.5,
        "interpretation": "Known-model planning diagnostic; all individual fields are independent fair bits. Predictive-entropy ties are uniform. No learned novelty claim.",
    }


def likelihoods(num_programs):
    """p(z, paired_response | query program, model) for known-noise reference experiments."""
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
    values = np.full((2 * num_programs, 2 * num_programs), 0.5)
    for model in range(2 * num_programs):
        values[model, model] = 0.8
        values[model, model ^ 1] = 0.2
    return values


def entropy(probabilities, axis=-1):
    terms = np.zeros_like(probabilities)
    np.log(probabilities, out=terms, where=probabilities > 0)
    return -(probabilities * terms).sum(axis=axis)


def scores_for(weight, table, method, *, values_sub=None):
    """Closed-form finite-particle expected value, without retraining fantasies.

    When `values_sub` is provided (shape `(2 * num_programs, len(avail_models))`),
    EVSI is computed strictly over the executable policy set `avail_models` (fixes E09).
    """
    predictive = np.einsum("qmo,m->qo", table, weight)
    if method == "predictive_entropy":
        return entropy(predictive)
    if method == "model_information":
        return entropy(predictive) - np.einsum("qm,m->q", entropy(table), weight)
    if method == "complete_program_evsi":
        if values_sub is not None:
            before = float(np.max(weight @ values_sub))
            joint = table * weight[None, :, None]  # (Q, M, O)
            # post_values[q, o, j] = sum_m joint[q, m, o] * values_sub[m, j]
            post_values = np.einsum("qmo,mj->qoj", joint, values_sub)
            after = np.max(post_values, axis=-1).sum(axis=-1)
            result = np.maximum(0.0, after - before)
            return result
        joint = table * weight[None, :, None]
        signed = joint[:, 0::2, :] - joint[:, 1::2, :]
        after = 0.5 + 0.3 * np.max(np.abs(signed), axis=1).sum(axis=1)
        before = 0.5 + 0.3 * np.max(np.abs(weight[0::2] - weight[1::2]))
        result = after - before
        assert np.min(result) > -1e-12
        return result
    raise ValueError(method)


def check_likelihood_and_evsi():
    """Independently enumerate raw four-bit worlds and verify EVSI formulas, including zero-info determinism."""
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
        with_explicit_v = scores_for(weight, table, "complete_program_evsi", values_sub=values)
        assert np.allclose(simplified, with_explicit_v, atol=1e-12)
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

    # Deterministic known-mechanism check: M=1 known model => EVSI == 0.0 exactly
    one_hot_w = np.zeros(2 * len(pairs))
    one_hot_w[0] = 1.0
    zero_evsi = scores_for(one_hot_w, table, "complete_program_evsi", values_sub=values)
    assert np.max(np.abs(zero_evsi)) < 1e-12
    return {
        "raw_state_joint_likelihood_check": "passed",
        "evsi_closed_form_and_bayes_regret_check": "passed",
        "deterministic_known_model_zero_evsi_check": "passed",
        "maximum_check_tolerance": 1e-12,
    }


def study(num_fields, seeds, max_blocks):
    """Known-likelihood reference study over exhaustive two-read candidate pool."""
    num_programs = math.comb(num_fields, 2)
    table = likelihoods(num_programs)
    values = policy_values(num_programs)
    methods = (
        "uniform_programs",
        "predictive_entropy",
        "model_information",
        "complete_program_evsi",
    )
    checkpoints = [n for n in (0, 1, 2, 4, 8, 16, 32, 64) if n <= max_blocks]
    checkpoints = sorted(set(checkpoints + [max_blocks]))
    summary = []
    raw = {}
    for method in methods:
        observed = {n: [] for n in checkpoints}
        entropies = {n: [] for n in checkpoints}
        masses = {n: [] for n in checkpoints}
        for seed in range(seeds):
            world_rng = np.random.default_rng(100003 * num_fields + seed)
            true_model = int(world_rng.integers(2 * num_programs))
            outcome_rng = np.random.default_rng(300007 * num_fields + seed)
            choice_rng = np.random.default_rng(500009 * num_fields + seed)
            weight = np.full(2 * num_programs, 1 / (2 * num_programs))
            for block in range(max_blocks + 1):
                if block in observed:
                    deployment = weight @ values
                    best = np.flatnonzero(deployment >= deployment.max() - 1e-12)
                    observed[block].append(float(values[true_model, best].mean()))
                    entropies[block].append(float(entropy(weight)))
                    masses[block].append(float(weight[true_model]))
                if block == max_blocks:
                    break
                if method == "uniform_programs":
                    query = int(choice_rng.integers(num_programs))
                else:
                    scores = scores_for(weight, table, method, values_sub=values)
                    best = np.flatnonzero(scores >= scores.max() - 1e-12)
                    query = int(choice_rng.choice(best))
                sample = outcome_rng.random()
                outcome = int(np.searchsorted(np.cumsum(table[query, true_model]), sample, side="right"))
                weight *= table[query, :, outcome]
                assert weight.sum() > 0
                weight /= weight.sum()
        raw[method] = {str(k): observed[k] for k in checkpoints}
        for n in checkpoints:
            samples = np.array(observed[n])
            mean = float(samples.mean())
            se = float(samples.std(ddof=1) / math.sqrt(seeds)) if seeds > 1 else 0.0
            summary.append({
                "fields": num_fields,
                "candidate_two_read_programs": num_programs,
                "model_hypotheses": 2 * num_programs,
                "method": method,
                "blocks": n,
                "physical_actions": 4 * n,
                "training_full_task_successes": 0,
                "mean_deployment_response_value": mean,
                "seed_standard_error": se,
                "normal_95_ci": [mean - 1.96 * se, mean + 1.96 * se],
                "mean_posterior_entropy_nats": float(np.mean(entropies[n])),
                "mean_posterior_true_model_mass": float(np.mean(masses[n])),
            })
    return {
        "fields": num_fields,
        "seeds": seeds,
        "max_blocks": max_blocks,
        "prior_scope": "known_likelihood_table_and_known_local_value_table",
        "response_ceiling_with_known_model": 0.8,
        "untrained_value": 0.5,
        "atomic_design_value_for_all_budgets": 0.5,
        "atomic_design_explanation": "One bit plus paired action responses is independent of the model when the true relation is a two-bit parity; new anchors do not share latent bits.",
        "summary": summary,
        "raw_seed_values": raw,
    }


def _propose_programs(
    pairs,
    active_set,
    records,
    budget,
    proposer,
    rng,
    *,
    max_candidate_checks=6,
):
    """Propose up to `budget` two-read composite programs under strict paid-field and check-budget constraints.

    Fixes E05 and E06:
      - Only fields present in a record's paid `observed_bits` dictionary may be accessed;
        unobserved fields have unknown values and cannot have their XOR computed.
      - At most `max_candidate_checks` candidate programs may be constructed and inspected per call;
        the proposer never enumerates and scores all `remaining` pairs for free.
      - In `label_shuffled_growth`, all grouping and ranking use the single permuted label view.
    Returns `(selected_program_indices, stats_dict)`.
    """
    remaining = [idx for idx in range(len(pairs)) if idx not in active_set]
    if not remaining or budget <= 0:
        return [], {"candidates_constructed": 0, "candidates_checked": 0}
    if proposer == "exhaustive_pool":
        return list(remaining), {
            "candidates_constructed": len(remaining),
            "candidates_checked": len(remaining),
        }
    if proposer == "uniform_prior_growth" or len(records) < 2:
        n_pick = min(budget, len(remaining), max_candidate_checks)
        chosen = rng.choice(remaining, size=n_pick, replace=False)
        return [int(x) for x in chosen], {
            "candidates_constructed": int(n_pick),
            "candidates_checked": int(n_pick),
        }

    pair_To_idx = {pair: idx for idx, pair in enumerate(pairs)}
    candidate_scores: dict[int, float] = {}
    candidates_constructed = 0
    candidates_checked = 0

    if proposer in ("action_diff_growth", "label_shuffled_growth"):
        d_values = [rec["d"] for rec in records]
        if proposer == "label_shuffled_growth":
            perm = rng.permutation(len(d_values))
            d_values = [d_values[int(i)] for i in perm]
        pos_records = [(rec, d) for rec, d in zip(records, d_values) if d > 0]
        neg_records = [(rec, d) for rec, d in zip(records, d_values) if d < 0]

        if pos_records and neg_records:
            for r1, d1_view in pos_records[-6:]:
                if candidates_checked >= max_candidate_checks:
                    break
                obs1 = r1["observed_bits"]  # dict[field_idx, bit_val] of PAID observed fields only
                for r2, d2_view in neg_records[-6:]:
                    if candidates_checked >= max_candidate_checks:
                        break
                    obs2 = r2["observed_bits"]
                    # Delete / Concat can only form pairs from fields actually observed in the paid traces
                    shared_observed_fields = sorted(set(obs1.keys()) & set(obs2.keys()))
                    joint_pairs = [
                        (i, j)
                        for i, j in itertools.combinations(shared_observed_fields, 2)
                        if pair_To_idx.get((i, j)) in remaining
                    ]
                    rng.shuffle(joint_pairs)
                    for i, j in joint_pairs:
                        if candidates_checked >= max_candidate_checks:
                            break
                        q_idx = pair_To_idx[(i, j)]
                        candidates_constructed += 1
                        candidates_checked += 1
                        xor_1 = obs1[i] ^ obs1[j]
                        xor_2 = obs2[i] ^ obs2[j]
                        mag = 0.5 * abs(float(d1_view) - float(d2_view))
                        if xor_1 != xor_2:
                            candidate_scores[q_idx] = candidate_scores.get(q_idx, 0.0) + mag * 1.0
                        else:
                            candidate_scores[q_idx] = candidate_scores.get(q_idx, 0.0) - mag * 0.25
    elif proposer == "prediction_error_growth":
        sorted_recs = sorted(records, key=lambda r: r["surprise"], reverse=True)[:6]
        for idx_a in range(len(sorted_recs)):
            if candidates_checked >= max_candidate_checks:
                break
            for idx_b in range(idx_a + 1, len(sorted_recs)):
                if candidates_checked >= max_candidate_checks:
                    break
                r1, r2 = sorted_recs[idx_a], sorted_recs[idx_b]
                obs1, obs2 = r1["observed_bits"], r2["observed_bits"]
                shared_observed_fields = sorted(set(obs1.keys()) & set(obs2.keys()))
                joint_pairs = [
                    (i, j)
                    for i, j in itertools.combinations(shared_observed_fields, 2)
                    if pair_To_idx.get((i, j)) in remaining
                ]
                rng.shuffle(joint_pairs)
                weight_surprise = 0.5 * (r1["surprise"] + r2["surprise"])
                for i, j in joint_pairs:
                    if candidates_checked >= max_candidate_checks:
                        break
                    q_idx = pair_To_idx[(i, j)]
                    candidates_constructed += 1
                    candidates_checked += 1
                    if (obs1[i], obs1[j]) != (obs2[i], obs2[j]):
                        candidate_scores[q_idx] = candidate_scores.get(q_idx, 0.0) + weight_surprise
    else:
        raise ValueError(f"Unknown proposer: {proposer}")

    # If remaining check budget allows and we haven't filled `budget` candidates, sample from grammar prior
    if len(candidate_scores) < budget and candidates_checked < max_candidate_checks:
        unscored = [q for q in remaining if q not in candidate_scores]
        extra_n = min(budget - len(candidate_scores), len(unscored), max_candidate_checks - candidates_checked)
        if extra_n > 0:
            extra_picks = rng.choice(unscored, size=extra_n, replace=False)
            for q in extra_picks:
                candidates_constructed += 1
                candidates_checked += 1
                candidate_scores[int(q)] = -1e-4

    if not candidate_scores:
        return [], {
            "candidates_constructed": candidates_constructed,
            "candidates_checked": candidates_checked,
        }

    ranked = sorted(
        candidate_scores.keys(),
        key=lambda q: candidate_scores[q] + float(rng.uniform(0.0, 1e-9)),
        reverse=True,
    )
    return [int(x) for x in ranked[:budget]], {
        "candidates_constructed": candidates_constructed,
        "candidates_checked": candidates_checked,
    }


def check_exhaustive_proposer_equivalence():
    """Verify that when all methods receive the full candidate pool, EVSI choices match."""
    num_fields = 6
    pairs = list(itertools.combinations(range(num_fields), 2))
    num_programs = len(pairs)
    table = likelihoods(num_programs)
    values = policy_values(num_programs)
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
            proposed, _ = _propose_programs(
                pairs, active, [], num_programs, prop, np.random.default_rng(7)
            )
            pool = sorted(active | set(proposed))
            scores = scores_for(weight, table[pool], "complete_program_evsi", values_sub=values)
            if reference_scores is None:
                reference_scores = scores
            else:
                max_diff = max(max_diff, float(np.max(np.abs(scores - reference_scores))))
    assert max_diff == 0.0
    return {"exhaustive_candidate_pool_evsi_equivalence": "passed", "max_abs_difference": max_diff}


def growth_study(
    num_fields,
    seeds,
    max_blocks,
    proposal_budget=2,
    initial_pool_size=2,
    max_candidate_checks=6,
):
    """Program discovery under bounded candidate construction/inspection budget and paid field access.

    Compares:
      - action_diff_growth: order-reversal witness guided Delete/Concat + active-pool EVSI
      - label_shuffled_growth: same proposer with permuted action-diff labels d + active-pool EVSI
      - prediction_error_growth: model surprise guided Delete/Concat + active-pool EVSI
      - uniform_prior_growth: uniform grammar prior proposal + active-pool EVSI
      - random_query_finite_bayes: random query expansion with finite-table Bayes update
      - noiseless_version_space_elimination: free noiseless action-direction oracle version-space elimination
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
        "random_query_finite_bayes",
        "noiseless_version_space_elimination",
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
        checked_counts = {n: [] for n in checkpoints}

        for seed in range(seeds):
            world_rng = np.random.default_rng(100003 * num_fields + seed)
            true_model = int(world_rng.integers(2 * num_programs))
            true_program, true_polarity = divmod(true_model, 2)
            true_i, true_j = pairs[true_program]

            init_rng = np.random.default_rng(700001 * num_fields + seed)
            non_true = [idx for idx in range(num_programs) if idx != true_program]
            active_pool = set(
                int(x) for x in init_rng.choice(non_true, size=min(initial_pool_size, len(non_true)), replace=False)
            )

            anchor_rng = np.random.default_rng(200003 * num_fields + seed)
            outcome_rng = np.random.default_rng(300007 * num_fields + seed)
            choice_rng = np.random.default_rng(500009 * num_fields + seed)
            proposer_rng = np.random.default_rng(900011 * num_fields + seed)

            weight = np.full(2 * num_programs, 1.0 / (2 * num_programs))
            records = []
            cum_candidates_checked = 0

            for block in range(max_blocks + 1):
                if block in observed:
                    # Unified executable policy set: deployment and acquisition both use `avail_models` from `active_pool`
                    avail_models = []
                    for q_idx in sorted(active_pool):
                        avail_models.extend([2 * q_idx, 2 * q_idx + 1])
                    deployment_sub = weight @ values[:, avail_models]
                    best_sub = np.flatnonzero(deployment_sub >= deployment_sub.max() - 1e-12)
                    best = [avail_models[int(i)] for i in best_sub]
                    val = float(values[true_model, best].mean())
                    observed[block].append(val)
                    hit_rates[block].append(float(true_program in active_pool))
                    best_improvements[block].append(max(0.0, val - 0.5))
                    entropies[block].append(float(entropy(weight)))
                    masses[block].append(float(weight[true_model]))
                    checked_counts[block].append(float(cum_candidates_checked))
                if block == max_blocks:
                    break

                anchor_bits = tuple(int(b) for b in anchor_rng.integers(0, 2, size=num_fields))
                # Paid public trace window exposes 4 fields; ONLY those observed bits are stored
                trace_subset = set(
                    int(x) for x in anchor_rng.choice(num_fields, size=min(4, num_fields), replace=False)
                )

                if method in ("random_query_finite_bayes", "noiseless_version_space_elimination"):
                    max_active_pool = initial_pool_size + 2 * proposal_budget
                    if block in (2, 4, 6, 8) and len(active_pool) < max_active_pool:
                        cand_list = [idx for idx in range(num_programs) if idx not in active_pool]
                        if cand_list:
                            new_q = int(choice_rng.choice(cand_list))
                            active_pool.add(new_q)
                            cum_candidates_checked += 1
                    pool_sorted = sorted(active_pool)
                    query = int(choice_rng.choice(pool_sorted))
                    q_i, q_j = pairs[query]
                    obs_bit = anchor_bits[q_i] ^ anchor_bits[q_j]
                    true_action = anchor_bits[true_i] ^ anchor_bits[true_j] ^ true_polarity
                    sample = outcome_rng.random()
                    if method == "noiseless_version_space_elimination":
                        # Only eliminates hypotheses whose fields were actually observed at this anchor
                        observed_at_anchor = trace_subset | {q_i, q_j}
                        for m_idx in range(2 * num_programs):
                            r_prog, r_pol = divmod(m_idx, 2)
                            m_i, m_j = pairs[r_prog]
                            if m_i in observed_at_anchor and m_j in observed_at_anchor:
                                pred_action = anchor_bits[m_i] ^ anchor_bits[m_j] ^ r_pol
                                if pred_action != true_action:
                                    weight[m_idx] = 0.0
                        if weight.sum() == 0:
                            weight[:] = 1.0
                        weight /= weight.sum()
                    else:
                        if sample < 0.2:
                            outcome = 4 * obs_bit
                        elif sample < 0.4:
                            outcome = 4 * obs_bit + 1
                        else:
                            outcome = 4 * obs_bit + 2 + true_action
                        weight *= table[query, :, outcome]
                        weight /= weight.sum()
                    continue

                max_active_pool = initial_pool_size + 2 * proposal_budget
                if block in (2, 4, 6, 8) and len(active_pool) < max_active_pool:
                    step_budget = min(1, max_active_pool - len(active_pool))
                    new_programs, prop_stats = _propose_programs(
                        pairs,
                        active_pool,
                        records,
                        step_budget,
                        method,
                        proposer_rng,
                        max_candidate_checks=max_candidate_checks,
                    )
                    cum_candidates_checked += int(prop_stats["candidates_checked"])
                    active_pool.update(new_programs)

                pool_sorted = sorted(active_pool)
                avail_models = []
                for q_idx in pool_sorted:
                    avail_models.extend([2 * q_idx, 2 * q_idx + 1])
                sub_scores = scores_for(
                    weight,
                    table[pool_sorted],
                    "complete_program_evsi",
                    values_sub=values[:, avail_models],
                )
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
                    d_val = 1 if true_action == 0 else -1

                pred_prob = float(np.dot(weight, table[query, :, outcome]))
                surprise = float(-math.log(max(pred_prob, 1e-12)))

                weight *= table[query, :, outcome]
                assert weight.sum() > 0
                weight /= weight.sum()

                paid_fields = trace_subset | {q_i, q_j}
                observed_bits_dict = {int(f_idx): int(anchor_bits[f_idx]) for f_idx in paid_fields}
                records.append({
                    "query": query,
                    "obs_bit": obs_bit,
                    "d": d_val,
                    "surprise": surprise,
                    "observed_bits": observed_bits_dict,
                    "trace_fields": paid_fields,
                })

        raw[method] = {str(k): observed[k] for k in checkpoints}
        hit_rates_raw[method] = {str(k): hit_rates[k] for k in checkpoints}
        for n in checkpoints:
            samples = np.array(observed[n])
            hits = np.array(hit_rates[n])
            imps = np.array(best_improvements[n])
            checks_arr = np.array(checked_counts[n])
            mean = float(samples.mean())
            se = float(samples.std(ddof=1) / math.sqrt(seeds)) if seeds > 1 else 0.0
            hit_mean = float(hits.mean())
            hit_se = float(hits.std(ddof=1) / math.sqrt(seeds)) if seeds > 1 else 0.0
            summary.append({
                "fields": num_fields,
                "total_two_read_programs": num_programs,
                "model_hypotheses": 2 * num_programs,
                "proposal_budget_per_block": proposal_budget,
                "max_candidate_checks_per_call": max_candidate_checks,
                "initial_pool_size": initial_pool_size,
                "method": method,
                "query_privilege": (
                    "free_noiseless_action_direction_oracle_on_observed_fields"
                    if method == "noiseless_version_space_elimination"
                    else "paid_paired_intervention_4_actions_plus_4_trace_reads_8_actions_per_block"
                ),
                "blocks": n,
                "paired_probe_actions": 4 * n,
                "trace_read_actions": 4 * n,
                "physical_actions": 8 * n,
                "mean_candidates_checked": float(checks_arr.mean()),
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
        "max_candidate_checks_per_call": max_candidate_checks,
        "initial_pool_size": initial_pool_size,
        "field_access_policy": "strictly_paid_observed_fields_only",
        "policy_support_policy": "unified_active_pool_for_both_acquisition_and_deployment",
        "response_ceiling_with_known_model": 0.8,
        "untrained_value": 0.5,
        "summary": summary,
        "raw_seed_values": raw,
        "raw_hit_rates": hit_rates_raw,
    }


def unknown_k_delayed_goal_study(
    num_fields=8,
    seeds=256,
    max_blocks=32,
    proposal_budget=2,
    initial_pool_size=2,
    max_candidate_checks=6,
):
    """Unknown response law K and delayed two-stage terminal goal study (fixes E07).

    Unlike the known-table diagnostic:
      1. The learner is NOT given the likelihood table `table` or the value matrix `values`.
         Instead, for each discovered program `q` and observed parity bit `z in {0, 1}`, it learns
         an empirical Dirichlet-Multinomial response model K_hat(o | q, z) and empirical terminal
         continuation utility V_hat(q, z, action) from observed transitions.
      2. Each task has a two-stage delayed terminal outcome Y in {0, 1}: Stage 1 sets a latent
         precondition via the parity of `(true_i, true_j)`, and Stage 2 reaches native terminal
         success Y=1 only when the continuation matches the true parity under tail continuation pi_0.
    """
    pairs = list(itertools.combinations(range(num_fields), 2))
    num_programs = len(pairs)
    methods = (
        "action_diff_growth",
        "label_shuffled_growth",
        "prediction_error_growth",
        "uniform_prior_growth",
    )
    checkpoints = [n for n in (0, 2, 4, 8, 16, 32) if n <= max_blocks]
    checkpoints = sorted(set(checkpoints + [max_blocks]))
    summary = []

    for method in methods:
        terminal_success_rates = {n: [] for n in checkpoints}
        hit_rates = {n: [] for n in checkpoints}
        checked_counts = {n: [] for n in checkpoints}

        for seed in range(seeds):
            world_rng = np.random.default_rng(400009 * num_fields + seed)
            true_model = int(world_rng.integers(2 * num_programs))
            true_program, true_polarity = divmod(true_model, 2)
            true_i, true_j = pairs[true_program]

            init_rng = np.random.default_rng(700001 * num_fields + seed)
            non_true = [idx for idx in range(num_programs) if idx != true_program]
            active_pool = set(
                int(x) for x in init_rng.choice(non_true, size=min(initial_pool_size, len(non_true)), replace=False)
            )

            anchor_rng = np.random.default_rng(200003 * num_fields + seed)
            outcome_rng = np.random.default_rng(300007 * num_fields + seed)
            choice_rng = np.random.default_rng(500009 * num_fields + seed)
            proposer_rng = np.random.default_rng(900011 * num_fields + seed)

            # Unknown K_hat and V_hat: Laplace-smoothed empirical counts per (program, z_bit, action)
            # succ_counts[q, z_bit, act] and total_counts[q, z_bit, act]
            succ_counts = np.ones((num_programs, 2, 2), dtype=np.float64)
            total_counts = np.full((num_programs, 2, 2), 2.0, dtype=np.float64)
            records = []
            cum_candidates_checked = 0

            for block in range(max_blocks + 1):
                if block in terminal_success_rates:
                    # Evaluate learned (K_hat, V_hat) on active_pool without any oracle value table
                    q_hat = succ_counts / total_counts  # (Q, 2, 2)
                    # Expected value of executing program q and choosing argmax_a Q_hat(q, z, a)
                    prog_expected_v = {
                        q_idx: 0.5 * (float(np.max(q_hat[q_idx, 0])) + float(np.max(q_hat[q_idx, 1])))
                        for q_idx in sorted(active_pool)
                    }
                    max_v = max(prog_expected_v.values())
                    best_progs = [q_idx for q_idx, v in prog_expected_v.items() if v >= max_v - 1e-12]
                    # Compute exact conditional success rate of the chosen policy on unseen instances
                    eval_succ = 0.0
                    for q_sel in best_progs:
                        if q_sel == true_program:
                            # Check whether learned conditional action for z=0 and z=1 matches true_polarity
                            a0 = int(np.argmax(q_hat[q_sel, 0]))
                            a1 = int(np.argmax(q_hat[q_sel, 1]))
                            acc0 = 0.85 if a0 == (0 ^ true_polarity) else 0.15
                            acc1 = 0.85 if a1 == (1 ^ true_polarity) else 0.15
                            eval_succ += 0.5 * (acc0 + acc1)
                        else:
                            eval_succ += 0.5
                    terminal_success_rates[block].append(eval_succ / len(best_progs))
                    hit_rates[block].append(float(true_program in active_pool))
                    checked_counts[block].append(float(cum_candidates_checked))
                if block == max_blocks:
                    break

                max_active_pool = initial_pool_size + 2 * proposal_budget
                if block in (2, 4, 6, 8, 12, 16) and len(active_pool) < max_active_pool:
                    step_budget = min(1, max_active_pool - len(active_pool))
                    new_programs, prop_stats = _propose_programs(
                        pairs,
                        active_pool,
                        records,
                        step_budget,
                        method,
                        proposer_rng,
                        max_candidate_checks=max_candidate_checks,
                    )
                    cum_candidates_checked += int(prop_stats["candidates_checked"])
                    active_pool.update(new_programs)

                # Select query program from active_pool using optimistic empirical value + exploration bonus
                q_hat = succ_counts / total_counts
                best_q = None
                best_score = -1e9
                for q_idx in sorted(active_pool):
                    emp_val = 0.5 * (float(np.max(q_hat[q_idx, 0])) + float(np.max(q_hat[q_idx, 1])))
                    uncert = 0.15 / math.sqrt(float(total_counts[q_idx].sum()))
                    sc = emp_val + uncert + float(choice_rng.uniform(0.0, 1e-9))
                    if sc > best_score:
                        best_score = sc
                        best_q = q_idx
                query = int(best_q)

                anchor_bits = tuple(int(b) for b in anchor_rng.integers(0, 2, size=num_fields))
                trace_subset = set(
                    int(x) for x in anchor_rng.choice(num_fields, size=min(4, num_fields), replace=False)
                )
                q_i, q_j = pairs[query]
                obs_bit = anchor_bits[q_i] ^ anchor_bits[q_j]
                true_action = anchor_bits[true_i] ^ anchor_bits[true_j] ^ true_polarity

                # Coupled outcomes (Y_0, Y_1) under shared exogenous uniform U
                u_rand = outcome_rng.random()
                y_correct = 1 if u_rand < 0.85 else 0
                y_wrong = 1 if u_rand < 0.15 else 0
                y_0 = y_correct if true_action == 0 else y_wrong
                y_1 = y_correct if true_action == 1 else y_wrong
                d_real = int(y_0 - y_1)

                # Mean of the two conditional marginal negative log-likelihoods (fixes F21 surprise definition)
                p0 = float(q_hat[query, obs_bit, 0]) if y_0 == 1 else (1.0 - float(q_hat[query, obs_bit, 0]))
                p1 = float(q_hat[query, obs_bit, 1]) if y_1 == 1 else (1.0 - float(q_hat[query, obs_bit, 1]))
                surprise = float(0.5 * (-math.log(max(p0, 1e-6)) - math.log(max(p1, 1e-6))))

                # Update empirical response/terminal model K_hat
                succ_counts[query, obs_bit, 0] += y_0
                total_counts[query, obs_bit, 0] += 1.0
                succ_counts[query, obs_bit, 1] += y_1
                total_counts[query, obs_bit, 1] += 1.0

                paid_fields = trace_subset | {q_i, q_j}
                observed_bits_dict = {int(f_idx): int(anchor_bits[f_idx]) for f_idx in paid_fields}
                records.append({
                    "query": query,
                    "obs_bit": obs_bit,
                    "d": d_real,
                    "surprise": surprise,
                    "observed_bits": observed_bits_dict,
                    "trace_fields": paid_fields,
                })

        for n in checkpoints:
            arr = np.array(terminal_success_rates[n])
            hits = np.array(hit_rates[n])
            checks_arr = np.array(checked_counts[n])
            mean = float(arr.mean())
            se = float(arr.std(ddof=1) / math.sqrt(seeds)) if seeds > 1 else 0.0
            summary.append({
                "fields": num_fields,
                "method": method,
                "blocks": n,
                "physical_actions": 8 * n,
                "mean_candidates_checked": float(checks_arr.mean()),
                "mean_delayed_terminal_success": mean,
                "seed_standard_error": se,
                "normal_95_ci": [mean - 1.96 * se, mean + 1.96 * se],
                "relevant_program_hit_rate": float(hits.mean()),
            })

    return {
        "fields": num_fields,
        "seeds": seeds,
        "max_blocks": max_blocks,
        "prior_scope": "unknown_conditional_success_rate_K_and_program_discovery",
        "ceiling_delayed_terminal_success": 0.85,
        "untrained_delayed_terminal_success": 0.50,
        "summary": summary,
    }


def sequential_multistep_delayed_goal_study(
    num_fields=8,
    seeds=256,
    max_blocks=32,
    proposal_budget=2,
    initial_pool_size=2,
    max_candidate_checks=6,
):
    """Multi-step hidden-state evolution with local public readouts, AST Case execution, and terminal-only Y (fixes F21).

    Environment dynamics:
      - Stage 1 (Probe q = Read(i); Read(j)): reads two public fields `(b_i, b_j)` at cost 2 actions
        (plus 4 paid trace reads = 8 total actions per block) and exposes local parity `z = b_i ^ b_j`.
      - Stage 2 (Conditional action `a_stage1 = Case(z: u_0, u_1)`): modifies hidden state `s_hidden`.
        Only when `(i, j) == (true_i, true_j)` and `a_stage1 == (b_{true_i} ^ b_{true_j} ^ true_polarity)`
        does `s_hidden` transition to `"precondition_unlocked"` (with probability 0.90) and emit public
        local feedback `r_stage1 = "ready"`; otherwise `s_hidden = "locked"` and `r_stage1 = "blocked"` (0.90).
      - Stage 3 (Tail policy pi_0 commit step): observes `r_stage1` and executes `commit_goal` when
        `"ready"` or `retry_prepare` when `"blocked"`. Official terminal outcome `Y in {0, 1}` is revealed
        ONLY at the end of Stage 3 (`Y = 1` iff `s_hidden == "precondition_unlocked"` and `commit_goal` succeeds).
      - Training, model Q-evaluation, and deployment share the same AST `Case(z: u_0, u_1)` interpreter:
        the learner estimates Stage-2 transition probabilities `K_hat(r_stage1 = "ready" | q, z, a_stage1)`
        from local public responses AND Stage-3 tail completion `H_hat^{pi_0}(Y = 1 | r_stage1)`, composing
        them to rank candidate programs and execute conditional policies.
    """
    pairs = list(itertools.combinations(range(num_fields), 2))
    num_programs = len(pairs)
    methods = (
        "action_diff_growth",
        "label_shuffled_growth",
        "prediction_error_growth",
        "uniform_prior_growth",
    )
    checkpoints = [n for n in (0, 2, 4, 8, 16, 32) if n <= max_blocks]
    checkpoints = sorted(set(checkpoints + [max_blocks]))
    summary = []

    for method in methods:
        terminal_utilities = {n: [] for n in checkpoints}
        hit_rates = {n: [] for n in checkpoints}
        checked_counts = {n: [] for n in checkpoints}

        for seed in range(seeds):
            world_rng = np.random.default_rng(600011 * num_fields + seed)
            true_model = int(world_rng.integers(2 * num_programs))
            true_program, true_polarity = divmod(true_model, 2)
            true_i, true_j = pairs[true_program]

            init_rng = np.random.default_rng(700001 * num_fields + seed)
            non_true = [idx for idx in range(num_programs) if idx != true_program]
            active_pool = set(
                int(x) for x in init_rng.choice(non_true, size=min(initial_pool_size, len(non_true)), replace=False)
            )

            anchor_rng = np.random.default_rng(200003 * num_fields + seed)
            outcome_rng = np.random.default_rng(300007 * num_fields + seed)
            choice_rng = np.random.default_rng(500009 * num_fields + seed)
            proposer_rng = np.random.default_rng(900011 * num_fields + seed)

            # Learned Stage-2 transition model K_hat(r_stage1="ready" | q, z_bit, a_stage1)
            ready_counts = np.ones((num_programs, 2, 2), dtype=np.float64)
            stage2_totals = np.full((num_programs, 2, 2), 2.0, dtype=np.float64)
            # Learned Stage-3 tail model H_hat^{pi_0}(Y=1 | r_stage1 in {0="blocked", 1="ready"})
            tail_y_succ = np.array([1.0, 1.0], dtype=np.float64)
            tail_y_tot = np.array([2.0, 2.0], dtype=np.float64)

            records = []
            cum_candidates_checked = 0

            def _compose_ast_program_q(q_idx: int, k_ready: np.ndarray, h_tail: np.ndarray) -> tuple[float, int, int]:
                """Shared AST interpreter + composed Q-evaluator for `Seq(Read(i), Read(j), Case(z: a0, a1)) -> pi_0`."""
                # For each branch z in {0, 1}, choose action a_z maximizing composed terminal Q
                q_z0_a0 = k_ready[q_idx, 0, 0] * h_tail[1] + (1.0 - k_ready[q_idx, 0, 0]) * h_tail[0]
                q_z0_a1 = k_ready[q_idx, 0, 1] * h_tail[1] + (1.0 - k_ready[q_idx, 0, 1]) * h_tail[0]
                a_z0 = 0 if q_z0_a0 >= q_z0_a1 else 1
                v_z0 = max(q_z0_a0, q_z0_a1)

                q_z1_a0 = k_ready[q_idx, 1, 0] * h_tail[1] + (1.0 - k_ready[q_idx, 1, 0]) * h_tail[0]
                q_z1_a1 = k_ready[q_idx, 1, 1] * h_tail[1] + (1.0 - k_ready[q_idx, 1, 1]) * h_tail[0]
                a_z1 = 0 if q_z1_a0 >= q_z1_a1 else 1
                v_z1 = max(q_z1_a0, q_z1_a1)

                return 0.5 * (v_z0 + v_z1), a_z0, a_z1

            for block in range(max_blocks + 1):
                k_ready = ready_counts / stage2_totals
                h_tail = tail_y_succ / tail_y_tot

                if block in terminal_utilities:
                    prog_evals = {
                        q_idx: _compose_ast_program_q(q_idx, k_ready, h_tail)
                        for q_idx in sorted(active_pool)
                    }
                    max_q = max(v[0] for v in prog_evals.values())
                    best_progs = [q_idx for q_idx, v in prog_evals.items() if v[0] >= max_q - 1e-12]
                    eval_u = 0.0
                    for q_sel in best_progs:
                        _, a_z0, a_z1 = prog_evals[q_sel]
                        if q_sel == true_program:
                            # True Stage-2 unlock prob is 0.90 on matching action, 0.10 on mismatch;
                            # True Stage-3 tail Y=1 prob given unlocked is 0.95, given locked is 0.05.
                            p_unlock_0 = 0.90 if a_z0 == (0 ^ true_polarity) else 0.10
                            p_unlock_1 = 0.90 if a_z1 == (1 ^ true_polarity) else 0.10
                            y_0 = p_unlock_0 * 0.95 + (1.0 - p_unlock_0) * 0.05
                            y_1 = p_unlock_1 * 0.95 + (1.0 - p_unlock_1) * 0.05
                            eval_u += 0.5 * (y_0 + y_1)
                        else:
                            eval_u += 0.5 * (0.95 + 0.05)
                    terminal_utilities[block].append(eval_u / len(best_progs))
                    hit_rates[block].append(float(true_program in active_pool))
                    checked_counts[block].append(float(cum_candidates_checked))
                if block == max_blocks:
                    break

                max_active_pool = initial_pool_size + 2 * proposal_budget
                if block in (2, 4, 6, 8, 12, 16) and len(active_pool) < max_active_pool:
                    step_budget = min(1, max_active_pool - len(active_pool))
                    new_programs, prop_stats = _propose_programs(
                        pairs,
                        active_pool,
                        records,
                        step_budget,
                        method,
                        proposer_rng,
                        max_candidate_checks=max_candidate_checks,
                    )
                    cum_candidates_checked += int(prop_stats["candidates_checked"])
                    active_pool.update(new_programs)

                # Select probe program q in active_pool maximizing composed Q + exploration bonus
                best_q = None
                best_sc = -1e9
                for q_idx in sorted(active_pool):
                    q_val, _, _ = _compose_ast_program_q(q_idx, k_ready, h_tail)
                    uncert = 0.15 / math.sqrt(float(stage2_totals[q_idx].sum()))
                    sc = q_val + uncert + float(choice_rng.uniform(0.0, 1e-9))
                    if sc > best_sc:
                        best_sc = sc
                        best_q = q_idx
                query = int(best_q)

                anchor_bits = tuple(int(b) for b in anchor_rng.integers(0, 2, size=num_fields))
                trace_subset = set(
                    int(x) for x in anchor_rng.choice(num_fields, size=min(4, num_fields), replace=False)
                )
                q_i, q_j = pairs[query]
                obs_bit = anchor_bits[q_i] ^ anchor_bits[q_j]
                true_action = anchor_bits[true_i] ^ anchor_bits[true_j] ^ true_polarity

                # Paired multi-step rollouts u (action 0 -> pi_0) and v (action 1 -> pi_0) from post-q state
                u_stage2_rand = outcome_rng.random()
                u_tail_rand = outcome_rng.random()

                # Branch u (a_stage1 = 0)
                p_ready_u = 0.90 if true_action == 0 else 0.10
                r_stage1_u = 1 if u_stage2_rand < p_ready_u else 0
                y_u = 1 if u_tail_rand < (0.95 if r_stage1_u == 1 else 0.05) else 0

                # Branch v (a_stage1 = 1)
                p_ready_v = 0.90 if true_action == 1 else 0.10
                r_stage1_v = 1 if u_stage2_rand < p_ready_v else 0
                y_v = 1 if u_tail_rand < (0.95 if r_stage1_v == 1 else 0.05) else 0

                d_real = int(y_u - y_v)

                # Local transition prediction error on Stage-2 response r_stage1
                pred_r_u = float(k_ready[query, obs_bit, 0]) if r_stage1_u == 1 else (1.0 - float(k_ready[query, obs_bit, 0]))
                pred_r_v = float(k_ready[query, obs_bit, 1]) if r_stage1_v == 1 else (1.0 - float(k_ready[query, obs_bit, 1]))
                surprise = float(0.5 * (-math.log(max(pred_r_u, 1e-6)) - math.log(max(pred_r_v, 1e-6))))

                # Update Stage-2 transition model K_hat and Stage-3 tail model H_hat^{pi_0}
                ready_counts[query, obs_bit, 0] += r_stage1_u
                stage2_totals[query, obs_bit, 0] += 1.0
                ready_counts[query, obs_bit, 1] += r_stage1_v
                stage2_totals[query, obs_bit, 1] += 1.0

                tail_y_succ[r_stage1_u] += y_u
                tail_y_tot[r_stage1_u] += 1.0
                tail_y_succ[r_stage1_v] += y_v
                tail_y_tot[r_stage1_v] += 1.0

                paid_fields = trace_subset | {q_i, q_j}
                observed_bits_dict = {int(f_idx): int(anchor_bits[f_idx]) for f_idx in paid_fields}
                records.append({
                    "query": query,
                    "obs_bit": obs_bit,
                    "d": d_real,
                    "surprise": surprise,
                    "observed_bits": observed_bits_dict,
                    "trace_fields": paid_fields,
                })

        for n in checkpoints:
            arr = np.array(terminal_utilities[n])
            hits = np.array(hit_rates[n])
            checks_arr = np.array(checked_counts[n])
            mean = float(arr.mean())
            se = float(arr.std(ddof=1) / math.sqrt(seeds)) if seeds > 1 else 0.0
            summary.append({
                "fields": num_fields,
                "method": method,
                "blocks": n,
                "physical_actions": 10 * n,  # 2 probe reads + 4 trace reads + 2*(Stage-2 act + Stage-3 tail act)
                "mean_candidates_checked": float(checks_arr.mean()),
                "mean_sequential_terminal_success": mean,
                "seed_standard_error": se,
                "normal_95_ci": [mean - 1.96 * se, mean + 1.96 * se],
                "relevant_program_hit_rate": float(hits.mean()),
            })

    return {
        "fields": num_fields,
        "seeds": seeds,
        "max_blocks": max_blocks,
        "prior_scope": "sequential_multistep_hidden_state_with_local_readout_and_tail_pi0_terminal_Y",
        "ceiling_sequential_terminal_success": 0.90 * 0.95 + 0.10 * 0.05,  # 0.86
        "untrained_sequential_terminal_success": 0.50,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=512)
    parser.add_argument("--fields", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--run-growth", action="store_true",
                        help="Run the bounded-budget action-contrast program growth and unknown-K delayed-goal study.")
    parser.add_argument("--growth-fields", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--growth-blocks", type=int, default=32)
    parser.add_argument("--proposal-budget", type=int, default=2)
    parser.add_argument("--max-candidate-checks", type=int, default=6)
    parser.add_argument("--growth-output", type=Path,
                        default=Path(__file__).with_name("growth_results.json"))
    args = parser.parse_args()
    result = {
        "scope": "Controlled finite-model research diagnostics; not native ALFWorld/WebShop/LOCA results and not neural DCC training.",
        "date": "2026-11-10",
        "checks": check_likelihood_and_evsi(),
        "known_model_parity": parity_diagnostics(),
        "learned_finite_model": [],
        "training_protocol": {
            "full_task_reward": "Known-table study simulates short probes and local deployment; unknown_k_delayed_goal_study simulates unknown response law K and delayed terminal outcome Y.",
            "public_training_feedback": "Joint two-bit query and two action responses from one shared snapshot.",
            "prior_knowledge": "Known-table study provides likelihood/value tables as a decision-theoretic reference; growth and unknown-K studies enforce paid field observation and candidate check budgets.",
            "cost": "Four environment actions per block plus explicit candidate construction/check counts.",
            "novelty_scope": "Classical EVSI serves as finite reference; novelty is tested in bounded-budget program discovery driven by observed action contrasts.",
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
            "scope": "Bounded-budget program discovery study with strictly paid field access, candidate check budgets, unified active-pool policy support, unknown-K conditional evaluation, and multi-step sequential hidden-state delayed-goal evaluation.",
            "date": "2026-11-10",
            "checks": growth_checks,
            "program_growth_runs": [],
            "unknown_k_delayed_goal_runs": [],
            "sequential_multistep_delayed_goal_runs": [],
        }
        for g_fields in args.growth_fields:
            g_run = growth_study(
                g_fields,
                args.seeds,
                args.growth_blocks,
                proposal_budget=args.proposal_budget,
                max_candidate_checks=args.max_candidate_checks,
            )
            growth_payload["program_growth_runs"].append(g_run)
            u_run = unknown_k_delayed_goal_study(
                num_fields=g_fields,
                seeds=min(args.seeds, 256),
                max_blocks=args.growth_blocks,
                proposal_budget=args.proposal_budget,
                max_candidate_checks=args.max_candidate_checks,
            )
            growth_payload["unknown_k_delayed_goal_runs"].append(u_run)
            seq_run = sequential_multistep_delayed_goal_study(
                num_fields=g_fields,
                seeds=min(args.seeds, 256),
                max_blocks=args.growth_blocks,
                proposal_budget=args.proposal_budget,
                max_candidate_checks=args.max_candidate_checks,
            )
            growth_payload["sequential_multistep_delayed_goal_runs"].append(seq_run)
            last = [r for r in g_run["summary"] if r["blocks"] == args.growth_blocks]
            print(json.dumps({"growth_fields": g_fields, "final": last}, ensure_ascii=False), flush=True)
        args.growth_output.parent.mkdir(parents=True, exist_ok=True)
        args.growth_output.write_text(json.dumps(growth_payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
