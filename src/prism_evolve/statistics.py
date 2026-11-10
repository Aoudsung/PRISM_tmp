"""Paired method comparisons for the registered native experiment (§7.3).

Inference is stratified by benchmark and backbone. Repeated seeds are averaged
within each task before scenario-level resampling; tasks have equal weight in
both the reported estimate and every bootstrap replicate. McNemar is reported
separately for each seed because its paired observations must remain binary.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


def official_benchmark_summary(records: Sequence[Mapping[str, Any]],
                               expected_tasks: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Aggregate native evaluator receipts, never substitute budget success for TGC/accuracy.

    Call separately for each method/model/split/seed. SGC is reported only when
    every task of every represented official scenario has a native score.
    """
    if not records:
        return {"status": "no_records", "evaluated_count": 0}
    identities = {(r.get("benchmark"), r.get("backbone"), r.get("seed"), r.get("method"),
                   r.get("split")) for r in records}
    if len(identities) != 1:
        raise ValueError("Official metrics require one benchmark/model/method/split/seed stratum.")
    by_id = {str(r["task_id"]): r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate task IDs in official benchmark aggregation.")
    scored = [(r, r.get("metadata", {}).get("official_evaluation")) for r in records]
    scored = [(r, e) for r, e in scored if isinstance(e, Mapping) and isinstance(e.get("success"), bool)]
    count = len(scored)
    expected = {str(r["task_id"]): r for r in expected_tasks} if expected_tasks is not None else None
    benchmark = str(records[0].get("benchmark", "")).split(":", 1)[0]
    result = {"benchmark": benchmark, "evaluated_count": count, "record_count": len(records),
              "missing_native_scores": sorted(set(by_id) - {str(r["task_id"]) for r, _ in scored}),
              "expected_task_count": len(expected) if expected is not None else None,
              "missing_tasks": sorted(set(expected) - set(by_id)) if expected is not None else None,
              "unexpected_tasks": sorted(set(by_id) - set(expected)) if expected is not None else None,
              "status": "complete" if count == len(records) else "missing_native_scores",
              "correct_count": sum(e["success"] for _, e in scored),
              "budget_compliant_success_count": sum(_binary(r["success"]) for r in records),
              "budget_compliant_success_rate": sum(_binary(r["success"]) for r in records) / len(records),
              "mean_project_utility": math.fsum(float(r["utility"]) for r in records) / len(records)}
    rate = result["correct_count"] / count if count else None
    result["dataset_complete"] = expected is not None and set(expected) == set(by_id) and count == len(expected)
    if expected is not None and (result["missing_tasks"] or result["unexpected_tasks"]):
        result["status"] = "partial_or_mismatching_dataset"
    def grouped_success(field: str, aliases: Mapping[str, str] | None = None) -> dict[str, Any]:
        groups: dict[str, list[bool]] = defaultdict(list)
        missing = []
        for record, evaluation in scored:
            task = expected.get(str(record["task_id"]), {}) if expected is not None else {}
            label = evaluation.get(field, task.get(field))
            if not isinstance(label, str) or not label:
                missing.append(str(record["task_id"]))
                continue
            label = aliases.get(label, label) if aliases else label
            groups[label].append(evaluation["success"])
        return {"status": "complete" if not missing and count else "missing_native_groups",
                "missing_task_ids": missing,
                "groups": {label: {"task_count": len(values), "correct_count": sum(values),
                                   "success_rate": sum(values) / len(values)}
                           for label, values in sorted(groups.items())}}

    if benchmark == "appworld":
        groups: dict[str, dict[str, bool]] = defaultdict(dict)
        declared: dict[str, set[str]] = {}
        if expected is not None:
            for task in expected.values():
                declared.setdefault(str(task["scenario_id"]), set()).add(str(task["task_id"]))
        for record, evaluation in scored:
            scenario = str(record["scenario_id"])
            groups[scenario][str(record["task_id"])] = evaluation["success"]
            members = evaluation.get("scenario_task_ids")
            if members:
                members = set(map(str, members))
                if scenario in declared and declared[scenario] != members:
                    raise ValueError("Inconsistent official scenario membership.")
                declared[scenario] = members
        incomplete = sorted(s for s in groups if not declared.get(s) or set(groups[s]) != declared[s])
        all_complete = bool(groups) and not incomplete and count == len(records)
        result.update(task_goal_completion=rate, scenario_goal_completion=(
            sum(all(values.values()) for values in groups.values()) / len(groups) if all_complete else None),
            scenario_count=len(groups), incomplete_scenarios=incomplete,
            sgc_status="complete_scenarios" if all_complete else "incomplete_or_unknown_scenario_membership",
            score_unit="fraction", task_goal_completion_percent=100 * rate if rate is not None else None)
        result["scenario_goal_completion_percent"] = (
            100 * result["scenario_goal_completion"] if all_complete else None)
    elif benchmark == "planbench_xl":
        metrics = [e.get("metrics", {}) for _, e in scored]
        required = ("search_turns", "call_turns", "invalid_tool_call_count",
                    "untrusted_input_rejection_count", "noisy_tool_call_count", "explored_datatype_count")
        complete = count > 0 and all(all(key in row for key in required) for row in metrics)
        result.update(accuracy=rate, diagnostic_metrics_status="complete" if complete else "missing_native_metrics")
        if complete:
            totals = {key: sum(row[key] for row in metrics) for key in required}
            calls = totals["call_turns"]
            result["metrics"] = {**totals,
                "search_to_call_ratio": totals["search_turns"] / calls if calls else None,
                "invalid_tool_call_rate": totals["invalid_tool_call_count"] / calls if calls else None,
                "untrusted_input_rejection_rate": totals["untrusted_input_rejection_count"] / calls if calls else None,
                "noisy_tool_call_rate": totals["noisy_tool_call_count"] / calls if calls else None,
                "mean_explored_datatypes": totals["explored_datatype_count"] / count}
            for key in ("executed_gt_datatype_precision", "executed_gt_datatype_recall"):
                values = [row[key] for row in metrics if row.get(key) is not None]
                result["metrics"]["mean_" + key] = math.fsum(values) / len(values) if values else None
            result["metrics"]["mean_turn_count"] = math.fsum(
                float(e.get("result", {}).get("steps", r["environment_steps"])) for r, e in scored) / count
            result["metrics"]["total_turn_count"] = sum(
                e.get("result", {}).get("steps", r["environment_steps"]) for r, e in scored)
            combined = totals["invalid_tool_call_count"] + totals["untrusted_input_rejection_count"]
            result["metrics"].update(combined_invalid_tool_call_count=combined,
                combined_invalid_tool_call_rate=combined / calls if calls else None,
                noisy_tool_task_count=sum(row["noisy_tool_call_count"] > 0 for row in metrics),
                noisy_tool_task_rate=sum(row["noisy_tool_call_count"] > 0 for row in metrics) / count)
        result["per_query_metrics"] = [{"task_id": r["task_id"], "success": e["success"],
                                         "metrics": e.get("metrics")} for r, e in scored]
    elif benchmark == "alfworld":
        # PoS reports task-micro success, alongside Place/Transform/Inspect groups.
        result.update(success_rate=rate, aggregation="task_micro", score_unit="fraction",
                      categories=grouped_success("category"), task_types=grouped_success("task_type"))
    elif benchmark == "webshop":
        scores = []
        missing_scores = []
        for record, evaluation in scored:
            score = evaluation.get("native_score", evaluation.get("task_score"))
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)):
                missing_scores.append(str(record["task_id"]))
                continue
            scores.append(float(score))
        mean_score = math.fsum(scores) / len(scores) if scores else None
        result.update(
            success_rate=rate,
            success_rate_percent=100 * rate if rate is not None else None,
            mean_native_score=mean_score,
            mean_task_score=mean_score,
            mean_task_score_percent=100 * mean_score if mean_score is not None else None,
            native_score_count=len(scores),
            missing_native_task_scores=missing_scores,
            native_score_status="complete" if len(scores) == count and count else "missing_native_task_scores",
            aggregation="task_micro",
            score_unit="fraction",
        )
    elif benchmark == "clindiag":
        for _, evaluation in scored:
            if evaluation.get("diagnosis_correct") is not evaluation["success"]:
                raise ValueError("ClinDiag success must equal the native diagnosis_correct verdict.")
        submitted = [e for _, e in scored if e.get("submitted") is True]
        differential = [e["differential_hit"] for e in submitted
                        if isinstance(e.get("differential_hit"), bool)]
        result.update(diagnosis_accuracy=rate, aggregation="case_micro", score_unit="fraction",
                      subsets=grouped_success("subset", {"challenging": "common"}),
                      submitted_count=len(submitted), differential_scored_count=len(differential),
                      differential_hit_rate_among_submissions=(
                          sum(differential) / len(differential) if differential else None),
                      differential_status="complete" if len(differential) == len(submitted)
                                          else "missing_native_differential_verdicts")
    elif benchmark == "loca":
        result.update(success_rate=rate, aggregation="task_instance_micro", score_unit="fraction",
                      categories=grouped_success("task_category"),
                      environment_lengths=grouped_success("environment_length"),
                      task_types=grouped_success("task_type"))
        values = []
        missing = []
        for record, evaluation in scored:
            score = evaluation.get("score")
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)):
                missing.append(str(record["task_id"]))
                continue
            if evaluation["success"] != (float(score) >= 1.0):
                raise ValueError("LOCA native success must use the official score >= 1 criterion.")
            values.append(float(score))
        result.update(mean_native_score=math.fsum(values) / len(values) if values else None,
                      native_score_count=len(values), missing_native_task_scores=missing,
                      native_score_status="complete" if len(values) == count and count
                                          else "missing_native_task_scores")
    return result


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("seed must be a nonnegative integer.")
    return int(value)


def _binary(value: Any) -> int:
    if not isinstance(value, (Real, bool)) or value not in (0, 1):
        raise ValueError(f"success must be binary, got {value!r}.")
    return int(value)


def exact_mcnemar(reference: Sequence[int], baseline: Sequence[int]) -> dict[str, Any]:
    """Two-sided exact conditional binomial test on one seed's task pairs."""
    if len(reference) != len(baseline) or len(reference) == 0:
        raise ValueError("McNemar requires nonempty, equally sized paired outcomes.")
    wins = losses = both_success = both_failure = 0
    for ref, other in zip(reference, baseline):
        ref, other = _binary(ref), _binary(other)
        wins += ref == 1 and other == 0
        losses += ref == 0 and other == 1
        both_success += ref == 1 and other == 1
        both_failure += ref == 0 and other == 0
    discordant = wins + losses
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    p_value = min(1.0, 2 * tail / (1 << discordant)) if discordant else 1.0
    return {
        "test": "exact_mcnemar_two_sided",
        "num_pairs": len(reference),
        "reference_only_success": wins,
        "baseline_only_success": losses,
        "both_success": both_success,
        "both_failure": both_failure,
        "p_value": p_value,
        "sampling_unit": "task_within_one_seed",
        "assumption": "independent_discordant_task_pairs",
    }


def _cluster_totals(
    differences: Sequence[float], scenario_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(differences) != len(scenario_ids) or len(differences) == 0:
        raise ValueError("Provide one finite difference and scenario ID per task.")
    grouped: dict[str, list[float]] = defaultdict(list)
    for difference, scenario_id in zip(differences, scenario_ids):
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("scenario_id must be a nonempty string.")
        if not isinstance(difference, Real) or not math.isfinite(float(difference)):
            raise ValueError("Paired differences must be finite numbers.")
        grouped[scenario_id].append(float(difference))
    ordered = [grouped[key] for key in sorted(grouped)]
    return (
        np.asarray([math.fsum(values) for values in ordered], dtype=np.float64),
        np.asarray([len(values) for values in ordered], dtype=np.int64),
        len(differences),
    )


def scenario_bootstrap_ci(
    differences: Sequence[float],
    scenario_ids: Sequence[str],
    *,
    seed: int = 0,
    num_resamples: int = 10000,
) -> dict[str, Any]:
    """Percentile 95% CI from resampling whole scenarios with replacement.

    Each draw contains all tasks in each selected scenario. Its statistic is the
    sum of task differences divided by its task count, including multiplicities.
    A single observed scenario provides no between-scenario uncertainty estimate.
    """
    num_resamples = _positive_int(num_resamples, "num_resamples")
    rng = np.random.default_rng(_seed(seed))
    sums, counts, n_tasks = _cluster_totals(differences, scenario_ids)
    n_scenarios = len(sums)
    result: dict[str, Any] = {
        "estimate": float(sums.sum() / n_tasks),
        "ci95": None,
        "ci_method": "scenario_cluster_percentile_bootstrap",
        "num_tasks": n_tasks,
        "num_scenarios": n_scenarios,
        "num_resamples": 0,
        "seed": int(seed),
    }
    if n_scenarios < 2:
        result["ci_unavailable_reason"] = "fewer_than_two_scenarios"
        return result
    statistics = np.empty(num_resamples, dtype=np.float64)
    for start in range(0, num_resamples, 1024):
        stop = min(num_resamples, start + 1024)
        indices = rng.integers(0, n_scenarios, size=(stop - start, n_scenarios))
        statistics[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    result["ci95"] = [float(x) for x in np.quantile(statistics, [0.025, 0.975])]
    result["num_resamples"] = num_resamples
    return result


def paired_sign_flip_test(
    differences: Sequence[float],
    scenario_ids: Sequence[str],
    *,
    seed: int = 0,
    num_permutations: int = 20000,
    exact_max_clusters: int = 16,
) -> dict[str, Any]:
    """Two-sided sign-flip test, swapping methods jointly within each scenario.

    Exact enumeration uses every distinct sign vector for nonzero scenario sums.
    Monte Carlo uses the plus-one correction. The null requires exchangeability
    of method labels (equivalently sign symmetry) at the scenario level.
    """
    num_permutations = _positive_int(num_permutations, "num_permutations")
    if (
        isinstance(exact_max_clusters, bool)
        or not isinstance(exact_max_clusters, Integral)
        or not 0 <= exact_max_clusters <= 20
    ):
        raise ValueError("exact_max_clusters must be an integer in [0, 20].")
    rng = np.random.default_rng(_seed(seed))
    sums, _, n_tasks = _cluster_totals(differences, scenario_ids)
    nonzero = sums[sums != 0.0]
    observed = abs(float(sums.sum()))
    tolerance = 16 * np.finfo(np.float64).eps * max(1.0, observed, float(np.abs(sums).sum()))
    exact = len(nonzero) <= exact_max_clusters
    n_draws = (1 << len(nonzero)) if exact else num_permutations
    extreme = 0
    bit_positions = np.arange(len(nonzero), dtype=np.uint64) if exact else None
    for start in range(0, n_draws, 1024):
        stop = min(n_draws, start + 1024)
        if exact:
            indices = np.arange(start, stop, dtype=np.uint64)[:, None]
            bits = ((indices >> bit_positions) & 1).astype(np.float64)
            signs = 2.0 * bits - 1.0
        else:
            signs = 2 * rng.integers(0, 2, size=(stop - start, len(nonzero))) - 1
        permuted = np.abs(signs @ nonzero)
        extreme += int(np.count_nonzero(permuted >= observed - tolerance))
    p_value = extreme / n_draws if exact else (extreme + 1) / (n_draws + 1)
    return {
        "test": "scenario_paired_sign_flip_two_sided",
        "estimate": float(sums.sum() / n_tasks),
        "p_value": float(p_value),
        "calculation": "exact" if exact else "monte_carlo_plus_one",
        "num_permutations": n_draws,
        "num_scenarios": len(sums),
        "num_nonzero_scenarios": len(nonzero),
        "seed": int(seed),
        "assumption": "scenario_level_method_label_exchangeability",
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values in input order."""
    values = [float(p) for p in p_values]
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in values):
        raise ValueError("p-values must be finite numbers in [0, 1].")
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(values) - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


@dataclass(frozen=True)
class _Record:
    benchmark: str
    backbone: str
    seed: int
    task_id: str
    scenario_id: str
    success: int
    utility: float


def _field(record: Any, name: str) -> Any:
    try:
        return record[name] if isinstance(record, Mapping) else getattr(record, name)
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"Evaluation record is missing {name!r}.") from exc


def _index_records(records: Sequence[Any], method: str) -> dict[tuple[str, str, int, str], _Record]:
    if len(records) == 0:
        raise ValueError(f"Method {method!r} has no evaluation records.")
    indexed = {}
    for record in records:
        text_fields = {name: _field(record, name) for name in ("benchmark", "backbone", "task_id", "scenario_id")}
        if any(not isinstance(value, str) or not value.strip() for value in text_fields.values()):
            raise ValueError("benchmark, backbone, task_id and scenario_id must be nonempty strings.")
        run_seed = _seed(_field(record, "seed"))
        utility = _field(record, "utility")
        if not isinstance(utility, Real) or not math.isfinite(float(utility)) or not 0.0 <= utility <= 1.0:
            raise ValueError("utility must be a finite number in [0, 1].")
        normalized = _Record(**text_fields, seed=run_seed, success=_binary(_field(record, "success")), utility=float(utility))
        key = (normalized.benchmark, normalized.backbone, normalized.seed, normalized.task_id)
        if key in indexed:
            raise ValueError(f"Duplicate evaluation pair for method {method!r}: {key!r}.")
        indexed[key] = normalized
    return indexed


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
    return {"mean": mean, "std": math.sqrt(variance)}


def evaluate_method_comparisons(
    records_by_method: Mapping[str, Sequence[Any]],
    reference_method: str = "prism_evolve",
    *,
    seed: int = 0,
    num_bootstrap: int = 10000,
    num_permutations: int = 20000,
    exact_max_clusters: int = 16,
    alpha: float = 0.05,
    method_backbones: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Compare a registered reference method with every baseline on identical pairs.

    Every record must include benchmark/backbone/training seed/task/scenario and
    binary success plus utility. Dataclasses and mappings are both accepted. Each
    stratum must be a complete task × seed grid, identical across its methods. An
    optional pre-registered method_backbones mapping must list every method,
    including the reference; all records in each declared scope are required.
    Without that mapping every method must cover the full reference grid. CI and
    permutation tests target task-population performance conditional on the fixed
    registered seeds; seed variability is separately reported as mean/sample std.

    Holm correction covers all returned utility/success cluster tests and all
    per-seed McNemar tests across all strata in this call. Call once with the full
    registered comparison family; filtering methods first changes that family.
    """
    seed = _seed(seed)
    num_bootstrap = _positive_int(num_bootstrap, "num_bootstrap")
    num_permutations = _positive_int(num_permutations, "num_permutations")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    if reference_method not in records_by_method or len(records_by_method) < 2:
        raise ValueError("Provide the reference method and at least one baseline.")
    if any(not isinstance(method, str) or not method.strip() for method in records_by_method):
        raise ValueError("Method IDs must be nonempty strings.")
    indexed = {method: _index_records(records, method) for method, records in records_by_method.items()}
    reference = indexed[reference_method]
    keys = set(reference)
    reference_backbones = {key[1] for key in keys}
    if method_backbones is None:
        scopes = {method: set(reference_backbones) for method in indexed}
    else:
        if set(method_backbones) != set(indexed):
            raise ValueError("The registered backbone scope must name every method, including the reference.")
        scopes = {}
        for method, backbones in method_backbones.items():
            if (isinstance(backbones, str) or not backbones
                    or any(not isinstance(model, str) or not model.strip() for model in backbones)
                    or len(set(backbones)) != len(backbones)):
                raise ValueError(f"Method {method!r} requires a nonempty, unique registered backbone list.")
            scopes[method] = set(backbones)
        if scopes[reference_method] != reference_backbones:
            raise ValueError("Reference records must cover exactly their registered backbone scope.")
        if any(not models <= reference_backbones for models in scopes.values()):
            raise ValueError("Every registered method backbone must belong to the reference scope.")
        # A backbone-scoped design retains the entire common benchmark/seed/task
        # schedule. An absent stratum cannot silently narrow the reference grid.
        reference_pairs = {(benchmark, run_seed, task_id) for benchmark, _, run_seed, task_id in keys}
        for backbone in reference_backbones:
            actual = {(benchmark, run_seed, task_id) for benchmark, model, run_seed, task_id in keys
                      if model == backbone}
            if actual != reference_pairs:
                raise ValueError(f"Reference backbone {backbone!r} lacks common registered benchmark/seed/task pairs.")
    for method, records in indexed.items():
        expected = {key for key in keys if key[1] in scopes[method]}
        if set(records) != expected:
            missing, extra = expected - records.keys(), records.keys() - expected
            raise ValueError(f"Unpaired records for {method!r}: {len(missing)} missing, {len(extra)} extra.")
        for key in expected:
            if records[key].scenario_id != reference[key].scenario_id:
                raise ValueError(f"Scenario mismatch for {method!r} at {key!r}.")
    strata = sorted({key[:2] for key in keys})
    method_ids = sorted(indexed)
    all_tests: list[dict[str, Any]] = []
    stratum_results = []
    stream_index = 0
    for benchmark, backbone in strata:
        stratum_methods = [method for method in method_ids if backbone in scopes[method]]
        baselines = [method for method in stratum_methods if method != reference_method]
        stratum_keys = [key for key in keys if key[:2] == (benchmark, backbone)]
        seeds = sorted({key[2] for key in stratum_keys})
        task_ids = sorted({key[3] for key in stratum_keys})
        if len(stratum_keys) != len(seeds) * len(task_ids):
            raise ValueError(f"Incomplete task × seed grid for {(benchmark, backbone)!r}.")
        scenarios = []
        for task_id in task_ids:
            scenario_set = {reference[(benchmark, backbone, run_seed, task_id)].scenario_id for run_seed in seeds}
            if len(scenario_set) != 1:
                raise ValueError(f"Task {task_id!r} changes scenario across seeds.")
            scenarios.append(next(iter(scenario_set)))
        method_summaries = {}
        for method in stratum_methods:
            by_seed = []
            for run_seed in seeds:
                rows = [indexed[method][(benchmark, backbone, run_seed, task_id)] for task_id in task_ids]
                by_seed.append({
                    "seed": run_seed,
                    "num_tasks": len(rows),
                    "success_rate": math.fsum(row.success for row in rows) / len(rows),
                    "mean_utility": math.fsum(row.utility for row in rows) / len(rows),
                })
            method_summaries[method] = {
                "by_seed": by_seed,
                "success_rate": _mean_std([row["success_rate"] for row in by_seed]),
                "mean_utility": _mean_std([row["mean_utility"] for row in by_seed]),
            }
        comparisons = []
        for baseline in baselines:
            comparison: dict[str, Any] = {"baseline": baseline, "mcnemar_by_seed": []}
            for metric in ("utility", "success"):
                differences = [
                    math.fsum(
                        getattr(reference[(benchmark, backbone, run_seed, task_id)], metric)
                        - getattr(indexed[baseline][(benchmark, backbone, run_seed, task_id)], metric)
                        for run_seed in seeds
                    ) / len(seeds)
                    for task_id in task_ids
                ]
                child_seed = int(np.random.SeedSequence([seed, stream_index]).generate_state(1)[0])
                stream_index += 1
                ci = scenario_bootstrap_ci(differences, scenarios, seed=child_seed, num_resamples=num_bootstrap)
                test = paired_sign_flip_test(
                    differences, scenarios, seed=child_seed,
                    num_permutations=num_permutations, exact_max_clusters=exact_max_clusters,
                )
                test.update({"difference": ci["estimate"], "ci95": ci["ci95"], "ci_method": ci["ci_method"], "num_bootstrap": ci["num_resamples"]})
                if "ci_unavailable_reason" in ci:
                    test["ci_unavailable_reason"] = ci["ci_unavailable_reason"]
                comparison[metric] = test
                all_tests.append(test)
            for run_seed in seeds:
                pairs = [(benchmark, backbone, run_seed, task_id) for task_id in task_ids]
                test = exact_mcnemar(
                    [reference[key].success for key in pairs],
                    [indexed[baseline][key].success for key in pairs],
                )
                test["seed"] = run_seed
                comparison["mcnemar_by_seed"].append(test)
                all_tests.append(test)
            comparisons.append(comparison)
        stratum_results.append({
            "benchmark": benchmark, "backbone": backbone,
            "seeds": seeds, "num_seeds": len(seeds),
            "num_tasks": len(task_ids), "num_scenarios": len(set(scenarios)),
            "num_seed_task_pairs": len(stratum_keys),
            "registered_methods": stratum_methods,
            "methods": method_summaries, "comparisons": comparisons,
        })
    adjusted = holm_adjust([test["p_value"] for test in all_tests])
    for test, p_adjusted in zip(all_tests, adjusted):
        test["p_holm"] = p_adjusted
        test["reject_at_alpha"] = p_adjusted <= alpha
    return {
        "schema_version": 1,
        "reference_method": reference_method,
        "protocol": {
            "difference_direction": "reference_minus_baseline",
            "stratification": ["benchmark", "backbone"],
            "method_scope": "strict_full_reference" if method_backbones is None else "pre_registered_backbones",
            "method_backbones": {method: sorted(scopes[method]) for method in method_ids},
            "estimand": "task_equal_mean_over_registered_seeds",
            "resampling_unit": "native_scenario_with_all_tasks_and_seeds",
            "seed_population_inference": False,
            "success_primary_test": "scenario_paired_sign_flip_two_sided",
            "mcnemar_reporting": "separate_binary_task_pairs_for_each_seed",
            "ci_coverage": "marginal_95_percent",
            "multiple_testing": "holm_all_reported_tests_across_all_strata",
            "holm_family_size": len(all_tests), "alpha": alpha,
            "random_seed": seed,
        },
        "strata": stratum_results,
    }
