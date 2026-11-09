"""Collect the registered native-v1 site shards into one comparison family.

Inputs are already-transferred run directories. Collection never launches work,
changes a source run, or computes partial-family inferential statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


_HOST_CONFIG_FIELDS = {
    "site_id", "execution_host", "project_root", "python_executable", "external_root",
    "appworld_data_root", "planbench_repo_root", "output_root", "scheduler",
    "alfworld_data_root", "alfworld_repo_root", "alfworld_python_executable",
    "clindiag_pos_repo_root", "clindiag_data_archive",
    "loca_pos_repo_root", "loca_python_executable", "loca_runtime_paths", "loca_runtime_roots",
    "loca_bwrap_executable",
    "activation_offload_cpu", "activation_offload_min_bytes", "backbones",
    "official_repos", "execution_policy", "defer_global_statistics",
    "gpu", "gpu_id", "allowed_gpu_ids", "required_gpu",
    "coordination", "execution_backbones",
}
_RESOURCE_FIELDS = (
    "env_steps", "replayed_env_steps", "input_tokens", "output_tokens", "forward_tokens",
    "total_model_tokens", "model_seconds", "environment_seconds", "restore_seconds",
    "snapshot_saves", "snapshot_restores", "snapshot_save_seconds", "gradient_steps",
    "update_seconds", "wall_seconds",
)


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_immutable(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_encoded(value))


def _experiment_description(manifest: Mapping[str, Any]) -> dict[str, Any]:
    config = manifest["config"]
    return {"parameters": dict(config), "data_splits": {
        benchmark: {"splits": {split: row["tasks"] for split, row in data["splits"].items()},
                    **({"identity": {key: data[key] for key in (
                        "scheme", "package_version", "split_seed", "archive_sha256",
                        "test_manifest_sha256", "upstream_source_sha256",
                    ) if key in data}} if benchmark in {"alfworld", "clindiag", "loca"} else {})}
        for benchmark, data in manifest["data_splits"].items()},
        "upstream_versions": manifest.get("upstream_commits", {})}


def _method_backbones(config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Use only the declared main/ablation scope, never observed result coverage."""
    methods, ablations = config["methods"], config["ablation_methods"]
    mechanism = config["mechanism_backbone"]
    return {method: [mechanism] if method in ablations else list(config["global_backbone_ids"])
            for method in methods}


def _planned_sites(config: Mapping[str, Any]) -> set[str]:
    return {work["site_id"] for work in config["workload_plan"]}


def _workload_cohorts(config: Mapping[str, Any]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    """Each model/seed/environment cohort has exactly one registered owner."""
    cohorts = {}
    for work in config["workload_plan"]:
        for benchmark in work["benchmarks"]:
            key = (work["backbone"], work["seed"], benchmark)
            if key in cohorts:
                raise ValueError(f"Multiply assigned cohort: {key!r}")
            cohorts[key] = work
    return cohorts


def _evaluation_resources(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals, observed = {}, {}
    calls: dict[str, int] = {}
    for field in _RESOURCE_FIELDS:
        values = [record["resource_usage"][field] for record in records
                  if field in record.get("resource_usage", {})]
        if values:
            totals[field] = sum(values) if all(isinstance(value, int) for value in values) else math.fsum(values)
            observed[field] = len(values)
    for record in records:
        for role, count in record.get("resource_usage", {}).get("calls_by_role", {}).items():
            calls[role] = calls.get(role, 0) + count
    return {"scope": "sum_of_test_episode_usage", "num_records": len(records),
            "totals": totals, "records_with_each_field": observed, "calls_by_role": calls,
            "wall_seconds_meaning": "sum_of_episode_wall_times_not_host_makespan"}


def _training_outcome(directory: Path) -> dict[str, Any]:
    phase = _read(directory / "training_phase.json")
    training = phase.get("training", {})
    return {
        "parameter_training": phase.get("parameter_training", {"status": "applicable"}),
        **{key: training.get(key) for key in (
            "learning_status", "updates", "nonzero_gradient_steps", "parameter_change_steps",
            "memory_update_status", "memory_changed_episodes", "stop_reason", "training_usage",
        )},
    }


def _feedback_tasks(parameters: Mapping[str, Any], dev_tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reproduce the registered scenario-round-robin selection using IDs only."""
    selection = parameters["feedback_diagnostic"]
    count = int(selection["num_tasks"])
    if count <= 0 or len(dev_tasks) < count:
        raise ValueError("The registered feedback subset exceeds the development task list.")
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, task in enumerate(dev_tasks):
        groups.setdefault(task["scenario_id"], []).append(dict(task, original_index=index))
    rng = random.Random(int(selection["selection_seed"]))
    scenarios = sorted(groups)
    rng.shuffle(scenarios)
    for scenario in scenarios:
        rng.shuffle(groups[scenario])
    chosen, depth = [], 0
    while len(chosen) < count:
        for scenario in scenarios:
            if depth < len(groups[scenario]):
                chosen.append(groups[scenario][depth])
                if len(chosen) == count:
                    break
        depth += 1
    return chosen


def _feedback_statistics(samples: Sequence[Mapping[str, Any]], parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Secondary two-environment family, conditional on observed legal anchors."""
    from .statistics import holm_adjust, paired_sign_flip_test, scenario_bootstrap_ci
    settings = parameters["statistics"]
    results = []
    for index, benchmark in enumerate(parameters["benchmarks"]):
        rows = [row for row in samples if row["benchmark"] == benchmark]
        task_means, by_seed = [], []
        for task_id in sorted({row["task_id"] for row in rows}):
            task_rows = [row for row in rows if row["task_id"] == task_id]
            eligible = [row for row in task_rows if row["status"] == "complete_pair"]
            means = {name: math.fsum(row[name] for row in eligible) / len(eligible) if eligible else None
                     for name in ("delta_initial", "delta_final", "delta_change")}
            task_means.append({"task_id": task_id, "scenario_id": task_rows[0]["scenario_id"],
                               "eligible_seeds": [row["seed"] for row in eligible],
                               "num_eligible_seeds": len(eligible), **means})
        for seed in parameters["seeds"]:
            seed_rows = [row for row in rows if row["seed"] == seed]
            eligible = [row for row in seed_rows if row["status"] == "complete_pair"]
            by_seed.append({"seed": seed, "num_registered_tasks": len(seed_rows),
                            "num_eligible_tasks": len(eligible),
                            "num_no_anchor": sum(row["status"] == "no_anchor" for row in seed_rows),
                            "mean_delta_initial": math.fsum(row["delta_initial"] for row in eligible) / len(eligible)
                                                  if eligible else None,
                            "mean_delta_final": math.fsum(row["delta_final"] for row in eligible) / len(eligible)
                                                if eligible else None,
                            "mean_delta_change": math.fsum(row["delta_change"] for row in eligible) / len(eligible)
                                                 if eligible else None})
        available = [row for row in task_means if row["num_eligible_seeds"]]
        seed_values = [row["mean_delta_change"] for row in by_seed if row["mean_delta_change"] is not None]
        seed_mean = math.fsum(seed_values) / len(seed_values) if seed_values else None
        seed_std = (math.sqrt(math.fsum((value - seed_mean) ** 2 for value in seed_values) / (len(seed_values) - 1))
                    if len(seed_values) > 1 else None)
        result: dict[str, Any] = {
            "benchmark": benchmark, "backbone": parameters["mechanism_backbone"],
            "num_registered_tasks": len(task_means), "num_eligible_tasks": len(available),
            "num_registered_seed_task_pairs": len(rows),
            "num_eligible_seed_task_pairs": sum(row["status"] == "complete_pair" for row in rows),
            "num_no_anchor": sum(row["status"] == "no_anchor" for row in rows),
            "task_coverage": len(available) / len(task_means),
            "anchor_coverage": sum(row["status"] == "complete_pair" for row in rows) / len(rows),
            "task_means": task_means, "by_seed": by_seed,
            "seed_mean_std": {"mean": seed_mean, "std": seed_std, "num_eligible_seeds": len(seed_values)},
            "estimate": None, "ci95": None, "p_value": None, "p_holm": None,
            "mean_delta_initial": None, "mean_delta_final": None,
            "reject_at_alpha": None, "status": "no_eligible_anchors",
        }
        if available:
            differences = [row["delta_change"] for row in available]
            scenarios = [row["scenario_id"] for row in available]
            random_seed = int(settings["seed"]) + 2000003 + index
            ci = scenario_bootstrap_ci(differences, scenarios, seed=random_seed,
                                        num_resamples=int(settings["num_bootstrap"]))
            test = paired_sign_flip_test(differences, scenarios, seed=random_seed,
                                          num_permutations=int(settings["num_permutations"]),
                                          exact_max_clusters=int(settings.get("exact_max_clusters", 16)))
            result.update(test, ci95=ci["ci95"], ci_method=ci["ci_method"],
                          num_bootstrap=ci["num_resamples"], status="estimated",
                          mean_delta_initial=math.fsum(row["delta_initial"] for row in available) / len(available),
                          mean_delta_final=math.fsum(row["delta_final"] for row in available) / len(available))
            if "ci_unavailable_reason" in ci:
                result["ci_unavailable_reason"] = ci["ci_unavailable_reason"]
        results.append(result)
    # A non-estimable hypothesis retains its registered family slot. Its reported
    # p-value remains null; no zero effect or artificial eligible task is inserted.
    adjusted = holm_adjust([row["p_value"] if row["p_value"] is not None else 1.0 for row in results])
    for row, p_value in zip(results, adjusted):
        if row["p_value"] is not None:
            row.update(p_holm=p_value, reject_at_alpha=p_value <= settings["alpha"])
    return results


def _collect_feedback(shards: Mapping[str, Any], contract: Mapping[str, Any], experiment_id: str,
                      output_dir: Path, cohort_shards: Mapping[Any, Any] | None = None) -> dict[str, Any]:
    """Read the fixed theta0-anchor measurement; partial feedback leaves main inference free."""
    parameters = contract["parameters"]
    selection = parameters["feedback_diagnostic"]
    if not selection["enabled"]:
        report = {
            "experiment_id": experiment_id, "analysis_family": "secondary_feedback_change",
            "status": "not_requested", "complete": True, "num_registered_seed_task_pairs": 0,
            "num_complete_pairs": 0, "num_no_anchor": 0, "num_partial": 0, "num_missing": 0,
            "measurements": [], "samples": [], "results": None, "anchor_coverage": None,
        }
        _write_progress(output_dir / "feedback_change_comparisons.json", report)
        return report
    backbone = parameters["mechanism_backbone"]
    cohorts = _workload_cohorts(parameters)
    feedback_sites = {work["site_id"] for (model, _, _), work in cohorts.items() if model == backbone}
    anchor_distribution = "initial_policy_first_nonterminal_information_action_on_registered_dev_tasks"
    samples, measurements, identities = [], [], []
    endpoints = ("initial_visible_utility", "initial_masked_utility", "final_visible_utility", "final_masked_utility")
    for benchmark in parameters["benchmarks"]:
        tasks = _feedback_tasks(parameters, contract["data_splits"][benchmark]["splits"]["dev"])
        for seed in parameters["seeds"]:
            work = cohorts[(backbone, seed, benchmark)]
            registered_site = work["site_id"]
            shard = (cohort_shards.get((backbone, seed, benchmark)) if cohort_shards is not None
                     else shards.get(registered_site))
            source_site = shard[1]["site_id"] if shard is not None else registered_site
            source_host = shard[1]["host"] if shard is not None else None
            actual_gpu = shard[3] if shard is not None and len(shard) == 4 else work["gpu"]
            registered = [dict(
                task, task_id=str(task["task_id"]), scenario_id=str(task["scenario_id"]), split="dev",
                environment_seed=int(task.get("environment_seed", int(seed) * 1000003 + task["original_index"])),
                actor_seed=int(seed) * 1000003 + task["original_index"],
            ) for task in tasks]
            unit: dict[str, Any] = {"site_id": source_site, "host": source_host, "registered_gpu": work["gpu"],
                                   "registered_site": registered_site,
                                   "physical_gpu": None, "benchmark": benchmark, "backbone": backbone,
                                   "seed": seed, "status": "missing", "complete": False,
                                   "num_registered_tasks": len(registered), "errors": []}
            rows_by_task: dict[str, Any] = {}
            identity = {"site_id": source_site, "benchmark": benchmark, "seed": seed, "files": {}}
            if shard is not None:
                directory, manifest, schedule = shard[:3]
                identity["manifest_id"] = manifest["manifest_id"]
                identity["execution_id"] = manifest.get("execution", {}).get("execution_id")
                unit["execution_id"] = identity["execution_id"]
                bundle = next(item for item in schedule if item["backbone_id"] == backbone and item["seed"] == seed
                              and benchmark in item["benchmarks"])
                folder = directory / "feedback" / bundle["bundle_id"] / benchmark
                paths = {"status": folder / "status.json",
                         "protocol": folder / "measurement" / "feedback_change_protocol.json",
                         "records": folder / "measurement" / "feedback_change_records.jsonl"}
                identity["files"] = {name: str(path) for name, path in paths.items()}
                unit["bundle_id"] = bundle["bundle_id"]
                try:
                    receipt, protocol = _read(paths["status"]), _read(paths["protocol"])
                    if any(receipt.get(key) != value for key, value in {
                        "manifest_id": manifest["manifest_id"], "host": source_host,
                        "site_id": source_site, "physical_gpu": actual_gpu,
                        "bundle_id": bundle["bundle_id"], "benchmark": benchmark, "backbone": backbone, "seed": seed,
                    }.items()):
                        raise ValueError("Feedback receipt identity differs from its registered run.")
                    if (any(protocol.get(key) != value for key, value in {
                            "benchmark": benchmark, "backbone": backbone, "seed": seed, "method": "prism_evolve",
                            "selection_seed": selection["selection_seed"], "num_tasks": len(registered),
                            "anchor_distribution": anchor_distribution,
                        }.items()) or protocol.get("tasks") != registered):
                        raise ValueError("Feedback protocol differs from the registered initial-anchor task scope.")
                    unit.update(status=receipt.get("status"), physical_gpu=receipt["physical_gpu"],
                                resource_usage=receipt.get("resource_usage"),
                                initial_checkpoint=protocol.get("initial_checkpoint"), final_checkpoint=protocol.get("final_checkpoint"))
                    expected = {row["task_id"]: row for row in registered}
                    for line in paths["records"].read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        task = expected.get(row.get("task_id"))
                        if (task is None or row["task_id"] in rows_by_task
                                or any(row.get(key) != value for key, value in {
                                    "benchmark": benchmark, "backbone": backbone, "seed": seed,
                                    "scenario_id": task["scenario_id"], "split": "dev",
                                    "original_index": task["original_index"], "environment_seed": task["environment_seed"],
                                    "actor_seed": task["actor_seed"],
                                }.items()) or row.get("method", "prism_evolve") != "prism_evolve"):
                            raise ValueError("Feedback record has a duplicate or mismatching dataset/method/seed/task identity.")
                        state = row.get("status")
                        if state == "complete_pair":
                            values = [row.get(key) for key in endpoints]
                            if (row.get("anchor_found") is not True or not isinstance(row.get("information_step"), int)
                                    or row["information_step"] < 0
                                    or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                                           or not 0 <= value <= 1 for value in values)):
                                raise ValueError("A complete feedback pair requires one shared initial anchor and four finite endpoints.")
                            initial, final = values[0] - values[1], values[2] - values[3]
                            for key, value in (("delta_initial", initial), ("delta_final", final), ("delta_change", final - initial)):
                                if row.get(key) is None or not math.isclose(float(row[key]), value, rel_tol=1e-9, abs_tol=1e-12):
                                    raise ValueError("Feedback delta differs from its paired endpoint utilities.")
                        elif state == "no_anchor":
                            if row.get("anchor_found") is not False or any(row.get(key) is not None for key in (*endpoints, "delta_initial", "delta_final", "delta_change")):
                                raise ValueError("A no_anchor task must retain null paired outcomes.")
                        elif state != "partial":
                            raise ValueError("Unknown feedback task status.")
                        rows_by_task[row["task_id"]] = row
                    unit["complete"] = receipt.get("status") == "done" and len(rows_by_task) == len(registered) and all(
                        row["status"] in {"complete_pair", "no_anchor"} for row in rows_by_task.values())
                except (OSError, ValueError, KeyError, TypeError) as error:
                    unit["errors"].append(f"{type(error).__name__}: {error}")
                    unit["complete"] = False
            unit_samples = []
            for task in registered:
                row = rows_by_task.get(task["task_id"])
                sample = {**task, "benchmark": benchmark, "backbone": backbone, "seed": seed,
                          "site_id": source_site, "host": source_host, "physical_gpu": unit["physical_gpu"],
                          "status": row["status"] if row else "missing",
                          "anchor_found": row.get("anchor_found") if row else None,
                          "information_step": row.get("information_step") if row else None,
                          **{name: row.get(name) if row and row["status"] == "complete_pair" else None
                             for name in (*endpoints, "delta_initial", "delta_final", "delta_change")}}
                if row:
                    for name in ("traces", "reason", "initial_natural_utility", "initial_natural_success", "initial_natural_termination"):
                        if name in row:
                            sample[name] = row[name]
                unit_samples.append(sample)
            unit.update(num_complete_pairs=sum(row["status"] == "complete_pair" for row in unit_samples),
                        num_no_anchor=sum(row["status"] == "no_anchor" for row in unit_samples),
                        num_partial=sum(row["status"] == "partial" for row in unit_samples),
                        num_missing=sum(row["status"] == "missing" for row in unit_samples))
            samples.extend(unit_samples)
            measurements.append(unit)
            identities.append(identity)
    complete = all(unit["complete"] for unit in measurements)
    inputs = {"measurements": identities,
              "source_manifest_ids": {site: shards[site][1]["manifest_id"] for site in shards}}
    report: dict[str, Any] = {
        "schema_version": 1, "experiment_id": experiment_id, "analysis_family": "secondary_feedback_change",
        "complete": complete, "status": "complete" if complete else "partial",
        "inputs": inputs, "backbone": backbone, "method": "prism_evolve", "split": "dev",
        "anchor_distribution": anchor_distribution,
        "estimand": "task_equal_change_averaged_over_available_initial_anchor_seeds",
        "claim_scope": "registered_dev_tasks_with_at_least_one_legal_initial_policy_anchor",
        "no_anchor_treatment": "report_coverage_and_null_outcomes_without_replacement_or_zero_imputation",
        "seed_population_inference": False, "holm_family_size": len(parameters["benchmarks"]),
        "hardware_scope": "initial_final_visible_masked_pairs_share_the_actual_model_seed_environment_host_gpu_cohort",
        "multiple_testing": "separate_secondary_family_one_delta_change_test_per_environment",
        "primary_family_included": False, "alpha": parameters["statistics"]["alpha"],
        "num_registered_seed_task_pairs": len(samples),
        "num_complete_pairs": sum(row["status"] == "complete_pair" for row in samples),
        "num_no_anchor": sum(row["status"] == "no_anchor" for row in samples),
        "num_partial": sum(row["status"] == "partial" for row in samples),
        "num_missing": sum(row["status"] == "missing" for row in samples),
        "measurements": measurements, "samples": samples, "results": None,
    }
    report["anchor_coverage"] = report["num_complete_pairs"] / len(samples) if complete else None
    target = output_dir / "feedback_change_comparisons.json"
    if complete:
        report["results"] = _feedback_statistics(samples, parameters)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_bytes(_encoded(report))
    temporary.replace(target)
    return report


def _write_progress(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(_encoded(value))
    temporary.replace(path)


def collect_runs(run_dirs: Sequence[Path], output_dir: Path, coordination_registry: Path | None = None) -> dict[str, Any]:
    """Collect existing records by run, execution and cell identity."""
    from .native_coordination import cohort_key
    if not run_dirs or coordination_registry is None:
        raise ValueError("Provide source runs and their coordination registry.")
    registry = _read(coordination_registry)
    output_dir = output_dir.resolve()
    sources, shards, site_resources, expected = {}, {}, {}, {}
    execution_paths, seen_paths = [], set()
    contract, experiment_id = None, None
    for supplied in run_dirs:
        directory = Path(supplied).resolve()
        if directory == output_dir or directory in output_dir.parents:
            raise ValueError("Collection output must be separate from original runs.")
        manifest = _read(directory / "manifest.json")
        if "execution" in manifest:
            execution_paths.append(directory)
            continue
        site = manifest["site_id"]
        if site in sources:
            raise ValueError("Duplicate origin site in the collection.")
        if experiment_id is None:
            experiment_id, contract = manifest["experiment_id"], _experiment_description(manifest)
        elif experiment_id != manifest["experiment_id"]:
            raise ValueError("Source runs belong to different experiments.")
        else:
            other = _experiment_description(manifest)
            for benchmark in {"alfworld", "clindiag", "loca"} & set(contract["data_splits"]):
                if other["data_splits"].get(benchmark) != contract["data_splits"][benchmark]:
                    raise ValueError(f"Source sites registered different {benchmark} datasets.")
        schedule = _read(directory / "schedule.json")
        source = {"site_id": site, "host": manifest["host"], "run_id": manifest["run_id"],
                  "manifest_id": manifest["manifest_id"], "collected_run_dir": str(directory),
                  "original_run_dir": manifest["run_dir"],
                  "backbone_ids": [model["id"] for model in manifest["config"]["backbones"]]}
        sources[site], shards[site] = source, (directory, manifest, schedule)
        site_resources[site] = {"site_id": site, "host": manifest["host"],
                                "runtime": manifest["runtime"], "config": manifest["config"], "cells": []}
        for bundle in schedule:
            for cell in bundle["cells"]:
                cell_id = cell["cell_id"]
                if cell_id in expected:
                    raise ValueError("A logical cell is assigned more than once.")
                expected[cell_id] = dict(cell, bundle_id=bundle["bundle_id"],
                                        backbone_id=bundle["backbone_id"], registered_site=site,
                                        registered_gpu=bundle["required_gpu"])
        execution_paths.extend(path.parent for path in sorted((directory / "executions").glob("*/manifest.json")))
    if contract is None:
        raise ValueError("Include the origin run directories.")
    pending, completed, records_by_method = [], [], {}
    cohort_shards, execution_sources, seen_cells = {}, {}, set()
    for directory in execution_paths:
        if directory in seen_paths:
            continue
        seen_paths.add(directory)
        manifest = _read(directory / "manifest.json")
        schedule = _read(directory / "schedule.json")
        grant = manifest["execution"]
        execution_id = grant["execution_id"]
        if execution_id in execution_sources:
            raise ValueError("An execution is present in more than one source directory.")
        job = registry["jobs"][grant["job_key"]]
        if job.get("execution_id") != execution_id or manifest["manifest_id"] != execution_id:
            raise ValueError("Results belong to another registered execution.")
        bundle = schedule[0]
        execution_sources[execution_id] = {**{key: grant[key] for key in
            ("execution_id", "job_key", "site_id", "host", "gpu", "gpu_uuid", "origin_site",
             "planned_gpu", "backbone_id", "seed", "benchmarks")}, "collected_run_dir": str(directory)}
        for benchmark in bundle["benchmarks"]:
            key = (bundle["backbone_id"], bundle["seed"], benchmark)
            if key in cohort_shards:
                raise ValueError("A cohort cannot have duplicate result executions.")
            cohort_shards[key] = (directory, manifest, schedule, grant["gpu"])
        for cell in bundle["cells"]:
            cell_id = cell["cell_id"]
            if cell_id not in expected or cell_id in seen_cells:
                raise ValueError("Duplicate or unregistered cell result.")
            seen_cells.add(cell_id)
            entry = dict(expected[cell_id], site_id=grant["site_id"], host=grant["host"],
                         physical_gpu=grant["gpu"], gpu_uuid=grant["gpu_uuid"], execution_id=execution_id,
                         manifest_id=manifest["manifest_id"])
            folder = directory / "cells" / cell_id
            status_path, records_path = folder / "status.json", folder / "test_records.json"
            if not status_path.exists():
                pending.append(dict(entry, status="not_started"))
                continue
            receipt = _read(status_path)
            if receipt.get("cell_id") != cell_id or receipt.get("manifest_id") != manifest["manifest_id"]:
                raise ValueError("A cell receipt belongs to another execution.")
            if job["status"] != "done" or receipt["status"] != "done" or not records_path.exists():
                pending.append(dict(entry, status=receipt["status"], registry_status=job["status"],
                                    phase=receipt.get("phase"), error=receipt.get("error")))
                continue
            records = _read(records_path)
            records_by_method.setdefault(cell["method"], []).extend(records)
            completed.append(dict(entry, num_records=len(records), records_file=str(records_path),
                                  training_file=str(folder / "training_phase.json")))
            site_resources[grant["site_id"]]["cells"].append({
                **entry, "receipt": receipt, "training_outcome": _training_outcome(folder),
                "evaluation_usage": _evaluation_resources(records)})
    for cell_id in expected.keys() - seen_cells:
        pending.append(dict(expected[cell_id], site_id=expected[cell_id]["registered_site"],
                            status="not_started_or_not_transferred"))
    return _finish_collection(sources, contract, experiment_id, records_by_method, pending, completed,
                              site_resources, shards, output_dir, cohort_shards=cohort_shards,
                              execution_sources=execution_sources)


def _finish_collection(sources, contract, experiment_id, records_by_method, pending, completed,
                       site_resources, shards, output_dir, *, cohort_shards=None, execution_sources=None,
                       ):
    assert contract is not None
    parameters = contract["parameters"]
    expected_sites = _planned_sites(parameters)
    missing_sites = sorted(expected_sites - sources.keys())
    missing_models = sorted(set(parameters["global_backbone_ids"]) -
                            {model for source in sources.values() for model in source["backbone_ids"]})
    method_scope = _method_backbones(parameters)
    planned_cells = [(work, benchmark, method) for work in parameters["workload_plan"]
                     for benchmark in work["benchmarks"] for method, models in method_scope.items()
                     if work["backbone"] in models]
    expected_cells = len(planned_cells)
    expected_test_episodes = sum(
        len(contract["data_splits"][benchmark]["splits"][split])
        for _, benchmark, _ in planned_cells
        for split in (parameters["appworld_test_splits"] if benchmark == "appworld" else ["test"]))
    completed_test_episodes = sum(cell["num_records"] for cell in completed)
    completed.sort(key=lambda cell: (cell["site_id"], cell["cell_id"]))
    pending.sort(key=lambda cell: (cell["site_id"], cell["cell_id"]))
    ready = (not missing_sites and not missing_models and not pending and len(completed) == expected_cells
             and completed_test_episodes == expected_test_episodes)
    feedback = _collect_feedback(shards, contract, experiment_id, output_dir, cohort_shards=cohort_shards)
    completeness = {
        "schema_version": 1, "experiment_id": experiment_id, "run_kind": "formal",
        "expected_sites": sorted(expected_sites), "missing_sites": missing_sites,
        "actual_hosts": {site: source["host"] for site, source in sorted(sources.items())},
        "global_backbone_ids": list(parameters["global_backbone_ids"]), "missing_backbones": missing_models,
        "method_backbones": method_scope,
        "source_manifest_ids": {site: sources[site]["manifest_id"] for site in sorted(sources)},
        "expected_cells": expected_cells, "completed_cells": len(completed),
        "pending_cells": expected_cells - len(completed), "completed": completed, "pending": pending,
        "expected_test_episodes": expected_test_episodes, "completed_test_episodes": completed_test_episodes,
        "pending_test_episodes": expected_test_episodes - completed_test_episodes,
        "formal_inference_ready": ready,
        "primary_comparison_family_complete": ready,
        "feedback_measurements_complete": feedback["complete"],
        "feedback_registered_seed_task_pairs": feedback["num_registered_seed_task_pairs"],
        "feedback_complete_pairs": feedback["num_complete_pairs"],
        "feedback_no_anchor": feedback["num_no_anchor"],
        "feedback_pending_seed_task_pairs": feedback["num_partial"] + feedback["num_missing"],
        "feedback_anchor_coverage": feedback["anchor_coverage"],
        "feedback_measurements": feedback["measurements"],
        "overall_experiment_complete": ready and feedback["complete"],
    }
    manifest_target = output_dir / "collection_manifest.json"
    result_target = output_dir / "global_method_comparisons.json"
    inputs = {"sources": list(sources.values()), "completed": completed}
    if execution_sources is not None:
        inputs["execution_sources"] = execution_sources
        completeness["execution_sources"] = execution_sources
    _write_progress(output_dir / "global_completeness.json", completeness)
    if not ready:
        return completeness
    if manifest_target.exists() or result_target.exists():
        raise FileExistsError("Final comparison files already exist; retain them and select a new output directory.")
    from . import statistics
    official_strata = {}
    for method, rows in records_by_method.items():
        for row in rows:
            key = (method, row["backbone"], row["benchmark"], row["seed"], row.get("split"))
            official_strata.setdefault(key, []).append(row)
    official_summaries = []
    for (method, backbone, benchmark, seed, split), rows in sorted(official_strata.items(), key=lambda item: str(item[0])):
        base_benchmark, _, embedded_split = benchmark.partition(":")
        selected_split = split or embedded_split or "test"
        # _experiment_description has already extracted the complete row["tasks"] list.
        tasks = contract["data_splits"][base_benchmark]["splits"][selected_split]
        official_summaries.append({"method": method, "backbone": backbone, "benchmark": benchmark,
                                   "seed": seed, "split": selected_split,
                                   **statistics.official_benchmark_summary(rows, tasks)})
    result = statistics.evaluate_method_comparisons(
        records_by_method, method_backbones=method_scope, **parameters["statistics"],
    )
    result["official_benchmark_metrics"] = official_summaries
    result["protocol"]["existing_success_metric"] = "budget_compliant_success; native evaluator metrics reported separately"
    collection = {
        "schema_version": 1, "experiment_id": experiment_id, "run_kind": "formal",
        "inputs": inputs,
        "sources": [source for _, source in sorted(sources.items())],
        "parameter_contract": contract, "site_resources": site_resources,
        "method_backbones": method_scope,
        "site_strata": {site: [{"site_id": site, "host": sources[site]["host"], "backbone": work["backbone"],
                                "seed": work["seed"], "registered_gpu": work["gpu"], "benchmark": benchmark, "split": split,
                                "methods": [method for method, models in method_scope.items() if work["backbone"] in models]}
                               for work in parameters["workload_plan"] if work["site_id"] == site
                               for benchmark in work["benchmarks"]
                               for split in (parameters["appworld_test_splits"] if benchmark == "appworld" else ["test"])]
                        for site in sorted(sources)},
        "hardware_scope": "method_pairs_share_registered_backbone_seed_environment_site_gpu_cohorts; cross_cohort_hardware_differs",
        "holm_family_size": result["protocol"]["holm_family_size"],
        "analysis_families": {
            "primary": {"file": "global_method_comparisons.json", "scope": "registered_method_comparisons"},
            "secondary": {"file": "feedback_change_comparisons.json", "holm_family_size": len(parameters["benchmarks"]),
                          "scope": "initial_to_final_prism_feedback_change_on_eligible_registered_dev_tasks"},
        },
    }
    if execution_sources is not None:
        collection["execution_sources"] = execution_sources
        collection["site_strata"] = {site: [
            {"site_id": site, "host": execution["host"], "backbone": execution["backbone_id"],
             "seed": execution["seed"], "benchmark": benchmark, "split": split,
             "registered_site": execution["origin_site"], "registered_gpu": execution["planned_gpu"],
             "physical_gpu": execution["gpu"], "gpu_uuid": execution["gpu_uuid"],
             "execution_id": execution["execution_id"],
             "methods": [method for method, models in method_scope.items() if execution["backbone_id"] in models]}
            for execution in execution_sources.values() if execution["site_id"] == site
            for benchmark in execution["benchmarks"]
            for split in (parameters["appworld_test_splits"] if benchmark == "appworld" else ["test"])
        ] for site in expected_sites}
        collection["hardware_scope"] = "method_pairs_share_actual_cohort_host_gpu; original_placement_and_transfers_are_preserved"
    _write_immutable(result_target, result)
    _write_immutable(manifest_target, collection)
    return completeness


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the native experiment sites declared in the workload plan.")
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coordination-registry", type=Path)
    args = parser.parse_args()
    result = collect_runs(args.run_dirs, args.output_dir, args.coordination_registry)
    summary = {key: result[key] for key in ("experiment_id", "formal_inference_ready", "expected_cells",
                                           "completed_cells", "pending_cells", "expected_test_episodes",
                                           "completed_test_episodes", "pending_test_episodes",
                                           "missing_sites", "missing_backbones", "feedback_measurements_complete",
                                           "feedback_complete_pairs", "feedback_no_anchor", "overall_experiment_complete")}
    summary["output_dir"] = str(args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
