#!/usr/bin/env python3
"""Export saved three-backbone results for paper writing; run from the repo root."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


BENCHMARKS = {"appworld": "part2_appworld", "planbench_xl": "part4_planbench_xl"}
PEERS = ("gigpo", "step_dpo", "infopo", "arew", "worldevolver", "pos_envprobe")


def pointer(*parts: object) -> str:
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def mean(value: dict | float) -> float:
    return value["mean"] if isinstance(value, dict) else value


def endpoints(ours: float, reference: float, unit: str) -> dict:
    out = {"ours": ours, "reference": reference, "difference": ours - reference, "unit": unit}
    if unit == "utility":
        out["difference_pts"] = 100 * (ours - reference)
    elif unit == "rate":
        out["difference_percentage_points"] = 100 * (ours - reference)
    return out


def method_comparison(model: str, benchmark: str, methods: dict, reference: str, role: str) -> dict:
    ours, ref = methods["prism_evolve"], methods[reference]
    part = BENCHMARKS[benchmark]
    cost = endpoints(mean(ours["final_test"]["mean_cumulative_cost"]),
                     mean(ref["final_test"]["mean_cumulative_cost"]), "macro_action_cost")
    cost["reduction_percent"] = (100 * (cost["reference"] - cost["ours"]) / cost["reference"]
                                 if cost["reference"] else None)
    return {
        "model": model, "benchmark": benchmark, "ours_method": "prism_evolve",
        "reference_method": reference, "reference_role": role,
        "source_pointers": {
            "ours": pointer("backbones", model, part, "methods", "prism_evolve"),
            "reference": pointer("backbones", model, part, "methods", reference),
        },
        "utility": endpoints(mean(ours["final_test"]["mean_utility"]),
                             mean(ref["final_test"]["mean_utility"]), "utility"),
        "success_rate": endpoints(mean(ours["final_test"]["success_rate"]),
                                  mean(ref["final_test"]["success_rate"]), "rate"),
        "cost": cost,
        "dev_delta_use": endpoints(mean(ours["final_dev"]["delta_use"]),
                                   mean(ref["final_dev"]["delta_use"]), "utility"),
    }


def integer_count(value: float, context: str) -> int:
    result = round(value)
    if not math.isclose(value, result, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"Non-integer reconstructed count at {context}: {value}")
    return result


def export(source: dict, source_file: str) -> tuple[dict, dict]:
    metrics = {"source_file": source_file, "config": source["config"],
               "environment": source["environment"], "main_results": [], "seed_results": [],
               "stage_results": [], "training_progress": [], "budgets": [], "split_steps": [],
               "cost_ledgers": [], "backbone_metadata": [],
               "exact_verification": {"source_pointer": "/part1_verification",
                                      "values": source["part1_verification"]}}
    comparisons = {"source_file": source_file, "method_comparisons": [], "stage_changes": [],
                   "budget_vs_unweighted": [], "split_vs_root": [], "trajectory_outcomes": []}

    for model, backbone in source["backbones"].items():
        metrics["backbone_metadata"].append({
            "model": model, "source_pointer": pointer("backbones", model),
            "llm_backbone": backbone["llm_backbone"], "elapsed_sec": backbone["elapsed_sec"],
        })
        for benchmark, part in BENCHMARKS.items():
            section = backbone[part]
            methods = section["methods"]
            for method, aggregate in methods.items():
                identity = {"model": model, "benchmark": benchmark, "method": method}
                base = ("backbones", model, part)
                metrics["main_results"].append({
                    **identity, "source_pointer": pointer(*base, "methods", method),
                    "num_seeds": aggregate["num_seeds"], "split_sizes": section["split_sizes"],
                    "final_test": aggregate["final_test"], "final_dev": aggregate["final_dev"],
                    "p_root_init": aggregate["p_root_init"], "p_root_final": aggregate["p_root_final"],
                })
                metrics["cost_ledgers"].append({
                    **identity, "source_pointer": pointer(*base, "methods", method, "cost_ledger_mean"),
                    "num_seeds": aggregate["num_seeds"], "values": aggregate["cost_ledger_mean"],
                })
                for index, stage in enumerate(aggregate["dev_by_stage"]):
                    metrics["stage_results"].append({
                        **identity, "source_pointer": pointer(*base, "methods", method, "dev_by_stage", index),
                        "stage_k": stage["stage_k"], "values": stage,
                    })

                pooled = {"seed_task_evaluations": 0, "successful_evaluations": 0,
                          "evaluations_with_recorded_failure": 0, "success_without_recorded_failure": 0,
                          "recovered_success": 0, "final_failure": 0}
                seed_counts = []
                for index, run in enumerate(section["seed_runs"][method]):
                    run_pointer = pointer(*base, "seed_runs", method, index)
                    metrics["seed_results"].append({
                        **identity, "seed": run["seed"], "source_pointer": run_pointer,
                        "p_root_init": run["p_root_init"], "p_root_final": run["p_root_final"],
                        "final_dev": run["final_dev"], "final_test": run["final_test"],
                        "cost_ledger": run["cost_ledger"],
                    })
                    for log_index, log in enumerate(run["train_logs"]):
                        metrics["training_progress"].append({
                            **identity, "seed": run["seed"], "stage_k": log["stage_k"],
                            "source_pointer": run_pointer + pointer("train_logs", log_index), "values": log,
                        })
                    test = run["final_test"]
                    total = test["num_tasks"]
                    success = integer_count(test["success_rate"] * total, run_pointer + "/final_test/success_rate")
                    failed_once = test["had_failure_count"]
                    recovered = integer_count(test["post_failure_recovery_rate"] * failed_once,
                                              run_pointer + "/final_test/post_failure_recovery_rate")
                    if not (0 <= recovered <= min(success, failed_once) <= total):
                        raise ValueError(f"Inconsistent trajectory counts at {run_pointer}")
                    counts = {"seed_task_evaluations": total, "successful_evaluations": success,
                              "evaluations_with_recorded_failure": failed_once,
                              "success_without_recorded_failure": success - recovered,
                              "recovered_success": recovered, "final_failure": total - success}
                    assert sum(counts[k] for k in ("success_without_recorded_failure", "recovered_success", "final_failure")) == total
                    seed_counts.append({"seed": run["seed"], "source_pointer": run_pointer + "/final_test", **counts})
                    for key, count in counts.items():
                        pooled[key] += count
                comparisons["trajectory_outcomes"].append({
                    **identity, **pooled, "seed_counts": seed_counts,
                    "evaluation_unit": "seed-task evaluation (the fixed task set is repeated across seeds)",
                    "recovery_denominator": "evaluations_with_recorded_failure",
                    "pooled_recovery_rate": (pooled["recovered_success"] / pooled["evaluations_with_recorded_failure"]
                                             if pooled["evaluations_with_recorded_failure"] else None),
                    "mean_seed_recovery_rate": mean(aggregate["final_test"]["post_failure_recovery_rate"]),
                })

            best_peer = max(PEERS, key=lambda m: mean(methods[m]["final_test"]["mean_utility"]))
            best_other = max((m for m in methods if m != "prism_evolve"),
                             key=lambda m: mean(methods[m]["final_test"]["mean_utility"]))
            for role, reference in (("frozen", "frozen_base"), ("strongest_named_peer", best_peer),
                                    ("strongest_other_method", best_other),
                                    ("unweighted", "ablation_unweighted"),
                                    ("unstratified", "unstratified_2branch"), ("outcome_rloo", "outcome_rloo")):
                comparisons["method_comparisons"].append(method_comparison(model, benchmark, methods, reference, role))
            progression = methods["prism_evolve"]["dev_by_stage"]
            comparisons["stage_changes"].append({
                "model": model, "benchmark": benchmark, "from_stage": progression[0]["stage_k"],
                "to_stage": progression[-1]["stage_k"],
                "source_pointers": {"from": pointer("backbones", model, part, "methods", "prism_evolve", "dev_by_stage", 0),
                                    "to": pointer("backbones", model, part, "methods", "prism_evolve", "dev_by_stage", len(progression)-1)},
                **{name: endpoints(mean(progression[-1][name]), mean(progression[0][name]), unit)
                   for name, unit in (("mean_utility", "utility"), ("success_rate", "rate"), ("delta_use", "utility"))},
            })

        diagnostics = backbone["part3_diagnostics"]
        for tag, budget in diagnostics["budget_sensitivity"].items():
            for method, values in budget["methods"].items():
                metrics["budgets"].append({
                    "model": model, "benchmark": "appworld", "split": "dev", "method": method,
                    "budget_tag": tag, "budget": budget["budget"],
                    "source_pointer": pointer("backbones", model, "part3_diagnostics", "budget_sensitivity", tag, "methods", method),
                    "values": values,
                })
            ours, ref = budget["methods"]["prism_evolve"], budget["methods"]["ablation_unweighted"]
            comparisons["budget_vs_unweighted"].append({
                "model": model, "benchmark": "appworld", "split": "dev", "budget_tag": tag,
                "budget": budget["budget"],
                "source_pointers": {m: pointer("backbones", model, "part3_diagnostics", "budget_sensitivity", tag, "methods", m)
                                    for m in ("prism_evolve", "ablation_unweighted")},
                "utility": endpoints(ours["mean_utility"], ref["mean_utility"], "utility"),
                "success_rate": endpoints(ours["success_rate"], ref["success_rate"], "rate"),
            })
        root = diagnostics["split_step_ablation"]["fixed_T0"]
        for schedule, values in diagnostics["split_step_ablation"].items():
            metrics["split_steps"].append({
                "model": model, "benchmark": "appworld", "schedule": schedule,
                "source_pointer": pointer("backbones", model, "part3_diagnostics", "split_step_ablation", schedule),
                "values": values, "reuses_main_result": schedule == "fixed_T0",
            })
            if schedule != "fixed_T0":
                comparisons["split_vs_root"].append({
                    "model": model, "benchmark": "appworld", "schedule": schedule,
                    "difference_direction": "schedule minus fixed_T0",
                    "source_pointers": {s: pointer("backbones", model, "part3_diagnostics", "split_step_ablation", s)
                                        for s in (schedule, "fixed_T0")},
                    "test_utility": endpoints(mean(values["final_test"]["mean_utility"]), mean(root["final_test"]["mean_utility"]), "utility"),
                    "dev_delta_use": endpoints(mean(values["final_dev"]["delta_use"]), mean(root["final_dev"]["delta_use"]), "utility"),
                })
    metrics["counts"] = {key: len(value) for key, value in metrics.items() if isinstance(value, list)}
    comparisons["counts"] = {key: len(value) for key, value in comparisons.items() if isinstance(value, list)}
    return metrics, comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/prism_evolve_experiments/prism_evolve_full_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/acl2027/evidence/data"))
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    metrics, comparisons = export(source, args.input.as_posix())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("metrics", metrics), ("comparisons", comparisons)):
        (args.output_dir / f"{name}.json").write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"metrics": metrics["counts"], "comparisons": comparisons["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
