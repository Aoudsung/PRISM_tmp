"""Report paired official WebShop outcomes from finalized study cells."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
from pathlib import Path
from statistics import fmean, stdev


PROTOCOL = "fgpo-r3-published-v4"
STUDY_PROTOCOL = "webshop-150-paired-v1"
MENU_ADAPTATION = "WebShop_public_search_menu_matches_native_commands"
SEEDS = (0, 1, 2)
TASK_COUNT = 128
METRICS = ("SR", "Score")
OUTCOMES = ("gained_full", "lost_full", "both_full", "neither_full")
TASK_FIELDS = (
    "comparison", "seed", "ordinal", "task_id", "environment_seed", "session_index", "sampling_seed",
    "reference_model_version", "target_model_version", "reference_purchased", "target_purchased",
    "reference_success", "target_success", "reference_score", "target_score", "score_delta", "outcome",
)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def require_fields(record, expected, source):
    for key, value in expected.items():
        actual = record.get(key)
        require(actual == value and (type(value) not in (bool, int) or type(actual) is type(value)),
                f"{source} has an unexpected {key}")


def task_key(task, split, worker_seed):
    session = task["session_index"]
    require(type(session) is int and (0 <= session < 500 if split == "test" else session >= 500),
            f"official {split} catalog has an invalid session_index")
    task_id = f"{split}/{worker_seed}/{session}"
    require_fields(task, {"task_id": task_id, "environment_seed": worker_seed, "split": split},
                   f"official task {task_id}")
    return task_id, worker_seed, session


def read_native_evaluation(records_path, test_tasks, evaluation):
    eval_identity = {key: evaluation[key] for key in ("protocol", "method", "seed", "benchmark", "split", "model_version")}
    eval_identity["study_protocol"] = evaluation.get("study_protocol")
    test_draw = [task_key(task, "test", evaluation["seed"] + 1000 + ordinal) for ordinal, task in enumerate(test_tasks)]
    require(len(test_draw) == TASK_COUNT and len({key[2] for key in test_draw}) == TASK_COUNT,
            "official test draw must contain 128 distinct sessions")
    records = read_json(records_path)
    require(len(records) == TASK_COUNT, f"{records_path.name} must contain 128 native receipts")
    outcomes = []
    for ordinal, ((task_id, worker_seed, session), row) in enumerate(zip(test_draw, records, strict=True)):
        require_fields(row, {**eval_identity, "task_id": task_id,
                             "trajectory_id": f"evaluation:test:{ordinal}:slot-0"}, f"evaluation row {ordinal}")
        native = row["official_evaluation"]
        require_fields(native, {"task_id": task_id, "split": "test", "benchmark": "webshop",
                                "environment_seed": worker_seed, "session_index": session,
                                "native_evaluator": "WebAgentTextEnv.step"}, f"native receipt for {task_id}")
        purchased, score, success = (native[key] for key in ("purchased", "score", "success"))
        require(type(purchased) is bool and type(score) in (float, int) and 0 <= score <= 1
                and (purchased or score == 0) and success is (purchased and score == 1),
                f"native receipt for {task_id} has inconsistent purchased, score or success")
        outcomes.append({"purchased": purchased, "score": score, "success": success})
    success_count = sum(row["success"] for row in outcomes)
    metrics = {"SR": 100 * (success_count / TASK_COUNT), "Score": 100 * fmean(row["score"] for row in outcomes)}
    require_fields(evaluation, {**metrics, "success_count": success_count}, "test/evaluation_summary.json")
    return test_draw, outcomes, metrics


def read_cell(root, method, seed, rounds):
    cell = root / "comparison" / "webshop" / method / f"seed-{seed}"
    result = {"path": str(cell), "method": method, "rounds": rounds, "status": "pending"}
    marker_path = cell / ".finalization_complete.json"
    if not marker_path.exists():
        result["reason"] = "等待终评完成" if cell.exists() else "等待启动"
        return result, None

    try:
        config = read_json(cell / "config.json")
        study = STUDY_PROTOCOL if rounds == 150 else None
        identity = {"protocol": PROTOCOL, "method": method, "seed": seed, "benchmark": "webshop",
                    "study_protocol": study}
        require_fields(config, {**identity, "rounds": rounds, "groups": 16, "trajectories": 8}, "config.json")
        require(isinstance(config["adaptations"], list) and MENU_ADAPTATION in config["adaptations"],
                "config.json is missing the corrected WebShop menu")
        model_path = config["model_path"]
        require(isinstance(model_path, str) and model_path, "config.json is missing model_path")

        cell_summary = read_json(cell / "cell_summary.json")
        require_fields(cell_summary, {**identity, "suite": "comparison", "model_path": model_path,
                                      "status": "complete", "finalization_stage": "finalized"}, "cell_summary.json")
        snapshot = cell_summary["completed_checkpoint"]["snapshot"]
        require(isinstance(snapshot, str) and snapshot, "cell_summary.json has no final checkpoint snapshot")
        require_fields(read_json(marker_path), {"snapshot": snapshot, "protocol": PROTOCOL,
                                                "terminal_export_completed": True}, marker_path.name)
        training = read_json(cell / "training_summary.json")
        require_fields(training, {**identity, "status": "complete", "rounds_completed": rounds},
                       "training_summary.json")
        require(training == cell_summary["training"], "training_summary.json differs from the terminal cell")
        version = training["optimizer_steps"]
        require(type(version) is int and version >= 0, "training_summary.json has an invalid optimizer_steps")

        origin = None
        if rounds == 150 and method == "fgpo":
            origin = read_json(cell / "continuation.json")
            require_fields(origin, {"protocol": PROTOCOL, "study_protocol": STUDY_PROTOCOL, "seed": seed,
                                    "rounds": 60, "target_rounds": rounds, "target_cell": str(cell)}, "continuation.json")
            source = Path(origin["source_cell"])
            require(origin == cell_summary["continuation"] and source.is_absolute()
                    and source.parts[-4:] == cell.parts[-4:] and source != cell,
                    "continuation.json does not name this seed's original V4 cell")
            origin_version = origin["optimizer_steps"]
            require(type(origin_version) is int and 0 <= origin_version <= version,
                    "continuation.json has an invalid round-60 model version")

        eval_identity = {**identity, "split": "test", "model_version": version}
        evaluation = read_json(cell / "test" / "evaluation_summary.json")
        require_fields(evaluation, {**eval_identity, "model_path": model_path, "dataset_complete": True,
                                    "official_task_count": TASK_COUNT, "evaluated_count": TASK_COUNT,
                                    "missing_tasks": [], "unexpected_tasks": [], "score_unit": "percent"},
                       "test/evaluation_summary.json")
        require(evaluation == cell_summary["evaluation"]["test"], "evaluation summary differs from the terminal cell")

        catalog = read_json(cell / "official_tasks.json")
        require(len(catalog["train"]) == rounds * 16, "official training draw has the wrong number of tasks")
        train_draw = [task_key(task, "train", seed + ordinal % 16) for ordinal, task in enumerate(catalog["train"])]
        records_path = cell / "test" / "evaluation_records.json"
        test_draw, outcomes, metrics = read_native_evaluation(records_path, catalog["test"], evaluation)
        settings = {key: config[key] for key in (
            "model_path", "webshop_data_root", "max_context_tokens", "output_tokens", "temperature", "top_p",
            "eval_temperature", "eval_top_p",
        )}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result.update(status="invalid", reason=str(error))
        return result, None

    result.update(status="complete", study_protocol=study, native_protocol=PROTOCOL, continuation=origin,
                  split="test", model_version=version, snapshot=snapshot, settings=settings,
                  native_receipts=str(records_path), task_count=TASK_COUNT, metrics=metrics)
    return result, (train_draw, test_draw, outcomes)


def pair_seed(comparison, seed, reference, target):
    (reference_cell, reference_data), (target_cell, target_data) = reference, target
    row = {"seed": seed, "reference": reference_cell, "target": target_cell}
    statuses = (reference_cell["status"], target_cell["status"])
    if statuses != ("complete", "complete"):
        row["status"] = "invalid" if "invalid" in statuses else "pending"
        return row, []

    reference_train, reference_tasks, reference_outcomes = reference_data
    target_train, target_tasks, target_outcomes = target_data
    try:
        if reference_cell["rounds"] == 60:
            origin = target_cell["continuation"]
            require(origin["source_cell"] == reference_cell["path"]
                    and origin["optimizer_steps"] == reference_cell["model_version"],
                    "the 60-round result is from a different cell or model than the continued checkpoint")
        require(reference_cell["settings"] == target_cell["settings"], "model or WebShop sampling settings differ")
        require(reference_train == target_train[:len(reference_train)], "official training draws differ in content or order")
        require(reference_tasks == target_tasks, "official test draws differ in content or order")
    except ValueError as error:
        row.update(status="invalid", reason=str(error))
        return row, []

    counts = dict.fromkeys(OUTCOMES, 0)
    tasks = []
    for ordinal, ((task_id, worker_seed, session), before, after) in enumerate(
        zip(reference_tasks, reference_outcomes, target_outcomes, strict=True)
    ):
        outcome = {
            (False, True): "gained_full", (True, False): "lost_full",
            (True, True): "both_full", (False, False): "neither_full",
        }[before["success"], after["success"]]
        counts[outcome] += 1
        tasks.append({"comparison": comparison, "seed": seed, "ordinal": ordinal, "task_id": task_id,
                      "environment_seed": worker_seed, "session_index": session,
                      # Engine assigns its sampling seed by this native trajectory ordinal.
                      "sampling_seed": seed * 1000003 + ordinal,
                      "reference_model_version": reference_cell["model_version"],
                      "target_model_version": target_cell["model_version"],
                      **{f"reference_{key}": value for key, value in before.items()},
                      **{f"target_{key}": value for key, value in after.items()},
                      "score_delta": after["score"] - before["score"], "outcome": outcome})
    row.update(status="complete", outcomes=counts,
               metrics={"reference": reference_cell["metrics"], "target": target_cell["metrics"],
                        "delta": {key: target_cell["metrics"][key] - reference_cell["metrics"][key] for key in METRICS}})
    return row, tasks


def summarize_pairs(pairs):
    ready = [row for row in pairs if row["status"] == "complete"]
    aggregate = None
    if len(ready) == len(SEEDS):
        aggregate = {"metrics": {}, "outcomes": {key: sum(row["outcomes"][key] for row in ready) for key in OUTCOMES}}
        for side in ("reference", "target", "delta"):
            aggregate["metrics"][side] = {}
            for key in METRICS:
                values = [row["metrics"][side][key] for row in ready]
                aggregate["metrics"][side][key] = {"mean": fmean(values), "sample_sd": stdev(values)}
    if any(row["status"] == "invalid" for row in pairs):
        status = "invalid"
    elif aggregate:
        status = "complete"
    else:
        status = "partial" if ready else "awaiting_pairs"
    return {"status": status, "complete_seeds": [row["seed"] for row in ready], "pairs": pairs, "aggregate": aggregate}


def markdown(comparisons):
    lines = ["# WebShop 官方 128 题配对结果", "",
             "SR 与 Score 均用百分制，差值为目标组减参照组，单位为百分点。逐题文件保留原始 0–1 Score。"]
    for summary in comparisons.values():
        lines.extend(["", f"## {summary['title']}", ""])
        complete = len(summary["complete_seeds"])
        lines.append(f"已完成 {complete}/3 个种子配对。")
        for row in summary["pairs"]:
            if row["status"] != "complete":
                parts = [row["reason"]] if "reason" in row else []
                for side, label in (("reference", "参照组"), ("target", "目标组")):
                    cell = row[side]
                    if "reason" in cell:
                        parts.append(f"{label}{cell['reason']}")
                lines.append(f"\n种子 {row['seed']}　{'，'.join(parts)}。")
        if not complete:
            continue
        lines.extend(["", "| 种子 | 参照组 SR | 目标组 SR | ΔSR | 参照组 Score | 目标组 Score | ΔScore | 新增满分 | 失去满分 | 双方满分 | 双方未满分 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in summary["pairs"]:
            if row["status"] != "complete":
                continue
            metrics = [f"{row['metrics'][side][key]:{fmt}}" for key in METRICS
                       for side, fmt in (("reference", ".2f"), ("target", ".2f"), ("delta", "+.2f"))]
            values = [str(row["seed"]), *metrics, *(str(row["outcomes"][key]) for key in OUTCOMES)]
            lines.append("| " + " | ".join(values) + " |")
        aggregate = summary["aggregate"]
        if aggregate:
            metrics = []
            for key in METRICS:
                for side, fmt in (("reference", ".2f"), ("target", ".2f"), ("delta", "+.2f")):
                    stat = aggregate["metrics"][side][key]
                    metrics.append(f"{stat['mean']:{fmt}} ± {stat['sample_sd']:.2f}")
            values = ["均值 ± 样本标准差，题数合计", *metrics, *(str(aggregate["outcomes"][key]) for key in OUTCOMES)]
            lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def update_report(study_root, v4_60_root, output):
    pairs = {"candidate_vs_v4_150": []}
    if v4_60_root is not None:
        pairs["v4_150_vs_v4_60"] = []
    tasks = []
    for seed in SEEDS:
        v4 = read_cell(study_root, "fgpo", seed, 150)
        candidate = read_cell(study_root, "fgpo_exact_first_hit", seed, 150)
        runs = {"candidate_vs_v4_150": (v4, candidate)}
        if v4_60_root is not None:
            runs["v4_150_vs_v4_60"] = (read_cell(v4_60_root, "fgpo", seed, 60), v4)
        for comparison, (reference, target) in runs.items():
            pair, paired_tasks = pair_seed(comparison, seed, reference, target)
            pairs[comparison].append(pair)
            tasks.extend(paired_tasks)
    titles = {"candidate_vs_v4_150": "精确观测＋首次命中 150 轮 vs V4 150 轮",
              "v4_150_vs_v4_60": "V4 150 轮 vs V4 60 轮"}
    comparisons = {key: {"title": titles[key], **summarize_pairs(rows)} for key, rows in pairs.items()}
    report = {"study_protocol": STUDY_PROTOCOL, "study_root": str(study_root),
              "v4_60_root": str(v4_60_root) if v4_60_root is not None else None, "task_count_per_seed": TASK_COUNT,
              "seeds": SEEDS, "score_unit": "percent", "task_score_unit": "raw_0_to_1", "delta": "target_minus_reference",
              "comparisons": comparisons, "tasks": tasks}
    tasks_path = output / "tasks.tsv.incoming"
    with tasks_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, TASK_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(tasks)
    summary_path = output / "summary.md.incoming"
    summary_path.write_text(markdown(comparisons), encoding="utf-8")
    report_path = output / "report.json.incoming"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tasks_path.replace(output / "tasks.tsv")
    summary_path.replace(output / "summary.md")
    report_path.replace(output / "report.json")
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True, help="150-round paired study output root")
    parser.add_argument("--v4-60-root", type=Path, help="Optional 60-round V4 comparison output root")
    args = parser.parse_args(argv)
    study_root = args.study_root.resolve()
    v4_60_root = args.v4_60_root.resolve() if args.v4_60_root else None
    if v4_60_root == study_root:
        parser.error("--v4-60-root must point to the 60-round study")
    output = study_root / "webshop-paired-report"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".report.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        comparisons = update_report(study_root, v4_60_root, output)
    for key, summary in comparisons.items():
        print(f"{key} {summary['status']} ({len(summary['complete_seeds'])}/{len(SEEDS)} paired seeds)")
    print(output / "summary.md")
    return int(any(summary["status"] == "invalid" for summary in comparisons.values()))


if __name__ == "__main__":
    raise SystemExit(main())
