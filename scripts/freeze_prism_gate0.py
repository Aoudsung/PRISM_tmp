#!/usr/bin/env python3
"""Recover r12 source evidence and preregister Gate 0 without loading a model.

Run on mgt01 only. A missing window remains an explicit missing slot; this
command never calls an actor/compiler and never substitutes dev/test episodes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
from typing import Any

from prism_evolve.native_trace_artifacts import (
    canonical_bytes, export_trace, public_view, sha256, verify_blob_references,
    write_json_exclusive,
)


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "--git-dir=" + str(root / ".git-remote-r1.git"),
                                    *args], text=True).strip()


def has_public_error(observation: dict[str, Any]) -> bool:
    text = str(observation["native_observation"])
    return bool(re.search(r"not found|traceback|\berror\b|invalid|missing required|"
                          r"nothing happens|cannot|can't|does not exist|failed", text, re.I))


def window_options(view: dict[str, Any]) -> list[dict[str, Any]]:
    """Contract §3.3: locate evidence first, retain unknown native effects.

    Text ActionRefs are grounded in the current public action list. JSON tool
    Ref consumption needs a structured return leaf with the same value and a
    non-error consumer; schema-valid literals alone are never a local witness.
    """
    result = []
    events = view["events"]
    for index, event in enumerate(events):
        action = str(event["action"].get("text", event["response"]))
        after = event["obs_after"]
        repeated = (index > 0 and action == str(events[index - 1]["action"].get("text", events[index - 1]["response"]))
                    and after["native_observation"] == events[index - 1]["obs_after"]["native_observation"])
        if has_public_error(after) or repeated:
            result.append({"role": "counterexample", "start_event": 0, "end_event_exclusive": index + 1,
                           "evidence_event": index, "basis": "public_error" if has_public_error(after) else "unjustified_identical_repeat"})
            continue
        if not index:
            continue
        before = event["obs_before"]
        available = before["classic_text_action_state"].get("available_actions", [])
        if action in available and not repeated:
            # Re-reading the same item without a downstream consumer is not loop evidence.
            if action == events[index - 1]["response"]:
                continue
            result.append({"role": "positive", "start_event": 0, "end_event_exclusive": index + 1,
                           "evidence_event": index, "basis": "returned_ActionRef_consumed",
                           "source_event": index - 1, "source_class": "LOCAL_WITNESS"})
        else:
            try:
                call = json.loads(action)
            except (TypeError, ValueError):
                continue
            if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
                continue
            for key, value in call["arguments"].items():
                # ID bindings require the corresponding public field, not an unrelated equal number.
                field = "id" if key.endswith("_id") else key
                pattern = r'"' + re.escape(field) + r'"\s*:\s*' + re.escape(json.dumps(value)) + r'(?=[,}\s])'
                if re.search(pattern, str(before["native_observation"]).replace('\\"', '"').replace('\\n', '\n')):
                    result.append({"role": "positive", "start_event": 0, "end_event_exclusive": index + 1,
                                   "evidence_event": index, "basis": "returned_field_consumed",
                                   "argument_field": key, "source_event": index - 1,
                                   "source_class": "LOCAL_WITNESS"})
                    break
    return result


def select_cases(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases, coverage = [], {}
    for benchmark in config["benchmarks"]:
        benchmark_cases = []
        for role, limit in (("positive", 4), ("counterexample", 2)):
            grouped = defaultdict(list)
            for record in records:
                if record["benchmark"] != benchmark or record["split"] != "train" or record["integrity_errors"]:
                    continue
                for option in record["window_options"]:
                    if option["role"] == role:
                        grouped[record["task_id"]].append({**option, "episode_id": record["episode_id"],
                            "task_id": record["task_id"], "source_path": record["copied_path"],
                            "source_sha256": record["source_sha256"], "native_success": record["native_success"],
                            "actor_seed": record["actor_seed"], "source_policy": record["source_policy"]})
            selected = []
            # Stable task round-robin: diverse tasks precede more windows from the same task.
            tasks = sorted(grouped, key=lambda task: (
                all(item["source_policy"] != "frozen_base" for item in grouped[task]), task))
            for task in tasks:
                grouped[task].sort(key=lambda item: (item["source_policy"] != "frozen_base", item["episode_id"], item["evidence_event"]))
            while tasks and len(selected) < limit:
                for task in tasks[:]:
                    if grouped[task]:
                        selected.append(grouped[task].pop(0))
                        if len(selected) == limit:
                            break
                    else:
                        tasks.remove(task)
            for index in range(limit):
                case = {"case_id": f"{benchmark}-{role}-{index + 1}", "benchmark": benchmark,
                        "role": role, "status": "MISSING_INPUT"}
                if index < len(selected):
                    case.update(selected[index], status="SOURCE_FROZEN")
                benchmark_cases.append(case)
        positives = [case for case in benchmark_cases if case["role"] == "positive" and case["status"] == "SOURCE_FROZEN"]
        negatives = [case for case in benchmark_cases if case["role"] == "counterexample" and case["status"] == "SOURCE_FROZEN"]
        coverage[benchmark] = {"positive_windows": len(positives), "counterexample_windows": len(negatives),
            "failed_positive_windows": sum(not case["native_success"] for case in positives),
            "positive_tasks": len({case["task_id"] for case in positives}),
            "input_counts_satisfied": (len(positives) == 4 and len(negatives) == 2
                and sum(not case["native_success"] for case in positives) >= 2
                and len({case["task_id"] for case in positives}) >= 2),
            "expression_coverage": "NOT_YET_COMPILED"}
        for case in benchmark_cases:
            case["preregistered_order"] = len(cases)
            cases.append(case)
    return cases, coverage


def collection_plan(config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze exact fallback IDs before any collection; at most 10 per benchmark."""
    host = json.loads(Path(config["host_config"]).read_text())
    excluded = {row["task_id"] for row in records if row["split"] == "train"}
    alf_root = Path(host["alfworld_data_root"])
    if alf_root.name != "json_2.1.1":
        alf_root = alf_root / "json_2.1.1"
    alf_tasks = sorted(p.relative_to(alf_root).as_posix() for p in (alf_root / "train").rglob("game.tw-pddl"))
    loca_source = Path(host["loca_pos_repo_root"]) / "benchmarks/LOCA-Bench/task-configs/final_8k_set_config.json"
    templates = sorted({str(case["name"]) for case in json.loads(loca_source.read_text())["configurations"]})
    loca_tasks = [f"loca/8k/{template}/seed_{seed}" for template in templates for seed in host["loca_train_seeds"]]
    task_lists = {"alfworld": alf_tasks,
                  "webshop": [f"webshop/{i}" for i in range(1500, 6910)], "loca": sorted(loca_tasks)}
    plan = []
    for b_index, benchmark in enumerate(config["benchmarks"]):
        tasks = sorted(task for task in task_lists[benchmark] if task not in excluded)[:10]
        if len(tasks) != 10:
            raise ValueError(f"Cannot preregister 10 train tasks for {benchmark}")
        for index, task in enumerate(tasks):
            plan.append({"benchmark": benchmark, "task_id": task, "split": "train",
                         "seed": config["collection"]["actor_seed_base"] + b_index * 100 + index,
                         "order": len(plan), "policy": "frozen_base", "attempt_limit": 1})
    return {"episodes": plan, "maximum_per_benchmark": 10, "model_calls_performed": 0,
            "run_only_if_input_coverage_missing": True, "requires_complete_implementation": True}


def recover(config_path: Path, contract_dir: Path) -> Path:
    config = json.loads(config_path.read_text())
    root = Path(config["project_root"]).resolve()
    if socket.gethostname().split(".")[0] != config["execution_host"] or Path.cwd().resolve() != root:
        raise RuntimeError("Gate 0 recovery must run in the registered mgt01 project")
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    collection_plan(config, [])  # validate dataset roots before copying any trace
    source = root / config["source_root"]
    # Load the frozen tokenizer.json directly; recovery needs no model runtime.
    from tokenizers import Tokenizer
    backend = Tokenizer.from_file(str(Path(config["model_path"]) / "tokenizer.json"))

    class RecordedPromptTokenizer:
        def decode(self, ids, **kwargs):
            return backend.decode(ids, skip_special_tokens=False)

        def encode(self, text, **kwargs):
            return backend.encode(text, add_special_tokens=False).ids

    tokenizer = RecordedPromptTokenizer()
    records, reference_checks, perturbation_checks = [], 0, 0
    for benchmark in config["benchmarks"]:
        for policy_dir in ("training_prism_evolve", "eval_frozen_base", "eval_prism_evolve"):
            for path in sorted((source / benchmark / policy_dir / "traces").glob("*.json")):
                relative = path.relative_to(source)
                episode = benchmark + "-" + policy_dir + "-" + path.stem
                trace = json.loads(path.read_text(encoding="utf-8"))
                destination = output / "raw_traces" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                errors = []
                try:
                    view = public_view(trace)
                    altered = deepcopy(trace)
                    altered["metadata"] = {"state_changed": True, "facts": "PRIVATE_SENTINEL", "native_score": 123}
                    for event in altered["events"]:
                        event["native_step"] = {"state_changed": False, "world_state": "PRIVATE_SENTINEL"}
                        event["receipts"] = [{"hidden_evaluator": "PRIVATE_SENTINEL"}]
                        event["sampling"]["dcc_decision"] = "PRIVATE_SENTINEL"
                    if canonical_bytes(view) != canonical_bytes(public_view(altered)):
                        raise ValueError("Private diagnostics changed compiler input")
                    perturbation_checks += 1
                    exported = export_trace(output, trace, episode, tokenizer=tokenizer)
                    reference_checks += verify_blob_references(output, exported)
                    if any(event["actor_input_ref"] is None for event in exported["events"]):
                        errors.append("MISSING_ACTOR_INPUT")
                    options = window_options(view)
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(str(exc))
                    options = []
                official = trace.get("metadata", {}).get("official_evaluation", {})
                records.append({"episode_id": episode, "benchmark": benchmark, "task_id": trace["task_id"],
                    "split": official.get("split", "UNKNOWN"), "native_split": official.get("native_split"),
                    "native_success": bool(trace["success"]), "source_policy": "prism_evolve" if policy_dir != "eval_frozen_base" else "frozen_base",
                    "actor_seed": trace.get("metadata", {}).get("environment_seed"), "events": len(trace["events"]),
                    "source_path": str(path), "copied_path": destination.relative_to(output).as_posix(),
                    "source_sha256": file_identity(path)["sha256"], "integrity_errors": errors,
                    "window_options": options, "physical_call_identity": "UNAVAILABLE_LEGACY"})
        # Existing physical records are archived as legacy evidence, never promoted to the new ledger.
        for path in sorted((source / benchmark).rglob("*.jsonl")):
            if "cost" in str(path.relative_to(source)):
                destination = output / "legacy_costs" / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
    cases, coverage = select_cases(records, config)
    model_root = Path(config["model_path"])
    model_files = [file_identity(path) for path in sorted(model_root.iterdir())
                   if path.name.endswith(".safetensors") or path.name in (
                       "config.json", "generation_config.json", "model.safetensors.index.json",
                       "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json")]
    code = []
    for relative in ("src/prism_evolve/native_core.py", "src/prism_evolve/native_policy.py",
                     "src/prism_evolve/native_training.py", "src/prism_evolve/native_runner.py",
                     "src/prism_evolve/native_trace_artifacts.py", "scripts/freeze_prism_gate0.py"):
        current = file_identity(root / relative)
        baseline = subprocess.run(["git", "--git-dir=" + str(root / ".git-remote-r1.git"),
            "show", config["audit_baseline"] + ":" + relative], capture_output=True, check=False)
        code.append({**current, "audit_baseline_sha256": sha256(baseline.stdout) if not baseline.returncode else None,
                     "matches_audit_baseline": baseline.returncode == 0 and current["sha256"] == sha256(baseline.stdout)})
    contracts = []
    for path in sorted(contract_dir.iterdir()):
        if path.is_file():
            dest = output / "contract" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            contracts.append(file_identity(dest))
    shutil.copyfile(config_path, output / "frozen_config.json")
    write_json_exclusive(output / "recovery_inventory.json", {"episodes": records,
        "counts": dict(Counter(f"{row['benchmark']}/{row['split']}/{row['source_policy']}" for row in records)),
        "verified_blob_references": reference_checks, "private_perturbation_checks": perturbation_checks,
        "model_calls_performed": 0, "legacy_accounting_is_new_ledger": False})
    write_json_exclusive(output / "collection_preregistration.json", collection_plan(config, records))
    write_json_exclusive(output / "case_manifest.json", {"cases": cases, "coverage": coverage,
        "status": "RECOVERED_INPUT_COUNTS_READY" if all(v["input_counts_satisfied"] for v in coverage.values()) else "NEEDS_PREREGISTERED_COLLECTION",
        "selection_frozen": True, "compilation_allowed": False,
        "reason": "Final 18-window freeze and complete implementation required before compilation"})
    manifest = {"run_id": config["run_id"], "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "project_root": str(root), "audit_baseline": config["audit_baseline"],
        "bare_refs": git_output(root, "show-ref"), "implementation_files": code,
        "config_sha256": sha256(config_path.read_bytes()), "model_files": model_files,
        "model_identity_sha256": sha256(canonical_bytes(model_files)), "reference": config["reference"],
        "reference_code_contract": "No reference adapter installed: frozen_reference() -> raw_base_reference() -> disable_adapter()",
        "contract_files": contracts, "model_calls_performed": 0,
        "actual_prompt_recovery": "Recorded prompt IDs decoded losslessly; messages_before saved separately as pre-policy history",
        "legacy_limits": ["Legacy calls have no immutable physical IDs or exception-closure evidence",
                          "Legacy reset envelope reconstructed from decoded public reset message",
                          "Legacy program-bound origin may be absent; never inferred as a physical model call"],
        "gate_status": "NOT_RUN"}
    write_json_exclusive(output / "manifest.json", manifest)
    # Seal the exact files so later phases can reject changed recovery inputs.
    sealed = {path.relative_to(output).as_posix(): file_identity(path)["sha256"]
              for path in sorted(output.rglob("*")) if path.is_file()}
    write_json_exclusive(output / "recovery_seal.json", sealed)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    args = parser.parse_args()
    print(recover(args.config, args.contract_dir))


if __name__ == "__main__":
    main()
