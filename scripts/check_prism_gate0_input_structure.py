#!/usr/bin/env python3
"""Post-collection Gate 0 structural input check, without reselection or models.

The supervisor-approved boundary revision leaves the collecting process and its
frozen implementation unchanged. --guard writes explicit control artifacts at
exclusive compiler-input paths, so the old count-only boundary cannot issue a
model request. These artifacts are never represented as actual model prompts.
--settle preserves the count-only decision and settles structural input failure.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import socket
import zipfile

from freeze_prism_gate0 import file_identity, has_public_error
from prism_evolve.native_trace_artifacts import canonical_bytes, public_view, write_json_exclusive
from prism_evolve.native_trace_compile import executed_action
from run_prism_gate0 import audit_physical, cost_event, read_json, report
from prism_evolve.native_core import monotonic_seconds


def words(text):
    return re.findall(r"[\w.-]+", text)


def evidence_for_window(view, limit):
    events = view["events"][:limit]
    frontier, bindings, pairs, single_calls = [], [], [], []
    for index, event in enumerate(events):
        action = executed_action(event)
        before = event["obs_before"]
        available = before["classic_text_action_state"].get("available_actions", [])
        if len(set(available)) >= 2:
            frontier.append({"event": index, "finite_public_action_refs": len(set(available)), "complete_collection": False})
        if index and action in available and not has_public_error(event["obs_after"]):
            bindings.append({"source_event": index - 1, "consumer_event": index, "kind": "ActionRef", "value": action})
            previous = executed_action(events[index - 1])
            left, right = words(previous), words(action)
            common = set(" ".join(left[start:start + size]) for size in range(1, min(5, len(left)) + 1)
                         for start in range(len(left) - size + 1)) & set(
                " ".join(right[start:start + size]) for size in range(1, min(5, len(right)) + 1)
                for start in range(len(right) - size + 1))
            if common:
                value = max(common, key=lambda text: (len(words(text)), len(text), text))
                if value in previous and value in action and value in str(before["native_observation"]):
                    pairs.append({"events": [index - 1, index], "binding": value,
                        "template": [previous.replace(value, "${ref}"), action.replace(value, "${ref}")]})
        try:
            call = json.loads(action)
        except ValueError:
            continue
        if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict) or has_public_error(event["obs_after"]):
            continue
        for key, value in call["arguments"].items():
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                continue
            field = "id" if key.endswith("_id") else key
            pattern = r'"' + re.escape(field) + r'"\s*:\s*' + re.escape(json.dumps(value)) + r'(?=[,}\s])'
            for source_index in range(index):
                source = str(events[source_index]["obs_after"]["native_observation"]).replace('\\"', '"').replace('\\n', '\n')
                if re.search(pattern, source):
                    bindings.append({"source_event": source_index, "consumer_event": index, "kind": "returned_field",
                                     "field": key, "value": value})
                    template = {**call, "arguments": {**call["arguments"], key: "${ref}"}}
                    single_calls.append({"events": [index], "binding": value,
                                         "template": canonical_bytes(template).decode("utf-8")})
                    break
    loops = []
    for candidates in (pairs, single_calls):
        by_start = defaultdict(list)
        for candidate in candidates:
            by_start[candidate["events"][0]].append(candidate)
        for first in candidates:
            for second in by_start[first["events"][-1] + 1]:
                if first["template"] == second["template"] and first["binding"] != second["binding"]:
                    loops.append({"iterations": [first["events"], second["events"]],
                        "bindings": [first["binding"], second["binding"]], "template": first["template"],
                        "basis": "two distinct bindings; same contiguous native structure; public result consumed"})
    return {"events_in_fixed_window": len(events), "frontier_witnesses": frontier,
        "cross_event_bindings": bindings, "loop_witnesses": loops,
        "if3_public_predicate_basis": bool(frontier or bindings),
        "if3_note": "Only public predicate material is certified here; F/U grammar hypotheses and actual If3 AST require compilation and independent review"}


def guard(output):
    initial = read_json(output / "case_manifest.json")
    for case in initial["cases"]:
        write_json_exclusive(output / "compiler_inputs" / (case["case_id"] + ".compiler.annotate_trace.json"), {
            "artifact_type": "CONTROL_BOUNDARY_GUARD_NOT_A_MODEL_PROMPT", "case_id": case["case_id"],
            "reason": "Supervisor-approved deterministic structural input check required before any compiler model call",
            "model_call_id": None, "physical_model_tokens": 0})
    write_json_exclusive(output / "structural_boundary_guard.json", {
        "installed_at": datetime.now(timezone.utc).isoformat(), "guarded_case_ids": [case["case_id"] for case in initial["cases"]],
        "collection_implementation_unchanged": True, "checker": file_identity(Path(__file__)),
        "supervisor_approval": "finish same 30 registered episodes; deterministic structural check before first compiler call; no reselection"})


def settle(output):
    started = monotonic_seconds()
    attempts = read_json(output / "execution_manifest.json")["collection_episodes"]
    collected = [json.loads(line) for line in (output / "collected_episodes.jsonl").read_text().splitlines()]
    if len(collected) != len(attempts) or {row["task_id"] for row in collected} != {row["task_id"] for row in attempts}:
        raise ValueError("Collection is not closed for the same complete preregistered task list")
    audit = audit_physical(output)
    if not audit["closed"]:
        raise ValueError("Cannot settle while a physical generation is still open")
    compiler_calls = [row for row in audit["model_calls"] if row["role"].startswith("compiler.")]
    if compiler_calls:
        raise ValueError("Structural check must precede every compiler model call")
    original = read_json(output / "final_case_manifest.json")
    cases = original["cases"]
    evidence = {}
    for case in cases:
        if case["status"] != "SOURCE_FROZEN":
            evidence[case["case_id"]] = {"input_missing": True}
            continue
        source = output / case["source_path"]
        if file_identity(source)["sha256"] != case["source_sha256"]:
            raise ValueError("Frozen source changed")
        evidence[case["case_id"]] = evidence_for_window(public_view(read_json(source)), case["end_event_exclusive"])
    per_benchmark = {}
    for benchmark in ("alfworld", "webshop", "loca"):
        positives = [case for case in cases if case["benchmark"] == benchmark and case["role"] == "positive"]
        rows = [evidence[case["case_id"]] for case in positives]
        per_benchmark[benchmark] = {
            "loop_source_cases": [case["case_id"] for case in positives if evidence[case["case_id"]].get("loop_witnesses")],
            "frontier_source_cases": [case["case_id"] for case in positives if evidence[case["case_id"]].get("frontier_witnesses")],
            "if3_public_basis_cases": [case["case_id"] for case in positives if evidence[case["case_id"]].get("if3_public_predicate_basis")],
            "return_ref_consumption_cases": [case["case_id"] for case in positives if any(
                binding["kind"] == "returned_field" for binding in evidence[case["case_id"]].get("cross_event_bindings", []))],
            "cross_event_binding_cases": [case["case_id"] for case in positives if evidence[case["case_id"]].get("cross_event_bindings")]}
    failures = []
    for benchmark in ("alfworld", "webshop"):
        if not per_benchmark[benchmark]["loop_source_cases"]:
            failures.append(benchmark + ": no two-distinct-binding closed-loop witness in the fixed positive windows")
        if not per_benchmark[benchmark]["frontier_source_cases"]:
            failures.append(benchmark + ": no finite public Frontier witness")
    if not per_benchmark["loca"]["return_ref_consumption_cases"]:
        failures.append("loca: no real returned Ref consumed in the fixed positive windows")
    if not any(row["if3_public_basis_cases"] for row in per_benchmark.values()):
        failures.append("no source-proven If3 predicate material")
    if not any(row["cross_event_binding_cases"] for row in per_benchmark.values()):
        failures.append("no source-proven cross-event binding")
    if not all(row["input_counts_satisfied"] for row in original["coverage"].values()):
        failures.append("fixed sample counts, failed-source strata or task diversity are incomplete")
    check = {"checker": file_identity(Path(__file__)), "collection_implementation_manifest_sha256": file_identity(output / "execution_manifest.json")["sha256"],
        "fixed_case_order": [case["case_id"] for case in cases], "case_evidence": evidence, "benchmarks": per_benchmark,
        "failures": failures, "pass": not failures, "compiler_model_calls": 0,
        "reselected_windows": False, "repeated_episodes": False}
    write_json_exclusive(output / "input_structure_check.json", check)
    preserved = output / "boundary_evidence"
    preserved.mkdir(exist_ok=True)
    (output / "final_case_manifest.json").rename(preserved / "count_only_case_manifest.json")
    updated = {**original, "status": "FAIL_INPUT_COVERAGE" if failures else "FROZEN", "compilation_allowed": not failures,
        "structure_check": "input_structure_check.json", "structure_check_implementation": check["checker"],
        "collection_implementation": "execution_manifest.json", "deterministic_case_selection_unchanged": True}
    write_json_exclusive(output / "final_case_manifest.json", updated)
    for path in (output / "compiler_inputs").glob("*.compiler.annotate_trace.json"):
        if read_json(path).get("artifact_type") == "CONTROL_BOUNDARY_GUARD_NOT_A_MODEL_PROMPT":
            path.rename(preserved / path.name)
    for path in (output / "compiler_outputs").glob("*.result.json"):
        path.rename(preserved / path.name)
    for name in ("gate0_report.json", "gate0_compile_report.json", "model_calls.jsonl", "cost_events.jsonl"):
        path = output / name
        if path.exists():
            path.rename(preserved / name)
    decoding = read_json(output / "decoding_manifest.json")
    actual = defaultdict(set)
    for row in collected:
        trace = read_json(output / row["copied_path"])
        for event in trace["events"]:
            actual[row["benchmark"]].add(event["sampling"].get("requested_max_new_tokens"))
    write_json_exclusive(output / "decoding_report.json", {**decoding,
        "actual_recorded_requested_limits": {key: sorted(value) for key, value in actual.items()},
        "all_actual_requests_at_most_512": all(all(value is not None and value <= 512 for value in values) for values in actual.values())})
    cost_event(output, "gate0-final-input-check", "input_structure_check_outer_interval", started,
               nested_costs_not_additive=True, implementation=check["checker"])
    audit = audit_physical(output)
    if not failures:
        print(json.dumps({"status": "FROZEN_REQUIRES_ACCOUNTED_COMPILE_RESUME", "evidence": str(output / "input_structure_check.json")}))
        return
    config = read_json(output / "execution_manifest.json")["config"]
    result = report(config, output, final=False)
    (output / "gate0_compile_report.json").rename(output / "gate0_report.json")
    for name in ("types_and_sources", "liveness_and_effects", "compiler_yield", "failure_source_start",
                 "counterexamples", "execution_cost", "startup_cost"):
        result["criteria"][name]["status"] = "NOT_RUN_INPUT_COVERAGE"
    result["criteria"]["expression_coverage"] = {"pass": False, "status": "FAIL_INPUT_COVERAGE",
                                                  "evidence": "input_structure_check.json", "failures": failures}
    check_wall = sum(row["elapsed_seconds"] for row in audit["cost_events"]
                     if row["kind"] == "input_structure_check_outer_interval")
    result = {**result, "input_structure_failures": failures, "effective_decoding": read_json(output / "decoding_report.json"),
              "structure_check_implementation": check["checker"], "compiler_model_calls": 0,
              "total_charged_outer_wall_seconds_including_input_check": audit["total_charged_outer_wall_seconds"] + check_wall,
              "no_gate1_or_gate2_authorization": True}
    # Preserve the unaugmented settlement; the new report explicitly identifies this boundary revision.
    (output / "gate0_report.json").rename(preserved / "report_before_structure_metadata.json")
    write_json_exclusive(output / "gate0_report.json", result)
    for name, rows in (("model_calls.jsonl", audit["model_calls"]), ("cost_events.jsonl", audit["cost_events"])):
        with (output / name).open("x", encoding="utf-8") as handle:
            for row in rows:handle.write(canonical_bytes(row).decode("utf-8") + "\n")
    archive = output.parent / (output.name + "-checked.zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(output.rglob("*")):
            if not path.is_file():continue
            name = path.relative_to(output).as_posix()
            if name == "manifest.json":name = "recovery_manifest.json"
            elif name == "case_manifest.json":name = "recovery_case_manifest.json"
            elif name == "final_case_manifest.json":name = "case_manifest.json"
            package.write(path, name)
        package.writestr("manifest.json", canonical_bytes({"recovery": read_json(output / "manifest.json"),
            "execution": read_json(output / "execution_manifest.json"), "effective_decoding": decoding,
            "input_structure_check": check, "gate0_report": "gate0_report.json", "posterior_learning": False,
            "online_graph_scheduling": False}).decode("utf-8"))
    write_json_exclusive(output.parent / (archive.name + ".identity.json"), file_identity(archive))
    print(json.dumps({"status": "FAIL_INPUT_COVERAGE", "failures": failures, "archive": str(archive)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--guard", action="store_true")
    mode.add_argument("--settle", action="store_true")
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "mgt01" or not args.output.resolve().is_relative_to(Path.cwd().resolve()):
        raise RuntimeError("Boundary check is restricted to the mgt01 project run")
    guard(args.output) if args.guard else settle(args.output)


if __name__ == "__main__":
    main()
