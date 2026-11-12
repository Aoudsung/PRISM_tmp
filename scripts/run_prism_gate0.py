#!/usr/bin/env python3
"""mgt01-only Gate 0: paid collection, final freeze, compilation and settlement.

Each registered base episode and compiler case is attempted once. Coverage
failure ends the round. Compiled candidates await independent public review;
--probe consumes an explicit version-matched review file, never model self-review.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import traceback
import zipfile
from typing import Any

from freeze_prism_gate0 import file_identity, select_cases, window_options
from prism_evolve.native_core import BudgetLimits, BudgetMeter, ExplorationLedger, Resources, monotonic_seconds
from prism_evolve.native_program_ir import IR_VERSION, SEMANTIC_SCHEMA_VERSION, Source, candidate_from_dict, validate, walk
from prism_evolve.native_program_execution import ProgramYield, execute
from prism_evolve.native_trace_artifacts import canonical_bytes, export_trace, public_view, sha256, verify_blob_references, write_json_exclusive
from prism_evolve.native_trace_compile import build_material, compile_window, executed_action, on_episode_trace_closed

IMPLEMENTATION_FILES = [
    "src/prism_evolve/native_core.py", "src/prism_evolve/native_policy.py",
    "src/prism_evolve/native_training.py", "src/prism_evolve/native_runner.py",
    "src/prism_evolve/native_trace_artifacts.py", "src/prism_evolve/native_program_ir.py",
    "src/prism_evolve/native_program_execution.py", "src/prism_evolve/native_trace_compile.py",
    "scripts/freeze_prism_gate0.py", "scripts/run_prism_gate0.py", "docs/prism-program-ir.md",
]
HARD_CRITERIA = ["raw_integrity", "public_boundary", "physical_accounting", "deadline_and_leases",
    "types_and_sources", "liveness_and_effects", "compiler_yield", "failure_source_start",
    "expression_coverage", "counterexamples", "execution_cost", "startup_cost"]


def append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_bytes(value).decode("utf-8") + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_host(config: dict[str, Any]) -> Path:
    root = Path(config["project_root"]).resolve()
    if socket.gethostname().split(".")[0] != "mgt01" or Path.cwd().resolve() != root:
        raise RuntimeError("Execution is restricted to the registered mgt01 project")
    return root / config["output_root"]


def freeze_implementation(config: dict[str, Any], output: Path) -> dict[str, Any]:
    identities = {path: file_identity(Path(path))["sha256"] for path in IMPLEMENTATION_FILES}
    previous = read_json(output / "manifest.json")
    plan = read_json(output / "collection_preregistration.json")
    # The approved startup extension keeps exactly the already registered IDs/seeds.
    if any(sum(row["benchmark"] == benchmark for row in plan["episodes"]) > 10 for benchmark in config["benchmarks"]):
        raise ValueError("Preregistered base collection exceeds the ten-episode ceiling")
    if len({(row["benchmark"], row["task_id"], row["seed"]) for row in plan["episodes"]}) != len(plan["episodes"]):
        raise ValueError("Duplicate collection episode")
    for identity in previous["model_files"]:
        if file_identity(Path(identity["path"]))["sha256"] != identity["sha256"]:
            raise ValueError("Frozen model/tokenizer asset changed")
    manifest = {"run_id": config["run_id"], "frozen_at": datetime.now(timezone.utc).isoformat(),
        "implementation_files": identities, "audit_baseline": config["audit_baseline"],
        "model_identity_sha256": previous["model_identity_sha256"], "reference": config["reference"],
        "config": config, "collection_episodes": plan["episodes"],
        "collection_authorization": "all three benchmarks: registered training episodes for input coverage and paid startup credit; maximum ten each",
        "ir_version": IR_VERSION, "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "ir_conventions": {"Frontier": "finite currently disclosed source-proven RefSet; never hidden completeness",
            "Loop": "one binding per finite item; per-node/per-iteration budget; invalidation or unknown yields",
            "If3": "T/F/U only; missing public evidence is U"},
        "startup_leases": {"compile": {"wall_seconds": 60, "model_tokens": 18432, "env_steps": 0},
            "trial": {"wall_seconds": 60, "model_tokens": 18432, "env_steps": 3}},
        "probe_protocol": {"prefix_event": 0, "actions_per_arm": 1,
            "selection": "first two positive slots with distinct source episodes per benchmark",
            "order": config["cost_probe_order"], "base_and_fallback_greedy": False,
            "compiler_cost_amortization": "none", "independent_public_review_required": True},
        "environment": {"host": socket.gethostname(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "packages": {name: version(name) for name in ("torch", "transformers", "peft", "tokenizers", "requests")},
            "offline_http_dependencies": read_json(output.parent / "operation_logs" / "offline-dependency-paths.json"),
            "gpu_inventory": subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"], text=True)},
        "complete_implementation_before_first_model_call": True,
        "posterior_learning": False, "online_graph_scheduling": False}
    write_json_exclusive(output / "execution_manifest.json", manifest)
    return manifest


def verify_implementation(output: Path) -> dict[str, Any]:
    manifest = read_json(output / "execution_manifest.json")
    for path, digest in manifest["implementation_files"].items():
        if file_identity(Path(path))["sha256"] != digest:
            raise ValueError("Implementation changed after preregistration: " + path)
    return manifest


def cost_event(output: Path, round_id: str, kind: str, started: float, **extra: Any) -> None:
    append_json(output / "physical_events.jsonl", {"kind": kind, "charged_round_id": round_id,
        "cost_event_id": sha256(canonical_bytes([round_id, kind, started])),
        "elapsed_seconds": monotonic_seconds() - started, "status": "OK", **extra})


def load_policy(config: dict[str, Any], output: Path):
    import torch
    from prism_evolve.native_policy import NativeCausalPolicy
    started = monotonic_seconds()
    try:
        policy = NativeCausalPolicy(config["model_path"], device="cuda:0", max_context_tokens=32768,
            max_new_tokens=512, chat_template_kwargs={"enable_thinking": False}, gradient_checkpointing=False,
            activation_offload_cpu=False)
        if policy._reference_adapter is not None:
            raise ValueError("Gate 0 requires the frozen raw-base reference")
        policy.model_metadata["frozen_asset_identity"] = read_json(output / "manifest.json")["model_identity_sha256"]
        policy.model_metadata["gate0_reference"] = "raw_base_disable_adapter"
        policy.eval()
        return policy
    finally:
        cost_event(output, "gate0-setup", "model_setup", started)


def collect(config: dict[str, Any], output: Path, policy: Any) -> list[dict[str, Any]]:
    import torch
    from prism_evolve.native_runner import _environment, _save_trace
    from prism_evolve.native_training import rollout, public_trace
    records = []
    plan = read_json(output / "execution_manifest.json")["collection_episodes"]
    host = read_json(Path(config["host_config"]))
    for benchmark in config["benchmarks"]:
        episode_cfg = {**host, "benchmark": benchmark, "max_context_tokens": 32768,
            "episode_max_steps": config["collection"]["episode_max_steps"][benchmark],
            "episode_max_model_tokens": config["collection"]["episode_max_model_tokens"],
            "episode_max_wall_seconds": config["collection"]["episode_max_wall_seconds"],
            "max_new_tokens": host["benchmark_max_new_tokens"][benchmark], "evaluation_greedy": False}
        setup_started = monotonic_seconds()
        env = _environment(episode_cfg, benchmark, output / "collection_envs" / benchmark)
        cost_event(output, f"{benchmark}-environment-setup", "environment_setup", setup_started)
        try:
            allowed = {str(task["task_id"]): task for task in env.list_tasks("train")}
            for row in (item for item in plan if item["benchmark"] == benchmark):
                episode_id = f"{benchmark}-base-{row['order']:03d}"
                round_id = "collect-" + episode_id
                write_json_exclusive(output / "collection_attempts" / (episode_id + ".json"), row)
                if row["task_id"] not in allowed:
                    raise ValueError("Preregistered task is not in the native training split")
                task = {**allowed[row["task_id"]], "environment_seed": row["seed"]}
                torch.manual_seed(row["seed"])
                torch.cuda.manual_seed_all(row["seed"])
                meter = BudgetMeter(BudgetLimits(episode_cfg["episode_max_steps"],
                    episode_cfg["episode_max_model_tokens"], episode_cfg["episode_max_wall_seconds"]),
                    event_path=output / "physical_events.jsonl", charged_round_id=round_id)
                outer = monotonic_seconds()
                capture = {}
                try:
                    with policy.frozen_reference():
                        trajectory = rollout(policy, env, task, episode_cfg, meter, seed=row["seed"],
                                             method="frozen_base", training=False, trajectory_capture=capture)
                    trace = public_trace(trajectory, method="frozen_base")
                    path = output / "raw_traces" / "new_frozen_base" / benchmark / (episode_id + ".json")
                    write_json_exclusive(path, trace)
                    exported = export_trace(output, trace, episode_id, tokenizer=policy.tokenizer)
                    refs = verify_blob_references(output, exported)
                    view = public_view(trace)
                    private = deepcopy(trace)
                    private["metadata"] = {"private_facts": "PERTURBED", "state_changed": True}
                    for event in private["events"]:
                        event["native_step"] = {"state_changed": True, "evaluator": "PERTURBED"}
                    if canonical_bytes(view) != canonical_bytes(public_view(private)):
                        raise ValueError("Public compiler input depends on private diagnostics")
                    closed = on_episode_trace_closed(view, window_options, budget_available=True)
                    record = {"episode_id": episode_id, "benchmark": benchmark, "task_id": row["task_id"],
                        "split": "train", "native_success": bool(trace["success"]), "source_policy": "frozen_base",
                        "actor_seed": row["seed"], "events": len(trace["events"]), "copied_path": path.relative_to(output).as_posix(),
                        "source_sha256": file_identity(path)["sha256"], "integrity_errors": [],
                        "window_options": closed["windows"], "compile_trigger": closed["status"],
                        "verified_blob_references": refs, "private_perturbation_passed": True,
                        "meter": meter.snapshot(), "paid_resources": asdict(Resources(
                            monotonic_seconds() - outer, meter.total_model_tokens, meter.env_steps))}
                    append_json(output / "collected_episodes.jsonl", record)
                    records.append(record)
                    print(json.dumps({"episode": episode_id, "events": len(trace["events"]),
                        "success": bool(trace["success"]), "tokens": meter.total_model_tokens,
                        "termination": trace["termination"]}), flush=True)
                    if not meter.usage_complete or meter.exploration_stopped:
                        raise RuntimeError("Collection accounting/deadline failure; later exploration is stopped")
                except BaseException as exc:
                    partial = capture.get("trajectory")
                    if partial is not None:
                        partial.termination = "interrupted:" + type(exc).__name__
                        write_json_exclusive(output / "raw_traces" / "interrupted" / (episode_id + ".json"),
                                             public_trace(partial, method="frozen_base"))
                    raise
                finally:
                    cost_event(output, round_id, "episode_outer_interval", outer, nested_costs_not_additive=True)
        finally:
            env.close()
    return records


def final_freeze(config: dict[str, Any], output: Path, collected: list[dict[str, Any]]) -> dict[str, Any]:
    recovered = read_json(output / "recovery_inventory.json")["episodes"]
    cases, coverage = select_cases(collected + recovered, config)
    probes, startup = {}, {}
    for benchmark in config["benchmarks"]:
        positives = [case for case in cases if case["benchmark"] == benchmark and case["role"] == "positive" and case["status"] == "SOURCE_FROZEN"]
        selected, episodes = [], set()
        for case in positives:
            if case["episode_id"] not in episodes:
                selected.append(case["case_id"])
                episodes.add(case["episode_id"])
            if len(selected) == 2:
                break
        probes[benchmark] = selected
        startup[benchmark] = next((case["case_id"] for case in positives if not case["native_success"]), None)
    ready = all(row["input_counts_satisfied"] for row in coverage.values())
    manifest = {"cases": cases, "coverage": coverage, "cost_probe_cases": probes, "startup_cases": startup,
        "status": "FROZEN" if ready else "FAIL_INPUT_COVERAGE", "compilation_allowed": ready,
        "frozen_at": datetime.now(timezone.utc).isoformat(), "config": config,
        "implementation_manifest_sha256": file_identity(output / "execution_manifest.json")["sha256"],
        "all_new_episodes_included": [row["episode_id"] for row in collected]}
    write_json_exclusive(output / "final_case_manifest.json", manifest)
    return manifest


def canonical_action(text: str) -> str:
    try:
        value = json.loads(text)
    except ValueError:
        return text
    return canonical_bytes(value).decode("utf-8")


def replay(candidate: Any, sources: dict[str, Source], documents: dict[str, str], view: dict[str, Any]) -> dict[str, Any]:
    validate(candidate, sources, documents)
    state = {"cursor": 0, "status": None, "divergence": None}
    def native(action, node, history):
        cursor = state["cursor"]
        if cursor >= len(view["events"]):
            state["status"] = "SOURCE_INCOMPLETE"
            raise ProgramYield("SourceIncomplete")
        if canonical_action(action) != canonical_action(executed_action(view["events"][cursor])):
            state.update(status="DIVERGED_AT_EVENT", divergence=cursor)
            raise ProgramYield("ActionDiverged")
        # Only a matched action releases this source return to the program.
        state["cursor"] += 1
        return view["events"][cursor]["obs_after"]
    result = execute(candidate, sources, resolve_source=lambda source, prefix: source.value,
        call_native=native, action_is_bound=lambda action, prefix: True,
        semantic_batch=lambda *args: (_ for _ in ()).throw(ProgramYield("SemanticExecutionRequired")),
        budget_available=lambda: True)
    return {"status": state["status"] or ("EXACT_WITNESS_REPLAY" if state["cursor"] == len(view["events"]) else "PREFIX_CONSISTENT"),
        "source_events_consumed": state["cursor"], "first_divergence_event": state["divergence"],
        "exit": result.exit, "reason": result.unknown_reason, "native_execution_count": 0, "native_success_count": 0}


def compile_cases(config: dict[str, Any], output: Path, policy: Any, frozen: dict[str, Any], collected: list[dict[str, Any]]) -> None:
    manifests = read_json(output / "execution_manifest.json")
    for benchmark in config["benchmarks"]:
        ledger = ExplorationLedger(event_path=output / "cost_probes" / (benchmark + ".startup_leases.jsonl"))
        paid = [row for row in collected if row["benchmark"] == benchmark]
        for row in paid:
            ledger.credit_paid_base(row["episode_id"], Resources(**row["paid_resources"]), usage_complete=row["meter"]["usage_complete"])
        cases = [case for case in frozen["cases"] if case["benchmark"] == benchmark]
        # The designated failed source is first; remaining slots retain preregistered order.
        cases.sort(key=lambda case: (case["case_id"] != frozen["startup_cases"][benchmark], case["preregistered_order"]))
        for case in cases:
            round_id = "compile-" + case["case_id"]
            ledger.open_round(round_id, Resources(config["collection"]["episode_max_wall_seconds"],
                config["collection"]["episode_max_model_tokens"], config["collection"]["episode_max_steps"][benchmark]))
            leases = ledger.reserve_compile_and_trial(case["case_id"], round_id,
                Resources(**manifests["startup_leases"]["compile"]), Resources(**manifests["startup_leases"]["trial"]))
            if leases is None:
                write_json_exclusive(output / "compiler_outputs" / (case["case_id"] + ".result.json"), {
                    "case_id": case["case_id"], "status": "DEFERRED_BY_BUDGET", "call_ids": [], "charged_round_id": round_id})
                continue
            meter = BudgetMeter(BudgetLimits(1, 18432, 60), event_path=output / "physical_events.jsonl", charged_round_id=round_id)
            policy.meter = meter
            started = monotonic_seconds()
            try:
                trace = read_json(output / case["source_path"])
                material = build_material(public_view(trace), case["end_event_exclusive"])
                result = compile_window(policy, material, output, case["case_id"], lease_seconds=60,
                    source_class=case.get("source_class", "INTERFACE_REPAIR_HYPOTHESIS"))
                if result["status"] == "AWAITING_INDEPENDENT_PUBLIC_REVIEW":
                    candidate = candidate_from_dict(read_json(output / "asts" / (case["case_id"] + ".json")))
                    source_replay = replay(candidate, material.sources, material.documents, material.public_view)
                    write_json_exclusive(output / "replay_results" / (case["case_id"] + ".json"), source_replay)
            finally:
                ledger.settle(leases[0], Resources(monotonic_seconds() - started, meter.total_model_tokens, 0), usage_complete=meter.usage_complete)
                cost_event(output, round_id, "compile_outer_interval", started, nested_costs_not_additive=True)
                write_json_exclusive(output / "cost_probes" / (case["case_id"] + ".startup.json"), {
                    "designated_startup_case": case["case_id"] == frozen["startup_cases"][benchmark],
                    "paid_base_episodes": len(paid), "paid": asdict(ledger.paid),
                    "compile_lease_id": leases[0], "trial_lease_id": leases[1],
                    "compile_lease": {key: asdict(value) if isinstance(value, Resources) else value
                                      for key, value in ledger.leases[leases[0]].items()},
                    "trial_lease": {key: asdict(value) if isinstance(value, Resources) else value
                                    for key, value in ledger.leases[leases[1]].items()},
                    "stopped": ledger.stopped, "trial_executed": False})
                policy.meter = None
            if ledger.stopped:
                break


def probe(config: dict[str, Any], output: Path, policy: Any, reviews_path: Path) -> dict[str, Any]:
    import torch
    frozen = read_json(output / "final_case_manifest.json")
    reviews = read_json(reviews_path)
    if reviews.get("independent_reviewer") in (None, "", "compiler", "self"):
        raise ValueError("A named independent public-evidence reviewer is required")
    write_json_exclusive(output / "public_evidence_reviews.json", reviews)
    results = []
    for benchmark in config["benchmarks"]:
        for index, case_id in enumerate(frozen["cost_probe_cases"][benchmark]):
            case = next(case for case in frozen["cases"] if case["case_id"] == case_id)
            candidate = candidate_from_dict(read_json(output / "asts" / (case_id + ".json")))
            review = reviews["cases"].get(case_id, {})
            if review.get("version") != candidate.version or review.get("approved") is not True:
                raise ValueError("Probe candidate lacks matching independent public-evidence approval")
            prov = read_json(output / "provenance" / (case_id + ".json"))
            sources = {key: Source(**row) for key, row in prov["sources"].items()}
            validate(candidate, sources, prov["documents"])
            trace = read_json(output / case["source_path"])
            view = public_view(trace)
            event = trace["events"][0]
            messages = event["messages_before"]
            schema = view["events"][0]["schema"]
            if not schema:
                raise ValueError("Source-aligned base prefix lacks the native public schema")
            order = ("base", "program") if index % 2 == 0 else ("program", "base")
            arms = {}
            for arm in order:
                round_id = f"probe-{case_id}-{arm}"
                meter = BudgetMeter(BudgetLimits(1, 262144, 1800), event_path=output / "physical_events.jsonl", charged_round_id=round_id)
                policy.meter = meter
                started = monotonic_seconds()
                action, fallback, batches = None, False, 0
                seed = 2026111200 + case["preregistered_order"]
                def base(role):
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        with policy.frozen_reference():
                            generation = policy._generate_native_text(messages, schema, mode=None,
                                greedy=False, role=role, max_new_tokens=512)
                    return generation.action.get("text", generation.text)
                def native(value, node, history):
                    nonlocal action
                    action = value
                    raise ProgramYield("SourceAlignedActionReady")
                def semantic(fields, history, version):
                    nonlocal batches
                    prompt = [{"role": "system", "content": "Return a JSON object with exactly these requested public predicate fields. Missing evidence is U. Schema=" + json.dumps(fields)},
                        {"role": "user", "content": canonical_bytes({"goal": view["goal"], "observation": view["events"][0]["obs_before"], "history": history}).decode("utf-8")}]
                    if len(policy._tokenize_messages(prompt)) > 8192:
                        raise ProgramYield("SemanticContextTooLarge")
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(seed + 100000)
                        torch.cuda.manual_seed_all(seed + 100000)
                        with policy.frozen_reference():
                            generation = policy.complete(prompt, role="program.semantic_batch", greedy=True, max_new_tokens=1024)
                    batches += 1
                    return json.loads(generation.text)
                try:
                    if arm == "base":
                        action = base("probe.base.action")
                    else:
                        available = view["events"][0]["obs_before"]["classic_text_action_state"].get("available_actions", [])
                        def bound(value, history):
                            return value in available or (schema.get("search_available") and value.startswith("search[")) or bool(schema.get("native_action_schemas_by_mode"))
                        execute(candidate, sources, resolve_source=lambda source, history: source.value,
                            call_native=native, action_is_bound=bound, semantic_batch=semantic,
                            budget_available=lambda: meter.elapsed_seconds < meter.limits.max_wall_seconds)
                        if action is None:
                            fallback = True
                            action = base("program.fallback.action")
                    arms[arm] = {"action": action, "model_tokens": meter.total_model_tokens,
                        "wall_seconds": monotonic_seconds() - started, "fallback": fallback,
                        "semantic_batches": batches, "usage_complete": meter.usage_complete,
                        "call_ids": [call.call_id for call in meter.model_calls]}
                    if not meter.usage_complete or meter.exploration_stopped:
                        raise RuntimeError("Cost probe has incomplete usage or a lease overrun")
                finally:
                    policy.meter = None
                    cost_event(output, round_id, "control_outer_interval", started, nested_costs_not_additive=True)
            row = {"benchmark": benchmark, "case_id": case_id, "episode_id": case["episode_id"],
                "source_prefix_event": 0, "order": order, "arms": arms, "native_execution_count": 0}
            results.append(row)
            write_json_exclusive(output / "cost_probes" / (case_id + ".control.json"), row)
    ratios = {}
    for benchmark in config["benchmarks"] + ["overall"]:
        rows = [row for row in results if benchmark == "overall" or row["benchmark"] == benchmark]
        ratios[benchmark] = {dimension: sum(row["arms"]["program"][dimension] for row in rows) /
            sum(row["arms"]["base"][dimension] for row in rows) for dimension in ("model_tokens", "wall_seconds")}
    write_json_exclusive(output / "cost_probes" / "control_ratios.json", ratios)
    return ratios


def audit_physical(output: Path) -> dict[str, Any]:
    path = output / "physical_events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    starts = {row["call_id"]: row for row in rows if row["kind"] == "call_started"}
    finishes = {row["call_id"]: row for row in rows if row["kind"] == "call_finished"}
    costs = [row for row in rows if row["kind"] not in {"call_started", "call_finished"}]
    model_costs = [row for row in costs if row["kind"] == "model_call"]
    closed = set(starts) == set(finishes)
    usage = all(row["usage_complete"] for row in finishes.values())
    unique = len({row["cost_event_id"] for row in costs}) == len(costs)
    billed = {row["call_id"] for row in model_costs}
    ids_match = billed == set(finishes) and all(row.get("charged_round_id") == starts[key]["charged_round_id"] for key, row in finishes.items())
    totals_match = (sum(row["known_input_tokens"] + row["known_output_tokens"] for row in model_costs) ==
                    sum((row["input_tokens"] or 0) + (row["output_tokens"] or 0) for row in finishes.values()))
    return {"calls_started": len(starts), "calls_finished": len(finishes), "closed": closed,
        "usage_complete": usage, "unique_cost_ids": unique, "call_cost_ids_match": ids_match,
        "known_token_totals_match": totals_match, "pass": closed and usage and unique and ids_match and totals_match,
        "lease_overruns": [key for key, row in finishes.items() if row["lease_status"] == "LEASE_OVERRUN"],
        "total_charged_outer_wall_seconds": sum(row["elapsed_seconds"] for row in costs
            if row["kind"] in {"model_setup", "environment_setup", "episode_outer_interval", "compile_outer_interval", "control_outer_interval"}),
        "model_calls": [row for row in rows if row["kind"] in {"call_started", "call_finished"}], "cost_events": costs}


def report(config: dict[str, Any], output: Path, *, fatal: str | None = None, final: bool) -> dict[str, Any]:
    audit = audit_physical(output)
    frozen = read_json(output / "final_case_manifest.json") if (output / "final_case_manifest.json").exists() else None
    inventory = read_json(output / "recovery_inventory.json")
    collected_path = output / "collected_episodes.jsonl"
    collected = [json.loads(line) for line in collected_path.read_text().splitlines()] if collected_path.exists() else []
    criteria = {name: {"pass": False, "status": "NOT_RUN", "failed_case_ids": []} for name in HARD_CRITERIA}
    criteria["raw_integrity"] = {"pass": not any(row["integrity_errors"] for row in inventory["episodes"] + collected), "status": "CHECKED",
        "recovered_blob_references": inventory["verified_blob_references"],
        "new_blob_references": sum(row["verified_blob_references"] for row in collected)}
    criteria["public_boundary"] = {"pass": inventory["private_perturbation_checks"] == len(inventory["episodes"])
        and all(row["private_perturbation_passed"] for row in collected), "status": "CHECKED"}
    criteria["physical_accounting"] = {"pass": audit["pass"], "status": "CHECKED", **{k: v for k, v in audit.items() if k not in {"model_calls", "cost_events", "pass"}}}
    outer_overruns = []
    for path in (output / "cost_probes").glob("*.startup_leases.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "LEASE_OVERRUN":
                outer_overruns.append(row["lease_id"])
    criteria["deadline_and_leases"] = {"pass": not audit["lease_overruns"] and not outer_overruns,
        "status": "CHECKED", "overruns": audit["lease_overruns"] + outer_overruns}
    cases = frozen["cases"] if frozen else []
    reviews = read_json(output / "public_evidence_reviews.json") if (output / "public_evidence_reviews.json").exists() else {"cases": {}}
    by_benchmark = {}
    qualified = set()
    for case in cases:
        ast_path = output / "asts" / (case["case_id"] + ".json")
        replay_path = output / "replay_results" / (case["case_id"] + ".json")
        if ast_path.exists() and replay_path.exists():
            ast = read_json(ast_path)
            review = reviews["cases"].get(case["case_id"], {})
            if review.get("approved") is True and review.get("version") == ast["version"] and read_json(replay_path)["status"] in {"EXACT_WITNESS_REPLAY", "PREFIX_CONSISTENT"}:
                qualified.add(case["case_id"])
    structures = set()
    for benchmark in config["benchmarks"]:
        positives = [case for case in cases if case["benchmark"] == benchmark and case["role"] == "positive"]
        good = [case for case in positives if case["case_id"] in qualified]
        ops = set()
        for case in good:
            candidate = candidate_from_dict(read_json(output / "asts" / (case["case_id"] + ".json")))
            ops.update(node.op for node in walk(candidate.ast))
            if any(node.reads for node in walk(candidate.ast) if node.op == "Call"):
                ops.add("cross_event_binding")
        structures.update(ops)
        by_benchmark[benchmark] = {"qualified": len(good), "denominator": 4,
            "failed_source_qualified": sum(not case["native_success"] for case in good), "structures": sorted(ops),
            "failed_case_ids": [case["case_id"] for case in positives if case["case_id"] not in qualified]}
    criteria["compiler_yield"] = {"pass": all(row["qualified"] >= 3 for row in by_benchmark.values()) and len(by_benchmark) == 3,
        "status": "CHECKED" if frozen else "NOT_RUN", "benchmarks": by_benchmark}
    criteria["failure_source_start"] = {"pass": len(by_benchmark) == 3 and all(row["failed_source_qualified"] >= 1 for row in by_benchmark.values()), "status": "CHECKED" if frozen else "NOT_RUN"}
    criteria["types_and_sources"] = {"pass": len(qualified) >= 9, "status": "INDEPENDENT_REVIEW_REQUIRED" if not reviews["cases"] else "CHECKED"}
    criteria["liveness_and_effects"] = {"pass": len(qualified) >= 9 and all(reviews["cases"].get(case_id, {}).get("liveness_and_effects") is True for case_id in qualified), "status": "INDEPENDENT_REVIEW_REQUIRED"}
    criteria["expression_coverage"] = {"pass": {"Loop", "Frontier", "If3", "cross_event_binding"} <= structures
        and all({"Loop", "Frontier"} <= set(by_benchmark.get(benchmark, {}).get("structures", [])) for benchmark in ("alfworld", "webshop")), "status": "CHECKED" if frozen else "NOT_RUN", "structures": sorted(structures)}
    negatives = [case for case in cases if case["role"] == "counterexample"]
    rejected = []
    for case in negatives:
        path = output / "compiler_outputs" / (case["case_id"] + ".result.json")
        if path.exists() and read_json(path)["status"] == "REJECTED_SOURCE_OR_TYPE":
            rejected.append(case["case_id"])
    criteria["counterexamples"] = {"pass": len(rejected) == 6, "status": "CHECKED" if frozen else "NOT_RUN", "rejected": rejected, "denominator": 6}
    ratio_path = output / "cost_probes" / "control_ratios.json"
    if ratio_path.exists():
        ratios = read_json(ratio_path)
        criteria["execution_cost"] = {"pass": all(all(value <= 1 for value in ratios[benchmark].values()) for benchmark in config["benchmarks"])
            and all(value < 1 for value in ratios["overall"].values()), "status": "CHECKED", "ratios": ratios}
    startup_pass = []
    for benchmark, case_id in (frozen or {}).get("startup_cases", {}).items():
        if not case_id:
            continue
        path = output / "cost_probes" / (case_id + ".startup.json")
        result_path = output / "compiler_outputs" / (case_id + ".result.json")
        if path.exists() and result_path.exists():
            row = read_json(path)
            if row["paid_base_episodes"] <= 10 and not row["stopped"] and row["trial_lease"]["status"] == "RESERVED" and read_json(result_path)["status"] == "AWAITING_INDEPENDENT_PUBLIC_REVIEW":
                startup_pass.append(benchmark)
    criteria["startup_cost"] = {"pass": len(startup_pass) == 3, "status": "CHECKED" if frozen else "NOT_RUN", "passed_benchmarks": startup_pass}
    verdict = "PASS" if all(row["pass"] for row in criteria.values()) and not fatal and (frozen or {}).get("status") == "FROZEN" else "FAIL"
    result = {"run_id": config["run_id"], "overall": verdict,
        "input_status": (frozen or {}).get("status", "NOT_FROZEN"), "criteria": criteria,
        "fatal_error": fatal, "new_base_episodes": len(collected), "recovered_episodes": len(inventory["episodes"]),
        "qualified_case_ids": sorted(qualified), "all_attempts_retained": True,
        "posterior_learning": False, "online_graph_scheduling": False}
    write_json_exclusive(output / ("gate0_report.json" if final else "gate0_compile_report.json"), result)
    if final:
        for name, rows in (("model_calls.jsonl", audit["model_calls"]), ("cost_events.jsonl", audit["cost_events"])):
            with (output / name).open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(canonical_bytes(row).decode("utf-8") + "\n")
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as package:
            for path in sorted(output.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(output).as_posix()
                if relative == "manifest.json":
                    relative = "recovery_manifest.json"
                elif relative == "case_manifest.json":
                    relative = "recovery_case_manifest.json"
                elif relative == "final_case_manifest.json":
                    relative = "case_manifest.json"
                package.write(path, relative)
            package.writestr("manifest.json", canonical_bytes({
                "recovery": read_json(output / "manifest.json"),
                "execution": read_json(output / "execution_manifest.json") if (output / "execution_manifest.json").exists() else None,
                "final_case_manifest_sha256": file_identity(output / "final_case_manifest.json")["sha256"] if frozen else None,
                "gate0_report": "gate0_report.json", "wall_accounting": "outer intervals only; nested model/environment intervals explain components"}).decode("utf-8"))
        write_json_exclusive(output.parent / (output.name + ".zip.identity.json"), file_identity(archive))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    output = check_host(config)
    policy = None
    try:
        if args.probe:
            verify_implementation(output)
            if args.reviews is None:
                raise ValueError("--probe requires --reviews")
            policy = load_policy(config, output)
            probe(config, output, policy, args.reviews)
            result = report(config, output, final=True)
        else:
            freeze_implementation(config, output)
            policy = load_policy(config, output)
            collected = collect(config, output, policy)
            frozen = final_freeze(config, output, collected)
            if not frozen["compilation_allowed"]:
                result = report(config, output, final=True)
            else:
                compile_cases(config, output, policy, frozen, collected)
                result = report(config, output, final=False)
        print(json.dumps({"overall": result["overall"], "input_status": result["input_status"],
            "report": str(output), "awaiting_independent_review": not args.probe and result["input_status"] == "FROZEN"}), flush=True)
    except BaseException as exc:
        write_json_exclusive(output / "execution_failure.json", {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        if not (output / "gate0_report.json").exists():
            report(config, output, fatal=f"{type(exc).__name__}: {exc}", final=True)
        raise
    finally:
        if policy is not None:
            policy.meter = None


if __name__ == "__main__":
    main()
