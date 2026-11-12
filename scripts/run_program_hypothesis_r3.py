#!/usr/bin/env python3
"""mgt01 formal R3: existing 30 episodes -> <=60 proposals -> matched B/R/P.

No collection, unit tests, review gates, fixed-action probes, or R2 rewrites.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import socket
import traceback
from typing import Any

from prism_evolve.native_core import BudgetExhausted, BudgetLimits, BudgetMeter, Resources, monotonic_seconds
from prism_evolve.native_program_execution import ProgramYield
from prism_evolve.native_trace_artifacts import canonical_bytes, public_observation, write_json_exclusive
from prism_evolve.program_hypothesis_r3 import (IR_INSTRUCTION, VERSION, ProgramProposal,
    RepairContext, TrialRecord, WorkLeases, append, make_context, measured, parse_proposal, select)
from prism_evolve.program_hypothesis_runtime_r3 import PublicRuntime


def read(path):
    return json.loads(Path(path).read_text())


def timed_policy(config, output):
    import torch
    from prism_evolve.native_policy import NativeCausalPolicy

    class TimedFrozenPolicy(NativeCausalPolicy):
        def _complete_impl(self, *args, **kwargs):
            # CUDA events measure actual forwards without a synchronization per
            # decoded token. Total physical call wall time includes host work.
            intervals, pending = [], []
            def before(module, arguments, named):
                ids = named.get("input_ids", arguments[0] if arguments else None)
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                pending.append(("prefill" if ids is not None and ids.shape[-1] > 1 else "decode", start))
            def after(module, arguments, named, result):
                if pending:
                    name, start = pending.pop()
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    intervals.append((name, start, end))
            # PEFT.generate delegates to the underlying causal LM's generate;
            # hooks on the PEFT wrapper never observe those forwards.
            generation_model = self.model.get_base_model()
            pre = generation_model.register_forward_pre_hook(before, with_kwargs=True)
            post = generation_model.register_forward_hook(after, with_kwargs=True, always_call=True)
            try:
                return super()._complete_impl(*args, **kwargs)
            finally:
                pre.remove()
                post.remove()
                torch.cuda.synchronize()
                call = kwargs.get("call_record")
                if call is not None:
                    for kind in ("prefill", "decode"):
                        call.timings[kind + "_gpu_seconds"] = sum(
                            start.elapsed_time(end) / 1000 for name, start, end in intervals if name == kind)
                        call.timings[kind + "_forwards"] = sum(name == kind for name, _, _ in intervals)

    with measured(output / "outer_costs.jsonl", "setup", "model_setup"):
        policy = TimedFrozenPolicy(config["model_path"], device="cuda:0",
            max_context_tokens=config["max_context_tokens"], max_new_tokens=config["actor_output_tokens"],
            chat_template_kwargs={"enable_thinking": False}, gradient_checkpointing=False,
            activation_offload_cpu=False)
        policy.eval()
        policy.dcc_mode_enabled = False
    return policy


def compile_episode(config, output, policy, leases, context, row, previous_calls=()):
    owner = "proposal-" + context.episode_id
    # Interrupted output usage stays unknown in the physical record. Admission
    # consumes its complete reserved output ceiling, without imputing a bill.
    prior_upper_tokens = sum((call.get("input_tokens") or 0) +
        (call["output_tokens"] if call.get("output_tokens") is not None else config["proposal_output_tokens"])
        for call in previous_calls)
    prior_wall = sum(call["timings"]["total_wall"] for call in previous_calls)
    meter = BudgetMeter(BudgetLimits(1,
        max(1, config["proposal_calls_per_episode"] * config["max_context_tokens"] - prior_upper_tokens),
        max(0.001, config["proposal_wall_seconds"] - prior_wall)),

        event_path=output / "physical_events.jsonl", charged_round_id=owner)
    policy.meter = meter
    record = TrialRecord(owner, context.benchmark, context.task_id, "train", "source_posthoc",
                         "proposal", context.episode_id, [context.episode_id], row["actor_seed"], 0)
    runtime = PublicRuntime(policy, None, meter, leases, output, record,
        {"instruction": "", "observation": ""}, None, row["actor_seed"])
    messages = [{"role": "system", "content": IR_INSTRUCTION},
                {"role": "user", "content": canonical_bytes(context.work_view).decode()}]
    proposal = None
    started = monotonic_seconds()
    try:
        for index in range(len(previous_calls), config["proposal_calls_per_episode"]):
            try:
                value, generation = runtime.model_call(messages, f"proposal.{index + 1}",
                    config["proposal_output_tokens"], greedy=True)
            except (ValueError, TypeError) as exc:
                if index:
                    raise
                messages = [{"role": "system", "content": IR_INSTRUCTION +
                    "This is the final call. Produce one complete proposal or explicit negative status; return valid JSON."},
                    {"role": "user", "content": canonical_bytes(context.work_view).decode()}]
                continue
            if value.get("status") == "LOCATE" and index == 0:
                evidence = [{"reference": ref, "value": select(context.documents[ref["document_id"]], ref)}
                            for ref in value.get("evidence", [])]
                names = set(value.get("required_tools", []))
                schemas = {name: tool for catalog in context.catalogs.values()
                           for name, tool in catalog.items() if name in names}
                if names - set(schemas):
                    raise ValueError("Locator requested a nonpublic API")
                messages = [{"role": "system", "content": IR_INSTRUCTION +
                    "This is the second and final call. Produce a complete restricted AST or explicit negative status."},
                    {"role": "user", "content": canonical_bytes({"episode": context.work_view,
                        "locator": value, "selected_evidence": evidence, "expanded_schemas": schemas}).decode()}]
                continue
            proposal = parse_proposal(value, context,
                [call["call_id"] for call in previous_calls] + [call.call_id for call in meter.model_calls])
            break
        if proposal is None:
            raise ValueError("No main proposal from the allowed calls")
    except Exception as exc:
        proposal = ProgramProposal(context.episode_id, "REJECTED",
                                   call_ids=[call["call_id"] for call in previous_calls] + [call.call_id for call in meter.model_calls],
                                   error=f"{type(exc).__name__}: {exc}")
    finally:
        policy.meter = None
    write_json_exclusive(output / "program_proposals" / (context.episode_id + ".json"), asdict(proposal))
    append(output / "proposal_costs.jsonl", {"episode_id": context.episode_id, "status": proposal.status,
        "wall_seconds": monotonic_seconds() - started, "meter": meter.snapshot(),
        "call_ids": [call["call_id"] for call in previous_calls] + [call.call_id for call in meter.model_calls],
        "previous_interrupted_calls": list(previous_calls),
        "prior_upper_tokens_for_admission_only": prior_upper_tokens,
        "online_rho_satisfied": False})
    print(json.dumps({"proposal": context.episode_id, "status": proposal.status,
                      "calls": len(meter.model_calls), "error": proposal.error}), flush=True)
    return proposal


def source_replay_summary(record, source_trace):
    consumed, divergence = 0, None
    events = source_trace["events"]
    for executed in record.events:
        if consumed >= len(events):
            return {"status": "SOURCE_INCOMPLETE", "source_events_consumed": consumed,
                    "future_source_returns_delivered": 0}
        # No source observation after this point is read on a mismatch.
        source_action = events[consumed]["action"]["text"]
        def normalized(action):
            try:
                return canonical_bytes(json.loads(action))
            except ValueError:
                return action
        if normalized(executed["action"]) != normalized(source_action):
            divergence = consumed
            break
        consumed += 1
    return {"status": "DIVERGED_AT_EVENT" if divergence is not None else
            "EXACT_WITNESS_REPLAY" if consumed == len(events) else "PREFIX_CONSISTENT",
            "source_events_consumed": consumed, "first_divergence_event": divergence,
            "future_source_returns_delivered": 0, "adjudication": "real_environment_only"}


def run_trial(config, output, policy, env, leases, row, source_context, proposal,
              task_id, split, identity, order, arm):
    trial_id = f"{identity}-{row['episode_id']}-{order:03d}-{arm}"
    max_steps = config["episode_max_steps"][row["benchmark"]]
    owner_lease = leases.reserve(trial_id, "trial", Resources(config["episode_max_wall_seconds"],
        config["episode_max_model_tokens"], max_steps))
    meter = BudgetMeter(BudgetLimits(max_steps, config["episode_max_model_tokens"],
                                    config["episode_max_wall_seconds"]),
        event_path=output / "physical_events.jsonl", charged_round_id=trial_id)
    seed = row["actor_seed"] if identity == "source_repair" else config["seed"] + 100000 + order
    anchor = proposal.anchor_event if proposal and proposal.status == "PROPOSAL" and identity == "source_repair" else 0
    experience = None if arm == "B" else source_context
    record = TrialRecord(trial_id, row["benchmark"], task_id, split, identity, arm,
        row["episode_id"], [] if arm == "B" else [source_context.episode_id], seed, anchor)
    write_json_exclusive(output / "trial_attempts" / (trial_id + ".json"), asdict(record))
    policy.meter, env.meter = meter, meter
    started = monotonic_seconds()
    runtime = None
    try:
        with measured(output / "outer_costs.jsonl", trial_id, "reset"):
            context = env.reset(task_id, seed=seed)
        runtime = PublicRuntime(policy, env, meter, leases, output, record, context, experience, seed)
        runtime.snapshot("public_entry")
        if identity == "source_repair":
            # Rebuild the causal prefix by real actions. A mismatched reset/return
            # immediately closes source replay; its suffix never enters runtime.
            source_trace = read(source_context.source_path)
            initial_match = canonical_bytes(public_observation(runtime.observation)) == canonical_bytes(
                public_observation(source_trace["initial_observation"]))
            if initial_match:
                for index in range(anchor):
                    if not runtime.available():
                        break
                    event = source_trace["events"][index]
                    runtime.native(event["action"]["text"], "causal_prefix")
                    if canonical_bytes(public_observation(runtime.observation)) != canonical_bytes(
                            public_observation(event["observation"])):
                        record.replay["prefix_return_divergence"] = index
                        break
            else:
                record.replay["reset_divergence"] = True
        runtime.snapshot("local_anchor") if runtime.available() else None
        if arm == "P" and runtime.available():
            if proposal is not None and proposal.status == "PROPOSAL":
                runtime.program(proposal)
            else:
                record.local_exits.append({"reason": "NO_REUSABLE_PROGRAM", "history_length": len(runtime.history)})
        runtime.finish()
        if identity == "source_repair":
            record.replay.update(source_replay_summary(record, source_trace))
    except Exception as exc:
        record.termination = f"ERROR:{type(exc).__name__}:{exc}"
        append(output / "trial_failures.jsonl", {"trial_id": trial_id, "error": record.termination,
                                               "traceback": traceback.format_exc()})
        record.terminal_y = 0.0
        record.utility = 0.0
        record.costs = meter.snapshot()
    finally:
        record.costs = {**meter.snapshot(), "online_rho_satisfied": False}
        actual = Resources(monotonic_seconds() - started, meter.total_model_tokens, meter.env_steps)
        leases.settle(owner_lease, actual)
        append(output / "outer_costs.jsonl", {"kind": "trial_outer", "charged_round_id": trial_id,
            "elapsed_seconds": actual.wall_seconds, "nested_components_not_additive": True})
        with measured(output / "outer_costs.jsonl", trial_id, "trial_export"):
            write_json_exclusive(output / "trial_records" / (trial_id + ".json"), asdict(record))
        policy.meter, env.meter = None, None
    print(json.dumps({"trial": trial_id, "Y": record.terminal_y, "termination": record.termination,
                      "tokens": meter.total_model_tokens, "env_steps": meter.env_steps}), flush=True)
    return record


def summarize(config, output, records, proposals, rows, fatal=None, resumed=False):
    proposal_costs = [json.loads(line) for line in (output / "proposal_costs.jsonl").read_text().splitlines()] if (output / "proposal_costs.jsonl").exists() else []
    physical = [json.loads(line) for line in (output / "physical_events.jsonl").read_text().splitlines()] if (output / "physical_events.jsonl").exists() else []
    calls = [row for row in physical if row["kind"] == "call_finished"]
    outer = [json.loads(line) for line in (output / "outer_costs.jsonl").read_text().splitlines()]
    by_scope = {}
    for scope in ("source_repair", "development_leave_one_out", "formal_new_target"):
        by_scope[scope] = {}
        for benchmark in config["benchmarks"] + ["overall"]:
            selected = [record for record in records if record.target_identity == scope and
                        (benchmark == "overall" or record.benchmark == benchmark)]
            arms = {}
            for arm in ("B", "R", "P"):
                members = [row for row in selected if row.arm == arm]
                arms[arm] = {"trials": len(members), "successes": sum(row.terminal_y for row in members),
                    "utility_sum": sum(row.utility for row in members),
                    "model_tokens": sum(row.costs.get("total_model_tokens", 0) for row in members),
                    "wall_seconds": sum(row.costs.get("wall_seconds", 0) for row in members),
                    "env_steps": sum(row.costs.get("env_steps", 0) for row in members),
                    "errors": [row.trial_id for row in members if row.termination.startswith("ERROR")]}
            by_scope[scope][benchmark] = {"arms": arms,
                "program_Y_gain_over_R": arms["P"]["successes"] - arms["R"]["successes"],
                "program_utility_gain_over_R": arms["P"]["utility_sum"] - arms["R"]["utility_sum"]}
    interface = {}
    for benchmark in config["benchmarks"]:
        baseline = [row for row in rows if row["benchmark"] == benchmark]
        current = [row for row in records if row.benchmark == benchmark and row.target_identity == "source_repair" and row.arm == "B"]
        interface[benchmark] = {"historical_r2_source_episodes": len(baseline),
            "historical_r2_successes": sum(row["native_success"] for row in baseline),
            "historical_r2_tokens": sum(row["meter"]["total_model_tokens"] for row in baseline),
            "shared_interface_B_trials": len(current), "shared_interface_B_successes": sum(row.terminal_y for row in current),
            "shared_interface_B_tokens": sum(row.costs.get("total_model_tokens", 0) for row in current),
            "interpretation": "paired source-task historical interface comparison; program benefit is exclusively P minus R"}
    initial_groups = {}
    for record in records:
        key = (record.benchmark, record.target_identity, record.target_task_id, record.seed)
        initial_groups.setdefault(key, []).append(record)
    initial_equivalence = [{"benchmark": key[0], "scope": key[1], "target_task_id": key[2],
        "arms": [row.arm for row in members],
        "equivalent": len(members) == 3 and all(row.initial_policy == members[0].initial_policy for row in members)}
        for key, members in initial_groups.items()]
    result = {"run_id": config["run_id"], "status": "COMPLETED" if fatal is None else "FAILED",
        "fatal_error": fatal, "proposal_episodes": len(proposals),
        "proposal_status_counts": {status: sum(row.status == status for row in proposals)
            for status in sorted({row.status for row in proposals})},
        "proposal_calls": sum(row["role"].startswith("proposal.") for row in calls),
        "proposal_known_tokens": sum((row["input_tokens"] or 0) + (row["output_tokens"] or 0)
            for row in calls if row["role"].startswith("proposal.")),
        "proposal_wall_seconds": sum(row["wall_seconds"] for row in proposal_costs),
        "all_model_calls": len(calls), "all_input_tokens": sum(row["input_tokens"] or 0 for row in calls),
        "all_output_tokens": sum(row["output_tokens"] or 0 for row in calls),
        "usage_complete": all(row["usage_complete"] for row in calls),
        "unknown_output_usage_call_ids": [row["call_id"] for row in calls if row["output_tokens"] is None],
        "unavailable_prefill_decode_call_ids": [row["call_id"] for row in calls
            if row["generation_started"] and row["timings"].get("prefill_forwards", 0) == 0],
        "historical_zero_phase_timings": "unavailable; preserved verbatim, never treated as measured zero or backfilled",
        "model_prefill_gpu_seconds": sum(row["timings"].get("prefill_gpu_seconds", 0) for row in calls),
        "model_decode_gpu_seconds": sum(row["timings"].get("decode_gpu_seconds", 0) for row in calls),
        "outer_components": {kind: sum(row["elapsed_seconds"] for row in outer if row["kind"] == kind)
            for kind in sorted({row["kind"] for row in outer})},
        "outer_wall_accounting": "setup + preparation + proposal outer + trial_outer + nonnested exports; model/native/snapshot intervals are components, not added twice",
        "trial_records": len(records), "results": by_scope, "interface_effect": interface,
        "online_rho_satisfied": False, "additional_collection": 0,
        "all_failures_retained": True,
        "common_initial_policy": {"model": config["model_id"], "frozen_reference": True,
            "lora_loaded": False, "actor_entry": "PublicRuntime.actor",
            "empty_experience_B_R_P_prompt_and_grammar_equivalence": all(row["equivalent"] for row in initial_equivalence),
            "matched_initial_inputs": initial_equivalence,
            "probe_model_calls": 0},
        "cost_records": ["physical_events.jsonl", "outer_costs.jsonl", "leases.jsonl", "proposal_costs.jsonl"],
        "record_contracts": ["repair_contexts/", "program_proposals/", "trial_records/", "raw_generations.jsonl", "snapshots/"]}
    write_json_exclusive(output / ("r3_final_report.json" if resumed else "r3_report.json"), result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="continue only uncompleted work; preserve all existing artifacts")
    args = parser.parse_args()
    config = read(args.config)
    if socket.gethostname().split(".")[0] != config["execution_host"] or Path.cwd().resolve() != Path(config["project_root"]).resolve():
        raise RuntimeError("R3 execution requires the registered mgt01 project root")
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=args.resume)
    source = Path(config["source_root"])
    rows = [json.loads(line) for line in (source / "collected_episodes.jsonl").read_text().splitlines()]
    if len(rows) != 30:
        raise ValueError("R3 consumes exactly the existing 30 complete training episodes")
    rows.sort(key=lambda row: (row["benchmark"], row["episode_id"]))
    leases = WorkLeases(output / "leases.jsonl")
    contexts, proposals, records, policy, env = {}, [], [], None, None
    if not args.resume:
        write_json_exclusive(output / "execution_config.json", config)
    manifest_path = output / ("recovery_manifest.json" if args.resume else "execution_manifest.json")
    write_json_exclusive(manifest_path, {"version": VERSION,
        "host": socket.gethostname(), "cwd": str(Path.cwd()), "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_path": config["model_path"], "frozen_raw_base": True,
        "source_report": str(source / "gate0_report.json"), "source_episodes": [row["episode_id"] for row in rows],
        "complete_implementation_before_first_model_call": not args.resume,
        "recovery": "supervisor-approved measurement-path repair; completed episodes never reproposed" if args.resume else None,
        "online_rho_satisfied": False, "source_posthoc_and_transfer_are_separate": True})
    with measured(output / "outer_costs.jsonl", "setup", "implementation_snapshot"):
        for name in ("src/prism_evolve/program_hypothesis_r3.py",
                     "src/prism_evolve/program_hypothesis_runtime_r3.py", __file__, str(args.config)):
            destination = output / ("recovery_implementation_sources" if args.resume else "implementation_sources") / Path(name).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, destination)
    fatal = None
    try:
        for row in rows:
            if args.resume:
                contexts[row["episode_id"]] = RepairContext(**read(output / "repair_contexts" / (row["episode_id"] + ".json")))
                continue
            destination = output / "source_traces" / (row["episode_id"] + ".json")
            with measured(output / "outer_costs.jsonl", row["episode_id"], "source_copy_and_projection"):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / row["copied_path"], destination)
                context = make_context(read(destination), row, str(destination))
                write_json_exclusive(output / "repair_contexts" / (row["episode_id"] + ".json"), asdict(context))
                contexts[row["episode_id"]] = context
        previous_physical = [json.loads(line) for line in (output / "physical_events.jsonl").read_text().splitlines()] if args.resume else []
        previous_calls = [call for call in previous_physical if call["kind"] == "call_finished"]
        policy = timed_policy(config, output)
        for row in rows:
            path = output / "program_proposals" / (row["episode_id"] + ".json")
            if args.resume and path.exists():
                proposals.append(ProgramProposal(**read(path)))
                continue
            prior = [call for call in previous_calls if call["charged_round_id"] == "proposal-" + row["episode_id"]]
            proposals.append(compile_episode(config, output, policy, leases, contexts[row["episode_id"]], row, prior))
        proposal_by_id = {proposal.episode_id: proposal for proposal in proposals}
        host = read(config["host_config"])
        from prism_evolve.native_runner import _environment
        for benchmark in config["benchmarks"]:
            cfg = {**host, "episode_max_steps": config["episode_max_steps"][benchmark]}
            with measured(output / "outer_costs.jsonl", benchmark, "environment_setup"):
                env = _environment(cfg, benchmark, output / "environments" / benchmark)
            try:
                bench_rows = [row for row in rows if row["benchmark"] == benchmark]
                plan = []
                for row in bench_rows:
                    plan.append((row, row["task_id"], "train", "source_repair", proposal_by_id[row["episode_id"]]))
                source_task_ids = {row["task_id"] for row in rows}
                for split, identity, count in (("dev", "development_leave_one_out", config["development_targets_per_benchmark"]),
                                                ("test", "formal_new_target", config["formal_new_targets_per_benchmark"])):
                    tasks = sorted([task for task in env.list_tasks(split) if str(task["task_id"]) not in source_task_ids],
                                   key=lambda task: str(task["task_id"]))[:count]
                    for task in tasks:
                        eligible = [row for row in bench_rows if row["task_id"] != str(task["task_id"])]
                        chosen = next((row for row in eligible if proposal_by_id[row["episode_id"]].status == "PROPOSAL"), eligible[0])
                        plan.append((chosen, str(task["task_id"]), split, identity, proposal_by_id[chosen["episode_id"]]))
                plan_path = output / "trial_plans" / (benchmark + ".json")
                if not plan_path.exists():
                    write_json_exclusive(plan_path, [
                    {"source_episode": row["episode_id"], "target_task_id": task_id, "native_split": split,
                     "target_identity": identity, "proposal_status": proposal.status,
                     "experience_held_out_target": identity != "source_repair" and row["task_id"] != task_id,
                     "arm_order": config["arm_orders"][index % 3]}
                    for index, (row, task_id, split, identity, proposal) in enumerate(plan)])
                for index, (row, task_id, split, identity, proposal) in enumerate(plan):
                    for arm in config["arm_orders"][index % 3]:
                        record_path = output / "trial_records" / f"{identity}-{row['episode_id']}-{index:03d}-{arm}.json"
                        if args.resume and record_path.exists():
                            records.append(TrialRecord(**read(record_path)))
                            continue
                        records.append(run_trial(config, output, policy, env, leases, row,
                            contexts[row["episode_id"]], proposal, task_id, split, identity, index, arm))
            finally:
                with measured(output / "outer_costs.jsonl", benchmark, "environment_close"):
                    env.close()
                env = None
    except BaseException as exc:
        fatal = f"{type(exc).__name__}: {exc}"
        write_json_exclusive(output / ("recovery_execution_failure.json" if args.resume else "execution_failure.json"),
                             {"error": fatal, "traceback": traceback.format_exc()})
    finally:
        if policy is not None:
            policy.meter = None
        if env is not None:
            env.close()
        result = summarize(config, output, records, proposals, rows, fatal, args.resume)
        print(json.dumps({key: result[key] for key in ("status", "fatal_error", "proposal_calls", "trial_records", "all_model_calls")}), flush=True)
    if fatal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
