"""Measure the V5 public contract in AppWorld and history credit in the household benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
from statistics import mean
from threading import Lock
from time import perf_counter

import torch

from fgpo.envs import EnvironmentPool
from fgpo.filtration import TokenFiltration
from fgpo.policy import CausalPolicy
from fgpo.public import (HORIZONS, OUTPUT_LIMITS, TERMINATORS, encode_text,
                         pinned_prompt, project_public_observation)
from fgpo.records import CostLedger
from fgpo.rollout import (_Cohort, _advance_cohort, _execute_response, _merge_cohorts,
                          _release_cohort, _sample_turn, collect_group)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, contents):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(contents, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def manifest_tasks(root, benchmark):
    rows = sorted(read_json(root.parent / "domain_manifest.json")[benchmark + "_tasks"],
                  key=lambda row: row["index"])
    if (len(rows) != 8 or len({row["index"] for row in rows}) != 8
            or len({row["task_id"] for row in rows}) != 8):
        raise ValueError(f"Expected eight distinct preregistered {benchmark} tasks")
    return rows


def frozen_policy(config, benchmark, checkpoint, ledger):
    policy = CausalPolicy(
        str(checkpoint), gpu_ids=(0,), ledger=ledger, trainable=False, benchmark=benchmark,
        max_context_tokens=int(config["max_context_tokens"]), max_new_tokens=OUTPUT_LIMITS[benchmark],
        cache_initial_capacity=int(config["cache_initial_capacity"]),
        working_memory_gib=float(config["working_memory_gib"]),
    )
    policy.sampling_temperature = 1.
    return policy


def counts(rows):
    total = Counter(
        attempts=len(rows), parse_valid=0, native_invocations=0, local_rejections=0,
        native_returns_ok=0, native_execution_failures=0, nonempty_executed_code=0,
        empty_code_invocations=0, nonempty_native_returns_ok=0,
        public_errors=0, consecutive_identical_public_errors=0, error_followups=0,
        recovered_after_error=0, response_cap_hits=0, observation_truncations=0,
        task_submissions=0, successful_submissions=0, generated_tokens=0, logical_prompt_tokens=0,
        summed_step_wall_seconds=0.,
    )
    previous_error = None
    for row in rows:
        native_ok = row["native_return_ok"] is True
        nonempty = row["nonempty_executed_code"]
        effective_ok = native_ok and nonempty
        error = row["public_feedback"] if not effective_ok else None
        total.update(
            parse_valid=int(row["parse_valid"]), native_invocations=int(row["native_invoked"]),
            local_rejections=int(not row["native_invoked"]), native_returns_ok=int(native_ok),
            native_execution_failures=int(row["native_return_ok"] is False),
            nonempty_executed_code=int(nonempty), public_errors=int(error is not None),
            empty_code_invocations=int(row["native_invoked"] and not nonempty),
            nonempty_native_returns_ok=int(effective_ok),
            response_cap_hits=int(row["response_cap_hit"]),
            observation_truncations=int(row["observation_truncated"]),
            task_submissions=int(row["submitted"]), successful_submissions=int(row["successful_submission"]),
            generated_tokens=row["generated_tokens"],
            logical_prompt_tokens=row["prompt_tokens"], summed_step_wall_seconds=row["step_wall_seconds"],
        )
        if previous_error is not None:
            total["error_followups"] += 1
            total["recovered_after_error"] += effective_ok
            total["consecutive_identical_public_errors"] += error == previous_error
        previous_error = error
    return total


def appworld_arm(root, config, arm):
    benchmark = "appworld"
    tasks = manifest_tasks(root, benchmark)
    folder = root / f"appworld-{arm}"
    folder.mkdir(exist_ok=True)
    ledger = CostLedger(folder / "costs.jsonl", gpu_ids=(0,))
    started = perf_counter()
    policy = frozen_policy(config, benchmark, config["model_path"], ledger)
    policy_load_seconds = perf_counter() - started
    active_config = {**config, "appworld_data_root": str(root / "appworld-data")}
    tokenizer = policy.tokenizer
    ledger.begin_group(f"stateful-appworld-{arm}")
    histories, turns = [], [[] for _ in tasks]
    pool = EnvironmentPool(active_config, benchmark, folder / "workers", size=len(tasks))
    with ExitStack() as resources:
        resources.callback(pool.close)
        executor = resources.enter_context(ThreadPoolExecutor(max_workers=len(tasks)))

        def start(task):
            env = pool(task["task_id"], task["environment_seed"], str(task["index"]))
            try:
                return env, env.reset(task["task_id"], initial_seed=task["environment_seed"])
            except BaseException:
                env.close()
                raise

        reset_started = perf_counter()
        worlds = list(executor.map(start, tasks))
        ledger.record("environment", "domain.appworld.reset_batch", seconds=perf_counter() - reset_started)
        for (_, context), task in zip(worlds, tasks, strict=True):
            schema, goal = context["action_schema"], context["instruction"]
            initial = project_public_observation(tokenizer, benchmark, context["observation"],
                                                 action_schema=schema, goal=goal, initial=True)
            version_args = {"benchmark": benchmark} if arm == "v5" else {}
            pin = pinned_prompt(tokenizer, schema["native_system_prompt"], initial, **version_args)
            histories.append(TokenFiltration(tokenizer, pin, f"{arm}:{task['index']}",
                                             max_new_tokens=OUTPUT_LIMITS[benchmark],
                                             max_context_tokens=policy.max_context_tokens))

        root_cache = policy.prefill_batch([history.prompt_ids for history in histories], role="domain.appworld.roots")
        resources.callback(policy.release_cache, root_cache)
        cohort = _Cohort(tuple(range(len(tasks))), policy.fork_cache(root_cache, copies=1))
        generators = [policy.new_generator(task["sampling_seed"]) for task in tasks]
        tokenizer_lock = Lock()

        def execute(slot, ids):
            env, context = worlds[slot]
            step_started = perf_counter()
            action, valid, result, observation = _execute_response(
                tokenizer, tokenizer_lock, env, benchmark, ids, context["instruction"])
            return action, valid, result, observation, perf_counter() - step_started

        print(f"{arm} AppWorld roots ready on {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
        prefix_started = perf_counter()
        try:
            for turn in range(3):
                pending = {}

                def dispatch(slot, ids):
                    pending[slot] = executor.submit(execute, slot, ids)

                with ledger.aggregate_model_calls():
                    parked, responses, _ = _sample_turn(
                        policy, cohort, generators, OUTPUT_LIMITS[benchmark], on_response=dispatch)
                cohort.state = None
                advances = {}
                try:
                    for slot, response in zip(cohort.slots, responses, strict=True):
                        action, valid, result, observed, seconds = pending[slot].result()
                        invoked = bool(result.info["native_executed"])
                        submitted = bool(result.info["official_termination"])
                        native_ok = bool(result.info["valid"]) if invoked else None
                        raw = observed.raw_observation
                        with tokenizer_lock:
                            truncated = len(encode_text(tokenizer, raw)) > 1024
                            feedback = tokenizer.decode(observed.user_body_ids, skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False)
                            response_text = tokenizer.decode(response, skip_special_tokens=True,
                                                             clean_up_tokenization_spaces=False)
                        cost = ledger.record("environment", "domain.appworld.step", seconds=seconds,
                                             env_steps=int(invoked), details={"task_index": tasks[slot]["index"],
                                             "turn": turn + 1, "native_invoked": invoked})
                        turns[slot].append({
                            "turn": turn + 1, "parse_valid": valid, "native_invoked": invoked,
                            "native_return_ok": native_ok, "submitted": submitted,
                            "successful_submission": submitted and bool(result.success),
                            "nonempty_executed_code": invoked and bool(action.get("code", "").strip()),
                            "response_cap_hit": len(response) == OUTPUT_LIMITS[benchmark]
                                                and response[-1] not in TERMINATORS,
                            "observation_truncated": truncated, "generated_tokens": len(response),
                            "prompt_tokens": len(histories[slot].prompt_ids), "step_wall_seconds": seconds,
                            "cost_event_id": cost.event_id, "action": action, "native_response": raw,
                            "public_feedback": feedback, "response": response_text,
                        })
                        advances[slot] = histories[slot].append(f"{slot}:{turn}", response, observed,
                                                               terminal=result.done or turn == 2)
                except BaseException:
                    for _, state in parked:
                        policy.release_cache(state)
                    raise
                next_cohorts = _advance_cohort(policy, root_cache, cohort, parked, advances,
                                               history_length=None, independent_roots=True)
                if not next_cohorts:
                    break
                cohort = _merge_cohorts(policy, next_cohorts, history_length=None)[0]
                print(f"{arm} AppWorld completed turn {turn + 1}", flush=True)
        finally:
            _release_cohort(policy, cohort)
        prefix_seconds = perf_counter() - prefix_started

    write_json(folder / "transcripts.json", [
        {**task, "initial": context, "turns": rows}
        for task, (_, context), rows in zip(tasks, worlds, turns, strict=True)
    ])
    per_task = [{**task, **counts(rows)} for task, rows in zip(tasks, turns, strict=True)]
    total = counts([])
    for rows in turns:
        total.update(counts(rows))
    write_json(folder / "summary.json", {
        "benchmark": benchmark, "arm": arm, "checkpoint": config["model_path"],
        "source": str(Path(__import__("fgpo").__file__).parent),
        "native_horizon": HORIZONS[benchmark], "decision_budget": 3,
        "temperature": 1., "output_limit": OUTPUT_LIMITS[benchmark], "tasks": per_task, "total": total,
        "conditional_recovery": (total["recovered_after_error"] / total["error_followups"]
                                 if total["error_followups"] else None),
        "public_error_definition": "local rejection, native execution failure, or empty source sent to native",
        "recovery_definition": "next observed turn sends nonempty source and native returns without failure",
        "costs": ledger.snapshot(), "wall_seconds": perf_counter() - started,
        "policy_load_seconds": policy_load_seconds, "stateful_decision_wall_seconds": prefix_seconds,
        "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    })


def alf_actor(root, config, checkpoint):
    benchmark = "alfworld"
    tasks = manifest_tasks(root, benchmark)
    folder = root / "alfworld"
    folder.mkdir(exist_ok=True)
    started = perf_counter()
    ledger = CostLedger(folder / "actor-costs.jsonl", gpu_ids=(0,))
    policy = frozen_policy(config, benchmark, checkpoint, ledger)
    pool = EnvironmentPool(config, benchmark, folder / "workers", size=8)
    try:
        for task in tasks:
            path = folder / f"group-{task['index']}.pt"
            if path.exists():
                continue
            print(f"ALF collecting task {task['index']}", flush=True)
            group = collect_group(policy, pool, task_id=task["task_id"], benchmark=benchmark,
                                  initial_seed=task["environment_seed"], slot_seeds=task["slot_seeds"],
                                  group_id=f"v5-domain-alfworld:{task['index']}")
            tmp = path.with_suffix(".tmp")
            torch.save(group, tmp)
            tmp.replace(path)
            print(f"ALF task {task['index']} successes="
                  f"{sum(row.official_success for row in group.trajectories)}/8", flush=True)
    finally:
        pool.close()
    write_json(folder / "actor-receipt.json", {
        "checkpoint": str(checkpoint), "temperature": 1., "horizon": HORIZONS[benchmark],
        "output_limit": OUTPUT_LIMITS[benchmark], "costs": ledger.snapshot(),
        "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "wall_seconds": perf_counter() - started,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    })


def alf_credit(root, config):
    from fgpo.credit import CrossTrajectoryCredit
    from history_credit import StructuralHistoryCredit, prepare_round

    started = perf_counter()
    benchmark = "alfworld"
    tasks = manifest_tasks(root, benchmark)
    folder = root / "alfworld"
    ledger = CostLedger(folder / "reference-costs.jsonl", gpu_ids=(0,))
    policy = None
    records = []
    for task in tasks:
        path = folder / f"group-{task['index']}.pt"
        group = torch.load(path, map_location="cpu", weights_only=False)
        trajectories = group.trajectories
        outcomes = [int(row.official_success) for row in trajectories]
        record = {**task, "group_artifact": path.name, "outcomes": outcomes, "successes": sum(outcomes),
                  "steps_by_slot": [len(row.steps) for row in trajectories]}
        mixed = 0 < sum(outcomes) < len(outcomes)
        reference = None
        if mixed:
            if policy is None:
                policy = frozen_policy(config, benchmark, config["model_path"], ledger)
            ledger.begin_group(group.group_id)
            # The helper captures V4's same pre-action feature and V5's executed history in one pass.
            reference = prepare_round(policy, trajectories, config=config)
        for version, estimator_type in (("v4", CrossTrajectoryCredit), ("v5", StructuralHistoryCredit)):
            estimator = estimator_type(trajectories, reference, config=config)
            estimator.advantages()
            record[version] = estimator.diagnostics
        records.append(record)
        print(f"ALF credit task {task['index']} successes={sum(outcomes)}/8 mixed={mixed}", flush=True)

    mixed = [record for record in records if not record["v4"]["homogeneous_group"]]
    metrics = ("matched_step_fraction", "mean_support_mass", "mean_effective_peers",
               "loo_outcome_brier", "loo_task_prior_brier")
    comparisons = {}
    for metric in metrics:
        paired = [(row["v4"][metric], row["v5"][metric]) for row in mixed]
        n_steps = sum(row["v4"]["step_count"] for row in mixed)
        comparisons[metric] = {
            "task_mean_v4": mean(a for a, _ in paired) if paired else None,
            "task_mean_v5": mean(b for _, b in paired) if paired else None,
            "task_mean_v5_minus_v4": mean(b - a for a, b in paired) if paired else None,
            "step_weighted_v4": sum(a * row["v4"]["step_count"]
                for row, (a, _) in zip(mixed, paired, strict=True)) / n_steps if n_steps else None,
            "step_weighted_v5": sum(b * row["v4"]["step_count"]
                for row, (_, b) in zip(mixed, paired, strict=True)) / n_steps if n_steps else None,
            "step_weighted_v5_minus_v4": sum((b - a) * row["v4"]["step_count"]
                for row, (a, b) in zip(mixed, paired, strict=True)) / n_steps if n_steps else None,
        }
    write_json(folder / "summary.json", {
        "benchmark": benchmark, "reference_checkpoint": config["model_path"],
        "scheduled_groups": len(tasks), "collected_groups": len(records), "mixed_groups": len(mixed),
        "scheduled_attempts": sum(len(task["slot_seeds"]) for task in tasks),
        "collected_attempts": sum(len(row["outcomes"]) for row in records),
        "native_successes": sum(row["successes"] for row in records),
        "collected_steps": sum(sum(row["steps_by_slot"]) for row in records),
        "mixed_steps": sum(row["v4"]["step_count"] for row in mixed),
        "mixed_group_metrics": comparisons, "tasks": records, "reference_costs": ledger.snapshot(),
        "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "wall_seconds": perf_counter() - started,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    })


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("appworld-v4", "appworld-v5", "alf-actor", "alf-credit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--actor", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    with torch.no_grad():
        if args.phase.startswith("appworld-"):
            appworld_arm(args.root, config, args.phase.removeprefix("appworld-"))
        elif args.phase == "alf-actor":
            alf_actor(args.root, config, args.actor)
        else:
            alf_credit(args.root, config)
