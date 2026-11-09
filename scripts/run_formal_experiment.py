#!/usr/bin/env python3
"""Run the fixed native-r3 R/W/P discriminant experiment in one invocation.

Each GPU has one worker at a time. Training collects full native train-only
demonstrations, fits the shared warmstart, and executes the fixed PRISM groups.
R, W and P evaluation run in fresh processes that never load training-reference
providers. Every phase is scheduled once; no success threshold controls release.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("appworld", "planbench_xl")
IMPLEMENTATION_REVISION = "native-r3.4-official-alignment"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return datetime.fromtimestamp(time.clock_gettime(time.CLOCK_REALTIME), timezone.utc).isoformat()


def select_tasks(env: Any, config: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Read the predeclared IDs and seeds; no selection uses outcomes or GT."""
    registered = {split: {item["task_id"]: item for item in env.list_tasks(split)}
                  for split in ("train", "dev")}
    selection = config["tasks"][env.benchmark]
    selected_train, selected_dev = [], []
    for label, split, target in (("train", "train", selected_train), ("development", "dev", selected_dev)):
        for fixed in selection[label]:
            task_id = str(fixed["task_id"])
            if task_id not in registered[split] or "environment_seed" not in fixed:
                raise ValueError(f"An explicit registered {split} task and environment_seed are required: {task_id}.")
            target.append({**registered[split][task_id], **fixed, "split": split})
        if len(target) != 2 or len({item["task_id"] for item in target}) != len(target):
            raise ValueError("The shortest native-r3 experiment fixes exactly two distinct tasks per split and benchmark.")
    if {task["task_id"] for task in selected_train} & {task["task_id"] for task in selected_dev}:
        raise ValueError("Native training and development task identities overlap.")
    return selected_train, selected_dev


def evaluation_summary(records: list[dict[str, Any]], expected_tasks=None) -> dict[str, Any]:
    """Descriptive outcomes from the one formal evaluation stream."""
    from collections import Counter
    from prism_evolve.native_training import _distribution
    from prism_evolve.statistics import official_benchmark_summary
    total = len(records)
    if total == 0:
        raise ValueError("The formal evaluation returned no task records.")
    steps = sum(record["environment_steps"] for record in records)
    successes = sum(record["success"] for record in records)
    outcomes = [record["metadata"]["program_outcomes"] for record in records]
    token_records = [item["token_usage"] for item in outcomes]
    token_counts = {
        key: [value for record in token_records for value in record[key]]
        for key in ("action_prompt_tokens", "action_output_tokens", "mode_prompt_tokens", "mode_output_tokens")
    }
    stop_reasons: Counter[str] = Counter()
    for record in token_records:
        stop_reasons.update(record["stop_reasons"])
    token_evidence = {
        "generated_actions": sum(item["total_events"] for item in outcomes),
        **{key: _distribution(values) for key, values in token_counts.items()},
        "billed_input_tokens": _distribution([
            action + mode for action, mode in zip(token_counts["action_prompt_tokens"], token_counts["mode_prompt_tokens"], strict=True)
        ]),
        "billed_output_tokens": _distribution([
            action + mode for action, mode in zip(token_counts["action_output_tokens"], token_counts["mode_output_tokens"], strict=True)
        ]),
        **{key: sum(record[key] for record in token_records) for key in (
            "probability_steps", "finite_log_probability_steps", "aligned_sampling_logprob_steps",
            "complete_program_steps", "empty_response_steps", "length_capped_steps", "eos_stopped_steps",
        )},
        "stop_reasons": dict(stop_reasons),
    }
    tool_evidence = {
        key: sum(item[key] for item in outcomes) for key in (
            "attempted_api_calls", "successful_api_calls", "successful_business_calls",
            "successful_documentation_calls", "successful_submission_calls",
        )
    }
    tool_evidence.update({
        key: sum(item["tool_execution"][key] for item in outcomes) for key in (
            "native_execution_successes", "native_execution_failures", "successful_retrieval_actions",
        )
    })
    return {
        "official_metrics": official_benchmark_summary(records, expected_tasks),
        "task_count": total, "native_success_count": successes,
        "native_success_rate": successes / total,
        "success_metric": "budget_compliant_success",
        "budget_compliant_success_count": successes,
        "budget_compliant_success_rate": successes / total,
        "mean_utility": sum(record["utility"] for record in records) / total,
        "environment_steps": steps, "mean_trajectory_steps": steps / total,
        "terminations": dict(Counter(record["termination"] for record in records)),
        "token_generation": token_evidence,
        "tool_execution": tool_evidence,
        "per_task_execution": [
            {"task_id": record["task_id"], "success": record["success"],
             "execution_result": record["metadata"]["execution_result"],
             "program_outcomes": record["metadata"]["program_outcomes"]} for record in records
        ],
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != "native-r3" or config.get("arms") != ["R", "W", "P"]:
        raise ValueError("This entry point requires protocol=native-r3 and the fixed arms [R,W,P].")
    if config.get("branch_anchor") != "first_public_feasible_frontier":
        raise ValueError("Native-r3 uses the first public feasible discovery/execution frontier.")
    if int(config["warmstart_epochs"]) != 4 or int(config["prism_groups"]) != 4:
        raise ValueError("This discriminant experiment fixes four SFT epochs and four PRISM groups.")
    if config.get("evaluation_greedy") is not True:
        raise ValueError("The declared native-r3 evaluation is greedy pass@1.")
    return config


def _runtime_policy(config: dict[str, Any], directory: Path) -> Any:
    import sqlite3  # Load the simulator SQLite library before GPU extensions.
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from prism_evolve.native_policy import NativeCausalPolicy
    if not torch.cuda.is_available():
        raise RuntimeError("The configured formal experiment requires a CUDA GPU.")
    torch.set_num_threads(int(config["cpu_threads"]))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy = NativeCausalPolicy(
        model_path=config["model_path"], device="cuda:0", dtype=getattr(torch, config["dtype"]),
        lora_rank=int(config["lora_rank"]), lora_alpha=float(config["lora_alpha"]),
        max_new_tokens=int(config["max_new_tokens"]), max_context_tokens=int(config["max_context_tokens"]),
        temperature=float(config["temperature"]), chat_template_kwargs=config["chat_template_kwargs"],
        gradient_checkpointing=True, activation_offload_cpu=bool(config["activation_offload_cpu"]),
        activation_offload_min_bytes=int(config["activation_offload_min_bytes"]), local_files_only=True,
    )
    write_json(directory / "runtime.json", {
        "python": sys.version, "gpu": torch.cuda.get_device_name(0),
        "packages": {name: version(name) for name in ("torch", "transformers", "peft", "numpy", "xgrammar", "pydantic", "appworld")},
        "model": policy.model_metadata,
    })
    return policy


def _environment_factory(config: dict[str, Any], directory: Path, benchmark: str):
    from prism_evolve.native_envs import NativeAppWorldEnv, NativePlanBenchEnv

    def environment(label: str, *, reference_execution: bool = False) -> Any:
        if benchmark == "appworld":
            return NativeAppWorldEnv(
                config["appworld_data_root"],
                experiment_name=f"prism-native-r3/{directory.parent.name}/{directory.name}/{label}",
                max_steps=int(config["episode_max_steps"]),
                timeout_seconds=int(config["teacher_reference_timeout_seconds"] if reference_execution
                                    else config["appworld_code_timeout_seconds"]),
            )
        return NativePlanBenchEnv(
            config["planbench_repo_root"], output_dir=directory / "native_environment" / label,
            max_steps=int(config["episode_max_steps"]), split_seed=int(config["split_seed"]),
            run_config_path=config["planbench_run_config"],
        )
    return environment


def _worker_status(directory: Path, benchmark: str, phase: str, arm: str | None) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    state = {"protocol": "native-r3", "implementation_revision": IMPLEMENTATION_REVISION,
             "benchmark": benchmark, "phase": phase, "arm": arm,
             "status": "running", "started_at": now(), "pid": os.getpid(),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "unit_tests_run": False, "smoke_tests_run": False, "intermediate_success_gates": False}
    write_json(directory / "status.json", state)
    return state


def execute_training_worker(config_path: Path, directory: Path, benchmark: str) -> int:
    state = _worker_status(directory, benchmark, "full_training", None)
    policy, env = None, None
    try:
        config = _load_config(config_path)
        policy = _runtime_policy(config, directory)
        from prism_evolve.native_core import BudgetLimits, BudgetMeter
        # Evaluation workers do not execute this import or access train GT.
        from prism_evolve.native_curriculum import collect_native_demonstrations, train_warmstart
        from prism_evolve.native_training import run_training
        environment = _environment_factory(config, directory, benchmark)
        env = environment("task-identities")
        train_tasks, dev_tasks = select_tasks(env, config)
        write_json(directory / "task_selection.json", {
            "selection": "explicit_fixed_ids_before_any_teacher_or_actor_generation",
            "train": train_tasks, "development": dev_tasks, "native_splits": env.split_manifest,
        })
        env.close()
        env = None
        checkpoints: dict[str, str | None] = {"W": None, "P": None}
        write_json(directory / "checkpoint_manifest.json", checkpoints)
        meter = BudgetMeter(BudgetLimits(**config["training_budget"]), event_path=directory / "physical_costs.jsonl")
        local = {**config, "benchmark": benchmark, "method": "prism_evolve",
                 "max_train_groups": int(config["prism_groups"]), "learning_rate": float(config["prism_lr"])}
        demonstrations, supervisors, demonstration_summary = collect_native_demonstrations(
            policy, environment, train_tasks, local, meter, directory / "demonstrations")
        warmstart = train_warmstart(policy, demonstrations, local, meter, directory / "warmstart", seed=int(config["seed"]))
        checkpoints["W"] = warmstart["checkpoint"]
        write_json(directory / "checkpoint_manifest.json", checkpoints)
        env = environment("prism-training")
        prism = run_training(
            "prism_evolve", policy, env, train_tasks, [], local, meter, directory / "prism",
            int(config["seed"]), progress_supervisors=supervisors,
        )
        checkpoints["P"] = prism["selected_checkpoint"]
        write_json(directory / "checkpoint_manifest.json", checkpoints)
        result = {"demonstrations": demonstration_summary, "warmstart": warmstart, "prism": prism,
                  "checkpoints": checkpoints,
                  "usage": meter.snapshot(), "reference_visibility": "train_only",
                  "actor_success_denominator_excludes_demonstrations": True}
        write_json(directory / "training_result.json", result)
        state.update(status="completed", finished_at=now(), **result)
        write_json(directory / "status.json", state)
        return 0
    except Exception as error:
        state.update(status="runtime_error", finished_at=now(), error_type=type(error).__name__,
                     error=str(error), traceback=traceback.format_exc())
        write_json(directory / "status.json", state)
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()
        if policy is not None:
            policy.close()


def execute_evaluation_worker(config_path: Path, directory: Path, benchmark: str,
                              arm: str, training_directory: Path) -> int:
    state = _worker_status(directory, benchmark, "native_development_evaluation", arm)
    policy, env = None, None
    try:
        config = _load_config(config_path)
        policy = _runtime_policy(config, directory)
        from contextlib import nullcontext
        from prism_evolve.native_core import BudgetLimits, BudgetMeter
        from prism_evolve.native_training import _limits, evaluate_policy
        if arm not in config["arms"]:
            raise ValueError("The evaluation arm was not predeclared.")
        if arm != "R":
            checkpoints = json.loads((training_directory / "checkpoint_manifest.json").read_text(encoding="utf-8"))
            if not checkpoints[arm]:
                raise ValueError(f"The fixed training run did not produce the requested {arm} checkpoint.")
            policy.load_adapter(checkpoints[arm])
        tasks = [{**item, "split": "dev"} for item in config["tasks"][benchmark]["development"]]
        environment = _environment_factory(config, directory, benchmark)
        env = environment("development")
        meter = BudgetMeter(BudgetLimits(**config["dev_budget"]), event_path=directory / "physical_costs.jsonl")
        local = {**config, "benchmark": benchmark, "split": "dev", "arm": arm,
                 "output_dir": str(directory / "results"), "method": "prism_evolve"}
        policy.eval()
        with policy.raw_base_reference() if arm == "R" else nullcontext():
            records = evaluate_policy(
                policy, env, tasks, local, lambda: BudgetMeter(_limits(local), parent=meter),
                "prism_evolve", int(config["seed"]), config["model_id"],
            )
        summary = evaluation_summary(records, env.list_tasks("dev"))
        summary["benchmark_identity"] = env.split_manifest
        summary.update(arm=arm, usage=meter.snapshot(), reference_visibility="none",
                       evaluation="native_evaluator_greedy_pass_at_one", task_counting_unit="fixed_dev_tasks")
        write_json(directory / "evaluation_summary.json", summary)
        state.update(status="completed", finished_at=now(), evaluation=summary)
        write_json(directory / "status.json", state)
        return 0
    except Exception as error:
        state.update(status="runtime_error", finished_at=now(), error_type=type(error).__name__,
                     error=str(error), traceback=traceback.format_exc())
        write_json(directory / "status.json", state)
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()
        if policy is not None:
            policy.close()


def run_experiment(config_path: Path, output: Path, gpus: list[int], benchmarks: list[str], *, resume: bool = False) -> int:
    if len(set(gpus)) != len(gpus) or not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError("Assign distinct nonnegative GPU IDs, one worker per GPU at a time.")
    if len(set(benchmarks)) != len(benchmarks) or not benchmarks:
        raise ValueError("Select each native benchmark once.")
    config = _load_config(config_path)
    output.mkdir(parents=True, exist_ok=resume)
    frozen_config = output / "configuration.json"
    # Preserve the exact source text used by every phase; no fingerprint scheme.
    source_paths = [Path(__file__).resolve(), ROOT / "pyproject.toml", *sorted((ROOT / "src").rglob("*.py"))]
    source_text = {str(path.relative_to(ROOT)): path.read_text(encoding="utf-8") for path in source_paths}
    source_record = output / "source_snapshot.json"
    if resume:
        if not frozen_config.exists() or json.loads(frozen_config.read_text(encoding="utf-8")) != config:
            raise ValueError("Resume configuration differs from the preserved run; use a new output directory.")
        if not source_record.exists() or json.loads(source_record.read_text(encoding="utf-8")) != source_text:
            raise ValueError("Resume source differs from the preserved run; use a new output directory.")
        previous = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
        if previous.get("implementation_revision") != IMPLEMENTATION_REVISION or previous.get("benchmarks") != benchmarks:
            raise ValueError("Resume implementation or benchmark scope differs from the original run.")
        archive = output / ("experiment.before-resume." + str(time.clock_gettime_ns(time.CLOCK_REALTIME)) + ".json")
        with archive.open("x", encoding="utf-8") as stream:
            json.dump(previous, stream, ensure_ascii=False, indent=2)
    else:
        write_json(frozen_config, config)
        write_json(source_record, source_text)
    state = {"protocol": "native-r3", "implementation_revision": IMPLEMENTATION_REVISION,
             "status": "running", "started_at": now(),
             "benchmarks": benchmarks, "arms": config["arms"], "gpus": gpus,
             "configuration": str(frozen_config), "intermediate_success_gates": False,
             "unit_tests_run": False, "smoke_tests_run": False,
             "planned_actor_trajectories": 6 * len(benchmarks),
             "distinct_development_tasks": 2 * len(benchmarks)}
    write_json(output / "experiment.json", state)

    def lane(gpu: int, assignments: list[str]) -> list[dict[str, Any]]:
        records = []
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", PYTHONFAULTHANDLER="1",
            OMP_NUM_THREADS=str(config["cpu_threads"]), MKL_NUM_THREADS=str(config["cpu_threads"]),
            OPENBLAS_NUM_THREADS=str(config["cpu_threads"]), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        for benchmark in assignments:
            benchmark_dir = output / benchmark
            training_dir = benchmark_dir / "training"
            # Fixed scheduling is independent of teacher success, SFT change or
            # training success. Missing requested checkpoints remain runtime
            # errors in W/P; another policy is never substituted.
            schedule = [("training", None, training_dir)] + [
                ("evaluation", arm, benchmark_dir / arm) for arm in config["arms"]]
            for phase, arm, directory in schedule:
                status_path = directory / "status.json"
                if resume and status_path.exists():
                    existing = json.loads(status_path.read_text(encoding="utf-8"))
                    if existing.get("status") == "completed":
                        expected_phase = "full_training" if phase == "training" else "native_development_evaluation"
                        if (existing.get("implementation_revision") != IMPLEMENTATION_REVISION
                                or existing.get("benchmark") != benchmark or existing.get("arm") != arm
                                or existing.get("phase") != expected_phase):
                            raise ValueError("Completed phase identity differs from this run.")
                        existing["exit_code"] = 0
                        records.append(existing)
                        continue
                    raise ValueError(f"Incomplete phase is preserved at {directory}; choose a new run directory.")
                command = [sys.executable, str(Path(__file__).resolve()), "--config", str(frozen_config),
                           "--output-dir", str(directory), "--worker-benchmark", benchmark,
                           "--worker-phase", phase, "--training-directory", str(training_dir)]
                if arm is not None:
                    command.extend(["--worker-arm", arm])
                label = f"{benchmark}_{phase}" + (f"_{arm}" if arm is not None else "")
                log_path = output / (label + ".log")
                with log_path.open("x", encoding="utf-8") as log:
                    process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
                if status_path.exists():
                    record = json.loads(status_path.read_text(encoding="utf-8"))
                else:
                    record = {"benchmark": benchmark, "phase": phase, "arm": arm,
                              "status": "runtime_error", "error": "Worker exited without a status record."}
                record["exit_code"] = process.returncode
                if process.returncode != 0:
                    record["status"] = "runtime_error"
                records.append(record)
        return records

    assignments = [benchmarks[index::len(gpus)] for index in range(len(gpus))]
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(lane, gpu, items) for gpu, items in zip(gpus, assignments) if items]
            workers = [worker for future in futures for worker in future.result()]
    except Exception as error:
        state.update(status="runtime_error", finished_at=now(), error_type=type(error).__name__, error=str(error))
        write_json(output / "experiment.json", state)
        raise
    completed = all(item["status"] == "completed" and item["exit_code"] == 0 for item in workers)
    evaluations = [{"benchmark": item["benchmark"], "arm": item["arm"], **item["evaluation"]}
                   for item in workers if item["phase"] == "native_development_evaluation" and "evaluation" in item]
    state.update(status="completed" if completed else "runtime_error", finished_at=now(), workers=workers,
                 evaluation_cells=evaluations,
                 observed_actor_trajectories=sum(item["task_count"] for item in evaluations),
                 native_success_count=sum(item["native_success_count"] for item in evaluations))
    write_json(output / "experiment.json", state)
    print(json.dumps({"status": state["status"], "result": str(output / "experiment.json"),
                      "native_success_count": state["native_success_count"],
                      "observed_actor_trajectories": state["observed_actor_trajectories"]}, ensure_ascii=False), flush=True)
    return 0 if completed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--benchmarks", choices=BENCHMARKS, nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--resume", action="store_true", help="Reuse matching completed phases; start only phases with no prior artifacts.")
    parser.add_argument("--worker-benchmark", choices=BENCHMARKS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-phase", choices=("training", "evaluation"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-arm", choices=("R", "W", "P"), help=argparse.SUPPRESS)
    parser.add_argument("--training-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_benchmark is not None:
        if args.worker_phase == "training":
            return execute_training_worker(args.config.resolve(), args.output_dir.resolve(), args.worker_benchmark)
        if args.worker_phase == "evaluation" and args.worker_arm is not None and args.training_directory is not None:
            return execute_evaluation_worker(args.config.resolve(), args.output_dir.resolve(), args.worker_benchmark,
                                             args.worker_arm, args.training_directory.resolve())
        raise ValueError("A worker requires its explicit phase and evaluation arm/checkpoint directory.")
    return run_experiment(args.config.resolve(), args.output_dir.resolve(), args.gpus, args.benchmarks, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
