#!/usr/bin/env python3
"""Bounded native training batches, separate from the full evaluation matrix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gigpo", "infopo", "arew", "t3", "step_dpo", "prism_evolve",
           "ablation_unweighted", "unstratified_2branch", "frozen_base")
TYPED_METHODS = {"prism_evolve", "ablation_unweighted", "unstratified_2branch", "frozen_base"}
BENCHMARKS = ("appworld", "planbench_xl")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 43


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_server(config):
    if platform.system() != "Linux" or socket.gethostname() != config["execution_host"]:
        raise RuntimeError("Training executes only on its configured remote server.")
    if config["site_id"] not in {"host_a", "host_b"}:
        raise ValueError("This bounded batch is assigned to Host A and Host B.")


def summarize_dev_records(records, method, benchmark, checkpoint_label):
    from collections import Counter
    total = len(records)
    successes = sum(int(item.get("success", 0)) for item in records)
    mean_utility = (sum(float(item.get("utility", 0.0)) for item in records) / total) if total else 0.0
    mean_steps = (sum(int(item.get("environment_steps", 0)) for item in records) / total) if total else 0.0
    outcomes = [item.get("metadata", {}).get("program_outcomes", {}) for item in records]
    total_events = sum(int(o.get("total_events", 0)) for o in outcomes)
    parse_ok = sum(int(o.get("action_parse_successes", 0)) for o in outcomes)
    grammar_rej = sum(int(o.get("truncated_programs", 0)) for o in outcomes)
    guard_blocks = sum(int(o.get("mode_guard_interceptions", 0)) for o in outcomes)
    progress_ev = sum(int(o.get("progress_events", 0)) for o in outcomes)
    terminations = dict(Counter(str(item.get("termination", "unknown")) for item in records))
    return {
        "method": method,
        "benchmark": benchmark,
        "checkpoint": checkpoint_label,
        "num_tasks": total,
        "success_count": successes,
        "final_task_success_rate": (successes / total) if total else 0.0,
        "mean_utility": mean_utility,
        "mean_trajectory_steps": mean_steps,
        "total_events": total_events,
        "action_parse_success_rate": (parse_ok / total_events) if total_events else 0.0,
        "grammar_rejection_rate": (grammar_rej / total_events) if total_events else 0.0,
        "mode_guard_interception_rate": (guard_blocks / total_events) if total_events else 0.0,
        "progress_event_rate": (progress_ev / total_events) if total_events else 0.0,
        "terminations": terminations,
    }


def prepare(args):
    config = json.loads(args.config.read_text())
    require_server(config)
    site = config["site_id"]
    if args.batch_plan:
        plan = json.loads(args.batch_plan.read_text())
    else:
        plan = {
            "batch_id": args.output_dir.name, "run_kind": "training_and_dev24_evaluation",
            "seed": SEED, "backbone": MODEL, "max_updates": None, "max_train_groups": None,
            "group_size": 4, "episode_max_steps": 100, "stagnation_repeat_limit": 3,
            "dev_evaluation": True, "dev_num_tasks": 24, "dev_selection_seed": 20261105,
            "training_budget": {"max_env_steps": 3000, "max_model_tokens": 4000000, "max_wall_seconds": 3600},
            "cells": [{"cell_id": f"qwen25_7b.{benchmark}.{method}.s43", "site_id": site,
                       "method": method, "benchmark": benchmark}
                      for method in (args.methods or METHODS) for benchmark in (args.benchmarks or BENCHMARKS)],
        }
    selected = [cell for cell in plan["cells"] if cell["site_id"] == site]
    gpus = args.gpus if args.gpus is not None else [args.gpu]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output / "input_config.json")
    save(output / "batch_plan.json", plan)
    source = output / "source"
    frozen = (args.source_root or ROOT).resolve()
    for path in (frozen / "src/prism_evolve").rglob("*.py"):
        if path.name.startswith("._") or "__pycache__" in path.parts:
            continue
        target = source / path.relative_to(frozen)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    script = source / "scripts/run_prism_training_diagnostic.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(Path(__file__), script)
    backbone = next(item for item in [*config["backbones"], *config.get("execution_backbones", [])]
                    if item["id"] == MODEL)
    group_size = int(plan.get("group_size", 4))
    config.update(
        run_kind=plan.get("run_kind", "training_and_dev24_evaluation"),
        max_updates=plan.get("max_updates"),
        max_train_groups=plan.get("max_train_groups"),
        episode_max_steps=int(plan.get("episode_max_steps", config.get("episode_max_steps", 100))),
        stagnation_repeat_limit=int(plan.get("stagnation_repeat_limit", 3)),
        gigpo_group_size=group_size,
        infopo_group_size=group_size,
        arew_group_size=group_size,
        t3_group_size=group_size,
        dev_evaluation=bool(plan.get("dev_evaluation", True)),
        dev_num_tasks=int(plan.get("dev_num_tasks", 24)),
        dev_selection_seed=int(plan.get("dev_selection_seed", 20261105)),
        training_task_selection="registered_order",
        checkpoint_selection="budget_terminal",
        backbone=MODEL, model_id=MODEL,
        output_dir=str(output),
        training_budget=dict(plan["training_budget"]),
        dev_budget_limits=dict(config["dev_budget"]),
        feedback_diagnostic={"enabled": False},
    )
    for override in config.get("method_overrides", {}).values():
        for name in ("max_updates", "max_train_groups", "training_budget", "training_task_selection"):
            override.pop(name, None)
    external = Path(config["external_root"])
    config["official_repos"] = {name: str(Path(value) if Path(value).is_absolute() else external / value)
                                for name, value in config["official_repos"].items()}
    manifest = {
        "run_kind": config["run_kind"], "run_id": output.name, "batch_id": plan["batch_id"],
        "manifest_id": f"{site}/{output.name}", "site_id": site,
        "created_at": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
        "physical_gpus": gpus, "seed": SEED, "backbone": backbone,
        "cells": selected, "batch_cells": plan["cells"],
        "actor_contracts": {method: "prism-typed-native-v1" if method in TYPED_METHODS else "native-action-v1"
                            for method in METHODS},
        "config": config, "source_root": str(source), "base_source_root": str(frozen),
        "input_configuration": str(output / "input_config.json"),
        "task_selection": "first registered native train tasks, in order; same order across methods",
        "scope": "full_budget_training_plus_fixed_24_dev_evaluation",
        "interpretation": "Complete-group budget reservation; 24-task dev set evaluated on terminal and frozen_base checkpoints.",
        "runtime": {"python": sys.version, "packages": {}},
    }
    for name in ("torch", "transformers", "peft", "appworld", "numpy", "xgrammar", "apache-tvm-ffi", "triton"):
        try:
            manifest["runtime"]["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            manifest["runtime"]["packages"][name] = None
    path = output / "manifest.json"
    save(path, manifest)
    for cell in selected:
        save(output / "cells" / cell["cell_id"] / "status.json", {**cell, "status": "pending"})
    processes = []
    stop_requested = False

    def stop_workers(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        for _, process, _ in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_workers)
    py_lib = str(Path(config["python_executable"]).resolve().parent.parent / "lib")
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_library_path = f"{py_lib}:{existing_ld}" if existing_ld else py_lib
    for slot_index, gpu in enumerate(gpus):
        if stop_requested:
            break
        worker_id = f"slot-{slot_index:02d}-gpu-{gpu}"
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           LD_LIBRARY_PATH=ld_library_path,
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                           PRISM_NATIVE_NVML_GPU_IDS=str(gpu), PYTHONUNBUFFERED="1",
                           PYTHONFAULTHANDLER="1", TOKENIZERS_PARALLELISM="false",
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        log_path = output / "workers" / f"{worker_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("x")
        process = subprocess.Popen([config["python_executable"], str(script), "--execute-manifest", str(path),
                                    "--gpu", str(gpu), "--worker-id", worker_id],
                                   env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        processes.append((worker_id, process, log))
    worker_exits = {}
    for worker_id, process, log in processes:
        worker_exits[str(worker_id)] = process.wait()
        log.close()
    states = {cell["cell_id"]: json.loads((output / "cells" / cell["cell_id"] / "status.json").read_text())
              for cell in selected}
    if stop_requested:
        for cell_id, state in states.items():
            if state["status"] in {"training", "evaluating_dev"}:
                state.update(status="interrupted", reason="parent_received_stop_signal")
                save(output / "cells" / cell_id / "status.json", state)
    save(output / "diagnostic_summary.json", {"run_kind": config["run_kind"], "cells": states,
         "manifest_id": manifest["manifest_id"], "worker_exit_codes": worker_exits,
         "stop_requested": stop_requested})
    return int(any(code != 0 for code in worker_exits.values()) or
               any(state["status"] != "training_finished" for state in states.values()))


def execute(path, gpu, worker_id=None):
    import sqlite3  # Preload conda libsqlite3/libicui18n before torch/tvm-ffi
    manifest = json.loads(path.read_text())
    config = manifest["config"]
    require_server(config)
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from prism_evolve.native_core import BudgetLimits, BudgetMeter
    from prism_evolve.native_development_metrics import _registered_feedback_tasks
    from prism_evolve.native_envs import NativeAppWorldEnv, NativePlanBenchEnv
    from prism_evolve.native_policy import NativeCausalPolicy
    from prism_evolve.native_scheduler import nvidia_inventory
    from prism_evolve.native_training import _limits, evaluate_policy, run_training
    output = path.parent
    worker_label = worker_id or f"gpu-{gpu}"
    worker = output / "workers" / worker_label
    worker.mkdir(parents=True, exist_ok=False)
    save(worker / "runtime.json", {"pid": os.getpid(), "host": socket.gethostname(), "physical_gpu": gpu,
        "worker_id": worker_label,
        "manifest_id": manifest["manifest_id"], "started_at": datetime.now(timezone.utc).isoformat()})
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_per_process_memory_fraction(float(config["scheduler"]["memory_fraction"]), 0)
    save(worker / "hardware.json", {"devices": [vars(item) for item in nvidia_inventory()],
         "cpu_count": os.cpu_count(), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]})
    policy = NativeCausalPolicy(manifest["backbone"]["model_path"], device="cuda:0",
        lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"], max_new_tokens=config["max_new_tokens"],
        max_context_tokens=config["max_context_tokens"], chat_template_kwargs=config.get("chat_template_kwargs", {}),
        activation_offload_cpu=config["activation_offload_cpu"],
        activation_offload_min_bytes=config["activation_offload_min_bytes"])
    initial = worker / "initial_adapter"
    policy.save_adapter(initial)
    save(worker / "model_runtime.json", policy.model_metadata)
    failures = False
    for cell in manifest["cells"]:
        directory = output / "cells" / cell["cell_id"]
        try:
            (directory / "claimed").mkdir()
        except FileExistsError:
            continue
        method, benchmark = cell["method"], cell["benchmark"]
        eval_only = bool(cell.get("eval_only", False) or method == "frozen_base")
        state = {**cell, "status": "evaluating_dev" if eval_only else "training",
                 "physical_gpu": gpu, "worker_pid": os.getpid(),
                 "manifest_id": manifest["manifest_id"], "initial_adapter": str(initial), "seed": SEED}
        save(directory / "status.json", state)
        env, meter = None, None
        try:
            policy.load_adapter(initial)
            policy.zero_grad(set_to_none=True)
            if benchmark == "appworld":
                env = NativeAppWorldEnv(config["appworld_data_root"],
                    experiment_name=f"training-diagnostic/{output.name}/{cell['cell_id']}",
                    max_steps=config["episode_max_steps"])
            elif benchmark == "planbench_xl":
                env = NativePlanBenchEnv(config["planbench_repo_root"], output_dir=directory / "native_environment",
                    max_steps=config["episode_max_steps"], split_seed=config["split_seed"],
                    run_config_path=config.get("planbench_run_config"))
            else:
                raise ValueError(f"Unknown registered benchmark: {benchmark}")
            selected_dev = _registered_feedback_tasks(
                env.list_tasks("dev"),
                int(config.get("dev_selection_seed", 20261105)),
                int(config.get("dev_num_tasks", 24)),
            )
            dev_tasks_24 = [
                {**dict(task), "split": "dev", "original_index": orig_idx,
                 "environment_seed": SEED * 1000003 + orig_idx}
                for orig_idx, task in selected_dev
            ]
            if eval_only:
                policy.eval()
                dev_dir = directory / "dev_eval_initial_frozen"
                dev_cfg = dict(config, benchmark=benchmark, split="dev", output_dir=str(dev_dir))
                ep_limits = _limits(dev_cfg)
                dev_meter = BudgetMeter(BudgetLimits(**config["dev_budget_limits"]),
                                        event_path=directory / "dev_costs.jsonl")
                dev_records = evaluate_policy(
                    policy, env, dev_tasks_24, dev_cfg,
                    lambda: BudgetMeter(ep_limits, parent=dev_meter),
                    method, SEED, MODEL,
                )
                dev_summary = summarize_dev_records(dev_records, method, benchmark, "initial_frozen_theta0")
                dev_summary["usage"] = dev_meter.snapshot()
                save(directory / "dev_eval_summary.json", dev_summary)
                state.update(
                    status="training_finished",
                    learning_status="frozen_initial_control",
                    diagnostic_judgment="frozen_initial_theta0_dev24_evaluated",
                    dev_evaluation=dev_summary,
                    usage=dev_meter.snapshot(),
                )
            else:
                all_train_tasks = env.list_tasks("train")
                max_groups = config.get("max_train_groups")
                tasks = all_train_tasks[:int(max_groups)] if max_groups is not None else all_train_tasks
                save(directory / "task_identity.json", {"tasks": tasks, "dev_tasks_24": dev_tasks_24,
                    "train_split": env.split_manifest,
                    "selection": "registered_native_train_order", "manifest_id": manifest["manifest_id"],
                    "initial_adapter": str(initial)})
                local_config = dict(config, benchmark=benchmark, output_dir=str(directory))
                summary_path = directory / "training" / "training_summary.json"
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    train_usage = summary.get("usage")
                else:
                    meter = BudgetMeter(BudgetLimits(**config["training_budget"]), event_path=directory / "training_costs.jsonl")
                    summary = run_training(method, policy, env, tasks, [], local_config, meter, directory / "training", SEED)
                    train_usage = meter.snapshot()
                records = [json.loads(line) for line in Path(summary["training_logs"]).read_text().splitlines()]
                method_signal = any(record.get("advantage_nonzero_count", 0) > 0 or
                    (len(record.get("branch_utilities", [])) == 2 and record["branch_utilities"][0] != record["branch_utilities"][1])
                    or record.get("preference_status") == "paired" for record in records)
                judgment = ("no_update_opportunity" if summary["updates"] == 0 else
                            "method_signal_and_parameter_change_observed" if method_signal and summary["parameter_change_steps"] else
                            "parameter_movement_without_observed_method_signal" if summary["parameter_change_steps"] else "no_parameter_change")
                constructions = [record["preference_construction"] for record in records
                                 if record.get("preference_construction") is not None]
                native_successes = (sum(record.get("trajectory_native_successes", 0) for record in records)
                                    if method != "step_dpo" else
                                    sum(int(item["reference_trajectory_success"]) + item["verified_candidate_successes"]
                                        for item in constructions) if constructions else None)
                state.update(
                    status="evaluating_dev" if config.get("dev_evaluation", True) else "training_finished",
                    learning_status=summary["learning_status"],
                    diagnostic_judgment=judgment, method_signal_observed=method_signal,
                    native_successes=native_successes,
                    success_count_scope="DPO construction trajectories" if method == "step_dpo" else "training rollout native terminals",
                    stop_reason=summary["stop_reason"],
                    groups=summary.get("groups"),
                    complete_groups=summary.get("complete_groups"),
                    updates=summary["updates"],
                    nonzero_gradient_steps=summary["nonzero_gradient_steps"],
                    parameter_change_steps=summary["parameter_change_steps"],
                    parameter_l2_norm_delta=summary.get("parameter_l2_norm_delta"),
                    cumulative_parameter_l2_norm_delta=summary.get("cumulative_parameter_l2_norm_delta"),
                    valid_preference_pairs=summary.get("valid_preference_pairs"),
                    diagnostic_metrics=summary.get("diagnostic_metrics"),
                    usage=train_usage,
                )
                save(directory / "status.json", state)
                if config.get("dev_evaluation", True):
                    policy.load_adapter(summary["selected_checkpoint"])
                    policy.eval()
                    dev_dir = directory / "dev_eval_final"
                    dev_cfg = dict(config, benchmark=benchmark, split="dev", output_dir=str(dev_dir))
                    ep_limits = _limits(dev_cfg)
                    dev_meter = BudgetMeter(BudgetLimits(**config["dev_budget_limits"]),
                                            event_path=directory / "dev_costs.jsonl")
                    dev_records = evaluate_policy(
                        policy, env, dev_tasks_24, dev_cfg,
                        lambda: BudgetMeter(ep_limits, parent=dev_meter),
                        method, SEED, MODEL,
                    )
                    dev_summary = summarize_dev_records(dev_records, method, benchmark, "budget_terminal_final")
                    dev_summary["usage"] = dev_meter.snapshot()
                    save(directory / "dev_eval_summary.json", dev_summary)
                    state["dev_evaluation"] = dev_summary
                    state["status"] = "training_finished"
        except Exception as error:
            failures = True
            state.update(status="error", error=f"{type(error).__name__}: {error}",
                         traceback=traceback.format_exc(), usage=meter.snapshot() if meter else None)
            save(directory / "status.json", state)
            if isinstance(error, torch.cuda.OutOfMemoryError) or "CUDA error" in str(error):
                raise
        finally:
            if env is not None:
                env.close()
            policy.meter = None
            policy.zero_grad(set_to_none=True)
        save(directory / "status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
    policy.close()
    return int(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--batch-plan", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--worker-id", type=str)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS)
    parser.add_argument("--execute-manifest", type=Path)
    args = parser.parse_args()
    if args.execute_manifest:
        if args.gpu is None:
            parser.error("A worker requires its assigned --gpu")
        return execute(args.execute_manifest.resolve(), args.gpu, worker_id=args.worker_id)
    if args.config is None or args.output_dir is None or (args.gpu is None and args.gpus is None):
        parser.error("--config, --output-dir and --gpu or --gpus are required")
    return prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
