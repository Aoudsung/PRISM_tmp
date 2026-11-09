"""Immutable native-v1 host shards and model-resident worker cells."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import socket
import sys
import time
import traceback
from typing import Any, Mapping

from .native_core import BudgetLimits, BudgetMeter, monotonic_seconds


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "native-v1"
INFERENCE_METHODS = {"pos", "worldevolver", "envprobe_simple", "envprobe_judge"}
TEXT_BENCHMARKS = {"alfworld", "webshop", "clindiag", "loca"}
TEXT_EPISODE_STEPS = {"alfworld": 50, "webshop": 15, "clindiag": 30, "loca": 100}


def _actor_contract(method: str, benchmark: str | None = None) -> str:
    if method in INFERENCE_METHODS:
        return "official-controller-native-v1"
    from .native_training import actor_contract
    return actor_contract(method, classic_text=benchmark in TEXT_BENCHMARKS)


def _benchmark_config(config: Mapping[str, Any], benchmark: str) -> dict[str, Any]:
    """Use one registered horizon and generation limit throughout a benchmark."""
    name = benchmark.split(":", 1)[0]
    local = dict(config)
    steps = config.get("benchmark_episode_steps", {}).get(
        name, TEXT_EPISODE_STEPS.get(name, config["episode_max_steps"]))
    tokens = config.get("benchmark_max_new_tokens", {}).get(
        name, 4096 if name in {"clindiag", "loca"} else config["max_new_tokens"])
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("A benchmark episode horizon must be a positive integer.")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("A benchmark generation limit must be a positive integer.")
    local.update(episode_max_steps=steps, max_steps=steps, max_new_tokens=tokens)
    if name in TEXT_BENCHMARKS:
        local["action_contract_version"] = "prism-native-text-v1"
        probe = dict(config.get("envprobe", {}))
        probe["probe_budget"] = min(int(probe.get("probe_budget", 25)), steps)
        local["envprobe"] = probe
        sentinel_tokens = 512 if name in {"alfworld", "webshop", "loca"} else 4096
        local["pos"] = {**config.get("pos", {}), "belief_max_tokens": 4096,
                        "sentinel_max_tokens": sentinel_tokens, "progress_max_tokens": sentinel_tokens}
    if name in config.get("pos_by_benchmark", {}):
        local["pos"] = {**local.get("pos", {}), **config["pos_by_benchmark"][name]}
    return local


def _set_benchmark_policy(policy, config: Mapping[str, Any]) -> None:
    # The model remains resident; only its per-call generation ceiling changes.
    policy.max_new_tokens = int(config["max_new_tokens"])


def _registered_tasks(env, manifest: Mapping[str, Any], benchmark: str, split: str):
    tasks = env.list_tasks(split)
    if benchmark in TEXT_BENCHMARKS:
        registered = manifest["data_splits"][benchmark]["splits"][split]["tasks"]
        if tasks != registered:
            raise ValueError(f"The runtime {benchmark}/{split} task list differs from its frozen manifest.")
    return tasks


def _json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


def _now() -> str:
    return datetime.fromtimestamp(time.clock_gettime(time.CLOCK_REALTIME), timezone.utc).isoformat()


def _append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def _balanced_method_order(methods: list[str], seed: int, task_index: int) -> list[str]:
    """Every complete cycle assigns every method to every temporal position."""
    if not methods or len(set(methods)) != len(methods) or task_index < 0:
        raise ValueError("A paired block needs distinct methods and a nonnegative task index.")
    cycle, offset = divmod(task_index, len(methods))
    order = list(methods)
    random.Random(int(seed) + cycle * 1000003).shuffle(order)
    return order[offset:] + order[:offset]


def _gpu_load(gpu: int) -> dict[str, Any]:
    from .native_scheduler import nvidia_inventory
    try:
        device = next(item for item in nvidia_inventory() if item.index == gpu)
        return {"at": _now(), **asdict(device)}
    except Exception as error:
        # Missing telemetry is recorded separately from an experiment outcome.
        return {"at": _now(), "physical_gpu": gpu,
                "telemetry_error": f"{type(error).__name__}: {error}"}


def _host(config: Mapping[str, Any]) -> None:
    if config.get("action_contract_version") not in {
        "native-r3", "prism-typed-native-v1", "prism-native-text-v1",
    }:
        raise ValueError("Formal execution requires a registered native action contract.")
    coordination = config.get("coordination", {})
    if (coordination.get("enabled") is not True or coordination.get("protocol") != "native-cohort-claims-v1"
            or not isinstance(coordination.get("session_id"), str) or not coordination["session_id"]):
        raise ValueError("New formal execution requires an explicit unique-claim coordination session.")
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', coordination['session_id'])
            or not Path(coordination.get('registry_root', '')).is_absolute()):
        raise ValueError('Coordination requires a safe session name and an absolute Host A registry root.')
    if platform.system() != "Linux" or socket.gethostname() != config["execution_host"]:
        raise RuntimeError("Native execution is restricted to the configured remote host.")


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("A native-v1 configuration is required.")
    _host(config)
    external = Path(config["external_root"])
    config["official_repos"] = {
        name: str(Path(value) if Path(value).is_absolute() else external / value)
        for name, value in config.get("official_repos", {}).items()
    }
    config["dev_budget_limits"] = dict(config["dev_budget"])
    config["max_steps"] = int(config["episode_max_steps"])
    return config


def _environment(config: Mapping[str, Any], benchmark: str, output: Path, meter=None):
    config = _benchmark_config(config, benchmark)
    if benchmark == "appworld":
        from .native_envs import NativeAppWorldEnv
        return NativeAppWorldEnv(config["appworld_data_root"],
                                 experiment_name=f"native-v1/{output.name}",
                                 max_steps=int(config["episode_max_steps"]), meter=meter)
    if benchmark == "planbench_xl":
        from .native_envs import NativePlanBenchEnv
        return NativePlanBenchEnv(config["planbench_repo_root"], output_dir=output / "native_environment",
                                  max_steps=int(config["episode_max_steps"]),
                                  split_seed=int(config["split_seed"]),
                                  run_config_path=config.get("planbench_run_config"), meter=meter)
    if benchmark == "alfworld":
        from .native_text_envs import NativeALFWorldEnv, NativeEnvironmentProcess
        arguments = {"data_root": config["alfworld_data_root"],
                     "output_dir": str(output / "native_environment"),
                     "max_steps": int(config["episode_max_steps"])}
        if config.get("alfworld_python_executable"):
            return NativeEnvironmentProcess(
                config["alfworld_python_executable"],
                environment_class="prism_evolve.native_text_envs:NativeALFWorldEnv",
                constructor_kwargs=arguments, meter=meter,
            )
        return NativeALFWorldEnv(**arguments, meter=meter)
    if benchmark == "webshop":
        from .native_text_envs import NativeEnvironmentProcess, NativeWebShopEnv
        arguments = {"repo_root": config["webshop_repo_root"],
                     "output_dir": str(output / "native_environment"),
                     "max_steps": int(config["episode_max_steps"]),
                     "use_small": bool(config.get("webshop_use_small", True)),
                     "human_goals": bool(config.get("webshop_human_goals", False))}
        if config.get("webshop_python_executable"):
            return NativeEnvironmentProcess(
                config["webshop_python_executable"],
                environment_class="prism_evolve.native_text_envs:NativeWebShopEnv",
                constructor_kwargs=arguments, meter=meter,
            )
        return NativeWebShopEnv(**arguments, meter=meter)
    if benchmark == "clindiag":
        from .native_clindiag_env import NativeClinDiagEnv
        return NativeClinDiagEnv(
            config.get("clindiag_pos_repo_root") or config["official_repos"]["pos"],
            config["clindiag_data_archive"], split_seed=int(config["split_seed"]),
            max_steps=int(config["episode_max_steps"]), meter=meter,
            provider_config=config.get("clindiag_provider_model"),
            judge_config=config.get("clindiag_judge_model"),
        )
    if benchmark == "loca":
        from .native_text_envs import NativeEnvironmentProcess
        arguments = {
            "pos_repo_root": config.get("loca_pos_repo_root") or config["official_repos"]["pos"],
            "output_dir": str(output / "native_environment"),
            "environment_length": config.get("loca_environment_length", "8k"),
            "max_steps": int(config["episode_max_steps"]),
            "runtime_paths": list(config.get("loca_runtime_paths", ())),
            "runtime_roots": list(config.get("loca_runtime_roots", ())),
            "bwrap_executable": config.get("loca_bwrap_executable", "bwrap"),
            "vm_config": dict(config.get("loca_vm", {})),
        }
        for name in ("train_seeds", "dev_seeds", "test_seeds"):
            if f"loca_{name}" in config:
                arguments[name] = list(config[f"loca_{name}"])
        return NativeEnvironmentProcess(
            config["loca_python_executable"],
            environment_class="prism_evolve.native_loca_env:NativeLOCAEnv",
            constructor_kwargs=arguments, meter=meter,
        )
    raise ValueError(f"Unknown native benchmark: {benchmark}")


def _source_files() -> list[Path]:
    files = list((ROOT / "src" / "prism_evolve").rglob("*.py"))
    for relative in ("scripts/run_prism_evolve_experiments.py", "scripts/run_prism_native_experiments.py",
                     "scripts/collect_prism_native_experiments.py",
                     "configs/prism-native-v1.json", "configs/prism-native-host-a.json",
                     "configs/prism-native-host-b.json", "configs/prism-native-colab.json",
                     "scripts/deploy_prism_native_colab.py", "docs/native-experiment-protocol.md", "pyproject.toml"):
        path = ROOT / relative
        if path.exists():
            files.append(path)
    files.extend(path for path in (ROOT / "scripts").glob("*colab*")
                 if path.is_file() and path.suffix in {".py", ".sh"})
    files.extend((ROOT / "configs").glob("prism-native*.json"))
    # AppleDouble sidecars are binary metadata, including matches in scripts.
    return sorted({path for path in files if not path.name.startswith("._")})


def _model_identity(backbone: Mapping[str, Any]) -> dict[str, Any]:
    return {"display_id": backbone["id"], "path": str(backbone["model_path"])}


def _protect_legacy_outputs(config: Mapping[str, Any], run_dir: Path) -> None:
    roots = {ROOT.resolve(), Path(config["project_root"]).resolve()}
    forbidden = {root / relative for root in roots
                 for relative in ("outputs/prism_evolve_experiments", "results/prism_evolve")}
    if any(run_dir == path or path in run_dir.parents for path in forbidden):
        raise ValueError("Legacy result directories are immutable inputs.")


def _methods_for_backbone(config: Mapping[str, Any], backbone: str) -> list[str]:
    methods = list(config["main_methods"])
    if backbone == config["mechanism_backbone"]:
        methods.extend(config["ablation_methods"])
    return [method for method in methods if method in config["methods"]]


def prepare_run(config_path: Path, run_dir: Path, *,
                selected_backbones: list[str] | None = None,
                selected_methods: list[str] | None = None,
                pipeline_request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = _load_config(config_path)
    # The old configuration spelling is retained as provenance, never as the
    # identity of the current actor implementation.
    requested_action_contract = config["action_contract_version"]
    if all(benchmark in TEXT_BENCHMARKS for benchmark in config["benchmarks"]):
        config["action_contract_version"] = "prism-native-text-v1"
    else:
        from .native_program import PROGRAM_VERSION
        config["action_contract_version"] = PROGRAM_VERSION
    overrides = [config, *config.get("method_overrides", {}).values()]
    if any(values.get(name) is not None for values in overrides
           for name in ("max_updates", "max_train_groups")):
        raise ValueError("Formal schedules stop at registered resource budgets; debug group/update caps are retired.")
    run_dir = run_dir.resolve()
    _protect_legacy_outputs(config, run_dir)
    if (run_dir / "manifest.json").exists():
        raise FileExistsError("Run manifests are immutable; resume the existing schedule or choose a new run ID.")
    if selected_backbones:
        config["backbones"] = [item for item in config["backbones"]
                               if item["id"] in selected_backbones or item["slug"] in selected_backbones]
    if selected_methods:
        unknown = set(selected_methods) - set(config["methods"])
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")
        config["methods"] = list(selected_methods)
    if not config["backbones"] or not config["methods"] or not config["benchmarks"]:
        raise ValueError("The requested schedule is empty.")
    run_dir.mkdir(parents=True, exist_ok=True)
    source = run_dir / "source"
    for path in _source_files():
        relative = path.relative_to(ROOT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    splits = {}
    for benchmark in config["benchmarks"]:
        env = _environment(config, benchmark, run_dir / "preparation" / benchmark)
        try:
            names = ["train", "dev"] + (list(config["appworld_test_splits"])
                                              if benchmark == "appworld" else ["test"])
            for split in names:
                env.list_tasks(split)
            splits[benchmark] = env.split_manifest
        finally:
            env.close()
    upstream_manifest = Path(config["external_root"]) / "upstream_commits.json"
    methods_by_backbone = {model: _methods_for_backbone(config, model)
                           for model in config["global_backbone_ids"]}
    test_counts = {benchmark: sum(description["count"] for split, description in data["splits"].items()
                                   if split in (config["appworld_test_splits"] if benchmark == "appworld" else ["test"]))
                   for benchmark, data in splits.items()}
    model_identities = [_model_identity(item) for item in config["backbones"]]
    execution_identities = list(model_identities)
    seen_models = {item["id"] for item in config["backbones"]}
    for model in config.get("execution_backbones", []):
        if model["id"] in seen_models or model["id"] not in config["global_backbone_ids"]:
            raise ValueError("Execution capabilities must be distinct registered fixed backbones.")
        execution_identities.append(_model_identity(model))
        seen_models.add(model["id"])
    manifest = {"protocol_version": PROTOCOL, "run_kind": "formal",
                "action_contract_version": config["action_contract_version"],
                "requested_action_contract_version": requested_action_contract,
                "run_id": run_dir.name, "experiment_id": config.get("experiment_id", run_dir.name),
                "created_at": _now(), "host": socket.gethostname(),
                "site_id": config["site_id"], "workload_plan": config["workload_plan"],
                "run_dir": str(run_dir), "config": config,
                "manifest_id": f"{config['site_id']}/{run_dir.name}", "source_root": str(source),
                "model_identities": model_identities,
                "execution_model_identities": execution_identities,
                "data_splits": splits,
                "methods_by_backbone": methods_by_backbone,
                "actor_contracts": {
                    method: _actor_contract(method, config["benchmarks"][0]
                        if all(name in TEXT_BENCHMARKS for name in config["benchmarks"]) else None)
                    for method in config["methods"]},
                "actor_contracts_by_benchmark": {
                    benchmark: {method: _actor_contract(method, benchmark) for method in config["methods"]}
                    for benchmark in config["benchmarks"]},
                "benchmark_runtime": {
                    benchmark: {key: value for key, value in _benchmark_config(config, benchmark).items()
                                if key in {"episode_max_steps", "max_steps", "max_new_tokens", "pos", "envprobe"}}
                    for benchmark in config["benchmarks"]},
                "experiment_plan": {
                    "checkpoint_selection": "budget_terminal",
                    "main_methods": config["main_methods"],
                    "ablation_methods": config["ablation_methods"],
                    "mechanism_backbone": config["mechanism_backbone"],
                    "expected_cells": sum(len(methods) for methods in methods_by_backbone.values())
                                      * len(config["benchmarks"]) * len(config["seeds"]),
                    "expected_test_episodes": sum(len(methods) for methods in methods_by_backbone.values())
                                              * sum(test_counts.values()) * len(config["seeds"]),
                    "feedback_scope": "initial_vs_final_prism_on_fixed_dev_scenarios_after_main_evaluation",
                    "feedback_enabled": bool(config.get("feedback_diagnostic", {}).get("enabled", True)),
                },
                "upstream_commits": json.loads(upstream_manifest.read_text(encoding="utf-8"))
                                     if upstream_manifest.is_file() else {},
                "statistics_family": {"methods": config["methods"], "backbones": config["backbones"],
                                      "seeds": config["seeds"], "benchmarks": config["benchmarks"]},
                "evaluation_schedule": {"kind": "paired-task-balanced-v1", "train_before_test": True,
                                        "order_seed": int(config.get("evaluation_order_seed", config["split_seed"])),
                                        "gpu_load_condition": "shared-load" if config["scheduler"].get("share_occupied_gpus", False)
                                                              else "exclusive-lane"},
                "legacy_results_used": False}
    if pipeline_request is not None:
        manifest["pipeline_request"] = dict(pipeline_request)
    manifest["runtime"] = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for package in ("torch", "transformers", "peft", "accelerate", "appworld", "numpy", "xgrammar", "apache-tvm-ffi", "triton"):
        try:
            manifest["runtime"]["packages"][package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            manifest["runtime"]["packages"][package] = None
    manifest_path = run_dir / "manifest.json"
    _json(manifest_path, manifest, exclusive=True)
    bundles = []
    backbone_by_id = {backbone["id"]: backbone for backbone in config["backbones"]}
    for work in config["workload_plan"]:
        if work["site_id"] != config["site_id"]:
            continue
        backbone = backbone_by_id[work["backbone"]]
        scoped_methods = _methods_for_backbone(config, backbone["id"])
        seed = int(work["seed"])
        benchmarks = list(work["benchmarks"])
        cells = [{"cell_id": f"{backbone['slug']}.{benchmark}.{method}.s{seed}",
                  "benchmark": benchmark, "method": method, "seed": seed}
                 for benchmark in benchmarks for method in scoped_methods]
        bundles.append({"bundle_id": f"{backbone['slug']}.s{seed}." + "-".join(benchmarks),
                        "site_id": config["site_id"], "benchmarks": benchmarks, "run_dir": str(run_dir),
                        "manifest_path": str(manifest_path), "backbone_id": backbone["id"],
                        "model_path": backbone["model_path"], "seed": seed, "cells": cells,
                        "required_gpu": int(work["gpu"]), "parameters_b": backbone["params_b"],
                        "required_memory_mib": int(backbone["params_b"] * 2300),
                        "priority_weight": backbone["params_b"] * len(cells)})
    _json(run_dir / "schedule.json", bundles, exclusive=True)
    return {"manifest": str(manifest_path), "bundles": len(bundles),
            "cells": sum(len(item["cells"]) for item in bundles), "run_kind": manifest["run_kind"]}


def _episode_meter(config: Mapping[str, Any], path: Path, parent=None) -> BudgetMeter:
    return BudgetMeter(BudgetLimits(int(config["episode_max_steps"]),
                                   int(config["episode_max_model_tokens"]),
                                   float(config["episode_max_wall_seconds"])),
                       parent=parent, event_path=path)


def _save_trace(path: Path, trajectory) -> None:
    from .native_training import public_trace
    _json(path, public_trace(trajectory), exclusive=True)


def _train_cell(policy, config, manifest, cell, directory, initial):
    """Finish the registered training budget and retain its terminal checkpoint."""
    imported = config.get("completed_training_imports", {}).get(cell["cell_id"])
    if imported is not None:
        if cell["method"] not in {"frozen_base", "prism_evolve"}:
            raise ValueError("Terminal training imports are registered only for this initial/final comparison.")
        source = Path(imported).resolve()
        source_manifest = json.loads((source.parent.parent / "manifest.json").read_text())
        bundles = json.loads((source.parent.parent / "schedule.json").read_text())
        matches = [item for bundle in bundles
                   if bundle["backbone_id"] == cell["backbone_id"] and bundle["seed"] == cell["seed"]
                   for item in bundle["cells"] if item == {key: cell[key] for key in
                       ("cell_id", "benchmark", "method", "seed")}]
        if source_manifest["run_kind"] != "formal" or len(matches) != 1:
            raise ValueError("Imported training belongs to another formal cell.")
        source_config = source_manifest["config"]
        for key in ("training_budget", "lora_rank", "lora_alpha", "max_context_tokens",
                    "benchmark_episode_steps", "benchmark_max_new_tokens", "chat_template_kwargs"):
            if source_config[key] != config[key]:
                raise ValueError(f"Imported training has a different registered {key}.")
        if source_manifest["data_splits"][cell["benchmark"]] != manifest["data_splits"][cell["benchmark"]]:
            raise ValueError("Imported training has a different task registration.")
        summary = deepcopy(json.loads((source / "training_phase.json").read_text()))
        if summary["actor_contract"] != _actor_contract(cell["method"], cell["benchmark"]):
            raise ValueError("Imported training has a different actor contract.")
        checkpoint = Path(summary["selected_checkpoint"])
        if not checkpoint.is_dir() or (cell["method"] == "prism_evolve"
                                      and not (source / "training/training_summary.json").is_file()):
            raise ValueError("Imported training has no complete terminal checkpoint.")
        summary["training_reuse"] = {"source_cell": str(source),
                                     "source_manifest_id": source_manifest["manifest_id"],
                                     "source_training_phase": str(source / "training_phase.json"),
                                     "new_training_model_calls": 0,
                                     "cost_scope": "source_training_usage_inherited_once; no optimizer continuation"}
        _json(directory / "training_phase.json", summary, exclusive=True)
        return summary
    from .native_training import run_training
    method, benchmark, seed = cell["method"], cell["benchmark"], int(cell["seed"])
    local_config = dict(_benchmark_config(config, benchmark), benchmark=benchmark, backbone=cell["backbone_id"],
                        output_dir=str(directory), model_id=cell["backbone_id"])
    _set_benchmark_policy(policy, local_config)
    env = _environment(local_config, benchmark, directory)
    try:
        train = _registered_tasks(env, manifest, benchmark, "train")
        dev = _registered_tasks(env, manifest, benchmark, "dev")
        summary: dict[str, Any] = {"method": method, "run_kind": manifest["run_kind"],
                                   "actor_contract": _actor_contract(method, benchmark),
                                   "episode_max_steps": local_config["episode_max_steps"],
                                   "max_new_tokens": local_config["max_new_tokens"]}
        checkpoint = str(initial)
        if method in INFERENCE_METHODS or method == "frozen_base":
            summary["parameter_training"] = {"status": "not_applicable", "parameter_updates": False}
        memory_initialization = config.get("worldevolver", {}).get("memory_initialization", "empty")
        if method == "worldevolver" and memory_initialization not in {"empty", "train_prequential"}:
            raise ValueError("WorldEvolver memory_initialization must be empty or train_prequential.")
        if method == "worldevolver" and memory_initialization == "train_prequential":
            from .native_memory_training import warm_start_worldevolver
            training_meter = BudgetMeter(BudgetLimits(**config["training_budget"]),
                                         event_path=directory / "training_costs.jsonl")
            summary["training"] = warm_start_worldevolver(
                policy, env, train, local_config, training_meter, directory / "training", seed)
            summary["memory_checkpoint"] = summary["training"]["memory_checkpoint"]
            summary["memory_initialization"] = "train_prequential"
        elif method == "worldevolver":
            summary["memory_initialization"] = "empty"
            summary["memory_checkpoint"] = None
        elif method not in INFERENCE_METHODS and method != "frozen_base":
            training_meter = BudgetMeter(BudgetLimits(**config["training_budget"]),
                                         event_path=directory / "training_costs.jsonl")
            summary["training"] = run_training(method, policy, env, train, dev, local_config,
                                                training_meter, directory / "training", seed)
            checkpoint = summary["training"].get("selected_checkpoint")
            if (not summary["training"].get("selection_complete", True)
                    or not checkpoint or checkpoint == "None"):
                raise RuntimeError("The training budget did not produce a terminal checkpoint.")
        summary.update(selected_checkpoint=str(checkpoint), model=policy.model_metadata,
                       split_manifest=env.split_manifest, training_finished_at=_now())
        if method in INFERENCE_METHODS:
            summary["method_identity"] = {
                "implementation": "official-controller-native-environment-transfer",
                "environment_interface": "unrestricted_native_actions",
                "trial_protocol": "success_at_1",
                "parameter_updates": False,
                "paper_protocol_reproduction": False,
            }
        if method == "worldevolver":
            summary["method_identity"].update(
                memory_initialization=memory_initialization,
                memory_update="prequential_with_split_isolation",
                sf_threshold_kind="absolute_probability_percent",
                sf_confidence_pct=float(config.get("worldevolver", {}).get("sf_confidence_pct", 50.0)),
            )
        _json(directory / "training_phase.json", summary, exclusive=True)
        return summary
    finally:
        env.close()


def _release_work_buffers(policy) -> None:
    policy.meter = None
    policy.zero_grad(set_to_none=True)
    gc.collect()
    # Retain the resident model's allocator cache across paired episodes.
    # The common per-process memory fraction bounds its reservation.


def _evaluate_one(policy, env, task, config, state, split, task_index, position, gpu, memory):
    """Evaluate one task using the benchmark owner's reusable environment."""
    import torch
    from .native_baselines import run_inference_method
    from .native_training import rollout
    cell, directory = state["cell"], state["directory"]
    method, seed = cell["method"], int(cell["seed"])
    output = directory / "evaluation" / split
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "tasks" / f"task-{task_index:05d}.json"
    trace_path = output / "traces" / f"task-{task_index:05d}.json"
    if record_path.exists() or trace_path.exists():
        raise FileExistsError("A paired task already has evidence; incomplete evaluation is not silently rerun.")
    imports = config.get("completed_training_imports", {})
    if config.get("reuse_completed_paired_evaluations", False):
        # Reuse every complete pair, regardless of its score; incomplete pairs run afresh.
        source_cells = [Path(path) for key, path in imports.items()
                        if key.startswith(cell["cell_id"].split(".", 1)[0] + "." + cell["benchmark"] + ".")]
        relative = Path("evaluation") / split
        packet = Path("tasks") / record_path.name
        trace = Path("traces") / trace_path.name
        if source_cells and all((path / relative / packet).is_file()
                                and (path / relative / trace).is_file() for path in source_cells):
            source = Path(imports[cell["cell_id"]])
            result = json.loads((source / relative / packet).read_text())
            expected = {"task_id": str(task["task_id"]), "method": method, "seed": seed,
                        "backbone": cell["backbone_id"], "paired_task_index": task_index,
                        "actor_seed": seed * 1000003 + task_index,
                        "paired_method_position": position, "physical_gpu": gpu}
            if any(result.get(key) != value for key, value in expected.items()):
                raise ValueError("A completed paired evaluation differs from the current registered task/order/GPU.")
            provenance = {"source_record": str(source / relative / packet),
                          "source_trace": str(source / relative / trace),
                          "new_model_calls": 0, "cost_scope": "original episode usage retained once"}
            result["evaluation_reuse"] = provenance
            original_trace = json.loads((source / relative / trace).read_text())
            original_trace["evaluation_reuse"] = provenance
            _json(trace_path, original_trace, exclusive=True)
            _json(record_path, result, exclusive=True)
            _append_json(output / "records.jsonl", result)
            return result
    task = dict(task)
    actor_seed = seed * 1000003 + task_index
    task.setdefault("environment_seed", actor_seed)
    evaluation_config = dict(_benchmark_config(config, cell["benchmark"]), benchmark=f"{cell['benchmark']}:{split}",
                             backbone=cell["backbone_id"], model_id=cell["backbone_id"],
                             split=split, output_dir=str(output))
    _set_benchmark_policy(policy, evaluation_config)
    switch_start = monotonic_seconds()
    policy.load_adapter(state["summary"]["selected_checkpoint"])
    policy.zero_grad(set_to_none=True)
    adapter_switch_seconds = monotonic_seconds() - switch_start
    torch.manual_seed(actor_seed)
    meter = _episode_meter(evaluation_config, output / "costs" / f"task-{task_index:05d}.jsonl")
    started_at = _now()
    try:
        if method in INFERENCE_METHODS:
            trajectory = run_inference_method(method, policy, env, task, seed, evaluation_config, meter,
                                               output / "official", memory_state=memory)
        else:
            trajectory = rollout(policy, env, task, evaluation_config, meter,
                                 seed=task["environment_seed"], method=method, training=False)
        ended_at = _now()
        trajectory.metadata.update(host=socket.gethostname(), site_id=config["site_id"], physical_gpu=gpu,
                                   real_started_at=started_at, real_ended_at=ended_at)
        _save_trace(trace_path, trajectory)
        if trajectory.termination == "method_error":
            raise RuntimeError(f"Complete {method} controller failed: {trajectory.metadata.get('method_error')}")
        result = trajectory.result(benchmark=evaluation_config["benchmark"], backbone=cell["backbone_id"],
                                    seed=seed, method=method)
        result.update(split=split, environment_seed=task["environment_seed"], actor_seed=actor_seed,
                      actor_contract=_actor_contract(method, cell["benchmark"]),
                      episode_max_steps=evaluation_config["episode_max_steps"],
                      max_new_tokens=evaluation_config["max_new_tokens"],
                      program_outcomes=trajectory.metadata.get("program_outcomes"),
                      host=socket.gethostname(), site_id=config["site_id"], physical_gpu=gpu,
                      real_started_at=started_at, real_ended_at=ended_at,
                      paired_task_index=task_index, paired_method_position=position,
                      adapter_switch_seconds=adapter_switch_seconds,
                      gpu_load_condition=state["receipt"]["gpu_load_condition"],
                      resource_usage=meter.snapshot())
        _json(record_path, result, exclusive=True)
        _append_json(output / "records.jsonl", result)
        return result
    except Exception as error:
        _json(output / "errors" / f"task-{task_index:05d}.json",
              {"task_id": task["task_id"], "environment_seed": task["environment_seed"],
               "physical_gpu": gpu, "real_started_at": started_at, "real_ended_at": _now(),
               "error": f"{type(error).__name__}: {error}", "resource_usage": meter.snapshot()},
              exclusive=True)
        raise


def _fail_cell(state, error, *, phase: str) -> None:
    state["failed"] = True
    detail = dict(state["receipt"], status="failed", phase=phase, ended_at=_now(),
                  wall_seconds=monotonic_seconds() - state["started"],
                  active_work_seconds=state["active_seconds"],
                  error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    _json(state["directory"] / "status.json", detail)


def _run_feedback_change(policy, config, bundle, states, initial, gpu, identity) -> bool:
    """Measure the registered initial/final policy contrast after main evaluation."""
    if (not config.get("feedback_diagnostic", {}).get("enabled", True)
            or bundle["backbone_id"] != config["mechanism_backbone"]):
        return False
    from .native_development_metrics import evaluate_feedback_change
    failed = False
    for benchmark in bundle["benchmarks"]:
        directory = Path(bundle["run_dir"]) / "feedback" / bundle["bundle_id"] / benchmark
        receipt = {"manifest_id": identity, "bundle_id": bundle["bundle_id"],
                   "backbone": bundle["backbone_id"], "benchmark": benchmark, "seed": int(bundle["seed"]),
                   "host": socket.gethostname(), "site_id": config["site_id"],
                   "physical_gpu": gpu, "started_at": _now(),
                   "scope": "initial_vs_final_prism_fixed_dev_scenarios"}
        _json(directory / "status.json", dict(receipt, status="running"), exclusive=True)
        env = None
        try:
            state = next(state for state in states if state["cell"]["benchmark"] == benchmark
                         and state["cell"]["method"] == "prism_evolve")
            checkpoint = state["summary"]["selected_checkpoint"]
            local_config = dict(_benchmark_config(config, benchmark), benchmark=benchmark,
                                backbone=bundle["backbone_id"], model_id=bundle["backbone_id"],
                                output_dir=str(directory))
            _set_benchmark_policy(policy, local_config)
            env = _environment(local_config, benchmark, directory)
            meter = BudgetMeter(BudgetLimits(**config["formal_feedback_budget"]),
                                event_path=directory / "costs.jsonl")
            summary = evaluate_feedback_change(
                policy, env, env.list_tasks("dev"), local_config, meter, directory / "measurement",
                initial, checkpoint, int(bundle["seed"]), bundle["backbone_id"],
            )
            _json(directory / "summary.json", summary, exclusive=True)
            _json(directory / "status.json", dict(receipt,
                  status="done" if summary.get("complete", False) else "partial", ended_at=_now(),
                  resource_usage=meter.snapshot(), summary=str(directory / "summary.json")))
        except Exception as error:
            failed = True
            _json(directory / "status.json", dict(receipt, status="failed", ended_at=_now(),
                  error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
        finally:
            if env is not None:
                env.close()
            _release_work_buffers(policy)
    return failed


def _run_resident_cells(policy, config, manifest, bundle, gpu, initial, identity) -> bool:
    """Train all fresh cells, then evaluate complete paired task blocks."""
    if manifest["run_kind"] != "formal":
        raise ValueError("Only complete formal experiment manifests may execute.")
    run_dir = Path(bundle["run_dir"])
    states: list[dict[str, Any]] = []
    failed = False
    condition = "shared-load" if config["scheduler"].get("share_occupied_gpus", False) else "exclusive-lane"
    admission = json.loads(os.environ.get("PRISM_NATIVE_GPU_ADMISSION_JSON", "{}"))
    training_order = sorted(bundle["cells"], key=lambda cell: (
        cell["method"] == "frozen_base" or cell["method"] in INFERENCE_METHODS - {"worldevolver"},
        cell["method"] != "prism_evolve",
        config["benchmarks"].index(cell["benchmark"]),
        config["methods"].index(cell["method"]),
    ))
    for original in training_order:
        cell = dict(original, backbone_id=bundle["backbone_id"])
        directory = run_dir / "cells" / cell["cell_id"]
        status_path = directory / "status.json"
        receipt = {"cell_id": cell["cell_id"], "manifest_id": identity,
                   "host": socket.gethostname(), "site_id": config["site_id"],
                   "physical_gpu": gpu, "started_at": _now(),
                   "gpu_load_condition": condition, "gpu_admission": admission,
                   "evaluation_schedule": "paired-task-balanced-v1"}
        if status_path.exists():
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if existing.get("manifest_id") != identity:
                raise ValueError("Cell receipt belongs to another immutable manifest.")
            if existing.get("status") != "done":
                failed = True
            continue
        state = {"cell": cell, "directory": directory, "receipt": receipt,
                 "started": monotonic_seconds(), "active_seconds": 0.0,
                 "records": [], "failed": False, "expected_evaluations": 0,
                 "registered_tasks_by_split": {}}
        states.append(state)
        _json(status_path, dict(receipt, status="running", phase="training"))
        started = monotonic_seconds()
        try:
            policy.load_adapter(initial)
            policy.zero_grad(set_to_none=True)
            state["summary"] = _train_cell(policy, config, manifest, cell, directory, initial)
            state["active_seconds"] += monotonic_seconds() - started
            _json(status_path, dict(receipt, status="running", phase="training_complete",
                                   selected_checkpoint=state["summary"]["selected_checkpoint"],
                                   training_finished_at=_now()))
        except Exception as error:
            state["active_seconds"] += monotonic_seconds() - started
            _fail_cell(state, error, phase="training")
            failed = True
        finally:
            _release_work_buffers(policy)

    # The final task blocks use the complete set of terminal policies.
    for benchmark in bundle["benchmarks"]:
        benchmark_index = config["benchmarks"].index(benchmark)
        benchmark_states = [state for state in states if state["cell"]["benchmark"] == benchmark]
        if not any(not state["failed"] for state in benchmark_states):
            continue
        methods = [cell["method"] for cell in bundle["cells"] if cell["benchmark"] == benchmark]
        lookup = {state["cell"]["method"]: state for state in benchmark_states}
        env = None
        splits = list(config["appworld_test_splits"]) if benchmark == "appworld" else ["test"]
        try:
            env = _environment(config, benchmark, run_dir / "paired_environment" / bundle["bundle_id"] / benchmark)
            for split_index, split in enumerate(splits):
                tasks = _registered_tasks(env, manifest, benchmark, split)
                if not tasks or len({task["task_id"] for task in tasks}) != len(tasks):
                    raise ValueError("Each evaluation split must contain nonempty, unique task IDs.")
                for state in benchmark_states:
                    state["expected_evaluations"] += len(tasks)
                    state["registered_tasks_by_split"][split] = tasks
                order_seed = int(config.get("evaluation_order_seed", config["split_seed"])) + int(bundle["seed"])
                order_seed += benchmark_index * 1000003 + split_index * 1009
                memories = {method: {} for method in methods}
                memory_state = lookup.get("worldevolver")
                if (memory_state is not None and not memory_state["failed"]
                        and memory_state["summary"].get("memory_checkpoint") is not None):
                    memory_path = Path(memory_state["summary"]["memory_checkpoint"])
                    memories["worldevolver"]["checkpoint"] = json.loads(memory_path.read_text(encoding="utf-8"))
                try:
                    for task_index, task in enumerate(tasks):
                        planned = _balanced_method_order(methods, order_seed, task_index)
                        block = {"benchmark": benchmark, "split": split, "task_id": task["task_id"],
                                 "task_index": task_index, "seed": int(bundle["seed"]),
                                 "environment_seed": task.get("environment_seed", int(bundle["seed"]) * 1000003 + task_index),
                                 "backbone": bundle["backbone_id"], "physical_gpu": gpu,
                                 "order_seed": order_seed, "planned_method_order": planned,
                                 "gpu_load_condition": condition, "real_started_at": _now(),
                                 "load_before": _gpu_load(gpu), "completed_methods": [], "failed_methods": []}
                        for position, method in enumerate(planned):
                            state = lookup.get(method)
                            if state is None or state["failed"]:
                                continue
                            started = monotonic_seconds()
                            _json(state["directory"] / "status.json", dict(state["receipt"], status="running",
                                  phase="paired_evaluation", split=split, task_index=task_index,
                                  completed_evaluations=len(state["records"])))
                            try:
                                result = _evaluate_one(policy, env, task, config, state, split, task_index,
                                                       position, gpu, memories[method])
                                state["records"].append(result)
                                block["completed_methods"].append(method)
                                if "evaluation_reuse" in result:
                                    block.setdefault("reused_methods", []).append(method)
                            except Exception as error:
                                state["active_seconds"] += monotonic_seconds() - started
                                _fail_cell(state, error, phase="paired_evaluation")
                                block["failed_methods"].append(method)
                                failed = True
                            finally:
                                if not state["failed"]:
                                    state["active_seconds"] += monotonic_seconds() - started
                                _release_work_buffers(policy)
                        block.update(real_ended_at=_now(), load_after=_gpu_load(gpu))
                        _append_json(run_dir / "paired_order" / f"{bundle['bundle_id']}.jsonl", block)
                finally:
                    for memory in memories.values():
                        if "session" in memory:
                            memory["session"].close()
        except Exception as error:
            failed = True
            for state in benchmark_states:
                if not state["failed"]:
                    _fail_cell(state, error, phase="paired_evaluation")
        finally:
            if env is not None:
                env.close()

    for state in states:
        if state["failed"]:
            continue
        if not state["expected_evaluations"] or len(state["records"]) != state["expected_evaluations"]:
            _fail_cell(state, RuntimeError("The complete registered evaluation grid was not produced."),
                       phase="paired_evaluation")
            failed = True
            continue
        directory = state["directory"]
        _json(directory / "test_records.json", state["records"], exclusive=True)
        summary = state["summary"]
        from .statistics import official_benchmark_summary
        summary["official_metrics_by_split"] = {
            split: official_benchmark_summary(
                [record for record in state["records"] if record["split"] == split],
                expected_tasks=tasks,
            )
            for split, tasks in state["registered_tasks_by_split"].items()
        }
        summary.update(test_records=str(directory / "test_records.json"), num_evaluations=len(state["records"]),
                       evaluation_schedule="paired-task-balanced-v1", gpu_load_condition=condition,
                       active_work_seconds=state["active_seconds"])
        _json(directory / "summary.json", summary, exclusive=True)
        _json(directory / "status.json", dict(state["receipt"], status="done", phase="complete", ended_at=_now(),
              wall_seconds=monotonic_seconds() - state["started"], active_work_seconds=state["active_seconds"],
              num_evaluations=len(state["records"])))
    feedback_failed = _run_feedback_change(policy, config, bundle, states, initial, gpu, identity)
    return failed or feedback_failed


def worker_bundle(path: Path, gpu: int) -> int:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = Path(bundle["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    _host(config)
    if manifest["run_kind"] != "formal":
        raise ValueError("Historical validation runs cannot launch model workers.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("Worker GPU isolation must be supplied by the scheduler.")
    if config.get("coordination", {}).get("enabled"):
        from .native_gpu_quota import worker_start
        from .native_scheduler import nvidia_inventory
        if "execution" not in manifest:
            raise ValueError("Coordinated workers require an independently authorized execution shard.")
        actual_gpu = next(item for item in nvidia_inventory() if item.index == gpu)
        worker_start(manifest, manifest["manifest_id"], actual_gpu.uuid)
    import torch
    from .native_policy import NativeCausalPolicy
    torch.set_num_threads(int(os.environ.get("PRISM_NATIVE_TORCH_THREADS", "4")))
    torch.manual_seed(int(bundle["seed"]))
    torch.cuda.manual_seed_all(int(bundle["seed"]))
    run_dir = Path(bundle["run_dir"])
    identity = manifest["manifest_id"]
    try:
        memory_fraction = float(os.environ.get("PRISM_NATIVE_GPU_MEMORY_FRACTION", "0.85"))
        allocator_cap_mib = manifest.get("execution", {}).get("quota", {}).get("allocator_mib")
        if allocator_cap_mib is not None:
            memory_fraction = allocator_cap_mib * 1048576 / torch.cuda.get_device_properties(0).total_memory
        if not 0 < memory_fraction <= 1:
            raise ValueError("The per-worker GPU memory fraction must lie in (0, 1].")
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
        policy = NativeCausalPolicy(bundle["model_path"], device="cuda:0",
                                    lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"],
                                    max_new_tokens=config["max_new_tokens"],
                                    max_context_tokens=config["max_context_tokens"],
                                    chat_template_kwargs=config.get("chat_template_kwargs", {}),
                                    activation_offload_cpu=bool(config.get("activation_offload_cpu", True)),
                                    activation_offload_min_bytes=int(config.get("activation_offload_min_bytes", 1048576)))
    except Exception as error:
        detail = {"status": "failed", "manifest_id": identity, "host": socket.gethostname(),
                  "site_id": config["site_id"],
                  "physical_gpu": gpu, "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "model_calls": 0, "at": _now()}
        _json(run_dir / "model_runtime" / f"{bundle['bundle_id']}.error.json", detail, exclusive=True)
        for cell in bundle["cells"]:
            path = run_dir / "cells" / cell["cell_id"] / "status.json"
            if not path.exists():
                _json(path, dict(detail, cell_id=cell["cell_id"]))
        return 1
    initial = run_dir / "initial_adapters" / bundle["bundle_id"]
    if not initial.exists():
        policy.save_adapter(initial)
    _json(run_dir / "model_runtime" / f"{bundle['bundle_id']}.json",
          dict(policy.model_metadata, physical_gpu=gpu, gpu_memory_fraction=memory_fraction,
               gpu_allocator_cap_mib=allocator_cap_mib,
               gpu_admission=json.loads(os.environ.get("PRISM_NATIVE_GPU_ADMISSION_JSON", "{}"))))
    return int(_run_resident_cells(policy, config, manifest, bundle, gpu, initial, identity))


def aggregate(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    schedule = json.loads((run_dir / "schedule.json").read_text(encoding="utf-8"))
    if manifest["config"].get("coordination", {}).get("enabled") and "execution" not in manifest:
        scheduler = json.loads((run_dir / "scheduler_state.json").read_text())
        executions = [aggregate(path.parent) for path in sorted((run_dir / "executions").glob("*/manifest.json"))]
        completeness = {"role": "registration_and_dispatch", "run_kind": manifest["run_kind"],
                        "registered_origin_cells": sum(len(bundle["cells"]) for bundle in schedule),
                        "local_execution_shards": executions,
                        "global_claims_complete": scheduler.get("coordination", {}).get("global", {}).get("global_complete", False),
                        "formal_inference_ready": False, "requires_global_collection": True}
        _json(run_dir / "completeness.json", completeness)
        return completeness
    pending = []
    identity = manifest["manifest_id"]
    for bundle in schedule:
        for cell in bundle["cells"]:
            directory = run_dir / "cells" / cell["cell_id"]
            receipt = directory / "status.json"
            if not receipt.exists():
                pending.append(cell["cell_id"])
                continue
            status = json.loads(receipt.read_text(encoding="utf-8"))
            if status.get("cell_id") != cell["cell_id"] or status.get("manifest_id") != identity:
                raise ValueError(f"Aggregation receipt belongs to another immutable cell: {receipt}")
            if status.get("status") != "done":
                pending.append(cell["cell_id"])
                continue
    completeness = {"run_kind": manifest["run_kind"], "pending_cells": pending,
                    "completed_cells": sum(len(bundle["cells"]) for bundle in schedule) - len(pending),
                    "formal_inference_ready": not pending and manifest["run_kind"] == "formal"}
    feedback = {}
    for bundle in schedule:
        if (manifest["config"].get("feedback_diagnostic", {}).get("enabled", True)
                and bundle["backbone_id"] == manifest["config"]["mechanism_backbone"]):
            for benchmark in bundle["benchmarks"]:
                path = run_dir / "feedback" / bundle["bundle_id"] / benchmark / "status.json"
                feedback[f"{bundle['bundle_id']}/{benchmark}"] = (
                    json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "pending"})
    completeness["feedback_measurements"] = feedback
    completeness["global_statistics_deferred"] = bool(manifest["config"].get("defer_global_statistics", False))
    _json(run_dir / "completeness.json", completeness)
    return completeness


def schedule_run(run_dir: Path) -> dict[str, Any]:
    from .native_scheduler import run_schedule
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    _host(config)
    if manifest["run_kind"] != "formal":
        raise ValueError("Historical validation runs cannot be scheduled again.")
    # Every worker inherits the exact source snapshot. The orchestration process
    # can continue from the user's checkout without loading a model itself.
    os.environ["PYTHONPATH"] = str(Path(manifest["source_root"]) / "src")
    bundles = json.loads((run_dir / "schedule.json").read_text(encoding="utf-8"))
    options = config["scheduler"]
    result = run_schedule(bundles, run_dir, config["python_executable"],
                          allowed_gpu_ids=options["allowed_gpu_ids"], max_workers=options["max_workers"],
                          poll_seconds=options["poll_seconds"], cpu_threads=options["cpu_threads_per_lane"],
                          expected_hostname=config["execution_host"],
                          share_occupied_gpus=bool(options.get("share_occupied_gpus", False)),
                          memory_fraction=float(options.get("memory_fraction", 0.85)),
                          coordination=config.get("coordination"),
                          max_workers_per_gpu=int(options.get("max_workers_per_gpu", 1)),
                          worker_memory_mib=options.get("worker_memory_mib"),
                          allocator_memory_mib=options.get("allocator_memory_mib"),
                          safety_margin_mib=int(options.get("safety_margin_mib", 8192)))
    return {"scheduler": result, "completeness": aggregate(run_dir)}


def run_pipeline(config_path: Path, run_dir: Path, *,
                 selected_backbones: list[str] | None = None,
                 selected_methods: list[str] | None = None) -> dict[str, Any]:
    """Freeze the completed implementation and submit the unified formal queue."""
    config_path, run_dir = config_path.resolve(), run_dir.resolve()
    config = _load_config(config_path)
    _protect_legacy_outputs(config, run_dir)
    request = {"config_path": str(config_path),
               "selected_backbones": selected_backbones, "selected_methods": selected_methods}
    state_path = run_dir / "pipeline_state.json"
    manifest_path = run_dir / "manifest.json"
    if state_path.exists() or manifest_path.exists() or (run_dir / "schedule.json").exists():
        raise FileExistsError("Formal pipelines require a new run directory; previous runs remain immutable.")
    state = {"protocol_version": PROTOCOL, "request": request,
             "host": socket.gethostname(), "site_id": config["site_id"],
             "started_at": _now(), "status": "preparing_formal"}
    _json(state_path, state, exclusive=True)
    from .native_gpu_quota import process_identity
    _json(run_dir / "monitor-runtime.json", {"host": socket.gethostname(),
                                           "run_directory": str(run_dir), "pipeline_pid": os.getpid(),
                                           "pipeline_start": process_identity(os.getpid())}, exclusive=True)
    prepare_run(config_path, run_dir, selected_backbones=selected_backbones,
                selected_methods=selected_methods, pipeline_request=request)
    state.update(status="formal_running", formal_manifest=str(manifest_path),
                 manifest_id=f"{config['site_id']}/{run_dir.name}", updated_at=_now())
    _json(state_path, state)
    result = schedule_run(run_dir)
    state.update(status=result["scheduler"]["status"], result=result, updated_at=_now())
    _json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Native-v1 remote host shards; legacy results stay immutable.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/prism-native-v1.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--backbones", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--worker-bundle", type=Path)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.worker_bundle is not None:
        if args.gpu is None:
            parser.error("--worker-bundle requires --gpu")
        raise SystemExit(worker_bundle(args.worker_bundle, args.gpu))
    if args.run_dir is None:
        parser.error("Supply a new --run-dir (or the existing immutable run for --schedule/--aggregate).")
    if args.pipeline:
        result = run_pipeline(args.config, args.run_dir,
                              selected_backbones=args.backbones, selected_methods=args.methods)
    elif args.prepare:
        result = prepare_run(args.config, args.run_dir,
                             selected_backbones=args.backbones, selected_methods=args.methods)
    elif args.aggregate:
        result = aggregate(args.run_dir.resolve())
    elif args.schedule:
        result = schedule_run(args.run_dir.resolve())
    else:
        parser.error("Choose --prepare, --schedule, --aggregate, or --pipeline")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
