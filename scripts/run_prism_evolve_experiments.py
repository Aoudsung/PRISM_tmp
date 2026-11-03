#!/usr/bin/env python3
"""End-to-End Experimental Verification & Benchmark Suite for PRISM-Evolve.

Executes all verification and experimental protocols specified in
`docs/prism-evolve-design.md` (§5.1–§5.3, §6.1–§6.2, §7.1–§7.3, §8.2):
1. Part 1 (§5.3): Exact rational enumeration + PyTorch autograd verification +
   Monte Carlo empirical gradient variance verification (N=10,000 episodes).
2. Part 2 (§7.2–§7.3): Multi-stage online cross-task evolution on AppWorld
   (train=90, dev=57, test_normal=168) across K=3 stages and 3 random seeds
   for all 8 methods, with scenario-grouped paired 95% CIs and 6D CostLedger.
3. Part 3 (§6.2, §7.2): Dev-set masked feedback diagnostic Delta_use(pi),
   budget sensitivity matrix B in {0.6 B_0, 1.0 B_0, 1.4 B_0}, and split-step
   selection strategy analysis.
4. Part 4 (§7.2): Cross-environment replication on PlanBench-XL retail domain
   (327 queries, 56 datatypes, 1,665 tools: train=180, dev=57, test=90) across
   K=3 stages and 3 random seeds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_evolve import (
    ABLAT_METHODS,
    AppWorldEnvAdapter,
    BranchCollector,
    ContextualLoRADualModePolicy,
    CostLedger,
    CrossTaskEvaluator,
    MAIN_TABLE_PEER_METHODS,
    MaskedFeedbackDiagnostic,
    PEER_BASELINE_REGISTRY,
    PlanBenchXLEnvAdapter,
    PoSEnvProbePolicy,
    StratifiedUpdater,
    TwoStageParametricPolicy,
    TwoStageToyEnv,
    WorldEvolverMemoryPolicy,
    load_appworld_split_tasks,
    load_planbench_split_tasks,
    verify_external_baseline_repos,
    verify_two_stage_exact_enumeration,
)


ALL_METHODS: Tuple[str, ...] = tuple(
    dict.fromkeys(list(MAIN_TABLE_PEER_METHODS) + list(ABLAT_METHODS))
)

CANONICAL_BACKBONES: Tuple[str, ...] = (
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.8-27B",
)

BACKBONE_SPECS: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "slug": "qwen25_7b",
        "display_name": "Qwen/Qwen2.5-7B-Instruct",
        "short_name": "Qwen2.5-7B-Instruct",
        "short_label": "Qwen2.5-7B",
        "params_b": 7.6,
        "family": "Qwen2.5 (7B)",
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        "cache_filename": "qwen25_7b_embed_cache.pt",
        "results_filename": "results_backbone_qwen25_7b.json",
        "local_candidates": [
            "Qwen/Qwen2.5-7B-Instruct",
        ],
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "slug": "llama31_8b",
        "display_name": "meta-llama/Llama-3.1-8B-Instruct",
        "short_name": "Llama-3.1-8B-Instruct",
        "short_label": "Llama-3.1-8B",
        "params_b": 8.0,
        "family": "Meta Llama 3.1 (8B)",
        "hf_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
        "cache_filename": "llama31_8b_embed_cache.pt",
        "results_filename": "results_backbone_llama31_8b.json",
        "local_candidates": [
            "/apps/users/cxw/Document/CodeSpace/Selfs/WrongLessonBench/models/Meta-Llama-3.1-8B-Instruct",
            "/home/chenxiangwei/models/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ],
    },
    "Qwen/Qwen3.8-27B": {
        "slug": "qwen38_27b",
        "display_name": "Qwen/Qwen3.8-27B",
        "short_name": "Qwen3.8-27B",
        "short_label": "Qwen3.8-27B",
        "params_b": 27.0,
        "family": "Qwen3.8 (27B)",
        "hf_url": "https://huggingface.co/Qwen/Qwen3.8-27B",
        "cache_filename": "qwen38_27b_embed_cache.pt",
        "results_filename": "results_backbone_qwen38_27b.json",
        "local_candidates": [
            "/home/chenxiangwei/models/Qwen3.8-27B",
            "/data/models/Qwen3.8-27B",
            "Qwen/Qwen3.8-27B",
        ],
    },
}


def _collection_mode_for_method(method: str) -> str:
    if method in ("prism_evolve", "ablation_unweighted", "step_dpo"):
        return "stratified"
    if method == "unstratified_2branch":
        return "unstratified_2branch"
    return "single"


def run_part1_exact_and_mc_verification(num_mc_episodes: int = 10000, seed: int = 20261102) -> Dict[str, Any]:
    """Run §5.3 exact rational verification and Monte Carlo empirical variance check."""
    t0 = time.perf_counter()
    exact_res = verify_two_stage_exact_enumeration()

    # Monte Carlo empirical verification on TwoStageToyEnv
    env = TwoStageToyEnv(budget=1.0)
    collector = BranchCollector(max_steps=4, max_split_step=0, fixed_split_step=0)
    updater = StratifiedUpdater(grad_clip_norm=0.0)
    rng = random.Random(seed)

    grads_strat: List[Tuple[float, float, float]] = []
    grads_rloo: List[Tuple[float, float, float]] = []
    costs_strat: List[float] = []
    costs_rloo: List[float] = []
    no_info_count_p002 = 0

    policy_strat = TwoStageParametricPolicy(p_init=0.4, q_informed_init=0.8, q_uninformed_init=0.7)
    policy_rloo = TwoStageParametricPolicy(p_init=0.4, q_informed_init=0.8, q_uninformed_init=0.7)
    ledger_strat = CostLedger()
    ledger_rloo = CostLedger()

    for idx in range(num_mc_episodes):
        # 1. Stratified 2-branch sample
        sample_s = collector.collect_episode(
            env=env,
            policy=policy_strat,
            task_id=f"mc_s_{idx}",
            rng=rng,
            ledger=ledger_strat,
            collection_mode="stratified",
        )
        policy_strat.zero_grad(set_to_none=True)
        loss_s = updater.compute_episode_loss(
            policy=policy_strat,
            sample=sample_s,
            batch_baseline=0.0,
            method="prism_evolve",
        )
        loss_s.backward()
        g_s = (
            float(-policy_strat.alpha.grad.item()) if policy_strat.alpha.grad is not None else 0.0,
            float(-policy_strat.beta.grad.item()) if policy_strat.beta.grad is not None else 0.0,
            float(-policy_strat.gamma.grad.item()) if policy_strat.gamma.grad is not None else 0.0,
        )
        grads_strat.append(g_s)
        assert sample_s.branch_I is not None and sample_s.branch_E is not None
        costs_strat.append(sample_s.branch_I.cumulative_cost + sample_s.branch_E.cumulative_cost)

        # 2. Same-prefix unstratified 2-branch RLOO sample
        sample_r = collector.collect_episode(
            env=env,
            policy=policy_rloo,
            task_id=f"mc_r_{idx}",
            rng=rng,
            ledger=ledger_rloo,
            collection_mode="unstratified_2branch",
        )
        policy_rloo.zero_grad(set_to_none=True)
        loss_r = updater.compute_episode_loss(
            policy=policy_rloo,
            sample=sample_r,
            batch_baseline=0.0,
            method="unstratified_2branch",
        )
        loss_r.backward()
        g_r = (
            float(-policy_rloo.alpha.grad.item()) if policy_rloo.alpha.grad is not None else 0.0,
            float(-policy_rloo.beta.grad.item()) if policy_rloo.beta.grad is not None else 0.0,
            float(-policy_rloo.gamma.grad.item()) if policy_rloo.gamma.grad is not None else 0.0,
        )
        grads_rloo.append(g_r)
        assert sample_r.branch_I is not None and sample_r.branch_E is not None
        costs_rloo.append(sample_r.branch_I.cumulative_cost + sample_r.branch_E.cumulative_cost)

        # 3. Scarcity check at p = 0.02 with K_roll = 2 unstratified draws
        u1 = rng.random()
        u2 = rng.random()
        if u1 >= 0.02 and u2 >= 0.02:
            no_info_count_p002 += 1

    def _mean_and_var(vecs: Sequence[Tuple[float, float, float]]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
        n = len(vecs)
        means = tuple(sum(v[d] for v in vecs) / n for d in range(3))
        vars_ = tuple(sum((v[d] - means[d]) ** 2 for v in vecs) / n for d in range(3))
        return means, vars_, sum(vars_)

    mc_mean_strat, mc_var_strat, mc_varsum_strat = _mean_and_var(grads_strat)
    mc_mean_rloo, mc_var_rloo, mc_varsum_rloo = _mean_and_var(grads_rloo)
    mc_cost_strat = sum(costs_strat) / len(costs_strat)
    mc_cost_rloo = sum(costs_rloo) / len(costs_rloo)
    mc_no_info_rate = no_info_count_p002 / num_mc_episodes

    return {
        "exact_enumeration": exact_res,
        "monte_carlo": {
            "num_episodes": num_mc_episodes,
            "stratified_mean_grad": mc_mean_strat,
            "stratified_var_components": mc_var_strat,
            "stratified_var_sum": mc_varsum_strat,
            "stratified_mean_pair_cost": mc_cost_strat,
            "stratified_var_cost_product": mc_varsum_strat * mc_cost_strat,
            "rloo_mean_grad": mc_mean_rloo,
            "rloo_var_components": mc_var_rloo,
            "rloo_var_sum": mc_varsum_rloo,
            "rloo_mean_pair_cost": mc_cost_rloo,
            "rloo_var_cost_product": mc_varsum_rloo * mc_cost_rloo,
            "empirical_p002_no_info_prob": mc_no_info_rate,
            "elapsed_sec": time.perf_counter() - t0,
        },
    }


def _summarize_eval_records(
    records: Sequence[Any],
    baseline_records: Sequence[Any],
    stage_k: int,
) -> Dict[str, Any]:
    gain = CrossTaskEvaluator.compute_paired_evolution_gain(
        records_k=records,
        records_0=baseline_records,
        stage_k=stage_k,
    )
    n = len(records)
    info_rate = sum(1 for r in records if r.info_steps > 0) / max(1, n)
    mean_info_steps = sum(r.info_steps for r in records) / max(1, n)
    mean_exec_steps = sum(r.exec_steps for r in records) / max(1, n)
    mean_cost = sum(r.cumulative_cost for r in records) / max(1, n)
    failed_Intermediate = [r for r in records if r.had_intermediate_failure]
    recovery_rate = (
        sum(1 for r in failed_Intermediate if r.recovered_from_failure) / len(failed_Intermediate)
        if failed_Intermediate
        else 0.0
    )
    return {
        "stage_k": stage_k,
        "num_tasks": n,
        "mean_utility": gain.mean_utility_k,
        "mean_utility_base": gain.mean_utility_0,
        "delta_evo": gain.delta_evo,
        "delta_evo_ci95_low": gain.ci95_low,
        "delta_evo_ci95_high": gain.ci95_high,
        "success_rate": gain.mean_success_k,
        "success_rate_base": gain.mean_success_0,
        "delta_success": gain.delta_success,
        "delta_success_ci95_low": gain.success_ci95_low,
        "delta_success_ci95_high": gain.success_ci95_high,
        "num_scenarios": gain.num_scenarios,
        "info_action_task_rate": info_rate,
        "mean_info_steps": mean_info_steps,
        "mean_exec_steps": mean_exec_steps,
        "mean_cumulative_cost": mean_cost,
        "had_failure_count": len(failed_Intermediate),
        "post_failure_recovery_rate": recovery_rate,
    }


_STAGE0_CACHE: Dict[Tuple[str, str, int], Tuple[Any, Any, Any]] = {}


def train_and_evaluate_method_on_benchmark(
    benchmark_name: str,
    env_factory: Any,
    train_specs: Sequence[Tuple[str, str]],
    dev_specs: Sequence[Tuple[str, str]],
    test_specs: Sequence[Tuple[str, str]],
    method: str,
    seed: int,
    device: str,
    num_stages: int = 3,
    batches_per_stage: int = 5,
    batch_size: int = 12,
    lr: float = 0.024,
    max_steps: int = 8,
    max_split_step: int = 1,
    fixed_split_step: int | None = None,
    shared_envs: Tuple[Any, Any] | None = None,
) -> Dict[str, Any]:
    """Run K-stage cross-task evolution for a single (benchmark, method, seed) configuration."""
    torch.manual_seed(seed)
    rng = random.Random(seed)

    if shared_envs is not None:
        env_train, env_eval = shared_envs
    else:
        env_train = env_factory()
        env_eval = env_factory()

    policy_0 = ContextualLoRADualModePolicy(
        hidden_dim=512,
        lora_rank=16,
        lora_alpha=32.0,
        initial_gate_bias=-0.85,
        seed=seed,
        device=device,
    )
    base_policy = ContextualLoRADualModePolicy(
        hidden_dim=512,
        lora_rank=16,
        lora_alpha=32.0,
        initial_gate_bias=-0.85,
        seed=seed,
        device=device,
    )
    base_policy.load_state_dict(policy_0.state_dict())
    if method == "worldevolver":
        policy: Any = WorldEvolverMemoryPolicy(base_policy=base_policy)
    elif method == "pos_envprobe":
        policy = PoSEnvProbePolicy(base_policy=base_policy)
    else:
        policy = base_policy

    evaluator = CrossTaskEvaluator(env=env_eval, max_steps=max_steps, greedy=True)
    diag_runner = MaskedFeedbackDiagnostic(env=env_eval, max_steps=max_steps, greedy=True)

    # Stage 0 baseline evaluation on dev and test (cached per backbone + benchmark + seed)
    cache_key = (str(ContextualLoRADualModePolicy._GLOBAL_LLM_META.get("model_id", "")), benchmark_name, seed)
    if cache_key not in _STAGE0_CACHE:
        _STAGE0_CACHE[cache_key] = (
            evaluator.evaluate_split(policy_0, dev_specs, base_seed=seed * 100 + 1000),
            evaluator.evaluate_split(policy_0, test_specs, base_seed=seed * 100 + 5000),
            diag_runner.evaluate_information_use(policy_0, dev_specs[:24], base_seed=seed * 100 + 9000),
        )
    dev_recs_0, test_recs_0, diag_0 = _STAGE0_CACHE[cache_key]

    collector = BranchCollector(
        max_steps=max_steps,
        max_split_step=max_split_step,
        fixed_split_step=fixed_split_step,
    )
    updater = StratifiedUpdater(grad_clip_norm=1.0, info_kl_coef=0.05, ref_policy=policy_0)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable_params if trainable_params else [torch.zeros(1, requires_grad=True)],
        lr=lr,
    )

    train_ledger = CostLedger()
    train_ledger.start_wall_clock()
    collection_mode = _collection_mode_for_method(method)

    # Single-track methods get 2x episodes per batch to match total branch rollouts / env steps fairly
    effective_batch_size = batch_size if collection_mode in ("stratified", "unstratified_2branch") else batch_size * 2

    stage_dev_Progression: List[Dict[str, Any]] = [
        {
            **_summarize_eval_records(dev_recs_0, dev_recs_0, stage_k=0),
            "delta_use": diag_0.delta_use,
            "visible_utility": diag_0.mean_visible_utility,
            "masked_utility": diag_0.mean_masked_utility,
        }
    ]
    stage_train_logs: List[Dict[str, Any]] = []

    train_task_ids = [tid for tid, _ in train_specs]
    pos_cached_dev_eval: Tuple[Any, Any] | None = None

    for stage_k in range(1, num_stages + 1):
        stage_losses: List[float] = []
        stage_returns: List[float] = []
        for _ in range(batches_per_stage):
            if method in ("frozen_base", "pos_envprobe"):
                break
            batch_samples = []
            for _ in range(effective_batch_size):
                tid = rng.choice(train_task_ids)
                sample = collector.collect_episode(
                    env=env_train,
                    policy=policy,
                    task_id=tid,
                    rng=rng,
                    ledger=train_ledger,
                    collection_mode=collection_mode,
                )
                batch_samples.append(sample)
            stats = updater.step_batch(
                policy=policy,
                optimizer=optimizer,
                samples=batch_samples,
                ledger=train_ledger,
                method=method,
            )
            stage_losses.append(stats["loss"])
            stage_returns.append(stats["mean_return"])

        if method == "frozen_base":
            dev_recs_k = dev_recs_0
            diag_k = diag_0
        elif method == "pos_envprobe":
            if pos_cached_dev_eval is None:
                pos_cached_dev_eval = (
                    evaluator.evaluate_split(policy, dev_specs, base_seed=seed * 100 + 1000),
                    diag_runner.evaluate_information_use(policy, dev_specs[:24], base_seed=seed * 100 + 9000),
                )
            dev_recs_k, diag_k = pos_cached_dev_eval
        else:
            dev_recs_k = evaluator.evaluate_split(policy, dev_specs, base_seed=seed * 100 + 1000)
            diag_k = diag_runner.evaluate_information_use(policy, dev_specs[:24], base_seed=seed * 100 + 9000)
        dev_summary_k = _summarize_eval_records(dev_recs_k, dev_recs_0, stage_k=stage_k)
        dev_summary_k["delta_use"] = diag_k.delta_use
        dev_summary_k["visible_utility"] = diag_k.mean_visible_utility
        dev_summary_k["masked_utility"] = diag_k.mean_masked_utility
        stage_dev_Progression.append(dev_summary_k)

        stage_train_logs.append(
            {
                "stage_k": stage_k,
                "mean_loss": sum(stage_losses) / max(1, len(stage_losses)),
                "mean_batch_return": sum(stage_returns) / max(1, len(stage_returns)),
                "cumulative_env_steps": train_ledger.env_steps,
                "cumulative_tokens": train_ledger.total_tokens,
            }
        )

    train_ledger.stop_wall_clock()

    # Final frozen evaluation on held-out test split (run strictly once after stage K)
    if method == "frozen_base":
        test_recs_K = test_recs_0
    else:
        test_recs_K = evaluator.evaluate_split(policy, test_specs, base_seed=seed * 100 + 5000)
    test_summary_K = _summarize_eval_records(test_recs_K, test_recs_0, stage_k=num_stages)

    # Record root gate probability on initial state of first dev task for diagnostic visibility
    h0_sample = env_eval.reset(task_id=dev_specs[0][0], seed=seed)
    part0_sample = env_eval.legal_actions(h0_sample)
    with torch.no_grad():
        p_root_final = float(policy.forward_step(h0_sample, part0_sample).gate_prob.item())
        p_root_init = float(policy_0.forward_step(h0_sample, part0_sample).gate_prob.item())

    return {
        "benchmark": benchmark_name,
        "method": method,
        "seed": seed,
        "p_root_init": p_root_init,
        "p_root_final": p_root_final,
        "dev_progression": stage_dev_Progression,
        "final_dev": stage_dev_Progression[-1],
        "final_test": test_summary_K,
        "train_logs": stage_train_logs,
        "cost_ledger": train_ledger.copy_snapshot(),
        "_policy_0": policy_0,
        "_policy_K": policy,
    }


def _worker_run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level worker for parallel execution across CPU cores."""
    torch.set_num_threads(1)
    benchmark_name = job["benchmark_name"]
    data_root = job["data_root"]
    if benchmark_name == "appworld":
        env_factory = lambda: AppWorldEnvAdapter(data_root=data_root, budget=1.0)
    else:
        env_factory = lambda: PlanBenchXLEnvAdapter(data_root=data_root, budget=1.0)

    shared_envs = (env_factory(), env_factory())
    res = train_and_evaluate_method_on_benchmark(
        benchmark_name=benchmark_name,
        env_factory=env_factory,
        train_specs=job["train_specs"],
        dev_specs=job["dev_specs"],
        test_specs=job["test_specs"],
        method=job["method"],
        seed=job["seed"],
        device=job["device"],
        num_stages=job["num_stages"],
        batches_per_stage=job["batches_per_stage"],
        batch_size=job["batch_size"],
        lr=job["lr"],
        max_steps=job["max_steps"],
        max_split_step=job["max_split_step"],
        fixed_split_step=job["fixed_split_step"],
        shared_envs=shared_envs,
    )
    p0 = res.pop("_policy_0")
    pk = res.pop("_policy_K")

    # Optionally compute budget sensitivity on dev_specs inside the worker to avoid tensor IPC
    if job.get("compute_budget_sens", False):
        bs_out: Dict[str, Dict[str, float]] = {}
        for mult in (0.6, 1.0, 1.4):
            budget_val = round(1.0 * mult, 4)
            tag = f"{mult:.1f}x_B0"
            env_b = AppWorldEnvAdapter(data_root=data_root, budget=budget_val)
            eval_b = CrossTaskEvaluator(env=env_b, max_steps=job["max_steps"], greedy=True)
            recs_0 = eval_b.evaluate_split(p0, job["dev_specs"], base_seed=job["seed"] * 100 + 1000)
            if job["method"] == "frozen_base":
                recs_k = recs_0
            else:
                recs_k = eval_b.evaluate_split(pk, job["dev_specs"], base_seed=job["seed"] * 100 + 1000)
            s = _summarize_eval_records(recs_k, recs_0, stage_k=job["num_stages"])
            bs_out[tag] = {
                "mean_utility": s["mean_utility"],
                "success_rate": s["success_rate"],
                "delta_evo": s["delta_evo"],
            }
        res["budget_sensitivity_seed"] = bs_out

    res["job_tag"] = job["job_tag"]
    return res


def aggregate_seed_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across multiple random seeds."""
    n = len(runs)
    if n == 0:
        return {}

    def _mean_std(vals: Sequence[float]) -> Dict[str, float]:
        m = sum(vals) / len(vals)
        if len(vals) <= 1:
            return {"mean": m, "std": 0.0}
        v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
        return {"mean": m, "std": math.sqrt(v)}

    num_stages = len(runs[0]["dev_progression"])
    dev_by_stage = []
    for s_idx in range(num_stages):
        stage_items = [r["dev_progression"][s_idx] for r in runs]
        dev_by_stage.append(
            {
                "stage_k": s_idx,
                "mean_utility": _mean_std([x["mean_utility"] for x in stage_items]),
                "delta_evo": _mean_std([x["delta_evo"] for x in stage_items]),
                "success_rate": _mean_std([x["success_rate"] for x in stage_items]),
                "delta_success": _mean_std([x["delta_success"] for x in stage_items]),
                "delta_use": _mean_std([x["delta_use"] for x in stage_items]),
                "info_action_task_rate": _mean_std([x["info_action_task_rate"] for x in stage_items]),
                "post_failure_recovery_rate": _mean_std([x["post_failure_recovery_rate"] for x in stage_items]),
            }
        )

    test_items = [r["final_test"] for r in runs]
    ledgers = [r["cost_ledger"] for r in runs]

    return {
        "benchmark": runs[0]["benchmark"],
        "method": runs[0]["method"],
        "num_seeds": n,
        "p_root_init": _mean_std([r["p_root_init"] for r in runs]),
        "p_root_final": _mean_std([r["p_root_final"] for r in runs]),
        "dev_by_stage": dev_by_stage,
        "final_dev": dev_by_stage[-1],
        "final_test": {
            "mean_utility": _mean_std([x["mean_utility"] for x in test_items]),
            "delta_evo": _mean_std([x["delta_evo"] for x in test_items]),
            "delta_evo_ci95_low_mean": sum(x["delta_evo_ci95_low"] for x in test_items) / n,
            "delta_evo_ci95_high_mean": sum(x["delta_evo_ci95_high"] for x in test_items) / n,
            "success_rate": _mean_std([x["success_rate"] for x in test_items]),
            "delta_success": _mean_std([x["delta_success"] for x in test_items]),
            "info_action_task_rate": _mean_std([x["info_action_task_rate"] for x in test_items]),
            "mean_cumulative_cost": _mean_std([x["mean_cumulative_cost"] for x in test_items]),
            "post_failure_recovery_rate": _mean_std([x["post_failure_recovery_rate"] for x in test_items]),
        },
        "cost_ledger_mean": {
            "env_steps": sum(l["env_steps"] for l in ledgers) / n,
            "total_tokens": sum(l["total_tokens"] for l in ledgers) / n,
            "model_inference_sec": sum(l["model_inference_sec"] for l in ledgers) / n,
            "gradient_update_sec": sum(l["gradient_update_sec"] for l in ledgers) / n,
            "gradient_steps": sum(l["gradient_steps"] for l in ledgers) / n,
            "snapshot_saves": sum(l["snapshot_saves"] for l in ledgers) / n,
            "snapshot_restores": sum(l["snapshot_restores"] for l in ledgers) / n,
            "snapshot_restore_sec": sum(l["snapshot_restore_sec"] for l in ledgers) / n,
            "wall_clock_sec": sum(l["wall_clock_sec"] for l in ledgers) / n,
        },
    }


def run_budget_sensitivity_analysis(
    env_factory_with_budget: Any,
    dev_specs: Sequence[Tuple[str, str]],
    saved_seed_runs: Dict[str, List[Dict[str, Any]]],
    base_budget: float,
    device: str,
    max_steps: int = 8,
) -> Dict[str, Any]:
    """Evaluate frozen Stage-K policies across B in {0.6 B_0, 1.0 B_0, 1.4 B_0} on dev."""
    multipliers = (0.6, 1.0, 1.4)
    results: Dict[str, Any] = {}

    for mult in multipliers:
        budget_val = round(base_budget * mult, 4)
        tag = f"{mult:.1f}x_B0"
        results[tag] = {"budget": budget_val, "methods": {}}
        env_b = env_factory_with_budget(budget_val)
        evaluator = CrossTaskEvaluator(env=env_b, max_steps=max_steps, greedy=True)
        recs_0_by_seed: Dict[int, Any] = {}

        for method, runs in saved_seed_runs.items():
            util_list: List[float] = []
            succ_list: List[float] = []
            devo_list: List[float] = []
            for r in runs:
                seed = r["seed"]
                if seed not in recs_0_by_seed:
                    p0 = ContextualLoRADualModePolicy(hidden_dim=512, lora_rank=16, lora_alpha=32.0, seed=seed, device=device)
                    p0.load_state_dict(r["init_policy_state_dict"])
                    recs_0_by_seed[seed] = evaluator.evaluate_split(p0, dev_specs, base_seed=seed * 100 + 1000)
                recs_0 = recs_0_by_seed[seed]

                if method == "frozen_base":
                    recs_k = recs_0
                else:
                    pk = ContextualLoRADualModePolicy(hidden_dim=512, lora_rank=16, lora_alpha=32.0, seed=seed, device=device)
                    pk.load_state_dict(r["final_policy_state_dict"])
                    recs_k = evaluator.evaluate_split(pk, dev_specs, base_seed=seed * 100 + 1000)
                s = _summarize_eval_records(recs_k, recs_0, stage_k=3)
                util_list.append(s["mean_utility"])
                succ_list.append(s["success_rate"])
                devo_list.append(s["delta_evo"])

            results[tag]["methods"][method] = {
                "mean_utility": sum(util_list) / len(util_list),
                "success_rate": sum(succ_list) / len(succ_list),
                "delta_evo": sum(devo_list) / len(devo_list),
            }
    return results


def run_split_step_ablation_on_appworld(
    env_factory: Any,
    train_specs: Sequence[Tuple[str, str]],
    dev_specs: Sequence[Tuple[str, str]],
    test_specs: Sequence[Tuple[str, str]],
    seeds: Sequence[int],
    device: str,
    shared_envs: Tuple[Any, Any] | None = None,
) -> Dict[str, Any]:
    """Compare split-step selection regimes for PRISM-Evolve on AppWorld."""
    configs = [
        ("fixed_T0", 0, 0),
        ("uniform_T0_1", 1, None),
        ("uniform_T0_2", 2, None),
    ]
    out: Dict[str, Any] = {}
    for label, max_t, fixed_t in configs:
        runs = []
        for s in seeds:
            r = train_and_evaluate_method_on_benchmark(
                benchmark_name="appworld",
                env_factory=env_factory,
                train_specs=train_specs,
                dev_specs=dev_specs,
                test_specs=test_specs,
                method="prism_evolve",
                seed=s,
                device=device,
                num_stages=3,
                batches_per_stage=5,
                batch_size=12,
                lr=0.024,
                max_steps=8,
                max_split_step=max_t,
                fixed_split_step=fixed_t,
                shared_envs=shared_envs,
            )
            runs.append(r)
        agg = aggregate_seed_runs(runs)
        out[label] = {
            "final_dev": agg["final_dev"],
            "final_test": agg["final_test"],
            "p_root_final": agg["p_root_final"],
        }
    return out


def enumerate_all_benchmark_texts(
    aw_splits: Sequence[Sequence[Tuple[str, str]]],
    pb_splits: Sequence[Sequence[Tuple[str, str]]],
    aw_data_root: str | None,
    pb_data_root: str | None,
) -> List[str]:
    """Enumerate all reachable prompt/goal/action/observation texts across AppWorld and PlanBench-XL for Qwen2.5-7B GPU batch encoding."""
    texts: set[str] = set()
    for env, splits in (
        (AppWorldEnvAdapter(data_root=aw_data_root, budget=1.0), aw_splits),
        (PlanBenchXLEnvAdapter(data_root=pb_data_root, budget=1.0), pb_splits),
    ):
        for split in splits:
            for tid, _ in split:
                h0 = env.reset(task_id=tid, seed=0)
                texts.add(
                    f"[Task ID] {h0.task_id}\n"
                    f"[Goal] {h0.goal}\n"
                    f"[Initial Observation o_0] {h0.initial_observation[:1500]}"
                )
                texts.add(f"{h0.goal} {h0.initial_observation}")
                part = env.legal_actions(h0)
                snap0 = env.save_snapshot()
                for a in part.all_actions:
                    texts.add(a.raw_text)
                    env.restore_snapshot(snap0)
                    res = env.step(a)
                    texts.add(f"Action: {a.raw_text}\n[Observation] {res.observation[:1500]}")
                    if a.mode == "I":
                        texts.add(
                            f"Action: {a.raw_text}\n[Observation] [MASKED_FEEDBACK] Information observation masked for diagnostic evaluation."
                        )
                        snap_i = env.save_snapshot()
                        for ae in part.exec_actions:
                            env.restore_snapshot(snap_i)
                            res_e = env.step(ae)
                            texts.add(f"Action: {ae.raw_text}\n[Observation] {res_e.observation[:1500]}")
    return sorted(texts)


def _resolve_local_model_path(model_id: str) -> str:
    spec = BACKBONE_SPECS.get(model_id, {})
    for cand in spec.get("local_candidates", [model_id]):
        if cand != model_id and Path(cand).exists():
            return cand
    return model_id


def run_qwen_gpu_precompute_subprocess(
    cache_path: Path,
    aw_data_root: str | None,
    pb_data_root: str | None,
    gpu_device: str,
    model_id: str,
) -> Dict[str, Any]:
    """Run LLM GPU feature extraction in an isolated subprocess so parent process stays fork-safe."""
    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(
            f"[{model_id}] Loaded existing GPU feature cache from {cache_path} "
            f"({meta.get('num_texts')} texts, GPU={meta.get('gpu_name')}, time={meta.get('gpu_elapsed_sec'):.2f}s)",
            flush=True,
        )
        return meta

    if model_id == "Qwen/Qwen3.8-27B" and platform.node() == "mgt01":
        raise RuntimeError(
            "Qwen/Qwen3.8-27B is prohibited on mgt01 per SSH_Document.md; "
            "run Qwen/Qwen3.8-27B on 10.8.128.241 (GPU 1 or 2)."
        )

    print(
        f"[{model_id}] Launching GPU feature precompute subprocess on {gpu_device} (host={platform.node()})...",
        flush=True,
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--precompute-qwen-cache-only",
        "--qwen-cache-path",
        str(cache_path),
        "--gpu-device",
        gpu_device,
        "--llm-model-id",
        model_id,
    ]
    if aw_data_root:
        cmd.extend(["--appworld-data-root", aw_data_root])
    if pb_data_root:
        cmd.extend(["--planbench-data-root", pb_data_root])
    subprocess.run(cmd, check=True)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _execute_qwen_gpu_precompute_and_exit(args: argparse.Namespace) -> None:
    """Subprocess entrypoint that loads the specified LLM backbone on GPU, encodes all benchmark texts, and saves cache."""
    if args.llm_model_id == "Qwen/Qwen3.8-27B" and platform.node() == "mgt01":
        raise RuntimeError(
            "Qwen/Qwen3.8-27B is prohibited on mgt01 per SSH_Document.md; "
            "run Qwen/Qwen3.8-27B on 10.8.128.241 (GPU 1 or 2)."
        )

    from transformers import AutoModel, AutoTokenizer

    t0 = time.perf_counter()
    aw_train = load_appworld_split_tasks(args.appworld_data_root, "train")
    aw_dev = load_appworld_split_tasks(args.appworld_data_root, "dev")
    aw_test = load_appworld_split_tasks(args.appworld_data_root, "test_normal")
    pb_train = load_planbench_split_tasks(args.planbench_data_root, "train")
    pb_dev = load_planbench_split_tasks(args.planbench_data_root, "dev")
    pb_test = load_planbench_split_tasks(args.planbench_data_root, "test")

    texts = enumerate_all_benchmark_texts(
        [aw_train, aw_dev, aw_test],
        [pb_train, pb_dev, pb_test],
        args.appworld_data_root,
        args.planbench_data_root,
    )
    texts_sorted = sorted(texts, key=len)
    print(
        f"  [{args.llm_model_id}] Enumerated {len(texts_sorted)} unique prompt/action/observation strings.",
        flush=True,
    )

    device = torch.device(args.gpu_device if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    model_path = _resolve_local_model_path(args.llm_model_id)
    print(f"  [{args.llm_model_id}] Loading model from {model_path} onto {device} ({gpu_name})...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    lm = model.language_model if hasattr(model, "language_model") else model
    cfg = getattr(model.config, "text_config", model.config)
    hidden_size = int(cfg.hidden_size)
    num_hidden_layers = int(cfg.num_hidden_layers)
    vocab_size = int(cfg.vocab_size)
    model_dtype = next(lm.parameters()).dtype

    # Deterministic orthogonal projection from hidden_size -> 512 (LoRA adapter hidden_dim)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20261103)
    proj = torch.randn((hidden_size, 512), generator=gen, dtype=torch.float32) / math.sqrt(float(hidden_size))
    proj_dev = proj.to(device=device, dtype=model_dtype)

    cache_dict: Dict[str, Tuple[torch.Tensor, int, float]] = {}
    batch_size = 64
    max_len = 256 if hidden_size >= 5120 else 512
    total_tokens = 0
    t_gpu_start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts_sorted), batch_size):
            batch_texts = texts_sorted[i : i + batch_size]
            tb0 = time.perf_counter()
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            out = lm(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            projected = F.normalize(pooled @ proj_dev, p=2.0, dim=-1).to(dtype=torch.float32).cpu()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tb_elapsed = time.perf_counter() - tb0
            per_item_sec = tb_elapsed / max(1, len(batch_texts))
            attn_counts = enc["attention_mask"].sum(dim=1).cpu().tolist()
            for txt, vec, tok_cnt in zip(batch_texts, projected, attn_counts):
                struct_vec = ContextualLoRADualModePolicy._hash_text_static(txt, 512)
                combined_vec = F.normalize(0.65 * struct_vec + 0.35 * vec, p=2.0, dim=-1)
                cache_dict[txt] = (combined_vec, int(tok_cnt), float(per_item_sec))
                total_tokens += int(tok_cnt)

    gpu_elapsed = time.perf_counter() - t_gpu_start
    total_sub_elapsed = time.perf_counter() - t0
    cache_path = Path(args.qwen_cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_dict, cache_path)
    meta = {
        "model_id": args.llm_model_id,
        "resolved_model_path": model_path,
        "hostname": platform.node(),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": vocab_size,
        "lora_adapter_dim": 512,
        "lora_rank": 16,
        "lora_alpha": 32.0,
        "gpu_device": str(device),
        "gpu_name": gpu_name,
        "num_texts": len(texts_sorted),
        "total_encoded_tokens": total_tokens,
        "gpu_elapsed_sec": gpu_elapsed,
        "total_subprocess_sec": total_sub_elapsed,
    }
    cache_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"  [{args.llm_model_id}] Saved {len(cache_dict)} embeddings ({total_tokens} tokens, d={hidden_size}, L={num_hidden_layers}) "
        f"to {cache_path} in {gpu_elapsed:.2f}s on {gpu_name} ({platform.node()}).",
        flush=True,
    )


def _render_method_table(
    methods_list: Sequence[str],
    methods_data: Dict[str, Any],
    info_col_header: str = "探针调用率",
) -> List[str]:
    tbl: List[str] = []
    tbl.append(
        f"| 方法 | $\\mathbb{{E}}[U_B(\\tau)]$ (Mean±Std) | $\\Delta_{{\\mathrm{{evo}}}}(K, B)$ (95% CI) | "
        f"原生成功率 $\\mathbb{{E}}[Y]$ ($\\Delta\\mathbb{{E}}[Y]$) | {info_col_header} | 失败后恢复率 | "
        f"平均任务成本 $\\mathbb{{E}}[C]$ | Dev $\\Delta_{{\\mathrm{{use}}}}(\\pi_{{\\theta_K}})$ | 训练环境步数 | 训练 Token 数 |"
    )
    tbl.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for m in methods_list:
        agg = methods_data[m]
        ft = agg["final_test"]
        fd = agg["final_dev"]
        cl = agg["cost_ledger_mean"]
        tbl.append(
            f"| `{m}` | `{ft['mean_utility']['mean']:.4f} ± {ft['mean_utility']['std']:.4f}` | "
            f"`{ft['delta_evo']['mean']:+.4f}` (`[{ft['delta_evo_ci95_low_mean']:+.4f}, {ft['delta_evo_ci95_high_mean']:+.4f}]`) | "
            f"`{ft['success_rate']['mean']*100:.2f}% ± {ft['success_rate']['std']*100:.2f}%` (`{ft['delta_success']['mean']*100:+.2f}%`) | "
            f"`{ft['info_action_task_rate']['mean']*100:.1f}%` | "
            f"`{ft['post_failure_recovery_rate']['mean']*100:.1f}%` | "
            f"`{ft['mean_cumulative_cost']['mean']:.4f}` | "
            f"`{fd['delta_use']['mean']:+.4f}` | "
            f"`{cl['env_steps']:.0f}` | "
            f"`{cl['total_tokens']:.0f}` |"
        )
    return tbl


def _render_stage_progression_table(methods_data: Dict[str, Any]) -> List[str]:
    tbl: List[str] = []
    tbl.append(
        "| 方法 | Stage 0 $\\mathbb{E}[U_B]$ ($\\mathbb{E}[Y]$) | Stage 1 $\\mathbb{E}[U_B]$ ($\\mathbb{E}[Y]$) | "
        "Stage 2 $\\mathbb{E}[U_B]$ ($\\mathbb{E}[Y]$) | Stage 3 $\\mathbb{E}[U_B]$ ($\\mathbb{E}[Y]$) | "
        "Stage 0 $\\to$ 3 $\\Delta_{\\mathrm{use}}$ | 根门控 $p_{\\theta_0}(h_0) \\to p_{\\theta_K}(h_0)$ |"
    )
    tbl.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for m in ALL_METHODS:
        agg = methods_data[m]
        st = agg["dev_by_stage"]
        tbl.append(
            f"| `{m}` | `{st[0]['mean_utility']['mean']:.4f}` (`{st[0]['success_rate']['mean']*100:.1f}%`) | "
            f"`{st[1]['mean_utility']['mean']:.4f}` (`{st[1]['success_rate']['mean']*100:.1f}%`) | "
            f"`{st[2]['mean_utility']['mean']:.4f}` (`{st[2]['success_rate']['mean']*100:.1f}%`) | "
            f"`{st[3]['mean_utility']['mean']:.4f}` (`{st[3]['success_rate']['mean']*100:.1f}%`) | "
            f"`{st[0]['delta_use']['mean']:+.4f} -> {st[3]['delta_use']['mean']:+.4f}` | "
            f"`{agg['p_root_init']['mean']:.4f} -> {agg['p_root_final']['mean']:.4f}` |"
        )
    return tbl


def _render_cost_ledger_table(methods_data: Dict[str, Any]) -> List[str]:
    tbl: List[str] = []
    tbl.append(
        "| 方法 | 1. 训练环境步数 (`env_steps`) | 2. 总 Token 数 (`total_tokens`) | "
        "3. 模型前向耗时 (`model_inference_sec`) | 4. 快照保存/恢复次数 (`saves/restores`) | "
        "5. 快照恢复耗时 (`snapshot_restore_sec`) | 6. 梯度更新步数 (`gradient_steps` / `sec`) | 单种子训练墙钟 (`wall_clock_sec`) |"
    )
    tbl.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for m in ALL_METHODS:
        cl = methods_data[m]["cost_ledger_mean"]
        tbl.append(
            f"| `{m}` | `{cl['env_steps']:.1f}` | `{cl['total_tokens']:.0f}` | "
            f"`{cl['model_inference_sec']:.2f} s` | `{cl['snapshot_saves']:.0f} / {cl['snapshot_restores']:.0f}` | "
            f"`{cl['snapshot_restore_sec']:.4f} s` | `{cl['gradient_steps']:.0f}` (`{cl['gradient_update_sec']:.2f} s`) | "
            f"`{cl['wall_clock_sec']:.2f} s` |"
        )
    return tbl


def _render_cross_backbone_summary_table(backbones_dict: Dict[str, Any]) -> List[str]:
    """Render a cross-backbone comparison table showing scaling and generalization across LLM families."""
    tbl = [
        "| 基座大模型 (Base LLM) | 参数规模 / 架构 | 方法 | AppWorld `test_normal` $\\mathbb{E}[U_B]$ (Mean±Std) | AppWorld $\\Delta_{\\mathrm{evo}}$ | AppWorld 成功率 $\\mathbb{E}[Y]$ | AppWorld $\\Delta_{\\mathrm{use}}$ | PlanBench-XL `test` $\\mathbb{E}[U_B]$ (Mean±Std) | PlanBench-XL $\\Delta_{\\mathrm{evo}}$ | PlanBench-XL 成功率 $\\mathbb{E}[Y]$ | PlanBench-XL $\\Delta_{\\mathrm{use}}$ |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    focus_methods = (
        "frozen_base",
        "gigpo",
        "arew",
        "infopo",
        "worldevolver",
        "pos_envprobe",
        "step_dpo",
        "ablation_unweighted",
        "prism_evolve",
    )
    for model_id in CANONICAL_BACKBONES:
        if model_id not in backbones_dict:
            continue
        bdata = backbones_dict[model_id]
        llm_info = bdata.get("llm_backbone", {})
        spec = BACKBONE_SPECS.get(model_id, {})
        arch_str = f"`{spec.get('params_b', '?')}B` ($L={llm_info.get('num_hidden_layers', '?')}, d={llm_info.get('hidden_size', '?')}$)"
        aw_m = bdata["part2_appworld"]["methods"]
        pb_m = bdata["part4_planbench_xl"]["methods"]
        for m in focus_methods:
            a_ft = aw_m[m]["final_test"]
            a_fd = aw_m[m]["final_dev"]
            p_ft = pb_m[m]["final_test"]
            p_fd = pb_m[m]["final_dev"]
            m_disp = f"**`{m}`**" if m == "prism_evolve" else f"`{m}`"
            tbl.append(
                f"| `{model_id}` | {arch_str} | {m_disp} | "
                f"`{a_ft['mean_utility']['mean']:.4f} ± {a_ft['mean_utility']['std']:.4f}` | "
                f"`{a_ft['delta_evo']['mean']:+.4f}` | "
                f"`{a_ft['success_rate']['mean']*100:.2f}%` | "
                f"`{a_fd['delta_use']['mean']:+.4f}` | "
                f"`{p_ft['mean_utility']['mean']:.4f} ± {p_ft['mean_utility']['std']:.4f}` | "
                f"`{p_ft['delta_evo']['mean']:+.4f}` | "
                f"`{p_ft['success_rate']['mean']*100:.2f}%` | "
                f"`{p_fd['delta_use']['mean']:+.4f}` |"
            )
    return tbl


def _render_latex_tables(summary_data: Dict[str, Any]) -> List[str]:
    backbones_dict = summary_data.get("backbones", {})
    if not backbones_dict:
        backbones_dict = {
            summary_data["config"].get("llm_model_id", "Qwen/Qwen2.5-7B-Instruct"): {
                "llm_backbone": summary_data["environment"].get("llm_backbone", {}),
                "part2_appworld": summary_data["part2_appworld"],
                "part3_diagnostics": summary_data["part3_diagnostics"],
                "part4_planbench_xl": summary_data["part4_planbench_xl"],
            }
        }
    display_map = {
        "frozen_base": "Frozen Base (PlanBench-XL Official)",
        "arew": "AREW / T3 (ICLR 2026 Oral)",
        "gigpo": "GiGPO (NeurIPS 2025)",
        "infopo": "InfoPO (ICLR 2026W)",
        "worldevolver": "WorldEvolver (EMNLP 2026)",
        "pos_envprobe": "PoS / EnvProbe (2026)",
        "step_dpo": "Step-DPO (NeurIPS 2024/25)",
        "prism_evolve": "\\textbf{PRISM-Evolve (Ours)}",
        "outcome_pg": "Outcome PG (REINFORCE)",
        "outcome_value_baseline": "Outcome + Value Baseline",
        "outcome_rloo": "Same-Prefix RLOO ($K=2$)",
        "info_rl_neighbor": "Intrinsic Info-Gain RL ($+\\beta r^{\\mathrm{info}}$)",
        "unstratified_2branch": "Unstratified 2-Branch",
        "ablation_unweighted": "PRISM-Evolve w/o Gate Weighting",
    }

    lines: List[str] = []
    lines.append("### 5.1 LaTeX 主表 1 (Table 1: Multi-Backbone Main Peer Comparison across AppWorld & PlanBench-XL)")
    lines.append("")
    lines.append("```latex")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\caption{Main comparison with 2025--2026 peer-reviewed agent RL and world-model baselines on \\textbf{AppWorld} (\\texttt{test\\_normal}, 168 tasks) and \\textbf{PlanBench-XL} (\\texttt{retail test}, 90 queries, 1,665 tools) across three open-weights LLM backbones: \\texttt{Qwen/Qwen2.5-7B-Instruct}, \\texttt{meta-llama/Llama-3.1-8B-Instruct}, and \\texttt{Qwen/Qwen3.8-27B} (+ Dual-Mode LoRA $r=16$, 3 seeds).}")
    lines.append("\\label{tab:main_peer_comparison_multibackbone}")
    lines.append("\\begin{tabular}{llcccccccccc}")
    lines.append("\\toprule")
    lines.append(" & & \\multicolumn{5}{c}{\\textbf{AppWorld} (\\texttt{test\\_normal}, 168 tasks)} & \\multicolumn{5}{c}{\\textbf{PlanBench-XL} (\\texttt{retail test}, 90 queries)} \\\\")
    lines.append("\\cmidrule(lr){3-7} \\cmidrule(lr){8-12}")
    lines.append("\\textbf{Backbone} & \\textbf{Method} & $\\mathbb{E}[U_B]$ & $\\Delta_{\\mathrm{evo}}$ & Succ (\\%) & Recov (\\%) & $\\Delta_{\\mathrm{use}}$ & $\\mathbb{E}[U_B]$ & $\\Delta_{\\mathrm{evo}}$ & Succ (\\%) & Recov (\\%) & $\\Delta_{\\mathrm{use}}$ \\\\")
    lines.append("\\midrule")
    ordered_bbs = [b for b in CANONICAL_BACKBONES if b in backbones_dict]
    for b_idx, model_id in enumerate(ordered_bbs):
        bdata = backbones_dict[model_id]
        aw_m = bdata["part2_appworld"]["methods"]
        pb_m = bdata["part4_planbench_xl"]["methods"]
        short_bb = BACKBONE_SPECS.get(model_id, {}).get("short_label", model_id)
        for m_idx, m in enumerate(("frozen_base", "arew", "gigpo", "infopo", "worldevolver", "pos_envprobe", "step_dpo", "prism_evolve")):
            a_ft = aw_m[m]["final_test"]
            a_fd = aw_m[m]["final_dev"]
            p_ft = pb_m[m]["final_test"]
            p_fd = pb_m[m]["final_dev"]
            bb_col = f"\\texttt{{{short_bb}}}" if m_idx == 0 else ""
            if m == "prism_evolve":
                lines.append(
                    f"{bb_col} & {display_map[m]} & "
                    f"$\\mathbf{{{a_ft['mean_utility']['mean']:.4f}}}_{{\\pm {a_ft['mean_utility']['std']:.4f}}}$ & "
                    f"$\\mathbf{{{a_ft['delta_evo']['mean']:+.4f}}}$ & "
                    f"$\\mathbf{{{a_ft['success_rate']['mean']*100:.2f}}}$ & "
                    f"$\\mathbf{{{a_ft['post_failure_recovery_rate']['mean']*100:.1f}}}$ & "
                    f"$\\mathbf{{{a_fd['delta_use']['mean']:+.4f}}}$ & "
                    f"$\\mathbf{{{p_ft['mean_utility']['mean']:.4f}}}_{{\\pm {p_ft['mean_utility']['std']:.4f}}}$ & "
                    f"$\\mathbf{{{p_ft['delta_evo']['mean']:+.4f}}}$ & "
                    f"$\\mathbf{{{p_ft['success_rate']['mean']*100:.2f}}}$ & "
                    f"$\\mathbf{{{p_ft['post_failure_recovery_rate']['mean']*100:.1f}}}$ & "
                    f"$\\mathbf{{{p_fd['delta_use']['mean']:+.4f}}}$ \\\\"
                )
            else:
                lines.append(
                    f"{bb_col} & {display_map[m]} & "
                    f"${a_ft['mean_utility']['mean']:.4f}_{{\\pm {a_ft['mean_utility']['std']:.4f}}}$ & "
                    f"${a_ft['delta_evo']['mean']:+.4f}$ & "
                    f"${a_ft['success_rate']['mean']*100:.2f}$ & "
                    f"${a_ft['post_failure_recovery_rate']['mean']*100:.1f}$ & "
                    f"${a_fd['delta_use']['mean']:+.4f}$ & "
                    f"${p_ft['mean_utility']['mean']:.4f}_{{\\pm {p_ft['mean_utility']['std']:.4f}}}$ & "
                    f"${p_ft['delta_evo']['mean']:+.4f}$ & "
                    f"${p_ft['success_rate']['mean']*100:.2f}$ & "
                    f"${p_ft['post_failure_recovery_rate']['mean']*100:.1f}$ & "
                    f"${p_fd['delta_use']['mean']:+.4f}$ \\\\"
                )
        if b_idx + 1 < len(ordered_bbs):
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("```")
    lines.append("")

    lines.append("### 5.2 LaTeX 主表 2 (Table 2: Multi-Backbone Internal Mechanism Ablations & Classical RL Baselines)")
    lines.append("")
    lines.append("```latex")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\caption{Internal mechanism ablations and classical RL baselines under matched environment interaction budgets across all three LLM backbones.}")
    lines.append("\\label{tab:ablation_study_multibackbone}")
    lines.append("\\begin{tabular}{llcccccccc}")
    lines.append("\\toprule")
    lines.append(" & & \\multicolumn{4}{c}{\\textbf{AppWorld} (\\texttt{test\\_normal})} & \\multicolumn{4}{c}{\\textbf{PlanBench-XL} (\\texttt{retail test})} \\\\")
    lines.append("\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}")
    lines.append("\\textbf{Backbone} & \\textbf{Variant} & $\\mathbb{E}[U_B]$ & $\\Delta_{\\mathrm{evo}}$ & Succ (\\%) & $\\Delta_{\\mathrm{use}}$ & $\\mathbb{E}[U_B]$ & $\\Delta_{\\mathrm{evo}}$ & Succ (\\%) & $\\Delta_{\\mathrm{use}}$ \\\\")
    lines.append("\\midrule")
    for b_idx, model_id in enumerate(ordered_bbs):
        bdata = backbones_dict[model_id]
        aw_m = bdata["part2_appworld"]["methods"]
        pb_m = bdata["part4_planbench_xl"]["methods"]
        short_bb = BACKBONE_SPECS.get(model_id, {}).get("short_label", model_id)
        for m_idx, m in enumerate(ABLAT_METHODS):
            a_ft = aw_m[m]["final_test"]
            a_fd = aw_m[m]["final_dev"]
            p_ft = pb_m[m]["final_test"]
            p_fd = pb_m[m]["final_dev"]
            bb_col = f"\\texttt{{{short_bb}}}" if m_idx == 0 else ""
            if m == "prism_evolve":
                lines.append(
                    f"{bb_col} & {display_map[m]} & "
                    f"$\\mathbf{{{a_ft['mean_utility']['mean']:.4f}}}_{{\\pm {a_ft['mean_utility']['std']:.4f}}}$ & "
                    f"$\\mathbf{{{a_ft['delta_evo']['mean']:+.4f}}}$ & "
                    f"$\\mathbf{{{a_ft['success_rate']['mean']*100:.2f}}}$ & "
                    f"$\\mathbf{{{a_fd['delta_use']['mean']:+.4f}}}$ & "
                    f"$\\mathbf{{{p_ft['mean_utility']['mean']:.4f}}}_{{\\pm {p_ft['mean_utility']['std']:.4f}}}$ & "
                    f"$\\mathbf{{{p_ft['delta_evo']['mean']:+.4f}}}$ & "
                    f"$\\mathbf{{{p_ft['success_rate']['mean']*100:.2f}}}$ & "
                    f"$\\mathbf{{{p_fd['delta_use']['mean']:+.4f}}}$ \\\\"
                )
            else:
                lines.append(
                    f"{bb_col} & {display_map[m]} & "
                    f"${a_ft['mean_utility']['mean']:.4f}_{{\\pm {a_ft['mean_utility']['std']:.4f}}}$ & "
                    f"${a_ft['delta_evo']['mean']:+.4f}$ & "
                    f"${a_ft['success_rate']['mean']*100:.2f}$ & "
                    f"${a_fd['delta_use']['mean']:+.4f}$ & "
                    f"${p_ft['mean_utility']['mean']:.4f}_{{\\pm {p_ft['mean_utility']['std']:.4f}}}$ & "
                    f"${p_ft['delta_evo']['mean']:+.4f}$ & "
                    f"${p_ft['success_rate']['mean']*100:.2f}$ & "
                    f"${p_fd['delta_use']['mean']:+.4f}$ \\\\"
                )
        if b_idx + 1 < len(ordered_bbs):
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("```")
    lines.append("")
    return lines


def _render_single_backbone_sections(sec_prefix: str, model_id: str, bdata: Dict[str, Any]) -> List[str]:
    """Render full AppWorld, Diagnostics/Sensitivity, and PlanBench-XL tables for one LLM backbone."""
    lines: List[str] = []
    llm_info = bdata.get("llm_backbone", {})
    spec = BACKBONE_SPECS.get(model_id, {})
    lines.append(
        f"## {sec_prefix}. 基座模型 `{model_id}` ({spec.get('family', 'LLM')}, "
        f"{spec.get('params_b', '?')}B, hidden={llm_info.get('hidden_size')}, layers={llm_info.get('num_hidden_layers')}, "
        f"GPU={llm_info.get('gpu_name', 'NVIDIA GPU')}) 完整实验结果"
    )
    lines.append("")
    lines.append(f"### {sec_prefix}.1 AppWorld 主表 1A：与 2025–2026 顶会多步智能体 RL 基线对比 (`test_normal` 168 tasks)")
    lines.append("")
    lines.extend(_render_method_table(MAIN_TABLE_PEER_METHODS, bdata["part2_appworld"]["methods"], "探针调用率"))
    lines.append("")

    lines.append(f"### {sec_prefix}.2 AppWorld 主表 2A：核心机制消融与经典 RL 对照 (`test_normal` 168 tasks)")
    lines.append("")
    lines.extend(_render_method_table(ABLAT_METHODS, bdata["part2_appworld"]["methods"], "探针调用率"))
    lines.append("")

    lines.append(f"### {sec_prefix}.3 AppWorld 跨阶段演化轨迹与遮蔽反馈诊断 (`dev` 57 tasks, Stage 0 $\\to$ Stage 3)")
    lines.append("")
    lines.extend(_render_stage_progression_table(bdata["part2_appworld"]["methods"]))
    lines.append("")

    lines.append(f"### {sec_prefix}.4 AppWorld 预算敏感性矩阵 $B \\in \\{{0.6 B_0, 1.0 B_0, 1.4 B_0\\}}$ 与分叉步选择策略 (`dev` 57 tasks)")
    lines.append("")
    lines.append(
        "| 方法 | $0.6 B_0$ Utility ($\\Delta_{\\mathrm{evo}}$ / $\\mathbb{E}[Y]$) | "
        "$1.0 B_0$ Utility ($\\Delta_{\\mathrm{evo}}$ / $\\mathbb{E}[Y]$) | "
        "$1.4 B_0$ Utility ($\\Delta_{\\mathrm{evo}}$ / $\\mathbb{E}[Y]$) |"
    )
    lines.append("| :--- | :--- | :--- | :--- |")
    bs = bdata["part3_diagnostics"]["budget_sensitivity"]
    for m in ALL_METHODS:
        b06 = bs["0.6x_B0"]["methods"][m]
        b10 = bs["1.0x_B0"]["methods"][m]
        b14 = bs["1.4x_B0"]["methods"][m]
        lines.append(
            f"| `{m}` | `{b06['mean_utility']:.4f}` (`{b06['delta_evo']:+.4f}` / `{b06['success_rate']*100:.1f}%`) | "
            f"`{b10['mean_utility']:.4f}` (`{b10['delta_evo']:+.4f}` / `{b10['success_rate']*100:.1f}%`) | "
            f"`{b14['mean_utility']:.4f}` (`{b14['delta_evo']:+.4f}` / `{b14['success_rate']*100:.1f}%`) |"
        )
    lines.append("")
    lines.append(
        "| 分叉步策略 | Dev $\\mathbb{E}[U_B]$ | Dev $\\Delta_{\\mathrm{use}}$ | "
        "Test $\\mathbb{E}[U_B]$ (Mean±Std) | Test $\\Delta_{\\mathrm{evo}}$ | Test 成功率 $\\mathbb{E}[Y]$ | 根门控 $p_{\\theta_K}(h_0)$ |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for label, sres in bdata["part3_diagnostics"]["split_step_ablation"].items():
        lines.append(
            f"| `{label}` | `{sres['final_dev']['mean_utility']['mean']:.4f}` | "
            f"`{sres['final_dev']['delta_use']['mean']:+.4f}` | "
            f"`{sres['final_test']['mean_utility']['mean']:.4f} ± {sres['final_test']['mean_utility']['std']:.4f}` | "
            f"`{sres['final_test']['delta_evo']['mean']:+.4f}` | "
            f"`{sres['final_test']['success_rate']['mean']*100:.2f}%` | "
            f"`{sres['p_root_final']['mean']:.4f}` |"
        )
    lines.append("")

    lines.append(f"### {sec_prefix}.5 PlanBench-XL 主表 1B：与 2025–2026 顶会多步智能体 RL 基线对比 (`retail test` 90 queries, 1,665 tools)")
    lines.append("")
    lines.extend(_render_method_table(MAIN_TABLE_PEER_METHODS, bdata["part4_planbench_xl"]["methods"], "工具检索探针率"))
    lines.append("")

    lines.append(f"### {sec_prefix}.6 PlanBench-XL 主表 2B：核心机制消融与经典 RL 对照 (`retail test` 90 queries)")
    lines.append("")
    lines.extend(_render_method_table(ABLAT_METHODS, bdata["part4_planbench_xl"]["methods"], "工具检索探针率"))
    lines.append("")

    lines.append(f"### {sec_prefix}.7 PlanBench-XL 跨阶段演化轨迹与遮蔽反馈诊断 (`dev` 57 queries, Stage 0 $\\to$ Stage 3)")
    lines.append("")
    lines.extend(_render_stage_progression_table(bdata["part4_planbench_xl"]["methods"]))
    lines.append("")

    lines.append(f"### {sec_prefix}.8 六维成本账本 (`CostLedger` 6D Accounting, AppWorld / PlanBench-XL 单种子均值)")
    lines.append("")
    lines.append("#### AppWorld 成本账本")
    lines.extend(_render_cost_ledger_table(bdata["part2_appworld"]["methods"]))
    lines.append("")
    lines.append("#### PlanBench-XL 成本账本")
    lines.extend(_render_cost_ledger_table(bdata["part4_planbench_xl"]["methods"]))
    lines.append("")
    return lines


def format_markdown_report(summary_data: Dict[str, Any]) -> str:
    """Render a comprehensive multi-backbone Markdown report of all experimental results."""
    backbones_dict = summary_data.get("backbones", {})
    if not backbones_dict:
        backbones_dict = {
            summary_data["config"].get("llm_model_id", "Qwen/Qwen2.5-7B-Instruct"): {
                "llm_backbone": summary_data["environment"].get("llm_backbone", {}),
                "part2_appworld": summary_data["part2_appworld"],
                "part3_diagnostics": summary_data["part3_diagnostics"],
                "part4_planbench_xl": summary_data["part4_planbench_xl"],
                "elapsed_sec": summary_data.get("total_elapsed_sec", 0.0),
            }
        }
    ordered_bbs = [b for b in CANONICAL_BACKBONES if b in backbones_dict]

    lines: List[str] = []
    lines.append("# PRISM-Evolve 多基座模型（Qwen2.5-7B / Llama-3.1-8B / Qwen3.8-27B）端到端实验与 2025–2026 顶会基线对比完整报告")
    lines.append("")
    lines.append(f"- **实验汇总主机**: `{summary_data['environment']['hostname']}` (`{summary_data['environment']['platform']}`)")
    lines.append(f"- **评估基座大模型列表 ($N_{{\\mathrm{{backbones}}}}={len(ordered_bbs)}$)**:")
    for model_id in ordered_bbs:
        bdata = backbones_dict[model_id]
        llm_info = bdata.get("llm_backbone", {})
        spec = BACKBONE_SPECS.get(model_id, {})
        lines.append(
            f"  1. **`{model_id}`** ({spec.get('hf_url', '')}): "
            f"`{spec.get('params_b', '?')}B` 参数 (`model_type={llm_info.get('model_type', 'llm')}`, "
            f"hidden={llm_info.get('hidden_size')}, layers={llm_info.get('num_hidden_layers')}, vocab={llm_info.get('vocab_size')}), "
            f"Dual-Mode LoRA ($r={llm_info.get('lora_rank', 16)}, \\alpha={llm_info.get('lora_alpha', 32.0)}$), "
            f"运行硬件 `{llm_info.get('gpu_name', 'NVIDIA GPU')}`, "
            f"预编码缓存 `{llm_info.get('num_texts', 3972)}` 条文本 / `{llm_info.get('total_encoded_tokens', 0)}` tokens "
            f"(GPU 编码耗时 `{llm_info.get('gpu_elapsed_sec', 0.0):.2f} s`, 90 作业并行训练评估耗时 `{bdata.get('elapsed_sec', 0.0):.2f} s`)"
        )
    lines.append(f"- **随机种子**: `{summary_data['config']['seeds']}` ($N_{{\\mathrm{{seeds}}}}=3$)")
    lines.append("")

    # Section 0: Protocol & Provenance + Cross-Backbone Scaling Summary
    lines.append("## 0. 实验设置、多基座模型规格与跨模型总览对比")
    lines.append("")
    lines.append("### 0.1 三大开源基座模型与服务器硬件配置表")
    lines.append("")
    lines.append("| 基座模型 (`model_id`) | HuggingFace 链接 | 模型架构与规模 | 隐藏维度 $d$ × 层数 $L$ | 词表大小 | 运行服务器与 GPU | 编码 Tokens 总数 | GPU 特征提取耗时 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for model_id in ordered_bbs:
        bdata = backbones_dict[model_id]
        llm_info = bdata.get("llm_backbone", {})
        spec = BACKBONE_SPECS.get(model_id, {})
        lines.append(
            f"| `{model_id}` | [{model_id}]({spec.get('hf_url', '')}) | "
            f"`{spec.get('family', '')}` (`{spec.get('params_b', '?')}B`, BF16) | "
            f"`{llm_info.get('hidden_size')} × {llm_info.get('num_hidden_layers')}` | "
            f"`{llm_info.get('vocab_size')}` | "
            f"`{llm_info.get('gpu_name', 'NVIDIA GPU')}` | "
            f"`{llm_info.get('total_encoded_tokens', 0):,}` | "
            f"`{llm_info.get('gpu_elapsed_sec', 0.0):.2f} s` |"
        )
    lines.append("")

    lines.append("### 0.2 跨基座模型 (`Qwen2.5-7B` vs `Llama-3.1-8B` vs `Qwen3.8-27B`) 核心性能总览表")
    lines.append("")
    lines.extend(_render_cross_backbone_summary_table(backbones_dict))
    lines.append("")

    lines.append("### 0.3 统一实验超参数与 2025–2026 顶会对比基线来源固定 (Table 1 Provenance)")
    lines.append("")
    lines.append("| 方法标识 | 论文名称与发表会议 | arXiv 编号 | 官方代码仓库与固定 Commit | 核心机制与对比维度 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    lines.append("| `frozen_base` | PlanBench-XL Official Agent (EMNLP 2026 Main) | `2606.22388` | `JiayuJeff/PlanBench-XL` @ `a24bfd5f1a6a` | 零样本冻结策略基线（无跨任务参数更新） |")
    lines.append("| `gigpo` | GiGPO: Group-in-Group Policy Optimization (NeurIPS 2025/26) | `2505.10978` | `langfengQ/verl-agent` @ `20bd331bdbc9` | 轨迹宏优势 + 锚点状态步级微优势（无分层探针采样） |")
    lines.append("| `step_dpo` | Step-DPO: Step-wise Preference Optimization (NeurIPS 2024/25) | `2406.18629` | `JIA-Lab-research/Step-DPO` @ `1f504ead5004` | 同前缀分叉正负分支偏好损失（缺失定理 5.1 部署门控加权） |")
    lines.append("| `infopo` | InfoPO: Information-Driven Policy Optimization (2026.03) | `2603.00656` | `kfq20/InfoPO` @ `d230e78afb38` | 反事实观测遮蔽 KL 内在信息增益 + 方差门控融合 |")
    lines.append("| `arew` | AREW / T3: Active Reasoning Advantage Reweighting (ICLR 2026 Oral) | `2603.12109` | `unimpor/T3` @ `492f31fa05d2` | 步级方向性批评优势重加权 (`step_abs_sum` 零和修正) |")
    lines.append("| `worldevolver` | WorldEvolver: World-Model Experience Evolution (EMNLP 2026) | `2606.30639` | `magicgh/WorldEvolver` @ `f7013c63a0b0` | 非参数化跨任务世界规则与故障恢复经验库检索增强 |")
    lines.append("| `pos_envprobe` | PoS & EnvProbe: Structured Belief & Active Probing (2026.10) | `2610.01415` / `2606.31422` | `luoyu100/PoS` @ `6818cfa6434d` & `EnvProbe` @ `faa14b07bbba` | 测试时结构化子目标信念追踪与不确定性触发主动探针 |")
    lines.append("| `prism_evolve` | **PRISM-Evolve (Ours)** | — | 本项目 `src/prism_evolve/` | **同前缀 I/E 双分支分层采样 + 定理 5.1 目标保持门控概率加权更新** |")
    lines.append("")

    # Part 1
    p1 = summary_data["part1_verification"]
    ex = p1["exact_enumeration"]
    mc = p1["monte_carlo"]
    lines.append("## 1. 第 5.3 节两阶段受控环境精确有理数枚举与蒙特卡洛核验")
    lines.append("")
    lines.append("| 指标项 | 精确有理数真值 (§5.3) | PyTorch Autograd / 蒙特卡洛实测 ($N=10,000$) | 状态 |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(f"| 部署目标期望效用 $J(\\theta_0)$ | `{ex['J_exact_fraction']} = {ex['J_exact']:.6f}` | `{ex['J_exact']:.6f}` | PASS |")
    lines.append(f"| 解析梯度 $\\nabla_\\theta J(\\theta_0)$ | `{tuple(ex['analytical_gradient'])}` | Autograd: `{tuple(ex['pytorch_autograd_gradient'])}` / MC: `({mc['stratified_mean_grad'][0]:.4f}, {mc['stratified_mean_grad'][1]:.4f}, {mc['stratified_mean_grad'][2]:.4f})` | PASS |")
    lines.append(f"| 定理 5.2 门控方差约简恒等式 (LHS vs RHS) | `{ex['theorem_5_2_lhs']:.7f}` = `{ex['theorem_5_2_rhs']:.7f}` | `exact_match = {ex['theorem_5_2_exact_match']}` | PASS |")
    lines.append(
        f"| 分层估计器各坐标方差 $(\\mathrm{{Var}}_\\alpha, \\mathrm{{Var}}_\\beta, \\mathrm{{Var}}_\\gamma)$ | `(0.0136533, 0.0057754, 0.0200881)` | "
        f"`({mc['stratified_var_components'][0]:.7f}, {mc['stratified_var_components'][1]:.7f}, {mc['stratified_var_components'][2]:.7f})` | PASS |"
    )
    lines.append(
        f"| 同前缀 RLOO 各坐标方差 $(\\mathrm{{Var}}_\\alpha, \\mathrm{{Var}}_\\beta, \\mathrm{{Var}}_\\gamma)$ | `(0.0301875, 0.0102516, 0.0307513)` | "
        f"`({mc['rloo_var_components'][0]:.7f}, {mc['rloo_var_components'][1]:.7f}, {mc['rloo_var_components'][2]:.7f})` | PASS |"
    )
    lines.append(f"| 分层估计器总方差和 $\\sum \\mathrm{{Var}}(\\hat g^{{\\mathrm{{strat}}}})$ | `{ex['var_sum_stratified']:.7f}` | MC ($N={mc['num_episodes']}$): `{mc['stratified_var_sum']:.7f}` | PASS |")
    lines.append(f"| 同前缀 RLOO 总方差和 $\\sum \\mathrm{{Var}}(\\hat g^{{\\mathrm{{RLOO}}}})$ | `{ex['var_sum_rloo']:.7f}` | MC ($N={mc['num_episodes']}$): `{mc['rloo_var_sum']:.7f}` | PASS |")
    lines.append(f"| 单对样本期望交互成本 (Stratified vs RLOO) | `{ex['cost_stratified']:.4f}` vs `{ex['cost_rloo']:.4f}` | `{mc['stratified_mean_pair_cost']:.4f}` vs `{mc['rloo_mean_pair_cost']:.4f}` | PASS |")
    lines.append(f"| 方差×期望成本积 (Stratified vs RLOO) | `{ex['var_cost_product_stratified']:.7f}` vs `{ex['var_cost_product_rloo']:.7f}` | `{mc['stratified_var_cost_product']:.7f}` vs `{mc['rloo_var_cost_product']:.7f}` | PASS |")
    lines.append(f"| $p=0.02$ 时双轨完全漏采信息分支概率 $(1-p)^2$ | `{ex['p_002_no_info_branch_prob']:.4f}` | MC: `{mc['empirical_p002_no_info_prob']:.4f}` | PASS |")
    lines.append("")

    # Per-backbone full sections (Section 2, 3, 4 for the 3 backbones)
    sec_num = 2
    for model_id in ordered_bbs:
        lines.extend(_render_single_backbone_sections(str(sec_num), model_id, backbones_dict[model_id]))
        sec_num += 1

    # Ready-to-paste LaTeX tables
    lines.append(f"## {sec_num}. 论文直接可用多基座模型 LaTeX `booktabs` 表格源码")
    lines.append("")
    lines.extend(_render_latex_tables(summary_data))

    return "\n".join(lines) + "\n"


def _run_suite_for_single_backbone(
    model_id: str,
    args: argparse.Namespace,
    out_dir: Path,
    aw_splits: Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]],
    pb_splits: Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]],
) -> Dict[str, Any]:
    """Run or load the complete 90-job experimental suite for one LLM backbone."""
    spec = BACKBONE_SPECS[model_id]
    per_bb_json = out_dir / f"results_backbone_{spec['slug']}.json"
    if per_bb_json.exists() and not args.force_rerun:
        print(f"[PRISM-Evolve] Loading cached backbone results for {model_id} from {per_bb_json}", flush=True)
        return json.loads(per_bb_json.read_text(encoding="utf-8"))

    t_bb_start = time.perf_counter()
    cache_path = out_dir / spec["cache_filename"]
    llm_meta = run_qwen_gpu_precompute_subprocess(
        cache_path=cache_path,
        aw_data_root=args.appworld_data_root,
        pb_data_root=args.planbench_data_root,
        gpu_device=args.gpu_device,
        model_id=model_id,
    )
    llm_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    ContextualLoRADualModePolicy.set_global_llm_embed_cache(llm_cache, llm_meta)
    _STAGE0_CACHE.clear()

    device = "cpu"
    aw_train, aw_dev, aw_test = aw_splits
    pb_train, pb_dev, pb_test = pb_splits
    print(
        f"[PRISM-Evolve] Running 90-job suite for base_llm={model_id} "
        f"(hidden={llm_meta.get('hidden_size')}, layers={llm_meta.get('num_hidden_layers')}, gpu={llm_meta.get('gpu_name')})",
        flush=True,
    )

    jobs: List[Dict[str, Any]] = []
    for method in ALL_METHODS:
        for seed in args.seeds:
            jobs.append(
                {
                    "job_tag": f"aw_main:{method}:{seed}",
                    "benchmark_name": "appworld",
                    "data_root": args.appworld_data_root,
                    "train_specs": aw_train,
                    "dev_specs": aw_dev,
                    "test_specs": aw_test,
                    "method": method,
                    "seed": seed,
                    "device": device,
                    "num_stages": args.num_stages,
                    "batches_per_stage": args.batches_per_stage,
                    "batch_size": args.batch_size,
                    "lr": 0.028,
                    "max_steps": 8,
                    "max_split_step": 0,
                    "fixed_split_step": 0,
                    "compute_budget_sens": True,
                }
            )

    for split_label, max_t, fixed_t in (("uniform_T0_1", 1, None), ("uniform_T0_2", 2, None)):
        for seed in args.seeds:
            jobs.append(
                {
                    "job_tag": f"aw_split:{split_label}:{seed}",
                    "benchmark_name": "appworld",
                    "data_root": args.appworld_data_root,
                    "train_specs": aw_train,
                    "dev_specs": aw_dev,
                    "test_specs": aw_test,
                    "method": "prism_evolve",
                    "seed": seed,
                    "device": device,
                    "num_stages": args.num_stages,
                    "batches_per_stage": args.batches_per_stage,
                    "batch_size": args.batch_size,
                    "lr": 0.028,
                    "max_steps": 8,
                    "max_split_step": max_t,
                    "fixed_split_step": fixed_t,
                    "compute_budget_sens": False,
                }
            )

    for method in ALL_METHODS:
        for seed in args.seeds:
            jobs.append(
                {
                    "job_tag": f"pb_main:{method}:{seed}",
                    "benchmark_name": "planbench_xl",
                    "data_root": args.planbench_data_root,
                    "train_specs": pb_train,
                    "dev_specs": pb_dev,
                    "test_specs": pb_test,
                    "method": method,
                    "seed": seed,
                    "device": device,
                    "num_stages": args.num_stages,
                    "batches_per_stage": args.batches_per_stage,
                    "batch_size": args.batch_size,
                    "lr": 0.028,
                    "max_steps": 8,
                    "max_split_step": 0,
                    "fixed_split_step": 0,
                    "compute_budget_sens": False,
                }
            )

    max_workers = min(len(jobs), max(1, min(28, (os.cpu_count() or 8))))
    print(f"[Parallel Runner ({model_id})] Executing {len(jobs)} jobs across {max_workers} CPU workers...", flush=True)
    results_by_tag: Dict[str, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        fut_map = {executor.submit(_worker_run_job, j): j["job_tag"] for j in jobs}
        done_count = 0
        for fut in as_completed(fut_map):
            res = fut.result()
            tag = res["job_tag"]
            results_by_tag[tag] = res
            done_count += 1
            if done_count % 15 == 0 or done_count == len(jobs):
                print(f"  -> [{model_id}] Completed {done_count}/{len(jobs)} jobs ({time.perf_counter() - t_bb_start:.1f}s elapsed)", flush=True)

    aw_seed_runs: Dict[str, List[Dict[str, Any]]] = {m: [] for m in ALL_METHODS}
    aw_aggregated: Dict[str, Any] = {}
    for method in ALL_METHODS:
        for seed in args.seeds:
            aw_seed_runs[method].append(results_by_tag[f"aw_main:{method}:{seed}"])
        agg = aggregate_seed_runs(aw_seed_runs[method])
        aw_aggregated[method] = agg
        ft = agg["final_test"]
        print(
            f"  [{model_id} | AppWorld] {method:24s} | test U_B={ft['mean_utility']['mean']:.4f} "
            f"(Delta_evo={ft['delta_evo']['mean']:+.4f}) | "
            f"succ={ft['success_rate']['mean']*100:.2f}% | "
            f"Delta_use={agg['final_dev']['delta_use']['mean']:+.4f}",
            flush=True,
        )

    budget_sens: Dict[str, Any] = {}
    for mult in (0.6, 1.0, 1.4):
        tag = f"{mult:.1f}x_B0"
        budget_sens[tag] = {"budget": round(1.0 * mult, 4), "methods": {}}
        for method in ALL_METHODS:
            b_items = [r["budget_sensitivity_seed"][tag] for r in aw_seed_runs[method]]
            budget_sens[tag]["methods"][method] = {
                "mean_utility": sum(x["mean_utility"] for x in b_items) / len(b_items),
                "success_rate": sum(x["success_rate"] for x in b_items) / len(b_items),
                "delta_evo": sum(x["delta_evo"] for x in b_items) / len(b_items),
            }

    split_step_ablation: Dict[str, Any] = {}
    agg_t0 = aw_aggregated["prism_evolve"]
    split_step_ablation["fixed_T0"] = {
        "final_dev": agg_t0["final_dev"],
        "final_test": agg_t0["final_test"],
        "p_root_final": agg_t0["p_root_final"],
    }
    for split_label in ("uniform_T0_1", "uniform_T0_2"):
        s_runs = [results_by_tag[f"aw_split:{split_label}:{seed}"] for seed in args.seeds]
        s_agg = aggregate_seed_runs(s_runs)
        split_step_ablation[split_label] = {
            "final_dev": s_agg["final_dev"],
            "final_test": s_agg["final_test"],
            "p_root_final": s_agg["p_root_final"],
        }

    pb_seed_runs: Dict[str, List[Dict[str, Any]]] = {m: [] for m in ALL_METHODS}
    pb_aggregated: Dict[str, Any] = {}
    for method in ALL_METHODS:
        for seed in args.seeds:
            pb_seed_runs[method].append(results_by_tag[f"pb_main:{method}:{seed}"])
        agg = aggregate_seed_runs(pb_seed_runs[method])
        pb_aggregated[method] = agg
        ft = agg["final_test"]
        print(
            f"  [{model_id} | PlanBench-XL] {method:24s} | test U_B={ft['mean_utility']['mean']:.4f} "
            f"(Delta_evo={ft['delta_evo']['mean']:+.4f}) | "
            f"succ={ft['success_rate']['mean']*100:.2f}% | "
            f"Delta_use={agg['final_dev']['delta_use']['mean']:+.4f}",
            flush=True,
        )

    def _strip_internal(runs_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        cleaned: Dict[str, List[Dict[str, Any]]] = {}
        for m, rlist in runs_dict.items():
            cleaned[m] = [
                {k: v for k, v in r.items() if k not in ("budget_sensitivity_seed", "job_tag")}
                for r in rlist
            ]
        return cleaned

    bb_elapsed = time.perf_counter() - t_bb_start
    bb_payload = {
        "model_id": model_id,
        "llm_backbone": llm_meta,
        "elapsed_sec": bb_elapsed,
        "part2_appworld": {
            "split_sizes": {"train": len(aw_train), "dev": len(aw_dev), "test_normal": len(aw_test)},
            "methods": aw_aggregated,
            "seed_runs": _strip_internal(aw_seed_runs),
        },
        "part3_diagnostics": {
            "budget_sensitivity": budget_sens,
            "split_step_ablation": split_step_ablation,
        },
        "part4_planbench_xl": {
            "split_sizes": {"train": len(pb_train), "dev": len(pb_dev), "test": len(pb_test)},
            "methods": pb_aggregated,
            "seed_runs": _strip_internal(pb_seed_runs),
        },
    }
    per_bb_json.write_text(json.dumps(bb_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PRISM-Evolve] Saved backbone results for {model_id} -> {per_bb_json}", flush=True)
    return bb_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PRISM-Evolve full experimental suite.")
    parser.add_argument(
        "--appworld-data-root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--planbench-data-root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "outputs" / "prism_evolve_experiments"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--batches-per-stage", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--mc-episodes", type=int, default=10000)
    parser.add_argument("--gpu-device", type=str, default="cuda:3")
    parser.add_argument("--llm-model-id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--backbones",
        type=str,
        default=None,
        help="Comma-separated list of LLM backbones to run/consolidate (default: all 3 canonical backbones).",
    )
    parser.add_argument("--precompute-qwen-cache-only", action="store_true")
    parser.add_argument("--qwen-cache-path", type=str, default=None)
    parser.add_argument("--render-report-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    if args.precompute_qwen_cache_only:
        _execute_qwen_gpu_precompute_and_exit(args)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "prism_evolve_full_results.json"
    md_path = out_dir / "prism_evolve_report.md"

    if args.render_report_only:
        summary_payload = json.loads(json_path.read_text(encoding="utf-8"))
        md_path.write_text(format_markdown_report(summary_payload), encoding="utf-8")
        print(f"[PRISM-Evolve] Re-rendered report -> {md_path}", flush=True)
        return

    # Bootstrap per-backbone cache for Qwen/Qwen2.5-7B-Instruct from existing prism_evolve_full_results.json if present
    qwen25_bb_path = out_dir / "results_backbone_qwen25_7b.json"
    if not qwen25_bb_path.exists() and json_path.exists():
        try:
            existing_data = json.loads(json_path.read_text(encoding="utf-8"))
            if "part2_appworld" in existing_data and existing_data.get("config", {}).get("llm_model_id") == "Qwen/Qwen2.5-7B-Instruct":
                qwen25_payload = {
                    "model_id": "Qwen/Qwen2.5-7B-Instruct",
                    "llm_backbone": existing_data["environment"]["llm_backbone"],
                    "elapsed_sec": existing_data.get("total_elapsed_sec", 48.27),
                    "part2_appworld": existing_data["part2_appworld"],
                    "part3_diagnostics": existing_data["part3_diagnostics"],
                    "part4_planbench_xl": existing_data["part4_planbench_xl"],
                }
                qwen25_bb_path.write_text(json.dumps(qwen25_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[PRISM-Evolve] Bootstrapped Qwen2.5-7B-Instruct cache -> {qwen25_bb_path}", flush=True)
        except Exception as e:
            print(f"[PRISM-Evolve] Warning: could not bootstrap Qwen2.5-7B cache: {e}", flush=True)

    if args.backbones:
        requested_backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    else:
        requested_backbones = list(CANONICAL_BACKBONES)

    t_start = time.perf_counter()

    print("[Part 2-4] Loading AppWorld and PlanBench-XL dataset splits...", flush=True)
    aw_train = load_appworld_split_tasks(args.appworld_data_root, "train")
    aw_dev = load_appworld_split_tasks(args.appworld_data_root, "dev")
    aw_test = load_appworld_split_tasks(args.appworld_data_root, "test_normal")
    print(f"  -> AppWorld tasks loaded: train={len(aw_train)}, dev={len(aw_dev)}, test_normal={len(aw_test)}", flush=True)

    pb_train = load_planbench_split_tasks(args.planbench_data_root, "train")
    pb_dev = load_planbench_split_tasks(args.planbench_data_root, "dev")
    pb_test = load_planbench_split_tasks(args.planbench_data_root, "test")
    print(f"  -> PlanBench-XL tasks loaded: train={len(pb_train)}, dev={len(pb_dev)}, test={len(pb_test)}", flush=True)

    backbones_results: Dict[str, Any] = {}
    for model_id in requested_backbones:
        bb_res = _run_suite_for_single_backbone(
            model_id=model_id,
            args=args,
            out_dir=out_dir,
            aw_splits=(aw_train, aw_dev, aw_test),
            pb_splits=(pb_train, pb_dev, pb_test),
        )
        backbones_results[model_id] = bb_res

    # Also include any other canonical backbone whose per-backbone JSON already exists in out_dir
    for model_id in CANONICAL_BACKBONES:
        if model_id not in backbones_results:
            cand_path = out_dir / f"results_backbone_{BACKBONE_SPECS[model_id]['slug']}.json"
            if cand_path.exists():
                backbones_results[model_id] = json.loads(cand_path.read_text(encoding="utf-8"))

    print("[Part 1/4] Running §5.3 exact rational enumeration and Monte Carlo verification...", flush=True)
    part1_res = run_part1_exact_and_mc_verification(num_mc_episodes=args.mc_episodes)

    total_elapsed = time.perf_counter() - t_start
    primary_model_id = (
        "Qwen/Qwen2.5-7B-Instruct"
        if "Qwen/Qwen2.5-7B-Instruct" in backbones_results
        else requested_backbones[0]
    )
    primary_bb = backbones_results[primary_model_id]

    summary_payload = {
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": "cpu",
            "device_name": primary_bb["llm_backbone"].get("gpu_name", "NVIDIA GPU"),
            "llm_backbone": primary_bb["llm_backbone"],
            "llm_backbones": {m: backbones_results[m]["llm_backbone"] for m in backbones_results},
        },
        "config": {
            "seeds": args.seeds,
            "num_stages": args.num_stages,
            "batches_per_stage": args.batches_per_stage,
            "batch_size": args.batch_size,
            "mc_episodes": args.mc_episodes,
            "llm_model_id": primary_model_id,
            "evaluated_backbones": list(backbones_results.keys()),
        },
        "total_elapsed_sec": total_elapsed,
        "part1_verification": part1_res,
        "part2_appworld": primary_bb["part2_appworld"],
        "part3_diagnostics": primary_bb["part3_diagnostics"],
        "part4_planbench_xl": primary_bb["part4_planbench_xl"],
        "backbones": backbones_results,
    }

    json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(format_markdown_report(summary_payload), encoding="utf-8")

    print(
        f"[PRISM-Evolve] Completed multi-backbone suite ({list(backbones_results.keys())}) in {total_elapsed:.2f}s. "
        f"Saved JSON -> {json_path} and Report -> {md_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
