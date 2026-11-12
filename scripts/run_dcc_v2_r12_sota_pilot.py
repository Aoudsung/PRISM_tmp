#!/usr/bin/env python3
"""PRISM DCC v2 (r12 / unified-r5) lean verification and benchmark performance pilot.

Eliminates redundant ablation sweeps and focuses strictly on:
  1. Stage 1 (~10s): Exact single-prefill `DynamicCache.crop` KV-cache parity & physical token
     accounting, pure state-only tail Bellman neutrality, and open-argument (`search[...]`) vs
     closed-atom (`click[search]`) prior decomposition check.
  2. Stage 2: Direct head-to-head benchmark performance (`frozen_base` vs `prism_evolve`) on
     `WebShop`, `ALFWorld`, and `LOCA` under official `U_B = Y * max(0, 1 - C_seconds / B_seconds)`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Any, Mapping, Sequence

import torch


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _git_commit(repo_root: Path) -> str:
    env = dict(os.environ, GIT_DISCOVERY_ACROSS_FILESYSTEM="1")
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), env=env, capture_output=True, text=True, check=False
    )
    return res.stdout.strip() or "unknown"


def run_stage1_kernel_checks(
    policy: Any,
    host_cfg: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Fast (~10s) remote verification of r5 DynamicCache.crop KV reuse, Bellman neutrality, and WebShop search prior."""
    from prism_evolve.native_core import BudgetLimits, BudgetMeter
    from prism_evolve.native_dcc import (
        ASTNode,
        ExecutableProgram,
        PublicBenchmarkBinder,
    )
    from prism_evolve.native_runner import _environment

    stage1_dir = out_dir / "stage1_r5_checks"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    # 1. Analytical Bellman neutrality at phi_0 across depths 1, 2, 3
    view0 = policy.dcc_ensemble.active_model_view(use_initial_params=True)
    b0 = view0.reset_intra_task_belief(device=policy.device)
    q_depths: dict[str, float] = {}
    for d in (1, 2, 3):
        ast_nodes = tuple([ASTNode(op="Call", tool_or_action="look") for _ in range(d)] + [ASTNode(op="Stop")])
        prog = ExecutableProgram(
            program_id=f"neutral_d{d}",
            benchmark="alfworld",
            step_templates=tuple("look" for _ in range(d)),
            readout_fields=(),
            ast_nodes=ast_nodes,
        )
        res_q = view0.evaluate_candidate_q(
            goal_text="examine the bowl with the desklamp.",
            public_fields={"room_receptacles": "desk 1", "target_object_visible": "unknown"},
            belief=b0,
            candidate_action="look",
            program=prog,
            benchmark="alfworld",
            remaining_budget_ratio=1.0,
            remaining_steps_ratio=1.0,
            step_budget_fraction=0.04,
            prior_action_logprob=0.0,
        )
        q_depths[f"depth_{d}"] = float(res_q["q_mean"])
    report["phi0_bellman_neutrality"] = {
        **q_depths,
        "max_spread": max(q_depths.values()) - min(q_depths.values()),
    }

    # 2. Single-prefill DynamicCache.crop numerical parity + physical forward token billing + WebShop search prior
    ws_cfg = dict(host_cfg, episode_max_steps=15, max_steps=15)
    ws_env = _environment(ws_cfg, "webshop", stage1_dir / "env_ws_check")
    try:
        ws_task = ws_env.list_tasks("dev")[0]
        ctx0 = ws_env.reset(str(ws_task["task_id"]), seed=42)
        goal = str(ctx0.get("instruction", ws_task.get("instruction", "")))
        msgs = [
            {"role": "system", "content": str(ctx0.get("system_prompt", ""))},
            {"role": "user", "content": json.dumps({"feedback_type": "task", "instruction": goal, "observation": ctx0["observation"]}, ensure_ascii=False)},
        ]
        schema = ws_env.schema_for_history(msgs)
        bindings = PublicBenchmarkBinder.extract_role_bindings("webshop", goal, (), schema)
        cands = PublicBenchmarkBinder.enumerate_admissible_actions("webshop", schema, goal, bindings)

        # Fresh prefill scoring
        policy._retained_kv_cache = None
        policy._retained_prompt_ids = ()
        m_fresh = BudgetMeter(BudgetLimits(10, 100000, 60.0))
        old_meter = policy.meter
        try:
            policy.meter = m_fresh
            raw_fresh = policy.score_frozen_action_logprobs(msgs, schema, cands, return_raw_logprobs=True)
            adv_fresh = policy.score_frozen_action_logprobs(msgs, schema, cands, return_raw_logprobs=False)

            # Retained KV-cache scoring (after 1 prior generation)
            m_reuse = BudgetMeter(BudgetLimits(10, 100000, 60.0))
            policy.meter = m_reuse
            with policy.frozen_reference():
                p_gen = policy._generate_native_text(
                    msgs, schema, mode=None, greedy=True, role="actor.action", max_new_tokens=64, retain_kv_cache=True
                )
            toks_after_gen = int(m_reuse.total_model_tokens)
            raw_reused = policy.score_frozen_action_logprobs(
                msgs, schema, cands, prompt_ids=p_gen.prompt_ids, return_raw_logprobs=True
            )
            suffix_only_charged = int(m_reuse.total_model_tokens) - toks_after_gen
        finally:
            policy.meter = old_meter

        max_lp_diff = max(abs(raw_fresh[k] - raw_reused[k]) for k in raw_fresh) if raw_fresh else 0.0
        report["single_prefill_kv_crop_check"] = {
            "candidates": cands,
            "goal_query_bound": bindings.get("goal_query", ""),
            "raw_logprobs_fresh": {k: round(v, 5) for k, v in raw_fresh.items()},
            "raw_logprobs_reused": {k: round(v, 5) for k, v in raw_reused.items()},
            "centered_advantages": {k: round(v, 5) for k, v in adv_fresh.items()},
            "max_abs_logprob_diff": float(max_lp_diff),
            "fresh_charged_tokens": int(m_fresh.total_model_tokens) // 2,
            "reused_suffix_only_charged_tokens": int(suffix_only_charged),
            "prompt_tokens_len": len(p_gen.prompt_ids),
        }
    finally:
        ws_env.close()

    _write_json(stage1_dir / "stage1_summary.json", report)
    return report


def _evaluate_slice(
    policy: Any,
    env: Any,
    tasks_slice: Sequence[Mapping[str, Any]],
    eval_cfg: Mapping[str, Any],
    ckpt_dir: Path,
    method_name: str,
    eval_out_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    from prism_evolve.native_core import BudgetLimits, BudgetMeter
    from prism_evolve.native_training import public_trace, rollout

    eval_out_dir.mkdir(parents=True, exist_ok=True)
    policy.load_adapter(ckpt_dir)
    policy.zero_grad(set_to_none=True)

    records = []
    prior_overrides = 0
    program_attributed_steps = 0
    total_dcc_steps = 0
    stagnant_self_loops = 0

    for t_idx, raw_task in enumerate(tasks_slice):
        task = dict(raw_task)
        actor_seed = seed * 1000003 + t_idx
        task["environment_seed"] = actor_seed
        torch.manual_seed(actor_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actor_seed)

        ep_meter = BudgetMeter(
            BudgetLimits(
                int(eval_cfg["episode_max_steps"]),
                int(eval_cfg["episode_max_model_tokens"]),
                float(eval_cfg["episode_max_wall_seconds"]),
            ),
            event_path=eval_out_dir / "costs" / f"task-{t_idx:05d}.jsonl",
        )
        if method_name == "frozen_base":
            with policy.frozen_reference():
                traj = rollout(policy, env, task, eval_cfg, ep_meter, seed=actor_seed, method="frozen_base", training=False)
        else:
            traj = rollout(policy, env, task, eval_cfg, ep_meter, seed=actor_seed, method=method_name, training=False)

        _write_json(eval_out_dir / "traces" / f"task-{t_idx:05d}.json", public_trace(traj, method=method_name))

        prev_act = None
        for ev in traj.events:
            act_txt = str(ev.generation.text).strip()
            if prev_act is not None and act_txt == prev_act:
                stagnant_self_loops += 1
            prev_act = act_txt
            dcc_dec = ev.generation.metadata.get("dcc_decision")
            if isinstance(dcc_dec, Mapping):
                total_dcc_steps += 1
                if dcc_dec.get("program_attributed"):
                    program_attributed_steps += 1
                p_act = str(dcc_dec.get("prior_action", "")).strip()
                e_act = str(dcc_dec.get("executed_concrete_action", "")).strip()
                if p_act and e_act and p_act != e_act:
                    prior_overrides += 1

        records.append({
            "task_index": t_idx,
            "task_id": str(task["task_id"]),
            "success": int(bool(traj.success)),
            "utility_u_b": float(traj.utility),
            "elapsed_seconds": float(traj.elapsed_seconds),
            "env_steps": len(traj.events),
            "model_tokens": int(ep_meter.total_model_tokens),
            "termination": str(traj.termination),
            "actions": [str(e.generation.text).strip() for e in traj.events[:10]],
        })

    n_t = max(1, len(records))
    summary = {
        "method": method_name,
        "tasks": len(records),
        "successes": sum(r["success"] for r in records),
        "success_rate": sum(r["success"] for r in records) / float(n_t),
        "mean_utility_u_b": sum(r["utility_u_b"] for r in records) / float(n_t),
        "mean_elapsed_seconds": sum(r["elapsed_seconds"] for r in records) / float(n_t),
        "mean_env_steps": sum(r["env_steps"] for r in records) / float(n_t),
        "total_model_tokens": sum(r["model_tokens"] for r in records),
        "consecutive_repeat_steps": stagnant_self_loops,
        "dcc_steps_total": total_dcc_steps,
        "prior_action_overrides": prior_overrides,
        "program_attributed_steps": program_attributed_steps,
        "per_task": records,
    }
    _write_json(eval_out_dir / "eval_summary.json", summary)
    return summary


def _summarize_train_jsonl(t_dir: Path) -> dict[str, Any]:
    jpath = t_dir / "training.jsonl"
    if not jpath.is_file():
        return {}
    rows = [json.loads(line) for line in jpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    d_reals = [r.get("dcc_d_real") for r in rows if r.get("dcc_d_real") is not None]
    ckeys = [str(r.get("dcc_comparison_key", "")) for r in rows if r.get("dcc_comparison_key")]
    key_counts = Counter(ckeys)
    modes = Counter(str(r.get("dcc_fork_selection_mode", "")) for r in rows if r.get("dcc_fork_selection_mode"))
    nlls = [float(r["dcc_side_loss"]) for r in rows if r.get("dcc_side_loss") is not None]
    return {
        "blocks_completed": len(rows),
        "first_3_losses": [round(x, 4) for x in nlls[:3]],
        "last_3_losses": [round(x, 4) for x in nlls[-3:]],
        "nonzero_d_real_blocks": sum(1 for x in d_reals if abs(float(x)) > 1e-6),
        "fork_selection_mode_counts": dict(modes),
        "distinct_comparison_keys": len(key_counts),
        "reused_comparison_keys": sum(1 for v in key_counts.values() if v >= 2),
        "branch_success_pairs": [(r.get("first_success"), r.get("second_success")) for r in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-config", type=Path, default=Path("configs/prism-native-host-a.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, default="/apps/users/adminuser/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-groups", type=int, default=10)
    parser.add_argument("--eval-tasks", type=int, default=16)
    args = parser.parse_args()

    out_dir: Path = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    host_cfg = json.loads(args.host_config.read_text(encoding="utf-8"))
    external_root = Path(host_cfg["external_root"])
    host_cfg["official_repos"] = {
        k: str(Path(v) if Path(v).is_absolute() else external_root / v)
        for k, v in host_cfg.get("official_repos", {}).items()
    }

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_per_process_memory_fraction(0.80, 0)

    from prism_evolve.native_core import BudgetLimits, BudgetMeter
    from prism_evolve.native_dcc import DCC_CONTRACT_VERSION
    from prism_evolve.native_policy import NativeCausalPolicy
    from prism_evolve.native_runner import _environment
    from prism_evolve.native_training import run_training

    resolved_manifest = {
        "git_commit": _git_commit(repo_root),
        "dcc_contract_version": DCC_CONTRACT_VERSION,
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "backbone": "Qwen/Qwen2.5-7B-Instruct",
        "model_path": args.model_path,
        "seed": args.seed,
        "train_groups": args.train_groups,
        "eval_tasks": args.eval_tasks,
        "methods": ["frozen_base", "prism_evolve"],
    }
    _write_json(out_dir / "resolved_manifest.json", resolved_manifest)

    policy = NativeCausalPolicy(
        args.model_path,
        device="cuda:0",
        lora_rank=16,
        lora_alpha=32.0,
        max_new_tokens=256,
        max_context_tokens=32768,
        chat_template_kwargs={"enable_thinking": False},
        activation_offload_cpu=True,
        activation_offload_min_bytes=1048576,
    )
    init_adapter_dir = out_dir / "initial_adapter"
    policy.save_adapter(init_adapter_dir)

    print("[Stage 1] Running r5 kernel checks...", flush=True)
    stage1 = run_stage1_kernel_checks(policy, host_cfg, out_dir)
    print("[Stage 1] Done:", json.dumps(stage1, ensure_ascii=False), flush=True)

    benchmarks_summary: dict[str, Any] = {}
    for bench, ep_steps in [("webshop", 15), ("alfworld", 25)]:
        print(f"[Stage 2] Benchmark={bench} (train_groups={args.train_groups}, eval_tasks={args.eval_tasks})...", flush=True)
        bench_dir = out_dir / bench
        bench_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            **host_cfg,
            "benchmark": bench,
            "backbone": "Qwen/Qwen2.5-7B-Instruct",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "episode_max_steps": ep_steps,
            "max_steps": ep_steps,
            "episode_max_model_tokens": 131072,
            "episode_max_wall_seconds": 600.0,
            "max_new_tokens": 256,
            "max_train_groups": args.train_groups,
            "groups_per_update": 1,
            "dcc_max_probe_steps": 3,
            "dcc_continuation_steps": ep_steps,
            "dcc_full_tail_rollout": True,
            "dcc_proposal_budget": 3,
            "dcc_learning_rate": 0.001,
            "grad_clip_norm": 1.0,
            "evaluation_greedy": False,
            "dcc_controller_mode": "expected_utility",
            "dcc_fork_acquisition": "decision_disagreement",
            "dcc_cluster_updates_per_block": 4,
            "dcc_proposer_mode": "conditional_gain",
            "dcc_quantize_prompt_wall_seconds": True,
            "training_objective": "U_B=Y*max(0,1-C/B)",
        }
        env = _environment(cfg, bench, bench_dir / f"env_{bench}")
        try:
            train_tasks = env.list_tasks("train")
            dev_tasks = env.list_tasks("dev")[: args.eval_tasks]

            policy.max_new_tokens = 256
            policy.load_adapter(init_adapter_dir)
            meter = BudgetMeter(
                BudgetLimits(3200, 3600000, 3600.0),
                event_path=bench_dir / "training_costs.jsonl",
            )
            train_sum = run_training(
                "prism_evolve", policy, env, train_tasks, dev_tasks, cfg,
                meter, bench_dir / "training_prism_evolve", args.seed,
            )
            main_ckpt = Path(train_sum["selected_checkpoint"])

            eval_res: dict[str, Any] = {}
            for m_name, ckpt in [("frozen_base", init_adapter_dir), ("prism_evolve", main_ckpt)]:
                print(f"  [Eval {bench}] {m_name} ({len(dev_tasks)} tasks)...", flush=True)
                eval_res[m_name] = _evaluate_slice(
                    policy, env, dev_tasks, cfg, ckpt, m_name, bench_dir / f"eval_{m_name}", seed=args.seed
                )

            benchmarks_summary[bench] = {
                "training_usage": train_sum["usage"],
                "dcc_summary": {
                    k: v for k, v in train_sum.get("dcc_summary", {}).items() if k != "mechanism_ablations"
                },
                "train_diagnostics": _summarize_train_jsonl(bench_dir / "training_prism_evolve"),
                "eval_comparison": {
                    k: {mk: mv for mk, mv in v.items() if mk != "per_task"}
                    for k, v in eval_res.items()
                },
                "paired_outcomes": {
                    k: [(r["task_id"], r["success"], round(r["utility_u_b"], 4), r["env_steps"], r["actions"][:5]) for r in v["per_task"]]
                    for k, v in eval_res.items()
                },
            }
            _write_json(bench_dir / f"{bench}_summary.json", benchmarks_summary[bench])
            print(
                f"  [{bench} RESULT] frozen_base={eval_res['frozen_base']['success_rate']:.4f} "
                f"(util={eval_res['frozen_base']['mean_utility_u_b']:.4f}, repeats={eval_res['frozen_base']['consecutive_repeat_steps']}) "
                f"vs prism_evolve={eval_res['prism_evolve']['success_rate']:.4f} "
                f"(util={eval_res['prism_evolve']['mean_utility_u_b']:.4f}, repeats={eval_res['prism_evolve']['consecutive_repeat_steps']})",
                flush=True,
            )
        finally:
            env.close()

    # Fast LOCA QEMU closed-loop check (1 train block + 2 eval tasks)
    print("[LOCA] Running QEMU VM closed-loop benchmark check...", flush=True)
    loca_dir = out_dir / "loca"
    loca_dir.mkdir(parents=True, exist_ok=True)
    loca_cfg = {
        **host_cfg,
        "benchmark": "loca",
        "backbone": "Qwen/Qwen2.5-7B-Instruct",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "episode_max_steps": 6,
        "max_steps": 6,
        "episode_max_model_tokens": 131072,
        "episode_max_wall_seconds": 600.0,
        "max_new_tokens": 512,
        "max_train_groups": 1,
        "groups_per_update": 1,
        "dcc_max_probe_steps": 2,
        "dcc_continuation_steps": 6,
        "dcc_full_tail_rollout": True,
        "dcc_proposal_budget": 2,
        "dcc_learning_rate": 0.001,
        "grad_clip_norm": 1.0,
        "evaluation_greedy": False,
        "dcc_controller_mode": "expected_utility",
        "dcc_fork_acquisition": "decision_disagreement",
        "dcc_cluster_updates_per_block": 4,
        "dcc_proposer_mode": "conditional_gain",
        "dcc_quantize_prompt_wall_seconds": True,
        "training_objective": "U_B=Y*max(0,1-C/B)",
    }
    policy.max_new_tokens = 512
    policy.load_adapter(init_adapter_dir)
    loca_env = _environment(loca_cfg, "loca", loca_dir / "env_loca")
    try:
        l_train = loca_env.list_tasks("train")
        l_dev = loca_env.list_tasks("dev")[:2]
        l_meter = BudgetMeter(BudgetLimits(200, 600000, 1200.0), event_path=loca_dir / "training_costs.jsonl")
        l_train_sum = run_training(
            "prism_evolve", policy, loca_env, l_train, l_dev, loca_cfg,
            l_meter, loca_dir / "training_prism_evolve", args.seed,
        )
        l_eval = {}
        for lm_name, lm_ckpt in [("frozen_base", init_adapter_dir), ("prism_evolve", Path(l_train_sum["selected_checkpoint"]))]:
            l_eval[lm_name] = _evaluate_slice(
                policy, loca_env, l_dev, loca_cfg, lm_ckpt, lm_name, loca_dir / f"eval_{lm_name}", seed=args.seed
            )
        benchmarks_summary["loca"] = {
            "complete_groups": l_train_sum["complete_groups"],
            "gradient_steps": l_train_sum["gradient_steps"],
            "training_usage": l_train_sum["usage"],
            "eval_comparison": {
                k: {mk: mv for mk, mv in v.items() if mk != "per_task"}
                for k, v in l_eval.items()
            },
        }
    finally:
        loca_env.close()

    final_payload = {
        "resolved_manifest": resolved_manifest,
        "stage1_r5_checks": stage1,
        "benchmarks": benchmarks_summary,
    }
    _write_json(out_dir / "r12_sota_pilot_summary.json", final_payload)
    print("[COMPLETE] Wrote summary to", out_dir / "r12_sota_pilot_summary.json", flush=True)


if __name__ == "__main__":
    main()
