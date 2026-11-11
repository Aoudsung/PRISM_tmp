#!/usr/bin/env python3
"""PRISM DCC v2 (r11) staged remote verification, matched-budget acquisition/replay pilot,
and Round 3 fixed-D / four-cell structure attribution driver.

Executes the three-stage experimental protocol on a designated remote GPU:
  Stage 1 (`round1_existing_data_replay`):
    - 1A: Single-decision snapshot replay on existing `r10p-v3-gpu4` ALFWorld `task-00002` Step 1,
      isolating sub-millisecond wall-clock prompt jitter vs true alias equivalence under unified
      `DecisionContext` and quantized prompt wall-seconds.
    - 1B: Paid fork-cluster NLL replay on existing `r10p-v3-gpu4` clusters (`1 update/block` vs
      `4 updates/block`), recording readout vs upstream gradient startup and accurate pre/post
      deployment `d_model` (`u -> pi_0` vs `v -> pi_0`).
  Stage 2 (`round2_acquisition_and_replay_comparison`):
    - Matched-budget live native training on `WebShop` and `ALFWorld` comparing legacy
      `lexicographic + 1 update/block` against `decision_disagreement R_h(u,v) + 4 updates/block`
      under the pure `expected_utility` controller (`argmax_rho Q_bar^rho`, full finite candidate
      support, Bellman-neutral `phi_0`), holding `witness_hard` proposer fixed.
  Stage 3 (`round3_structure_and_four_cell_attribution`):
    - Paired dev-set evaluation across `frozen_base`, the four-cell matrix
      `(phi_0/phi_T) x (Xi_0/Xi_T)`, `dcc_fixed_d_shuffled`, and fixed-D `conditional_gain`
      (`Gamma(P; u, v)` vs `conditional_gain_shuffled`), plus `LOCA` QEMU VM closed-loop check.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any, Mapping

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


def run_stage1_existing_data_replay(
    policy: Any,
    host_a_cfg: Mapping[str, Any],
    prev_run_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Replay existing paid artifacts from `r10p-v3-gpu4` without new environment rollouts."""
    from prism_evolve.native_dcc import (
        ControlledResponseEnsemble,
        DecisionContext,
        PublicBenchmarkBinder,
        PublicTransition,
        TailExample,
        _deterministic_seed,
    )
    from prism_evolve.native_runner import _environment

    stage1_dir = out_dir / "stage1_existing_data_replay"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    # --- 1A. Single-Decision Snapshot Alias Replay on ALFWorld task-00002 Step 1 ---
    alf_prev = prev_run_dir / "alfworld"
    t2_full_path = alf_prev / "eval_prism_evolve/traces/task-00002.json"
    t2_nogr_path = alf_prev / "eval_dcc_no_program_growth/traces/task-00002.json"
    final_ckpt = alf_prev / "training_prism_evolve/final_adapter"
    init_ckpt = prev_run_dir / "initial_adapter"

    if t2_full_path.is_file() and t2_nogr_path.is_file() and final_ckpt.is_dir():
        t2_full = json.loads(t2_full_path.read_text(encoding="utf-8"))
        t2_nogr = json.loads(t2_nogr_path.read_text(encoding="utf-8"))
        ev0_full = t2_full["events"][0]
        ev1_full = t2_full["events"][1]
        ev1_nogr = t2_nogr["events"][1]

        # Reconstruct live ALFWorld environment at task-00002 Step 1 (`1` env step replay)
        alf_cfg = dict(host_a_cfg, episode_max_steps=25, max_steps=25)
        env = _environment(alf_cfg, "alfworld", stage1_dir / "env_alfworld_replay")
        try:
            dev_tasks = env.list_tasks("dev")
            task_2 = dev_tasks[2]
            actor_seed = 42 * 1000003 + 2
            ctx0 = env.reset(str(task_2["task_id"]), seed=actor_seed)
            step0_act = ev0_full["action"]
            env.step(step0_act)

            msgs_full_raw = list(ev1_full["messages_before"])
            msgs_nogr_raw = list(ev1_nogr["messages_before"])
            schema_s1 = env.schema_for_history(msgs_full_raw)
            goal_text = str(ctx0.get("instruction", task_2.get("instruction", "")))

            # Build quantized-wall-second shared messages (`599.521...` -> `600` or `int(round(...))`)
            shared_quant_msgs = list(msgs_full_raw[:-1])
            last_budget_obj = json.loads(msgs_full_raw[-1]["content"])
            last_budget_obj["remaining_wall_seconds"] = int(round(float(last_budget_obj["remaining_wall_seconds"])))
            shared_quant_msgs.append(
                {"role": "user", "content": json.dumps(last_budget_obj, sort_keys=True, separators=(",", ":"))}
            )

            # 1) Confirm raw unquantized timestamps reproduce the prompt-token divergence
            ids_raw_a = policy._tokenize_messages(policy._text_action_messages(msgs_full_raw, schema_s1, None))
            ids_raw_b = policy._tokenize_messages(policy._text_action_messages(msgs_nogr_raw, schema_s1, None))

            # 2) Replay identical quantized snapshot across phi_T aliases (`full`, `no_program_growth`, `fixed_d_shuffled`)
            alias_outputs: dict[str, Any] = {}
            shuf_ckpt = alf_prev / "training_prism_evolve/fixed_d_shuffled_adapter"
            for alias_name, ckpt_dir, variant in [
                ("prism_evolve", final_ckpt, "full"),
                ("dcc_no_program_growth", final_ckpt, "no_program_growth"),
                ("dcc_fixed_d_shuffled", shuf_ckpt if shuf_ckpt.is_dir() else final_ckpt, "full"),
            ]:
                policy.load_adapter(ckpt_dir)
                policy.dcc_mode_enabled = True
                policy.dcc_controller_mode = "expected_utility"
                policy.dcc_eval_variant = variant
                policy.reset_dcc_episode("alfworld", task_instruction=goal_text, member_index=None)
                # Replay step-0 belief update identically
                policy.observe_dcc_transition(
                    "alfworld",
                    goal_text,
                    [],
                    schema_s1,
                    actual_action=str(step0_act.get("text", "")),
                    remaining_budget_ratio=1.0,
                    spent_seconds_ratio=0.0,
                    remaining_steps_ratio=1.0,
                )
                torch.manual_seed(actor_seed + 101)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(actor_seed + 101)
                gen = policy.act(
                    shared_quant_msgs,
                    schema_s1,
                    forced_mode=None,
                    greedy=False,
                    role="actor",
                    events=[],
                    task_instruction=goal_text,
                    remaining_budget_ratio=599.52 / 600.0,
                    remaining_steps_ratio=24.0 / 25.0,
                    max_env_steps=25,
                )
                dcc_dec = gen.metadata.get("dcc_decision", {})
                alias_outputs[alias_name] = {
                    "action": gen.text.strip(),
                    "prompt_ids_hash": hash(tuple(gen.prompt_ids)),
                    "selected_q_mean": dcc_dec.get("selected_q_mean"),
                    "candidates_evaluated": dcc_dec.get("planning_decision", {}).get("planning_compute_cost", {}).get("candidates_evaluated"),
                }

            # 3) Replay phi_0 aliases (`dcc_k0_planner` vs `dcc_reset_parameters`)
            phi0_outputs: dict[str, Any] = {}
            for alias_name, ckpt_dir, variant in [
                ("dcc_k0_planner", init_ckpt, "k0_planner"),
                ("dcc_reset_parameters", final_ckpt, "reset_parameters"),
            ]:
                policy.load_adapter(ckpt_dir)
                policy.dcc_mode_enabled = True
                policy.dcc_controller_mode = "expected_utility"
                policy.dcc_eval_variant = variant
                policy.reset_dcc_episode("alfworld", task_instruction=goal_text, member_index=None)
                torch.manual_seed(actor_seed + 202)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(actor_seed + 202)
                gen0 = policy.act(
                    shared_quant_msgs,
                    schema_s1,
                    forced_mode=None,
                    greedy=False,
                    role="actor",
                    events=[],
                    task_instruction=goal_text,
                    remaining_budget_ratio=599.52 / 600.0,
                    remaining_steps_ratio=24.0 / 25.0,
                    max_env_steps=25,
                )
                dcc_dec0 = gen0.metadata.get("dcc_decision", {})
                phi0_outputs[alias_name] = {
                    "action": gen0.text.strip(),
                    "selected_q_mean": dcc_dec0.get("selected_q_mean"),
                }

            phi_t_actions = {v["action"] for v in alias_outputs.values()}
            phi_t_qs = {round(float(v["selected_q_mean"]), 7) for v in alias_outputs.values()}
            phi_0_actions = {v["action"] for v in phi0_outputs.values()}
            phi_0_qs = {round(float(v["selected_q_mean"]), 7) for v in phi0_outputs.values()}

            report["alias_single_snapshot_replay"] = {
                "historical_step1_wall_seconds": [
                    ev1_full["messages_before"][-1]["content"],
                    ev1_nogr["messages_before"][-1]["content"],
                ],
                "raw_float_prompt_tokens_identical": ids_raw_a == ids_raw_b,
                "quantized_snapshot_phi_T_aliases": alias_outputs,
                "phi_T_aliases_strictly_identical": (len(phi_t_actions) == 1 and len(phi_t_qs) == 1),
                "quantized_snapshot_phi_0_aliases": phi0_outputs,
                "phi_0_aliases_strictly_identical": (len(phi_0_actions) == 1 and len(phi_0_qs) == 1),
            }
        finally:
            env.close()

    # --- 1B. Paid-Cluster NLL Replay (`1 update/block` vs `4 updates/block`) on `r10p-v3-gpu4` ---
    cluster_replay_results: dict[str, Any] = {}
    tail_vec = policy.extract_frozen_action_embeddings(["frozen-prior-tail-v2"])[0]

    for bench in ("webshop", "alfworld"):
        traces_dir = prev_run_dir / bench / "training_prism_evolve/traces"
        if not traces_dir.is_dir():
            continue
        clusters = []
        for g_idx in range(4):
            tp = traces_dir / f"group-{g_idx:05d}-trajectory-0.json"
            if not tp.is_file():
                continue
            tdata = json.loads(tp.read_text(encoding="utf-8"))
            meta = tdata.get("metadata", {})
            raw_trs = meta.get("dcc_cluster_transitions", [])
            raw_tex = meta.get("dcc_tail_examples", [])
            rec_m = meta.get("dcc_measurement_record", {})
            if not raw_trs:
                continue
            max_steps_b = 15 if bench == "webshop" else 25
            trs = [
                PublicTransition(
                    transition_id=str(t["transition_id"]),
                    task_id=str(t["task_id"]),
                    episode_id=str(t["episode_id"]),
                    cluster_id=str(t["cluster_id"]),
                    benchmark=str(t["benchmark"]),
                    step_index=int(t["step_index"]),
                    goal_text=str(t["goal_text"]),
                    pre_history_ref=str(t["pre_history_ref"]),
                    pre_public_fields=dict(t["pre_public_fields"]),
                    action_text=str(t["action_text"]),
                    action_template=str(t["action_template"]),
                    role_bindings=dict(t["role_bindings"]),
                    raw_response=str(t["raw_response"]),
                    typed_response=dict(t["typed_response"]),
                    resource_delta={str(k): float(v) for k, v in dict(t["resource_delta"]).items()},
                    remaining_budget={str(k): float(v) for k, v in dict(t["remaining_budget"]).items()},
                    termination=str(t["termination"]),
                    native_outcome=t.get("native_outcome"),
                    parent_transition_id=t.get("parent_transition_id"),
                    resource_before={str(k): float(v) for k, v in dict(t.get("resource_before", {})).items()},
                    resource_after={str(k): float(v) for k, v in dict(t.get("resource_after", {})).items()},
                    frozen_context_vec=tuple(float(x) for x in t.get("frozen_context_vec", ())),
                    frozen_action_vec=tuple(float(x) for x in t.get("frozen_action_vec", ())),
                    controller_id=str(t.get("controller_id", "dcc")),
                )
                for t in raw_trs
            ]
            texs = [
                TailExample(
                    example_id=str(tx["example_id"]),
                    cluster_id=str(tx["cluster_id"]),
                    task_id=str(tx["task_id"]),
                    benchmark=str(tx["benchmark"]),
                    step_index=int(tx["step_index"]),
                    tail_policy_version=str(tx["tail_policy_version"]),
                    goal_text=str(tx["goal_text"]),
                    pre_public_fields=dict(tx["pre_public_fields"]),
                    resource_before={str(k): float(v) for k, v in dict(tx["resource_before"]).items()},
                    frozen_context_vec=tuple(float(x) for x in tx.get("frozen_context_vec", ())),
                    frozen_tail_action_vec=tail_vec,
                    step_budget_fraction=1.0 / float(max_steps_b),
                    observed_y=tx.get("observed_y"),
                    tail_seconds_cost_ratio=tx.get("tail_seconds_cost_ratio"),
                    total_seconds_cost_ratio=tx.get("total_seconds_cost_ratio"),
                    censored=bool(tx.get("censored", False)),
                    prefix_node_id=tx.get("prefix_node_id"),
                    episode_id=str(tx.get("episode_id", "")),
                )
                for tx in raw_tex
            ]
            cid = trs[0].cluster_id
            g_rng = torch.Generator(device="cpu").manual_seed(_deterministic_seed(42, cid))
            bw = torch.poisson(torch.ones(8, dtype=torch.float32), generator=g_rng)
            if float(bw.sum().item()) <= 0:
                bw = torch.ones(8, dtype=torch.float32)
            clusters.append({"cluster_id": cid, "transitions": trs, "tail_examples": texs, "bw": bw.to(policy.device), "rec": rec_m})

        if not clusters:
            continue

        arm_stats: dict[str, Any] = {}
        for arm_name, updates_per_block in [("single_step_1upd", 1), ("cluster_replay_4upd", 4)]:
            ens_test = ControlledResponseEnsemble().to(policy.device)
            opt_t = torch.optim.AdamW(ens_test.parameters(), lr=1e-3)
            buf = []
            grad_trace = []
            for c_item in clusters:
                buf.append(c_item)
                for u_i in range(updates_per_block):
                    chosen_c = c_item if u_i == 0 else buf[(len(buf) + u_i) % len(buf)]
                    opt_t.zero_grad(set_to_none=True)
                    loss_t, _, _ = ens_test.transition_negative_log_likelihood(
                        chosen_c["transitions"], tail_examples=chosen_c["tail_examples"], bootstrap_weights=chosen_c["bw"]
                    )
                    loss_t.backward()
                    up_g = math.sqrt(
                        sum(
                            float(p.grad.square().sum().item())
                            for n, p in ens_test.named_parameters()
                            if p.grad is not None and (n == "transition_logits" or n.startswith(("context_encoder.", "transition_residual_proj.")))
                        )
                    )
                    rd_g = math.sqrt(
                        sum(
                            float(p.grad.square().sum().item())
                            for n, p in ens_test.named_parameters()
                            if p.grad is not None and not (n == "transition_logits" or n.startswith(("context_encoder.", "transition_residual_proj.")))
                        )
                    )
                    torch.nn.utils.clip_grad_norm_(ens_test.parameters(), 1.0)
                    opt_t.step()
                    grad_trace.append(
                        {"step": len(grad_trace), "loss": round(float(loss_t.item()), 5), "grad_readout": round(rd_g, 6), "grad_upstream": round(up_g, 6)}
                    )

            # Evaluate final NLL across all 4 paid clusters + ensemble member std on last cluster
            with torch.no_grad():
                eval_losses = [
                    float(ens_test.transition_negative_log_likelihood(c["transitions"], tail_examples=c["tail_examples"])[0].item())
                    for c in clusters
                ]
                last_tr = clusters[-1]["transitions"][-1]
                b_root = ens_test.reset_intra_task_belief(device=policy.device)
                q_last = ens_test.evaluate_candidate_q(
                    goal_text=last_tr.goal_text,
                    public_fields=last_tr.pre_public_fields,
                    belief=b_root,
                    candidate_action=last_tr.action_text,
                    frozen_context_vec=last_tr.frozen_context_vec,
                    frozen_action_vec=last_tr.frozen_action_vec,
                    frozen_tail_action_vec=tail_vec,
                    remaining_budget_ratio=0.9,
                    remaining_steps_ratio=0.8,
                    step_budget_fraction=1.0 / (15.0 if bench == "webshop" else 25.0),
                )
            arm_stats[arm_name] = {
                "total_side_updates": len(grad_trace),
                "step0_gradients": grad_trace[0],
                "step1_gradients": grad_trace[1] if len(grad_trace) > 1 else None,
                "final_step_gradients": grad_trace[-1],
                "mean_cluster_nll_after": round(sum(eval_losses) / len(eval_losses), 5),
                "post_update_q_mean": round(float(q_last["q_mean"]), 5),
                "post_update_q_member_std": round(float(q_last["q_std"]), 6),
                "latent_emission_std": round(float(ens_test.emission_logits.std(dim=1).mean().item()), 6),
            }
        cluster_replay_results[bench] = arm_stats

    report["paid_cluster_nll_replay"] = cluster_replay_results
    _write_json(stage1_dir / "stage1_replay_summary.json", report)
    return report


def _evaluate_dev_slice(
    policy: Any,
    env: Any,
    dev_tasks: list[dict[str, Any]],
    eval_cfg: Mapping[str, Any],
    ckpt_dir: Path,
    method_name: str,
    eval_out_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    from prism_evolve.native_core import BudgetLimits, BudgetMeter
    from prism_evolve.native_training import public_trace, rollout

    eval_out_dir.mkdir(parents=True, exist_ok=True)
    policy.load_adapter(ckpt_dir)
    policy.zero_grad(set_to_none=True)

    # Map synthetic evaluation arms (`prism_evolve_lex_1upd`, `dcc_conditional_gain*`) to `"prism_evolve"` rollout contract
    rollout_method = (
        method_name
        if method_name in {
            "frozen_base", "prism_evolve", "dcc_k0_planner",
            "dcc_no_program_growth", "dcc_reset_parameters", "dcc_fixed_d_shuffled",
        }
        else "prism_evolve"
    )

    records = []
    prior_overrides = 0
    program_attributed_steps = 0
    total_dcc_steps = 0
    action_change_examples: list[dict[str, Any]] = []

    for t_idx, raw_task in enumerate(dev_tasks):
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
        if rollout_method == "frozen_base":
            with policy.frozen_reference():
                traj = rollout(policy, env, task, eval_cfg, ep_meter, seed=actor_seed, method="frozen_base", training=False)
        else:
            traj = rollout(policy, env, task, eval_cfg, ep_meter, seed=actor_seed, method=rollout_method, training=False)

        _write_json(eval_out_dir / "traces" / f"task-{t_idx:05d}.json", public_trace(traj, method=rollout_method))

        for s_i, ev in enumerate(traj.events):
            dcc_dec = ev.generation.metadata.get("dcc_decision")
            if isinstance(dcc_dec, Mapping):
                total_dcc_steps += 1
                if dcc_dec.get("program_attributed"):
                    program_attributed_steps += 1
                p_act = str(dcc_dec.get("prior_action", "")).strip()
                e_act = str(dcc_dec.get("executed_concrete_action", "")).strip()
                if p_act and e_act and p_act != e_act:
                    prior_overrides += 1
                    if len(action_change_examples) < 8:
                        action_change_examples.append({
                            "task_id": str(task["task_id"]),
                            "task_index": t_idx,
                            "step_index": s_i,
                            "public_fields": ev.metadata.get("dcc_pre_public_fields", {}),
                            "prior_action": p_act,
                            "executed_action": e_act,
                            "selected_program_id": dcc_dec.get("selected_program_id"),
                            "selected_program_origin": dcc_dec.get("selected_program_origin"),
                            "prior_q_mean": dcc_dec.get("prior_q_mean"),
                            "selected_q_mean": dcc_dec.get("selected_q_mean"),
                        })

        records.append({
            "task_index": t_idx,
            "task_id": str(task["task_id"]),
            "success": int(bool(traj.success)),
            "utility_u_b": float(traj.utility),
            "elapsed_seconds": float(traj.elapsed_seconds),
            "env_steps": len(traj.events),
            "model_tokens": int(ep_meter.total_model_tokens),
            "termination": str(traj.termination),
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
        "dcc_steps_total": total_dcc_steps,
        "prior_action_overrides": prior_overrides,
        "program_attributed_steps": program_attributed_steps,
        "per_task": records,
        "action_change_examples": action_change_examples,
    }
    _write_json(eval_out_dir / "eval_summary.json", summary)
    return summary


def _build_fixed_d_checkpoint(
    base_ckpt_dir: Path,
    target_ckpt_dir: Path,
    benchmark: str,
    records_json_path: Path,
    *,
    proposer_mode: str,
    shuffle_labels: bool,
    seed: int,
) -> dict[str, Any]:
    """Construct an offline fixed-D checkpoint sharing exact `phi_T` tensors with `base_ckpt_dir`
    while re-running `ActionContrastProposer` under `(proposer_mode, shuffle_labels)` on the exact same
    recorded `MeasurementRecord` sequence and budget.
    """
    from prism_evolve.native_dcc import ActionContrastProposer, MeasurementRecord, ProgramLibrary

    if target_ckpt_dir.exists():
        shutil.rmtree(target_ckpt_dir)
    shutil.copytree(base_ckpt_dir, target_ckpt_dir)

    raw_records = json.loads(records_json_path.read_text(encoding="utf-8")) if records_json_path.is_file() else []
    lib = ProgramLibrary(benchmark)
    prop = ActionContrastProposer(
        benchmark,
        max_probe_steps=3,
        proposal_budget=3,
        shuffle_witness_labels=shuffle_labels,
        proposer_mode=proposer_mode,
        seed=seed,
    )
    for idx, rdict in enumerate(raw_records):
        rec = MeasurementRecord(
            record_id=str(rdict["record_id"]),
            benchmark=str(rdict["benchmark"]),
            task_id=str(rdict["task_id"]),
            goal_category=str(rdict["goal_category"]),
            anchor_signature=str(rdict["anchor_signature"]),
            program_id=str(rdict["program_id"]),
            step_templates=tuple(rdict["step_templates"]),
            readout_fields=tuple(rdict["readout_fields"]),
            public_readout_z=dict(rdict["public_readout_z"]),
            z_parity_bit=int(rdict["z_parity_bit"]),
            continuation_pair_key=str(rdict["continuation_pair_key"]),
            u_score=float(rdict["u_score"]),
            v_score=float(rdict["v_score"]),
            u_cost=int(rdict["u_cost"]),
            v_cost=int(rdict["v_cost"]),
            u_done=bool(rdict["u_done"]),
            v_done=bool(rdict["v_done"]),
            action_diff_d=float(rdict["action_diff_d"]),
            outcome_bin=int(rdict["outcome_bin"]),
            surprise=float(rdict.get("surprise", 0.0)),
            rebound_trace_templates=tuple(rdict["rebound_trace_templates"]),
            rebound_trace_fields=tuple(rdict["rebound_trace_fields"]),
            post_q_checkpoint_id=str(rdict.get("post_q_checkpoint_id", "")),
            u_policy_id=str(rdict.get("u_policy_id", "u")),
            v_policy_id=str(rdict.get("v_policy_id", "v")),
            tail_policy_version=str(rdict.get("tail_policy_version", "frozen-prior-tail-v2")),
            factual_tail_ref=str(rdict.get("factual_tail_ref", "")),
            counterfactual_tail_ref=str(rdict.get("counterfactual_tail_ref", "")),
            u_native_outcome=rdict.get("u_native_outcome"),
            v_native_outcome=rdict.get("v_native_outcome"),
            d_real=rdict.get("d_real"),
            d_model=rdict.get("d_model"),
            d_source=str(rdict.get("d_source", "observed_native_terminal")),
            role_bindings={},
            comparison_key=str(rdict.get("comparison_key", "")),
            program_executed_cleanly=bool(rdict.get("program_executed_cleanly", True)),
        )
        prop.record_measurement(rec)
        prop.grow_library(lib, dataset_version=f"step_{idx+1}", model_version=f"step_{idx+1}")

    libs_path = target_ckpt_dir / "dcc_program_libraries.json"
    libs_payload = json.loads(libs_path.read_text(encoding="utf-8"))
    libs_payload[benchmark] = lib.to_dict()
    _write_json(libs_path, libs_payload)
    return {
        "proposer_mode": proposer_mode,
        "shuffle_labels": shuffle_labels,
        "library_size": len(lib.active_programs()),
        "added_programs": [
            {"program_id": p.program_id, "origin": p.origin, "steps": list(p.step_templates), "case_field": p.case_condition_field, "case_val": p.case_condition_value, "true_branch": p.case_true_template, "false_branch": p.case_false_template}
            for p in lib.active_programs()
            if p.origin != "starter"
        ],
        "unique_witness_pairs": prop.witness_pairs_found,
        "permutable_groups": prop.proposal_runs[-1].permutable_groups if prop.proposal_runs else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-config", type=Path, default=Path("configs/prism-native-host-a.json"))
    parser.add_argument("--prev-pilot-dir", type=Path,
                        default=Path("outputs/prism_evolve_pos_benchmarks/prism-qwen7b-dcc-v2-pilot-20261110-r10p-v3-gpu4"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, default="/apps/users/adminuser/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-groups", type=int, default=6)
    parser.add_argument("--dev-tasks", type=int, default=6)
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
        torch.cuda.set_per_process_memory_fraction(0.75, 0)

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
        "training_seed": args.seed,
        "loca_official_environment_seeds": [42, 43, 44, 45, 46],
        "train_groups_per_arm": args.train_groups,
        "dev_tasks_evaluated": args.dev_tasks,
        "controller_mode": "expected_utility",
        "quantize_prompt_wall_seconds": True,
        "benchmark_horizons": {"webshop": 15, "alfworld": 25, "loca": 8},
        "training_budget": {"max_env_steps": 1600, "max_model_tokens": 1800000, "max_wall_seconds": 2400},
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

    # === STAGE 1: Existing Data & Snapshot Replay ===
    print("[Stage 1] Running existing-data snapshot & cluster NLL replay...", flush=True)
    stage1_summary = run_stage1_existing_data_replay(
        policy, host_cfg, args.prev_pilot_dir.resolve(), out_dir
    )
    print("[Stage 1] Done:", json.dumps(stage1_summary.get("alias_single_snapshot_replay", {}), ensure_ascii=False), flush=True)

    # === STAGE 2 & STAGE 3: Matched-Budget Live Acquisition/Replay Pilot + 4-Cell & Fixed-D Attribution ===
    benchmark_results: dict[str, Any] = {}
    for bench, ep_steps in [("webshop", 15), ("alfworld", 25)]:
        print(f"[Stage 2/3] Starting benchmark={bench} (ep_steps={ep_steps}, groups={args.train_groups})...", flush=True)
        bench_dir = out_dir / bench
        bench_dir.mkdir(parents=True, exist_ok=True)

        base_b_cfg = {
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
            "dcc_proposer_mode": "witness_hard",
            "dcc_quantize_prompt_wall_seconds": True,
            "training_objective": "U_B=Y*max(0,1-C/B)",
        }

        env = _environment(base_b_cfg, bench, bench_dir / "env_shared")
        try:
            train_tasks = env.list_tasks("train")
            dev_tasks = env.list_tasks("dev")[: args.dev_tasks]

            # Arm 1: Legacy lexicographic fork pair + 1 update/block under pure expected_utility
            policy.load_adapter(init_adapter_dir)
            cfg_lex = {
                **base_b_cfg,
                "dcc_fork_acquisition": "lexicographic",
                "dcc_cluster_updates_per_block": 1,
            }
            meter_lex = BudgetMeter(
                BudgetLimits(1600, 1800000, 2400.0),
                event_path=bench_dir / "train_lex_1upd_costs.jsonl",
            )
            train_lex_summary = run_training(
                "prism_evolve", policy, env, train_tasks, dev_tasks, cfg_lex,
                meter_lex, bench_dir / "training_lex_1upd", args.seed,
            )

            # Arm 2: Decision-disagreement R_h(u,v) + 4-step cluster NLL replay (`prism_evolve`)
            policy.load_adapter(init_adapter_dir)
            cfg_dis = {
                **base_b_cfg,
                "dcc_fork_acquisition": "decision_disagreement",
                "dcc_cluster_updates_per_block": 4,
            }
            meter_dis = BudgetMeter(
                BudgetLimits(1600, 1800000, 2400.0),
                event_path=bench_dir / "train_disagreement_4upd_costs.jsonl",
            )
            train_dis_summary = run_training(
                "prism_evolve", policy, env, train_tasks, dev_tasks, cfg_dis,
                meter_dis, bench_dir / "training_prism_evolve", args.seed,
            )

            # Parse block-level acquisition & d_model metrics from training.jsonl for both arms
            def _summarize_train_jsonl(t_dir: Path) -> dict[str, Any]:
                jpath = t_dir / "training.jsonl"
                rows = [json.loads(line) for line in jpath.read_text(encoding="utf-8").splitlines() if line.strip()]
                d_reals = [r.get("dcc_d_real") for r in rows if r.get("dcc_d_real") is not None]
                d_pre = [r.get("dcc_d_model_preupdate") for r in rows if r.get("dcc_d_model_preupdate") is not None]
                d_post = [r.get("dcc_d_model_postupdate") for r in rows if r.get("dcc_d_model_postupdate") is not None]
                comp_keys = [r.get("dcc_comparison_key") for r in rows if r.get("dcc_comparison_key")]
                key_counts: dict[str, int] = {}
                for ck in comp_keys:
                    key_counts[ck] = key_counts.get(ck, 0) + 1
                return {
                    "blocks_completed": len(rows),
                    "d_real_values": d_reals,
                    "nonzero_d_real_blocks": sum(1 for x in d_reals if abs(float(x)) > 1e-6),
                    "gt_001_d_real_blocks": sum(1 for x in d_reals if abs(float(x)) > 0.01),
                    "d_model_preupdate_values": [round(float(x), 6) for x in d_pre],
                    "d_model_postupdate_values": [round(float(x), 6) for x in d_post],
                    "fork_disagreement_r_values": [round(float(r.get("dcc_fork_disagreement_r", 0.0)), 6) for r in rows],
                    "fork_selection_modes": [r.get("dcc_fork_selection_mode") for r in rows],
                    "distinct_comparison_keys": len(key_counts),
                    "reused_comparison_keys": sum(1 for v in key_counts.values() if v >= 2),
                    "factual_branch_successes": [(r.get("first_success"), r.get("second_success")) for r in rows],
                    "side_updates_total": sum(int(r.get("dcc_side_update_steps_this_block", 0)) for r in rows),
                    "side_update_seconds_total": round(sum(float(r.get("dcc_side_update_seconds", 0.0)) for r in rows), 4),
                }

            lex_diag = _summarize_train_jsonl(bench_dir / "training_lex_1upd")
            dis_diag = _summarize_train_jsonl(bench_dir / "training_prism_evolve")

            # Stage 3 Fixed-D Proposer Comparison on Arm 2's exact dataset D and exact phi_T
            main_ckpt = Path(train_dis_summary["selected_checkpoint"])
            shuf_witness_ckpt = Path(
                train_dis_summary.get("fixed_d_shuffled_checkpoint", bench_dir / "training_prism_evolve/fixed_d_shuffled_adapter")
            )

            records_path = bench_dir / "training_prism_evolve/dcc_measurement_records.json"
            cond_gain_ckpt = bench_dir / "training_prism_evolve/conditional_gain_adapter"
            cond_gain_shuf_ckpt = bench_dir / "training_prism_evolve/conditional_gain_shuffled_adapter"
            cg_info = _build_fixed_d_checkpoint(
                main_ckpt, cond_gain_ckpt, bench, records_path,
                proposer_mode="conditional_gain", shuffle_labels=False, seed=args.seed,
            )
            cg_shuf_info = _build_fixed_d_checkpoint(
                main_ckpt, cond_gain_shuf_ckpt, bench, records_path,
                proposer_mode="conditional_gain", shuffle_labels=True, seed=args.seed + 999,
            )

            # Evaluate all paired arms on the exact same dev tasks under pure `expected_utility`
            eval_arms = [
                ("frozen_base", init_adapter_dir),
                ("dcc_k0_planner", init_adapter_dir),
                ("prism_evolve_lex_1upd", Path(train_lex_summary["selected_checkpoint"])),
                ("dcc_no_program_growth", main_ckpt),
                ("dcc_reset_parameters", main_ckpt),
                ("prism_evolve", main_ckpt),
                ("dcc_fixed_d_shuffled", shuf_witness_ckpt),
                ("dcc_conditional_gain", cond_gain_ckpt),
                ("dcc_conditional_gain_shuffled", cond_gain_shuf_ckpt),
            ]
            eval_results: dict[str, Any] = {}
            for m_name, m_ckpt in eval_arms:
                if m_name == "dcc_conditional_gain" and cg_info["library_size"] <= 4 and "dcc_no_program_growth" in eval_results:
                    ref_copy = deepcopy(eval_results["dcc_no_program_growth"])
                    ref_copy["method"] = m_name
                    ref_copy["identity_alias_of"] = "dcc_no_program_growth (Xi_Gamma == Xi_0)"
                    eval_results[m_name] = ref_copy
                    continue
                if m_name == "dcc_conditional_gain_shuffled" and cg_shuf_info["library_size"] <= 4 and "dcc_no_program_growth" in eval_results:
                    ref_copy = deepcopy(eval_results["dcc_no_program_growth"])
                    ref_copy["method"] = m_name
                    ref_copy["identity_alias_of"] = "dcc_no_program_growth (Xi_Gamma_shuf == Xi_0)"
                    eval_results[m_name] = ref_copy
                    continue
                print(f"  [Eval {bench}] {m_name}...", flush=True)
                eval_results[m_name] = _evaluate_dev_slice(
                    policy, env, dev_tasks, cfg_dis, m_ckpt, m_name, bench_dir / f"eval_{m_name}", seed=args.seed
                )

            benchmark_results[bench] = {
                "round2_acquisition_comparison": {
                    "lexicographic_1upd": {
                        "training_usage": train_lex_summary["usage"],
                        "dcc_summary": train_lex_summary.get("dcc_summary", {}),
                        "block_diagnostics": lex_diag,
                    },
                    "decision_disagreement_4upd": {
                        "training_usage": train_dis_summary["usage"],
                        "dcc_summary": train_dis_summary.get("dcc_summary", {}),
                        "block_diagnostics": dis_diag,
                    },
                },
                "round3_fixed_d_proposers": {
                    "conditional_gain_true_d": cg_info,
                    "conditional_gain_shuffled_d": cg_shuf_info,
                },
                "eval_comparison": {
                    k: {
                        mk: mv for mk, mv in v.items() if mk not in ("per_task",)
                    }
                    for k, v in eval_results.items()
                },
                "paired_task_outcomes": {
                    k: [(r["task_id"], r["success"], round(r["utility_u_b"], 4), r["env_steps"]) for r in v["per_task"]]
                    for k, v in eval_results.items()
                },
            }
            _write_json(bench_dir / f"{bench}_summary.json", benchmark_results[bench])
        finally:
            env.close()

    # === LOCA QEMU VM Closed-Loop Verification under r4 ===
    print("[LOCA] Running QEMU VM closed-loop verification under unified r4...", flush=True)
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
        "dcc_proposer_mode": "witness_hard",
        "dcc_quantize_prompt_wall_seconds": True,
        "training_objective": "U_B=Y*max(0,1-C/B)",
    }
    policy.max_new_tokens = 512
    policy.load_adapter(init_adapter_dir)
    loca_env = _environment(loca_cfg, "loca", loca_dir / "env_loca")
    try:
        l_train = loca_env.list_tasks("train")
        l_dev = loca_env.list_tasks("dev")[:1]
        l_meter = BudgetMeter(BudgetLimits(200, 500000, 1200.0), event_path=loca_dir / "training_costs.jsonl")
        l_train_sum = run_training(
            "prism_evolve", policy, loca_env, l_train, l_dev, loca_cfg,
            l_meter, loca_dir / "training_prism_evolve", args.seed,
        )
        l_eval = {}
        for lm_name, lm_ckpt in [
            ("frozen_base", init_adapter_dir),
            ("dcc_k0_planner", init_adapter_dir),
            ("prism_evolve", Path(l_train_sum["selected_checkpoint"])),
        ]:
            l_eval[lm_name] = _evaluate_dev_slice(
                policy, loca_env, l_dev, loca_cfg, lm_ckpt, lm_name, loca_dir / f"eval_{lm_name}", seed=args.seed
            )
        benchmark_results["loca"] = {
            "complete_groups": l_train_sum["complete_groups"],
            "gradient_steps": l_train_sum["gradient_steps"],
            "training_usage": l_train_sum["usage"],
            "eval_comparison": {
                k: {mk: mv for mk, mv in v.items() if mk not in ("per_task",)}
                for k, v in l_eval.items()
            },
        }
    finally:
        loca_env.close()

    final_payload = {
        "resolved_manifest": resolved_manifest,
        "stage1_existing_data_replay": stage1_summary,
        "benchmarks": benchmark_results,
    }
    _write_json(out_dir / "r11_staged_summary.json", final_payload)
    print("[COMPLETE] Wrote summary to", out_dir / "r11_staged_summary.json", flush=True)


if __name__ == "__main__":
    main()
