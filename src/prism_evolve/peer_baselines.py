"""Legacy controlled baseline adaptations for the fixed macro-action experiments.

The registry records the papers and repositories that inspired these adaptations.
The functions use the local finite-action scorer, scenario groups and rule-based
wrappers; they do not invoke the complete official training or inference pipelines.
Native-benchmark comparisons must use the separate full-method integration path.
Historical method IDs remain available for reading and reproducing controlled runs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .collector import StratifiedBranchSample
from .event_log import CandidateAction, EventRecord, PublicHistory
from .policy import ActionSpacePartition, DualModePolicy, PolicyForwardOutput


DEFAULT_EXTERNAL_ROOT = Path("/apps/users/adminuser/prism-exact-external")


@dataclass(frozen=True)
class PeerBaselineProvenance:
    """Reference provenance for a locally adapted controlled baseline."""

    method_id: str
    display_name: str
    venue: str
    arxiv_id: str
    repo_url: str
    local_dir_name: str
    pinned_commit: str
    key_source_file: str
    implementation_kind: str = "legacy_controlled_adaptation"


PEER_BASELINE_REGISTRY: Dict[str, PeerBaselineProvenance] = {
    "frozen_base": PeerBaselineProvenance(
        method_id="frozen_base",
        display_name="Frozen controlled scorer",
        venue="EMNLP 2026 Main",
        arxiv_id="2606.22388",
        repo_url="https://github.com/JiayuJeff/PlanBench-XL",
        local_dir_name="pbxl-repo",
        pinned_commit="a24bfd5f1a6ad7ad3c5d5204525272c6488225ac",
        key_source_file="src/data/retail/queries.json",
    ),
    "gigpo": PeerBaselineProvenance(
        method_id="gigpo",
        display_name="GiGPO-inspired controlled adaptation",
        venue="NeurIPS 2025 / 2026",
        arxiv_id="2505.10978",
        repo_url="https://github.com/langfengQ/verl-agent",
        local_dir_name="gigpo-official",
        pinned_commit="20bd331bdbc9026a5668e11362178e10ab7400c8",
        key_source_file="gigpo/core_gigpo.py",
    ),
    "step_dpo": PeerBaselineProvenance(
        method_id="step_dpo",
        display_name="Branch-DPO controlled adaptation",
        venue="NeurIPS 2024 / 2025",
        arxiv_id="2406.18629",
        repo_url="https://github.com/JIA-Lab-research/Step-DPO",
        local_dir_name="step-dpo-official",
        pinned_commit="1f504ead5004f252025cb234017dfd9897cd542c",
        key_source_file="stepdpo_trainer.py",
    ),
    "infopo": PeerBaselineProvenance(
        method_id="infopo",
        display_name="InfoPO-inspired controlled KL adaptation",
        venue="ICLR 2026 Workshop / arXiv 2026.03",
        arxiv_id="2603.00656",
        repo_url="https://github.com/kfq20/InfoPO",
        local_dir_name="infopo-official",
        pinned_commit="d230e78afb3890eb1c7eec2f27f75bcc6dceed8d",
        key_source_file="verl/trainer/ppo/core_algos.py",
    ),
    "arew": PeerBaselineProvenance(
        method_id="arew",
        display_name="AREW-inspired controlled feedback adaptation",
        venue="ICLR 2026 Oral & ICML 2026",
        arxiv_id="2603.12109",
        repo_url="https://github.com/unimpor/T3",
        local_dir_name="arew-official",
        pinned_commit="492f31fa05d2065c750a72d5e798385af282fa5d",
        key_source_file="verl/verl/trainer/ppo/ray_trainer.py",
    ),
    "worldevolver": PeerBaselineProvenance(
        method_id="worldevolver",
        display_name="WorldEvolver-inspired controlled rule memory",
        venue="EMNLP 2026",
        arxiv_id="2606.30639",
        repo_url="https://github.com/magicgh/WorldEvolver",
        local_dir_name="worldevolver-official",
        pinned_commit="f7013c63a0b012d3ba1f0fbbefab9218ba4d110e",
        key_source_file="src/agents/wm_react_agent.py",
    ),
    "pos_envprobe": PeerBaselineProvenance(
        method_id="pos_envprobe",
        display_name="PoS/EnvProbe-inspired controlled entropy wrapper",
        venue="2026.10",
        arxiv_id="2610.01415 / 2606.31422",
        repo_url="https://github.com/luoyu100/PoS",
        local_dir_name="envprobe-official",
        pinned_commit="faa14b07bbbab2a10d0b45f9ceafb67c73ca85c0",
        key_source_file="src/methods/self_uncertainty_probe.py",
    ),
}


MAIN_TABLE_PEER_METHODS: Tuple[str, ...] = (
    "frozen_base",
    "gigpo",
    "step_dpo",
    "infopo",
    "arew",
    "worldevolver",
    "pos_envprobe",
    "prism_evolve",
)

ABLAT_METHODS: Tuple[str, ...] = (
    "frozen_base",
    "outcome_pg",
    "outcome_value_baseline",
    "outcome_rloo",
    "info_rl_neighbor",
    "unstratified_2branch",
    "ablation_unweighted",
    "prism_evolve",
)


def verify_external_baseline_repos(
    external_root: Path | str = DEFAULT_EXTERNAL_ROOT,
) -> Dict[str, Dict[str, Any]]:
    """Inspect reference files; presence does not attest full-method integration."""
    root = Path(external_root)
    report: Dict[str, Dict[str, Any]] = {}
    for method_id, prov in PEER_BASELINE_REGISTRY.items():
        repo_dir = root / prov.local_dir_name
        src_path = repo_dir / prov.key_source_file
        report[method_id] = {
            "display_name": prov.display_name,
            "implementation_kind": prov.implementation_kind,
            "venue": prov.venue,
            "arxiv_id": prov.arxiv_id,
            "repo_url": prov.repo_url,
            "pinned_commit": prov.pinned_commit,
            "repo_dir": str(repo_dir),
            "key_source_file": str(src_path),
            "repo_exists": repo_dir.is_dir(),
            "key_source_exists": src_path.is_file(),
            "exists": (repo_dir.is_dir() and src_path.is_file()) if prov.local_dir_name else True,
        }
    return report


def _prefix_at_step(history: PublicHistory, step_idx: int) -> PublicHistory:
    return PublicHistory(
        task_id=history.task_id,
        scenario_id=history.scenario_id,
        goal=history.goal,
        initial_observation=history.initial_observation,
        total_budget=history.total_budget,
        events=history.events[:step_idx],
    )


def _anchor_state_key(history: PublicHistory) -> str:
    """Compute canonical anchor-state key for GiGPO `build_step_group` (`gigpo/core_gigpo.py`).

    Groups steps across trajectories that share the same structural observation state
    (task domain, step depth, whether informative probe occurred, whether an unrecovered
    execution failure is present, and whether terminal submission is ready).
    """
    vis_events = [ev for ev in history.events if "[MASKED_FEEDBACK]" not in ev.observation]
    has_probe = any(ev.mode == "I" and "[probe_empty]" not in ev.observation.lower() for ev in vis_events)
    has_fail = any(
        ev.mode == "E"
        and (
            "error" in ev.observation.lower()
            or "failed" in ev.observation.lower()
            or "blocked" in ev.observation.lower()
            or bool(ev.metadata.get("step_failed", False))
        )
        for ev in vis_events
    )
    ready_sub = any(
        "verified_exec_result=true" in ev.observation.lower()
        or "ready_submit=" in ev.observation.lower()
        for ev in vis_events
    )
    scen = history.scenario_id or history.task_id.split("__")[0]
    return f"scen={scen}|s_depth={min(history.step_count, 3)}|probe={int(has_probe)}|fail={int(has_fail)}|ready={int(ready_sub)}"


def compute_gigpo_batch_losses(
    policy: DualModePolicy,
    samples: Sequence[StratifiedBranchSample],
    step_advantage_w: float = 1.0,
    gamma: float = 0.95,
    epsilon: float = 1e-6,
    batch_baseline: float = 0.0,
) -> List[torch.Tensor]:
    """Compute a controlled finite-action adaptation of the GiGPO advantage loss.

    Implements:
    - Eq. 3 (`episode_norm_reward`): macro trajectory-level normalized relative advantage.
    - Controlled structural-state groups approximate the Eq. 6 anchor groups.
    - Eq. 7 (`step_norm_reward`): micro step-level relative advantage from discounted return-to-go.
    - Eq. 8: combined advantage A_{i,t} = A^{macro}_i + w * A^{micro}_{i,t}.
    """
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype

    # Extract all rollout trajectories (each unstratified 2-branch sample contributes 2 rollouts
    # in the same task group; single-trajectory samples are grouped by scenario_id).
    trajectories: List[Tuple[str, Any, Sequence[ActionSpacePartition], int]] = []
    for s_idx, sample in enumerate(samples):
        if sample.did_split and sample.branch_I is not None and sample.branch_E is not None:
            grp_id = f"task_{sample.branch_I.history.task_id}_{s_idx}"
            trajectories.append((grp_id, sample.branch_I, sample.step_partitions_I, s_idx))
            trajectories.append((grp_id, sample.branch_E, sample.step_partitions_E, s_idx))
        elif sample.single_trajectory is not None:
            out = sample.single_trajectory
            grp_id = f"scen_{out.history.scenario_id or out.history.task_id.split('__')[0]}"
            trajectories.append((grp_id, out, sample.step_partitions_single, s_idx))

    if not trajectories:
        return [torch.zeros((), device=device, dtype=dtype)]

    # 1. Episode-level macro advantage (Eq. 3 of gigpo/core_gigpo.py)
    grp_scores: Dict[str, List[float]] = defaultdict(list)
    all_scores: List[float] = []
    for grp_id, out, _, _ in trajectories:
        u = float(out.utility)
        grp_scores[grp_id].append(u)
        all_scores.append(u)

    batch_mean = sum(all_scores) / len(all_scores)
    batch_std = math.sqrt(sum((x - batch_mean) ** 2 for x in all_scores) / max(1, len(all_scores)))

    macro_advs: List[float] = []
    for grp_id, out, _, _ in trajectories:
        u = float(out.utility)
        g_vals = grp_scores[grp_id]
        if len(g_vals) > 1:
            g_mean = sum(g_vals) / len(g_vals)
            g_std = math.sqrt(sum((x - g_mean) ** 2 for x in g_vals) / (len(g_vals) - 1))
            if g_std > 1e-4:
                macro_advs.append((u - g_mean) / (g_std + epsilon))
            else:
                macro_advs.append((u - batch_mean) / (batch_std + epsilon))
        else:
            macro_advs.append((u - batch_mean) / (batch_std + epsilon))

    # 2. Step-level anchor-state grouping & micro advantage (Eq. 6-7 of gigpo/core_gigpo.py)
    anchor_returns: Dict[str, List[float]] = defaultdict(list)
    traj_step_keys_and_rtg: List[List[Tuple[str, float]]] = []

    for _, out, _, _ in trajectories:
        events = out.history.events
        n_steps = len(events)
        u_term = float(out.utility)
        step_info: List[Tuple[str, float]] = []
        for t_idx in range(n_steps):
            h_t = _prefix_at_step(out.history, t_idx)
            akey = _anchor_state_key(h_t)
            rtg = u_term * (gamma ** (n_steps - 1 - t_idx))
            anchor_returns[akey].append(rtg)
            step_info.append((akey, rtg))
        traj_step_keys_and_rtg.append(step_info)

    anchor_mean_std: Dict[str, Tuple[float, float]] = {}
    for akey, vals in anchor_returns.items():
        if len(vals) <= 1:
            anchor_mean_std[akey] = (vals[0] if vals else 0.0, 1.0)
        else:
            m = sum(vals) / len(vals)
            s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
            anchor_mean_std[akey] = (m, s)

    # 3. Combine macro + micro advantages (Eq. 8 of gigpo/core_gigpo.py) and evaluate log-probs
    sample_losses: Dict[int, torch.Tensor] = {
        idx: torch.zeros((), device=device, dtype=dtype) for idx in range(len(samples))
    }
    sample_counts: Dict[int, int] = defaultdict(int)

    for traj_idx, (_, out, parts, s_idx) in enumerate(trajectories):
        events = out.history.events
        n_steps = len(events)
        if n_steps == 0:
            continue
        traj_loss = torch.zeros((), device=device, dtype=dtype)
        macro_a = macro_advs[traj_idx]
        step_info = traj_step_keys_and_rtg[traj_idx]

        for t_idx, ev in enumerate(events):
            h_t = _prefix_at_step(out.history, t_idx)
            out_t = policy.forward_step(h_t, parts[t_idx])
            lp = out_t.full_log_prob(ev.action)

            akey, rtg = step_info[t_idx]
            a_mean, a_std = anchor_mean_std[akey]
            if len(anchor_returns[akey]) > 1 and a_std > 1e-4:
                micro_a = (rtg - a_mean) / (a_std + epsilon)
            else:
                micro_a = 0.0

            combined_adv = macro_a + step_advantage_w * micro_a
            traj_loss = traj_loss - float(combined_adv) * lp

        sample_losses[s_idx] = sample_losses[s_idx] + traj_loss
        sample_counts[s_idx] += 1

    return [
        sample_losses[idx] / max(1, sample_counts[idx])
        for idx in range(len(samples))
    ]


def compute_step_dpo_episode_loss(
    policy: DualModePolicy,
    ref_policy: DualModePolicy | None,
    sample: StratifiedBranchSample,
    beta_dpo: float = 0.4,
    batch_baseline: float = 0.0,
) -> torch.Tensor:
    """Compute reference-policy DPO on a controlled continuation preference pair.

    The unweighted DPO loss follows the paper's reference log-ratio objective.
    Pair construction here ranks complete branches by terminal utility; the original
    Step-DPO first-error localization and step rectification require a separate dataset.
    Equal-utility branches and episodes without a pair contribute zero loss.
    """
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype
    if ref_policy is None or ref_policy is policy:
        raise ValueError("Branch DPO requires a distinct frozen reference policy")
    eff_ref = ref_policy

    zero_loss = next(p for p in policy.parameters() if p.requires_grad).reshape(-1)[0] * 0.0

    if sample.did_split and sample.branch_I is not None and sample.branch_E is not None:
        T = sample.split_step
        G_1 = float(sample.branch_I.utility)
        G_2 = float(sample.branch_E.utility)
        if G_1 == G_2:
            return zero_loss
        hist_1 = sample.branch_I.history
        hist_2 = sample.branch_E.history
        parts_1 = sample.step_partitions_I
        parts_2 = sample.step_partitions_E

        # Evaluate full log-prob of continuation >= T under current policy and frozen ref_policy
        ell_1 = torch.zeros((), device=device, dtype=dtype)
        ell_1_ref = 0.0
        for t_idx in range(T, len(hist_1.events)):
            h_t = _prefix_at_step(hist_1, t_idx)
            out_t = policy.forward_step(h_t, parts_1[t_idx])
            ell_1 = ell_1 + out_t.full_log_prob(hist_1.events[t_idx].action)
            with torch.no_grad():
                out_ref = eff_ref.forward_step(h_t, parts_1[t_idx])
                ell_1_ref += float(out_ref.full_log_prob(hist_1.events[t_idx].action).detach().cpu().item())

        ell_2 = torch.zeros((), device=device, dtype=dtype)
        ell_2_ref = 0.0
        for t_idx in range(T, len(hist_2.events)):
            h_t = _prefix_at_step(hist_2, t_idx)
            out_t = policy.forward_step(h_t, parts_2[t_idx])
            ell_2 = ell_2 + out_t.full_log_prob(hist_2.events[t_idx].action)
            with torch.no_grad():
                out_ref = eff_ref.forward_step(h_t, parts_2[t_idx])
                ell_2_ref += float(out_ref.full_log_prob(hist_2.events[t_idx].action).detach().cpu().item())

        if G_1 >= G_2:
            chosen_logratios = ell_1 - ell_1_ref
            rejected_logratios = ell_2 - ell_2_ref
        else:
            chosen_logratios = ell_2 - ell_2_ref
            rejected_logratios = ell_1 - ell_1_ref

        dpo_logits = beta_dpo * (chosen_logratios - rejected_logratios)
        return -F.logsigmoid(dpo_logits)

    return zero_loss


def compute_infopo_batch_losses(
    policy: DualModePolicy,
    samples: Sequence[StratifiedBranchSample],
    intrinsic_weight: float = 0.5,
    intrinsic_gate_temperature: float = 0.05,
    epsilon: float = 1e-6,
    batch_baseline: float = 0.0,
) -> List[torch.Tensor]:
    """Compute a legacy finite-action KL-shaping adaptation inspired by InfoPO.

    This controlled variant uses batch-normalized action-distribution KL. Original
    InfoPO uses realized next-action token likelihood differences and same-prompt
    rollout groups; native comparisons must use that full integration.

    Controlled adaptation:
    1. Turn-level intrinsic information gain via counterfactual observation masking:
       I_t = KL(pi_theta(. | h_{t+1}^{vis}) || pi_theta(. | h_{t+1}^{mask}))
    2. Group/batch normalization of non-zero intrinsic rewards.
    3. Two-stage variance-gated intrinsic fusion:
       - Stage 1 bounded magnitude ratio: min(1.0, max(outcome_std, 1.0) / (intrinsic_std + eps))
       - Stage 2 variance gate: sigmoid(-group_std / intrinsic_gate_temperature)
       - Combined step advantage: A^{out}_i + (intrinsic_weight * gate * ratio) * A^{int}_{i,t}
    """
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype

    outcomes_and_parts: List[Tuple[Any, Sequence[ActionSpacePartition]]] = []
    for s in samples:
        if s.single_trajectory is not None:
            outcomes_and_parts.append((s.single_trajectory, s.step_partitions_single))
        elif s.branch_I is not None:
            outcomes_and_parts.append((s.branch_I, s.step_partitions_I))

    if not outcomes_and_parts:
        return [torch.zeros((), device=device, dtype=dtype)]

    raw_outcomes = [float(out.utility) for out, _ in outcomes_and_parts]
    out_mean = sum(raw_outcomes) / len(raw_outcomes)
    out_std = math.sqrt(sum((x - out_mean) ** 2 for x in raw_outcomes) / max(1, len(raw_outcomes)))

    # 1. Compute turn-level counterfactual KL intrinsic rewards for each trajectory
    traj_intrinsic_steps: List[List[float]] = []
    non_zero_intrinsics: List[float] = []

    with torch.no_grad():
        for out, parts in outcomes_and_parts:
            events = out.history.events
            step_kls: List[float] = [0.0] * len(events)
            for t_idx, ev in enumerate(events):
                if ev.mode == "I" and t_idx + 1 < len(events):
                    h_next_vis = _prefix_at_step(out.history, t_idx + 1)
                    h_next_mask = h_next_vis.with_masked_step(t_idx)
                    part_next = parts[t_idx + 1]
                    out_vis = policy.forward_step(h_next_vis, part_next)
                    out_mask = policy.forward_step(h_next_mask, part_next)

                    p_vis = float(out_vis.gate_prob.clamp(1e-6, 1.0 - 1e-6).item())
                    p_msk = float(out_mask.gate_prob.clamp(1e-6, 1.0 - 1e-6).item())
                    kl_gate = p_vis * math.log(p_vis / p_msk) + (1.0 - p_vis) * math.log(
                        (1.0 - p_vis) / (1.0 - p_msk)
                    )
                    kl_exec = 0.0
                    if out_vis.log_probs_E is not None and out_mask.log_probs_E is not None:
                        probs_e_vis = torch.exp(out_vis.log_probs_E)
                        kl_exec = float(
                            torch.sum(probs_e_vis * (out_vis.log_probs_E - out_mask.log_probs_E))
                            .clamp(0.0, 2.0)
                            .item()
                        )
                    kl_val = max(0.0, float(kl_gate) + 0.35 * kl_exec)
                    step_kls[t_idx] = kl_val
                    if kl_val > 0.0:
                        non_zero_intrinsics.append(kl_val)
            traj_intrinsic_steps.append(step_kls)

    if len(non_zero_intrinsics) > 1:
        int_mean = sum(non_zero_intrinsics) / len(non_zero_intrinsics)
        int_std = math.sqrt(
            sum((x - int_mean) ** 2 for x in non_zero_intrinsics) / (len(non_zero_intrinsics) - 1)
        )
    elif len(non_zero_intrinsics) == 1:
        int_mean = non_zero_intrinsics[0]
        int_std = 1.0
    else:
        int_mean = 0.0
        int_std = 1.0

    # Stage 1 bounded scaling & Stage 2 variance gating (lines 1262-1287 of core_algos.py)
    baseline_std = 1.0
    target_int_std = max(out_std, baseline_std)
    magnitude_ratio = min(1.0, target_int_std / (int_std + epsilon))
    var_gate = 1.0 / (1.0 + math.exp(min(20.0, max(-20.0, out_std / max(1e-4, intrinsic_gate_temperature)))))
    eff_int_weight = intrinsic_weight * max(0.25, var_gate) * magnitude_ratio

    losses: List[torch.Tensor] = []
    for idx, (out, parts) in enumerate(outcomes_and_parts):
        events = out.history.events
        n_steps = len(events)
        if n_steps == 0:
            losses.append(torch.zeros((), device=device, dtype=dtype))
            continue

        out_adv = raw_outcomes[idx] - out_mean
        step_kls = traj_intrinsic_steps[idx]

        ep_loss = torch.zeros((), device=device, dtype=dtype)
        for t_idx, ev in enumerate(events):
            h_t = _prefix_at_step(out.history, t_idx)
            out_t = policy.forward_step(h_t, parts[t_idx])
            lp = out_t.full_log_prob(ev.action)

            kl_t = step_kls[t_idx]
            if kl_t > 0.0:
                norm_int = (kl_t - int_mean) / (int_std + epsilon)
                int_bonus = eff_int_weight * (0.65 + 0.35 * norm_int)
            else:
                int_bonus = 0.0

            step_adv = out_adv + int_bonus
            ep_loss = ep_loss - float(step_adv) * lp

        losses.append(ep_loss)

    return losses


def compute_arew_batch_losses(
    policy: DualModePolicy,
    samples: Sequence[StratifiedBranchSample],
    arew_bonus_scale: float = 0.35,
    batch_baseline: float = 0.0,
) -> List[torch.Tensor]:
    """Compute a controlled feedback-label adaptation inspired by AREW.

    Explicit observed failures receive -1 and explicit successful probe/execution
    feedback receives +1. Unlabelled feedback stays neutral. The local zero-sum
    modifier is defined only when both signs occur. Original AREW requires its
    task-specific AS/BT directional critiques; this function does not implement T3.
    """
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype

    losses: List[torch.Tensor] = []
    for s in samples:
        out = s.single_trajectory if s.single_trajectory is not None else s.branch_I
        parts = s.step_partitions_single if s.single_trajectory is not None else s.step_partitions_I
        if out is None or not out.history.events:
            losses.append(torch.zeros((), device=device, dtype=dtype))
            continue

        events = out.history.events
        n_steps = len(events)
        G = float(out.utility)

        # Assign directional step labels y_t in {-1, 0, +1}
        labels: List[int] = []
        for ev in events:
            obs_l = ev.observation.lower()
            if "step_failed" in ev.metadata:
                step_failed = bool(ev.metadata["step_failed"])
            else:
                step_failed = obs_l.startswith(("[exec_error]", "[submit_fail]"))
            step_succeeded = bool(ev.metadata.get("step_succeeded", False)) or obs_l.startswith(
                ("[probe_ok]", "[exec_ok]", "[submit_pass]")
            )
            if step_failed:
                labels.append(-1)
            elif step_succeeded:
                labels.append(1)
            else:
                labels.append(0)

        pos_indices = [i for i, y in enumerate(labels) if y > 0]
        neg_indices = [i for i, y in enumerate(labels) if y < 0]
        step_bonuses = [0.0] * n_steps
        if pos_indices and neg_indices:
            pos_budget = arew_bonus_scale / (2.0 * len(pos_indices))
            neg_budget = -arew_bonus_scale / (2.0 * len(neg_indices))
            for i in pos_indices:
                step_bonuses[i] = pos_budget
            for i in neg_indices:
                step_bonuses[i] = neg_budget

        base_adv = G - float(batch_baseline)

        ep_loss = torch.zeros((), device=device, dtype=dtype)
        for t_idx, ev in enumerate(events):
            h_t = _prefix_at_step(out.history, t_idx)
            out_t = policy.forward_step(h_t, parts[t_idx])
            lp = out_t.full_log_prob(ev.action)
            total_step_adv = base_adv + step_bonuses[t_idx]
            ep_loss = ep_loss - float(total_step_adv) * lp

        losses.append(ep_loss)

    return losses


@dataclass(frozen=True)
class WorldModelRuleEntry:
    """Rule extracted by the legacy controlled memory wrapper."""

    scenario_id: str
    domain_token: str
    recommended_probe_id: Optional[str]
    verified_exec_id: Optional[str]
    blocked_exec_ids: Tuple[str, ...]
    rule_text: str
    observed_utility: float


class WorldEvolverMemoryPolicy(DualModePolicy):
    """Legacy controlled rule-memory wrapper inspired by WorldEvolver.

    Uses observed macro-action tags to retrieve rules and adjust finite-action scores.
    Full WorldEvolver additionally requires model predictions, semantic memory
    revision and selective foresight. Overhead fields here use the legacy cost model.
    """

    def __init__(
        self,
        base_policy: DualModePolicy,
        max_memory_entries: int = 128,
        top_k_retrieve: int = 3,
        retrieval_logit_boost: float = 0.95,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.max_memory_entries = max_memory_entries
        self.top_k_retrieve = top_k_retrieve
        self.retrieval_logit_boost = retrieval_logit_boost
        self.memory_bank: List[WorldModelRuleEntry] = []

    def distill_batch(self, samples: Sequence[StratifiedBranchSample]) -> int:
        """Extract world-model transition rules from completed rollout trajectories."""
        added = 0
        for s in samples:
            out = s.single_trajectory if s.single_trajectory is not None else s.branch_I
            if out is None:
                continue
            hist = out.history
            init_l = hist.initial_observation.lower()
            dom_tok = ""
            for part in init_l.split():
                if part.startswith("active_domain="):
                    dom_tok = part.split("=", 1)[1].strip(",.; ")
                    break

            rec_probe: Optional[str] = None
            ver_exec: Optional[str] = None
            blocked_execs: List[str] = []

            for ev in hist.events:
                obs_l = ev.observation.lower()
                if ev.mode == "I" and "[probe_empty]" not in obs_l:
                    rec_probe = ev.action.action_id
                    for tok in obs_l.split():
                        if tok.startswith("verified_action="):
                            ver_exec = tok.split("=", 1)[1].strip(",.; ")
                        elif tok.startswith("blocked_action="):
                            blocked_execs.append(tok.split("=", 1)[1].strip(",.; "))
                elif ev.mode == "E":
                    if "error" in obs_l or "failed" in obs_l or "blocked" in obs_l:
                        blocked_execs.append(ev.action.action_id)
                    elif out.terminal_success == 1 and ver_exec is None and not ev.action.is_terminal_submit:
                        ver_exec = ev.action.action_id

            if rec_probe or ver_exec or blocked_execs:
                rule_txt = (
                    f"[WorldEvolver Rule scen={hist.scenario_id} dom={dom_tok}] "
                    f"probe={rec_probe} verified={ver_exec} blocked={blocked_execs}"
                )
                entry = WorldModelRuleEntry(
                    scenario_id=hist.scenario_id,
                    domain_token=dom_tok,
                    recommended_probe_id=rec_probe,
                    verified_exec_id=ver_exec,
                    blocked_exec_ids=tuple(sorted(set(blocked_execs))),
                    rule_text=rule_txt,
                    observed_utility=float(out.utility),
                )
                self.memory_bank.append(entry)
                added += 1

        if len(self.memory_bank) > self.max_memory_entries:
            self.memory_bank = self.memory_bank[-self.max_memory_entries :]
        return added

    def distill_from_samples(self, samples: Sequence[StratifiedBranchSample]) -> int:
        """Alias for updater compatibility (`step_batch` calls `distill_from_samples`)."""
        return self.distill_batch(samples)

    def _retrieve_rules(self, history: PublicHistory) -> List[WorldModelRuleEntry]:
        if not self.memory_bank:
            return []
        init_l = history.initial_observation.lower()
        scored: List[Tuple[float, int, WorldModelRuleEntry]] = []
        for idx, entry in enumerate(self.memory_bank):
            score = 0.1 + 0.3 * entry.observed_utility
            if entry.domain_token and f"active_domain={entry.domain_token}" in init_l:
                score += 1.5
            if entry.scenario_id and entry.scenario_id == history.scenario_id:
                score += 1.0
            scored.append((score, idx, entry))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item[2] for item in scored[: self.top_k_retrieve]]

    def forward_step(
        self,
        history: PublicHistory,
        partition: ActionSpacePartition,
    ) -> PolicyForwardOutput:
        base_out = self.base_policy.forward_step(history, partition)
        retrieved = self._retrieve_rules(history)
        if not retrieved:
            return base_out

        # Non-parametric in-context memory retrieval adds prompt tokens & shifts logits
        extra_tokens = 38 * len(retrieved)
        device = base_out.gate_prob.device
        dtype = base_out.gate_prob.dtype

        vis_events = [ev for ev in history.events if "[MASKED_FEEDBACK]" not in ev.observation]
        has_probed = any(ev.mode == "I" and "[probe_empty]" not in ev.observation.lower() for ev in vis_events)
        domain_matched_rules = [
            r for r in retrieved if r.domain_token and f"active_domain={r.domain_token}" in history.initial_observation.lower()
        ]
        probe_rules = [r for r in domain_matched_rules if r.recommended_probe_id is not None and r.observed_utility >= 0.45]

        gate_prob = base_out.gate_prob
        if not has_probed and partition.both_modes_available:
            # Retrieved world-model rules remind the agent to inspect preconditions when a matching
            # domain rule with positive probe outcome exists in the sliding memory window, but suffer
            # from incomplete coverage and frozen continuation parameters theta_0.
            if probe_rules and history.step_count == 0:
                boost = 1.18 * self.retrieval_logit_boost
            elif domain_matched_rules and history.step_count == 0:
                boost = 0.65 * self.retrieval_logit_boost
            else:
                boost = 0.22 * self.retrieval_logit_boost
            logit_g = torch.logit(gate_prob.clamp(1e-6, 1.0 - 1e-6)) + boost
            gate_prob = torch.sigmoid(logit_g)

        log_probs_E = base_out.log_probs_E
        if log_probs_E is not None and partition.exec_actions and domain_matched_rules:
            blocked_set = {bid.lower() for r in domain_matched_rules for bid in r.blocked_exec_ids}
            adj = torch.zeros_like(log_probs_E, device=device, dtype=dtype)
            for idx, cand in enumerate(partition.exec_actions):
                if cand.action_id.lower() in blocked_set and not has_probed:
                    adj[idx] -= 0.35
            log_probs_E = F.log_softmax(log_probs_E + adj, dim=0)

        return PolicyForwardOutput(
            gate_prob=gate_prob,
            log_probs_I=base_out.log_probs_I,
            log_probs_E=log_probs_E,
            partition=partition,
            input_tokens=base_out.input_tokens + extra_tokens,
            output_tokens=base_out.output_tokens + 12,
            inference_time_sec=base_out.inference_time_sec + 0.0004 * len(retrieved),
            value_estimate=base_out.value_estimate,
        )


class PoSEnvProbePolicy(DualModePolicy):
    """Legacy controlled entropy wrapper inspired by PoS and EnvProbe.

    This combined heuristic lacks PoS belief validation/trapping recovery and the
    full EnvProbe field-selection/correction pipeline. Use independent native
    integrations for those methods. Overhead fields use the legacy cost model.
    """

    def __init__(
        self,
        base_policy: DualModePolicy,
        uncertainty_threshold: float = 0.52,
        probing_logit_shift: float = 1.35,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.uncertainty_threshold = uncertainty_threshold
        self.probing_logit_shift = probing_logit_shift

    def forward_step(
        self,
        history: PublicHistory,
        partition: ActionSpacePartition,
    ) -> PolicyForwardOutput:
        base_out = self.base_policy.forward_step(history, partition)
        vis_events = [ev for ev in history.events if "[MASKED_FEEDBACK]" not in ev.observation]
        has_probed = any(ev.mode == "I" and "[probe_empty]" not in ev.observation.lower() for ev in vis_events)
        has_discovered = any(ev.action.intent == "discover" for ev in vis_events)

        # Compute normalized predictive Shannon entropy over execution candidates
        exec_entropy_norm = 0.0
        if base_out.log_probs_E is not None and len(partition.exec_actions) > 1:
            probs_e = torch.exp(base_out.log_probs_E)
            ent = -float(torch.sum(probs_e * base_out.log_probs_E).detach().cpu().item())
            exec_entropy_norm = ent / math.log(len(partition.exec_actions))

        gate_prob = base_out.gate_prob
        if partition.both_modes_available:
            if not has_probed and exec_entropy_norm >= self.uncertainty_threshold:
                # Controlled entropy-based exploration heuristic.
                logit_g = torch.logit(gate_prob.clamp(1e-6, 1.0 - 1e-6)) + self.probing_logit_shift
                gate_prob = torch.sigmoid(logit_g)
            elif has_probed and not has_discovered and history.step_count == 1:
                # Legacy macro-action wrapper's explicit second-probe heuristic.
                logit_g = torch.logit(gate_prob.clamp(1e-6, 1.0 - 1e-6)) + 3.8
                gate_prob = torch.sigmoid(logit_g)

        # Legacy estimated overhead; these fields are not measured model usage.
        return PolicyForwardOutput(
            gate_prob=gate_prob,
            log_probs_I=base_out.log_probs_I,
            log_probs_E=base_out.log_probs_E,
            partition=partition,
            input_tokens=base_out.input_tokens + 96,
            output_tokens=base_out.output_tokens + 44,
            inference_time_sec=base_out.inference_time_sec + 0.0008,
            value_estimate=base_out.value_estimate,
        )
