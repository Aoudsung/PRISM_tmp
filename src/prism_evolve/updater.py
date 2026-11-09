"""Module 5: Objective-Preserving Stratified Parameter Updater and Exact Rational Verifier.

Implements §4.2, §5.1, §5.2, §5.3, and §7.3 of the PRISM-Evolve design specification:
- Single-backward objective-preserving loss L_k(theta):
    L_k(theta) = -sg(bar_G - b_k) * ell_pre(theta)
                 -sg(G_I - G_E) * p_theta(h_T, B_T)
                 -sg(p * (G_I - G_E)) * ell_I(theta)
                 -sg((1 - p) * (G_E - G_I)) * ell_E(theta)
  where bar_G = p * G_I + (1 - p) * G_E, all return terms and weight p = p_{theta_k}(h_T, B_T)
  have stop_gradient applied, and every occurred action's score is evaluated strictly once.
- Comparison and ablation updaters:
  - `prism_evolve`: Full objective-preserving stratified update
  - `ablation_unweighted`: Ablation 1 removing deployment probability weights p and 1-p
  - `unstratified_2branch`: Same-prefix unstratified 2-branch RLOO update
  - `outcome_pg`: Standard Outcome RL with moving batch baseline
  - `outcome_value_baseline`: Standard Outcome RL with learned state-value baseline V_phi
  - `outcome_rloo`: Two-sample Leave-One-Out Outcome RL
  - `info_rl_neighbor`: Information-aware RL baseline (InfoPO-style visible vs. masked KL shaping)
- Exact rational enumeration verifier for the §5.3 two-stage benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import os
from typing import Any, Sequence

import torch

from .collector import StratifiedBranchSample
from .event_log import CandidateAction, EventRecord, PublicHistory
from .peer_baselines import (
    compute_arew_batch_losses,
    compute_gigpo_batch_losses,
    compute_infopo_batch_losses,
    compute_step_dpo_episode_loss,
)
from .policy import ActionSpacePartition, DualModePolicy, TwoStageParametricPolicy
from .utility import CostLedger, TrajectoryOutcome


@dataclass(frozen=True)
class TrajectoryLogProbTerms:
    """Decomposed log-probability sums for a trajectory evaluated in a single pass."""

    ell_prefix: torch.Tensor  # sum_{0 <= t < T} log pi_theta(a_t | h_t, B_t)
    root_gate_prob: torch.Tensor | None  # p_theta(h_T, B_T) at t = T
    ell_branch_conditional_root: torch.Tensor  # log pi_{theta, m}(a_T) + sum_{t > T} log pi_theta(a_t)
    ell_branch_full_root: torch.Tensor  # sum_{t >= T} log pi_theta(a_t | h_t, B_t)
    value_losses: torch.Tensor  # MSE for optional value baseline


def _prefix_at_step(history: PublicHistory, step_idx: int) -> PublicHistory:
    """Reconstruct public history h_{step_idx} right before event `step_idx` occurred."""
    return PublicHistory(
        task_id=history.task_id,
        scenario_id=history.scenario_id,
        goal=history.goal,
        initial_observation=history.initial_observation,
        total_budget=history.total_budget,
        events=history.events[:step_idx],
    )


class StratifiedUpdater:
    """Parameter updater implementing PRISM-Evolve L_k(theta), baselines, and ablations."""

    def __init__(
        self,
        grad_clip_norm: float = 1.0,
        info_kl_coef: float = 0.05,
        baseline_momentum: float = 0.9,
        ref_policy: DualModePolicy | None = None,
    ) -> None:
        self.grad_clip_norm = grad_clip_norm
        self.info_kl_coef = info_kl_coef
        self.baseline_momentum = baseline_momentum
        self.ref_policy = ref_policy
        self.running_baseline: float = 0.0
        self._baseline_initialized: bool = False

    def _update_running_baseline(self, batch_mean_return: float) -> float:
        """Return prior independent batch baseline b_k, then update moving average (§4.2)."""
        # The first batch uses the fixed zero prior; its own returns must not set b_k.
        prior_b = self.running_baseline
        if not self._baseline_initialized:
            self.running_baseline = batch_mean_return
            self._baseline_initialized = True
        else:
            self.running_baseline = (
                self.baseline_momentum * self.running_baseline
                + (1.0 - self.baseline_momentum) * batch_mean_return
            )
        return prior_b

    def _evaluate_single_trajectory(
        self,
        policy: DualModePolicy,
        outcome: TrajectoryOutcome,
        partitions: Sequence[ActionSpacePartition],
        target_return: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Evaluate full log-prob sum, value loss, and optional InfoPO KL bonus in a single pass."""
        device = next(policy.parameters()).device
        dtype = next(policy.parameters()).dtype
        total_log_prob = torch.zeros((), device=device, dtype=dtype)
        total_val_loss = torch.zeros((), device=device, dtype=dtype)
        info_kl_bonus = 0.0

        events = outcome.history.events
        for t_idx, ev in enumerate(events):
            h_t = _prefix_at_step(outcome.history, t_idx)
            part_t = partitions[t_idx]
            out_t = policy.forward_step(h_t, part_t)
            # Each occurred action's score is evaluated strictly once
            total_log_prob = total_log_prob + out_t.full_log_prob(ev.action)
            if out_t.value_estimate is not None:
                total_val_loss = total_val_loss + 0.5 * (
                    out_t.value_estimate - float(target_return)
                ) ** 2

            # Optional InfoPO-style intrinsic KL bonus between visible and masked next history
            if ev.mode == "I" and t_idx + 1 < len(events):
                with torch.no_grad():
                    h_next_vis = _prefix_at_step(outcome.history, t_idx + 1)
                    h_next_mask = h_next_vis.with_masked_step(t_idx)
                    part_next = partitions[t_idx + 1]
                    out_vis = policy.forward_step(h_next_vis, part_next)
                    out_mask = policy.forward_step(h_next_mask, part_next)
                    p_vis = float(out_vis.gate_prob.clamp(1e-6, 1.0 - 1e-6).item())
                    p_msk = float(out_mask.gate_prob.clamp(1e-6, 1.0 - 1e-6).item())
                    kl_gate = p_vis * (
                        torch.log(torch.tensor(p_vis / p_msk)).item()
                    ) + (1.0 - p_vis) * (
                        torch.log(torch.tensor((1.0 - p_vis) / (1.0 - p_msk))).item()
                    )
                    kl_exec = 0.0
                    if out_vis.log_probs_E is not None and out_mask.log_probs_E is not None:
                        probs_e_vis = torch.exp(out_vis.log_probs_E)
                        kl_exec = float(
                            torch.sum(probs_e_vis * (out_vis.log_probs_E - out_mask.log_probs_E)).clamp(0.0, 2.0).item()
                        )
                    info_kl_bonus += max(0.0, float(kl_gate) + 0.25 * kl_exec)

        return total_log_prob, total_val_loss, info_kl_bonus

    def compute_episode_loss(
        self,
        policy: DualModePolicy,
        sample: StratifiedBranchSample,
        batch_baseline: float = 0.0,
        method: str = "prism_evolve",
    ) -> torch.Tensor:
        """Compute differentiable scalar loss for one episode sample under `method`.

        Strict invariants for `prism_evolve` (§4.2):
        1. All return terms (G_I, G_E, bar_G, b_k) and weight p = p_{theta_k}(h_T, B_T)
           are strictly stop-gradient constants.
        2. Root gate p_theta(h_T, B_T) is differentiated in the explicit gate term
           -sg(G_I - G_E) * p_theta(h_T, B_T), and NOT duplicated inside ell_I or ell_E.
        3. Every occurred action's score is computed strictly once in the computation graph.
        """
        device = next(policy.parameters()).device
        dtype = next(policy.parameters()).dtype

        if method == "step_dpo":
            return compute_step_dpo_episode_loss(
                policy=policy,
                ref_policy=self.ref_policy,
                sample=sample,
                batch_baseline=batch_baseline,
            )

        if sample.did_split and sample.branch_I is not None and sample.branch_E is not None:
            T = sample.split_step
            G_I = float(sample.branch_I.utility)
            G_E = float(sample.branch_E.utility)
            hist_I = sample.branch_I.history
            hist_E = sample.branch_E.history
            parts_I = sample.step_partitions_I
            parts_E = sample.step_partitions_E

            # 1. Shared prefix 0 <= t < T: evaluate each shared prefix action strictly once
            ell_pre = torch.zeros((), device=device, dtype=dtype)
            for t_idx in range(T):
                h_t = _prefix_at_step(hist_I, t_idx)
                out_t = policy.forward_step(h_t, parts_I[t_idx])
                ell_pre = ell_pre + out_t.full_log_prob(hist_I.events[t_idx].action)

            # 2. Split step t = T: evaluate h_T once to obtain p_theta(h_T, B_T) and root conditional log-probs
            h_T = _prefix_at_step(hist_I, T)
            part_T = parts_I[T]
            out_T = policy.forward_step(h_T, part_T)
            p_theta_T = out_T.gate_prob
            p_detached = float(sample.root_gate_prob_detached)

            if method in ("prism_evolve", "ablation_unweighted"):
                # Root uses conditional log-probabilities log pi_{theta, I}(q) and log pi_{theta, E}(a)
                ell_I = out_T.conditional_log_prob(hist_I.events[T].action)
                ell_E = out_T.conditional_log_prob(hist_E.events[T].action)

                # Continuation steps t > T on branch I
                for t_idx in range(T + 1, len(hist_I.events)):
                    h_t = _prefix_at_step(hist_I, t_idx)
                    out_t = policy.forward_step(h_t, parts_I[t_idx])
                    ell_I = ell_I + out_t.full_log_prob(hist_I.events[t_idx].action)

                # Continuation steps t > T on branch E
                for t_idx in range(T + 1, len(hist_E.events)):
                    h_t = _prefix_at_step(hist_E, t_idx)
                    out_t = policy.forward_step(h_t, parts_E[t_idx])
                    ell_E = ell_E + out_t.full_log_prob(hist_E.events[t_idx].action)

                diff_IE = G_I - G_E
                if method == "prism_evolve":
                    w_I = p_detached * diff_IE
                    w_E = (1.0 - p_detached) * (-diff_IE)
                    bar_G = p_detached * G_I + (1.0 - p_detached) * G_E
                    return (
                        -(bar_G - float(batch_baseline)) * ell_pre
                        - diff_IE * p_theta_T
                        - w_I * ell_I
                        - w_E * ell_E
                    )
                else:
                    # Ablation 1: remove deployment gating probability weights p and 1-p
                    # (degenerates to unweighted equal-weight mixture over the two branches)
                    w_I = 0.5 * diff_IE
                    w_E = 0.5 * (-diff_IE)
                    bar_G = 0.5 * (G_I + G_E)
                    return (
                        -(bar_G - float(batch_baseline)) * ell_pre
                        - diff_IE * p_theta_T
                        - w_I * ell_I
                        - w_E * ell_E
                    )

            if method == "unstratified_2branch":
                # Unstratified 2-branch RLOO: root actions were sampled from full pi_theta
                ell_1 = out_T.full_log_prob(hist_I.events[T].action)
                ell_2 = out_T.full_log_prob(hist_E.events[T].action)
                for t_idx in range(T + 1, len(hist_I.events)):
                    h_t = _prefix_at_step(hist_I, t_idx)
                    out_t = policy.forward_step(h_t, parts_I[t_idx])
                    ell_1 = ell_1 + out_t.full_log_prob(hist_I.events[t_idx].action)
                for t_idx in range(T + 1, len(hist_E.events)):
                    h_t = _prefix_at_step(hist_E, t_idx)
                    out_t = policy.forward_step(h_t, parts_E[t_idx])
                    ell_2 = ell_2 + out_t.full_log_prob(hist_E.events[t_idx].action)

                bar_G = 0.5 * (G_I + G_E)
                adv_1 = 0.5 * (G_I - G_E)
                adv_2 = 0.5 * (G_E - G_I)
                return -(bar_G - float(batch_baseline)) * ell_pre - adv_1 * ell_1 - adv_2 * ell_2

            raise ValueError(f"Unsupported method {method!r} for two-branch sample")

        # Single-trajectory fallback / single-track baseline methods
        assert sample.single_trajectory is not None
        outcome = sample.single_trajectory
        parts = sample.step_partitions_single
        G = float(outcome.utility)
        if method == "outcome_value_baseline":
            device = next(policy.parameters()).device
            dtype = next(policy.parameters()).dtype
            loss = torch.zeros((), device=device, dtype=dtype)
            for t_idx, ev in enumerate(outcome.history.events):
                h_t = _prefix_at_step(outcome.history, t_idx)
                out_t = policy.forward_step(h_t, parts[t_idx])
                lp = out_t.full_log_prob(ev.action)
                v_est = out_t.value_estimate if out_t.value_estimate is not None else torch.tensor(batch_baseline, device=device, dtype=dtype)
                base_val = 0.5 * float(v_est.detach().cpu().item()) + 0.5 * float(batch_baseline)
                adv = G - base_val
                val_loss = 0.5 * (v_est - G) ** 2
                loss = loss - adv * lp + val_loss
            return loss

        total_lp, _, info_kl = self._evaluate_single_trajectory(
            policy=policy,
            outcome=outcome,
            partitions=parts,
            target_return=G,
        )
        if method == "info_rl_neighbor":
            effective_return = G + self.info_kl_coef * info_kl
        else:
            effective_return = G
        adv = effective_return - float(batch_baseline)
        return -adv * total_lp

    def step_batch(
        self,
        policy: DualModePolicy,
        optimizer: torch.optim.Optimizer,
        samples: Sequence[StratifiedBranchSample],
        ledger: CostLedger,
        method: str = "prism_evolve",
    ) -> dict[str, float]:
        """Perform one on-policy batch parameter update theta_k -> theta_{k+1}."""
        if not samples:
            return {"loss": 0.0, "mean_return": 0.0}

        if method in ("frozen_base", "ablation_frozen_adapter", "worldevolver", "pos_envprobe"):
            if method == "worldevolver" and hasattr(policy, "distill_from_samples"):
                policy.distill_from_samples(samples)
            mean_ret = sum(s.weighted_prefix_return for s in samples) / len(samples)
            return {"loss": 0.0, "mean_return": float(mean_ret)}

        t0 = float(os.times().elapsed)
        policy.train()
        optimizer.zero_grad(set_to_none=True)

        batch_returns = [s.weighted_prefix_return for s in samples]
        batch_mean = sum(batch_returns) / len(batch_returns)
        b_k = self._update_running_baseline(batch_mean)

        losses: list[torch.Tensor] = []
        if method == "gigpo":
            losses = compute_gigpo_batch_losses(
                policy=policy,
                samples=samples,
                batch_baseline=b_k,
            )
        elif method == "infopo":
            losses = compute_infopo_batch_losses(
                policy=policy,
                samples=samples,
                batch_baseline=b_k,
            )
        elif method == "arew":
            losses = compute_arew_batch_losses(
                policy=policy,
                samples=samples,
                batch_baseline=b_k,
            )
        elif method == "step_dpo":
            losses = [
                compute_step_dpo_episode_loss(
                    policy=policy,
                    ref_policy=self.ref_policy,
                    sample=s,
                    batch_baseline=b_k,
                )
                for s in samples
            ]
        elif method == "outcome_rloo" and len(samples) >= 2:
            # Leave-One-Out baseline across single trajectories in the batch
            total_ret = sum(batch_returns)
            n = len(samples)
            for idx, s in enumerate(samples):
                loo_baseline = (total_ret - batch_returns[idx]) / (n - 1)
                losses.append(
                    self.compute_episode_loss(
                        policy=policy,
                        sample=s,
                        batch_baseline=loo_baseline,
                        method="outcome_pg",
                    )
                )
        else:
            for s in samples:
                losses.append(
                    self.compute_episode_loss(
                        policy=policy,
                        sample=s,
                        batch_baseline=b_k,
                        method=method,
                    )
                )

        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()

        if self.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip_norm)
        optimizer.step()

        elapsed = max(0.0, float(os.times().elapsed) - t0)
        ledger.record_gradient_update(elapsed_sec=elapsed, steps=1)

        return {
            "loss": float(batch_loss.detach().cpu().item()),
            "mean_return": float(batch_mean),
            "baseline_b_k": float(b_k),
            "grad_time_sec": elapsed,
        }

    def update_batch(
        self,
        policy: DualModePolicy,
        optimizer: torch.optim.Optimizer,
        samples: Sequence[StratifiedBranchSample],
        ledger: CostLedger | None = None,
        method: str = "prism_evolve",
    ) -> dict[str, float]:
        """Alias for step_batch with optional CostLedger."""
        return self.step_batch(
            policy=policy,
            optimizer=optimizer,
            samples=samples,
            ledger=ledger if ledger is not None else CostLedger(),
            method=method,
        )


def verify_two_stage_exact_enumeration(
    p_frac: Fraction = Fraction(2, 5),
    q_inf_frac: Fraction = Fraction(4, 5),
    q_uninf_frac: Fraction = Fraction(7, 10),
    c_info_frac: Fraction = Fraction(3, 10),
    c_exec_frac: Fraction = Fraction(1, 5),
) -> dict[str, Any]:
    """Exact rational enumeration + PyTorch autograd verification for §5.3.

    Verifies:
    1. Exact expected utility J = 0.464 (58/125).
    2. Exact analytical gradient nabla J = (0.0384, 0.0448, 0.0).
    3. Exact enumeration over all stratified two-branch outcomes using PyTorch autograd
       on TwoStageParametricPolicy + StratifiedUpdater matches nabla J to machine precision.
    4. Theorem 5.2 root gating variance reduction identity:
       Var(g_{alpha, RLOO}) - Var(g_{alpha, strat}) = u * (1/2 - u) * E[D^2].
    5. Full-gradient variance sum:
       Stratified = 0.0395168 (24698 / 625000) vs. RLOO = 0.0711904 (44494 / 625000),
       expected pair cost (0.50 B vs. 0.48 B), and variance * cost product
       (0.0197584 B vs. 0.034171392 B).
    6. Finite-sample scarcity at p = 0.02: P(N_I = 0) = (1 - 0.02)^2 = 0.9604.
    """
    one = Fraction(1, 1)
    half = Fraction(1, 2)

    r_I = one - c_info_frac  # 7/10 = 0.7
    r_E = one - c_exec_frac  # 4/5 = 0.8

    V_I = r_I * q_inf_frac  # 7/10 * 4/5 = 14/25 = 0.56
    V_E = r_E * half  # 4/5 * 1/2 = 2/5 = 0.40
    J_exact = p_frac * V_I + (one - p_frac) * V_E  # 58/125 = 0.464

    u_frac = p_frac * (one - p_frac)  # 6/25 = 0.24
    v_inf = q_inf_frac * (one - q_inf_frac)  # 4/25 = 0.16
    v_uninf = q_uninf_frac * (one - q_uninf_frac)  # 21/100 = 0.21

    # Analytical true gradient nabla J
    grad_alpha_true = u_frac * (V_I - V_E)  # 24/625 = 0.0384
    grad_beta_true = p_frac * r_I * v_inf  # 28/625 = 0.0448
    grad_gamma_true = Fraction(0, 1)  # 0.0

    # Enumerate all 8 stratified two-branch outcomes:
    # Branch I outcome: y_I in {1, 0} with prob {q_inf, 1 - q_inf}, G_I = r_I * y_I, score_beta = y_I - q_inf
    # Branch E outcome: a_E in {0, 1} with prob {q_uninf, 1 - q_uninf},
    #                   y_E in {1, 0} with prob {1/2, 1/2}, G_E = r_E * y_E, score_gamma = (1 - a_E) - q_uninf
    strat_outcomes: list[tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]] = []
    for y_I in (1, 0):
        prob_I = q_inf_frac if y_I == 1 else (one - q_inf_frac)
        G_I = r_I * y_I
        score_beta = Fraction(y_I, 1) - q_inf_frac
        for a_E in (0, 1):
            prob_aE = q_uninf_frac if a_E == 0 else (one - q_uninf_frac)
            score_gamma = (one if a_E == 0 else Fraction(0, 1)) - q_uninf_frac
            for y_E in (1, 0):
                prob_yE = half
                G_E = r_E * y_E
                weight = prob_I * prob_aE * prob_yE
                D = G_I - G_E
                g_alpha = u_frac * D
                g_beta = p_frac * D * score_beta
                g_gamma = (one - p_frac) * (-D) * score_gamma
                strat_outcomes.append((weight, g_alpha, g_beta, g_gamma, G_I, G_E))

    mean_g_alpha_strat = sum(w * ga for w, ga, _, _, _, _ in strat_outcomes)
    mean_g_beta_strat = sum(w * gb for w, _, gb, _, _, _ in strat_outcomes)
    mean_g_gamma_strat = sum(w * gg for w, _, _, gg, _, _ in strat_outcomes)

    var_g_alpha_strat = sum(w * (ga - mean_g_alpha_strat) ** 2 for w, ga, _, _, _, _ in strat_outcomes)
    var_g_beta_strat = sum(w * (gb - mean_g_beta_strat) ** 2 for w, _, gb, _, _, _ in strat_outcomes)
    var_g_gamma_strat = sum(w * (gg - mean_g_gamma_strat) ** 2 for w, _, _, gg, _, _ in strat_outcomes)
    var_sum_strat = var_g_alpha_strat + var_g_beta_strat + var_g_gamma_strat

    E_D2 = sum(w * (G_I - G_E) ** 2 for w, _, _, _, G_I, G_E in strat_outcomes)

    # Enumerate same-prefix (shared state s in {0, 1}) 2-sample RLOO distribution
    # 2-sample RLOO estimator: g_hat = 1/2 * (G_1 - G_2) * (score_1 - score_2)
    mean_g_alpha_rloo = Fraction(0, 1)
    mean_g_beta_rloo = Fraction(0, 1)
    mean_g_gamma_rloo = Fraction(0, 1)
    second_alpha_rloo = Fraction(0, 1)
    second_beta_rloo = Fraction(0, 1)
    second_gamma_rloo = Fraction(0, 1)

    for s_hidden in (0, 1):
        outcomes_s: list[tuple[Fraction, Fraction, Fraction, Fraction, Fraction]] = []
        # Mode I trajectories
        for y_I in (1, 0):
            prob_y = q_inf_frac if y_I == 1 else (one - q_inf_frac)
            w = p_frac * prob_y
            G = r_I * y_I
            sa = one - p_frac
            sb = Fraction(y_I, 1) - q_inf_frac
            sg = Fraction(0, 1)
            outcomes_s.append((w, G, sa, sb, sg))
        # Mode E trajectories on shared state s_hidden
        for a_E in (0, 1):
            prob_aE = q_uninf_frac if a_E == 0 else (one - q_uninf_frac)
            w = (one - p_frac) * prob_aE
            y_E = 1 if a_E == s_hidden else 0
            G = r_E * y_E
            sa = -p_frac
            sb = Fraction(0, 1)
            sg = (one if a_E == 0 else Fraction(0, 1)) - q_uninf_frac
            outcomes_s.append((w, G, sa, sb, sg))

        for w1, G1, sa1, sb1, sg1 in outcomes_s:
            for w2, G2, sa2, sb2, sg2 in outcomes_s:
                w12 = half * w1 * w2
                dG = half * (G1 - G2)
                ga = dG * (sa1 - sa2)
                gb = dG * (sb1 - sb2)
                gg = dG * (sg1 - sg2)
                mean_g_alpha_rloo += w12 * ga
                mean_g_beta_rloo += w12 * gb
                mean_g_gamma_rloo += w12 * gg
                second_alpha_rloo += w12 * ga * ga
                second_beta_rloo += w12 * gb * gb
                second_gamma_rloo += w12 * gg * gg

    var_g_alpha_rloo = second_alpha_rloo - mean_g_alpha_rloo ** 2
    var_g_beta_rloo = second_beta_rloo - mean_g_beta_rloo ** 2
    var_g_gamma_rloo = second_gamma_rloo - mean_g_gamma_rloo ** 2
    var_sum_rloo = var_g_alpha_rloo + var_g_beta_rloo + var_g_gamma_rloo

    # Theorem 5.2 identity check:
    # Var(g_{alpha, RLOO}) - Var(g_{alpha, strat}) == u * (1/2 - u) * E[D^2]
    lhs_thm52 = var_g_alpha_rloo - var_g_alpha_strat
    rhs_thm52 = u_frac * (half - u_frac) * E_D2

    cost_strat = c_info_frac + c_exec_frac  # 0.50
    cost_rloo = 2 * (p_frac * c_info_frac + (one - p_frac) * c_exec_frac)  # 0.48
    var_cost_strat = var_sum_strat * cost_strat  # 0.0197584
    var_cost_rloo = var_sum_rloo * cost_rloo  # 0.034171392

    # Verify PyTorch autograd through TwoStageParametricPolicy and StratifiedUpdater
    policy = TwoStageParametricPolicy(
        p_init=float(p_frac),
        q_informed_init=float(q_inf_frac),
        q_uninformed_init=float(q_uninf_frac),
        dtype=torch.float64,
    )
    updater = StratifiedUpdater(grad_clip_norm=0.0)

    probe_action = CandidateAction(action_id="probe_state", intent="probe", tool_name="inspect_state")
    sub0_action = CandidateAction(action_id="submit_0", intent="submit", tool_name="submit_guess", arguments={"guess": 0})
    sub1_action = CandidateAction(action_id="submit_1", intent="submit", tool_name="submit_guess", arguments={"guess": 1})
    root_part = ActionSpacePartition.from_candidates([probe_action, sub0_action, sub1_action])
    stage2_part = ActionSpacePartition.from_candidates([sub0_action, sub1_action])

    from .event_log import EventRecord

    autograd_grad_accum = torch.zeros(3, dtype=torch.float64)
    for s_hidden in (0, 1):
        base_hist = PublicHistory(
            task_id="verify",
            goal="Verify exact gradient",
            initial_observation="Hidden state unrevealed",
            total_budget=1.0,
        )
        for y_I in (1, 0):
            prob_I = float(q_inf_frac if y_I == 1 else (one - q_inf_frac))
            # If y_I == 1, informed action matches s_hidden; else mismatches
            act_I_stage2 = sub0_action if ((s_hidden == 0 and y_I == 1) or (s_hidden == 1 and y_I == 0)) else sub1_action
            ev_I0 = EventRecord(step_index=0, action=probe_action, observation=f"Probe result: state={s_hidden}", step_cost=0.1)
            ev_I1 = EventRecord(step_index=1, action=act_I_stage2, observation=f"Submitted, success={y_I}", step_cost=0.2)
            hist_I = base_hist.append(ev_I0).append(ev_I1)
            out_I = TrajectoryOutcome.from_history(hist_I, native_success=y_I, terminated_by_submit=True)

            for a_E in (0, 1):
                prob_aE = float(q_uninf_frac if a_E == 0 else (one - q_uninf_frac))
                act_E = sub0_action if a_E == 0 else sub1_action
                y_E = 1 if a_E == s_hidden else 0
                ev_E0 = EventRecord(step_index=0, action=act_E, observation=f"Submitted, success={y_E}", step_cost=0.2)
                hist_E = base_hist.append(ev_E0)
                out_E = TrajectoryOutcome.from_history(hist_E, native_success=y_E, terminated_by_submit=True)

                sample = StratifiedBranchSample(
                    collection_mode="stratified",
                    split_step=0,
                    did_split=True,
                    prefix_history=base_hist,
                    root_gate_prob_detached=float(p_frac),
                    branch_I=out_I,
                    branch_E=out_E,
                    single_trajectory=None,
                    step_partitions_I=(root_part, stage2_part),
                    step_partitions_E=(root_part,),
                )
                w = 0.5 * prob_I * prob_aE
                policy.zero_grad()
                loss = updater.compute_episode_loss(policy, sample, batch_baseline=0.0, method="prism_evolve")
                loss.backward()
                # Note: -nabla L_k(theta) is the ascent gradient estimator g_hat_branch
                g_vec = torch.tensor(
                    [
                        -float(policy.alpha.grad.item()),
                        -float(policy.beta.grad.item()),
                        -float(policy.gamma.grad.item()),
                    ],
                    dtype=torch.float64,
                )
                autograd_grad_accum += w * g_vec

    # Scarcity diagnostic at p = 0.02
    p_low = Fraction(1, 50)
    prob_no_info_n2 = (one - p_low) ** 2  # (49/50)^2 = 2401/2500 = 0.9604

    return {
        "J_exact_fraction": str(J_exact),
        "J_exact": float(J_exact),
        "analytical_gradient": (
            float(grad_alpha_true),
            float(grad_beta_true),
            float(grad_gamma_true),
        ),
        "enumerated_stratified_gradient": (
            float(mean_g_alpha_strat),
            float(mean_g_beta_strat),
            float(mean_g_gamma_strat),
        ),
        "enumerated_rloo_gradient": (
            float(mean_g_alpha_rloo),
            float(mean_g_beta_rloo),
            float(mean_g_gamma_rloo),
        ),
        "pytorch_autograd_gradient": (
            round(float(autograd_grad_accum[0].item()), 12),
            round(float(autograd_grad_accum[1].item()), 12),
            round(float(autograd_grad_accum[2].item()), 12),
        ),
        "theorem_5_2_lhs": float(lhs_thm52),
        "theorem_5_2_rhs": float(rhs_thm52),
        "theorem_5_2_exact_match": lhs_thm52 == rhs_thm52,
        "var_sum_stratified": float(var_sum_strat),
        "var_sum_rloo": float(var_sum_rloo),
        "cost_stratified": float(cost_strat),
        "cost_rloo": float(cost_rloo),
        "var_cost_product_stratified": float(var_cost_strat),
        "var_cost_product_rloo": float(var_cost_rloo),
        "p_002_no_info_branch_prob": float(prob_no_info_n2),
    }
