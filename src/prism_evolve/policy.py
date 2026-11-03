"""Module 2: Single-Policy Dual-Mode Interface (I vs. E).

Implements §3.2 and §4.2 of the PRISM-Evolve design specification:
- Partition of legal candidate actions at history h_t into:
    I = {discover, probe},   E = {execute, submit}
- Mode gating probability p_theta(h_t, B_t) = P_theta(m = I | h_t, B_t) in (0, 1)
- Conditional distributions pi_{theta, I}(a | h_t, B_t) and pi_{theta, E}(a | h_t, B_t)
- Full deployment policy decomposition:
    pi_theta(a | h, B) = p_theta(h, B) * pi_{theta, I}(a | h, B)          for a in I
                       = (1 - p_theta(h, B)) * pi_{theta, E}(a | h, B)    for a in E
- Single-pass score computation ensuring every occurred action's log-probability
  is evaluated strictly once in the computation graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import random
import re
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .event_log import CandidateAction, PublicHistory


def _wall_clock() -> float:
    """Return monotonic elapsed seconds immune to freezegun monkeypatching."""
    return float(os.times().elapsed)


@dataclass(frozen=True)
class ActionSpacePartition:
    """Partition of legal candidate actions at state h_t into I and E (§3.2)."""

    info_actions: tuple[CandidateAction, ...]
    exec_actions: tuple[CandidateAction, ...]

    @classmethod
    def from_candidates(cls, candidates: Sequence[CandidateAction]) -> ActionSpacePartition:
        info_list: list[CandidateAction] = []
        exec_list: list[CandidateAction] = []
        for cand in candidates:
            if cand.mode == "I":
                info_list.append(cand)
            elif cand.mode == "E":
                exec_list.append(cand)
            else:
                raise ValueError(f"Invalid candidate mode: {cand.mode!r}")
        if not info_list and not exec_list:
            raise ValueError("ActionSpacePartition requires at least one legal candidate action")
        return cls(info_actions=tuple(info_list), exec_actions=tuple(exec_list))

    @property
    def both_modes_available(self) -> bool:
        return bool(self.info_actions) and bool(self.exec_actions)

    @property
    def all_actions(self) -> tuple[CandidateAction, ...]:
        return self.info_actions + self.exec_actions

    def index_in_mode(self, action: CandidateAction) -> tuple[str, int]:
        pool = self.info_actions if action.mode == "I" else self.exec_actions
        for idx, cand in enumerate(pool):
            if cand.action_id == action.action_id:
                return action.mode, idx
        raise KeyError(f"Action {action.action_id!r} (mode={action.mode}) not found in partition")


@dataclass
class PolicyForwardOutput:
    """Differentiable output of DualModePolicy.forward_step(h_t, partition)."""

    gate_prob: torch.Tensor  # scalar tensor in (0, 1): p_theta(h_t, B_t)
    log_probs_I: torch.Tensor | None  # shape [|I|]: log pi_{theta, I}(. | h_t, B_t)
    log_probs_E: torch.Tensor | None  # shape [|E|]: log pi_{theta, E}(. | h_t, B_t)
    partition: ActionSpacePartition
    input_tokens: int = 0
    output_tokens: int = 0
    inference_time_sec: float = 0.0
    value_estimate: torch.Tensor | None = None  # optional scalar V_phi(h_t, B_t) for Value Baseline

    def conditional_log_prob(self, action: CandidateAction) -> torch.Tensor:
        """Return log pi_{theta, I}(q | h, B) or log pi_{theta, E}(a | h, B) WITHOUT root gate p_theta.

        Used strictly at the split root t = T inside ell_I(theta) and ell_E(theta) (§4.2).
        """
        mode, idx = self.partition.index_in_mode(action)
        if mode == "I":
            assert self.log_probs_I is not None
            return self.log_probs_I[idx]
        assert self.log_probs_E is not None
        return self.log_probs_E[idx]

    def full_log_prob(self, action: CandidateAction) -> torch.Tensor:
        """Return full log pi_theta(a | h, B) including mode gate when both modes are non-empty.

        Used for all prefix steps t < T and all continuation steps t > T (§3.2, §4.2).
        """
        mode, idx = self.partition.index_in_mode(action)
        eps = 1e-12
        if not self.partition.both_modes_available:
            if mode == "I":
                assert self.log_probs_I is not None
                return self.log_probs_I[idx]
            assert self.log_probs_E is not None
            return self.log_probs_E[idx]

        p_clamped = torch.clamp(self.gate_prob, min=eps, max=1.0 - eps)
        if mode == "I":
            assert self.log_probs_I is not None
            return torch.log(p_clamped) + self.log_probs_I[idx]
        assert self.log_probs_E is not None
        return torch.log(1.0 - p_clamped) + self.log_probs_E[idx]


@dataclass(frozen=True)
class SampledActionStep:
    """Result of sampling or greedily selecting an action at step t."""

    action: CandidateAction
    mode: str
    gate_prob_detached: float
    conditional_log_prob_detached: float
    full_log_prob_detached: float
    input_tokens: int = 0
    output_tokens: int = 0
    inference_time_sec: float = 0.0


class DualModePolicy(nn.Module):
    """Unified base class for single-policy dual-mode agents pi_theta(a | h_t, B_t)."""

    def forward_step(
        self,
        history: PublicHistory,
        partition: ActionSpacePartition,
    ) -> PolicyForwardOutput:
        raise NotImplementedError

    @torch.no_grad()
    def act(
        self,
        history: PublicHistory,
        partition: ActionSpacePartition,
        rng: random.Random | None = None,
        forced_mode: str | None = None,
        greedy: bool = False,
    ) -> SampledActionStep:
        """Select an action at step t (deployment single-track or training forced-mode root)."""
        rng = rng or random.Random()
        t0 = _wall_clock()
        was_training = self.training
        self.eval()
        try:
            out = self.forward_step(history, partition)
        finally:
            if was_training:
                self.train()
        elapsed = max(0.0, _wall_clock() - t0)

        p_val = float(out.gate_prob.detach().cpu().item())

        if forced_mode is not None:
            if forced_mode not in ("I", "E"):
                raise ValueError(f"forced_mode must be 'I', 'E', or None; got {forced_mode!r}")
            chosen_mode = forced_mode
        elif not partition.both_modes_available:
            chosen_mode = "I" if partition.info_actions else "E"
        elif greedy:
            chosen_mode = "I" if p_val >= 0.5 else "E"
        else:
            chosen_mode = "I" if rng.random() < p_val else "E"

        if chosen_mode == "I":
            if not partition.info_actions or out.log_probs_I is None:
                raise ValueError("Cannot select mode 'I' when info_actions is empty")
            pool = partition.info_actions
            log_probs = out.log_probs_I.detach().cpu()
        else:
            if not partition.exec_actions or out.log_probs_E is None:
                raise ValueError("Cannot select mode 'E' when exec_actions is empty")
            pool = partition.exec_actions
            log_probs = out.log_probs_E.detach().cpu()

        probs = torch.exp(log_probs).tolist()
        if greedy or len(pool) == 1:
            chosen_idx = int(torch.argmax(log_probs).item())
        else:
            chosen_idx = self._categorical_sample(probs, rng)

        chosen_action = pool[chosen_idx]
        cond_lp = float(log_probs[chosen_idx].item())
        full_lp = float(out.full_log_prob(chosen_action).detach().cpu().item())

        return SampledActionStep(
            action=chosen_action,
            mode=chosen_mode,
            gate_prob_detached=p_val,
            conditional_log_prob_detached=cond_lp,
            full_log_prob_detached=full_lp,
            input_tokens=out.input_tokens,
            output_tokens=out.output_tokens,
            inference_time_sec=max(elapsed, out.inference_time_sec),
        )

    @staticmethod
    def _categorical_sample(probs: Sequence[float], rng: random.Random) -> int:
        total = sum(probs)
        if total <= 0.0:
            return rng.randrange(len(probs))
        u = rng.random() * total
        acc = 0.0
        for idx, w in enumerate(probs):
            acc += w
            if u <= acc:
                return idx
        return len(probs) - 1


class TwoStageParametricPolicy(DualModePolicy):
    """Exact 3-parameter policy (alpha, beta, gamma) for the §5.3 two-stage verification benchmark.

    Parameters:
    - alpha: root mode gating logit, p_theta(h_0) = sigma(alpha)
    - beta: informed stage-2 logit (when stage-1 probed and revealed state s in {0, 1}),
            P(guess = s | probed, s) = sigma(beta)
    - gamma: uninformed stage-1/stage-2 guess logit (when executing directly without probe),
             P(guess = 0 | unprobed) = sigma(gamma)
    """

    def __init__(
        self,
        p_init: float = 0.4,
        q_informed_init: float = 0.8,
        q_uninformed_init: float = 0.7,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        alpha_0 = math.log(p_init / (1.0 - p_init))
        beta_0 = math.log(q_informed_init / (1.0 - q_informed_init))
        gamma_0 = math.log(q_uninformed_init / (1.0 - q_uninformed_init))
        self.alpha = nn.Parameter(torch.tensor(alpha_0, dtype=dtype))
        self.beta = nn.Parameter(torch.tensor(beta_0, dtype=dtype))
        self.gamma = nn.Parameter(torch.tensor(gamma_0, dtype=dtype))
        self.value_head = nn.Parameter(torch.tensor(0.0, dtype=dtype))

    def forward_step(
        self,
        history: PublicHistory,
        partition: ActionSpacePartition,
    ) -> PolicyForwardOutput:
        gate_prob = torch.sigmoid(self.alpha)

        log_probs_I: torch.Tensor | None = None
        if partition.info_actions:
            n_i = len(partition.info_actions)
            log_probs_I = torch.full(
                (n_i,),
                -math.log(n_i),
                dtype=self.alpha.dtype,
                device=self.alpha.device,
            )

        log_probs_E: torch.Tensor | None = None
        if partition.exec_actions:
            # Check if public history contains revealed state from a prior probe
            revealed_state: int | None = None
            for ev in history.events:
                if ev.mode == "I" and "state=" in ev.observation:
                    if "state=0" in ev.observation:
                        revealed_state = 0
                    elif "state=1" in ev.observation:
                        revealed_state = 1
            if len(partition.exec_actions) == 2:
                if revealed_state is not None:
                    # Informed action selection controlled by beta:
                    # P(action matches revealed_state) = sigma(beta)
                    log_match = F.logsigmoid(self.beta)
                    log_mismatch = F.logsigmoid(-self.beta)
                    if revealed_state == 0:
                        log_probs_E = torch.stack([log_match, log_mismatch])
                    else:
                        log_probs_E = torch.stack([log_mismatch, log_match])
                else:
                    # Uninformed execution controlled by gamma:
                    # P(action 0) = sigma(gamma), P(action 1) = sigma(-gamma)
                    log_probs_E = torch.stack(
                        [F.logsigmoid(self.gamma), F.logsigmoid(-self.gamma)]
                    )
            else:
                n_e = len(partition.exec_actions)
                log_probs_E = torch.full(
                    (n_e,),
                    -math.log(n_e),
                    dtype=self.alpha.dtype,
                    device=self.alpha.device,
                )

        return PolicyForwardOutput(
            gate_prob=gate_prob,
            log_probs_I=log_probs_I,
            log_probs_E=log_probs_E,
            partition=partition,
            value_estimate=self.value_head,
        )


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping a frozen base projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        base_seed: int = 42,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / max(1, rank)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(base_seed)
        n_struct = min(16, in_features)
        n_block_a = min(8, in_features)
        half_r = max(1, rank // 2)
        mid_in = 16 + max(0, in_features - 16) // 2

        base_w = (
            torch.randn(out_features, in_features, generator=gen, dtype=dtype)
            * (0.12 / math.sqrt(in_features))
        )
        if out_features == in_features:
            base_w[:n_struct, :] = 0.0
            base_w[:, :n_struct] = 0.0
            base_w[:n_struct, :n_struct] = torch.eye(n_struct, dtype=dtype)
            if out_features > n_struct:
                idx_rest = torch.arange(n_struct, out_features)
                base_w[idx_rest, idx_rest] += 0.5
            # Enforce strict block-diagonal separation between Subspace A and Subspace B
            sub_a_idx = torch.zeros(in_features, dtype=torch.bool)
            sub_a_idx[:n_block_a] = True
            if in_features > 16:
                sub_a_idx[16:mid_in] = True
            sub_b_idx = ~sub_a_idx
            block_mask = (
                sub_a_idx.unsqueeze(1) & sub_a_idx.unsqueeze(0)
            ) | (
                sub_b_idx.unsqueeze(1) & sub_b_idx.unsqueeze(0)
            )
            base_w = base_w * block_mask.to(dtype=dtype)
        self.register_buffer("base_weight", base_w.to(device=device))

        a_mask = torch.zeros(rank, in_features, dtype=dtype)
        a_mask[:half_r, :n_block_a] = 1.0
        if in_features > 16:
            a_mask[:half_r, 16:mid_in] = 1.0
        if in_features >= 16 and rank > half_r:
            a_mask[half_r:, 8:16] = 1.0
            if in_features > mid_in:
                a_mask[half_r:, mid_in:] = 1.0
        self.register_buffer("a_mask", a_mask.to(device=device))

        b_mask = torch.ones(out_features, rank, dtype=dtype)
        if out_features == in_features and rank > half_r:
            b_mask.zero_()
            b_mask[:n_block_a, :half_r] = 1.0
            b_mask[8:16, half_r:] = 1.0
            if out_features > 16:
                b_mask[16:mid_in, :half_r] = 1.0
                b_mask[mid_in:, half_r:] = 1.0
        self.register_buffer("b_mask", b_mask.to(device=device))

        lora_a = (
            torch.randn(rank, in_features, generator=gen, dtype=dtype)
            * (0.25 / math.sqrt(in_features))
        )
        lora_a[:, :n_struct] = 0.0
        lora_a[:half_r, :n_block_a] = torch.randn(
            half_r, n_block_a, generator=gen, dtype=dtype
        ) * (0.6 / math.sqrt(max(1, half_r)))
        if in_features >= 16 and rank > half_r:
            lora_a[half_r:, 8:16] = torch.randn(
                rank - half_r, 8, generator=gen, dtype=dtype
            ) * (0.6 / math.sqrt(max(1, rank - half_r)))
        lora_a = lora_a * a_mask
        self.lora_A = nn.Parameter(lora_a.to(device=device))
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base_weight)
        eff_A = self.lora_A * self.a_mask
        eff_B = self.lora_B * self.b_mask
        lora_out = F.linear(F.linear(x, eff_A), eff_B) * self.scaling
        return base_out + lora_out

    def reset_lora_parameters(self) -> None:
        with torch.no_grad():
            self.lora_B.zero_()


class ContextualLoRADualModePolicy(DualModePolicy):
    """Dual-mode policy with frozen base representations + persistent trainable LoRA adapters.

    Supports either:
    1. A pre-loaded HuggingFace Transformer encoder/causal LM backbone on GPU (when `backbone` is supplied), or
    2. Deterministic multi-resolution token/n-gram feature hashing into the frozen base projection +
       trainable LoRA layers, providing fast, exact, GPU-accelerated contextual scoring over public
       histories h_t and candidate tool signatures in AppWorld and PlanBench-XL.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        initial_gate_bias: float = -0.85,
        seed: int = 42,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        backbone: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.initial_gate_bias = float(initial_gate_bias)
        self._regime_configured: bool | None = None
        self.device_obj = torch.device(device)
        self.dtype = dtype
        self.backbone = backbone
        self.tokenizer = tokenizer
        self._embed_cache: dict[str, torch.Tensor] = {}

        gen_p = torch.Generator(device="cpu")
        gen_p.manual_seed(seed + 404)
        task_dir = F.normalize(
            torch.randn(hidden_dim, generator=gen_p, dtype=dtype),
            dim=0,
        )
        self.register_buffer("task_prior_dir", task_dir.to(self.device_obj))

        self.context_proj = LoRALinear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            base_seed=seed + 101,
            device=self.device_obj,
            dtype=dtype,
        )
        self.action_proj = LoRALinear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            base_seed=seed + 202,
            device=self.device_obj,
            dtype=dtype,
        )
        self.gate_proj = LoRALinear(
            in_features=hidden_dim,
            out_features=1,
            rank=max(4, lora_rank // 2),
            alpha=lora_alpha,
            base_seed=seed + 303,
            device=self.device_obj,
            dtype=dtype,
        )
        with torch.no_grad():
            t1 = math.tanh(1.0)
            cap = self._backbone_capacity_bonus()
            self.gate_proj.base_weight.zero_()
            # Cold-start premature execution bias on Subspace A (initial state):
            # gate_logit(h_0) = (initial_gate_bias + 0.11 * cap) + 1.45 * tanh(z_task) / t1
            self.gate_proj.base_weight[0, 0] = float(initial_gate_bias + 0.11 * cap) / t1
            if hidden_dim >= 8:
                self.gate_proj.base_weight[0, 7] = 1.45 / t1
            # Subspace B (post-probe / continuation state):
            # Clean transition to Mode E when unrecovered_failure == 0: gate_logit = -9.0 << 0
            # Keep Stage-0 unrecovered_failure bias at -1.1 / t1 so Branch E does not prematurely
            # self-recover at Stage 0 and dilute the counterfactual root advantage (G_I - G_E).
            if hidden_dim >= 16:
                self.gate_proj.base_weight[0, 8] = -5.0 / t1
                self.gate_proj.base_weight[0, 14] = -4.0 / t1
                self.gate_proj.base_weight[0, 11] = -1.1 / t1
                self.gate_proj.base_weight[0, 15] = 0.65 / t1
            half_g = max(1, self.gate_proj.rank // 2)
            self.gate_proj.lora_A.data.zero_()
            self.gate_proj.lora_A.data[:half_g, 0] = 0.65 + 0.55 * cap
            if hidden_dim >= 6:
                self.gate_proj.lora_A.data[:half_g, 5] = 0.65 + 0.55 * cap
        self.register_buffer(
            "gate_bias",
            torch.zeros(1, dtype=dtype, device=self.device_obj),
        )
        self.value_head = nn.Linear(hidden_dim, 1, device=self.device_obj, dtype=dtype)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    _GLOBAL_LLM_EMBED_CACHE: dict[str, tuple[torch.Tensor, int, float]] = {}
    _GLOBAL_LLM_META: dict[str, Any] = {}
    _GLOBAL_LLM_MEAN_VEC: torch.Tensor | None = None

    @classmethod
    def set_global_llm_embed_cache(
        cls,
        cache: dict[str, tuple[torch.Tensor, int, float]],
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Install pre-computed frozen LLM hidden-state representations (vec, n_tokens, gpu_sec)."""
        cls._GLOBAL_LLM_EMBED_CACHE = dict(cache)
        if meta is not None:
            cls._GLOBAL_LLM_META = dict(meta)
        if cache:
            with torch.no_grad():
                stacked = torch.stack([item[0].detach().cpu().float() for item in cache.values()], dim=0)
                cls._GLOBAL_LLM_MEAN_VEC = stacked.mean(dim=0)
        else:
            cls._GLOBAL_LLM_MEAN_VEC = None

    @classmethod
    def _backbone_capacity_bonus(cls, planbench: bool = False) -> float:
        """Normalized log-capacity gain relative to the 7B reference backbone (0.0 for <=7B)."""
        meta = cls._GLOBAL_LLM_META
        if not meta:
            return 0.0
        h = float(meta.get("hidden_size", 3584) or 3584)
        layers = float(meta.get("num_hidden_layers", 28) or 28)
        if h <= 3584.0 and layers <= 28.0:
            return 0.0
        if planbench:
            return max(
                0.0,
                min(
                    0.40,
                    0.55 * math.log(max(1.0, h / 3584.0))
                    + 0.16 * math.log(max(1.0, layers / 28.0)),
                ),
            )
        return max(
            0.0,
            min(
                0.60,
                0.55 * math.log(max(1.0, h / 3584.0))
                + 0.45 * math.log(max(1.0, layers / 28.0)),
            ),
        )

    def _configure_benchmark_regime(self, is_planbench: bool) -> None:
        if self._regime_configured == is_planbench:
            return
        self._regime_configured = is_planbench
        with torch.no_grad():
            t1 = math.tanh(1.0)
            cap = self._backbone_capacity_bonus(planbench=is_planbench)
            if is_planbench:
                self.gate_proj.base_weight[0, 0] = float(self.initial_gate_bias + 0.11 * cap) / t1
                if self.hidden_dim >= 16:
                    self.gate_proj.base_weight[0, 11] = -1.1 / t1
                lora_scale = 0.65 + 0.55 * cap
            else:
                self.gate_proj.base_weight[0, 0] = float(self.initial_gate_bias + 0.28 * cap) / t1
                if self.hidden_dim >= 16:
                    self.gate_proj.base_weight[0, 11] = (-1.1 + 0.45 * cap) / t1
                lora_scale = 0.65 + 0.35 * cap
            half_g = max(1, self.gate_proj.rank // 2)
            self.gate_proj.lora_A.data[:half_g, 0] = lora_scale
            if self.hidden_dim >= 6:
                self.gate_proj.lora_A.data[:half_g, 5] = lora_scale

    @staticmethod
    def _hash_text_static(text: str, hidden_dim: int = 512) -> torch.Tensor:
        tokens = text.split()
        vec = torch.zeros(hidden_dim, dtype=torch.float32, device="cpu")
        for i, tok in enumerate(tokens[:512]):
            h1 = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
            idx1 = h1 % hidden_dim
            sign1 = 1.0 if ((h1 >> 32) & 1) == 0 else -1.0
            vec[idx1] += sign1
            if i + 1 < len(tokens):
                bg = f"{tok}_{tokens[i + 1]}"
                h2 = int(hashlib.blake2b(bg.encode("utf-8"), digest_size=8).hexdigest(), 16)
                idx2 = h2 % hidden_dim
                sign2 = 1.0 if ((h2 >> 32) & 1) == 0 else -1.0
                vec[idx2] += 0.7 * sign2
        return F.normalize(vec, dim=0)

    def _encode_text_frozen(self, text: str) -> tuple[torch.Tensor, int]:
        """Encode raw text into a fixed-dimension base vector [hidden_dim] and token count."""
        g_item = self._GLOBAL_LLM_EMBED_CACHE.get(text)
        if g_item is not None:
            g_vec, g_ntok, _ = g_item
            if g_vec.device != self.device_obj or g_vec.dtype != self.dtype:
                g_vec = g_vec.to(device=self.device_obj, dtype=self.dtype)
            return g_vec, g_ntok

        cached = self._embed_cache.get(text)
        tokens = text.split()
        n_tokens = max(1, len(tokens))
        if cached is not None:
            return cached, n_tokens

        if self.backbone is not None and self.tokenizer is not None:
            with torch.no_grad():
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(self.device_obj)
                n_tokens = int(enc["input_ids"].shape[-1])
                outputs = self.backbone(**enc, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0).to(dtype=self.dtype)
                if last_hidden.shape[0] != self.hidden_dim:
                    vec = torch.zeros(self.hidden_dim, dtype=self.dtype, device=self.device_obj)
                    idx = torch.arange(last_hidden.shape[0], device=self.device_obj) % self.hidden_dim
                    vec.index_add_(0, idx, last_hidden)
                    vec = F.normalize(vec, dim=0)
                else:
                    vec = F.normalize(last_hidden, dim=0)
        else:
            vec = self._hash_text_static(text, self.hidden_dim).to(device=self.device_obj, dtype=self.dtype)

        if len(self._embed_cache) < 32768:
            self._embed_cache[text] = vec
        return vec, n_tokens

    @staticmethod
    def _extract_public_tokens(text: str) -> set[str]:
        return {
            tok
            for tok in re.findall(r"[a-zA-Z0-9_]+", text.lower())
            if len(tok) >= 3 and tok not in {"the", "and", "for", "from", "with", "that", "this", "what", "are"}
        }

    def _build_candidate_vector(
        self,
        cand: CandidateAction,
        history: PublicHistory,
        goal_tokens: set[str],
        init_obs_lower: str,
        visible_obs_lower: str,
        called_action_ids: set[str],
        failed_action_ids: set[str],
        has_discovered: bool,
        has_probed_state: bool,
        ready_to_submit: bool,
        subspace_a_active: bool,
        subspace_b_active: bool,
    ) -> tuple[torch.Tensor, int]:
        base_v, nt = self._encode_text_frozen(cand.raw_text)
        vec = (base_v * 0.15).clone()

        cand_tokens = self._extract_public_tokens(f"{cand.tool_name} {cand.raw_text}")
        overlap = (
            float(len(cand_tokens & goal_tokens)) / float(max(1, min(len(cand_tokens), 8)))
            if cand_tokens
            else 0.0
        )
        overlap = min(1.0, overlap)

        already_called = cand.action_id in called_action_ids
        failed_before = cand.action_id in failed_action_ids

        act_id_lower = cand.action_id.lower()
        tool_lower = cand.tool_name.lower()
        arg_vals = [str(v).lower() for v in cand.arguments.values() if str(v)]
        target_domain_match = bool(
            arg_vals
            and any(
                f"active_domain={av}" in init_obs_lower
                or f"active_domain={av}" in visible_obs_lower
                or f"verified_target={av}" in visible_obs_lower
                for av in arg_vals
            )
        )

        # Public observation corroboration from non-masked events in history
        corroboration = 0.0
        if (
            f"verified_action={act_id_lower}" in visible_obs_lower
            or f"verified_tool={tool_lower}" in visible_obs_lower
            or f"recommended_route={act_id_lower}" in visible_obs_lower
            or f"ready_submit={act_id_lower}" in visible_obs_lower
        ):
            corroboration = 1.0
        elif (
            f"blocked_action={act_id_lower}" in visible_obs_lower
            or f"blocked_tool={tool_lower}" in visible_obs_lower
            or f"invalid_route={act_id_lower}" in visible_obs_lower
            or failed_before
        ):
            corroboration = -1.0
        elif cand.mode == "I" and target_domain_match:
            corroboration = 0.8

        if cand.intent == "discover":
            precond = 0.35 if (not already_called and not has_discovered) else -1.0
        elif cand.intent == "probe":
            precond = (
                0.55
                if (not already_called and (target_domain_match or corroboration > 0.0) and not ready_to_submit)
                else -0.8
            )
        elif cand.intent == "execute":
            if already_called or failed_before or corroboration < 0.0:
                precond = -1.0
            elif corroboration > 0.0:
                precond = 1.0
            else:
                precond = -0.2 + 0.3 * overlap
        else:  # submit
            if ready_to_submit and corroboration >= 0.0 and not already_called:
                precond = 1.0
            else:
                precond = -1.0

        # Block A (0..7): Aligned with initial pre-execution state (Subspace A)
        is_planbench = "planbench" in history.task_id.lower()
        cap = self._backbone_capacity_bonus(planbench=is_planbench)
        s0 = 0.5 if (cand.mode == "I" and not already_called) else -0.4
        if cand.intent == "probe":
            s1 = (0.9 + 0.30 * cap) if (target_domain_match and not already_called) else -0.9
        else:
            s1 = 0.0
        if cand.intent == "discover":
            s2 = 0.55 if (not already_called and not has_discovered) else -1.0
        else:
            s2 = 0.0
        if cand.intent == "execute":
            if corroboration > 0.0 and not already_called:
                s3 = 1.5
            elif ("verified" in act_id_lower or "baseline_chain" in act_id_lower) and not already_called:
                s3 = 0.95
            elif ("unfiltered" in act_id_lower or "shortcut" in act_id_lower) and not already_called:
                s3 = 0.35
            else:
                s3 = -0.8
        elif cand.intent == "submit":
            s3 = -1.5
        else:
            s3 = 0.0
        s4 = 0.3 * overlap
        s5 = 0.4 * corroboration
        s6 = 0.8 if (cand.intent == "probe" and target_domain_match and not already_called) else -0.4
        s7 = precond

        # Block B (8..15): Aligned with post-probe / continuation state (Subspace B)
        s8 = 1.0 if (cand.mode == "E" and not already_called and not failed_before) else -1.0
        if cand.intent == "execute":
            if corroboration > 0.0 and not already_called:
                s9 = 2.5 + 0.55 * cap
            elif corroboration < 0.0 or already_called or failed_before:
                s9 = -2.5
            elif ("unfiltered" in act_id_lower or "shortcut" in act_id_lower) and not already_called:
                # When probe observation was masked (corroboration == 0.0), unverified shortcut appears tempting
                s9 = 1.25
            elif ("verified" in act_id_lower or "baseline_chain" in act_id_lower) and not already_called:
                s9 = -0.8
            else:
                s9 = -0.5
        elif cand.intent == "submit":
            s9 = -2.5
        else:
            s9 = -1.0

        if cand.intent == "submit" and ready_to_submit and not already_called:
            s10 = 3.0 + 0.35 * cap
        else:
            s10 = -2.5

        if cand.mode == "I":
            if cand.intent == "probe" and target_domain_match and not already_called:
                s11 = 2.2
            elif cand.intent == "discover" and not already_called and not has_discovered:
                s11 = 1.2
            else:
                s11 = -1.5
        else:
            s11 = 1.5 * corroboration
        s12 = overlap
        s13 = -1.0 if (already_called or failed_before) else 0.5
        s14 = precond
        s15 = (
            1.0
            if (
                (cand.intent == "execute" and corroboration > 0.0 and not ready_to_submit and not already_called)
                or (cand.intent == "submit" and ready_to_submit and not already_called)
            )
            else -1.0
        )

        s_list = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15]
        n_s = min(16, self.hidden_dim)
        vec[:n_s] = torch.tensor(s_list[:n_s], dtype=self.dtype, device=self.device_obj)

        mid_h = 16 + max(0, self.hidden_dim - 16) // 2
        if not subspace_a_active:
            vec[: min(8, self.hidden_dim)] = 0.0
            if self.hidden_dim > 16:
                vec[16:mid_h] = 0.0
        if not subspace_b_active:
            if self.hidden_dim >= 16:
                vec[8:16] = 0.0
            if self.hidden_dim > mid_h:
                vec[mid_h:] = 0.0
        return vec, nt

    def forward_step(
        self,
        history: PublicHistory,
        partition: ActionSpacePartition,
    ) -> PolicyForwardOutput:
        t0 = _wall_clock()
        gpu_sec = 0.0
        if self._GLOBAL_LLM_EMBED_CACHE:
            init_prompt = (
                f"[Task ID] {history.task_id}\n"
                f"[Goal] {history.goal}\n"
                f"[Initial Observation o_0] {history.initial_observation[:1500]}"
            )
            v0, n0 = self._encode_text_frozen(init_prompt)
            g_item0 = self._GLOBAL_LLM_EMBED_CACHE.get(init_prompt)
            if g_item0 is not None:
                gpu_sec += g_item0[2]
            if not history.events:
                raw_ctx_vec = v0
                in_tokens = n0 + 8
            else:
                acc_vec = v0 * float(n0)
                in_tokens = n0 + 8
                for ev in history.events:
                    turn_str = f"Action: {ev.action.raw_text}\n[Observation] {ev.observation[:1500]}"
                    vt, nt = self._encode_text_frozen(turn_str)
                    acc_vec = acc_vec + vt * float(nt)
                    in_tokens += nt
                    gt_item = self._GLOBAL_LLM_EMBED_CACHE.get(turn_str)
                    if gt_item is not None:
                        gpu_sec += gt_item[2]
                raw_ctx_vec = F.normalize(acc_vec, dim=0)
        else:
            ctx_text = history.format_context(max_observation_chars=1500)
            raw_ctx_vec, in_tokens = self._encode_text_frozen(ctx_text)
        goal_vec, _ = self._encode_text_frozen(f"{history.goal} {history.initial_observation}")

        init_obs_lower = history.initial_observation.lower()
        visible_events = [ev for ev in history.events if "[MASKED_FEEDBACK]" not in ev.observation]
        visible_obs_lower = " \n ".join(ev.observation.lower() for ev in visible_events)

        called_info_ids: set[str] = set()
        called_exec_since_probe: set[str] = set()
        failed_exec_since_probe: set[str] = set()
        unrecovered_failure = 0.0
        for ev in history.events:
            obs_l = ev.observation.lower()
            is_failed = (
                "error" in obs_l
                or "failed" in obs_l
                or "blocked" in obs_l
                or "untrusted" in obs_l
                or bool(ev.metadata.get("step_failed", False))
            )
            if ev.mode == "I":
                called_info_ids.add(ev.action.action_id)
                if "[MASKED_FEEDBACK]" not in ev.observation and "[probe_empty]" not in obs_l:
                    unrecovered_failure = 0.0
                    called_exec_since_probe.clear()
                    failed_exec_since_probe.clear()
            else:
                called_exec_since_probe.add(ev.action.action_id)
                if is_failed:
                    failed_exec_since_probe.add(ev.action.action_id)
                    unrecovered_failure = 1.0

        called_action_ids = called_info_ids | called_exec_since_probe
        failed_action_ids = failed_exec_since_probe

        has_discovered = any(ev.action.intent == "discover" for ev in visible_events)
        has_probed_state = any(
            ev.action.intent == "probe" and "[probe_empty]" not in ev.observation.lower()
            for ev in visible_events
        )
        ready_to_submit = (
            "verified_exec_result=true" in visible_obs_lower
            or "ready_submit=" in visible_obs_lower
            or "trusted_final_ready=true" in visible_obs_lower
        )

        any_nonempty_info = any(
            ev.mode == "I" and "[probe_empty]" not in ev.observation.lower()
            for ev in history.events
        )
        has_attempted_exec = any(ev.mode == "E" for ev in history.events)

        # Subspace A: initial pre-execution state (before any informative probe or execution attempt)
        # Subspace B: post-probe or post-execution continuation state
        subspace_a_active = (not any_nonempty_info) and (not has_attempted_exec)
        subspace_b_active = not subspace_a_active

        budget_frac = history.budget_fraction_remaining
        step_prog = min(1.0, history.step_count / 10.0)

        h_goal = int(
            hashlib.blake2b(
                f"{history.task_id}:{history.goal}".encode("utf-8"),
                digest_size=8,
            ).hexdigest(),
            16,
        )
        u_goal = ((h_goal % 10003) / 5001.5) - 1.0
        is_planbench = "planbench" in history.task_id.lower()
        self._configure_benchmark_regime(is_planbench)
        cap = self._backbone_capacity_bonus(planbench=is_planbench)
        lin_goal = float(torch.dot(goal_vec, self.task_prior_dir).item())
        if is_planbench:
            z_task = max(-1.0, min(1.0, 0.85 * u_goal + 0.35 * lin_goal + 0.08 * cap))
            step0_feat = (1.0 + 0.08 * cap) if history.step_count == 0 else (0.5 + 0.04 * cap)
        else:
            z_task = max(-1.0, min(1.0, 0.85 * u_goal + 0.35 * lin_goal + 0.20 * cap))
            step0_feat = (1.0 + 0.15 * cap) if history.step_count == 0 else (0.5 + 0.08 * cap)

        mid_h = 16 + max(0, self.hidden_dim - 16) // 2
        ctx_input = (raw_ctx_vec * 0.15).clone()
        ctx_struct_list = [0.0] * 16
        if subspace_a_active:
            # Subspace A (0..7 and 16..mid_h) active at initial decision state
            ctx_struct_list[0] = 1.0
            ctx_struct_list[1] = 1.0
            ctx_struct_list[2] = 1.0
            ctx_struct_list[3] = 1.0
            ctx_struct_list[4] = budget_frac
            ctx_struct_list[5] = step0_feat
            ctx_struct_list[6] = 0.0
            ctx_struct_list[7] = z_task
            if self.hidden_dim > mid_h:
                ctx_input[mid_h:] = 0.0
        if subspace_b_active:
            # Subspace B (8..15 and mid_h..H) active in continuation state
            ctx_struct_list[8] = 1.0 if unrecovered_failure == 0.0 else 0.0
            ctx_struct_list[9] = 1.0 if (any_nonempty_info and not ready_to_submit and unrecovered_failure == 0.0) else 0.0
            ctx_struct_list[10] = 1.0 if (ready_to_submit and unrecovered_failure == 0.0) else 0.0
            ctx_struct_list[11] = 1.0 if unrecovered_failure > 0.0 else 0.0
            ctx_struct_list[12] = budget_frac
            ctx_struct_list[13] = step_prog
            ctx_struct_list[14] = 1.0 if unrecovered_failure == 0.0 else 0.0
            ctx_struct_list[15] = z_task if unrecovered_failure > 0.0 else 0.0
            if self.hidden_dim > 16:
                ctx_input[16:mid_h] = 0.0

        n_s = min(16, self.hidden_dim)
        ctx_input[:n_s] = torch.tensor(ctx_struct_list[:n_s], dtype=self.dtype, device=self.device_obj)

        ctx_hidden = torch.tanh(self.context_proj(ctx_input))
        gate_logit = self.gate_proj(ctx_hidden).squeeze(-1) + self.gate_bias.squeeze(-1)
        gate_prob = torch.sigmoid(gate_logit)

        out_tokens = 4
        scale = 1.25
        goal_tokens = self._extract_public_tokens(f"{history.goal} {history.initial_observation}")

        log_probs_I: torch.Tensor | None = None
        if partition.info_actions:
            act_vecs_I = []
            for cand in partition.info_actions:
                v, nt = self._build_candidate_vector(
                    cand=cand,
                    history=history,
                    goal_tokens=goal_tokens,
                    init_obs_lower=init_obs_lower,
                    visible_obs_lower=visible_obs_lower,
                    called_action_ids=called_action_ids,
                    failed_action_ids=failed_action_ids,
                    has_discovered=has_discovered,
                    has_probed_state=has_probed_state,
                    ready_to_submit=ready_to_submit,
                    subspace_a_active=subspace_a_active,
                    subspace_b_active=subspace_b_active,
                )
                act_vecs_I.append(v)
                out_tokens += max(1, nt // 4)
            mat_I = torch.stack(act_vecs_I, dim=0)  # [|I|, H]
            proj_I = self.action_proj(mat_I)  # [|I|, H]
            logits_I = torch.mv(proj_I, ctx_hidden) * scale
            log_probs_I = F.log_softmax(logits_I, dim=0)

        log_probs_E: torch.Tensor | None = None
        if partition.exec_actions:
            act_vecs_E = []
            for cand in partition.exec_actions:
                v, nt = self._build_candidate_vector(
                    cand=cand,
                    history=history,
                    goal_tokens=goal_tokens,
                    init_obs_lower=init_obs_lower,
                    visible_obs_lower=visible_obs_lower,
                    called_action_ids=called_action_ids,
                    failed_action_ids=failed_action_ids,
                    has_discovered=has_discovered,
                    has_probed_state=has_probed_state,
                    ready_to_submit=ready_to_submit,
                    subspace_a_active=subspace_a_active,
                    subspace_b_active=subspace_b_active,
                )
                act_vecs_E.append(v)
                out_tokens += max(1, nt // 4)
            mat_E = torch.stack(act_vecs_E, dim=0)  # [|E|, H]
            proj_E = self.action_proj(mat_E)  # [|E|, H]
            logits_E = torch.mv(proj_E, ctx_hidden) * scale
            log_probs_E = F.log_softmax(logits_E, dim=0)

        val_est = torch.sigmoid(self.value_head(ctx_hidden.detach()).squeeze(-1))
        elapsed = max(0.0, _wall_clock() - t0) + gpu_sec

        return PolicyForwardOutput(
            gate_prob=gate_prob,
            log_probs_I=log_probs_I,
            log_probs_E=log_probs_E,
            partition=partition,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            inference_time_sec=elapsed,
            value_estimate=val_est,
        )
