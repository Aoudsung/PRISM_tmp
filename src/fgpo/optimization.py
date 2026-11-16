"""Bounded CPU-master AdamW and streaming-head PPO.

Only a vocabulary chunk keeps a loss graph.  Its hidden-state derivative is
collected before the packed Transformer graph is traversed once.  Transfer
buffers are bounded independently of the size of a parameter tensor.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from typing import Any

import torch


@dataclass(frozen=True)
class _ParameterSlice:
    name: str
    parameter: Any
    start: int
    size: int
    offset: int


class _TransferSlot:
    """One of two reusable slots; host storage is never reused during DMA."""

    def __init__(self, device, dtype, capacity, *, pin_memory):
        self.device = device
        self.capacity = int(capacity)
        self.pin_memory = bool(pin_memory)
        self.stream = torch.cuda.Stream(device=device)
        self.downloaded = torch.cuda.Event()
        self.uploaded = torch.cuda.Event()
        self.has_upload = False
        self.host_grad = torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=self.pin_memory)
        self.host_weight = torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=self.pin_memory)
        with torch.cuda.stream(self.stream):
            self.device_grad = torch.empty(capacity, dtype=dtype, device=device)
            self.device_weight = torch.empty(capacity, dtype=dtype, device=device)

    def download(self, entries, fresh):
        # A previous H2D copy may still be reading host_weight.  A stream wait
        # alone would not protect that host storage from the next CPU update.
        if self.has_upload:
            self.uploaded.synchronize()
            self.has_upload = False
        self.entries = entries
        self.length = sum(entry.size for entry in entries)
        producer = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(producer)
            initialize = False
            for entry in entries:
                section = slice(entry.offset, entry.offset + entry.size)
                source = slice(entry.start, entry.start + entry.size)
                self.device_grad[section].copy_(entry.parameter.grad.detach().view(-1)[source])
                if entry.name in fresh:
                    self.device_weight[section].copy_(entry.parameter.detach().view(-1)[source])
                    initialize = True
            self.host_grad[:self.length].copy_(self.device_grad[:self.length], non_blocking=self.pin_memory)
            if initialize:
                self.host_weight[:self.length].copy_(self.device_weight[:self.length], non_blocking=self.pin_memory)
            self.downloaded.record(self.stream)

    def upload(self):
        with torch.cuda.stream(self.stream):
            self.device_weight[:self.length].copy_(self.host_weight[:self.length], non_blocking=self.pin_memory)
            for entry in self.entries:
                section = slice(entry.offset, entry.offset + entry.size)
                destination = slice(entry.start, entry.start + entry.size)
                entry.parameter.view(-1)[destination].copy_(self.device_weight[section])
            self.uploaded.record(self.stream)
        self.has_upload = True

    def synchronize(self):
        # Completion also makes state_dict/save_model and the next forward safe.
        self.stream.synchronize()
        self.has_upload = False


class OffloadedAdamW:
    def __init__(self, model, *, lr=1e-6, weight_decay=.01, betas=(.9,.999), eps=1e-8,
                 bucket_mib=64, pin_memory=True):
        self.params = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
        self.lr, self.weight_decay, self.betas, self.eps = float(lr),float(weight_decay),tuple(betas),float(eps)
        if self.lr < 0 or self.weight_decay < 0 or self.eps <= 0 or len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas):
            raise ValueError("Invalid AdamW hyperparameters")
        self.bucket_bytes = int(float(bucket_mib) * 1024 * 1024)
        if self.bucket_bytes < 1:
            raise ValueError("optimizer bucket_mib must be positive")
        self.pin_memory = bool(pin_memory)
        self.state, self.steps = {},0
        self._transfer_slots = {}

    def zero_grad(self):
        for _,p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, max_norm=1.):
        if max_norm is not None and (not math.isfinite(max_norm) or max_norm < 0):
            raise ValueError("max_norm must be finite and nonnegative, or None")
        active = [(name, p) for name, p in self.params if p.grad is not None]
        device_norms, groups = {}, {}
        for name, parameter in active:
            gradient = parameter.grad
            if gradient.is_sparse or not parameter.is_contiguous() or not gradient.is_contiguous():
                raise ValueError("OffloadedAdamW requires dense contiguous parameters and gradients")
            if parameter.device.type not in ("cpu", "cuda"):
                raise ValueError("OffloadedAdamW supports CPU and CUDA parameters")
            groups.setdefault((parameter.device, parameter.dtype), []).append((name, parameter))
            # Norm accumulation never squares bf16 values.  The scalar norms
            # stay on their device until its whole reduction is complete.
            norm_part = torch.linalg.vector_norm(gradient.detach(), dtype=torch.float32)
            device_norms.setdefault(parameter.device, []).append(norm_part)
        squared = [torch.stack(parts).double().square().sum() for parts in device_norms.values()]
        norm = math.sqrt(sum(float(value.cpu()) for value in squared))
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        scale = min(1.,max_norm/(norm+1e-6)) if max_norm is not None else 1.
        # No master, moment, step count, or model weight is mutated until the
        # complete norm has been checked.
        fresh = set()
        for name, parameter in active:
            if name not in self.state:
                master = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
                self.state[name] = {"master": master, "exp_avg": torch.zeros_like(master),
                                    "exp_avg_sq": torch.zeros_like(master), "step": 0}
                fresh.add(name)
            self.state[name]["step"] += 1
        self.steps += 1
        for (device, dtype), parameters in groups.items():
            capacity = max(1, self.bucket_bytes // parameters[0][1].element_size())
            buckets = self._buckets(parameters, capacity)
            if device.type == "cpu":
                for entries in buckets:
                    for entry in entries:
                        part = slice(entry.start, entry.start + entry.size)
                        values = entry.parameter.detach().view(-1)[part]
                        gradient = entry.parameter.grad.detach().view(-1)[part].float().mul(scale)
                        master = self._update_slice(entry, gradient, values if entry.name in fresh else None)
                        entry.parameter.view(-1)[part].copy_(master)
                continue
            capacity = max(sum(entry.size for entry in bucket) for bucket in buckets)
            key = (device, dtype)
            if key in self._transfer_slots and self._transfer_slots[key][0].capacity < capacity:
                del self._transfer_slots[key]
            if key not in self._transfer_slots:
                self._transfer_slots[key] = [_TransferSlot(device, dtype, capacity, pin_memory=self.pin_memory)
                                             for _ in range(min(2, len(buckets)))]
            slots = self._transfer_slots[key]
            if len(slots) == 1 and len(buckets) > 1:
                slots.append(_TransferSlot(device, dtype, slots[0].capacity, pin_memory=self.pin_memory))
            try:
                for index in range(min(len(slots), len(buckets))):
                    slots[index].download(buckets[index], fresh)
                for index, entries in enumerate(buckets):
                    slot = slots[index % len(slots)]
                    slot.downloaded.synchronize()
                    # CPU kernels use PyTorch's configured thread pool.  Only
                    # this bounded bucket is converted to fp32 for arithmetic.
                    gradients = slot.host_grad[:slot.length].float().mul(scale)
                    for entry in entries:
                        section = slice(entry.offset, entry.offset + entry.size)
                        initial = slot.host_weight[section] if entry.name in fresh else None
                        master = self._update_slice(entry, gradients[section], initial)
                        slot.host_weight[section].copy_(master)
                    slot.upload()
                    following = index + len(slots)
                    if following < len(buckets):
                        slot.download(buckets[following], fresh)
            finally:
                for slot in slots:
                    slot.synchronize()
        return norm

    @staticmethod
    def _buckets(parameters, capacity):
        buckets, current, used = [], [], 0
        for name, parameter in parameters:
            start = 0
            while start < parameter.numel():
                size = min(capacity - used, parameter.numel() - start)
                current.append(_ParameterSlice(name, parameter, start, size, used))
                start, used = start + size, used + size
                if used == capacity:
                    buckets.append(current)
                    current, used = [], 0
        if current:
            buckets.append(current)
        return buckets

    def _update_slice(self, entry, gradient, initial):
        state = self.state[entry.name]
        section = slice(entry.start, entry.start + entry.size)
        master = state["master"].view(-1)[section]
        first = state["exp_avg"].view(-1)[section]
        second = state["exp_avg_sq"].view(-1)[section]
        if initial is not None:
            master.copy_(initial)
        b1, b2 = self.betas
        first.mul_(b1).add_(gradient, alpha=1 - b1)
        second.mul_(b2).addcmul_(gradient, gradient, value=1 - b2)
        denominator = second.sqrt().div_(math.sqrt(1 - b2 ** state["step"])).add_(self.eps)
        master.mul_(1 - self.lr * self.weight_decay).addcdiv_(first, denominator,
                                                          value=-self.lr / (1 - b1 ** state["step"]))
        return master

    def state_dict(self):
        return {"format":"fgpo-adamw-v1","parameter_names":[n for n,_ in self.params],
                "initialized_parameter_names":list(self.state),"state":self.state,"steps":self.steps,
                "lr":self.lr,"weight_decay":self.weight_decay,"betas":self.betas,"eps":self.eps,
                "bucket_bytes":self.bucket_bytes,"pin_memory":self.pin_memory}

    def load_state_dict(self, state):
        if state["format"] != "fgpo-adamw-v1" or state["parameter_names"] != [n for n,_ in self.params]:
            raise ValueError("Optimizer state belongs to a different exact parameter space")
        if set(state["initialized_parameter_names"]) != set(state["state"]):
            raise ValueError("Checkpoint is missing saved Adam master weights or moments")
        self.state = state["state"]
        self.steps,self.lr,self.weight_decay = state["steps"],state["lr"],state["weight_decay"]
        self.betas,self.eps = tuple(state["betas"]),state["eps"]
        self.bucket_bytes = int(state.get("bucket_bytes", self.bucket_bytes))
        self.pin_memory = bool(state.get("pin_memory", self.pin_memory))
        self._transfer_slots.clear()

    def gradient_state(self):
        return {name:p.grad.detach().to(device="cpu", copy=True) for name,p in self.params if p.grad is not None}

    def restore_gradients(self, values):
        self.zero_grad()
        params = dict(self.params)
        for name,value in values.items():
            params[name].grad = value.to(params[name])


class WeightReference:
    def __init__(self, model=None, *, trainable_only=False, state=None):
        self.weights = (state if state is not None else {name:p.detach().to(device="cpu", copy=True)
                        for name,p in model.named_parameters() if not trainable_only or p.requires_grad})

    @contextmanager
    def activate(self, model):
        current,previous = {},model.training
        try:
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in self.weights:
                        current[name] = p.detach().to(device="cpu", copy=True)
                        p.copy_(self.weights[name])
            model.eval()
            yield
        finally:
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in current:
                        p.copy_(current[name])
            model.train(previous)

    @torch.no_grad()
    def update_ema(self, model, decay=.99):
        for name,p in model.named_parameters():
            if name in self.weights:
                self.weights[name].mul_(decay).add_(p.detach().cpu(),alpha=1-decay)


@dataclass
class UpdateCursor:
    minibatch: int = 0
    segment: int = 0
    optimizer_ready: bool = False
    finished: bool = False
    denominator: float = 0.
    metrics: dict[str,Any] = field(default_factory=lambda:{"loss":0.,"entropy":0.,"reference_kl":0.,"steps":0})


def ppo_update(policy, trajectories, reference_data, advantages, optimizer, *,
               minibatch_rows=256, chunk_tokens=64, cursor=None, checkpoint_callback=None,
               constant_denominator=True, loss_normalization="token", clip_ratio=.2,
               dual_clip=3., entropy_coef=.001, kl_coef=.01, max_grad_norm=1.):
    """PPO with one live vocabulary chunk and one backbone backward per segment.

    ``trajectory`` sums all response-token scores along each trajectory and
    divides by a fixed trajectory count; it does not divide by trajectory
    length.  ``token`` preserves the previous token-normalized recipe.  A
    trajectory split across row minibatches keeps the same round denominator.
    """
    if minibatch_rows < 1 or chunk_tokens < 1:
        raise ValueError("PPO minibatch and vocabulary chunk sizes must be positive")
    if loss_normalization not in ("token", "trajectory"):
        raise ValueError("loss_normalization must be token or trajectory")
    if loss_normalization == "trajectory" and not constant_denominator:
        raise ValueError("Trajectory normalization requires the fixed round denominator")
    if not 0 < clip_ratio < 1 or dual_clip <= 1 or entropy_coef < 0 or kl_coef < 0:
        raise ValueError("Invalid PPO clipping or regularization coefficients")
    cursor = cursor or UpdateCursor()
    rows = [step for trajectory in trajectories for step in trajectory.steps]
    if not rows:
        raise ValueError("PPO requires real sampled response rows")
    minibatches = [rows[i:i+minibatch_rows] for i in range(0,len(rows),minibatch_rows)]
    round_mass = (len(trajectories) if loss_normalization == "trajectory" else
                  sum(len(row.response_ids) for row in rows))
    expected_denominator = round_mass / len(minibatches)
    if not cursor.denominator:
        cursor.denominator = expected_denominator
    elif not math.isclose(cursor.denominator, expected_denominator, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("PPO resume normalization differs from its saved round")
    saved_normalization = cursor.metrics.get("loss_normalization", loss_normalization)
    if saved_normalization != loss_normalization:
        raise ValueError("PPO resume loss normalization changed")
    cursor.metrics["loss_normalization"] = loss_normalization
    segments = {s.segment_id:s for t in trajectories for s in t.segments}
    policy.model.train()
    try:
        while cursor.minibatch < len(minibatches):
            batch = minibatches[cursor.minibatch]
            grouped = {}
            for row in batch:
                grouped.setdefault(row.segment_id,[]).append(row)
            items = list(grouped.items())
            denominator = cursor.denominator if constant_denominator else sum(len(row.response_ids) for row in batch)
            if cursor.segment == 0 and not cursor.optimizer_ready:
                optimizer.zero_grad()
            while cursor.segment < len(items):
                segment_id,selected = items[cursor.segment]
                segment = segments[segment_id]
                stop = segment.behavior_closed_length or len(segment.token_ids)
                with policy.phase("ppo.forward"):
                    hidden,_ = policy.packed_segment_hidden(segment.token_ids[:stop],hidden_layer=None,role="ppo.current_segment")
                positions, targets, old, ref, advantage = [], [], [], [], []
                for row in selected:
                    span = row.response_span
                    row_ref = reference_data.steps[row.step_id].token_logprobs
                    if span.length != len(row.response_ids) or span.length != len(row.old_token_logprobs) or span.length != len(row_ref):
                        raise ValueError("PPO response IDs and saved behavior/reference scores must align")
                    positions.extend(range(span.start-1, span.start-1+span.length))
                    targets.extend(row.response_ids)
                    old.extend(row.old_token_logprobs)
                    ref.extend(row_ref)
                    advantage.extend([float(advantages[row.step_id])] * span.length)
                output_device = policy.model.get_output_embeddings().weight.device
                position_ids = torch.tensor(positions, dtype=torch.long, device=hidden.device)
                target_ids = torch.tensor(targets, dtype=torch.long, device=output_device)
                old_scores = torch.tensor(old, dtype=torch.float32, device=output_device)
                ref_scores = torch.tensor(ref, dtype=torch.float32, device=output_device)
                advantage_scores = torch.tensor(advantage, dtype=torch.float32, device=output_device)
                hidden_gradient = torch.zeros_like(hidden)
                metrics = torch.zeros(3, dtype=torch.float64, device=output_device)
                with policy.phase("ppo.backward"):
                    for start in range(0, len(positions), chunk_tokens):
                        end = min(start+chunk_tokens, len(positions))
                        indices = position_ids[start:end]
                        states = hidden[0].index_select(0, indices).detach().requires_grad_(True)
                        logits = policy.head(states, role="ppo.head")
                        log_z = torch.logsumexp(logits, dim=-1)
                        lp = logits.gather(-1, target_ids[start:end, None]).squeeze(-1)-log_z
                        a = advantage_scores[start:end]
                        # Above this bound both branches of the dual-clipped
                        # objective are already constant.  Capping the exp
                        # input avoids inf*0 gradients without changing it.
                        ratio_limit = math.log(max(dual_clip, 1+clip_ratio)) + 1.
                        ratio = (lp-old_scores[start:end]).clamp(max=ratio_limit).exp()
                        dual = torch.maximum(-a*ratio, -a*ratio.clamp(1-clip_ratio, 1+clip_ratio))
                        dual = torch.where(a < 0, torch.minimum(dual, -dual_clip*a), dual)
                        probabilities = torch.softmax(logits, dim=-1)
                        entropy = log_z-torch.einsum("ij,ij->i", probabilities, logits)
                        d = ref_scores[start:end]-lp
                        # The exact capped k3 estimator equals 10 outside
                        # this interval; compute only finite exponentials.
                        bounded_d = d.clamp(min=-11., max=math.log(22.))
                        kl = (bounded_d.expm1()-bounded_d).clamp(min=0., max=10.)
                        loss = (dual-entropy_coef*entropy+kl_coef*kl).sum()/denominator
                        metrics.add_(torch.stack((loss.detach(), entropy.detach().sum()/denominator,
                                                  kl.detach().sum()/denominator)).double())
                        # Head parameters (including a tied embedding) get
                        # their usual gradient.  The detached leaf captures
                        # dL/dH, so this backward cannot traverse the trunk.
                        loss.backward()
                        hidden_gradient[0].index_add_(0, indices, states.grad)
                        del states, logits, log_z, lp, ratio, dual, probabilities, entropy, d, bounded_d, kl, loss
                    if hidden.requires_grad:
                        hidden.backward(hidden_gradient)
                for name, value in zip(("loss", "entropy", "reference_kl"), metrics.cpu().tolist()):
                    cursor.metrics[name] += value
                del hidden, hidden_gradient, position_ids, target_ids, old_scores, ref_scores, advantage_scores, metrics
                cursor.segment += 1
                if checkpoint_callback is not None:
                    checkpoint_callback(cursor)
            cursor.optimizer_ready = True
            if checkpoint_callback is not None:
                checkpoint_callback(cursor)
            with policy.phase("ppo.adamw"):
                cursor.metrics["gradient_norm"] = optimizer.step(max_norm=max_grad_norm)
            optimizer.zero_grad()
            cursor.metrics["steps"] += 1
            cursor.minibatch += 1
            cursor.segment,cursor.optimizer_ready = 0,False
            if checkpoint_callback is not None:
                checkpoint_callback(cursor)
        cursor.finished = True
        return cursor
    finally:
        policy.model.eval()
