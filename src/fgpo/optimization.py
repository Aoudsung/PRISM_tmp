"""Bounded CPU-master AdamW and streaming-head PPO.

Only a vocabulary chunk keeps a loss graph.  Its hidden-state derivative is
collected before the packed Transformer graph is traversed once.  Transfer
buffers are bounded independently of the size of a parameter tensor.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch.autograd.function import once_differentiable


class _LogProbabilityEntropy(torch.autograd.Function):
    """First-order response statistics without retaining a softmax loss graph.

    With p=softmax(z), d log p_y / dz = onehot(y)-p and
    d H / dz = -p*(log p+H). Only logits and row-sized statistics are saved;
    probabilities are reconstructed once when their combined adjoint is used:
    g_logp*(onehot(y)-p) - g_H*p*(log p+H), where g_* are the upstream
    derivatives of the normalized loss. The head supplies FP32 logits.
    """

    @staticmethod
    def forward(ctx, logits, targets):
        log_z = torch.logsumexp(logits, dim=-1)
        log_probability = logits.gather(1, targets[:, None]).squeeze(1)-log_z
        probabilities = (logits-log_z[:, None]).exp_()
        entropy = log_z-torch.einsum("ij,ij->i", probabilities, logits)
        ctx.save_for_backward(logits, log_z, entropy, targets)
        ctx.set_materialize_grads(False)
        return log_probability, entropy

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_log_probability, grad_entropy):
        if grad_log_probability is None and grad_entropy is None:
            return None, None
        logits, log_z, entropy, targets = ctx.saved_tensors
        log_probabilities = logits-log_z[:, None]
        if grad_entropy is not None:
            gradient = log_probabilities.exp()
            # g_H * dH/dz plus the dense -g_logp*p term. These two
            # temporaries are private to backward; saved logits stay intact.
            log_probabilities.add_(entropy[:, None]).mul_(-grad_entropy[:, None])
            if grad_log_probability is not None:
                log_probabilities.sub_(grad_log_probability[:, None])
            gradient.mul_(log_probabilities)
        else:
            gradient = log_probabilities.exp_().mul_(-grad_log_probability[:, None])
        del log_probabilities
        if grad_log_probability is not None:
            gradient.scatter_add_(1, targets[:, None], grad_log_probability[:, None])
        return gradient, None


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
        self.host_grad = torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=self.pin_memory)
        self.host_weight = torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=self.pin_memory)
        with torch.cuda.stream(self.stream):
            self.device_grad = torch.empty(capacity, dtype=dtype, device=device)
            self.device_weight = torch.empty(capacity, dtype=dtype, device=device)

    def download(self, entries, fresh):
        # Reuse is ordered on this slot's stream: prior H2D/writeback -> next
        # D2H -> downloaded event. The CPU waits for that final event before
        # reading gradients or overwriting host_weight, so no earlier host
        # wait is needed while the other slot's CPU update can proceed.
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

    def synchronize(self):
        # Completion also makes state_dict/save_model and the next forward safe.
        self.stream.synchronize()


class OffloadedAdamW:
    def __init__(self, model, *, lr=1e-6, weight_decay=.01, betas=(.9,.999), eps=1e-8,
                 bucket_mib=64, pin_memory=True):
        self.params = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
        self.lr, self.weight_decay, self.betas, self.eps = float(lr),float(weight_decay),tuple(betas),float(eps)
        if (not all(math.isfinite(value) for value in (self.lr,self.weight_decay,self.eps,*self.betas))
            or self.lr < 0 or self.weight_decay < 0 or self.eps <= 0
            or len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas)):
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
                        gradient = entry.parameter.grad.detach().view(-1)[part].to(dtype=torch.float32,copy=True).mul_(scale)
                        master = self._update_slice(entry, gradient, values if entry.name in fresh else None)
                        entry.parameter.view(-1)[part].copy_(master)
                        del gradient
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
                    gradients = slot.host_grad[:slot.length].float().mul_(scale)
                    for entry in entries:
                        section = slice(entry.offset, entry.offset + entry.size)
                        initial = slot.host_weight[section] if entry.name in fresh else None
                        master = self._update_slice(entry, gradients[section], initial)
                        slot.host_weight[section].copy_(master)
                    del gradients
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
        # checkpoint.save serializes synchronously. torch.save stages each
        # storage on CPU in turn; retaining device aliases here avoids an
        # additional full-gradient CPU copy at an interrupted minibatch.
        # checkpoint.load explicitly restores these tensors on CPU.
        return {name:p.grad.detach() for name,p in self.params if p.grad is not None}

    def restore_gradients(self, values):
        self.zero_grad()
        params = dict(self.params)
        for name,value in values.items():
            params[name].grad = value.to(params[name])

    def reference_restore_sources(self, model, initial_weights):
        """Borrow actor sources; uninitialized parameters are still initial.

        The engine changes actor parameters only through this optimizer.
        Every updated parameter therefore has a persistent FP32 master; a
        parameter without optimizer state has never been updated. The caller
        must supply the immutable initial reference, not an EMA reference.
        Validate all sources before installing any reference parameters.
        """
        parameters, owned = dict(model.named_parameters()), dict(self.params)
        if any(parameters.get(name) is not parameter for name,parameter in self.params):
            raise ValueError("Reference restoration requires this actor's optimizer")
        sources = {}
        for name,initial in initial_weights.items():
            parameter = parameters[name]
            if parameter.requires_grad and owned.get(name) is not parameter:
                raise ValueError("Every trainable reference parameter must belong to the actor optimizer")
            state = self.state.get(name)
            source = initial if state is None else state["master"]
            expected_dtype = parameter.dtype if state is None else torch.float32
            if (source.device.type != "cpu" or source.shape != parameter.shape or
                source.dtype != expected_dtype or not source.is_contiguous() or
                not parameter.is_contiguous()):
                raise ValueError("Reference restoration requires matching contiguous CPU source tensors")
            sources[name] = source
        return sources

    @torch.no_grad()
    def restore_reference_parameter(self, parameter, source):
        # Match step() exactly: round FP32 master -> actor dtype on CPU,
        # then transfer that dtype to the actor. A cross-device FP32 copy
        # would move rounding to CUDA. Temporary storage is one CPU bucket.
        size = max(1,self.bucket_bytes // parameter.element_size())
        target, values = parameter.view(-1), source.view(-1)
        for start in range(0,parameter.numel(),size):
            end = min(start+size,parameter.numel())
            rounded = values[start:end].to(dtype=parameter.dtype)
            target[start:end].copy_(rounded)
            del rounded


class WeightReference:
    def __init__(self, model=None, *, trainable_only=False, state=None, fixed=False):
        self.weights = (state if state is not None else {name:p.detach().to(device="cpu", copy=True)
                        for name,p in model.named_parameters() if not trainable_only or p.requires_grad})
        self.fixed = bool(fixed)

    @contextmanager
    def activate(self, model, *, optimizer=None):
        if optimizer is not None and not self.fixed:
            raise ValueError("Master-weight restoration requires an immutable initial reference")
        current = (optimizer.reference_restore_sources(model,self.weights) if optimizer is not None else {})
        installed,previous = [],model.training
        try:
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in self.weights:
                        if optimizer is None:
                            current[name] = p.detach().to(device="cpu", copy=True)
                        # Register before copy_: finally restores even if
                        # installing this parameter fails part-way through.
                        installed.append((name,p))
                        p.copy_(self.weights[name])
            model.eval()
            yield
        finally:
            with torch.no_grad():
                for name,p in installed:
                    if optimizer is None:
                        p.copy_(current[name])
                    else:
                        optimizer.restore_reference_parameter(p,current[name])
            model.train(previous)

    @torch.no_grad()
    def update_ema(self, model, decay=.99):
        if self.fixed:
            raise ValueError("The initial reference must remain immutable")
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
               minibatch_rows: int | None = None, chunk_tokens=64, cursor=None,
               checkpoint_callback=None, constant_denominator=True, loss_normalization="token",
               entropy_normalization="trajectory", clip_ratio=.2, dual_clip=3.,
               entropy_coef=.001, kl_coef=.01, max_grad_norm=1., published_plan=None,
               trajectory_groups=None, round_index=None):
    """PPO with one live vocabulary chunk and one backbone backward per segment.

    ``trajectory`` sums all response-token scores along each trajectory and
    divides by a fixed trajectory count; it does not divide by trajectory
    length.  ``token`` preserves the previous token-normalized recipe.  When
    ``trajectory_groups`` is provided, each mixed-outcome group is updated as
    one atomic K-trajectory minibatch with fixed trajectory denominator K.
    Only the selected responses' causal prefix is forwarded. The gathered
    response states retain the one trunk graph; their detached chunks carry
    vocabulary gradients until a single gather/backbone backward.

    A published plan preserves the released virtual-rank/microbatch weighting
    and optimizer boundaries. Its per-token coefficient already includes the
    fixed gradient accumulation and rank average; repeated copies of one row
    within one update are summed before a forward, never across updates.
    """
    if chunk_tokens < 1:
        raise ValueError("PPO vocabulary chunk size must be positive")
    if published_plan is not None and trajectory_groups is not None:
        raise ValueError("published_plan and trajectory_groups are mutually exclusive")
    if trajectory_groups is None and published_plan is None and (minibatch_rows is None or minibatch_rows < 1):
        raise ValueError("Row-sliced PPO requires a positive minibatch_rows")
    if loss_normalization not in ("token", "trajectory"):
        raise ValueError("loss_normalization must be token or trajectory")
    if entropy_normalization not in ("token", "trajectory"):
        raise ValueError("entropy_normalization must be token or trajectory")
    if loss_normalization == "trajectory" and not constant_denominator:
        raise ValueError("Trajectory normalization requires the fixed round denominator")
    if not 0 < clip_ratio < 1 or dual_clip <= 1 or entropy_coef < 0 or kl_coef < 0:
        raise ValueError("Invalid PPO clipping or regularization coefficients")
    cursor = cursor or UpdateCursor()
    if trajectory_groups is not None:
        if loss_normalization != "trajectory" or not constant_denominator:
            raise ValueError("Atomic mixed-group updates require trajectory loss normalization")
        unpacked = [tuple(g.trajectories if hasattr(g, "trajectories") else g) for g in trajectory_groups]
        if not unpacked:
            raise ValueError("Atomic mixed-group PPO update requires at least one mixed group")
        group_size = len(unpacked[0])
        if group_size < 2 or any(len(g) != group_size for g in unpacked):
            raise ValueError("Every atomic mixed group must have trajectory count K >= 2")
        trajectories = [t for g in unpacked for t in g]
        rows = [step for t in trajectories for step in t.steps]
        if not rows:
            raise ValueError("PPO requires real sampled response rows")
        coefficients = None
        minibatches = [[step for t in g for step in t.steps] for g in unpacked]
        if any(not batch for batch in minibatches):
            raise ValueError("Every atomic mixed group must contain actual response rows")
        expected_denominator, weighting = float(group_size), "atomic_mixed_group_trajectory"
    elif published_plan is not None:
        rows = [step for trajectory in trajectories for step in trajectory.steps]
        if not rows or constant_denominator or loss_normalization != "token":
            raise ValueError("Published source weights require their token-mean update definition")
        indexed = {row.step_id:row for row in rows}
        coefficients = [dict(batch) for batch in published_plan["minibatches"]]
        minibatches = [[indexed[step_id] for step_id in batch] for batch in coefficients]
        if not minibatches or any(not batch for batch in minibatches):
            raise ValueError("A published optimizer step requires actual response rows")
        expected_denominator, weighting = 1., "published_virtual_microbatch_mean"
    else:
        rows = [step for trajectory in trajectories for step in trajectory.steps]
        if not rows:
            raise ValueError("PPO requires real sampled response rows")
        coefficients = None
        minibatches = [rows[i:i+minibatch_rows] for i in range(0,len(rows),minibatch_rows)]
        round_mass = (len(trajectories) if loss_normalization == "trajectory" else
                      sum(len(row.response_ids) for row in rows))
        expected_denominator = round_mass / len(minibatches)
        weighting = "round_trajectory" if constant_denominator else "minibatch_token_mean"
    if not cursor.denominator:
        cursor.denominator = expected_denominator
    elif not math.isclose(cursor.denominator, expected_denominator, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("PPO resume normalization differs from its saved round")
    for key, value in (("loss_normalization", loss_normalization),
                       ("entropy_normalization", entropy_normalization),
                       ("loss_weighting", weighting),
                       ("minibatch_count", len(minibatches))):
        if cursor.metrics.get(key, value) != value:
            raise ValueError(f"PPO resume {key} changed")
        cursor.metrics[key] = value
    cursor.metrics.setdefault("batch_gradient_norms", [])
    cursor.metrics.setdefault("batch_clip_scales", [])
    cursor.metrics.setdefault("evaluated_response_tokens", 0)
    cursor.metrics.setdefault("ppo_clipped_tokens", 0)
    cursor.metrics.setdefault("first_batch_tokens", 0)
    cursor.metrics.setdefault("first_batch_abs_logprob_diff_sum", 0.)
    cursor.metrics.setdefault("first_batch_max_abs_logprob_diff", 0.)
    if trajectory_groups is not None:
        cursor.metrics.update(mixed_groups=len(minibatches), group_trajectory_denominator=expected_denominator,
                              homogeneous_round_bypass=False)
    segments = {s.segment_id:s for t in trajectories for s in t.segments}
    segment_groups = {s.segment_id:t.group_id for t in trajectories for s in t.segments}
    ledger = getattr(policy, "ledger", None)
    policy.model.train()
    try:
        while cursor.minibatch < len(minibatches):
            batch = minibatches[cursor.minibatch]
            grouped = {}
            for row in batch:
                grouped.setdefault(row.segment_id,[]).append(row)
            items = list(grouped.items())
            batch_group_ids = {segment_groups[row.segment_id] for row in batch}
            batch_group_id = next(iter(batch_group_ids)) if len(batch_group_ids) == 1 else None
            denominator = (1. if coefficients is not None else cursor.denominator if constant_denominator
                           else sum(len(row.response_ids) for row in batch))
            batch_response_tokens = sum(len(row.response_ids) for row in batch)
            entropy_denominator = (float(batch_response_tokens)
                                   if coefficients is None and entropy_normalization == "token"
                                   else denominator)
            with (ledger.context(group_id=batch_group_id, round_index=round_index)
                  if ledger is not None else nullcontext()):
                if cursor.segment == 0 and not cursor.optimizer_ready:
                    optimizer.zero_grad()
                while cursor.segment < len(items):
                    segment_id,selected = items[cursor.segment]
                    segment = segments[segment_id]
                    segment_group_id = segment_groups[segment_id]
                    closed = segment.behavior_closed_length or len(segment.token_ids)
                    positions, targets, old, ref, advantage, weights = [], [], [], [], [], []
                    for row in selected:
                        span = row.response_span
                        row_ref = reference_data.steps[row.step_id].token_logprobs
                        if (span.segment_id != segment_id or span.start < 1 or span.length < 1
                            or span.stop > closed or span.length != len(row.response_ids)
                            or span.length != len(row.old_token_logprobs) or span.length != len(row_ref)):
                            raise ValueError("PPO response IDs and saved behavior/reference scores must align")
                        positions.extend(range(span.start-1, span.start-1+span.length))
                        targets.extend(row.response_ids)
                        old.extend(row.old_token_logprobs)
                        ref.extend(row_ref)
                        advantage.extend([float(advantages[row.step_id])] * span.length)
                        if coefficients is not None:
                            weights.extend([coefficients[cursor.minibatch][row.step_id]] * span.length)
                    # Future tokens in this packed segment cannot affect a
                    # selected response under the causal mask. Retain every
                    # preceding token and its original position, stopping only
                    # after the last target owned by this minibatch.
                    stop = max(row.response_span.stop for row in selected)
                    track_first_batch = cursor.minibatch == 0
                    with (ledger.context(group_id=segment_group_id, round_index=round_index)
                          if ledger is not None else nullcontext()), \
                         (ledger.aggregate_model_calls() if ledger is not None else nullcontext()):
                        with policy.phase("ppo.forward"):
                            hidden,_ = policy.packed_segment_hidden(segment.token_ids[:stop],hidden_layer=None,role="ppo.current_segment")
                            position_ids = torch.tensor(positions, dtype=torch.long, device=hidden.device)
                            response_hidden = hidden[0].index_select(0,position_ids)
                            # IndexSelectBackward retains the scatter indices and
                            # input shape, not this full prompt/observation output.
                            del hidden,position_ids
                        output_device = policy.model.get_output_embeddings().weight.device
                        target_ids = torch.tensor(targets, dtype=torch.long, device=output_device)
                        old_scores = torch.tensor(old, dtype=torch.float32, device=output_device)
                        ref_scores = torch.tensor(ref, dtype=torch.float32, device=output_device)
                        advantage_scores = torch.tensor(advantage, dtype=torch.float32, device=output_device)
                        loss_weights = torch.tensor(weights,dtype=torch.float32,device=output_device) if coefficients is not None else None
                        response_gradient = torch.empty_like(response_hidden) if response_hidden.requires_grad else None
                        metrics = torch.zeros(3, dtype=torch.float64, device=output_device)
                        clipped_tokens = torch.zeros((), dtype=torch.int64, device=output_device)
                        if track_first_batch:
                            first_batch_diff_sum = torch.zeros((), dtype=torch.float64, device=output_device)
                            first_batch_diff_max = torch.zeros((), dtype=torch.float64, device=output_device)
                        with policy.phase("ppo.backward"):
                            for start in range(0, len(positions), chunk_tokens):
                                end = min(start+chunk_tokens, len(positions))
                                states = response_hidden[start:end].detach().requires_grad_(response_gradient is not None)
                                logits = policy.head(states, role="ppo.head")
                                lp,entropy = _LogProbabilityEntropy.apply(logits,target_ids[start:end])
                                a = advantage_scores[start:end]
                                # Above this bound both branches of the dual-clipped
                                # objective are already constant.  Capping the exp
                                # input avoids inf*0 gradients without changing it.
                                ratio_limit = math.log(max(dual_clip, 1+clip_ratio)) + 1.
                                ratio = (lp-old_scores[start:end]).clamp(max=ratio_limit).exp()
                                with torch.no_grad():
                                    if track_first_batch:
                                        lp_diff = (lp.detach() - old_scores[start:end]).abs()
                                        first_batch_diff_sum.add_(lp_diff.double().sum())
                                        torch.maximum(first_batch_diff_max, lp_diff.max().double(), out=first_batch_diff_max)
                                        del lp_diff
                                    clipped_mask = ((a > 0) & (ratio > 1 + clip_ratio)) | ((a < 0) & ((ratio < 1 - clip_ratio) | (ratio > dual_clip)))
                                    clipped_tokens.add_(clipped_mask.sum(dtype=torch.int64))
                                    del clipped_mask
                                dual = torch.maximum(-a*ratio, -a*ratio.clamp(1-clip_ratio, 1+clip_ratio))
                                dual = torch.where(a < 0, torch.minimum(dual, -dual_clip*a), dual)
                                d = ref_scores[start:end]-lp
                                # The exact capped k3 estimator equals 10 outside
                                # this interval; compute only finite exponentials.
                                bounded_d = d.clamp(min=-11., max=math.log(22.))
                                kl = (bounded_d.expm1()-bounded_d).clamp(min=0., max=10.)
                                ppo_kl = dual+kl_coef*kl
                                # This small scalar graph supplies the exact clipped
                                # PPO/k3 coefficient g_logp and entropy's coefficient
                                # (trajectory denominator, token-mean entropy, or published row weight)
                                # to the combined vocabulary adjoint above.
                                if loss_weights is None:
                                    loss = ppo_kl.sum()/denominator
                                    if entropy_coef:
                                        loss = loss-(entropy_coef/entropy_denominator)*entropy.sum()
                                    entropy_metric = entropy.detach().sum()/entropy_denominator
                                    kl_metric = kl.detach().sum()/denominator
                                else:
                                    weight = loss_weights[start:end]
                                    loss = ((ppo_kl-entropy_coef*entropy if entropy_coef else ppo_kl)*weight).sum()
                                    entropy_metric,kl_metric = (entropy.detach()*weight).sum(),(kl.detach()*weight).sum()
                                metrics.add_(torch.stack((loss.detach(),entropy_metric,kl_metric)).double())
                                # Head parameters (including a tied embedding) get
                                # their usual gradient.  The detached leaf captures
                                # dL/dH, so this backward cannot traverse the trunk.
                                loss.backward()
                                if response_gradient is not None:
                                    response_gradient[start:end].copy_(states.grad)
                                del states, logits, lp, ratio, dual, entropy, d, bounded_d, kl, ppo_kl, loss
                            if response_gradient is not None:
                                response_hidden.backward(response_gradient)
                        del response_hidden,response_gradient
                    if track_first_batch:
                        summary = torch.cat((
                            metrics,
                            torch.stack((clipped_tokens.double(), first_batch_diff_sum, first_batch_diff_max)),
                        )).cpu().tolist()
                        loss_val, entropy_val, kl_val, clipped_val, diff_sum, diff_max = summary
                        cursor.metrics["first_batch_tokens"] += len(positions)
                        cursor.metrics["first_batch_abs_logprob_diff_sum"] += diff_sum
                        previous_max = cursor.metrics["first_batch_max_abs_logprob_diff"]
                        cursor.metrics["first_batch_max_abs_logprob_diff"] = (
                            diff_max if math.isnan(diff_max) else max(previous_max, diff_max)
                        )
                        del first_batch_diff_sum, first_batch_diff_max
                    else:
                        summary = torch.cat((metrics, clipped_tokens.double().unsqueeze(0))).cpu().tolist()
                        loss_val, entropy_val, kl_val, clipped_val = summary
                    cursor.metrics["loss"] += loss_val
                    cursor.metrics["entropy"] += entropy_val
                    cursor.metrics["reference_kl"] += kl_val
                    cursor.metrics["evaluated_response_tokens"] += len(positions)
                    cursor.metrics["ppo_clipped_tokens"] += int(clipped_val)
                    del target_ids, old_scores, ref_scores, advantage_scores, loss_weights, metrics, clipped_tokens
                    cursor.segment += 1
                    if checkpoint_callback is not None:
                        checkpoint_callback(cursor)
                cursor.optimizer_ready = True
                if checkpoint_callback is not None:
                    checkpoint_callback(cursor)
                with policy.phase("ppo.adamw"):
                    grad_norm = float(optimizer.step(max_norm=max_grad_norm))
                    clip_scale = min(1., max_grad_norm / (grad_norm + 1e-6)) if max_grad_norm is not None else 1.
                    cursor.metrics["gradient_norm"] = grad_norm
                    cursor.metrics["batch_gradient_norms"].append(grad_norm)
                    cursor.metrics["batch_clip_scales"].append(clip_scale)
                optimizer.zero_grad()
                cursor.metrics["steps"] += 1
                cursor.minibatch += 1
                cursor.segment,cursor.optimizer_ready = 0,False
                if checkpoint_callback is not None:
                    checkpoint_callback(cursor)
        cursor.metrics["first_batch_mean_abs_logprob_diff"] = (
            cursor.metrics["first_batch_abs_logprob_diff_sum"] / cursor.metrics["first_batch_tokens"]
            if cursor.metrics["first_batch_tokens"] else None
        )
        cursor.metrics["ppo_clip_fraction"] = (
            cursor.metrics["ppo_clipped_tokens"] / cursor.metrics["evaluated_response_tokens"]
            if cursor.metrics["evaluated_response_tokens"] else 0.0
        )
        cursor.finished = True
        return cursor
    finally:
        policy.model.eval()
