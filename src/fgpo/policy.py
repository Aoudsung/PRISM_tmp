"""Qwen causal execution with bounded workspace and reusable inference KV storage."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .cache import ArenaCache, ArenaLayer


def clock() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC)


@dataclass
class KVState:
    cache: Any
    valid_mask: torch.Tensor
    next_logits: torch.Tensor
    lengths: tuple[int, ...]
    # Half-open physical column ranges.  Host-known ranges let compaction build
    # gather indices without torch.nonzero's device-to-host shape readback.
    valid_spans: tuple[tuple[tuple[int, int], ...], ...]
    right_padded: bool = True


class _DecodeWorkspace:
    """One response's bounded, exclusively owned device input buffers.

    Each draw completes its existing packed CPU readback before these mutable
    buffers are reused. All writes/consumers on each device use its current
    stream; cached selection tensors are immutable until the active set changes.
    """
    def __init__(self, policy, state, max_tokens):
        self.batch = len(state.lengths)
        self.device = policy.device
        self.start = state.valid_mask.shape[1]
        self.steps = 0
        self.max_appends = max(0, int(max_tokens) - 1)
        self.ids = torch.empty((self.batch, 1), dtype=torch.long, device=self.device)
        self.positions = torch.empty_like(self.ids)
        self.lengths = torch.tensor(state.lengths, dtype=torch.long, device=self.device)
        self.mask = torch.empty((self.batch, self.start + self.max_appends),
                                dtype=torch.bool, device=self.device)
        self.mask[:, :self.start].copy_(state.valid_mask)
        self.cache_positions = torch.arange(self.start, self.start + self.max_appends,
                                            dtype=torch.long, device=self.device)
        self.sample_input = torch.empty(self.batch, dtype=torch.long, device=self.device)
        self.sample_gather = torch.empty(self.batch, dtype=torch.long, device=state.next_logits.device)
        self.pad_token_id = policy.tokenizer.pad_token_id
        self.selections, self.zero_columns = {}, {}
        self.sampled_tokens, self.sampled_rows = None, ()

    def indices(self, rows, device, purpose="rows"):
        rows, device = tuple(rows), torch.device(device)
        key = purpose, device
        previous = self.selections.get(key)
        if previous is None or previous[0] != rows:
            previous = rows, torch.tensor(rows, dtype=torch.long, device=device)
            self.selections[key] = previous
        return previous[1]

    def zeros(self, count, device):
        device = torch.device(device)
        if device not in self.zero_columns:
            self.zero_columns[device] = torch.zeros(self.batch, dtype=torch.long, device=device)
        return self.zero_columns[device][:count]

    def retain(self, tokens, rows):
        if self.sampled_tokens is not None:
            raise RuntimeError("Consume the retained draw before sampling again")
        self.sampled_tokens, self.sampled_rows = tokens, tuple(rows)

    def clear_sample(self):
        self.sampled_tokens, self.sampled_rows = None, ()

    def prepare(self, state, continuing):
        continuing = tuple(continuing)
        if self.sampled_tokens is None or not continuing:
            raise ValueError("One-token append requires a retained draw and continuing rows")
        if self.steps >= self.max_appends or state.valid_mask.shape[1] != self.start + self.steps:
            raise ValueError("One-token workspace must follow its response's physical cache positions")
        keep = set(continuing)
        selected = tuple(i for i, row in enumerate(self.sampled_rows) if row in keep)
        if tuple(self.sampled_rows[i] for i in selected) != continuing:
            raise ValueError("Continuing rows must preserve the retained draw's slot order")
        if continuing == self.sampled_rows:
            tokens = self.sampled_tokens
        else:
            positions = self.indices(selected, self.sampled_tokens.device, "sample_positions")
            tokens = self.sample_gather[:len(continuing)]
            torch.index_select(self.sampled_tokens, 0, positions, out=tokens)
        rows = self.indices(continuing, self.device)
        self.sample_input[:len(continuing)].copy_(tokens)
        self.ids.fill_(self.pad_token_id)
        self.ids.index_copy_(0, rows, self.sample_input[:len(continuing), None])
        physical = self.start + self.steps
        valid = self.mask[:, physical]
        valid.zero_()
        valid.index_fill_(0, rows, True)
        # Match the general width-one path for both real and padded queries:
        # real rows use their old length, inactive rows use max(old length-1,0).
        self.positions[:, 0].copy_(self.lengths)
        self.positions[:, 0].sub_(1).add_(valid).clamp_min_(0)
        self.lengths.add_(valid)
        cache_position = self.cache_positions[self.steps:self.steps + 1]
        self.steps += 1
        additions = tuple(int(row in keep) for row in range(self.batch))
        return (self.ids, self.mask[:, :physical + 1], self.positions, cache_position, additions)

    def close(self):
        self.clear_sample()
        self.selections.clear()
        self.zero_columns.clear()
        self.ids = self.positions = self.lengths = self.mask = None
        self.cache_positions = self.sample_input = self.sample_gather = None


class _LayerCaptured(Exception):
    """Internal control flow for a no-grad intermediate-layer representation."""


def layer_device_map(config, gpu_ids, free_gib=None, *, full_training=True, working_memory_gib=6.):
    if not gpu_ids:
        raise ValueError("Qwen requires an explicitly allocated healthy CUDA device")
    capacities = [float(free_gib[g]) if free_gib and g in free_gib else
                  torch.cuda.mem_get_info(g)[0] / 2**30 for g in gpu_ids]
    h, intermediate = int(config.hidden_size), int(config.intermediate_size)
    kv = int(config.num_key_value_heads) * (h // int(config.num_attention_heads))
    width = 4 if full_training else 2
    layer_size = (2*h*h + 2*h*kv + 3*h*intermediate + 4*h) * width / 2**30
    embedding = int(config.vocab_size) * h * width / 2**30
    capacities = [v - float(working_memory_gib) for v in capacities]
    capacities[0] -= embedding
    if not config.tie_word_embeddings:
        capacities[-1] -= embedding
    if min(capacities) < 0 or sum(capacities) < config.num_hidden_layers * layer_size:
        raise ValueError("Allocated free VRAM does not cover Qwen weights/gradients and working space")
    total, cumulative, boundaries = sum(capacities), 0., []
    for capacity in capacities:
        cumulative += capacity / total
        boundaries.append(cumulative)
    mapping = {}
    for layer in range(config.num_hidden_layers):
        fraction = (layer + .5) / config.num_hidden_layers
        preferred = next(i for i, boundary in enumerate(boundaries) if fraction <= boundary + 1e-9)
        fitting = [i for i, value in enumerate(capacities) if value >= layer_size]
        if not fitting:
            raise ValueError("Allocation cannot hold the next whole Qwen decoder layer")
        selected = preferred if preferred in fitting else max(fitting, key=lambda i: capacities[i])
        mapping[f"model.layers.{layer}"] = f"cuda:{gpu_ids[selected]}"
        capacities[selected] -= layer_size
    mapping.update({"model.embed_tokens": f"cuda:{gpu_ids[0]}", "model.norm": f"cuda:{gpu_ids[-1]}",
                    "lm_head": f"cuda:{gpu_ids[0] if config.tie_word_embeddings else gpu_ids[-1]}"})
    return mapping


class CausalPolicy(nn.Module):
    def __init__(self, model_path: str, *, gpu_ids: Sequence[int], ledger,
                 trainable: bool = True, adaptation: str | None = None,
                 checkpoint: str | None = None, free_memory_gib: Mapping[int, float] | None = None,
                 activation_offload: bool = False, benchmark: str = "alfworld",
                 max_context_tokens: int | None = None, max_new_tokens: int | None = None,
                 cache_initial_capacity: int = 1024, representation_token_budget: int = 8192,
                 representation_batch_size: int = 8, working_memory_gib: float = 6.,
                 checkpoint_shard_mib: float = 1024.):
        super().__init__()
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        # These are the actual compute devices. The Engine/ledger separately
        # retain the full allocated device set and all saved RNG contexts.
        self.model_path, self.gpu_ids, self.ledger = str(model_path), tuple(gpu_ids), ledger
        self.adaptation, self.benchmark = adaptation, benchmark
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if cfg.model_type != "qwen2":
            raise ValueError("The formal study uses the existing official Qwen2.5-7B checkpoint")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device_map = layer_device_map(cfg, self.gpu_ids, free_memory_gib,
            full_training=trainable and adaptation is None, working_memory_gib=working_memory_gib)
        source = checkpoint if checkpoint and adaptation is None else model_path
        self.model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=self.device_map, low_cpu_mem_usage=True)
        if adaptation == "self_lora":
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            self.model = (PeftModel.from_pretrained(self.model, checkpoint, is_trainable=trainable)
                if checkpoint else get_peft_model(self.model, LoraConfig(task_type=TaskType.CAUSAL_LM,
                    r=96, lora_alpha=128, lora_dropout=.05, target_modules="all-linear", bias="none")))
        elif adaptation is not None:
            raise ValueError(f"Unknown author adaptation {adaptation!r}")
        else:
            self.model.requires_grad_(trainable)
        if adaptation != "self_lora":
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.
                if hasattr(module, "attention_dropout"):
                    module.attention_dropout = 0.
        if trainable:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        self.model.eval()
        self.activation_offload = bool(activation_offload)
        supported_context = min(32768, int(cfg.max_position_embeddings))
        self.max_context_tokens = supported_context if max_context_tokens is None else int(max_context_tokens)
        self.max_new_tokens = (256 if benchmark == "alfworld" else 1024) if max_new_tokens is None else int(max_new_tokens)
        if not 0 < self.max_new_tokens < self.max_context_tokens <= supported_context:
            raise ValueError("Generation/context limits must fit the configured Qwen context")
        self.cache_initial_capacity = int(cache_initial_capacity)
        self.representation_token_budget = int(representation_token_budget)
        self.representation_batch_size = int(representation_batch_size)
        if min(self.cache_initial_capacity, self.representation_token_budget, self.representation_batch_size) < 1:
            raise ValueError("Cache and representation batch budgets must be positive")
        if not math.isfinite(float(checkpoint_shard_mib)) or float(checkpoint_shard_mib) <= 0:
            raise ValueError("checkpoint_shard_mib must be finite and positive")
        self.checkpoint_shard_bytes = int(float(checkpoint_shard_mib) * 1024 * 1024)
        if self.checkpoint_shard_bytes < 1:
            raise ValueError("Checkpoint shard size must be at least one byte")
        self._phase_depth = 0
        self.eos_token_ids = (151645, 151643)
        self.sampling_temperature = 1.
        self._reference_active = False
        self.metadata = {"base_model": model_path, "adaptation": adaptation, "dtype": "bfloat16",
                         "compute_logical_gpu_ids":list(self.gpu_ids),
                         "attention": "sdpa", "device_map": dict(self.device_map),
                         "kv_cache": "geometric_arena", "cache_initial_capacity": self.cache_initial_capacity,
                         "rollout_batching": "turn_barrier_owned_cohort_merge",
                         "rollout_decode_inputs": "retained_device_tokens_bounded_workspace_v2",
                         "activation_offload": self.activation_offload,
                         "representation_token_budget": self.representation_token_budget,
                         "representation_batch_size": self.representation_batch_size,
                         "working_memory_reserve_gib": float(working_memory_gib),
                         "checkpoint_shard_bytes": self.checkpoint_shard_bytes,
                         "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)}

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @property
    def base(self):
        return self.model.get_base_model() if self.adaptation else self.model

    @property
    def sampling_temperature(self):
        return self._sampling_temperature

    @sampling_temperature.setter
    def sampling_temperature(self, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Sampling temperature must be finite and positive")
        self._sampling_temperature = value

    def synchronize(self):
        for gpu in self.gpu_ids:
            torch.cuda.synchronize(gpu)

    def record(self, kind, role, **fields):
        self.ledger.record(kind, role, gpu_ids=list(self.gpu_ids), **fields)

    @contextmanager
    def activation_saving(self):
        if self.activation_offload and torch.is_grad_enabled():
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                yield
        else:
            yield

    @contextmanager
    def phase(self, role):
        self.synchronize()
        outermost = self._phase_depth == 0
        if outermost:
            for gpu in self.gpu_ids:
                torch.cuda.reset_peak_memory_stats(gpu)
        self._phase_depth += 1
        start, status = clock(), "OK"
        try:
            yield
        except BaseException:
            status = "ERROR"
            raise
        finally:
            self.synchronize()
            self._phase_depth -= 1
            memory = {str(gpu): {"allocated_bytes": torch.cuda.memory_allocated(gpu),
                                "reserved_bytes": torch.cuda.memory_reserved(gpu),
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(gpu),
                                "peak_reserved_bytes": torch.cuda.max_memory_reserved(gpu)}
                      for gpu in self.gpu_ids}
            self.record("compute", role, seconds=clock()-start,
                        details={"status": status, "time_basis": "synchronized_wall",
                                 "memory_scope": "phase" if outermost else "enclosing_phase",
                                 "cuda_memory": memory})

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True))

    def new_generator(self, seed: int):
        generator = torch.Generator(device=self.model.get_output_embeddings().weight.device)
        generator.manual_seed(int(seed))
        return generator

    def head(self, hidden: torch.Tensor, *, role="vocabulary_projection"):
        started, status = clock(), "OK"
        try:
            module = self.model.get_output_embeddings()
            return module(hidden.to(module.weight.device)).float()
        except BaseException:
            status = "ERROR"
            raise
        finally:
            self.record("model", role, seconds=clock()-started,
                        details={"status": status, "time_basis": "host_dispatch",
                                 "projection_continuation": True,
                                 "positions": hidden.numel() // hidden.shape[-1]})

    def body(self, input_ids, *, attention_mask=None, position_ids=None, cache=None,
             cache_position=None, use_cache=False, hidden_layer=None, role="forward", teacher_forced=False,
             actual_tokens=None, logical_length=None, stop_at_hidden_layer=False, hidden_positions=None):
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("A causal forward requires a nonempty batch of token sequences")
        if attention_mask is None:
            past = cache.get_seq_length() if cache is not None else 0
            mask = torch.ones((ids.shape[0], past + ids.shape[1]), dtype=torch.bool, device=self.device)
            if logical_length is None:
                logical_length = past + ids.shape[1]
            if actual_tokens is None:
                actual_tokens = ids.numel()
        else:
            mask = attention_mask.to(self.device)
        if position_ids is None:
            position_ids = (mask.long().cumsum(-1)-1).clamp_min(0)[:, -ids.shape[1]:]
        else:
            position_ids = position_ids.to(self.device)
        # The rollout passes lengths already known on the host.  No per-token
        # scalar readback is needed for capacity checks or work accounting.
        if logical_length is None:
            logical_length = int(position_ids.max()) + 1
        if logical_length > self.max_context_tokens:
            raise ValueError("Exact Qwen context capacity exceeded")
        if actual_tokens is None:
            actual_tokens = int(mask[:, -ids.shape[1]:].sum())
        if stop_at_hidden_layer and (hidden_layer is None or use_cache or torch.is_grad_enabled()):
            raise ValueError("Layer early exit is restricted to no-grad, uncached representation forwards")
        if hidden_positions is not None and hidden_layer is None:
            raise ValueError("Selected representation positions require an intermediate layer")
        captures = []
        hook = None
        if hidden_layer is not None:
            layers = self.base.model.layers
            index = len(layers) + hidden_layer if hidden_layer < 0 else hidden_layer
            if not 0 <= index < len(layers):
                raise ValueError("Representation layer is outside the model")
            def capture(_module, _args, output):
                representation = output[0] if isinstance(output, tuple) else output
                if hidden_positions is not None:
                    positions = torch.as_tensor(hidden_positions, dtype=torch.long, device=representation.device)
                    representation = representation.index_select(1, positions)
                captures.append(representation)
                if stop_at_hidden_layer:
                    raise _LayerCaptured
            hook = layers[index].register_forward_hook(capture)
        kwargs = dict(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                      past_key_values=cache, use_cache=use_cache)
        if cache_position is not None:
            kwargs["cache_position"] = cache_position.to(self.device)
        start, status = clock(), "OK"
        try:
            with self.activation_saving():
                output = self.base.model(**kwargs)
            selected = captures[0] if captures else None
            return output.last_hidden_state, selected, output.past_key_values if use_cache else None
        except _LayerCaptured:
            return captures[0], captures[0], None
        except BaseException:
            status = "ERROR"
            raise
        finally:
            if hook is not None:
                hook.remove()
            sampling_cache = use_cache and not teacher_forced
            self.record("model", role, input_tokens=actual_tokens if sampling_cache else 0,
                        forward_tokens=0 if sampling_cache else actual_tokens, seconds=clock()-start,
                        details={"status": status, "time_basis": "host_dispatch",
                                 "padded_compute_tokens": ids.numel(), "use_cache": use_cache,
                                 "executed_decoder_layers": index + 1 if stop_at_hidden_layer else len(self.base.model.layers)})

    def packed_segment_hidden(self, token_ids, *, hidden_layer=-8, role="packed_segment",
                              hidden_positions=None, stop_at_hidden_layer=False):
        """Score causal tokens, optionally retaining only requested layer positions.

        ``hidden_positions`` selects sequence positions in the intermediate
        representation, without changing the full final hidden states used
        for response scoring. Representation-only callers may exit at that
        layer; both returned tensors then contain the selected positions.
        """
        return self.body(tuple(token_ids)[:-1], hidden_layer=hidden_layer, role=role,
                         hidden_positions=hidden_positions, stop_at_hidden_layer=stop_at_hidden_layer)[:2]

    @torch.no_grad()
    def observation_embeddings(self, rows, *, role="reference.observation_embedding", layer=-8):
        rows = tuple(tuple(row) for row in rows)
        if not rows:
            return torch.empty((0, self.base.config.hidden_size), dtype=torch.float32)
        if any(not row or len(row) > self.max_context_tokens for row in rows):
            raise ValueError("Observation representations require nonempty in-context token sequences")
        ordered = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        batches, current = [], []
        for index in ordered:
            width = len(rows[index])
            if current and (len(current) >= self.representation_batch_size or
                            width * (len(current) + 1) > self.representation_token_budget):
                batches.append(current)
                current = []
            current.append(index)
        if current:
            batches.append(current)
        result = torch.empty((len(rows), self.base.config.hidden_size), dtype=torch.float32)
        for indices in batches:
            selected_rows = [rows[i] for i in indices]
            lengths = [len(row) for row in selected_rows]
            width = max(lengths)
            ids = torch.tensor([row + (self.tokenizer.pad_token_id,) * (width - len(row))
                                for row in selected_rows], dtype=torch.long, device=self.device)
            valid = torch.arange(width, device=self.device)[None, :] < torch.tensor(lengths, device=self.device)[:, None]
            hidden, _, _ = self.body(ids, attention_mask=valid, hidden_layer=layer,
                stop_at_hidden_layer=True, actual_tokens=sum(lengths), logical_length=width, role=role)
            vectors = hidden[torch.arange(len(indices), device=hidden.device),
                             torch.tensor(lengths, device=hidden.device) - 1]
            result[indices] = torch.nn.functional.normalize(vectors.float(), dim=-1).cpu()
            # Release this batch before evaluating the next RHS: assignment to
            # `hidden` alone would retain the previous full output meanwhile.
            del hidden, vectors, ids, valid
        return result

    @torch.no_grad()
    def prefill_root(self, token_ids, *, role="rollout.root"):
        if self.model.training:
            self.model.eval()
        ids = tuple(token_ids)
        if not ids:
            raise ValueError("A causal root needs at least one prompt token")
        cache = ArenaCache(len(self.base.model.layers), initial_capacity=self.cache_initial_capacity,
                           growth_limit=self.max_context_tokens)
        hidden, _, cache = self.body(ids, cache=cache, use_cache=True, role=role,
                                     actual_tokens=len(ids), logical_length=len(ids))
        logits = self.head(hidden[:, -1], role=role+".head")
        return KVState(cache, torch.ones((1,len(ids)), dtype=torch.bool, device=self.device),
                       logits, (len(ids),), (((0, len(ids)),),))

    @torch.no_grad()
    def prefill_batch(self, prompts, *, role="rollout.batch_root"):
        """Prefill complete independent prompts with explicit right padding.

        Each row keeps its own logical RoPE positions and last real-token
        distribution. Padding is masked throughout subsequent cache appends;
        prompt tokens are never truncated to form a batch.
        """
        if self.model.training:
            self.model.eval()
        prompts = tuple(tuple(prompt) for prompt in prompts)
        lengths = tuple(len(prompt) for prompt in prompts)
        if not lengths or not all(lengths):
            raise ValueError("A batched causal prefill requires nonempty prompts")
        width = max(lengths)
        if width > self.max_context_tokens:
            raise ValueError("A complete prompt exceeds the logical context capacity")
        ids = torch.tensor([prompt + (self.tokenizer.pad_token_id,) * (width - len(prompt))
                            for prompt in prompts], dtype=torch.long, device=self.device)
        counts = torch.tensor(lengths, dtype=torch.long, device=self.device)
        valid = torch.arange(width, device=self.device)[None, :] < counts[:, None]
        cache = ArenaCache(len(self.base.model.layers), initial_capacity=self.cache_initial_capacity,
                           growth_limit=self.max_context_tokens)
        hidden, _, cache = self.body(ids, attention_mask=valid, cache=cache, use_cache=True, role=role,
                                     actual_tokens=sum(lengths), logical_length=width)
        row_indices = torch.arange(len(prompts), device=hidden.device)
        last = counts.to(hidden.device) - 1
        logits = self.head(hidden[row_indices, last], role=role+".head")
        spans = tuple(((0, length),) for length in lengths)
        return KVState(cache, valid, logits, lengths, spans)

    def cache_tensors(self, cache):
        return [(entry[0],entry[1]) for entry in cache]

    @torch.no_grad()
    def restore_cache(self, tensors):
        layers = []
        for i, (key, value) in enumerate(tensors):
            device = next(self.base.model.layers[i].parameters()).device
            key, value = key.to(device), value.to(device)
            layer = ArenaLayer(initial_capacity=self.cache_initial_capacity, growth_limit=self.max_context_tokens)
            layer.update(key, value)
            layers.append(layer)
        return ArenaCache(layers=layers)

    @torch.no_grad()
    def fork_cache(self, root: KVState, copies=8):
        if copies < 1:
            raise ValueError("Cache fork requires at least one copy")
        indices = torch.arange(len(root.lengths), dtype=torch.long).repeat_interleave(copies)
        return KVState(root.cache.copy_rows(indices), root.valid_mask.repeat_interleave(copies,0),
                       root.next_logits.repeat_interleave(copies,0),
                       tuple(n for n in root.lengths for _ in range(copies)),
                       tuple(spans for spans in root.valid_spans for _ in range(copies)), root.right_padded)

    @torch.no_grad()
    def select_cache(self, state: KVState, indices, *, consume=False):
        """Select independent rows, or transfer an exclusively owned state.

        The consuming form drops departing rows one layer at a time and keeps
        identity selections without copying.  Callers relinquish the old state.
        """
        indices = tuple(int(index) for index in indices)
        selection = torch.tensor(indices, dtype=torch.long)
        lengths = tuple(state.lengths[index] for index in indices)
        spans = tuple(state.valid_spans[index] for index in indices)
        if consume:
            if indices == tuple(range(state.valid_mask.shape[0])):
                return state
            state.cache.batch_select_indices(selection)
            state.valid_mask = state.valid_mask.index_select(0, selection.to(state.valid_mask.device))
            state.next_logits = state.next_logits.index_select(0, selection.to(state.next_logits.device))
            state.lengths = lengths
            state.valid_spans = spans
            return state
        return KVState(state.cache.copy_rows(selection),
                       state.valid_mask.index_select(0, selection.to(state.valid_mask.device)),
                       state.next_logits.index_select(0, selection.to(state.next_logits.device)),
                       lengths, spans, state.right_padded)

    def release_cache(self, state: KVState):
        """Relinquish an obsolete KVState without modifying any transferred cache."""
        state.cache = state.valid_mask = state.next_logits = None
        state.lengths = ()
        state.valid_spans = ()

    @torch.no_grad()
    def merge_cache(self, states):
        """Consume compact sibling states without recomputing any model token."""
        states = tuple(states)
        if not states:
            raise ValueError("Cache merging requires at least one owned state")
        if len(states) == 1:
            return states[0]
        if any(not state.right_padded or state.valid_mask.shape[1] != max(state.lengths)
               for state in states):
            raise ValueError("Merge complete compacted prefixes at the turn barrier")
        lengths = tuple(length for state in states for length in state.lengths)
        spans = tuple(row for state in states for row in state.valid_spans)
        width = max(lengths)
        counts = torch.tensor(lengths, device=self.device)
        valid = torch.arange(width, device=self.device)[None, :] < counts[:, None]
        logits = torch.cat([state.next_logits for state in states], dim=0)
        try:
            cache = ArenaCache.merge_owned([state.cache for state in states])
            return KVState(cache, valid, logits, lengths, spans, True)
        finally:
            for state in states:
                self.release_cache(state)

    @torch.no_grad()
    def compact_cache(self, state: KVState):
        """Compact real padding only; retained arena capacity is reusable."""
        length = max(state.lengths, default=0)
        if state.right_padded and state.valid_mask.shape[1] == length:
            return state
        if state.right_padded:
            state.cache.crop(length)
        else:
            positions = [torch.cat([torch.arange(start, stop, device=self.device)
                                    for start, stop in spans]) if spans else
                         torch.empty(0, dtype=torch.long, device=self.device)
                         for spans in state.valid_spans]
            state.cache.compact(positions, state.lengths)
        counts = torch.tensor(state.lengths, device=self.device)
        state.valid_mask = torch.arange(length, device=self.device)[None, :] < counts[:, None]
        state.valid_spans = tuple(((0, count),) if count else () for count in state.lengths)
        state.right_padded = True
        return state

    @torch.no_grad()
    def append_cache(self, state: KVState, rows: Sequence[Sequence[int]], *, role="rollout.increment"):
        """Extend exclusively owned KV storage and transfer it to the result."""
        if self.model.training:
            self.model.eval()
        lengths = [len(row) for row in rows]
        if len(lengths) != len(state.lengths):
            raise ValueError("Cache append rows must match the owned trajectory slots")
        width = max(lengths, default=0)
        if width == 0:
            return state
        batch = len(rows)
        totals = tuple(old + new for old, new in zip(state.lengths, lengths, strict=True))
        if max(totals) > self.max_context_tokens:
            raise ValueError("Cache append exceeds the logical context capacity")
        ids = torch.tensor([tuple(row) + (self.tokenizer.pad_token_id,) * (width-len(row))
                            for row in rows], dtype=torch.long, device=self.device)
        added = torch.tensor(lengths, device=self.device)
        offsets = torch.arange(width, device=self.device)
        valid = offsets[None, :] < added[:, None]
        mask = torch.cat((state.valid_mask.to(self.device),valid),dim=1)
        previous = torch.tensor(state.lengths, device=self.device)
        positions = torch.minimum(previous[:, None] + offsets[None, :],
                                  (previous + added - 1)[:, None]).clamp_min(0)
        physical = state.valid_mask.shape[1]
        active = tuple(i for i, count in enumerate(lengths) if count)
        return self._append_prepared_cache(state, ids, mask, positions,
            torch.arange(physical,physical+width,device=self.device),
            lengths=lengths, totals=totals, active=active, role=role)

    @torch.no_grad()
    def decode_workspace(self, state: KVState, max_tokens):
        return _DecodeWorkspace(self, state, max_tokens)

    @torch.no_grad()
    def append_sampled_cache(self, state: KVState, workspace, continuing, *, role="rollout.decode"):
        """Append retained GPU samples; EOS/cap decisions remain with the caller."""
        if self.model.training:
            self.model.eval()
        continuing = tuple(continuing)
        keep = set(continuing)
        totals = tuple(old + int(row in keep) for row, old in enumerate(state.lengths))
        if max(totals) > self.max_context_tokens:
            raise ValueError("Cache append exceeds the logical context capacity")
        try:
            ids, mask, positions, cache_position, lengths = workspace.prepare(state, continuing)
            return self._append_prepared_cache(state, ids, mask, positions, cache_position,
                lengths=lengths, totals=totals, active=continuing, role=role, workspace=workspace)
        finally:
            workspace.clear_sample()

    def _append_prepared_cache(self, state, ids, mask, positions, cache_position, *,
                               lengths, totals, active, role, workspace=None):
        batch, width = ids.shape
        physical = state.valid_mask.shape[1]
        hidden,_,cache = self.body(ids,attention_mask=mask,position_ids=positions,cache=state.cache,
            cache_position=cache_position,use_cache=True,role=role,
            actual_tokens=sum(lengths), logical_length=max(totals))
        indices = (torch.tensor(active, dtype=torch.long, device=hidden.device) if workspace is None else
                   workspace.indices(active, hidden.device))
        last = (torch.tensor([lengths[i] - 1 for i in active], device=hidden.device) if workspace is None else
                workspace.zeros(len(active), hidden.device))
        selected = hidden[indices, last]
        projected = self.head(selected, role=role+".head")
        if len(active) == batch:
            logits = projected
        else:
            # Inactive slots keep their last causal distribution. Like the KV
            # arena, this buffer is exclusively owned and transferred by the
            # caller, so it can be updated without cloning a full vocabulary
            # table or projecting padding tokens for completed responses.
            logits = state.next_logits
            target_indices = (indices.to(logits.device) if workspace is None else
                              workspace.indices(active, logits.device))
            logits.index_copy_(0, target_indices, projected)
        right_padded = state.right_padded and all(not n or state.lengths[i] == physical
                                                for i, n in enumerate(lengths))
        spans = tuple(old if not count else
                      old[:-1] + ((old[-1][0], physical + count),)
                      if old and old[-1][1] == physical else old + ((physical, physical + count),)
                      for old, count in zip(state.valid_spans, lengths, strict=True))
        return KVState(cache,mask,logits,totals,spans,right_padded)

    @torch.no_grad()
    def draw_tokens(self, state: KVState, active, generators, *, temperature=None, top_p=1.0, workspace=None):
        """Return the same CPU draw and optionally retain its device token IDs."""
        active, generators = tuple(active), tuple(generators)
        if len(active) != len(generators):
            raise ValueError("Each active trajectory requires its own random generator")
        if not active:
            return [], []
        temperature = self.sampling_temperature if temperature is None else float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive")
        if float(top_p) != 1.:
            raise ValueError("The published sampling protocol uses top_p=1.0")
        selected = (torch.tensor(active, dtype=torch.long, device=state.next_logits.device) if workspace is None else
                    workspace.indices(active, state.next_logits.device))
        logits = state.next_logits.index_select(0, selected).float()
        if temperature != 1.:
            logits.div_(temperature)
        logp = torch.log_softmax(logits, dim=-1)
        del logits
        probabilities = logp.exp()
        tokens = torch.cat([torch.multinomial(probabilities[row], 1, generator=generator)
                            for row, generator in enumerate(generators)])
        scores = logp.gather(1, tokens[:, None]).squeeze(1)
        # The CPU needs termination decisions before dispatching environments.
        # One packed readback replaces per-slot int/float synchronization while
        # multinomial still consumes each slot's independent generator stream.
        sampled = torch.stack((tokens.double(), scores.double()), dim=1).cpu().tolist()
        ids = [int(row[0]) for row in sampled]
        logprobs = [row[1] for row in sampled]
        if workspace is not None:
            workspace.retain(tokens, active)
        self.record("model","rollout.draw",output_tokens=len(ids),seconds=0.)
        return ids,logprobs

    def save_model(self, directory):
        # Keep only detached device aliases here. Transformers first groups
        # them into shards; safetensors moves the current shard to CPU during
        # serialization. An individual tensor larger than the configured
        # shard budget occupies its own shard. The official save path retains
        # config, generation_config, tied-weight handling and index metadata.
        self.synchronize()
        self.model.save_pretrained(directory,state_dict=self.model.state_dict(),
                                   safe_serialization=True,max_shard_size=self.checkpoint_shard_bytes)
        self.tokenizer.save_pretrained(directory)
