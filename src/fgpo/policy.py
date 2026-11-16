"""Qwen full-softmax causal execution, explicit CUDA layer shards and exact KV state."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn


def clock() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC)


@dataclass
class KVState:
    cache: Any
    valid_mask: torch.Tensor
    next_logits: torch.Tensor


def layer_device_map(config, gpu_ids, free_gib=None, *, full_training=True):
    if not gpu_ids:
        raise ValueError("Qwen requires an explicitly allocated healthy CUDA device")
    capacities = [float(free_gib[g]) if free_gib and g in free_gib else
                  torch.cuda.mem_get_info(g)[0] / 2**30 for g in gpu_ids]
    h, intermediate = int(config.hidden_size), int(config.intermediate_size)
    kv = int(config.num_key_value_heads) * (h // int(config.num_attention_heads))
    width = 4 if full_training else 2
    layer_size = (2*h*h + 2*h*kv + 3*h*intermediate + 4*h) * width / 2**30
    embedding = int(config.vocab_size) * h * width / 2**30
    capacities = [v - 3. for v in capacities]
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
                 activation_offload: bool = True, benchmark: str = "alfworld"):
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
                                          full_training=trainable and adaptation is None)
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
        self.max_context_tokens = min(32768, int(cfg.max_position_embeddings))
        self.max_new_tokens = 256 if benchmark == "alfworld" else 1024
        self.eos_token_ids = (151645, 151643)
        self._reference_active = False
        self.metadata = {"base_model": model_path, "adaptation": adaptation, "dtype": "bfloat16",
                         "compute_logical_gpu_ids":list(self.gpu_ids),
                         "attention": "sdpa", "device_map": dict(self.device_map),
                         "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)}

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @property
    def base(self):
        return self.model.get_base_model() if self.adaptation else self.model

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
        start, status = clock(), "OK"
        try:
            yield
        except BaseException:
            status = "ERROR"
            raise
        finally:
            self.synchronize()
            self.record("compute", role, seconds=clock()-start, details={"status": status})

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True))

    def new_generator(self, seed: int):
        generator = torch.Generator(device=self.model.get_output_embeddings().weight.device)
        generator.manual_seed(int(seed))
        return generator

    def head(self, hidden: torch.Tensor, *, role="vocabulary_projection"):
        self.synchronize()
        started, status = clock(), "OK"
        try:
            module = self.model.get_output_embeddings()
            value = module(hidden.to(module.weight.device)).float()
            self.synchronize()
            return value
        except BaseException:
            status = "ERROR"
            raise
        finally:
            self.record("model", role, seconds=clock()-started,
                        details={"status": status, "projection_continuation": True, "positions": hidden.shape[-2]})

    def body(self, input_ids, *, attention_mask=None, position_ids=None, cache=None,
             cache_position=None, use_cache=False, hidden_layer=None, role="forward", teacher_forced=False):
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        mask = (torch.ones_like(ids) if attention_mask is None else attention_mask.to(self.device))
        if position_ids is None:
            position_ids = (mask.long().cumsum(-1)-1).clamp_min(0)[:, -ids.shape[1]:]
        else:
            position_ids = position_ids.to(self.device)
        if int(position_ids.max()) >= self.max_context_tokens:
            raise ValueError("Exact Qwen context capacity exceeded")
        captures = []
        hook = None
        if hidden_layer is not None:
            layers = self.base.model.layers
            index = len(layers) + hidden_layer if hidden_layer < 0 else hidden_layer
            def capture(_module, _args, output):
                captures.append(output[0] if isinstance(output, tuple) else output)
            hook = layers[index].register_forward_hook(capture)
        kwargs = dict(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                      past_key_values=cache, use_cache=use_cache)
        if cache_position is not None:
            kwargs["cache_position"] = cache_position.to(self.device)
        self.synchronize()
        start, status = clock(), "OK"
        try:
            with self.activation_saving():
                output = self.base.model(**kwargs)
            self.synchronize()
            selected = captures[0] if captures else None
            return output.last_hidden_state, selected, output.past_key_values if use_cache else None
        except BaseException:
            status = "ERROR"
            raise
        finally:
            if hook is not None:
                hook.remove()
            actual = int(mask[:, -ids.shape[1]:].sum())
            sampling_cache = use_cache and not teacher_forced
            self.record("model", role, input_tokens=actual if sampling_cache else 0,
                        forward_tokens=0 if sampling_cache else actual, seconds=clock()-start,
                        details={"status": status, "padded_compute_tokens": ids.numel(), "use_cache": use_cache})

    def packed_segment_hidden(self, token_ids, *, hidden_layer=-8, role="packed_segment"):
        return self.body(tuple(token_ids)[:-1], hidden_layer=hidden_layer, role=role)[:2]

    @torch.no_grad()
    def observation_embeddings(self, rows, *, role="reference.observation_embedding", layer=-8):
        from torch.nn import ModuleList, Identity
        original_layers,original_norm = self.base.model.layers,self.base.model.norm
        stop = len(original_layers)+layer+1 if layer < 0 else layer+1
        try:
            self.base.model.layers = ModuleList(list(original_layers[:stop]))
            self.base.model.norm = Identity()
            width = max(len(row) for row in rows)
            ids = torch.full((len(rows),width),self.tokenizer.pad_token_id,dtype=torch.long,device=self.device)
            valid = torch.zeros_like(ids,dtype=torch.bool)
            for i,row in enumerate(rows):
                ids[i,:len(row)] = torch.tensor(row,device=self.device)
                valid[i,:len(row)] = True
            hidden,_,_ = self.body(ids,attention_mask=valid,role=role)
            vectors = hidden[torch.arange(len(rows),device=hidden.device),
                             torch.tensor([len(row)-1 for row in rows],device=hidden.device)]
            return torch.nn.functional.normalize(vectors.float(),dim=-1).cpu()
        finally:
            self.base.model.layers,self.base.model.norm = original_layers,original_norm

    @torch.no_grad()
    def prefill_root(self, token_ids, *, role="rollout.root"):
        self.model.eval()
        ids = tuple(token_ids)
        if not ids:
            raise ValueError("A causal root needs at least one prompt token")
        hidden, _, cache = self.body(ids, use_cache=True, role=role)
        logits = self.head(hidden[:, -1], role=role+".head")
        return KVState(cache, torch.ones((1,len(ids)), dtype=torch.bool, device=self.device), logits)

    def cache_tensors(self, cache):
        return [(entry[0],entry[1]) for entry in cache]

    def restore_cache(self, tensors):
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer
        result = DynamicCache()
        for i, (key, value) in enumerate(tensors):
            device = next(self.base.model.layers[i].parameters()).device
            key, value = key.to(device), value.to(device)
            # DynamicLayer.update concatenates even its initially empty KV.
            # These tensors already contain the complete layer: transfer them
            # directly, preserving the installed layer's initialization state.
            layer = DynamicLayer()
            layer.lazy_initialization(key)
            layer.keys, layer.values = key, value
            result.layers.append(layer)
        return result

    @torch.no_grad()
    def fork_cache(self, root: KVState, copies=8):
        tensors = ((layer.keys.repeat_interleave(copies, 0), layer.values.repeat_interleave(copies, 0))
                   for layer in root.cache.layers)
        return KVState(self.restore_cache(tensors), root.valid_mask.repeat_interleave(copies,0),
                       root.next_logits.repeat_interleave(copies,0))

    @torch.no_grad()
    def select_cache(self, state: KVState, indices, *, consume=False):
        """Select independent rows, or transfer an exclusively owned state.

        The consuming form drops departing rows one layer at a time and keeps
        identity selections without copying.  Callers relinquish the old state.
        """
        indices = tuple(int(index) for index in indices)
        selection = torch.tensor(indices, dtype=torch.long)
        if consume:
            if indices == tuple(range(state.valid_mask.shape[0])):
                return state
            for layer in state.cache.layers:
                layer.keys = layer.keys.index_select(0, selection.to(layer.keys.device))
                layer.values = layer.values.index_select(0, selection.to(layer.values.device))
            state.valid_mask = state.valid_mask.index_select(0, selection.to(state.valid_mask.device))
            state.next_logits = state.next_logits.index_select(0, selection.to(state.next_logits.device))
            return state
        tensors = ((layer.keys.index_select(0, selection.to(layer.keys.device)),
                    layer.values.index_select(0, selection.to(layer.values.device)))
                   for layer in state.cache.layers)
        return KVState(self.restore_cache(tensors),
                       state.valid_mask.index_select(0, selection.to(state.valid_mask.device)),
                       state.next_logits.index_select(0, selection.to(state.next_logits.device)))

    def release_cache(self, state: KVState):
        """Relinquish an obsolete KVState without modifying any transferred cache."""
        state.cache = state.valid_mask = state.next_logits = None

    @torch.no_grad()
    def compact_cache(self, state: KVState):
        """Pack valid positions in owned layer storage, without a second full KV.

        Each row gather is copied back before the next row.  The shortened
        views keep exact K/V and RoPE values; the next normal cache append
        replaces their backing storage layer by layer with its concatenation.
        """
        counts = state.valid_mask.sum(-1)
        length = int(counts.max())
        positions = [torch.where(row)[0] for row in state.valid_mask]
        valid = torch.arange(length,device=self.device)[None,:] < counts.to(self.device)[:,None]
        if state.valid_mask.shape[1] == length and torch.equal(state.valid_mask.to(self.device), valid):
            return state
        for layer in state.cache.layers:
            key, value = layer.keys, layer.values
            for row, idx in enumerate(positions):
                n = len(idx)
                key[row,:,:n].copy_(key[row].index_select(1,idx.to(key.device)))
                value[row,:,:n].copy_(value[row].index_select(1,idx.to(value.device)))
                key[row,:,n:length].zero_()
                value[row,:,n:length].zero_()
            layer.keys, layer.values = key[:,:,:length], value[:,:,:length]
        state.valid_mask = valid
        return state

    @torch.no_grad()
    def append_cache(self, state: KVState, rows: Sequence[Sequence[int]], *, role="rollout.increment"):
        self.model.eval()
        lengths = [len(row) for row in rows]
        width = max(lengths, default=0)
        if width == 0:
            return state
        batch = len(rows)
        ids = torch.full((batch,width),self.tokenizer.pad_token_id,dtype=torch.long,device=self.device)
        valid = torch.zeros_like(ids,dtype=torch.bool)
        for i,row in enumerate(rows):
            if row:
                ids[i,:len(row)] = torch.tensor(row,device=self.device)
                valid[i,:len(row)] = True
        mask = torch.cat((state.valid_mask.to(self.device),valid),dim=1)
        positions = (mask.long().cumsum(-1)-1).clamp_min(0)[:,-width:]
        physical = state.valid_mask.shape[1]
        hidden,_,cache = self.body(ids,attention_mask=mask,position_ids=positions,cache=state.cache,
            cache_position=torch.arange(physical,physical+width,device=self.device),use_cache=True,role=role)
        last = torch.tensor([max(0,n-1) for n in lengths],device=hidden.device)
        selected = hidden[torch.arange(batch,device=hidden.device),last]
        logits = self.head(selected,role=role+".head")
        for i,n in enumerate(lengths):
            if not n:
                logits[i] = state.next_logits[i]
        return KVState(cache,mask,logits)

    @torch.no_grad()
    def draw_tokens(self, state: KVState, active, generators):
        ids, logprobs = [], []
        for row,generator in zip(active,generators):
            logp = torch.log_softmax(state.next_logits[row].float(),dim=-1)
            token = int(torch.multinomial(logp.exp(),1,generator=generator))
            ids.append(token)
            logprobs.append(float(logp[token]))
        self.record("model","rollout.draw",output_tokens=len(ids),seconds=0.)
        return ids,logprobs

    def save_model(self, directory):
        self.model.save_pretrained(directory,state_dict={k:v.detach().cpu() for k,v in self.model.state_dict().items()},
                                   safe_serialization=True,max_shard_size="4GB")
        self.tokenizer.save_pretrained(directory)
