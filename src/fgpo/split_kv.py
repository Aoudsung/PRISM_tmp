"""Single-token BF16 Qwen GQA over the occupied Arena KV rows."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


_ATTENTION_NAME = "fgpo_split_kv"
_KEYS_PER_SPLIT = 512
_HEADS_PER_KV = 7
_HEAD_DIM = 128


@triton.jit(do_not_specialize=["Tail", "Width", "KV_B", "KV_H", "S"])
def _partial(Q, K, V, Prefix, Partial, Stats, Tail, Width, Scale, KV_B, KV_H, S,
             QB: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
             SPLIT: tl.constexpr, TILE: tl.constexpr,
             HEADS: tl.constexpr, D: tl.constexpr):
    group = tl.program_id(0)
    part = tl.program_id(1)
    batch, kv_head = group // 4, group % 4
    qh = tl.arange(0, 8)
    d = tl.arange(0, D)
    keys = tl.arange(0, TILE)
    prefix = tl.load(Prefix + batch)
    length = prefix + Width - Tail
    base = batch * KV_B + kv_head * KV_H
    q = tl.load(Q + batch * QB + (kv_head * HEADS + qh[:, None]) * QH + d[None, :] * QD,
                qh[:, None] < HEADS, other=0).to(tl.float32)
    maximum = tl.full((8,), -float("inf"), tl.float32)
    denominator = tl.full((8,), 0., tl.float32)
    weighted = tl.full((8, D), 0., tl.float32)
    for start in range(part * SPLIT, tl.minimum((part + 1) * SPLIT, length), TILE):
        logical = start + keys
        physical = tl.where(logical < prefix, logical, Tail + logical - prefix)
        valid = logical < length
        offsets = base + physical[:, None] * D + d[None, :]
        k = tl.load(K + offsets, valid[:, None], other=0).to(tl.float32)
        # Triton 3.3's matrix lowering does not support the 6000D's SM120.
        # The seven queries share these loads and reduce in FP32 on CUDA cores.
        score = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * Scale
        score = tl.where(valid[None, :], score, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(score, axis=1))
        rescale = tl.exp(maximum - new_maximum)
        probability = tl.exp(score - new_maximum[:, None])
        v = tl.load(V + offsets, valid[:, None], other=0).to(tl.float32)
        weighted = weighted * rescale[:, None] + tl.sum(probability[:, :, None] * v[None, :, :], axis=1)
        denominator = denominator * rescale + tl.sum(probability, axis=1)
        maximum = new_maximum
    head = group * HEADS + qh
    tl.store(Partial + (head[:, None] * S + part) * D + d[None, :], weighted, qh[:, None] < HEADS)
    tl.store(Stats + head * 2 * S + part, maximum, qh < HEADS)
    tl.store(Stats + head * 2 * S + S + part, denominator, qh < HEADS)


@triton.jit(do_not_specialize=["Parts", "S"])
def _reduce(Partial, Stats, Output, Parts, S, BLOCK: tl.constexpr, D: tl.constexpr):
    head = tl.program_id(0)
    s = tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    m = tl.load(Stats + head * 2 * S + s, s < Parts, other=-float("inf"))
    factor = tl.exp(m - tl.max(m, axis=0))
    sums = tl.load(Stats + head * 2 * S + S + s, s < Parts, other=0.)
    weighted = tl.load(Partial + (head * S + s[:, None]) * D + d[None, :], s[:, None] < Parts, other=0.)
    output = tl.sum(weighted * factor[:, None], axis=0) / tl.sum(sums * factor, axis=0)
    tl.store(Output + head * D + d, output)


class SplitKVWorkspace:
    """The right-padded response prefix and contiguous physical suffix."""

    def __init__(self, state, max_appends):
        self.rebase(state)
        self.capacity = triton.cdiv(max(state.lengths) + max_appends, _KEYS_PER_SPLIT)
        self.partial = None
        self.stats = None

    def rebase(self, state):
        if any(n <= 0 or spans != ((0, n),)
               for n, spans in zip(state.lengths, state.valid_spans, strict=True)):
            raise ValueError("Split KV sampling needs a nonempty right-padded prefix at the response boundary")
        self.prefix = torch.tensor(state.lengths, device=state.valid_mask.device, dtype=torch.int32)
        self.tail = state.valid_mask.shape[1]

    def select_rows(self, indices, physical):
        self.prefix = self.prefix.index_select(0, indices)
        self.tail = min(self.tail, physical)

    def attention(self, query, key, value, logical_length, scale):
        batch, _, width, _ = key.shape
        parts = triton.cdiv(logical_length, _KEYS_PER_SPLIT)
        heads = batch * 4 * _HEADS_PER_KV
        if (query.shape != (batch, 28, 1, _HEAD_DIM) or key.shape != value.shape
                or key.shape[1] != 4 or key.shape[-1] != _HEAD_DIM
                or key.stride() != value.stride() or key.stride()[2:] != (_HEAD_DIM, 1)
                or query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype
                or not 0 <= self.tail <= width or not 1 <= parts <= self.capacity
                or self.prefix.numel() != batch):
            raise ValueError("Split KV sampling received unsupported Qwen decode inputs")
        if self.partial is None:
            self.partial = torch.empty((heads, self.capacity, _HEAD_DIM), dtype=torch.float32, device=query.device)
            self.stats = torch.empty((heads, 2, self.capacity), dtype=torch.float32, device=query.device)
        output = torch.empty((batch, 1, 28, _HEAD_DIM), dtype=query.dtype, device=query.device)
        with torch.cuda.device(query.device):
            _partial[(batch * 4, parts)](query, key, value, self.prefix, self.partial, self.stats,
                self.tail, width, scale, key.stride(0), key.stride(1), self.capacity,
                query.stride(0), query.stride(1), query.stride(3), _KEYS_PER_SPLIT, 16, _HEADS_PER_KV, _HEAD_DIM,
                num_warps=4, num_stages=2)
            _reduce[(heads,)](self.partial, self.stats, output, parts, self.capacity,
                              triton.next_power_of_2(parts), _HEAD_DIM, num_warps=4)
        return output


def install_split_kv(model):
    decoder = model.model
    config = decoder.config
    embeddings = model.get_input_embeddings().weight
    if (config.model_type != "qwen2" or config.num_hidden_layers != 28 or config.num_attention_heads != 28
            or config.num_key_value_heads != 4 or decoder.layers[0].self_attn.head_dim != _HEAD_DIM
            or decoder.has_sliding_layers or embeddings.dtype != torch.bfloat16
            or embeddings.device.type != "cuda" or torch.cuda.get_device_capability(embeddings.device) < (8, 0)):
        raise ValueError("Split KV decode requires BF16 Qwen2.5-7B with full Ampere-or-newer CUDA attention")
    ALL_ATTENTION_FUNCTIONS.register(_ATTENTION_NAME, _attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(_ATTENTION_NAME, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    decoder.set_attn_implementation(_ATTENTION_NAME)


def _attention(module, query, key, value, attention_mask, *, fgpo_split_kv=None,
               dropout=0., scaling=None, **kwargs):
    if fgpo_split_kv is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    if attention_mask is not None or dropout or module.training or torch.is_grad_enabled():
        raise ValueError("Split KV attention requires the exclusive no-grad sampled decode path")
    workspace, logical_length = fgpo_split_kv
    return workspace.attention(query, key, value, logical_length, scaling), None
