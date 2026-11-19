"""Single-GPU Qwen2.5 masked rollout attention for Torch 2.7 and HF 4.57."""
from __future__ import annotations

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


_ATTENTION_NAME = "fgpo_masked_decode"


def install_masked_decode(model):
    decoder = model.model
    config = decoder.config
    embeddings = model.get_input_embeddings().weight
    if (config.model_type != "qwen2" or config.num_attention_heads != 28
            or config.num_key_value_heads != 4 or decoder.layers[0].self_attn.head_dim != 128
            or decoder.has_sliding_layers or embeddings.dtype != torch.bfloat16
            or embeddings.device.type != "cuda"):
        raise ValueError("Host B masked decode requires BF16 Qwen2.5-7B with full CUDA attention")
    # The packed FP32 readback must represent every possible sampled token ID exactly.
    if model.get_output_embeddings().weight.shape[0] > 2**24:
        raise ValueError("Host B masked decode vocabulary exceeds exact FP32 token IDs")
    ALL_ATTENTION_FUNCTIONS.register(_ATTENTION_NAME, _attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(_ATTENTION_NAME, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    decoder.set_attn_implementation(_ATTENTION_NAME)


def forward_masked_decode(decoder, *, attention_mask, **kwargs):
    batch, width = attention_mask.shape
    # Cutlass SDPA requires aligned bias strides even when the live key width is ragged.
    aligned_width = ((width + 7) // 8) * 8
    bias = torch.full((batch, 1, 1, aligned_width), -torch.inf, dtype=torch.bfloat16,
                      device=attention_mask.device)[..., :width]
    bias.masked_fill_(attention_mask[:, None, None, :], 0.)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return decoder(attention_mask=bias, fgpo_masked_decode=True, **kwargs)


def _attention(module, query, key, value, attention_mask, *, fgpo_masked_decode=False,
               dropout=0., scaling=None, **kwargs):
    if not fgpo_masked_decode:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)

    batch, heads, _, head_dim = query.shape
    # Heads sharing a KV head are independent queries against the same physical keys.
    grouped_query = query.squeeze(2).unflatten(1, (key.shape[1], module.num_key_value_groups))
    output = scaled_dot_product_attention(grouped_query, key, value, attn_mask=attention_mask,
                                          dropout_p=0., is_causal=False, scale=scaling)
    return output.reshape(batch, 1, heads, head_dim), None
