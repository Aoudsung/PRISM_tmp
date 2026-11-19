"""Collection-only Qwen backbone, without dispatch hooks or a vocabulary head."""
from copy import deepcopy

import torch
from torch import nn


@torch.no_grad()
def copy_backbone(source, device):
    # Meta construction consumes no sampling RNG and creates no weight staging
    # allocation. Take each parameter directly from its current learner shard.
    with torch.device("meta"):
        replica = type(source)(deepcopy(source.config))
    for name, parameter in source.named_parameters():
        path, _, leaf = name.rpartition(".")
        weight = parameter.detach().to(device=device, copy=True)
        setattr(replica.get_submodule(path), leaf, nn.Parameter(weight, requires_grad=False))
    for name, buffer in source.named_buffers():
        path, _, leaf = name.rpartition(".")
        setattr(replica.get_submodule(path), leaf, buffer.detach().to(device=device, copy=True))
    rotary = source.rotary_emb
    replica.rotary_emb.original_inv_freq = (
        replica.rotary_emb.inv_freq if rotary.original_inv_freq is rotary.inv_freq
        else rotary.original_inv_freq.detach().to(device=device, copy=True))
    replica.rotary_emb.max_seq_len_cached = rotary.max_seq_len_cached
    replica.rotary_emb.attention_scaling = rotary.attention_scaling
    return replica.eval()
