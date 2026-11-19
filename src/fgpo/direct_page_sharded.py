"""Device-local direct decode captures for the existing sharded Qwen actor."""
from __future__ import annotations

from copy import deepcopy
from itertools import groupby
from types import SimpleNamespace

import torch

from .direct_page_graph import _DirectGraph
from .paged_graph import _DIM


class _SegmentGraph(_DirectGraph):
    def __init__(self, pool, buffers, stage, batch, capacity, all_keys_valid):
        super().__init__(pool, buffers.decode, batch, capacity, all_keys_valid)
        self.stage = stage
        self.hidden = buffers.hidden[:batch]
        self.rotary = tuple(tensor[:batch] for tensor in buffers.rotary)

    def load_stage(self, ids, positions, mask, hidden, rotary):
        if self.stage.embedding is not None:
            self.ids.copy_(ids)
        else:
            self.hidden.copy_(hidden, non_blocking=True)
            for destination, source in zip(self.rotary, rotary, strict=True):
                if destination.device != source.device:
                    destination.copy_(source, non_blocking=True)
        self.positions.copy_(positions, non_blocking=True)
        self.column.fill_(mask.shape[1] - 1)
        if not self.all_keys_valid and self.stage.layers:
            width = mask.shape[1]
            self.valid[:, :width].copy_(mask, non_blocking=True)
            self.valid[:, width:].zero_()

    def _forward(self, _actor):
        stage = self.stage
        if stage.embedding is None:
            hidden = self.hidden
        else:
            hidden = stage.embedding(self.ids)
            for target, source in zip(self.rotary, stage.rotary_emb(hidden, self.positions), strict=True):
                target.copy_(source)
        if stage.layers:
            if self.all_keys_valid:
                torch.mul(self.cu_q, self.pool.width, out=self.cu_k)
                self.seqused_k.copy_(self.positions[:, 0] + 1)
            else:
                self.bias_values.copy_(torch.where(self.valid, 0., -torch.inf))
            for layer in stage.layers:
                hidden = layer(hidden, attention_mask=None, position_ids=self.positions, use_cache=False,
                               position_embeddings=self.rotary, fgpo_paged_graph=self)
        if stage.norm is not None:
            hidden = stage.norm(hidden)
        if stage.head is None:
            self.hidden.copy_(hidden)
        else:
            self.logits.copy_(stage.head(hidden[:, 0]))


class _ShardedShape:
    def __init__(self, owner, batch, capacity, all_keys_valid):
        self.parts = tuple((stage.device, _SegmentGraph(owner.pool, owner.buffers[stage.device],
                                                       stage, batch, capacity, all_keys_valid))
                           for stage in owner.stages)
        self.streams = owner.streams

    def run(self, actor, ids, positions, table, mask, capture_stream):
        hidden = rotary = None
        warmup = capture = 0.
        created = False
        for device, part in self.parts:
            with torch.cuda.device(device):
                part.load_stage(ids, positions, mask, hidden, rotary)
                setup = part.replay(actor, self.streams[device])
                if setup is not None:
                    created = True
                    warmup += setup[0]
                    capture += setup[1]
                hidden = part.hidden
                if rotary is None:
                    rotary = part.rotary
        return part.logits.clone(), (warmup, capture) if created else None

    def reset(self):
        for device, part in self.parts:
            with torch.cuda.device(device):
                part.reset()


class ShardedActor:
    def __init__(self, actor, pool):
        self.pool = pool
        decoder, head = actor.model, actor.get_output_embeddings()
        dimensions = decoder.config.hidden_size
        operations = [(pool.device, "embedding", decoder.embed_tokens),
                      *((device, "layer", layer) for device, layer in zip(pool.layer_devices, decoder.layers, strict=True)),
                      (decoder.norm.weight.device, "norm", decoder.norm), (head.weight.device, "head", head)]
        stage_rows = []
        for device, rows in groupby(operations, key=lambda row: row[0]):
            stage = SimpleNamespace(device=device, embedding=None, layers=[], norm=None, head=None)
            for _, role, module in rows:
                if role == "layer":
                    stage.layers.append(module)
                else:
                    setattr(stage, role, module)
            stage_rows.append(stage)
        self.stages = tuple(stage_rows)
        rotary = decoder.rotary_emb
        if rotary.inv_freq.device != pool.device:
            # Accelerate may leave the nonpersistent frequency table on CPU.
            # Hold a decode-only copy on the embedding device before capture.
            same_frequency = rotary.original_inv_freq is rotary.inv_freq
            rotary = deepcopy(rotary).to(device=pool.device)
            rotary.original_inv_freq = (rotary.inv_freq if same_frequency
                                        else rotary.original_inv_freq.to(device=pool.device))
        self.stages[0].rotary_emb = rotary
        self.buffers, self.streams = {}, {}
        for device in pool.devices:
            hidden = torch.empty((pool.active_rows, 1, dimensions), dtype=torch.bfloat16, device=device)
            cos = torch.empty((pool.active_rows, 1, _DIM), dtype=torch.bfloat16, device=device)
            self.buffers[device] = SimpleNamespace(decode=pool.device_buffers[device], hidden=hidden,
                                                   rotary=(cos, torch.empty_like(cos)))
            self.streams[device] = torch.cuda.Stream(device=device)

    def build(self, batch, capacity, all_keys_valid):
        return _ShardedShape(self, batch, capacity, all_keys_valid)
