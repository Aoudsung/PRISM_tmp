"""Direct native SDPA from durable NHD page runs for the AppWorld actor."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
import time

import torch
import triton
import triton.language as tl
from transformers.cache_utils import Cache

from .paged_graph import (_DecodeBuffers, _DecodeGraph, _DIM, _GROUP, _KV_HEADS,
                          _MASKED_BUCKET, _MAX_LIVE_GRAPHS, _PAGE)


@triton.jit
def _stage_runs(Pool, Meta, Key, Value, PLANE: tl.constexpr, STRIDE: tl.constexpr,
                SPANS: tl.constexpr, TILE: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0) * TILE + tl.arange(0, TILE)
    element = tl.arange(0, 4 * D)
    meta = Meta + row * (5 + 3 * SPANS)
    source, length = tl.load(meta), tl.load(meta + 2)
    physical = tl.full((TILE,), 0, tl.int32)
    for span in tl.static_range(SPANS):
        start = tl.load(meta + 5 + 3 * span)
        first = tl.load(meta + 6 + 3 * span)
        stop = tl.load(meta + 7 + 3 * span)
        physical = tl.where((col >= first) & (col < stop), start + col - first, physical)
    source_index = ((source * STRIDE + physical[:, None]) * 4 * D + element[None, :])
    stage_index = ((row * STRIDE + col[:, None]) * 4 * D + element[None, :])
    valid = col[:, None] < length
    key = tl.load(Pool + source_index, valid, other=0)
    value = tl.load(Pool + PLANE + source_index, valid, other=0)
    tl.store(Key + stage_index, key, valid)
    tl.store(Value + stage_index, value, valid)


@triton.jit
def _commit_runs(Pool, Meta, Key, Value, PLANE: tl.constexpr, STRIDE: tl.constexpr,
                 SPANS: tl.constexpr, TILE: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0) * TILE + tl.arange(0, TILE)
    element = tl.arange(0, 4 * D)
    meta = Meta + row * (5 + 3 * SPANS)
    destination, length = tl.load(meta + 1), tl.load(meta + 2)
    zero_start, zero_stop = tl.load(meta + 3), tl.load(meta + 4)
    target = ((destination * STRIDE + col[:, None]) * 4 * D + element[None, :])
    stage = ((row * STRIDE + col[:, None]) * 4 * D + element[None, :])
    copy = col[:, None] < length
    write = copy | ((col[:, None] >= zero_start) & (col[:, None] < zero_stop))
    key = tl.load(Key + stage, copy, other=0)
    value = tl.load(Value + stage, copy, other=0)
    tl.store(Pool + target, key, write)
    tl.store(Pool + PLANE + target, value, write)


@triton.jit(do_not_specialize=["Past", "Width", "K0", "K1", "K2", "V0", "V1", "V2"])
def _append_extent(K, V, Pool, Rows, DenseK, DenseV, Past, Width, K0, K1, K2, V0, V1, V2,
                   PLANE: tl.constexpr, STRIDE: tl.constexpr, TILE: tl.constexpr,
                   D: tl.constexpr, GATHER: tl.constexpr):
    group = tl.program_id(1)
    row, head = group // 4, group % 4
    cols = tl.program_id(0) * TILE + tl.arange(0, TILE)
    d = tl.arange(0, D)
    slot, added = tl.load(Rows + row * 2), tl.load(Rows + row * 2 + 1)
    if GATHER:
        physical, append = cols, cols - Past
    else:
        physical, append = cols + Past, cols
    new = (append >= 0) & (append < Width)
    valid = new & (append < added)
    load = new if GATHER else valid
    key = tl.load(K + (row * K0 + head * K1 + append[:, None] * K2 + d[None, :]),
                  load[:, None], other=0)
    value = tl.load(V + (row * V0 + head * V1 + append[:, None] * V2 + d[None, :]),
                    load[:, None], other=0)
    index = ((slot * STRIDE + physical[:, None]) * 4 + head) * D + d[None, :]
    tl.store(Pool + index, tl.where(valid[:, None], key, 0), new[:, None])
    tl.store(Pool + PLANE + index, tl.where(valid[:, None], value, 0), new[:, None])
    if GATHER:
        old = (physical < Past)[:, None]
        old_key = tl.load(Pool + index, old, other=0)
        old_value = tl.load(Pool + PLANE + index, old, other=0)
        width = Past + Width
        target = ((row * 4 + head) * width + cols[:, None]) * D + d[None, :]
        tl.store(DenseK + target, tl.where(old, old_key, key), cols[:, None] < width)
        tl.store(DenseV + target, tl.where(old, old_value, value), cols[:, None] < width)


@triton.jit
def _append_token(K, V, Pool, Column, K0: tl.constexpr, K1: tl.constexpr,
                  V0: tl.constexpr, V1: tl.constexpr, PLANE: tl.constexpr,
                  STRIDE: tl.constexpr, D: tl.constexpr):
    group = tl.program_id(0)
    row, head = group // 4, group % 4
    column = tl.load(Column).to(tl.int32)
    d = tl.arange(0, D)
    target = ((row * STRIDE + column) * 4 + head) * D + d
    tl.store(Pool + target, tl.load(K + row * K0 + head * K1 + d))
    tl.store(Pool + PLANE + target, tl.load(V + row * V0 + head * V1 + d))


@dataclass(eq=False)
class _Extent:
    slot: int
    owners: set = field(default_factory=set)

    @property
    def width(self):
        return max((row.width for row in self.owners), default=0)


@dataclass(eq=False)
class _Row:
    extent: _Extent
    width: int = 0

    def __post_init__(self):
        self.extent.owners.add(self)

    def release(self):
        self.extent.owners.remove(self)

    def relocate(self, extent):
        self.release()
        self.extent = extent
        extent.owners.add(self)


class _DirectPool:
    """A row owns a durable contiguous run of physical NHD pages."""
    def __init__(self, actor, rows, roots, context, record):
        decoder, head = actor.model, actor.get_output_embeddings()
        self.device = decoder.embed_tokens.weight.device
        self.layer_devices = tuple(layer.input_layernorm.weight.device for layer in decoder.layers)
        self.devices = tuple(dict.fromkeys((self.device, *self.layer_devices,
                                           decoder.norm.weight.device, head.weight.device)))
        self.width = triton.cdiv(context, _MASKED_BUCKET) * _MASKED_BUCKET
        self.pages_per_row = self.width // _PAGE
        self.active_rows = rows
        self.root_count = roots
        self.extents = [_Extent(slot) for slot in range(rows + roots)]
        self.dirty = [0] * len(self.extents)
        self.epoch = 0
        self.record = record
        self.plane = len(self.extents) * self.width * _KV_HEADS * _DIM
        shape = (2, len(self.extents), self.width, _KV_HEADS, _DIM)
        self.layers = tuple(torch.zeros(shape, dtype=torch.bfloat16, device=device)
                            for device in self.layer_devices)
        self.bytes_per_column = 2 * _KV_HEADS * _DIM * 2 * len(self.layers)
        self.record("model", "rollout.direct_kv_initialize",
                    details={"kv_zero_bytes":len(self.extents) * self.width * self.bytes_per_column,
                             "layer_counts":{str(device):self.layer_devices.count(device) for device in self.devices},
                             "time_basis":"included_in_kv_allocation"})
        self.device_buffers = {}
        for device in self.devices:
            with torch.cuda.device(device):
                self.device_buffers[device] = _DecodeBuffers(device, rows, self.width, self.pages_per_row,
                                                               head.weight.shape[0])
        self.buffers = self.device_buffers[self.device]

    def roots(self, count):
        first = self.active_rows if self.root_count else 0
        roots = [extent for extent in self.extents[first:] if not extent.owners][:count]
        if len(roots) != count:
            raise RuntimeError("The actor has exhausted its reserved immutable root runs")
        self._move([(extent.slot, extent.slot, (), 0, self.dirty[extent.slot]) for extent in roots
                    if self.dirty[extent.slot]], "root_reuse")
        return DirectRows(self, [_Row(extent) for extent in roots])

    def _move(self, moves, reason):
        if not moves:
            return
        started = time.monotonic()
        spans = max(len(spans) for _, _, spans, _, _ in moves)
        metadata = []
        read_width = write_width = copied = cleared = 0
        for source, destination, parts, zero_start, zero_stop in moves:
            runs, width = [], 0
            for start, stop in parts:
                runs.extend((start, width, width + stop - start))
                width += stop - start
            read_width = max(read_width, width)
            write_width = max(write_width, width, zero_stop)
            copied += width
            cleared += max(0, zero_stop - zero_start)
            metadata.append([source, destination, width, zero_start, zero_stop,
                             *runs, *([0] * (3 * (spans - len(parts))))])
        meta = {device:torch.tensor(metadata, dtype=torch.int32, device=device)
                for device in set(self.layer_devices)}
        for layer in self.layers:
            buffers = self.device_buffers[layer.device]
            with torch.cuda.device(layer.device):
                if read_width:
                    _stage_runs[(triton.cdiv(read_width, 8), len(moves))](
                        layer, meta[layer.device], buffers.keys, buffers.values,
                        self.plane, self.width, spans, 8, _DIM, num_warps=4)
                if write_width:
                    _commit_runs[(triton.cdiv(write_width, 8), len(moves))](
                        layer, meta[layer.device], buffers.keys, buffers.values,
                        self.plane, self.width, spans, 8, _DIM, num_warps=4)
        for destination, width, zero_start, zero_stop in (row[1:5] for row in metadata):
            self.dirty[destination] = max(width, zero_start)
        self.record("model", "rollout.direct_kv_" + reason, seconds=time.monotonic() - started,
                    details={"kv_copied_bytes":copied * self.bytes_per_column,
                             "kv_zero_bytes":cleared * self.bytes_per_column, "kv_layout_rows":len(moves),
                             "time_basis":"host_dispatch"})

    def prepare(self, rows, *, contiguous=False, compact=None, reason):
        # Root views share pages until their first mutable write. Mutable aliases
        # may share a common prefix when the writer owns the longest view.
        desired, replacements, specs = [], {}, {}
        for index, row in enumerate(rows):
            extent = row.extent
            parts = compact[index] if compact is not None else None
            if (extent.slot >= self.active_rows or extent in desired or extent.width > row.width
                    or (parts is not None and len(extent.owners) > 1)):
                extent = _Extent(-1)
                replacements[row] = extent
            desired.append(extent)
            if parts is not None or extent is not row.extent:
                specs[extent] = row.extent.slot, parts if parts is not None else ((0, row.width),)

        departed = Counter(row.extent for row in replacements)
        kept = [extent for extent in self.extents[:self.active_rows]
                if len(extent.owners) > departed[extent]]
        placed = {extent: slot for slot, extent in enumerate(desired)} if contiguous else {}
        for extent in kept:
            if extent not in placed and (not contiguous or extent.slot >= len(desired)):
                placed[extent] = extent.slot
        waiting = [extent for extent in (*desired, *kept) if extent not in placed]
        used = set(placed.values())
        available = [slot for slot in range(self.active_rows) if slot not in used]
        if len(waiting) > len(available):
            raise RuntimeError("The actor has more live independent histories than active page runs")
        placed.update(zip(waiting, available))

        moves = []
        for extent, destination in placed.items():
            source, parts = specs.get(extent, (extent.slot, None))
            if parts is None and source != destination:
                parts = ((0, extent.width),)
            if parts is not None:
                width = sum(stop - start for start, stop in parts)
                moves.append((source, destination, parts, width, self.dirty[destination]))
            elif self.dirty[destination] > extent.width:
                moves.append((source, destination, (), extent.width, self.dirty[destination]))
        self._move(moves, reason)

        for row, extent in replacements.items():
            row.relocate(extent)
        live_slots = set(placed.values())
        for extent, destination in placed.items():
            extent.slot = destination
            self.extents[destination] = extent
        for slot in range(self.active_rows):
            if slot not in live_slots:
                self.extents[slot] = _Extent(slot)
        if moves or replacements:
            self.epoch += 1

    def finish_append(self, rows, physical, lengths):
        for row, added in zip(rows, lengths, strict=True):
            if added:
                row.width = physical + added
                self.dirty[row.extent.slot] = max(self.dirty[row.extent.slot], row.width)


class DirectRows:
    """Cache row views preserve history identity while their backing runs move."""
    def __init__(self, pool, rows):
        self.pool, self.rows = pool, rows
        self.epoch = -1

    def close(self):
        if self.rows:
            for row in self.rows:
                row.release()
            self.rows.clear()
            self.pool.epoch += 1

    def __del__(self):
        self.close()

    def copy_rows(self, indices, **_):
        selected = [self.rows[i] for i in indices.tolist()]
        self.pool.epoch += 1
        return DirectRows(self.pool, [_Row(row.extent, row.width) for row in selected])

    def batch_select_indices(self, indices, **_):
        selected = self.copy_rows(indices)
        self.close()
        self.rows, selected.rows = selected.rows, []

    def split_owned(self, row_groups, **_):
        kept = {i for rows in row_groups for i in rows}
        outputs = [DirectRows(self.pool, [self.rows[i] for i in rows]) for rows in row_groups]
        for i, row in enumerate(self.rows):
            if i not in kept:
                row.release()
        self.rows = []
        self.pool.epoch += 1
        return outputs

    @classmethod
    def merge_owned(cls, caches, *, row_groups=None):
        pool = caches[0].pool
        if any(cache.pool is not pool for cache in caches):
            raise ValueError("Direct page runs must belong to the same actor session")
        rows = [row for cache in caches for row in cache.rows]
        if row_groups is not None:
            ordered = [None] * len(rows)
            for row, dest in zip(rows, (i for group in row_groups for i in group), strict=True):
                ordered[dest] = row
            rows = ordered
        for cache in caches:
            cache.rows = []
        pool.epoch += 1
        return cls(pool, rows)

    def prepare_sampled(self):
        if self.epoch != self.pool.epoch:
            self.pool.prepare(self.rows, contiguous=True, reason="cohort_layout")
            self.epoch = self.pool.epoch

    def compact_physical(self, spans, lengths):
        changed = [(row, parts, length) for row, parts, length in zip(self.rows, spans, lengths, strict=True)
                   if parts and parts != ((0, length),)]
        if changed:
            rows, parts, counts = zip(*changed, strict=True)
            self.pool.prepare(rows, compact=parts, reason="compact")
            for row, count in zip(rows, counts, strict=True):
                row.width = count


class EagerDirectAppend(Cache):
    """Stock prefill/feedback SDPA over exact physically padded columns."""
    def __init__(self, cache, physical, lengths, width):
        super().__init__(layers=[])
        self.pool, self.past = cache.pool, physical
        self.width = physical + width
        if self.width > self.pool.width:
            raise ValueError("Direct KV append exceeds the physical run capacity")
        if physical:
            self.pool.prepare(cache.rows, reason="eager_ownership")
        slots = tuple(row.extent.slot for row in cache.rows)
        self.batch, self.gather = len(slots), bool(physical)
        metadata = tuple(zip(slots, lengths, strict=True))
        self.rows = {device:torch.tensor(metadata, dtype=torch.int32, device=device)
                     for device in set(self.pool.layer_devices)}
        cache.pool.finish_append(cache.rows, physical, lengths)
        self.pool.record("model", "rollout.direct_kv_eager_append",
                         details={"kv_direct_write_bytes":self.batch * width * self.pool.bytes_per_column,
                                  "kv_copied_bytes":self.batch * self.width * self.pool.bytes_per_column
                                                    if self.gather else 0,
                                  "kv_layout_rows":self.batch,"time_basis":"included_in_actor_forward"})

    def get_seq_length(self, layer_idx=0):
        return self.past

    def get_mask_sizes(self, cache_position, layer_idx):
        return self.past + cache_position.shape[0], 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        batch, heads, width, dim = key_states.shape
        layer = self.pool.layers[layer_idx]
        device = layer.device
        buffers = self.pool.device_buffers[device]
        if self.gather:
            keys, values = (buf[:batch * self.width].view(batch, heads, self.width, dim)
                            for buf in (buffers.keys, buffers.values))
        else:
            keys, values = key_states, value_states
        columns = self.width if self.gather else width
        with torch.cuda.device(device):
            _append_extent[(triton.cdiv(columns, 16), batch * heads)](
                key_states, value_states, layer, self.rows[device], keys, values,
                self.past, width, *key_states.stride()[:3], *value_states.stride()[:3],
                self.pool.plane, self.pool.width, 16, _DIM, self.gather, num_warps=4)
        return keys, values


class _DirectGraph(_DecodeGraph):
    def __init__(self, pool, buffers, batch, capacity, all_keys_valid):
        super().__init__(pool, buffers, batch, capacity, all_keys_valid)
        self.column = buffers.table[0, :1]

    def load_inputs(self, ids, positions, table, mask):
        self.ids.copy_(ids)
        self.positions.copy_(positions)
        width = mask.shape[1]
        self.column.fill_(width - 1)
        self.valid[:, :width].copy_(mask)
        self.valid[:, width:].zero_()

    def attention(self, layer_idx, query, key, value, scale):
        pool, batch, capacity = self.pool, self.batch, self.capacity
        layer = pool.layers[layer_idx]
        _append_token[(batch * _KV_HEADS,)](
            key, value, layer, self.column, *key.stride()[:2], *value.stride()[:2],
            pool.plane, pool.width, _DIM, num_warps=4)
        keys, values = layer[0, :batch], layer[1, :batch]
        if self.all_keys_valid:
            output = torch.ops.aten._flash_attention_forward(
                query.squeeze(2), keys.flatten(0, 1), values.flatten(0, 1),
                self.cu_q, self.cu_k, 1, capacity, 0., False, False, scale=scale,
                seqused_k=self.seqused_k)[0]
        else:
            grouped_query = query.squeeze(2).unflatten(1, (_KV_HEADS, _GROUP))
            output = torch.ops.aten._scaled_dot_product_efficient_attention(
                grouped_query, keys[:, :capacity].transpose(1, 2), values[:, :capacity].transpose(1, 2),
                self.bias, False, 0., False, scale=scale)[0]
        return output.reshape(batch, 1, _KV_HEADS * _GROUP, _DIM)

    def _forward(self, model):
        if self.all_keys_valid:
            torch.mul(self.cu_q, self.pool.width, out=self.cu_k)
            self.seqused_k.copy_(self.positions[:, 0] + 1)
        else:
            self.bias_values.copy_(torch.where(self.valid, 0., -torch.inf))
        output = model.model(input_ids=self.ids, position_ids=self.positions,
                             attention_mask={"full_attention": None}, use_cache=False, fgpo_paged_graph=self)
        self.logits.copy_(model.get_output_embeddings()(output.last_hidden_state[:, 0]))
        return self.logits


class DirectGraphSession:
    def __init__(self, model, row_count, root_count, context, record, *, backbone=None):
        self.model = SimpleNamespace(model=model.model if backbone is None else backbone,
                                     get_output_embeddings=model.get_output_embeddings)
        self.pool = _DirectPool(self.model, row_count, root_count, context, record)
        self.stream = torch.cuda.Stream(device=self.pool.device)
        self.shards = None
        if len(self.pool.devices) > 1:
            from .direct_page_sharded import ShardedActor

            self.shards = ShardedActor(self.model, self.pool)
        self.graphs = {}

    def empty(self, count):
        return self.pool.roots(count)

    def decode(self, ids, positions, table, mask, all_keys_valid):
        tile = _PAGE if all_keys_valid else _MASKED_BUCKET
        capacity = triton.cdiv(mask.shape[1], tile) * tile
        if capacity > self.pool.width:
            raise ValueError("Direct KV decode exceeds the physical run capacity")
        shape = ids.shape[0], capacity, all_keys_valid
        with torch.cuda.device(self.pool.device):
            graph = self.graphs.pop(shape, None)
            if graph is None:
                if len(self.graphs) == _MAX_LIVE_GRAPHS:
                    self._retire_graph_family()
                graph = (_DirectGraph(self.pool, self.pool.buffers, *shape) if self.shards is None
                         else self.shards.build(*shape))
            logits, setup = graph.run(self.model, ids, positions, table, mask, self.stream)
            self.graphs[shape] = graph
        return logits, capacity, setup

    def close(self):
        self._retire_graph_family()

    def _retire_graph_family(self):
        if not self.graphs:
            return
        # Reset can free the cuBLAS workspace of the capture stream. Every
        # captured shape on that stream must retire before any further replay.
        for device in self.pool.devices:
            torch.cuda.synchronize(device)
        retired = tuple(self.graphs.values())
        self.graphs.clear()
        for graph in retired:
            graph.reset()
