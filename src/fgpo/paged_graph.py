"""Persistent BF16 KV pages and captured Qwen actor decode for AppWorld."""
from __future__ import annotations

import time

import torch
import triton
import triton.language as tl
from transformers.cache_utils import Cache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


_PAGE = 128
_MASKED_BUCKET = 512
_KV_HEADS = 4
_GROUP = 7
_DIM = 128
_NAME = "fgpo_paged_graph"
_MAX_LIVE_GRAPHS = 64


@triton.jit(do_not_specialize=["Past", "Width", "K0", "K1", "K2", "V0", "V1", "V2"])
def _eager_kv(K, V, Pool, Table, Layout, DenseK, DenseV, Past, Width, K0, K1, K2, V0, V1, V2,
              PAGES: tl.constexpr, SPANS: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr,
              TILE: tl.constexpr, GATHER: tl.constexpr):
    group = tl.program_id(1)
    row, head = group // 4, group % 4
    cols = tl.program_id(0) * TILE + tl.arange(0, TILE)
    d = tl.arange(0, D)
    meta = Layout + row * (2 + 3 * SPANS)
    append = cols - Past
    new = (append >= 0) & (append < Width)
    old_length, add_length = tl.load(meta), tl.load(meta + 1)
    k = tl.load(K + row * K0 + head * K1 + append[:, None] * K2 + d[None, :], new[:, None], other=0)
    v = tl.load(V + row * V0 + head * V1 + append[:, None] * V2 + d[None, :], new[:, None], other=0)
    write = new & (append < add_length)
    logical = old_length + append
    page = tl.load(Table + row * PAGES + logical // PAGE, write, other=0)
    offsets = ((page * 8 + head) * PAGE + logical % PAGE)[:, None] * D + d[None, :]
    tl.store(Pool + offsets, k, write[:, None])
    tl.store(Pool + offsets + 4 * PAGE * D, v, write[:, None])
    if GATHER:
        old_logical = tl.full((TILE,), -1, tl.int32)
        for span in tl.static_range(SPANS):
            start = tl.load(meta + 2 + 3 * span)
            stop = tl.load(meta + 3 + 3 * span)
            first = tl.load(meta + 4 + 3 * span)
            old_logical = tl.where((cols >= start) & (cols < stop), first + cols - start, old_logical)
        old = old_logical >= 0
        old_page = tl.load(Table + row * PAGES + old_logical // PAGE, old, other=0)
        old_offsets = ((old_page * 8 + head) * PAGE + old_logical % PAGE)[:, None] * D + d[None, :]
        old_k = tl.load(Pool + old_offsets, old[:, None], other=0)
        old_v = tl.load(Pool + old_offsets + 4 * PAGE * D, old[:, None], other=0)
        width = Past + Width
        dense = (group * width + cols[:, None]) * D + d[None, :]
        tl.store(DenseK + dense, tl.where(new[:, None], k, old_k), cols[:, None] < width)
        tl.store(DenseV + dense, tl.where(new[:, None], v, old_v), cols[:, None] < width)


@triton.jit
def _write_token(K, V, Pool, Table, Positions, K0: tl.constexpr, K1: tl.constexpr,
                 V0: tl.constexpr, V1: tl.constexpr, PAGES: tl.constexpr,
                 PAGE: tl.constexpr, D: tl.constexpr):
    group = tl.program_id(0)
    row, head = group // 4, group % 4
    position = tl.load(Positions + row).to(tl.int32)
    page = tl.load(Table + row * PAGES + position // PAGE)
    d = tl.arange(0, D)
    offset = ((page * 8 + head) * PAGE + position % PAGE) * D + d
    tl.store(Pool + offset, tl.load(K + row * K0 + head * K1 + d))
    tl.store(Pool + offset + 4 * PAGE * D, tl.load(V + row * V0 + head * V1 + d))


@triton.jit
def _gather_decode(Pool, Table, Logical, K, V, CAPACITY: tl.constexpr,
                   PAGES: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr):
    group = tl.program_id(1)
    row, head = group // 4, group % 4
    cols = tl.program_id(0) * TILE + tl.arange(0, TILE)
    d = tl.arange(0, D)
    logical = tl.load(Logical + row * CAPACITY + cols)
    valid = logical >= 0
    page = tl.load(Table + row * PAGES + logical // PAGE, valid, other=0)
    offsets = ((page * 8 + head) * PAGE + logical % PAGE)[:, None] * D + d[None, :]
    key = tl.load(Pool + offsets, valid[:, None], other=0)
    value = tl.load(Pool + offsets + 4 * PAGE * D, valid[:, None], other=0)
    dense = ((row * CAPACITY + cols[:, None]) * 4 + head) * D + d[None, :]
    tl.store(K + dense, key)
    tl.store(V + dense, value)


def install_paged_graph(model, *, sharded=False):
    decoder, weight = model.model, model.get_input_embeddings().weight
    config = decoder.config
    if (config.model_type != "qwen2" or config.num_hidden_layers != 28 or config.num_attention_heads != 28
            or config.num_key_value_heads != _KV_HEADS or decoder.layers[0].self_attn.head_dim != _DIM
            or decoder.has_sliding_layers or decoder.rotary_emb.rope_type != "default"
            or weight.dtype != torch.bfloat16 or weight.device.type != "cuda"
            or torch.cuda.get_device_capability(weight.device) < (8, 0)
            or any(p.dtype != weight.dtype or p.device.type != "cuda"
                   or (not sharded and p.device != weight.device) for p in model.parameters())):
        raise ValueError("Paged Graph decode requires CUDA BF16 Qwen2.5-7B with default RoPE")
    ALL_ATTENTION_FUNCTIONS.register(_NAME, _attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(_NAME, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    decoder.set_attn_implementation(_NAME)


def _attention(module, query, key, value, attention_mask, *, fgpo_paged_graph=None,
               dropout=0., scaling=None, **kwargs):
    if fgpo_paged_graph is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    return fgpo_paged_graph.attention(module.layer_idx, query, key, value, scaling), None


class _PagePool:
    def __init__(self, model, row_count, root_count, context):
        self.device = model.get_input_embeddings().weight.device
        self.pages_per_row = triton.cdiv(context, _PAGE)
        capacity = (row_count + root_count) * self.pages_per_row
        shape = (capacity, 2, _KV_HEADS, _PAGE, _DIM)
        self.layers = tuple(torch.empty(shape, dtype=torch.bfloat16, device=self.device)
                            for _ in model.model.layers)
        self.refs = [0] * capacity
        self.free = list(range(capacity - 1, -1, -1))

    def allocate(self):
        page = self.free.pop()
        self.refs[page] = 1
        return page

    def release(self, pages):
        for page in pages:
            self.refs[page] -= 1
            if not self.refs[page]:
                self.free.append(page)

    def own_tail(self, pages):
        page = pages[-1]
        if self.refs[page] > 1:
            replacement = self.allocate()
            for layer in self.layers:
                layer[replacement].copy_(layer[page])
            self.release((page,))
            pages[-1] = replacement
            return True
        return False


class PagedRows:
    """Own one page reference per logical row, including shared root pages."""
    def __init__(self, pool, rows):
        self.pool, self.rows = pool, rows
        self._table = None

    def close(self):
        for row in self.rows:
            self.pool.release(row)
        self.rows.clear()
        self._table = None

    def __del__(self):
        self.close()

    def copy_rows(self, indices, **_):
        rows = [self.rows[i].copy() for i in indices.tolist()]
        for row in rows:
            for page in row:
                self.pool.refs[page] += 1
        return PagedRows(self.pool, rows)

    def batch_select_indices(self, indices, **_):
        selected = self.copy_rows(indices)
        self.close()
        self.rows, selected.rows = selected.rows, []

    def split_owned(self, row_groups, **_):
        outputs = []
        for rows in row_groups:
            outputs.append(PagedRows(self.pool, [self.rows[i] for i in rows]))
            for i in rows:
                self.rows[i] = []
        self.close()
        return outputs

    @classmethod
    def merge_owned(cls, caches, *, row_groups=None):
        pool = caches[0].pool
        if any(cache.pool is not pool for cache in caches):
            raise ValueError("Paged rows must own pages from the same actor session")
        rows = [row for cache in caches for row in cache.rows]
        if row_groups is not None:
            ordered = [None] * len(rows)
            for row, dest in zip(rows, (i for group in row_groups for i in group), strict=True):
                ordered[dest] = row
            rows = ordered
        for cache in caches:
            cache.rows = []
            cache.close()
        return cls(pool, rows)

    def prepare(self, previous, totals):
        dirty = False
        for pages, old, new in zip(self.rows, previous, totals, strict=True):
            if old == new:
                continue
            if old % _PAGE:
                dirty |= self.pool.own_tail(pages)
            while len(pages) < triton.cdiv(new, _PAGE):
                pages.append(self.pool.allocate())
                dirty = True
        if self._table is None or dirty:
            size = self.pool.pages_per_row
            table = [row + [0] * (size - len(row)) for row in self.rows]
            self._table = torch.tensor(table, dtype=torch.int32, device=self.pool.device)
        return self._table


class EagerPagedAppend(Cache):
    """Native SDPA prefill/feedback: expose just the current layer's dense view."""
    def __init__(self, cache, physical, spans, previous, lengths, totals):
        super().__init__(layers=[])
        self.pool, self.past = cache.pool, physical
        self.table = cache.prepare(previous, totals)
        self.spans = max(map(len, spans), default=0)
        layout = []
        for row, old, new in zip(spans, previous, lengths, strict=True):
            metadata, logical = [old, new], 0
            for start, stop in row:
                metadata.extend((start, stop, logical))
                logical += stop - start
            layout.append(metadata + [0] * (3 * (self.spans - len(row))))
        self.layout = torch.tensor(layout, dtype=torch.int32, device=self.pool.device)

    def get_seq_length(self, layer_idx=0):
        return self.past

    def get_mask_sizes(self, cache_position, layer_idx):
        return self.past + cache_position.shape[0], 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        batch, heads, width, dim = key_states.shape
        full = self.past + width
        gather = self.past > 0
        keys = key_states.new_empty((batch, heads, full, dim)) if gather else key_states
        values = torch.empty_like(keys) if gather else value_states
        _eager_kv[(triton.cdiv(full, 16), batch * heads)](
            key_states, value_states, self.pool.layers[layer_idx], self.table, self.layout, keys, values,
            self.past, width, *key_states.stride()[:3], *value_states.stride()[:3],
            self.pool.pages_per_row, self.spans, _PAGE, _DIM, 16, gather, num_warps=4)
        return keys, values


class _DecodeBuffers:
    """Static inputs and outputs live outside the shared capture allocator."""
    def __init__(self, device, rows, width, pages_per_row, vocab):
        # Graphs run serially and depend only on this storage and the frozen actor.
        self.capture_pool = torch.cuda.graph_pool_handle()
        self.ids = torch.empty((rows, 1), dtype=torch.long, device=device)
        self.positions = torch.empty_like(self.ids)
        self.table = torch.empty((rows, pages_per_row), dtype=torch.int32, device=device)
        self.valid = torch.empty(rows * width, dtype=torch.bool, device=device)
        self.logical = torch.empty(rows * width, dtype=torch.int32, device=device)
        self.bias = torch.empty(rows * width, dtype=torch.bfloat16, device=device)
        self.seqused_k = torch.empty(rows, dtype=torch.int32, device=device)
        self.cu_q = torch.arange(rows + 1, dtype=torch.int32, device=device)
        self.cu_k = torch.empty_like(self.cu_q)
        self.logits = torch.empty((rows, vocab), dtype=torch.float32, device=device)
        shape = (rows * width, _KV_HEADS, _DIM)
        self.keys = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.values = torch.empty_like(self.keys)


class _DecodeGraph:
    def __init__(self, pool, buffers, batch, capacity, all_keys_valid):
        self.pool, self.batch, self.capacity = pool, batch, capacity
        self.all_keys_valid = all_keys_valid
        self.capture_pool = buffers.capture_pool
        self.ids = buffers.ids[:batch]
        self.positions = buffers.positions[:batch]
        self.table = buffers.table[:batch]
        self.logits = buffers.logits[:batch]
        self.valid, self.logical = (buffer[:batch * capacity].view(batch, capacity)
                                    for buffer in (buffers.valid, buffers.logical))
        self.keys, self.values = (buffer[:batch * capacity].view(batch, capacity, _KV_HEADS, _DIM)
                                  for buffer in (buffers.keys, buffers.values))
        if all_keys_valid:
            self.cu_q, self.cu_k = buffers.cu_q[:batch + 1], buffers.cu_k[:batch + 1]
            self.seqused_k = buffers.seqused_k[:batch]
        else:
            self.bias_values = buffers.bias[:batch * capacity].view(batch, capacity)
            self.bias = self.bias_values[:, None, None, :].expand(batch, _KV_HEADS, _GROUP, capacity)
        self.graph = None

    def attention(self, layer_idx, query, key, value, scale):
        batch, capacity = self.batch, self.capacity
        pool = self.pool.layers[layer_idx]
        _write_token[(batch * _KV_HEADS,)](
            key, value, pool, self.table, self.positions, *key.stride()[:2], *value.stride()[:2],
            self.pool.pages_per_row, _PAGE, _DIM, num_warps=4)
        _gather_decode[(capacity // 16, batch * _KV_HEADS)](
            pool, self.table, self.logical, self.keys, self.values,
            capacity, self.pool.pages_per_row, _PAGE, _DIM, 16, num_warps=4)
        if self.all_keys_valid:
            # The 128-key capacity preserves native FlashAttention's split count
            # while seqused_k supplies the current exact unmasked width.
            output = torch.ops.aten._flash_attention_forward(
                query.squeeze(2), self.keys.flatten(0, 1), self.values.flatten(0, 1),
                self.cu_q, self.cu_k, 1, capacity, 0., False, False, scale=scale,
                seqused_k=self.seqused_k)[0]
        else:
            # The seven independent queries share a KV head and physical bias.
            grouped_query = query.squeeze(2).unflatten(1, (_KV_HEADS, _GROUP))
            output = torch.ops.aten._scaled_dot_product_efficient_attention(
                grouped_query, self.keys.transpose(1, 2), self.values.transpose(1, 2),
                self.bias, False, 0., False, scale=scale)[0]
        return output.reshape(batch, 1, _KV_HEADS * _GROUP, _DIM)

    def _forward(self, model):
        self.logical.copy_(torch.where(self.valid, self.valid.cumsum(1, dtype=torch.int32) - 1, -1))
        if self.all_keys_valid:
            torch.mul(self.cu_q, self.capacity, out=self.cu_k)
            self.seqused_k.copy_(self.positions[:, 0] + 1)
        else:
            self.bias_values.copy_(torch.where(self.valid, 0., -torch.inf))
        output = model.model(input_ids=self.ids, position_ids=self.positions,
                             attention_mask={"full_attention": None}, use_cache=False, fgpo_paged_graph=self)
        self.logits.copy_(model.get_output_embeddings()(output.last_hidden_state[:, 0]))
        return self.logits

    def load_inputs(self, ids, positions, table, mask):
        self.ids.copy_(ids)
        self.positions.copy_(positions)
        width = mask.shape[1]
        self.valid[:, :width].copy_(mask)
        self.valid[:, width:].zero_()
        self.table.copy_(table)

    def run(self, model, ids, positions, table, mask, capture_stream):
        self.load_inputs(ids, positions, table, mask)
        setup = self.replay(model, capture_stream)
        # Other shapes can run next and write the same static output addresses.
        return self.logits.clone(), setup

    def replay(self, model, capture_stream):
        setup = None
        if self.graph is None:
            warmup_start = time.monotonic()
            capture_stream.wait_stream(torch.cuda.current_stream(self.ids.device))
            with torch.cuda.stream(capture_stream):
                # Warm the same signatures and initialize CUDA libraries before capture.
                self._forward(model)
                capture_stream.synchronize()
                capture_start = time.monotonic()
                # Capture cannot reclaim inactive blocks from startup's default pool.
                torch.cuda.empty_cache()
                graph = torch.cuda.CUDAGraph()
                graph.capture_begin(pool=self.capture_pool)
                try:
                    self._forward(model)
                finally:
                    graph.capture_end()
            self.graph = graph
            setup = (capture_start - warmup_start, time.monotonic() - capture_start)
        self.graph.replay()
        return setup

    def reset(self):
        if self.graph is not None:
            self.graph.reset()


class PagedGraphSession:
    """One frozen actor collection batch, with reused static addresses across turns."""
    def __init__(self, model, row_count, root_count, context, max_new_tokens):
        self.pool = _PagePool(model, row_count, root_count, context)
        width = triton.cdiv(context + max_new_tokens, _MASKED_BUCKET) * _MASKED_BUCKET
        vocab = model.get_output_embeddings().weight.shape[0]
        self.buffers = _DecodeBuffers(self.pool.device, row_count, width, self.pool.pages_per_row, vocab)
        self.model = model
        self.stream = torch.cuda.Stream(device=self.pool.device)
        self.graphs = {}

    def empty(self, count):
        return PagedRows(self.pool, [[] for _ in range(count)])

    def decode(self, ids, positions, table, mask, all_keys_valid):
        tile = _PAGE if all_keys_valid else _MASKED_BUCKET
        capacity = triton.cdiv(mask.shape[1], tile) * tile
        shape = ids.shape[0], capacity, all_keys_valid
        with torch.cuda.device(self.pool.device):
            graph = self.graphs.pop(shape, None)
            if graph is None:
                if len(self.graphs) == _MAX_LIVE_GRAPHS:
                    expired = self.graphs.pop(next(iter(self.graphs)))
                    expired.graph.reset()
                graph = _DecodeGraph(self.pool, self.buffers, *shape)
            logits, setup = graph.run(self.model, ids, positions, table, mask, self.stream)
            self.graphs[shape] = graph
        return logits, capacity, setup

    def close(self):
        for graph in self.graphs.values():
            if graph.graph is not None:
                graph.graph.reset()
        self.graphs.clear()
