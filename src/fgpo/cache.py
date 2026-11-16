"""Reusable KV storage implementing the Transformers Cache/Layer API.

Attention receives only the occupied prefix, not the allocation capacity.  The
storage grows geometrically and retains spare capacity after crop/compaction.
Padding positions remain explicit in the caller's attention mask; this cache
does not equate physical batch columns with logical RoPE positions.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicLayer


class ArenaLayer(DynamicLayer):
    """Inference-only append storage with amortized allocation and exact crop."""

    is_compileable = False
    is_sliding = False

    def __init__(self, *, initial_capacity=1024, growth_limit=32768):
        super().__init__()
        self.initial_capacity = int(initial_capacity)
        self.growth_limit = int(growth_limit)
        if self.initial_capacity < 1 or self.growth_limit < 1:
            raise ValueError("KV capacities must be positive")
        self._capacity = 0
        self._length = 0
        self._key_storage = self._value_storage = None

    def _refresh_views(self):
        self.keys = self._key_storage[..., :self._length, :]
        self.values = self._value_storage[..., :self._length, :]

    def lazy_initialization(self, key_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.is_initialized = True
        self._reserve(max(1, key_states.shape[-2]))

    def _reserve(self, required):
        if required <= self._capacity:
            return
        # The limit caps geometric over-allocation, not physical padding.  A
        # ragged batch can have more physical columns than logical positions;
        # the policy separately enforces each row's logical context capacity.
        grown = (self._capacity + self.initial_capacity if required > self.growth_limit else
                 min(self.growth_limit, max(self.initial_capacity, 2 * self._capacity)))
        capacity = max(int(required), grown)
        shape = (self.max_batch_size, self.num_heads, capacity, self.head_dim)
        keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        values = torch.empty_like(keys)
        if self._length:
            keys[..., :self._length, :].copy_(self.keys)
            values[..., :self._length, :].copy_(self.values)
        self._key_storage, self._value_storage = keys, values
        self._capacity = capacity
        self._refresh_views()

    def update(self, key_states, value_states, cache_kwargs=None):
        if torch.is_grad_enabled():
            raise RuntimeError("Arena KV updates require no_grad inference")
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        end = self._length + key_states.shape[-2]
        self._reserve(end)
        self._key_storage[..., self._length:end, :].copy_(key_states)
        self._value_storage[..., self._length:end, :].copy_(value_states)
        self._length = end
        self._refresh_views()
        return self.keys, self.values

    def get_seq_length(self):
        return self._length

    def get_max_cache_shape(self):
        # Mask length follows occupied tokens, as for DynamicLayer.
        return -1

    def crop(self, max_length):
        length = self._length + max_length if max_length < 0 else max_length
        self._length = min(self._length, max(0, int(length)))
        if self.is_initialized:
            self._refresh_views()

    def reset(self):
        self._length = 0
        if self.is_initialized:
            self._refresh_views()

    def copy_rows(self, indices):
        other = ArenaLayer(initial_capacity=self.initial_capacity, growth_limit=self.growth_limit)
        if not self.is_initialized:
            return other
        indices = indices.to(device=self.device, dtype=torch.long)
        other.dtype, other.device = self.dtype, self.device
        other.max_batch_size, other.num_heads, other.head_dim = indices.numel(), self.num_heads, self.head_dim
        other.is_initialized = True
        other._reserve(max(1, self._length))
        other._length = self._length
        other._refresh_views()
        if self._length:
            torch.index_select(self.keys, 0, indices, out=other.keys)
            torch.index_select(self.values, 0, indices, out=other.values)
        return other

    def batch_select_indices(self, indices):
        selected = self.copy_rows(indices)
        self.__dict__.update(selected.__dict__)

    def reorder_cache(self, beam_idx):
        self.batch_select_indices(beam_idx)

    def batch_repeat_interleave(self, repeats):
        if self.is_initialized:
            indices = torch.arange(self.max_batch_size, device=self.device).repeat_interleave(repeats)
            self.batch_select_indices(indices)

    def compact(self, positions, lengths):
        """Remove masked columns while preserving already-computed K/V values."""
        longest = max(lengths, default=0)
        for row, (indices, count) in enumerate(zip(positions, lengths, strict=True)):
            if count:
                # index_select creates a distinct row buffer, so an in-place
                # leftward copy cannot overwrite a later source position.
                self._key_storage[row, :, :count].copy_(self.keys[row].index_select(1, indices))
                self._value_storage[row, :, :count].copy_(self.values[row].index_select(1, indices))
            self._key_storage[row, :, count:longest].zero_()
            self._value_storage[row, :, count:longest].zero_()
        self._length = longest
        self._refresh_views()


class ArenaCache(Cache):
    """Transformers-compatible cache; each layer allocates on its own device."""

    def __init__(self, num_layers=None, *, initial_capacity=1024, growth_limit=32768, layers=None):
        if layers is None:
            if num_layers is None:
                raise ValueError("ArenaCache needs a layer count")
            layers = [ArenaLayer(initial_capacity=initial_capacity, growth_limit=growth_limit)
                      for _ in range(int(num_layers))]
        super().__init__(layers=layers)

    def copy_rows(self, indices):
        return ArenaCache(layers=[layer.copy_rows(indices) for layer in self.layers])

    def compact(self, positions, lengths):
        device_positions = {}
        for layer in self.layers:
            device = layer.device
            if device not in device_positions:
                device_positions[device] = [indices.to(device) for indices in positions]
            layer.compact(device_positions[device], lengths)
