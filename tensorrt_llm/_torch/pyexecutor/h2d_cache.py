# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skip host-to-device metadata copies whose values the destination already holds.

Steady decode re-uploads many small per-step metadata tensors whose values do not change (slot
indices, query lengths, prompt lengths, page tables between page boundaries). Each upload is a
separate copy on the compute stream, serialized between the decode graphs. An ``H2DSkipCache``
keeps one host shadow per destination buffer (the last values written through it, and the view
they were written to) and skips the copy when the same view receives the same values again.

Each engine that armed ``RuntimeHooks.skip_unchanged_h2d`` owns one cache and makes it the active
one for its forward (``active``); ``copy_h2d_if_changed`` acts on the calling context's active
cache and, with none, is ``dst.copy_(src, non_blocking=True)``.

Route a copy through it only for a destination that no other code writes (no device kernel, no
other copy path, no other engine, no graph replay): the shadow describes the device contents only
under that invariant. CUDA graph metadata pool-shared destinations use ``copy_h2d_pool_shared`` /
``fill_tail_pool_shared``, which never skip. A write through a different view of the same buffer
replaces the shadow, so overlapping views never skip on stale state.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator, Optional

import torch

from . import step_inputs

__all__ = [
    "H2DSkipCache",
    "SKIP_POOL_SHARED_UPLOADS",
    "active",
    "copy_h2d_if_changed",
    "copy_h2d_pool_shared",
    "enabled",
    "fill_tail_if_changed",
    "fill_tail_pool_shared",
]


class _HostShadowBuffer:
    """Host snapshot of the last values copied into a device buffer view."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor
        self._last_host_values: Optional[torch.Tensor] = None
        self._last_host_shape: Optional[torch.Size] = None

    @property
    def host_values(self) -> Optional[torch.Tensor]:
        """Last recorded host values reshaped to the submitted view; callers must not mutate."""
        if self._last_host_values is None or self._last_host_shape is None:
            return None
        count = self._last_host_shape.numel()
        return self._last_host_values[:count].view(self._last_host_shape)

    def invalidate(self) -> None:
        """Forget a pending or externally overwritten copy."""
        self._last_host_values = None
        self._last_host_shape = None

    def matches(self, host: torch.Tensor, *, exact: bool = False) -> bool:
        """Whether the recorded host inputs match ``host``."""
        last = self._last_host_values
        flat = host.reshape(-1)
        return (
            last is not None
            and last.dtype == host.dtype
            and (not exact or self._last_host_shape == host.shape)
            and last.numel() >= flat.numel()
            and torch.equal(last[: flat.numel()], flat)
        )

    def record(self, host: torch.Tensor) -> None:
        """Remember an immutable host snapshot of a submitted copy."""
        flat = host.reshape(-1)
        last = self._last_host_values
        if last is None or last.dtype != host.dtype or last.numel() <= flat.numel():
            self._last_host_values = flat.clone()
        else:
            last = last.clone()
            last[: flat.numel()] = flat
            self._last_host_values = last
        self._last_host_shape = host.shape


class H2DSkipCache:
    """One engine's host shadows of its skip-routed destinations (see the module docstring).

    ``changed_span``: a packed copy into a 2-D view with contiguous rows whose shadow covers the
    same rows (same first element and row stride) uploads, per row, only the span from its first
    to its last changed column, plus any columns the shadow does not cover.
    """

    # A changed-span upload is used only while it moves at most this fraction of the view's bytes.
    CHANGED_SPAN_MAX_FRACTION = 0.5

    def __init__(self, changed_span: bool = False) -> None:
        self.changed_span = changed_span
        # Device storage address -> (storage kept alive so the address is not reused; view key;
        # host shadow). Keyed by storage rather than by view base because tensors created under
        # inference mode do not record their base.
        self._shadow: dict[int, tuple[object, tuple, _HostShadowBuffer]] = {}
        # Device storage address -> (storage kept alive; view key; (first row, value) of the
        # view's last tail fill).
        self._tails: dict[int, tuple[object, tuple, tuple[int, int]]] = {}

    def clear(self) -> None:
        """Forget every shadow so the next copy to each destination is issued."""
        self._shadow.clear()
        self._tails.clear()

    def copy(self, dst: torch.Tensor, src: torch.Tensor, pack: bool) -> bool:
        storage = dst.untyped_storage()
        addr = storage.data_ptr()
        if src.is_cuda or not dst.is_cuda or torch.cuda.is_current_stream_capturing():
            self._shadow.pop(addr, None)
            dst.copy_(src, non_blocking=True)
            return True
        key = (dst.storage_offset(), tuple(dst.shape), tuple(dst.stride()), dst.dtype)
        entry = self._shadow.get(addr)
        if entry is not None and entry[1] == key and entry[2].matches(src, exact=True):
            return False
        old = entry[2].host_values if entry is not None else None
        buffer = entry[2] if entry is not None and entry[1] == key else _HostShadowBuffer(dst)
        buffer.invalidate()
        self._shadow.pop(addr, None)
        if not (
            pack
            and self.changed_span
            and old is not None
            and self._copy_changed_span(dst, src, (entry[1], old))
        ):
            if pack:
                step_inputs.copy_(dst, src)
            else:
                dst.copy_(src, non_blocking=True)
        buffer.record(src)
        self._shadow[addr] = (storage, key, buffer)
        return True

    def _copy_changed_span(self, dst: torch.Tensor, src: torch.Tensor, shadow: tuple) -> bool:
        """Stage only the changed span of each row of ``dst``; False if not applicable."""
        (offset, shape, stride, dtype), old = shadow
        if (
            dst.dim() != 2
            or dst.stride(1) != 1
            or offset != dst.storage_offset()
            or len(shape) != 2
            or shape[0] != dst.shape[0]
            or stride != tuple(dst.stride())
            or dtype != dst.dtype
            or old.dtype != src.dtype
            or tuple(src.shape) != tuple(dst.shape)
            or not step_inputs.is_open()
        ):
            return False
        cols = dst.shape[1]
        known = min(shape[1], cols)
        changed = src[:, :known] != old[:, :known]
        if cols > known:
            # Columns the shadow does not cover hold unknown device values.
            changed = torch.cat([changed, changed.new_ones(dst.shape[0], cols - known)], dim=1)
        rows = changed.any(dim=1).nonzero().flatten().tolist()
        spans = []
        for row in rows:
            columns = changed[row].nonzero().flatten()
            spans.append((row, int(columns[0]), int(columns[-1]) + 1))
        if sum(hi - lo for _, lo, hi in spans) > self.CHANGED_SPAN_MAX_FRACTION * dst.numel():
            return False
        for row, lo, hi in spans:
            step_inputs.copy_(dst[row, lo:hi], src[row, lo:hi])
        return True

    def forget(self, dst: torch.Tensor) -> None:
        """Drop the shadow and tail record of ``dst``'s storage."""
        addr = dst.untyped_storage().data_ptr()
        self._shadow.pop(addr, None)
        self._tails.pop(addr, None)

    def fill_tail(self, buf: torch.Tensor, start: int, value: int) -> None:
        storage = buf.untyped_storage()
        addr = storage.data_ptr()
        if start >= buf.shape[0]:
            # The live rows cover the whole view: the previous tail fill is overwritten.
            self._tails.pop(addr, None)
            return
        if torch.cuda.is_current_stream_capturing():
            self._tails.pop(addr, None)
            buf[start:].fill_(value)
            return
        key = (buf.storage_offset(), tuple(buf.shape), tuple(buf.stride()), buf.dtype)
        entry = self._tails.get(addr)
        if entry is not None and entry[1] == key and entry[2] == (start, value):
            return
        buf[start:].fill_(value)
        self._tails[addr] = (storage, key, (start, value))


_active: ContextVar[Optional[H2DSkipCache]] = ContextVar("trtllm_h2d_skip_cache", default=None)

# Kill switch for A/B only: True routes pool-shared metadata destinations through the skip cache;
# False keeps them always written.
SKIP_POOL_SHARED_UPLOADS = False


@contextlib.contextmanager
def active(cache: Optional[H2DSkipCache]) -> Iterator[None]:
    """Make ``cache`` the calling context's cache (None: no skipping) for the duration."""
    token = _active.set(cache)
    try:
        yield
    finally:
        _active.reset(token)


def enabled() -> bool:
    """Whether the calling context skips unchanged copies."""
    return _active.get() is not None


def copy_h2d_if_changed(dst: torch.Tensor, src: torch.Tensor, pack: bool = False) -> bool:
    """``dst.copy_(src, non_blocking=True)`` unless the active cache knows ``dst`` holds ``src``."""
    cache = _active.get()
    if cache is None:
        dst.copy_(src, non_blocking=True)
        return True
    return cache.copy(dst, src, pack)


def fill_tail_if_changed(buf: torch.Tensor, start: int, value: int) -> None:
    """``buf[start:].fill_(value)`` unless the active cache's last tail fill of ``buf`` matched."""
    cache = _active.get()
    if cache is None:
        if start < buf.shape[0]:
            buf[start:].fill_(value)
        return
    cache.fill_tail(buf, start, value)


def copy_h2d_pool_shared(dst: torch.Tensor, src: torch.Tensor, pack: bool = False) -> bool:
    """``copy_h2d_if_changed`` for a destination in a CUDA graph metadata buffer pool: always
    issued."""
    cache = _active.get()
    if cache is None:
        if pack:
            step_inputs.copy_metadata_(dst, src)
        else:
            dst.copy_(src, non_blocking=True)
        return True
    if SKIP_POOL_SHARED_UPLOADS:
        return cache.copy(dst, src, pack)
    cache.forget(dst)
    if pack:
        step_inputs.copy_(dst, src)
    else:
        dst.copy_(src, non_blocking=True)
    return True


def fill_tail_pool_shared(buf: torch.Tensor, start: int, value: int) -> None:
    """``fill_tail_if_changed`` for a buffer in a CUDA graph metadata buffer pool: always issued."""
    cache = _active.get()
    if cache is not None and SKIP_POOL_SHARED_UPLOADS:
        cache.fill_tail(buf, start, value)
        return
    if cache is not None:
        cache.forget(buf)
    if start < buf.shape[0]:
        buf[start:].fill_(value)
