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
import ctypes
from contextvars import ContextVar
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import torch

from . import step_inputs
from .numa_pinned import numa_pinned_empty


def _load_memcmp():
    try:
        libc = ctypes.CDLL(None)
        fn = libc.memcmp
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        fn.restype = ctypes.c_int
        return fn
    except Exception:
        return None


_MEMCMP = _load_memcmp()

__all__ = [
    "H2DSkipCache",
    "SKIP_POOL_SHARED_UPLOADS",
    "active",
    "copy_block_offsets_if_changed",
    "copy_h2d_if_changed",
    "copy_h2d_pool_shared",
    "enabled",
    "fill_tail_if_changed",
    "fill_tail_pool_shared",
    "incremental_block_offsets_enabled",
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
    # Maximum total block-offset entries patched incrementally before falling back to full copy.
    BLOCK_OFFSET_MAX_SPAN = 256

    def __init__(
        self,
        changed_span: bool = False,
        incremental_block_offsets: bool = False,
    ) -> None:
        self.changed_span = changed_span
        self.incremental_block_offsets = incremental_block_offsets
        # Device storage address -> (storage kept alive so the address is not reused; view key;
        # host shadow). Keyed by storage rather than by view base because tensors created under
        # inference mode do not record their base.
        self._shadow: dict[int, tuple[object, tuple, _HostShadowBuffer]] = {}
        # Device storage address -> (storage kept alive; view key; (first row, value) of the
        # view's last tail fill).
        self._tails: dict[int, tuple[object, tuple, tuple[int, int]]] = {}
        # Device storage address -> (storage kept alive; key; raw int32 snapshot [P, S, B]; tails).
        self._block_offsets: dict[int, tuple] = {}
        self._bo_stage_ring: list[torch.Tensor] = []
        self._bo_stage_pos: int = 0
        self._bo_k1 = torch.empty((1,), dtype=torch.int32)
        self._bo_v1 = torch.empty((1,), dtype=torch.int32)

    def clear(self) -> None:
        """Forget every shadow so the next copy to each destination is issued."""
        self._shadow.clear()
        self._tails.clear()
        self._block_offsets.clear()

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
        self._block_offsets.pop(addr, None)

    def _get_bo_stage(self, n_elems: int, device: torch.device) -> torch.Tensor:
        if not self._bo_stage_ring or self._bo_stage_ring[0].numel() < n_elems:
            cap = max(n_elems, self.BLOCK_OFFSET_MAX_SPAN * 2)
            self._bo_stage_ring = [
                numa_pinned_empty(cap, dtype=torch.int32, device=device) for _ in range(4)
            ]
            self._bo_stage_pos = 0
        buf = self._bo_stage_ring[self._bo_stage_pos]
        self._bo_stage_pos = (self._bo_stage_pos + 1) % len(self._bo_stage_ring)
        return buf

    def copy_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        host_src: torch.Tensor,
        copy_idx_list: Sequence[int],
        index_scales: Sequence[int],
        kv_offsets: Sequence[int],
        bad_page_index: int,
        fallback_fn: Callable[[], None],
        *,
        manager_id: object,
        stream: Optional[torch.cuda.Stream] = None,
        max_span: Optional[int] = None,
    ) -> str:
        """Skip or incrementally patch ``copy_batch_block_offsets_to_device`` when safe.

        Returns ``"skip"``, ``"incremental"``, or ``"full"``.
        """
        storage = dst_tensor.untyped_storage()
        addr = storage.data_ptr()
        num_seqs = len(copy_idx_list)
        if (
            not self.incremental_block_offsets
            or num_seqs == 0
            or not dst_tensor.is_cuda
            or not dst_tensor.is_contiguous()
            or dst_tensor.dim() != 4
            or dst_tensor.shape[2] != 2
            or dst_tensor.dtype != torch.int32
            or host_src.is_cuda
            or host_src.dtype != torch.int32
            or torch.cuda.is_current_stream_capturing()
        ):
            self._block_offsets.pop(addr, None)
            fallback_fn()
            return "full"

        num_pools = host_src.shape[0]
        if dst_tensor.shape[0] != num_pools or dst_tensor.shape[3] != host_src.shape[3]:
            self._block_offsets.pop(addr, None)
            fallback_fn()
            return "full"

        slots_tuple = tuple(int(s) for s in copy_idx_list)
        scales_tuple = tuple(int(v) for v in index_scales[:num_pools])
        offsets_tuple = tuple(int(v) for v in kv_offsets[:num_pools])
        key = (
            manager_id,
            dst_tensor.storage_offset(),
            tuple(dst_tensor.shape),
            tuple(dst_tensor.stride()),
            num_seqs,
            slots_tuple,
            scales_tuple,
            offsets_tuple,
        )
        src_np = host_src.numpy()
        entry = self._block_offsets.get(addr)
        if entry is None or entry[1] != key:
            self._block_offsets.pop(addr, None)
            fallback_fn()
            snap = np.stack([src_np[:, slot, 0, :] for slot in slots_tuple], axis=1).copy()
            tails = []
            for p in range(num_pools):
                row_tails = []
                for s in range(num_seqs):
                    r = snap[p, s, :]
                    bads = np.flatnonzero(r == bad_page_index)
                    row_tails.append(int(bads[0]) if bads.size > 0 else int(r.size))
                tails.append(row_tails)
            self._block_offsets[addr] = (storage, key, snap, tails)
            return "full"

        snap = entry[2]
        tails = entry[3]
        limit = self.BLOCK_OFFSET_MAX_SPAN if max_span is None else int(max_span)
        patches: list[tuple[int, int, int, int, np.ndarray]] = []
        total_span = 0
        n_cols = snap.shape[2]
        row_bytes = n_cols * 4
        memcmp = _MEMCMP
        for p in range(num_pools):
            for s, slot in enumerate(slots_tuple):
                cur_row = src_np[p, slot, 0, :]
                prev_row = snap[p, s, :]
                if (
                    memcmp is not None
                    and cur_row.flags.c_contiguous
                    and prev_row.flags.c_contiguous
                ):
                    c_ptr = cur_row.ctypes.data
                    p_ptr = prev_row.ctypes.data
                    if memcmp(c_ptr, p_ptr, row_bytes) == 0:
                        continue
                    t_hint = tails[p][s]
                    w_lo = max(0, min(t_hint, n_cols) - 1)
                    w_hi = min(n_cols, max(0, t_hint) + 8)
                    lo = -1
                    hi = -1
                    pre_ok = (w_lo == 0) or (memcmp(c_ptr, p_ptr, w_lo * 4) == 0)
                    suf_b = (n_cols - w_hi) * 4
                    suf_ok = (suf_b == 0) or (
                        memcmp(c_ptr + w_hi * 4, p_ptr + w_hi * 4, suf_b) == 0
                    )
                    if pre_ok and suf_ok:
                        for idx in range(w_lo, w_hi):
                            if cur_row[idx] != prev_row[idx]:
                                if lo < 0:
                                    lo = idx
                                hi = idx + 1
                else:
                    if np.array_equal(cur_row, prev_row):
                        continue
                    lo = -1
                    hi = -1
                if lo < 0:
                    diff = np.flatnonzero(cur_row != prev_row)
                    lo = int(diff[0])
                    hi = int(diff[-1]) + 1
                total_span += hi - lo
                if total_span > limit:
                    self._block_offsets.pop(addr, None)
                    fallback_fn()
                    new_snap = np.stack([src_np[:, sl, 0, :] for sl in slots_tuple], axis=1).copy()
                    new_tails = []
                    for pp in range(num_pools):
                        rt = []
                        for ss in range(num_seqs):
                            rr = new_snap[pp, ss, :]
                            bb = np.flatnonzero(rr == bad_page_index)
                            rt.append(int(bb[0]) if bb.size > 0 else int(rr.size))
                        new_tails.append(rt)
                    self._block_offsets[addr] = (storage, key, new_snap, new_tails)
                    return "full"
                tails[p][s] = hi
                patches.append((p, s, lo, hi, cur_row))

        if not patches:
            return "skip"

        cur_stream = torch.cuda.current_stream(dst_tensor.device)
        can_pack = step_inputs.is_open() and (stream is None or stream == cur_stream)
        for p, s, lo, hi, cur_row in patches:
            span = hi - lo
            scale = scales_tuple[p]
            kv_off = offsets_tuple[p]
            if span == 1:
                val = int(cur_row[lo])
                kv0 = 0 if val == bad_page_index else val * scale
                kv1 = 0 if val == bad_page_index else kv0 + kv_off
                self._bo_k1[0] = kv0
                self._bo_v1[0] = kv1
                k_t = self._bo_k1
                v_t = self._bo_v1
            else:
                sub = cur_row[lo:hi]
                bad = sub == bad_page_index
                k_t = torch.from_numpy(np.where(bad, 0, sub * scale).astype(np.int32, copy=False))
                v_t = torch.from_numpy(
                    np.where(bad, 0, sub * scale + kv_off).astype(np.int32, copy=False)
                )
            if can_pack:
                step_inputs.copy_(dst_tensor[p, s, 0, lo:hi], k_t)
                step_inputs.copy_(dst_tensor[p, s, 1, lo:hi], v_t)
            else:
                stage = self._get_bo_stage(span * 2, dst_tensor.device)
                stage[:span].copy_(k_t)
                stage[span : 2 * span].copy_(v_t)
                if stream is not None and stream != cur_stream:
                    with torch.cuda.stream(stream):
                        dst_tensor[p, s, 0, lo:hi].copy_(stage[:span], non_blocking=True)
                        dst_tensor[p, s, 1, lo:hi].copy_(stage[span : 2 * span], non_blocking=True)
                else:
                    dst_tensor[p, s, 0, lo:hi].copy_(stage[:span], non_blocking=True)
                    dst_tensor[p, s, 1, lo:hi].copy_(stage[span : 2 * span], non_blocking=True)
            snap[p, s, lo:hi] = cur_row[lo:hi]
        return "incremental"

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


def incremental_block_offsets_enabled() -> bool:
    """Whether the active cache enables incremental KV block-offset updates."""
    cache = _active.get()
    return cache is not None and cache.incremental_block_offsets


def copy_block_offsets_if_changed(
    dst_tensor: torch.Tensor,
    host_src: torch.Tensor,
    copy_idx_list: Sequence[int],
    index_scales: Sequence[int],
    kv_offsets: Sequence[int],
    bad_page_index: int,
    fallback_fn: Callable[[], None],
    *,
    manager_id: object,
    stream: Optional[torch.cuda.Stream] = None,
) -> str:
    """Route ``copy_batch_block_offsets_to_device`` through the active cache when enabled."""
    cache = _active.get()
    if cache is None or not cache.incremental_block_offsets:
        fallback_fn()
        return "full"
    return cache.copy_block_offsets(
        dst_tensor,
        host_src,
        copy_idx_list,
        index_scales,
        kv_offsets,
        bad_page_index,
        fallback_fn,
        manager_id=manager_id,
        stream=stream,
    )


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
