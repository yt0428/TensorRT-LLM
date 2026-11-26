# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack a decode step's small per-step input copies into one upload and one Triton scatter launch.

Between two decode-graph replays, the next step's metadata reaches the device as a chain of tiny
copies (positions, KV lengths, cache slots, and the graph runner's static-input copies). Each is a
separate operation on the stream the next graph waits for, so the GPU idles for the whole chain.

While a pack is open (``begin`` ... ``flush``), ``copy_(dst, src)`` records a host-to-device copy
by snapshotting the host bytes into a pinned staging buffer, and a device-to-device copy as a
(dst, src) descriptor. ``flush`` issues one host-to-device copy of [descriptors | data] into a
device staging buffer and a pure-Triton scatter launch, which writes every staged host copy and
then every device copy (in that order, so a device copy may read a destination a host copy just
filled). With no pack open, ``copy_`` is ``dst.copy_(src, non_blocking=True)``.

With ``parallel_scatter`` (``RuntimeHooks.pack_parallel_scatter``), a pack whose host-copy
destinations are pairwise disjoint and whose device copies have no overlap or read-after-write
hazard is issued with one program per copy (``_parallel_scatter_kernel``). Any other pack runs
``_serial_scatter_kernel`` (one program applying copies strictly in order, pure Triton with zero
C++ op dependency).

With ``in_graph`` (``RuntimeHooks.pack_in_graph``, on top of ``parallel_scatter``), the issue
moves into the replayed graph: a graph captured with ``capture_prologue`` copies a fixed pinned
window (``ghost``) to the device staging buffer, records ``ghost_free``, and runs
``_graph_scatter_kernel`` with the copy counts read from the staged header. ``flush(in_graph=True)``
fills the window on the host once ``ghost_free`` shows the previous replay has consumed it.

With ``early_copy`` (``RuntimeHooks.pack_early_copy``, on top of ``in_graph``), the window reaches
the device ahead of the replay on a side copy stream into a dedicated device window (``gdev``)
while the previous graph may still run. The prologue waits for ``staged_ready``, scatters device
to device, zeroes the header, and records ``prologue_done`` for the next side-stream upload.

With ``step_epoch`` (``RuntimeHooks.pack_step_epoch``, on top of ``in_graph``), the pack keeps
one int64 device counter ``StepInputPack.epoch`` (``step_epoch()`` for the active pack) that
program 0 of the prologue increments once per replay.

With ``prologue_op`` (``RuntimeHooks.pack_prologue_op``, on top of ``early_copy``), the prologue
calls ``prologue_op(gdev, epoch, MAX_DESC)`` in place of the default two scatter launches and
header zero.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned

__all__ = [
    "StepInputPack",
    "active",
    "begin",
    "capture_prologue",
    "copy_",
    "copy_metadata_",
    "discard",
    "flush",
    "is_active",
    "is_open",
    "step_epoch",
]


@triton.jit
def _serial_scatter_kernel(stage_u8, stage_i64, BLOCK: tl.constexpr):
    """Apply every staged host copy and then every device copy strictly in descriptor order."""
    num_h2d = tl.load(stage_i64)
    num_d2d = tl.load(stage_i64 + 1)
    for i in range(0, num_h2d):
        d = 2 + 3 * i
        dst = tl.load(stage_i64 + d).to(tl.pointer_type(tl.uint8))
        src = stage_u8 + tl.load(stage_i64 + d + 1)
        nbytes = tl.load(stage_i64 + d + 2)
        for start in range(0, nbytes, BLOCK):
            o = start + tl.arange(0, BLOCK)
            tl.store(dst + o, tl.load(src + o, mask=o < nbytes), mask=o < nbytes)
    for i in range(0, num_d2d):
        d = 2 + 3 * (num_h2d + i)
        dst = tl.load(stage_i64 + d).to(tl.pointer_type(tl.uint8))
        src = tl.load(stage_i64 + d + 1).to(tl.pointer_type(tl.uint8))
        nbytes = tl.load(stage_i64 + d + 2)
        for start in range(0, nbytes, BLOCK):
            o = start + tl.arange(0, BLOCK)
            tl.store(dst + o, tl.load(src + o, mask=o < nbytes), mask=o < nbytes)


# One compiled variant for every descriptor offset: specializing ``first`` (== 1, divisible by 16)
# would compile a new kernel when a step's pack first holds a new number of copies after warm-up.
@triton.jit(do_not_specialize=["first"])
def _parallel_scatter_kernel(
    stage_u8, stage_i64, first, DEVICE_SRC: tl.constexpr, BLOCK: tl.constexpr
):
    """Program i writes copy ``first + i`` ([dst, src, bytes] descriptors after the 2-word header).

    ``src`` is an offset into the staged buffer for a host copy or a device address for a device
    copy (``DEVICE_SRC``).
    """
    d = 2 + 3 * (first + tl.program_id(0))
    dst = tl.load(stage_i64 + d).to(tl.pointer_type(tl.uint8))
    src_word = tl.load(stage_i64 + d + 1)
    if DEVICE_SRC:
        src = src_word.to(tl.pointer_type(tl.uint8))
    else:
        src = stage_u8 + src_word
    nbytes = tl.load(stage_i64 + d + 2)
    for start in range(0, nbytes, BLOCK):
        o = start + tl.arange(0, BLOCK)
        tl.store(dst + o, tl.load(src + o, mask=o < nbytes), mask=o < nbytes)


@triton.jit
def _graph_scatter_kernel(
    stage_u8,
    stage_i64,
    epoch,
    DEVICE_SRC: tl.constexpr,
    BLOCK: tl.constexpr,
    BUMP_EPOCH: tl.constexpr,
):
    """Descriptor slot i of host copies (``DEVICE_SRC`` False) or device copies.

    Counts are read from the staged header; slots past their count write nothing.
    ``BUMP_EPOCH``: program 0 also increments ``epoch[0]``.
    """
    i = tl.program_id(0)
    if BUMP_EPOCH:
        if i == 0:
            tl.store(epoch, tl.load(epoch) + 1)
    num_h2d = tl.load(stage_i64)
    if DEVICE_SRC:
        count = tl.load(stage_i64 + 1)
        d = 2 + 3 * (num_h2d + i)
    else:
        count = num_h2d
        d = 2 + 3 * i
    if i < count:
        dst = tl.load(stage_i64 + d).to(tl.pointer_type(tl.uint8))
        src_word = tl.load(stage_i64 + d + 1)
        if DEVICE_SRC:
            src = src_word.to(tl.pointer_type(tl.uint8))
        else:
            src = stage_u8 + src_word
        nbytes = tl.load(stage_i64 + d + 2)
        for start in range(0, nbytes, BLOCK):
            o = start + tl.arange(0, BLOCK)
            tl.store(dst + o, tl.load(src + o, mask=o < nbytes), mask=o < nbytes)


def _disjoint(ranges: list[tuple[int, int]]) -> bool:
    """Whether the [start, start + bytes) ranges are pairwise disjoint."""
    end = -1
    for start, nbytes in sorted(ranges):
        if start < end:
            return False
        end = start + nbytes
    return True


def _parallel_ok(h2d: list[tuple[int, int, int]], d2d: list[tuple[int, int, int]]) -> bool:
    """Whether issuing host copies in parallel and then device copies in parallel matches serial.

    Host destinations must be pairwise disjoint, device destinations must be pairwise disjoint,
    and no device destination may overlap any device source (a host copy may still feed a device
    copy because the host scatter finishes before the device scatter launches).
    """
    if not _disjoint([(dst, n) for dst, _, n in h2d]):
        return False
    if not d2d:
        return True
    dsts = [(dst, n) for dst, _, n in d2d]
    if not _disjoint(dsts):
        return False
    return all(_disjoint(dsts + [(src, n)]) for _, src, n in d2d)


class StepInputPack:
    """One engine's pack of deferred step-input copies (see the module docstring)."""

    CAPACITY = 1 << 16
    MAX_DESC = 256
    # Pinned staging buffers in rotation: a forced flush reopens the pack without waiting on the
    # upload just issued.
    RING = 4
    HEADER_BYTES = 16
    DESC_BYTES = 24
    ALIGN = 16
    # Pinned window a graph's prologue copies (in_graph): header, descriptors, and ~10 KB data.
    GRAPH_WINDOW = 1 << 14
    # A model's own pack (RuntimeHooks.step_input_pack_factory) may set this on its class; the
    # stock pack keeps it off. The engine then issues the pack right after input preparation,
    # before it records the input-copy event, instead of at the graph replay, and uploads routed
    # through ``copy_metadata_`` join it.
    issue_after_prepare = False

    def __init__(
        self,
        device: torch.device,
        parallel_scatter: bool = False,
        in_graph: bool = False,
        early_copy: bool = False,
        step_epoch: bool = False,
        prologue_op: Optional[Callable[[torch.Tensor, Optional[torch.Tensor], int], None]] = None,
    ) -> None:
        self.parallel_scatter = parallel_scatter
        self.in_graph = in_graph and parallel_scatter
        self.early_copy = early_copy and self.in_graph
        self.prologue_op = prologue_op if self.early_copy else None
        self.epoch: Optional[torch.Tensor] = (
            torch.zeros(1, dtype=torch.int64, device=device)
            if step_epoch and self.in_graph
            else None
        )
        if self.early_copy:
            # The prologue's own device window: an eager flush (self.dev) never leaves a header here.
            self.gdev = torch.zeros(self.GRAPH_WINDOW, dtype=torch.uint8, device=device)
            self.copy_stream = torch.cuda.Stream(device=device)
            # Recorded by the side upload once it landed (the prologue waits on it) ...
            self.staged_ready = torch.cuda.Event(external=True)
            # ... and by every replayed prologue after its scatter (the next upload waits on it).
            self.prologue_done = torch.cuda.Event(external=True)
            # Create both underlying CUDA events now so capture never skips waiting on staged_ready.
            self.staged_ready.record(self.copy_stream)
            self.prologue_done.record(self.copy_stream)
        if self.in_graph:
            self.ghost = torch.zeros(
                self.GRAPH_WINDOW, dtype=torch.uint8, pin_memory=prefer_pinned()
            )
            # Recorded by every replayed prologue or side upload right after reading ghost.
            self.ghost_free = torch.cuda.Event(external=True)
        self.host = [
            torch.empty(self.CAPACITY, dtype=torch.uint8, pin_memory=prefer_pinned())
            for _ in range(self.RING)
        ]
        self.events: list[Optional[torch.cuda.Event]] = [None] * self.RING
        self.dev = torch.empty(self.CAPACITY, dtype=torch.uint8, device=device)
        self.data_start = self.HEADER_BYTES + self.DESC_BYTES * self.MAX_DESC
        self.parity = 0
        self.open = False
        self.h2d: list[tuple[int, int, int]] = []
        self.d2d: list[tuple[int, int, int]] = []
        # Every tensor a staged copy reads or writes, kept alive until the pack is issued.
        self.staged: list[torch.Tensor] = []
        self._graph_staged: list[torch.Tensor] = []
        self.used = self.data_start
        if device.type == "cuda" and torch.cuda.is_available():
            self._warm_up(device)
            if self.in_graph:
                self._warm_up_graph_kernel()

    def _warm_up(self, device: torch.device) -> None:
        """Compile the eager scatter kernels now, not on a serving step's flush."""
        if torch.cuda.is_current_stream_capturing():
            return
        # Always compile the serial Triton fallback with a zero-count header (writes nothing).
        self.dev[: self.HEADER_BYTES].zero_()
        _serial_scatter_kernel[(1,)](self.dev, self.dev.view(torch.int64), BLOCK=256, num_warps=4)
        if self.parallel_scatter:
            dst = torch.zeros(8, dtype=torch.int32, device=device)
            self.begin()
            self.stage(dst[:4], torch.arange(4, dtype=torch.int32))
            self.stage(dst[4:], dst[:4])
            self.flush()
        torch.cuda.synchronize(device)

    def _warm_up_graph_kernel(self) -> None:
        """Compile the prologue's scatter now (a zero-count header: it writes nothing)."""
        if torch.cuda.is_current_stream_capturing():
            return
        self.dev[: self.HEADER_BYTES].zero_()
        if self.prologue_op is not None:
            self.prologue_op(self.gdev, self.epoch, self.MAX_DESC)
        else:
            self._launch_graph_scatter(self.dev)
        torch.cuda.synchronize(self.dev.device)

    def _launch_graph_scatter(self, window: torch.Tensor) -> None:
        stage_i64 = window.view(torch.int64)
        for device_src in (False, True):
            bump = self.epoch is not None and not device_src
            _graph_scatter_kernel[(self.MAX_DESC,)](
                window,
                stage_i64,
                self.epoch if bump else stage_i64,
                DEVICE_SRC=device_src,
                BLOCK=256,
                BUMP_EPOCH=bump,
                num_warps=1,
            )

    def capture_prologue(self) -> bool:
        """Inside a graph capture: emit the prologue that issues the window ``flush(in_graph=True)``
        fills. Returns whether the graph has a prologue."""
        if not self.in_graph:
            return False
        if self.early_copy:
            torch.cuda.current_stream().wait_event(self.staged_ready)
            if self.prologue_op is not None:
                self.prologue_op(self.gdev, self.epoch, self.MAX_DESC)
            else:
                self._launch_graph_scatter(self.gdev)
                # Consumed: a replay with no new upload scatters nothing.
                self.gdev[: self.HEADER_BYTES].zero_()
            self.prologue_done.record()
            return True
        self.dev[: self.GRAPH_WINDOW].copy_(self.ghost, non_blocking=True)
        self.ghost_free.record()
        self._launch_graph_scatter(self.dev)
        return True

    def _upload_window(self, nbytes: int) -> None:
        """early_copy: upload the filled pinned window into gdev on the side stream once the last
        replayed prologue has consumed gdev."""
        stream = self.copy_stream
        stream.wait_event(self.prologue_done)
        with torch.cuda.stream(stream):
            self.gdev[:nbytes].copy_(self.ghost[:nbytes], non_blocking=True)
        self.ghost_free.record(stream)
        self.staged_ready.record(stream)

    def _ghost_writable(self) -> None:
        # The last replayed prologue or side upload has read ghost (no-op before first record).
        self.ghost_free.synchronize()

    def quiesce_graph_window(self) -> None:
        """Before replaying a graph with the prologue without a packed upload: ensure a zero-count
        window so the prologue writes nothing."""
        if self.in_graph and self.ghost[: self.HEADER_BYTES].any():
            self._ghost_writable()
            self.ghost[: self.HEADER_BYTES].zero_()
            if self.early_copy:
                # A window uploaded but never replayed would still sit in gdev: zero its header.
                self._upload_window(self.HEADER_BYTES)

    def begin(self) -> None:
        if self.open:
            self.flush()
        event = self.events[self.parity]
        if event is not None:
            # The upload that last read this pinned buffer has run.
            event.synchronize()
        self._clear()
        self.open = True

    def _clear(self) -> None:
        self.h2d.clear()
        self.d2d.clear()
        self.staged.clear()
        self.used = self.data_start

    def stage(self, dst: torch.Tensor, src: torch.Tensor) -> bool:
        if not dst.is_cuda or dst.numel() != src.numel():
            return False
        nbytes = dst.numel() * dst.element_size()
        if src.device.type == "cpu":
            # A contiguous destination is one descriptor; a 2-D slice with contiguous rows (e.g.
            # a page table's leading rows and columns) is one per row.
            if dst.is_contiguous():
                rows, row_bytes, row_stride = 1, nbytes, 0
            elif dst.dim() == 2 and dst.stride(1) == 1:
                rows = dst.shape[0]
                row_bytes = dst.shape[1] * dst.element_size()
                row_stride = dst.stride(0) * dst.element_size()
            else:
                return False
            offset = (self.used + self.ALIGN - 1) // self.ALIGN * self.ALIGN
            if (
                offset + nbytes > self.CAPACITY
                or len(self.h2d) + len(self.d2d) + rows > self.MAX_DESC
            ):
                return False
            # A dtype change converts on the host, as copy_ would on the device.
            src = src.to(dst.dtype).contiguous()
            self.host[self.parity][offset : offset + nbytes].copy_(src.view(-1).view(torch.uint8))
            base = dst.data_ptr()
            for row in range(rows):
                self.h2d.append((base + row * row_stride, offset + row * row_bytes, row_bytes))
            self.staged.append(dst)
            self.used = offset + nbytes
            return True
        if len(self.h2d) + len(self.d2d) >= self.MAX_DESC:
            return False
        if src.is_cuda and src.dtype == dst.dtype and src.is_contiguous() and dst.is_contiguous():
            self.d2d.append((dst.data_ptr(), src.data_ptr(), nbytes))
            self.staged += (dst, src)
            return True
        return False

    def flush(self, in_graph: bool = False) -> None:
        """Issue the pack. ``in_graph``: the next replay is of a graph captured with
        ``capture_prologue``."""
        if not self.open:
            if in_graph:
                self.quiesce_graph_window()
            return
        self.open = False
        if not self.h2d and not self.d2d:
            if in_graph:
                self.quiesce_graph_window()
            self._clear()
            return
        if in_graph and self.in_graph:
            if self._fill_graph_window():
                if self.early_copy:
                    self._upload_window(self.used)
                self._graph_staged.extend(self.staged)
                self._clear()
                return
            self.quiesce_graph_window()
        host = self.host[self.parity]
        desc = [len(self.h2d), len(self.d2d)] + [
            value for copy in self.h2d + self.d2d for value in copy
        ]
        host[: 8 * len(desc)].view(torch.int64).copy_(torch.tensor(desc, dtype=torch.int64))
        # Descriptors and data in one upload; the gap between them is unused padding.
        self.dev[: self.used].copy_(host[: self.used], non_blocking=True)
        stage_i64 = self.dev.view(torch.int64)
        if self.parallel_scatter and _parallel_ok(self.h2d, self.d2d):
            if self.h2d:
                _parallel_scatter_kernel[(len(self.h2d),)](
                    self.dev, stage_i64, 0, DEVICE_SRC=False, BLOCK=256, num_warps=2
                )
            if self.d2d:
                _parallel_scatter_kernel[(len(self.d2d),)](
                    self.dev, stage_i64, len(self.h2d), DEVICE_SRC=True, BLOCK=256, num_warps=2
                )
        else:
            _serial_scatter_kernel[(1,)](self.dev, stage_i64, BLOCK=256, num_warps=4)
        event = self.events[self.parity]
        if event is None:
            event = self.events[self.parity] = torch.cuda.Event()
        event.record()
        self.parity = (self.parity + 1) % self.RING
        self._release_after_submission(self.staged)
        self._clear()

    def _release_after_submission(self, tensors: list[torch.Tensor]) -> None:
        if tensors:
            stream = torch.cuda.current_stream(self.dev.device)
            for tensor in tensors:
                tensor.record_stream(stream)
            tensors.clear()

    def on_graph_replay(self) -> None:
        """Release window operands after the graph reading their addresses has been queued."""
        self._release_after_submission(self._graph_staged)

    def _fill_graph_window(self) -> bool:
        """Write the pack into the pinned window the replay's prologue issues; False when it does
        not fit or is not order-free (the caller issues it eagerly)."""
        if self.used > self.GRAPH_WINDOW or not _parallel_ok(self.h2d, self.d2d):
            return False
        desc = [len(self.h2d), len(self.d2d)] + [
            value for copy in self.h2d + self.d2d for value in copy
        ]
        self._ghost_writable()
        host = self.host[self.parity]
        self.ghost[: self.used].copy_(host[: self.used])
        self.ghost[: 8 * len(desc)].view(torch.int64).copy_(torch.tensor(desc, dtype=torch.int64))
        return True

    def discard(self) -> None:
        """Close the pack without issuing it (its step failed)."""
        self.open = False
        self._clear()


_active: ContextVar[Optional[StepInputPack]] = ContextVar("trtllm_step_input_pack", default=None)


@contextlib.contextmanager
def active(pack: Optional[StepInputPack]) -> Iterator[None]:
    """Make ``pack`` the calling context's pack (None: no packing) for the duration."""
    token = _active.set(pack)
    try:
        yield
    finally:
        _active.reset(token)


def begin() -> None:
    """Open the active pack for the coming copies. Call only when the step will reach ``flush``."""
    pack = _active.get()
    if pack is not None:
        pack.begin()


def is_active(pack: StepInputPack) -> bool:
    """Whether ``pack`` is the calling context's pack."""
    return _active.get() is pack


def is_open() -> bool:
    pack = _active.get()
    return pack is not None and pack.open


def copy_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """``dst.copy_(src, non_blocking=True)``, deferred into the active open pack when there is one."""
    pack = _active.get()
    if pack is not None and pack.open and not torch.cuda.is_current_stream_capturing():
        if pack.stage(dst, src):
            return dst
        # Keep the order: everything staged so far lands before this copy, and the copies after it
        # pack again.
        pack.flush()
        out = dst.copy_(src, non_blocking=True)
        pack.begin()
        return out
    return dst.copy_(src, non_blocking=True)


def copy_metadata_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """``copy_`` when the active open pack is issued right after input preparation
    (``StepInputPack.issue_after_prepare``), else ``dst.copy_(src, non_blocking=True)``.
    """
    pack = _active.get()
    if pack is not None and pack.open and pack.issue_after_prepare:
        return copy_(dst, src)
    return dst.copy_(src, non_blocking=True)


def discard() -> None:
    """Close the active pack without issuing it (its step failed)."""
    pack = _active.get()
    if pack is not None:
        pack.discard()


def flush(in_graph: bool = False) -> None:
    """Issue the active open pack (one upload and one scatter launch on the current stream) and
    close it.

    ``in_graph``: the next replay is of a graph whose prologue issues the pack
    (``capture_prologue``).
    """
    pack = _active.get()
    if pack is not None:
        pack.flush(in_graph=in_graph)


def step_epoch() -> Optional[torch.Tensor]:
    """The active pack's step epoch (an int64 [1] device tensor), or None when there is none."""
    pack = _active.get()
    return pack.epoch if pack is not None else None


def capture_prologue() -> Optional[StepInputPack]:
    """Inside a graph capture: add the active pack's in-graph prologue; returns the pack whose
    window the graph issues, or None when the graph has no prologue."""
    pack = _active.get()
    return pack if pack is not None and pack.capture_prologue() else None
