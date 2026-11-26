# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode top-k of the DSA indexer in one CuTe DSL launch: the trtllm op's selection, per row."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from tensorrt_llm.logger import logger

from .._op import Arch, Cell, OpWrapper
from . import _cluster_radix_topk_decode as _cluster_kernel
from . import _radix_topk_decode as _kernel
from .indexer_topk_decode import indexer_topk_decode

TOP_K = _kernel.TOP_K
_PAD_KEY = _kernel.PAD_KEY
_OP_FINAL_ITEMS = _kernel.OP_FINAL_ITEMS
#: fp16(-FLT_MAX) is -inf, 0xFC00, whose 11-bit ordered key is 0x7E0.
_PAD_FP16_BIN = 0x7E0

#: The trtllm op's split-work dispatch (`computeIndexerTopKDecodeBlocksPerRow`,
#: `kernels/indexerTopK.cu`): its column threshold for 10 blocks per row, its cap on blocks
#: per row, and the fewest columns per block.
_SPLIT_WORK_COLUMNS = 200_000
_MAX_BLOCKS_PER_ROW = 10
_MIN_COLUMNS_PER_BLOCK = 2048

#: The Mistral Large 4 NVFP4 targets' decode call: next_n 1, and their max_seq_len of 32768 as
#: the paged scorer's column count (a multiple of its 256-column alignment, so contiguous).
_ML4 = dict(next_n=1, width=32768)
#: KV lengths in the 8K/1K decode window: 8192-token prompts plus up to 1024 generated tokens.
_ML4_DECODE_WINDOW = [8600 + (37 * i) % 617 for i in range(256)]
#: The targets' other scorer widths: the decode scorer's column count is the engine's
#: max_seq_len, and a target builds this selector at any width from 4096 (where the op splits a
#: row over 2 blocks at every row count) to 65536 (its 10 blocks of 6553 columns at 1..15 rows).
_OTHER_WIDTHS = (4096, 8192, 16384, 65536)


def _lens_within(width: int, n: int) -> list[int]:
    """`n` KV lengths a `width`-column scorer row holds: one key, the top-k edge (2047, 2048,
    2049), the full row and one key short of it, half of it, then a spread over 1..width."""
    edges = [1, 2047, 2048, 2049, width, width - 1, width // 2 + 1, 3]
    return (edges + [1 + (7919 * i) % width for i in range(n)])[:n]


def blocks_per_row(num_rows: int, num_columns: int, sm_count: int) -> int:
    """How many blocks the trtllm op splits each row across, for this call's shape.

    The op's own host dispatch, line for line: a pure function of the row count, the column
    count and the device's SM count, so a captured graph keeps its capture's value. The
    selection (see `_radix_topk_decode`) depends on it through the op's padding."""
    if num_rows <= 0:
        return 1
    if num_columns >= _SPLIT_WORK_COLUMNS:
        return _MAX_BLOCKS_PER_ROW
    max_bp = min(max(1, num_columns // _MIN_COLUMNS_PER_BLOCK), _MAX_BLOCKS_PER_ROW)
    if num_rows < sm_count // 2:
        best_bp, best_waves = 1, 1
        for bp in range(2, max_bp + 1):
            waves = (num_rows * bp + sm_count - 1) // sm_count
            if waves * best_bp < best_waves * bp:
                best_bp, best_waves = bp, waves
        return max(1, best_bp)
    return max(1, min(2, max_bp))


@dataclass(frozen=True)
class CompiledRadixTopkDecode:
    """A compiled launcher with the facts its call needs: device, next_n and SM count."""

    fn: object
    device_id: int
    next_n: int
    sm_count: int


_COMPILED: dict[tuple[int, int], CompiledRadixTopkDecode] = {}


def build(device_id: int, *, next_n: int = 1) -> CompiledRadixTopkDecode:
    """Compile the kernel for `next_n` and launch it once, ahead of any capture.

    The row count, the column count and the logits row stride are symbolic, so one handle
    serves every batch and width. A few seconds of compile, cached per process."""
    key = (int(device_id), int(next_n))
    if key in _COMPILED:
        return _COMPILED[key]
    with torch.cuda.device(device_id):
        rows, cols, batch, stride = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int64()
        fake_logits = make_fake_tensor(
            cutlass.Float32, (rows, cols), stride=(stride, 1), assumed_align=4
        )
        fake_lens = make_fake_compact_tensor(cutlass.Int32, (batch,), assumed_align=4)
        fake_out = make_fake_compact_tensor(
            cutlass.Int32, (rows, TOP_K), stride_order=(1, 0), assumed_align=4
        )
        fn = cute.compile(
            _kernel.RadixTopkDecodeKernel(int(next_n)),
            fake_logits,
            fake_lens,
            fake_out,
            cutlass.Int32(2),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        sm_count = torch.cuda.get_device_properties(device_id).multi_processor_count
        handle = CompiledRadixTopkDecode(fn, int(device_id), int(next_n), int(sm_count))
        logits = torch.zeros((next_n, 4096), dtype=torch.float32, device="cuda")
        lens = torch.full((1,), 4096, dtype=torch.int32, device="cuda")
        out = torch.empty((next_n, TOP_K), dtype=torch.int32, device="cuda")
        fn(logits, lens, out, blocks_per_row(next_n, 4096, sm_count))
        torch.cuda.synchronize()
    _COMPILED[key] = handle
    return handle


def _keys(values: torch.Tensor) -> torch.Tensor:
    """fp32 scores as int64 keys of the op's order: ascending key is descending score."""
    bits = values.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(bits >= 0x80000000, bits, (~bits) & 0x7FFFFFFF)


def _fp16_bins(values: torch.Tensor) -> torch.Tensor:
    """The op's first-pass bins: the 11-bit ordered key of each score's fp16 image, with
    every NaN in the top bin (the device conversion yields the canonical positive NaN)."""
    bits = values.half().view(torch.int16).long() & 0xFFFF
    key = torch.where(bits >= 0x8000, bits, (~bits) & 0x7FFF)
    return torch.where(torch.isnan(values), 0, key >> 5)


def _top(keys: torch.Tensor, pad: int, need: int) -> tuple[torch.Tensor, int]:
    """Positions of the `need` best of `keys` and `pad` pads at key(-FLT_MAX): by key, a
    column before a pad, then by position. Returns the column positions and the pads taken."""
    rank = torch.cat([keys * 2, torch.full((pad,), _PAD_KEY * 2 + 1, device=keys.device)])
    order = torch.sort(rank, stable=True).indices[:need]
    cols = order[order < keys.numel()]
    return cols, need - cols.numel()


def _stage(values: torch.Tensor, keys: torch.Tensor, pad: int) -> tuple[torch.Tensor, int]:
    """One selection of the op's kernel over `values` and `pad` pads: its fp16 first pass
    decides when its threshold bin holds at most 2048 entries, its key passes otherwise.
    Returns the selected positions and the pads taken."""
    bins = torch.cat([_fp16_bins(values), torch.full((pad,), _PAD_FP16_BIN, device=keys.device)])
    counts = torch.bincount(bins, minlength=2048)
    t0 = int(torch.searchsorted(torch.cumsum(counts, 0), TOP_K))
    before, count = int(counts[:t0].sum()), int(counts[t0])
    if count > _OP_FINAL_ITEMS:
        return _top(keys, pad, TOP_K)
    vbins = bins[: values.numel()]
    above = (vbins < t0).nonzero().flatten()
    inbin = (vbins == t0).nonzero().flatten()
    picked, pads = _top(keys[inbin], pad if _PAD_FP16_BIN == t0 else 0, TOP_K - before)
    pads += pad if _PAD_FP16_BIN < t0 else 0
    return torch.cat([above, inbin[picked]]), pads


class CuteDslRadixTopkDecode(OpWrapper):
    """Select each decode row's `top_k` (2048) entries the way the trtllm op does, in one launch.

    Row `r = b * next_n + t` is sequence `b`'s `t`-th new token; its columns are `[0, end)`
    with `end = seq_lens[b] - next_n + t + 1`. The row's selection is the trtllm
    `indexer_topk_decode` op's on its split path: scores ranked by fp32 bit pattern
    (`key(x) = bits` for a negative `x`, else `~bits & 0x7FFFFFFF`, ascending), over the
    row's columns plus the op's padding entries -- `pad = sum(max(0, top_k - len))` entries
    at -FLT_MAX for the blocks of `len <= top_k` columns it cuts the row into
    (`blocks_per_row` of the call's shape), each standing for a -1. Selected columns come
    first in `indices[r]`, in no particular order (in column order when `end <= top_k`);
    the selected pads follow as -1. Where scores tie at the cut, which of them win is the
    op's freedom too; here a column wins a tie against a pad.

    So for scores above -FLT_MAX (every finite relu-weighted sum the paged scorer writes)
    the row holds its top `min(end, top_k)` columns and then -1s. Below that, the op's
    padding decides: a column scored -inf (and -FLT_MAX, which ties with a pad) is selected
    only where fewer than `top_k` columns and pads rank above it.

    One score is ranked by the op's fp16 first pass apart from its key: a negative NaN,
    whose fp16 image is the canonical positive NaN, the best score. A selection the op
    settles in that pass (its threshold bin holds at most 2048 entries) treats a -NaN as
    the best score, and one it settles by key as the worst. A row holding a negative NaN is
    therefore selected the op's way stage by stage: each of its blocks of more than top_k
    columns, then the merge of their winners, the shorter blocks' columns and the pads.

    `indices` (`[rows, top_k]` int32, contiguous) is written in place and returned.
    `next_n` and `compiled` are keyword-only.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The Mistral Large 4 NVFP4 tp4 / tep4 / dep4 targets (`_ML4`) score every generation token
    # of a step with the paged scorer, fp32 rows 32768 wide (their max_seq_len), and select
    # here: from their decode steps (CUDA-graph captured, padded to the engine's batch list
    # 1, 2, 4, 8, 16, 24, ..., 128, 192, 256; dep4 on each rank's own sequences) and from the
    # generation rows of a mixed step, which sit at the tail of the step's selection buffer.
    # At 32768 columns the op splits a row over 10 blocks at 1..15 rows, 9 at 16, and 2 from
    # 64 rows on; `scores` names the pattern the test writes into the rows.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Mistral Large 4 32768, 16 rows: ends 1..32768, the 2048/2049 top-k edge, the 8K/1K "
            "decode window and full rows",
            spec=dict(
                _ML4,
                seq_lens=[1, 2, 2047, 2048, 2049, 4096, 8600, 8960]
                + [9216, 12345, 16384, 20481, 30000, 32767, 32768, 9000],
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, row counts 1..16 and 17..256 (every blocks-per-row the op "
            "picks, 10 down to 2): leading rows of one 256-row batch in the decode window",
            spec=dict(
                _ML4,
                batches=tuple(range(1, 17)) + (17, 24, 31, 33, 64, 75, 76, 100, 128, 192, 256),
                seq_lens=[1, 2049, 32768, 2048] + _ML4_DECODE_WINDOW[4:],
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, captured decode batches 1..256 of the engine's padded list: "
            "replayed on new scores and seq_lens up to 32768, the padding dummy's row at 1",
            spec=dict(
                _ML4,
                graph_batches=(1, 2, 4, 8, 16, 24, 64, 128, 192, 256),
                seq_lens=_ML4_DECODE_WINDOW,
                replays=3,
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, indexer-like scores (head-weighted relu sums, many exact "
            "zeros) at 1, 16, 64 and 256 rows",
            spec=dict(
                _ML4,
                scores="indexer",
                batches=(1, 16, 64, 256),
                seq_lens=[9216, 32768, 2049, 20481] + _ML4_DECODE_WINDOW[4:],
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, tied scores at the cut at 16 and 256 rows: 16 levels, all "
            "+0.0 (relu's floor), mixed +-0.0",
            spec=dict(
                _ML4,
                scores="ties",
                batches=(16, 256),
                seq_lens=[32768, 32768, 32768, 9216, 9216, 9216, 20481, 20481]
                + [20481, 2049, 2049, 2049, 2048, 100, 1, 30000]
                + _ML4_DECODE_WINDOW[16:],
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, scores past fp16's range and under its resolution at 16 "
            "and 256 rows, both signs",
            spec=dict(
                _ML4,
                scores="wide",
                batches=(16, 256),
                seq_lens=[32768, 32767, 9216, 8600, 20481, 2049, 2048, 1]
                + [16385, 4097, 30000, 12345, 9000, 8192, 3000, 32768]
                + _ML4_DECODE_WINDOW[16:],
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, special values: +-NaN, +-inf, -FLT_MAX, -0.0 and denormals "
            "among few finite scores, short and long rows, at 1, 15, 16 and 64 rows",
            spec=dict(
                _ML4,
                scores="special",
                batches=(1, 15, 16, 64),
                seq_lens=[2100, 1000, 2048, 4000, 9216, 32768, 20481, 2049]
                + [3000, 12000, 1, 2, 5000, 6000, 30000, 2500]
                + _ML4_DECODE_WINDOW[16:64],
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, negative NaNs, the one score the op's fp16 pass ranks "
            "differently: among distinct scores (the pass decides) and in a crowd of +0.0 (it "
            "does not), short and long rows, at 1, 16 and 64 rows",
            spec=dict(
                _ML4,
                scores="negnan",
                batches=(1, 16, 64),
                seq_lens=[9216, 2048, 4000, 32768, 20481, 1500, 12000, 2049]
                + [30000, 5000, 2100, 16384, 8600, 3000, 25000, 7000]
                + _ML4_DECODE_WINDOW[16:64],
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, threshold groups past the 4096-entry list: up to 30000 "
            "equal scores at the cut, and 6000 sharing the top 11 key bits",
            spec=dict(
                _ML4,
                scores="crowded",
                batches=(1, 16),
                seq_lens=[32768, 9216, 30000, 20481, 12000, 8600, 32767, 4097]
                + [16384, 2049, 6145, 9000, 32768, 10000, 25000, 7000],
            ),
        ),
        Cell(
            why="Mistral Large 4 32768, mixed step: 15 generation rows at rows 32753..32767 of the "
            "step's 32768-row selection buffer; the context rows untouched",
            spec=dict(
                _ML4,
                tail_of=32768,
                seq_lens=[32768, 1, 2049, 20481, 2048, 8600, 32767, 3]
                + [9216, 7777, 4096, 1234, 30000, 6145, 16385],
            ),
        ),
    )
    # The targets' other scorer widths (an engine's max_seq_len of 4096, 8192, 16384 or 65536):
    # every row count 1..16 and up to 256, the captured batches replayed, ties at the cut and
    # special values, KV lengths within the width.
    CELLS += tuple(
        Cell(
            why=f"width {w}, row counts 1..16 and 17..256 (every blocks-per-row the op picks): "
            f"ends 1..{w}, the 2048/2049 top-k edge, full rows",
            spec=dict(
                next_n=1,
                width=w,
                batches=tuple(range(1, 17)) + (17, 33, 64, 76, 77, 128, 256),
                seq_lens=_lens_within(w, 256),
            ),
        )
        for w in _OTHER_WIDTHS
    )
    CELLS += tuple(
        Cell(
            why=f"width {w}, captured decode batches 1..256 of the engine's padded list: "
            f"replayed on new scores and seq_lens up to {w}, the padding dummy's row at 1",
            spec=dict(
                next_n=1,
                width=w,
                graph_batches=(1, 2, 4, 8, 16, 64, 256),
                seq_lens=_lens_within(w, 256),
                replays=2,
            ),
        )
        for w in _OTHER_WIDTHS
    )
    CELLS += tuple(
        Cell(
            why=f"width {w}, {what}",
            spec=dict(
                next_n=1, width=w, scores=kind, batches=batches, seq_lens=_lens_within(w, 256)
            ),
        )
        for w in _OTHER_WIDTHS
        for kind, what, batches in (
            (
                "ties",
                "tied scores at the cut at 16 and 256 rows: 16 levels, all +0.0, mixed +-0.0",
                (16, 256),
            ),
            (
                "special",
                "special values: +-NaN, +-inf, -FLT_MAX, -0.0 and denormals among few "
                "finite scores, at 1, 15, 16 and 64 rows",
                (1, 15, 16, 64),
            ),
        )
    )

    note = """
    `build(device_id, next_n=)` compiles and warms the kernel; a target builds it on its
    first forward, ahead of any capture, and passes the handle as `compiled`. Its row and
    column counts are symbolic, so one handle serves every batch, eager or captured, with
    nothing compiled later. No scratch: one 1024-thread CTA owns one row and works in about
    66 KB of shared memory. The call's host work is `blocks_per_row` of its shape, which a
    captured graph freezes as the op does. The call takes fp32 logits whose rows start
    32-byte aligned, a multiple of 8 columns wide and at most 65536 (the scorer's output at
    the Mistral Large 4 width is), and refuses any other.

    The op it reproduces is `indexer_topk_decode` (the catalog entry and
    torch.ops.trtllm.indexer_topk_decode) on its split path, which it takes for every
    Mistral Large 4 call: fp32 logits at least 4096 and under 200000 columns wide. The call
    refuses a shape the op would run as one block per row (`blocks_per_row` 1), whose
    single-block kernel orders scores differently (an insertion sort on float compares, and
    a short row listed without reading its scores).

    How the two compute the same set. The op ranks by the fp32 key above. Its first pass
    bins fp16(x), and every later comparison (radix digits 31..21, 20..10, 9..0 and a radix
    sort of the final bin) is on the full 32-bit key. fp16(x) is a monotone image of the key
    for every score but a negative NaN, so wherever that pass settles a selection it settles
    it as the key would. Each of its blocks keeps its own top_k, or all its columns plus
    pads, and a merge takes the top_k of those: the top_k of the row's columns and the pads,
    since a column a block drops has top_k better ones beside it. This kernel ranks by the
    same key (its digits are bits 31..20, 19..10 and 9..0, no narrower type) over the row's
    columns and the same number of pads, so the two select the same multiset of keys: the
    same columns wherever no two scores share the cut's bit pattern, and among the ones that
    do, each takes as many as the other. A row holding a negative NaN is selected through
    the op's own stages (see the class docstring), fp16 pass included.

    On the tp4 target's decode scorer logits (264 calls: 6 steps of its 44 indexer layers,
    KV lengths about 8200, rows tiled to the batch at 1, 16, 64 and 256 rows) no row is
    selected differently from the op. Cost: one CTA reads its whole row, so the time follows
    the KV length.

    Hard limits (sm_103)
      rows             grid x, one CTA per row                 envelope 256
      columns, end     int32; end clamped to [0, columns]      envelope 4096..65536 (cells at
                                                               4096, 8192, 16384, 32768 and
                                                               65536); is_valid refuses
                                                               an end past the row; at most
                                                               65536 columns (the emulation
                                                               path's column mask)
      logits stride(0) symbolic int64, stride(1) 1, rows 32-byte aligned
      threshold group  4096-entry shared-memory list; a larger group is narrowed over the
                       row in global memory (one pass per digit)
    """

    def __call__(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        *,
        next_n: int,
        compiled: CompiledRadixTopkDecode,
    ) -> torch.Tensor:
        rows, columns = logits.shape
        if compiled.next_n != next_n:
            raise ValueError(f"compiled for next_n={compiled.next_n}, called with {next_n}")
        if indices.shape != (rows, TOP_K) or rows != seq_lens.shape[0] * next_n:
            raise ValueError(
                f"indices must be [{rows}, {TOP_K}] and seq_lens hold {rows} / next_n entries; "
                f"got {tuple(indices.shape)} and {tuple(seq_lens.shape)}"
            )
        if columns > _kernel.MAX_COLUMNS:
            raise ValueError(f"logits rows wider than {_kernel.MAX_COLUMNS} columns are not served")
        # The kernel reads each row in 32-byte vectors from its start.
        if logits.data_ptr() % 32 or logits.stride(0) % 8 or columns % 8 or logits.stride(1) != 1:
            raise ValueError(
                "logits rows must start 32-byte aligned and be a multiple of 8 unit-stride "
                f"columns; got shape {tuple(logits.shape)}, stride {logits.stride()}"
            )
        bp = blocks_per_row(rows, columns, compiled.sm_count)
        if bp < 2:
            raise ValueError(
                f"the trtllm op runs [{rows}, {columns}] as one block per row; that path is not "
                "the one reproduced here"
            )
        if rows:
            compiled.fn(logits, seq_lens, indices, bp)
        return indices

    def reference(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        *,
        next_n: int,
        compiled: CompiledRadixTopkDecode,
    ) -> torch.Tensor:
        """The op's selection, as a fresh `[rows, top_k]` int32 tensor; `indices` is not read.

        Columns ranked by key, a column before a pad at the same key, then by column: one
        answer among the ones the op allows. Only `compiled.sm_count` is read off the handle,
        for the op's blocks per row."""
        rows, columns = logits.shape
        bp = blocks_per_row(rows, columns, compiled.sm_count)
        out = torch.full((rows, TOP_K), -1, dtype=torch.int32, device=logits.device)
        lens = seq_lens.tolist()
        keys = _keys(logits.float())
        for r in range(rows):
            end = max(0, min(columns, lens[r // next_n] - next_n + r % next_n + 1))
            block = end // bp
            last = end - block * (bp - 1)
            pad = (bp - 1) * max(0, TOP_K - block) + max(0, TOP_K - last)
            if not bool((keys[r, :end] > _kernel.NEG_INF_KEY).any()):
                won, _ = _top(keys[r, :end], pad, TOP_K)
            else:
                # A negative NaN: the op's own two launches, block by block, then the merge.
                passed = []
                for j in range(bp):
                    lo, hi = j * block, (end if j == bp - 1 else (j + 1) * block)
                    if hi - lo <= TOP_K:
                        passed.append(torch.arange(lo, hi, device=logits.device))
                    else:
                        picked, _ = _stage(logits[r, lo:hi], keys[r, lo:hi], 0)
                        passed.append(lo + picked)
                cand = torch.cat(passed)
                picked, _ = _stage(logits[r, cand], keys[r, cand], pad)
                won = cand[picked]
            won = won.sort().values
            out[r, : won.numel()] = won.to(torch.int32)
        return out

    def is_valid(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        *,
        next_n: int,
        compiled: CompiledRadixTopkDecode,
    ) -> None:
        # The kernel clamps a causal end past the row to the row's width and answers for the
        # truncated row; the op reads past it. The lengths are device data that a captured
        # call reads only at replay, so the check needs an eager call.
        if seq_lens.numel() and not torch.cuda.is_current_stream_capturing():
            widest = int(seq_lens.max())
            assert widest <= logits.shape[1], (
                f"a causal end of {widest} exceeds the {logits.shape[1]}-column logits"
            )


cute_dsl_radix_topk_decode = CuteDslRadixTopkDecode()


@dataclass(frozen=True)
class CompiledSplitRadixTopkDecode:
    """Precompiled partition and merge launches; scratch is supplied by the caller."""

    stage1: object
    stage2: object
    device_id: int
    next_n: int
    partitions: int
    use_pdl: bool
    sm_count: int


_SPLIT_COMPILED: dict[tuple[int, int, int, bool], CompiledSplitRadixTopkDecode] = {}


def build_split(
    device_id: int, *, next_n: int = 4, partitions: int = 10, use_pdl: bool = True
) -> CompiledSplitRadixTopkDecode:
    """Compile a two-launch selector, before capture, for SM107 component experiments.

    Rows, width and row stride are symbolic. The selector retains the long-row native
    split's ten logical partition boundaries and sentinel padding.
    """
    if next_n not in (1, 4) or partitions != 10:
        raise ValueError("split radix requires next_n 1/4 and exactly 10 native partitions")
    key = (int(device_id), int(next_n), int(partitions), bool(use_pdl))
    if key in _SPLIT_COMPILED:
        return _SPLIT_COMPILED[key]
    with torch.cuda.device(device_id):
        rows, cols, batch, size = (cute.sym_int() for _ in range(4))
        stride = cute.sym_int64()
        fake_logits = make_fake_tensor(
            cutlass.Float32, (rows, cols), stride=(stride, 1), assumed_align=4
        )
        fake_lens = make_fake_compact_tensor(cutlass.Int32, (batch,), assumed_align=4)
        fake_aux_i = make_fake_compact_tensor(cutlass.Int32, (size,), assumed_align=4)
        fake_aux_l = make_fake_compact_tensor(cutlass.Float32, (size,), assumed_align=4)
        fake_out = make_fake_compact_tensor(
            cutlass.Int32, (rows, TOP_K), stride_order=(1, 0), assumed_align=4
        )
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = _kernel.SplitRadixTopkDecodeKernel(next_n, partitions, use_pdl)
        stage1 = cute.compile(
            kernel.split,
            fake_logits,
            fake_lens,
            fake_aux_i,
            fake_aux_l,
            stream,
            options="--enable-tvm-ffi",
        )
        stage2 = cute.compile(
            kernel.merge,
            fake_aux_i,
            fake_aux_l,
            fake_out,
            stream,
            options="--enable-tvm-ffi",
        )
        sm_count = torch.cuda.get_device_properties(device_id).multi_processor_count
        handle = CompiledSplitRadixTopkDecode(stage1, stage2, *key, int(sm_count))
        logits = torch.zeros((next_n, 4096), dtype=torch.float32, device="cuda")
        lens = torch.full((1,), 4096, dtype=torch.int32, device="cuda")
        out = torch.empty((next_n, TOP_K), dtype=torch.int32, device="cuda")
        aux_i = torch.empty(next_n * partitions * TOP_K, dtype=torch.int32, device="cuda")
        aux_l = torch.empty_like(aux_i, dtype=torch.float32)
        stage1(logits, lens, aux_i, aux_l)
        stage2(aux_i, aux_l, out)
        torch.cuda.synchronize()
    _SPLIT_COMPILED[key] = handle
    return handle


#: Supported CTAs per cluster for `cluster_radix_topk_decode`.
CLUSTER_SIZES = (2, 4, 8, 16, 32)
#: Portable CUDA cluster limit; larger sizes require non-portable opt-in and device verification.
_PORTABLE_MAX_CLUSTER = 8
#: Default verification cluster ladder ordered from portable baseline to SM107 wide CGA.
DEFAULT_VERIFY_CLUSTERS = (8, 16, 32)
#: Minimum scorer width (columns) per cluster size so each CTA scans enough vectors to amortize
#: DSMEM histogram reduction: C=8 through 64K, C=16 above 64K, and C=32 from 256K upward when
#: supported by the device.
_CLUSTER_MIN_WIDTH = {8: 20480, 16: 65537, 32: 262144}


def select_cluster_size(
    rows: int,
    width: int,
    sm_count: int,
    available_clusters: tuple[int, ...] = DEFAULT_VERIFY_CLUSTERS,
) -> int:
    """Pick the largest cluster size in `available_clusters` that keeps `rows * cluster` within
    one SM wave and gives each CTA enough columns to amortize DSMEM histogram reduction.

    Pure function of host-known static shapes `(rows, width, sm_count)`, safe for CUDA-graph
    capture without inspecting device lengths.
    """
    candidates = tuple(sorted({int(c) for c in available_clusters if int(c) in CLUSTER_SIZES}))
    if not candidates:
        raise ValueError("available_clusters must contain at least one valid cluster size")
    if rows <= 0:
        return candidates[0]
    best = candidates[0]
    for cluster in candidates:
        if rows * cluster <= sm_count and width >= _CLUSTER_MIN_WIDTH.get(cluster, 0):
            best = cluster
    return best


def reachable_cluster_sizes(
    row_counts: tuple[int, ...],
    width: int,
    sm_count: int,
    candidates: tuple[int, ...] = DEFAULT_VERIFY_CLUSTERS,
) -> tuple[int, ...]:
    """Sorted cluster sizes up to the widest size `select_cluster_size` can pick across
    `row_counts` for `(width, sm_count)`, preserving smaller fallback sizes."""
    ordered = tuple(sorted({int(c) for c in candidates if int(c) in CLUSTER_SIZES}))
    if not ordered:
        raise ValueError("candidates must contain at least one valid cluster size")
    if not row_counts:
        return (ordered[0],)
    max_needed = max(select_cluster_size(r, width, sm_count, ordered) for r in row_counts)
    return tuple(c for c in ordered if c <= max_needed)


def _split_call_runs(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    aux_indices: torch.Tensor,
    aux_logits: torch.Tensor,
    *,
    next_n: int,
    device_id: int,
    sm_count: int,
) -> bool:
    """Validate a split or cluster selector call; whether its kernels run it, else the op.

    The kernels serve the shapes the op splits into ten blocks per row, with rows 32-byte
    aligned and a width that is a multiple of 8, which their vector reads require."""
    if logits.ndim != 2 or logits.dtype != torch.float32 or not logits.is_cuda:
        raise ValueError("logits must be a two-dimensional CUDA float32 tensor")
    rows, width = logits.shape
    if not 4096 <= width <= 1048584 or rows > 64:
        raise ValueError("split radix requires width 4096..1048584 and at most 64 rows")
    if logits.stride(1) != 1 or logits.stride(0) < width:
        raise ValueError("logits must have unit column stride and non-overlapping rows")
    if seq_lens.shape != (rows // next_n,) or rows % next_n:
        raise ValueError("seq_lens must contain one length per request")
    if indices.shape != (rows, TOP_K):
        raise ValueError(f"indices must have shape ({rows}, {TOP_K})")
    native_partitions = blocks_per_row(rows, width, sm_count)
    needed = rows * native_partitions * TOP_K
    for name, tensor, dtype in (
        ("seq_lens", seq_lens, torch.int32),
        ("indices", indices, torch.int32),
        ("aux_indices", aux_indices, torch.int32),
        ("aux_logits", aux_logits, torch.float32),
    ):
        if tensor.dtype != dtype or tensor.device != logits.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous {dtype} on {logits.device}")
    if logits.device.index != device_id:
        raise ValueError("compiled selector and logits must be on the same CUDA device")
    if aux_indices.numel() < needed or aux_logits.numel() < needed:
        raise ValueError(f"scratch must hold at least {needed} elements")
    vector_rows = logits.data_ptr() % 32 == 0 and logits.stride(0) % 8 == 0 and width % 8 == 0
    return native_partitions == _MAX_BLOCKS_PER_ROW and vector_rows


def split_radix_topk_decode(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    aux_indices: torch.Tensor,
    aux_logits: torch.Tensor,
    *,
    compiled: CompiledSplitRadixTopkDecode,
) -> torch.Tensor:
    """Select K=2048 from fp32 ``logits[batch * next_n, width]`` into int32 ``indices``.

    ``seq_lens[batch]`` is int32 and read on device for each replay. Both contiguous scratch
    buffers hold at least ``rows * partitions * 2048`` elements; their initial contents are
    ignored. Unit-stride logits may have padded rows and nonzero storage offsets. This
    selector uses the native op when its shape would select fewer than ten partitions,
    preserving that shape's sentinel and nonfinite behavior, and when its rows are not
    32-byte aligned with a width that is a multiple of 8, which its vector reads require.
    """
    runs = _split_call_runs(
        logits, seq_lens, indices, aux_indices, aux_logits,
        next_n=compiled.next_n, device_id=compiled.device_id, sm_count=compiled.sm_count,
    )  # fmt: skip
    if logits.shape[0]:
        if not runs:
            return indexer_topk_decode(
                logits, seq_lens, indices, compiled.next_n, TOP_K, 1, aux_indices, aux_logits
            )
        flat_i, flat_l = aux_indices.view(-1), aux_logits.view(-1)
        compiled.stage1(logits, seq_lens, flat_i, flat_l)
        compiled.stage2(flat_i, flat_l, indices)
    return indices


@dataclass(frozen=True)
class CompiledClusterRadixTopkDecode:
    """The precompiled one-launch selector; scratch is supplied by the caller."""

    launch: object
    device_id: int
    next_n: int
    cluster: int
    threads: int
    unroll: int
    pipe: bool
    min_blocks: int
    use_pdl: bool
    sm_count: int


_CLUSTER_COMPILED: dict[tuple, CompiledClusterRadixTopkDecode] = {}


def _probe_cubin_cluster_support(
    cubin: bytes | None,
    device_id: int,
    cluster: int,
    threads: int,
) -> bool | None:
    """Check via `cuOccupancyMaxActiveClusters` whether the compiled cubin can launch with
    `cluster` CTAs per cluster on `device_id`. Returns `None` if probing is unavailable."""
    if not cubin:
        return None
    from cuda.bindings import driver

    lib = None
    try:
        err, dev = driver.cuDeviceGet(device_id)
        if err != driver.CUresult.CUDA_SUCCESS:
            return None
        err, lib = driver.cuLibraryLoadData(cubin, None, None, 0, None, None, 0)
        if err != driver.CUresult.CUDA_SUCCESS:
            return None
        err, kernels = driver.cuLibraryEnumerateKernels(1, lib)
        if err != driver.CUresult.CUDA_SUCCESS or not kernels:
            return None
        err, fn = driver.cuKernelGetFunction(kernels[0])
        if err != driver.CUresult.CUDA_SUCCESS:
            return None
        err, max_smem = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            dev,
        )
        if err != driver.CUresult.CUDA_SUCCESS:
            return None
        driver.cuFuncSetAttribute(
            fn,
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            max_smem,
        )
        err_attr = driver.cuFuncSetAttribute(
            fn,
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
            1,
        )
        if err_attr[0] != driver.CUresult.CUDA_SUCCESS:
            return False
        smem_bytes = _cluster_kernel.cluster_kernel_smem_bytes(cluster, threads)
        cfg = driver.CUlaunchConfig()
        cfg.gridDimX = cluster
        cfg.gridDimY = 1
        cfg.gridDimZ = 1
        cfg.blockDimX = threads
        cfg.blockDimY = 1
        cfg.blockDimZ = 1
        cfg.sharedMemBytes = smem_bytes
        attr = driver.CUlaunchAttribute()
        attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        val = driver.CUlaunchAttributeValue()
        val.clusterDim.x = cluster
        val.clusterDim.y = 1
        val.clusterDim.z = 1
        attr.value = val
        cfg.attrs = [attr]
        cfg.numAttrs = 1
        err, active = driver.cuOccupancyMaxActiveClusters(fn, cfg)
        if err != driver.CUresult.CUDA_SUCCESS:
            return False
        return int(active) > 0
    finally:
        if lib is not None:
            driver.cuLibraryUnload(lib)


def build_cluster(
    device_id: int,
    *,
    next_n: int = 4,
    cluster: int = 8,
    threads: int = 512,
    unroll: int = 2,
    pipe: bool = True,
    min_blocks: int = 0,
    use_pdl: bool = True,
) -> CompiledClusterRadixTopkDecode:
    """Compile `cluster_radix_topk_decode`'s launch before capture: `cluster` CTAs of `threads`
    threads per row, `unroll` vectors per thread per batch, `pipe` and `min_blocks` as
    `_cluster_radix_topk_decode.ClusterRadixTopkDecodeKernel` takes them. Rows, width and row
    stride are symbolic."""
    if next_n not in (1, 4):
        raise ValueError("cluster radix requires next_n 1 or 4")
    if (
        cluster not in CLUSTER_SIZES
        or threads not in (256, 512, 1024)
        or unroll not in (1, 2, 4, 8)
    ):
        raise ValueError(
            "cluster radix requires cluster 2/4/8/16/32, threads 256/512/1024, unroll 1/2/4/8"
        )
    if min_blocks not in (0, 1, 2, 4):
        raise ValueError("cluster radix requires min_blocks 0, 1, 2 or 4")
    key = (
        int(device_id), int(next_n), int(cluster), int(threads), int(unroll), bool(pipe),
        int(min_blocks), bool(use_pdl),
    )  # fmt: skip
    if key in _CLUSTER_COMPILED:
        return _CLUSTER_COMPILED[key]
    with torch.cuda.device(device_id):
        rows, cols, batch, size = (cute.sym_int() for _ in range(4))
        stride = cute.sym_int64()
        fake_logits = make_fake_tensor(
            cutlass.Float32, (rows, cols), stride=(stride, 1), assumed_align=4
        )
        fake_lens = make_fake_compact_tensor(cutlass.Int32, (batch,), assumed_align=4)
        fake_aux_i = make_fake_compact_tensor(cutlass.Int32, (size,), assumed_align=4)
        fake_aux_l = make_fake_compact_tensor(cutlass.Float32, (size,), assumed_align=4)
        fake_out = make_fake_compact_tensor(
            cutlass.Int32, (rows, TOP_K), stride_order=(1, 0), assumed_align=4
        )
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = _cluster_kernel.ClusterRadixTopkDecodeKernel(
            next_n, cluster, threads, unroll, use_pdl, pipe=pipe, min_blocks=min_blocks
        )
        if cluster > _PORTABLE_MAX_CLUSTER:
            with tempfile.TemporaryDirectory() as dump_dir:
                launch = cute.compile(
                    kernel,
                    fake_logits,
                    fake_lens,
                    fake_aux_i,
                    fake_aux_l,
                    fake_out,
                    stream,
                    options=f"--enable-tvm-ffi --dump-dir={dump_dir} --keep-cubin",
                )
            cubin = getattr(getattr(launch, "artifacts", None), "CUBIN", None)
            supported = _probe_cubin_cluster_support(
                cubin, int(device_id), int(cluster), int(threads)
            )
            if supported is False:
                raise RuntimeError(
                    f"cluster size {cluster} (threads={threads}) is not supported on cuda:{device_id}"
                )
        else:
            launch = cute.compile(
                kernel,
                fake_logits,
                fake_lens,
                fake_aux_i,
                fake_aux_l,
                fake_out,
                stream,
                options="--enable-tvm-ffi",
            )
        sm_count = torch.cuda.get_device_properties(device_id).multi_processor_count
        handle = CompiledClusterRadixTopkDecode(launch, *key, int(sm_count))
        logits = torch.zeros((next_n, 4096), dtype=torch.float32, device="cuda")
        lens = torch.full((1,), 4096, dtype=torch.int32, device="cuda")
        out = torch.empty((next_n, TOP_K), dtype=torch.int32, device="cuda")
        aux_i = torch.empty(next_n * _MAX_BLOCKS_PER_ROW * TOP_K, dtype=torch.int32, device="cuda")
        aux_l = torch.empty_like(aux_i, dtype=torch.float32)
        launch(logits, lens, aux_i, aux_l, out)
        torch.cuda.synchronize()
    _CLUSTER_COMPILED[key] = handle
    return handle


def build_cluster_variants(
    device_id: int,
    *,
    next_n: int = 4,
    clusters: tuple[int, ...] = DEFAULT_VERIFY_CLUSTERS,
    threads: int = 512,
    unroll: int = 2,
    pipe: bool = True,
    min_blocks: int = 0,
    use_pdl: bool = True,
) -> dict[int, CompiledClusterRadixTopkDecode]:
    """Compile each requested cluster size supported on `device_id`, keyed by cluster size.

    Portable sizes (`<= 8`) propagate compilation/launch errors; non-portable sizes (`> 8`) are
    built in ascending order and stop cleanly at the first size the device/driver rejects.
    """
    ordered = tuple(sorted({int(c) for c in clusters}))
    if not ordered or any(c not in CLUSTER_SIZES for c in ordered):
        raise ValueError(f"clusters must be non-empty subset of {CLUSTER_SIZES}, got {clusters}")
    variants: dict[int, CompiledClusterRadixTopkDecode] = {}
    for cluster in ordered:
        if cluster <= _PORTABLE_MAX_CLUSTER:
            variants[cluster] = build_cluster(
                device_id,
                next_n=next_n,
                cluster=cluster,
                threads=threads,
                unroll=unroll,
                pipe=pipe,
                min_blocks=min_blocks,
                use_pdl=use_pdl,
            )
            continue
        try:
            variants[cluster] = build_cluster(
                device_id,
                next_n=next_n,
                cluster=cluster,
                threads=threads,
                unroll=unroll,
                pipe=pipe,
                min_blocks=min_blocks,
                use_pdl=use_pdl,
            )
        except RuntimeError as exc:
            logger.info(
                f"cluster_radix_topk_decode: skipping non-portable cluster={cluster} on "
                f"cuda:{device_id}: {exc}"
            )
            break
    if not variants:
        raise RuntimeError(
            f"no requested cluster size in {ordered} could be built on cuda:{device_id}"
        )
    return variants


def resolve_compiled_cluster(
    compiled: CompiledClusterRadixTopkDecode | Mapping[int, CompiledClusterRadixTopkDecode],
    rows: int,
    width: int,
) -> CompiledClusterRadixTopkDecode:
    """Resolve one compiled cluster launcher for `(rows, width)` from a handle or size map."""
    if isinstance(compiled, CompiledClusterRadixTopkDecode):
        return compiled
    if not compiled:
        raise ValueError("compiled cluster selector mapping must not be empty")
    first = next(iter(compiled.values()))
    chosen = select_cluster_size(rows, width, first.sm_count, tuple(compiled.keys()))
    return compiled[chosen]


def cluster_radix_topk_decode(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    aux_indices: torch.Tensor,
    aux_logits: torch.Tensor,
    *,
    compiled: CompiledClusterRadixTopkDecode | Mapping[int, CompiledClusterRadixTopkDecode],
) -> torch.Tensor:
    """`split_radix_topk_decode`'s selection and contract in one cluster launch per call.

    The same set per row (ties at the k-th key taken in no particular order, columns before
    the op's pads), valid columns first in no particular order, then -1. The scratch buffers
    are read only for rows holding a negative NaN, which take the split selector's code; the
    native op serves the shapes `split_radix_topk_decode` gives it. `compiled` may be one
    handle or a `{cluster_size: handle}` map from `build_cluster_variants`, selected per call
    by `(rows, width)` via `select_cluster_size`.
    """
    selected = resolve_compiled_cluster(compiled, logits.shape[0], logits.shape[1])
    runs = _split_call_runs(
        logits, seq_lens, indices, aux_indices, aux_logits,
        next_n=selected.next_n, device_id=selected.device_id, sm_count=selected.sm_count,
    )  # fmt: skip
    if logits.shape[0]:
        if not runs:
            return indexer_topk_decode(
                logits, seq_lens, indices, selected.next_n, TOP_K, 1, aux_indices, aux_logits
            )
        selected.launch(logits, seq_lens, aux_indices.view(-1), aux_logits.view(-1), indices)
    return indices


def validate_split_radix_topk_lengths(
    logits: torch.Tensor, seq_lens: torch.Tensor, *, next_n: int
) -> None:
    """Eager test guard for device metadata; never call from the production launch path.

    A legal request contains the preceding ``next_n - 1`` tokens, with its last causal end
    within the score row. CUDA graph capture must use already validated legal metadata.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("validate split radix lengths before CUDA graph capture")
    if seq_lens.numel():
        shortest, longest = torch.aminmax(seq_lens)
        if int(shortest) < next_n - 1 or int(longest) > logits.shape[1]:
            raise ValueError("seq_lens must be within [next_n - 1, logits width]")
