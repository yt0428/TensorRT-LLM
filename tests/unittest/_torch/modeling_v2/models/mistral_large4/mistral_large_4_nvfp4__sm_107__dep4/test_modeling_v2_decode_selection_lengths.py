# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The selection contract the Mistral-Large-4-1T-A52B NVFP4 sm_107 dep4 decode step relies on.

`DecodeTarget` passes FlashMLA's sparse decode `topk_length = _selection_lengths(kv_lens,
next_n, index_topk)`, and the decode then reads only a row's first `topk_length` slots. That is
exact only while every decode selector the target dispatches to (`_select_indexer_decode`)
writes, for row `r = b * next_n + t` of a step whose scores are finite, with
`end = kv_lens[b] - next_n + t + 1`:

* C1 count: exactly `min(max(end, 0), index_topk)` valid (>= 0) slots, that is
  `_selection_lengths(kv_lens, next_n, index_topk)[r]`;
* C2 prefix-packed: those valid slots are the row's first entries;
* C3 -1 fill: every entry after them is -1;
* C4 causal and distinct: each valid slot is a column in `[0, end)`, none repeated, whatever
  the scores past `end` hold;
* C5 padding dummies: a CUDA-graph padding dummy (one prompt key plus the draft slots, so
  `kv_len = next_n`) has row t select exactly the `t + 1` keys `[0, t]`.

A selector that breaks any of them would silently drop keys from the decode, so this test
runs the target's own dispatch on one GPU (the selectors built as `_derive_first_step` builds
them) and names the clause that fails. It covers next_n 1 and 4 at padded graph batches 16
and 24 (the split and the native paths at next_n 4) and scorer widths 32768 and 262144,
over padding dummies, sequences ending below, at and just past index_topk, and long ones;
scores random, tied (rounded) and mostly zero (relu), with every column at or past a row's end
set to 1e30 so a selector that reads past it picks it. The engine's dense selection for a
step that skips the indexer, `[0, end)` then -1 (`_get_dense_topk_indices` of
`DSAtrtllmAttentionMetadata`), meets C1-C5 by construction.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__dep4 import (  # noqa: E501
    modeling as M,
)

TOPK = 2048
#: The radix split's most blocks per row (the DSA metadata's arena).
AUX_BLOCKS = 10
#: What a column at or past a row's causal end holds: the best score, if it were read.
POISON = 1.0e30


def _kv_lens(next_n: int, batch: int, width: int) -> list[int]:
    """`batch` sequences: four padding dummies (`next_n` keys), sequences whose rows end below,
    at and just past index_topk, and long ones up to the scorer's width."""
    base = [next_n] * 4 + [
        next_n + 1, 100, TOPK - 1, TOPK, TOPK + 1, TOPK + next_n - 1, TOPK + next_n, 4096,
        12345, width // 3, width - 1, width,
    ]  # fmt: skip
    return (base * 2)[:batch]


def _selector(width: int, dev: int) -> SimpleNamespace:
    """The target's decode selectors at a scorer width, built through the target module's own
    builders as `_derive_first_step` builds them, so the test follows whichever selector the
    target imports."""
    lo, hi = M._TOPK_DECODE_COLUMNS
    ix_topk = None
    if M._TOPK_DECODE_CUTE_DSL and lo <= width <= hi:
        ix_topk = M.build_ix_topk_decode(dev, next_n=1)
    lo, hi = M._TOPK_VERIFY_COLUMNS
    ix_verify = None
    if M._TOPK_DECODE_CUTE_DSL and lo <= width <= hi:
        ix_verify = M.build_ix_topk_verify(
            dev, next_n=4, use_pdl=os.environ.get("TRTLLM_ENABLE_PDL") != "0"
        )
    return SimpleNamespace(_ix_topk=ix_topk, _ix_topk_verify=ix_verify, idx_topk=TOPK)


@pytest.mark.cpu_only
def test_selection_lengths_formula() -> None:
    kv = torch.tensor([3, 4, 5, 2049, 9000], dtype=torch.int32)
    got = M._selection_lengths(kv, 4, TOPK)
    want = [max(0, min(k - 4 + t + 1, TOPK)) for k in kv.tolist() for t in range(4)]
    assert got.tolist() == want and got.dtype == torch.int32
    assert M._selection_lengths(kv, 1, TOPK).tolist() == [3, 4, 5, 2048, 2048]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("width", [32768, 262144])
@pytest.mark.parametrize("batch", [16, 24])
@pytest.mark.parametrize("next_n", [1, 4])
def test_decode_selector_contract(next_n: int, batch: int, width: int) -> None:
    dev = torch.cuda.current_device()
    sel = _selector(width, dev)
    g = torch.Generator(device="cuda").manual_seed(next_n * 1000 + batch + width)
    lens = _kv_lens(next_n, batch, width)
    kv = torch.tensor(lens, dtype=torch.int32, device="cuda")
    rows = batch * next_n
    ends = (kv.unsqueeze(1) - next_n + 1 + torch.arange(next_n, device="cuda")).view(-1)
    past = torch.arange(width, device="cuda").unsqueeze(0) >= ends.unsqueeze(1)
    length = M._selection_lengths(kv, next_n, TOPK)
    col = torch.arange(TOPK, device="cuda")
    head = col.unsqueeze(0) < length.unsqueeze(1)
    for draw in ("random", "tied", "relu"):
        logits = torch.randn(rows, width, generator=g, device="cuda")
        if draw == "tied":
            logits = torch.round(logits)
        elif draw == "relu":
            logits = torch.relu(logits - 1.5)
        logits.masked_fill_(past, POISON)
        idx = torch.full((rows, TOPK), -7, dtype=torch.int32, device="cuda")
        aux_i = torch.empty((rows, AUX_BLOCKS, TOPK), dtype=torch.int32, device="cuda")
        aux_l = torch.empty((rows, AUX_BLOCKS, TOPK), dtype=torch.float32, device="cuda")
        M.MistralLarge4Dep4ModelingV2._select_indexer_decode(
            sel, logits, kv, idx, next_n, aux_i, aux_l
        )
        valid = idx >= 0
        what = f"{draw} scores, next_n {next_n}, batch {batch}, width {width}"
        assert torch.equal(valid.sum(1).int(), length), f"C1 count: {what}"
        assert bool(valid[head].all()), f"C2 prefix-packed: {what}"
        assert bool((idx[~head] == -1).all()), f"C3 -1 fill: {what}"
        for r in range(rows):
            s = idx[r, : int(length[r])]
            assert bool((s < ends[r]).all()), f"C4 causal: row {r}, {what}"
            assert s.unique().numel() == s.numel(), f"C4 distinct: row {r}, {what}"
        for b in range(4):  # the padding dummies
            for t in range(next_n):
                got = idx[b * next_n + t, : t + 1].sort().values
                assert got.tolist() == list(range(t + 1)), f"C5 dummy row {b}/{t}: {what}"
