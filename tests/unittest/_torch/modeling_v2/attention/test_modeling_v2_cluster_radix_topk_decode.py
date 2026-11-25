# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 tests of the one-launch cluster radix selector against the split selector's contract."""

from unittest.mock import patch

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("cluster radix requires SM107 validation", allow_module_level=True)

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_radix_topk_decode import (  # noqa: E402
    TOP_K,
    build_cluster,
    build_split,
    cluster_radix_topk_decode,
    split_radix_topk_decode,
)

#: key(-FLT_MAX), the op's pad entries.
_PAD_KEY = 0xFF7FFFFF
#: build_cluster arguments: the dep4 target's build, narrower and wider clusters, the unpipelined
#: passes and a register cap of two CTAs per SM.
_CONFIGS = [
    dict(),
    dict(cluster=2, threads=512, unroll=2, pipe=False),
    dict(cluster=4, threads=512, unroll=4),
    dict(cluster=16, threads=256, unroll=1),
    dict(cluster=4, threads=1024, unroll=8, pipe=False),
    dict(cluster=8, threads=512, unroll=2, min_blocks=2),
]


def _buffers(batch, width, next_n=4):
    rows = batch * next_n
    stride = (width + 255) // 256 * 256
    storage = torch.empty(rows * stride + 8, dtype=torch.float32, device="cuda")
    logits = storage[8:].as_strided((rows, width), (stride, 1))
    output_storage = torch.full((rows + 1, TOP_K), -7, dtype=torch.int32, device="cuda")
    output = output_storage[1:]
    aux_i = torch.full((rows, 10, TOP_K), -7, dtype=torch.int32, device="cuda")
    aux_l = torch.full((rows, 10, TOP_K), float("nan"), dtype=torch.float32, device="cuda")
    return logits, output, aux_i, aux_l


def _end(lengths, row, next_n=4):
    return lengths[row // next_n] - next_n + row % next_n + 1


def _fill(logits, lengths, next_n, seed, kind):
    """Scores of each row's causal window: a permutation (`ordinary`), seven values (`ties`),
    a permutation lifted at the ten blocks' edges (`boundaries`), or one value's neighbors in
    the last 23 mantissa bits (`dense`: a first-digit bin far wider than the gather list)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits.fill_(1.0e9)
    for row in range(logits.shape[0]):
        end = _end(lengths, row, next_n)
        scores = torch.randperm(end, generator=generator, device="cuda").float()
        if kind == "ties":
            scores.remainder_(7)
        elif kind == "boundaries":
            for part in range(1, 10):
                boundary = end // 10 * part
                if boundary < end:
                    scores[max(0, boundary - 1) : boundary + 1] += end
        elif kind == "dense":
            bits = 0x3F800000 + torch.randperm(end, generator=generator, device="cuda") % (1 << 20)
            scores = bits.to(torch.int32).view(torch.float32)
        logits[row, :end] = scores


def _check(logits, lengths, output, next_n=4):
    """Each row: min(top_k, end) distinct in-window columns first, then -1; their scores are
    the window's top_k scores."""
    prefix = output.as_strided((TOP_K,), (1,), storage_offset=0)
    assert bool((prefix == -7).all()), "the output view must not overwrite its leading guard"
    for row in range(logits.shape[0]):
        end = _end(lengths, row, next_n)
        indices = output[row]
        valid = indices[indices >= 0].long()
        expected_count = min(TOP_K, end)
        assert valid.numel() == expected_count
        assert bool((indices[:expected_count] >= 0).all()), "valid columns must form a prefix"
        assert bool((indices[expected_count:] == -1).all()), "padding must be a suffix"
        assert valid.unique().numel() == valid.numel()
        assert bool((valid < end).all())
        selected = logits[row, valid].sort().values
        reference = logits[row, :end].topk(expected_count).values.sort().values
        torch.testing.assert_close(selected, reference, rtol=0, atol=0)


def _keys(logits, row, selection):
    """The sorted keys a row's selection takes: its columns', then key(-FLT_MAX) per -1."""
    col = selection[selection >= 0].long()
    assert col.unique().numel() == col.numel()
    bits = logits[row, col].view(torch.int32).long() & 0xFFFFFFFF
    keys = torch.where(bits >= 0x80000000, bits, (~bits) & 0x7FFFFFFF)
    pads = torch.full((TOP_K - col.numel(),), _PAD_KEY, device="cuda", dtype=torch.int64)
    return torch.cat((keys, pads)).sort().values


def _same_sets(logits, lengths, output, reference, next_n=4):
    for row in range(logits.shape[0]):
        end = _end(lengths, row, next_n)
        count = int((output[row] >= 0).sum())
        assert bool((output[row, :count] >= 0).all()) and bool((output[row, count:] == -1).all())
        assert bool((output[row, :count] < end).all())
        assert torch.equal(_keys(logits, row, output[row]), _keys(logits, row, reference[row]))


def _dep4_lengths(width):
    """Sixteen KV lengths a DSpark scorer row of `width` columns holds: the shortest legal
    request, the top-k edge, the ten-block edge, the slice edges, half the row and the full
    row."""
    return [
        *(3, 4, 5, 2047, 2048, 2049, 2051, 20479, 20480, 20481, 32769),
        *(width // 2 + 1, width - 9, width - 8, width - 1, width),
    ]


@pytest.mark.parametrize("config", _CONFIGS)
@pytest.mark.parametrize("seed,kind", [(0, "ordinary"), (17, "ties"), (42, "boundaries")])
@pytest.mark.parametrize("batch,width", [(1, 32776), (4, 65536), (16, 1048584)])
def test_cluster_radix_causal_selection(config, seed, kind, batch, width):
    compiled = build_cluster(torch.cuda.current_device(), **config)
    logits, output, aux_i, aux_l = _buffers(batch, width)
    edges = _dep4_lengths(width)
    for offset in range(0, len(edges), batch):
        lengths = [edges[(offset + b) % len(edges)] for b in range(batch)]
        lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        _fill(logits, lengths, 4, seed + offset, kind)
        output.fill_(-7)
        assert (
            cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
            is output
        )
        _check(logits, lengths, output)


@pytest.mark.parametrize("config", _CONFIGS[:2])
@pytest.mark.parametrize("length", [20481, 300001, 1048584])
def test_cluster_radix_wide_threshold_bin(config, length):
    """Scores sharing their first 12 key bits: the first digit's threshold bin holds every
    column, so the select narrows over the slices (and gathers or claims the tie group)."""
    compiled = build_cluster(torch.cuda.current_device(), **config)
    logits, output, aux_i, aux_l = _buffers(1, 1048584)
    for kind in ("dense", "ties"):
        lengths = [length]
        lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        _fill(logits, lengths, 4, 5, kind)
        output.fill_(-7)
        cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
        _check(logits, lengths, output)


@pytest.mark.parametrize("seed,kind", [(0, "ordinary"), (17, "ties"), (3, "dense")])
def test_cluster_radix_matches_split_selector(seed, kind):
    """dep4's largest graph batch (64 rows) at its 1M width: the split selector's sets."""
    device = torch.cuda.current_device()
    compiled = build_cluster(device)
    split = build_split(device)
    width = 1048584
    logits, output, aux_i, aux_l = _buffers(16, width)
    lengths = _dep4_lengths(width)
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    _fill(logits, lengths, 4, seed, kind)
    reference = torch.empty_like(output)
    split_radix_topk_decode(logits, lens, reference, aux_i, aux_l, compiled=split)
    output.fill_(-7)
    cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
    _same_sets(logits, lengths, output, reference)


@pytest.mark.parametrize("length,seed", [(5, 0), (2049, 17), (65536, 42), (400003, 7)])
def test_cluster_radix_nonfinite_semantics(length, seed):
    """NaN, inf, -inf, -FLT_MAX and -0.0 in the windows: a row holding a negative NaN takes the
    split selector's code, the others the cluster select; both give the split selector's sets
    (key for key, -FLT_MAX columns and pads alike)."""
    device = torch.cuda.current_device()
    compiled = build_cluster(device)
    split = build_split(device)
    logits, output, aux_i, aux_l = _buffers(2, 1048584)
    lengths = [length, length]
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    _fill(logits, lengths, 4, seed, "ordinary")
    patterns = torch.tensor(
        [0x7FC00000, 0x7F800000, -0x400000, -0x800000, -0x800001, -0x80000000],
        dtype=torch.int32,
        device="cuda",
    ).view(torch.float32)
    for row in range(8):
        end = _end(lengths, row)
        positions = torch.arange(row % 4, end, 19, device="cuda")
        values = patterns[torch.arange(positions.numel(), device="cuda") % 6]
        if row >= 4:
            # The second request holds no negative NaN.
            values = torch.where(values.view(torch.int32) == -0x400000, 0.5, values)
        logits[row, positions] = values
    reference = torch.empty_like(output)
    split_radix_topk_decode(logits, lens, reference, aux_i, aux_l, compiled=split)
    aux_i.fill_(-7)
    aux_l.fill_(float("nan"))
    output.fill_(-7)
    cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
    _same_sets(logits, lengths, output, reference)


def test_cluster_radix_selector_contract():
    """The output layout the dep4 sparse MLA decode reads (its per-row `topk_length`), on a
    graph batch of 16 with padding dummies: for finite scores, exactly min(max(end, 0), 2048)
    distinct columns of [0, end) first, -1 after them, whatever the columns past `end` hold; a
    dummy (kv_len next_n) selects {0..t} in its row t. -FLT_MAX columns count as columns. A row
    holding a negative NaN (the split selector's code) keeps the layout: distinct in-window
    columns first, then -1."""
    compiled = build_cluster(torch.cuda.current_device())
    width = 1048584
    logits, output, aux_i, aux_l = _buffers(16, width)
    lengths = [3, 4, 4, 4, 5, 1000, 2048, 2051, 20481, 133337, 300001, 777777, width, 4, 4, 4]
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(9)
    logits.fill_(float("nan"))
    neg_max = -torch.finfo(torch.float32).max
    for row in range(64):
        end = _end(lengths, row)
        window = torch.randn(end, generator=generator, device="cuda")
        window[::97] = neg_max
        logits[row, :end] = window
        logits[row, end : end + 64] = torch.inf
    # Request 10 (rows 40..43) holds a negative NaN: the split selector's code serves it.
    nan_rows = range(40, 44)
    for row in nan_rows:
        logits[row, 5] = torch.tensor([-0x400000], dtype=torch.int32).view(torch.float32).item()
    output.fill_(-7)
    cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
    for row in range(64):
        end = _end(lengths, row)
        selection = output[row]
        count = int((selection >= 0).sum())
        assert bool((selection[:count] >= 0).all()), "valid columns must form a prefix"
        assert bool((selection[count:] == -1).all()), "-1 must fill the rest"
        valid = selection[:count].long()
        assert valid.unique().numel() == count
        assert bool((valid < max(end, 0)).all())
        if row in nan_rows:
            continue
        assert count == min(max(end, 0), TOP_K)
        if lengths[row // 4] == 4:
            assert sorted(valid.tolist()) == list(range(row % 4 + 1))


def test_cluster_radix_graph_replay():
    """dep4's largest graph batch, 16 requests (64 rows) at 1048584 columns, captured once and
    replayed over new lengths and scores."""
    compiled = build_cluster(torch.cuda.current_device())
    width = 1048584
    logits, output, aux_i, aux_l = _buffers(16, width)
    edges = _dep4_lengths(width)
    lens = torch.tensor(edges, dtype=torch.int32, device="cuda")
    _fill(logits, edges, 4, 0, "ordinary")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
    for replay in range(4):
        lengths = edges[replay + 1 :] + edges[: replay + 1]
        _fill(logits, lengths, 4, replay, ("ordinary", "ties", "boundaries", "dense")[replay])
        lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        output.fill_(-7)
        graph.replay()
        _check(logits, lengths, output)


@pytest.mark.parametrize("layout", ["unaligned_rows", "unaligned_width", "eight_blocks"])
def test_cluster_radix_native_shapes(layout):
    """The shapes the split selector leaves to the op take the op here too."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.indexer_topk_decode import (
        indexer_topk_decode as native_topk,
    )

    compiled = build_cluster(torch.cuda.current_device())
    _, output, aux_i, aux_l = _buffers(1, 65536)
    storage = torch.empty(4 * 65536 + 9, dtype=torch.float32, device="cuda")
    offset = 1 if layout == "unaligned_rows" else 8
    width = {"unaligned_rows": 65536, "unaligned_width": 65535, "eight_blocks": 16384}[layout]
    logits = storage[offset:].as_strided((4, width), (65536, 1))
    lengths = [width if layout != "eight_blocks" else 20]
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    _fill(logits, lengths, 4, 42, "boundaries")
    target = (
        "tensorrt_llm._torch._experimental.modeling_v2.catalog.attention."
        "cute_dsl_radix_topk_decode.indexer_topk_decode"
    )
    with patch(target, wraps=native_topk) as fallback:
        cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
        fallback.assert_called_once()
    _check(logits, lengths, output)


def test_cluster_radix_rejects_invalid_inputs():
    device = torch.cuda.current_device()
    for kwargs in (
        {"cluster": 32},
        {"threads": 128},
        {"unroll": 3},
        {"next_n": 2},
        {"min_blocks": 3},
    ):
        with pytest.raises(ValueError, match="cluster radix requires"):
            build_cluster(device, **kwargs)
    compiled = build_cluster(device)
    logits, output, aux_i, aux_l = _buffers(1, 16384)
    lens = torch.tensor([16384], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="float32"):
        cluster_radix_topk_decode(logits.half(), lens, output, aux_i, aux_l, compiled=compiled)
    with pytest.raises(ValueError, match="scratch"):
        cluster_radix_topk_decode(
            logits, lens, output, aux_i.flatten()[: 4 * 8 * TOP_K - 1], aux_l, compiled=compiled
        )
    logits, output, aux_i, aux_l = _buffers(17, 16384)
    lens = torch.full((17,), 16384, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="at most 64 rows"):
        cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
