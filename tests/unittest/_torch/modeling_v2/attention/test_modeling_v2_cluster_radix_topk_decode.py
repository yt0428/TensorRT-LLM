# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests and SM107 GPU tests for the 1-launch cluster radix decode top-k selector."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention import (
    _cluster_radix_topk_decode as cluster_kernel_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention import (
    cute_dsl_radix_topk_decode as topk_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_radix_topk_decode import (
    CLUSTER_SIZES,
    DEFAULT_VERIFY_CLUSTERS,
    TOP_K,
    CompiledClusterRadixTopkDecode,
    blocks_per_row,
    build_cluster,
    build_cluster_variants,
    build_split,
    cluster_radix_topk_decode,
    reachable_cluster_sizes,
    resolve_compiled_cluster,
    select_cluster_size,
    split_radix_topk_decode,
)

_IS_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
_requires_sm107 = pytest.mark.skipif(
    not _IS_SM107, reason="cluster radix GPU tests require SM107 validation"
)

#: key(-FLT_MAX), the op's pad entries.
_PAD_KEY = 0xFF7FFFFF
#: build_cluster arguments covering 2..32 CTAs/cluster, threads, unroll, unpipelined and min_blocks.
_CONFIGS = [
    dict(),
    dict(cluster=2, threads=512, unroll=2, pipe=False),
    dict(cluster=4, threads=512, unroll=4),
    dict(cluster=16, threads=512, unroll=2),
    dict(cluster=16, threads=256, unroll=1),
    dict(cluster=32, threads=512, unroll=2),
    dict(cluster=4, threads=1024, unroll=8, pipe=False),
    dict(cluster=8, threads=512, unroll=2, min_blocks=2),
]


# ============================================================================
# CPU-only routing, occupancy/SMEM footprint, and mathematical contract tests
# ============================================================================


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rows,width,sm_count,available,expected",
    [
        # VR200 (212 SMs), full {8, 16, 32} ladder
        (4, 1048584, 212, (8, 16, 32), 32),
        (4, 780000, 212, (8, 16, 32), 32),
        (4, 262144, 212, (8, 16, 32), 32),
        (4, 131072, 212, (8, 16, 32), 16),
        (4, 65536, 212, (8, 16, 32), 8),
        (4, 32768, 212, (8, 16, 32), 8),
        (4, 20480, 212, (8, 16, 32), 8),
        (8, 1048584, 212, (8, 16, 32), 16),
        (8, 780000, 212, (8, 16, 32), 16),
        (8, 131072, 212, (8, 16, 32), 16),
        (8, 65536, 212, (8, 16, 32), 8),
        (8, 32768, 212, (8, 16, 32), 8),
        (8, 20480, 212, (8, 16, 32), 8),
        (16, 1048584, 212, (8, 16, 32), 8),
        (32, 1048584, 212, (8, 16, 32), 8),
        (64, 1048584, 212, (8, 16, 32), 8),
        # Graceful downgrade when device caps non-portable cluster at 16 or 8
        (4, 780000, 212, (8, 16), 16),
        (4, 262144, 212, (8, 16), 16),
        (4, 65536, 212, (8, 16), 8),
        (4, 780000, 212, (8,), 8),
        # Blackwell (148 SMs), non-portable max 16
        (4, 1048584, 148, (8, 16), 16),
        (8, 1048584, 148, (8, 16), 16),
        (16, 1048584, 148, (8, 16), 8),
        # Zero rows safely returns smallest candidate
        (0, 1048584, 212, (8, 16, 32), 8),
    ],
)
def test_select_cluster_size_policy(rows, width, sm_count, available, expected):
    assert select_cluster_size(rows, width, sm_count, available) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "width,sm_count,candidates,expected",
    [
        (20480, 212, (8, 16, 32), (8,)),
        (32768, 212, (8, 16, 32), (8,)),
        (65536, 212, (8, 16, 32), (8,)),
        (131072, 212, (8, 16, 32), (8, 16)),
        (262144, 212, (8, 16, 32), (8, 16, 32)),
        (780000, 212, (8, 16, 32), (8, 16, 32)),
        (1048584, 212, (8, 16, 32), (8, 16, 32)),
        (1048584, 212, (8, 16), (8, 16)),
        (1048584, 212, (8,), (8,)),
    ],
)
def test_reachable_cluster_sizes(width, sm_count, candidates, expected):
    row_counts = (4, 8, 16, 32, 64)
    assert reachable_cluster_sizes(row_counts, width, sm_count, candidates) == expected


@pytest.mark.cpu_only
def test_resolve_compiled_cluster_single_and_mapping():
    def _fake(cluster: int, sm_count: int = 212) -> CompiledClusterRadixTopkDecode:
        return CompiledClusterRadixTopkDecode(
            launch=object(),
            device_id=0,
            next_n=4,
            cluster=cluster,
            threads=512,
            unroll=2,
            pipe=True,
            min_blocks=0,
            use_pdl=True,
            sm_count=sm_count,
        )

    h8, h16, h32 = _fake(8), _fake(16), _fake(32)
    assert resolve_compiled_cluster(h16, 64, 1048584) is h16

    variants = {8: h8, 16: h16, 32: h32}
    assert resolve_compiled_cluster(variants, 4, 780000) is h32
    assert resolve_compiled_cluster(variants, 4, 131072) is h16
    assert resolve_compiled_cluster(variants, 4, 65536) is h8
    assert resolve_compiled_cluster(variants, 4, 32768) is h8
    assert resolve_compiled_cluster(variants, 8, 780000) is h16
    assert resolve_compiled_cluster(variants, 16, 780000) is h8
    assert resolve_compiled_cluster(variants, 64, 1048584) is h8

    with pytest.raises(ValueError, match="must not be empty"):
        resolve_compiled_cluster({}, 4, 780000)


@pytest.mark.cpu_only
def test_build_cluster_variants_fallback_on_non_portable_rejection():
    built: list[int] = []

    def _fake_build(device_id, *, cluster, **kwargs):
        built.append(cluster)
        if cluster == 32:
            raise RuntimeError("cluster size 32 unsupported")
        return CompiledClusterRadixTopkDecode(
            launch=object(),
            device_id=device_id,
            next_n=4,
            cluster=cluster,
            threads=512,
            unroll=2,
            pipe=True,
            min_blocks=0,
            use_pdl=True,
            sm_count=212,
        )

    with patch.object(topk_mod, "build_cluster", side_effect=_fake_build):
        variants = build_cluster_variants(0, next_n=4, clusters=(8, 16, 32))
    assert built == [8, 16, 32]
    assert set(variants.keys()) == {8, 16}

    # If cluster=16 is rejected, cluster=32 is not attempted.
    built.clear()

    def _reject_16(device_id, *, cluster, **kwargs):
        if cluster >= 16:
            built.append(cluster)
            raise RuntimeError(f"cluster size {cluster} unsupported")
        return _fake_build(device_id, cluster=cluster, **kwargs)

    with patch.object(topk_mod, "build_cluster", side_effect=_reject_16):
        variants = build_cluster_variants(0, next_n=4, clusters=(8, 16, 32))
    assert built == [8, 16]
    assert set(variants.keys()) == {8}

    # Portable cluster=8 failures must propagate rather than be swallowed.
    with (
        patch.object(topk_mod, "build_cluster", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        build_cluster_variants(0, next_n=4, clusters=(8, 16))


@pytest.mark.cpu_only
def test_cluster_kernel_smem_and_validation_cpu():
    assert DEFAULT_VERIFY_CLUSTERS == (8, 16, 32)
    assert CLUSTER_SIZES == (2, 4, 8, 16, 32)
    for cluster in CLUSTER_SIZES:
        for threads in (256, 512, 1024):
            smem = cluster_kernel_mod.cluster_kernel_smem_bytes(cluster, threads)
            # Fits 2 CTAs/SM comfortably inside 228 KiB standard SMEM on SM100/SM107.
            assert 65536 <= smem <= 96 * 1024
            assert smem * 2 <= 228 * 1024
            # Verify kernel constructor accepts every valid power-of-two cluster in 2..32.
            k = cluster_kernel_mod.ClusterRadixTopkDecodeKernel(4, cluster, threads, 2, True)
            assert k.C == cluster and k.T == threads

    for kwargs in (
        {"cluster": 64},
        {"cluster": 12},
        {"threads": 128},
        {"unroll": 3},
        {"next_n": 2},
        {"min_blocks": 3},
    ):
        with pytest.raises(ValueError, match="cluster radix requires"):
            build_cluster(0, **kwargs)


@pytest.mark.cpu_only
@pytest.mark.parametrize("cluster", [8, 16, 32])
@pytest.mark.parametrize("end", [512, 2048, 2049, 16384, 20480, 262144, 780000, 1048584])
def test_cluster_slice_partition_and_ring_bijection_cpu(cluster, end):
    """Verify slice geometry, 8-float alignment, complete disjoint coverage of [0, end), and
    rank-staggered DSMEM ring bijection for cluster=8/16/32 across short through 780K/1M."""
    span = ((end + cluster - 1) // cluster + 7) // 8 * 8
    assert span % 8 == 0
    covered = 0
    prev_hi = 0
    for rank in range(cluster):
        lo = min(rank * span, end)
        hi = min(lo + span, end)
        assert lo % 8 == 0 or lo == end
        assert lo == prev_hi
        assert lo <= hi <= end
        covered += hi - lo
        prev_hi = hi
        # Rank-staggered peer ring visits every peer rank in {0..cluster-1} exactly once.
        peers = [(rank + r) & (cluster - 1) for r in range(cluster)]
        assert peers[0] == rank
        assert sorted(peers) == list(range(cluster))
    assert covered == end
    assert prev_hi == end
    # At rows=4 on VR200 (212 SMs), every width >= 20480 hits the 10-partition split contract.
    if end >= 20480:
        assert blocks_per_row(4, end, 212) == 10


# ============================================================================
# SM107 GPU correctness, edge-case, and CUDA Graph replay tests (VR200 ready)
# ============================================================================


@pytest.fixture(scope="module")
def _custom_ops():
    if _IS_SM107:
        import tensorrt_llm._torch.custom_ops  # noqa: F401


def _build_or_skip_cluster(device: int, **config) -> CompiledClusterRadixTopkDecode:
    cluster = config.get("cluster", 8)
    try:
        return build_cluster(device, **config)
    except RuntimeError as exc:
        if cluster > 8:
            pytest.skip(f"device does not support non-portable cluster={cluster}: {exc}")
        raise


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
    request, short/512, top-k edge, ten-block edge, 16K/256K/780K slices and the full row."""
    return [
        *(3, 4, 512, 2047, 2048, 2049, 16384, 20479, 20480, 20481, 32769),
        *(min(262144, width), min(780000, width), width - 8, width - 1, width),
    ]


@_requires_sm107
@pytest.mark.parametrize("config", _CONFIGS)
@pytest.mark.parametrize("seed,kind", [(0, "ordinary"), (17, "ties"), (42, "boundaries")])
@pytest.mark.parametrize("batch,width", [(1, 32776), (4, 65536), (16, 1048584)])
def test_cluster_radix_causal_selection(_custom_ops, config, seed, kind, batch, width):
    compiled = _build_or_skip_cluster(torch.cuda.current_device(), **config)
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


@_requires_sm107
@pytest.mark.parametrize("cluster", [8, 16, 32])
@pytest.mark.parametrize("length", [512, 2048, 16384, 262144, 780000, 1048584])
@pytest.mark.parametrize("kind", ["ordinary", "ties", "dense"])
def test_cluster_radix_c1_rows4_lengths(_custom_ops, cluster, length, kind):
    """Exact Top-2048 for c=1 (rows=4, next_n=4) across cluster=8/16/32 and 512..780K..1M."""
    compiled = _build_or_skip_cluster(torch.cuda.current_device(), cluster=cluster)
    logits, output, aux_i, aux_l = _buffers(1, 1048584)
    lengths = [length]
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    _fill(logits, lengths, 4, 13 + cluster, kind)
    output.fill_(-7)
    cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
    _check(logits, lengths, output)


@_requires_sm107
@pytest.mark.parametrize("config", _CONFIGS[:2] + [dict(cluster=16), dict(cluster=32)])
@pytest.mark.parametrize("length", [20481, 300001, 780000, 1048584])
def test_cluster_radix_wide_threshold_bin(_custom_ops, config, length):
    """Scores sharing their first 12 key bits: the first digit's threshold bin holds every
    column, so the select narrows over the slices (and gathers or claims the tie group)."""
    compiled = _build_or_skip_cluster(torch.cuda.current_device(), **config)
    logits, output, aux_i, aux_l = _buffers(1, 1048584)
    for kind in ("dense", "ties"):
        lengths = [length]
        lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        _fill(logits, lengths, 4, 5, kind)
        output.fill_(-7)
        cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
        _check(logits, lengths, output)


@_requires_sm107
@pytest.mark.parametrize("seed,kind", [(0, "ordinary"), (17, "ties"), (3, "dense")])
def test_cluster_radix_matches_split_selector(_custom_ops, seed, kind):
    """Largest graph batch (64 rows) at 1M width: exact match with the split selector's sets."""
    device = torch.cuda.current_device()
    compiled = build_cluster_variants(device)
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


@_requires_sm107
@pytest.mark.parametrize("cluster", [8, 16, 32])
@pytest.mark.parametrize("length,seed", [(5, 0), (2049, 17), (65536, 42), (400003, 7)])
def test_cluster_radix_nonfinite_semantics(_custom_ops, cluster, length, seed):
    """NaN, inf, -inf, -FLT_MAX and -0.0 in the windows: a row holding a negative NaN takes the
    split selector's code, the others the cluster select; both give the split selector's sets
    (key for key, -FLT_MAX columns and pads alike)."""
    device = torch.cuda.current_device()
    compiled = _build_or_skip_cluster(device, cluster=cluster)
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


@_requires_sm107
def test_cluster_radix_selector_contract(_custom_ops):
    """The output layout sparse MLA decode reads on a graph batch of 16 with padding dummies:
    for finite scores, exactly min(max(end, 0), 2048) distinct columns of [0, end) first, -1
    after them, whatever the columns past `end` hold; a dummy (kv_len next_n) selects {0..t} in
    its row t. -FLT_MAX columns count as columns."""
    compiled = build_cluster_variants(torch.cuda.current_device())
    width = 1048584
    logits, output, aux_i, aux_l = _buffers(16, width)
    lengths = [3, 4, 4, 4, 5, 512, 2048, 2051, 16384, 133337, 262144, 780000, width, 4, 4, 4]
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


@_requires_sm107
@pytest.mark.parametrize("batch", [1, 2, 16, 20, 24, 32])
def test_cluster_radix_graph_replay(_custom_ops, batch):
    """Graph batches 1 (c=1, widest cluster), 2 (c=2), and 16..32 (64..128 rows) at 1M width,
    captured once through `build_cluster_variants` and replayed over 512..780K..1M lengths."""
    variants = build_cluster_variants(torch.cuda.current_device())
    width = 1048584
    logits, output, aux_i, aux_l = _buffers(batch, width)
    edges = (_dep4_lengths(width) * 2)[: max(batch, len(_dep4_lengths(width)))]
    lengths = edges[:batch]
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    _fill(logits, lengths, 4, 0, "ordinary")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=variants)
    for replay in range(4):
        lengths = [edges[(replay * 3 + b) % len(edges)] for b in range(batch)]
        _fill(logits, lengths, 4, replay, ("ordinary", "ties", "boundaries", "dense")[replay])
        lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        output.fill_(-7)
        graph.replay()
        _check(logits, lengths, output)


@_requires_sm107
def test_cluster_radix_empty_slice_pass2_skip(_custom_ops):
    """Long row whose top-2048 values all live in the first CTA slice so all other slices
    have my_above == 0 and my_eq == 0 and skip Pass 2."""
    compiled = build_cluster(torch.cuda.current_device())
    width = 1048584
    logits, output, aux_i, aux_l = _buffers(2, width)
    lengths = [524288, 4]
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    logits.fill_(-10.0)
    for row in range(4):
        logits[row, :TOP_K] = torch.arange(TOP_K, 0, -1, dtype=torch.float32, device="cuda")
    for row in range(4, 8):
        logits[row, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    output.fill_(-7)
    cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
    _check(logits, lengths, output)


@_requires_sm107
@pytest.mark.parametrize("layout", ["unaligned_rows", "unaligned_width", "eight_blocks"])
def test_cluster_radix_native_shapes(_custom_ops, layout):
    """The shapes the split selector leaves to the op take the op here too."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.indexer_topk_decode import (
        indexer_topk_decode as native_topk,
    )

    compiled = build_cluster_variants(torch.cuda.current_device())
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


@_requires_sm107
def test_cluster_radix_rejects_invalid_cuda_inputs(_custom_ops):
    device = torch.cuda.current_device()
    compiled = build_cluster(device)
    logits, output, aux_i, aux_l = _buffers(1, 16384)
    lens = torch.tensor([16384], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="float32"):
        cluster_radix_topk_decode(logits.half(), lens, output, aux_i, aux_l, compiled=compiled)
    with pytest.raises(ValueError, match="scratch"):
        cluster_radix_topk_decode(
            logits, lens, output, aux_i.flatten()[: 4 * 8 * TOP_K - 1], aux_l, compiled=compiled
        )
    logits, output, aux_i, aux_l = _buffers(33, 16384)
    lens = torch.full((33,), 16384, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="at most 128 rows"):
        cluster_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
