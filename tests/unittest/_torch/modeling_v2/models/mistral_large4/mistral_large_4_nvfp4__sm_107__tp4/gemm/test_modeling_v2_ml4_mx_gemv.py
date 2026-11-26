# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 (Rubin VR200) GPU accuracy, Paradigm-B uGPU placement, fused LayerNorm, fused SwiGLU,
and CUDA-graph replay tests for ``ml4_mx_gemv`` across all ML4 TP8 & TP16 FP8 linear shapes."""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cute_dsl_fp8_block_scale_linear import (
    cute_dsl_fp8_block_scale_linear as baseline,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.gemm.locality import (  # noqa: E501
    LocalityMap,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.gemm.ml4_mx_gemv import (  # noqa: E501
    ml4_mx_gemv,
    ml4_mx_gemv_layernorm,
    ml4_mx_gemv_swiglu,
    prepare_ugpu_gemv_weight,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires VR200/SM107",
)


def _input(m: int, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=gen)
    exponents = torch.randint(-12, 13, (m, k // 128, 1), device="cuda", generator=gen)
    x.copy_(torch.ldexp(x.reshape(m, k // 128, 128).float(), exponents).reshape(m, k))
    x[0, :128] = 0
    return x.contiguous()


def _weights(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    w = _input(n, k, seed)
    return torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(w, True)


@pytest.mark.parametrize(
    "cell",
    ml4_mx_gemv.CELLS,
    ids=[f"m{c.spec['m']}-{c.spec['n']}x{c.spec['k']}" for c in ml4_mx_gemv.CELLS],
)
def test_ml4_mx_gemv_matches_k128_reference(cell) -> None:
    m, n, k = cell.spec["m"], cell.spec["n"], cell.spec["k"]
    w, ws = _weights(n, k, seed=101 + n + k)
    x = _input(m, k, seed=202 + m)
    out = ml4_mx_gemv(x, w, ws)
    assert out is not None, f"expected {cell.why} to be served"
    expected = ml4_mx_gemv.reference(x, w, ws)
    assert_within_ulp(out, expected, element_ulp=4.0, rms_ulp=2.0)


@pytest.mark.parametrize(
    "nk",
    (
        (512, 6144),  # TP16 wq (1.5 MiB/half -> padded)
        (1024, 6144),  # TP8 wq (3 MiB/half -> padded)
        (4096, 6144),  # ix_wq_b (12 MiB/half -> exact 2 MiB!)
        (6144, 512),  # TP16 wo (1.5 MiB/half -> padded)
        (6144, 1024),  # TP8 wo (3 MiB/half -> padded)
        (6144, 6144),  # TP8 dense0_gu (18 MiB/half -> exact 2 MiB!)
    ),
)
def test_ml4_mx_gemv_ugpu_bitwise_matches_unplaced(nk: tuple[int, int]) -> None:
    ok, why = LocalityMap.is_available(torch.cuda.current_device())
    if not ok:
        pytest.skip(f"requires dual-domain Rubin locality: {why}")
    n, k = nk
    w, ws = _weights(n, k, seed=303 + n)
    lm = LocalityMap.get(w.device)
    placed = prepare_ugpu_gemv_weight(w, ws, lm)
    assert placed is not None
    # Fail-closed VMM domain verification on weight and scale halves.
    assert lm.domain_of(placed.weight_split.half0.data_ptr()) == 0
    assert lm.domain_of(placed.weight_split.half1.data_ptr()) == 1
    assert lm.domain_of(placed.scale_split.half0.data_ptr()) == 0
    assert lm.domain_of(placed.scale_split.half1.data_ptr()) == 1

    x = _input(4, k, seed=404)
    y_unplaced = ml4_mx_gemv(x, w, ws)
    y_ugpu = ml4_mx_gemv(x, w, ws, placed=placed)
    assert y_unplaced is not None and y_ugpu is not None
    # Placement only: arithmetic is unchanged, so outputs are bitwise identical!
    assert torch.equal(y_ugpu.view(torch.int16), y_unplaced.view(torch.int16))


@pytest.mark.parametrize(
    "cell",
    ml4_mx_gemv_layernorm.CELLS,
    ids=[f"m{c.spec['m']}" for c in ml4_mx_gemv_layernorm.CELLS],
)
@pytest.mark.parametrize("use_ugpu", (False, True))
def test_ml4_mx_gemv_fused_layernorm_ix_wq_b(cell, use_ugpu: bool) -> None:
    m, n, k = cell.spec["m"], cell.spec["n"], cell.spec["k"]
    w, ws = _weights(n, k, seed=505)
    placed = None
    if use_ugpu:
        ok, why = LocalityMap.is_available(torch.cuda.current_device())
        if not ok:
            pytest.skip(f"requires dual-domain Rubin locality: {why}")
        placed = prepare_ugpu_gemv_weight(w, ws, LocalityMap.get(w.device))
        assert placed is not None

    gen = torch.Generator(device="cuda").manual_seed(606 + m)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=gen)
    ln_w = (torch.rand((k,), dtype=torch.float32, device="cuda", generator=gen) * 0.25 + 0.88).to(
        torch.bfloat16
    )
    ln_b = (torch.randn((k,), dtype=torch.float32, device="cuda", generator=gen) * 0.05).to(
        torch.bfloat16
    )
    out = ml4_mx_gemv_layernorm(x, ln_w, ln_b, 1e-6, w, ws, placed=placed)
    assert out is not None
    ref = ml4_mx_gemv_layernorm.reference(x, ln_w, ln_b, 1e-6, w, ws)
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


@pytest.mark.parametrize("k", (1536, 3072), ids=("tp16_dense0_dn", "tp8_dense0_dn"))
def test_ml4_mx_gemv_fused_swiglu(k: int) -> None:
    n = 6144
    w, ws = _weights(n, k, seed=707 + k)
    gu = torch.randn((4, 2 * k), dtype=torch.bfloat16, device="cuda")
    out = ml4_mx_gemv_swiglu(gu, w, ws)
    assert out is not None
    expected = baseline.reference(flashinfer_silu_and_mul(gu), w, ws)
    assert_within_ulp(out, expected, element_ulp=4.0, rms_ulp=2.0)


def test_ml4_mx_gemv_cuda_graph_replay_ugpu() -> None:
    ok, why = LocalityMap.is_available(torch.cuda.current_device())
    if not ok:
        pytest.skip(f"requires dual-domain Rubin locality: {why}")
    n, k = 4096, 6144
    w, ws = _weights(n, k, seed=808)
    placed = prepare_ugpu_gemv_weight(w, ws, LocalityMap.get(w.device))
    assert placed is not None
    x = _input(4, k, seed=809)
    ml4_mx_gemv(x, w, ws, placed=placed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ml4_mx_gemv(x, w, ws, placed=placed)
    for seed in range(10):
        x.copy_(_input(4, k, 900 + seed))
        graph.replay()
        torch.cuda.synchronize()
        expected = ml4_mx_gemv.reference(x, w, ws)
        assert_within_ulp(out, expected, element_ulp=4.0, rms_ulp=2.0)
