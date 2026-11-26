# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 (Rubin VR200) GPU accuracy, CUDA-graph, and Paradigm-B uGPU placement tests for
``ml4_moe_engine`` (TP8 ``I=768`` and TP16 ``I=384``).

Validates ``ml4_moe_engine`` against BOTH:
1. ``ml4_moe_reference`` (plain PyTorch fp64-accumulated reference), and
2. Stock ML4 decode MoE path (``cublas_mm`` router + ``renorm_moe_routing_op`` + ``fp4_quantize`` +
   ``fp4_block_scale_moe_runner`` + ``cute_dsl_fp8_block_scale_linear`` + ``flashinfer_silu_and_mul``).
"""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cute_dsl_fp8_block_scale_linear import (
    cute_dsl_fp8_block_scale_linear,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.gemm.locality import (  # noqa: E501
    LocalityMap,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.moe.ml4_moe_kernel import (  # noqa: E501
    HIDDEN,
    NUM_EXPERTS,
    TOP_K,
    Ml4MoeState,
    expand_fp8_128x128_scale_to_ue8m0_r128c4,
    ml4_moe_engine,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires VR200/SM107",
)

_STATES: dict[tuple, Ml4MoeState] = {}


def _state(m: int, inter: int, shared: bool, locality_mode: str) -> Ml4MoeState:
    key = (m, inter, shared, locality_mode)
    if key not in _STATES:
        if locality_mode != "none":
            ok, why = LocalityMap.is_available(torch.cuda.current_device())
            if not ok:
                pytest.skip(f"requires dual-domain Rubin locality: {why}")
        _STATES[key] = Ml4MoeState(
            torch.device("cuda"),
            num_tokens=m,
            inter=inter,
            shared_inter=inter,
            shared=shared,
            locality_mode=locality_mode,
        )
    return _STATES[key]


def _make_operands(inter: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    fc1_w = torch.randint(
        0,
        256,
        (NUM_EXPERTS, 2 * inter, HIDDEN // 2),
        generator=gen,
        device="cuda",
        dtype=torch.uint8,
    )
    fc1_s = (
        torch.rand((NUM_EXPERTS, 2 * inter, HIDDEN // 16), generator=gen, device="cuda") * 0.75
        + 0.25
    ).to(torch.float8_e4m3fn)
    fc2_w = torch.randint(
        0, 256, (NUM_EXPERTS, HIDDEN, inter // 2), generator=gen, device="cuda", dtype=torch.uint8
    )
    fc2_s = (
        torch.rand((NUM_EXPERTS, HIDDEN, inter // 16), generator=gen, device="cuda") * 0.75 + 0.25
    ).to(torch.float8_e4m3fn)
    out1_gate = torch.full((NUM_EXPERTS,), 2e-3, dtype=torch.float32, device="cuda")
    out1 = torch.full((NUM_EXPERTS,), 2e-3, dtype=torch.float32, device="cuda")
    out2 = torch.full((NUM_EXPERTS,), 1e-3, dtype=torch.float32, device="cuda")
    router = (torch.randn((NUM_EXPERTS, HIDDEN), generator=gen, device="cuda") * 0.02).to(
        torch.bfloat16
    )
    gscale = torch.tensor([134.4], dtype=torch.float32, device="cuda")

    # Shared expert FP8_PB_WO with exact power-of-2 128x128 block scales expanded to R128c4 UE8M0.
    gu_w = (torch.randn((2 * inter, HIDDEN), generator=gen, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    gu_exp = torch.randint(-6, 4, (2 * inter // 128, HIDDEN // 128), generator=gen, device="cuda")
    gu_s_f32 = torch.ldexp(torch.ones_like(gu_exp, dtype=torch.float32), gu_exp)
    gu_sf, ok = expand_fp8_128x128_scale_to_ue8m0_r128c4(gu_s_f32, 2 * inter, HIDDEN, strict=True)
    assert ok

    dn_w = (torch.randn((HIDDEN, inter), generator=gen, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    dn_exp = torch.randint(-6, 4, (HIDDEN // 128, inter // 128), generator=gen, device="cuda")
    dn_s_f32 = torch.ldexp(torch.ones_like(dn_exp, dtype=torch.float32), dn_exp)
    dn_sf, ok = expand_fp8_128x128_scale_to_ue8m0_r128c4(dn_s_f32, HIDDEN, inter, strict=True)
    assert ok

    return dict(
        fc1_w=fc1_w,
        fc1_s=fc1_s,
        fc2_w=fc2_w,
        fc2_s=fc2_s,
        out1_gate=out1_gate,
        out1=out1,
        out2=out2,
        router=router,
        gscale=gscale,
        shared=(gu_w, gu_sf, dn_w, dn_sf),
    )


def _stock_ml4_moe(
    hidden: torch.Tensor, ops: dict, inter: int, include_shared: bool
) -> torch.Tensor:
    logits = (hidden.float() @ ops["router"].float().t()).contiguous()
    ids, weights = renorm_moe_routing_op(logits, TOP_K, torch.bfloat16)
    xq, xs = fp4_quantize(hidden, ops["gscale"], 16, False, False)
    routed = fp4_block_scale_moe_runner(
        None,
        None,
        xq,
        xs.view(torch.float8_e4m3fn),
        ops["fc1_w"],
        ops["fc1_s"],
        None,
        None,
        None,
        None,
        ops["fc2_w"],
        ops["fc2_s"],
        None,
        ops["out1"],
        ops["out1_gate"],
        ops["out2"],
        NUM_EXPERTS,
        TOP_K,
        None,
        None,
        inter,
        0,
        NUM_EXPERTS,
        None,
        1,
        True,
        0,
        topk_weights=weights,
        topk_ids=ids,
        tune_max_num_tokens=16,
    )[0]
    if not include_shared:
        return routed
    gu_w, gu_sf, dn_w, dn_sf = ops["shared"]
    act = flashinfer_silu_and_mul(cute_dsl_fp8_block_scale_linear(hidden, gu_w, gu_sf))
    sh = cute_dsl_fp8_block_scale_linear(act, dn_w, dn_sf)
    return torch.add(routed, sh)


@pytest.mark.parametrize(
    "cell",
    ml4_moe_engine.CELLS,
    ids=[
        f"{c.spec['locality_mode']}-m{c.spec['tokens']}-I{c.spec['inter']}"
        for c in ml4_moe_engine.CELLS
    ],
)
def test_ml4_moe_matches_stock_and_reference(cell) -> None:
    inter = cell.spec["inter"]
    m = cell.spec["tokens"]
    locality_mode = cell.spec["locality_mode"]
    ops = _make_operands(inter, seed=1000 + inter + m)
    st = _state(m, inter, shared=True, locality_mode=locality_mode)
    layer = st.layer(
        ops["fc1_w"],
        ops["fc1_s"],
        ops["fc2_w"],
        ops["fc2_s"],
        ops["out1_gate"],
        ops["out1"],
        ops["out2"],
        ops["router"],
        ops["gscale"],
        shared_weights=ops["shared"],
    )
    gen = torch.Generator(device="cuda").manual_seed(2026 + m)
    hidden = torch.randn((m, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)

    out = ml4_moe_engine(hidden, layer)
    again = ml4_moe_engine(hidden, layer)
    assert torch.equal(out.view(torch.int16), again.view(torch.int16)), "non-deterministic repeat"

    stock = _stock_ml4_moe(hidden, ops, inter, include_shared=True)
    ref = ml4_moe_engine.reference(hidden, layer)
    assert_within_ulp(out, stock, element_ulp=4.0, rms_ulp=0.5)
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=0.5)


@pytest.mark.parametrize("inter", (768, 384), ids=("tp8", "tp16"))
@pytest.mark.parametrize("locality_mode", ("n_split", "expert_half"))
def test_ml4_moe_ugpu_placement_is_bitwise_equal_to_unplaced(
    inter: int, locality_mode: str
) -> None:
    ops = _make_operands(inter, seed=777 + inter)
    st_ref = _state(4, inter, shared=True, locality_mode="none")
    st_loc = _state(4, inter, shared=True, locality_mode=locality_mode)
    layer_ref = st_ref.layer(
        ops["fc1_w"],
        ops["fc1_s"],
        ops["fc2_w"],
        ops["fc2_s"],
        ops["out1_gate"],
        ops["out1"],
        ops["out2"],
        ops["router"],
        ops["gscale"],
        shared_weights=ops["shared"],
    )
    layer_loc = st_loc.layer(
        ops["fc1_w"],
        ops["fc1_s"],
        ops["fc2_w"],
        ops["fc2_s"],
        ops["out1_gate"],
        ops["out1"],
        ops["out2"],
        ops["router"],
        ops["gscale"],
        shared_weights=ops["shared"],
    )
    # Fail-closed VMM domain verification on placed halves.
    lm = st_loc.locality_map
    assert lm is not None
    if locality_mode == "n_split":
        assert lm.domain_of(layer_loc.a1_d0.data_ptr()) == 0
        assert lm.domain_of(layer_loc.a1_d1.data_ptr()) == 1
        assert lm.domain_of(layer_loc.a2_d0.data_ptr()) == 0
        assert lm.domain_of(layer_loc.a2_d1.data_ptr()) == 1
    else:
        assert lm.domain_of(layer_loc.s2w_d0.data_ptr()) == 0
        assert lm.domain_of(layer_loc.s2w_d1.data_ptr()) == 1

    hidden = torch.randn((4, HIDDEN), device="cuda", dtype=torch.bfloat16)
    y_ref = ml4_moe_engine(hidden, layer_ref)
    y_loc = ml4_moe_engine(hidden, layer_loc)
    assert_within_ulp(y_loc, y_ref, element_ulp=2.0, rms_ulp=0.2)


@pytest.mark.parametrize("inter", (768, 384), ids=("tp8", "tp16"))
def test_ml4_moe_cuda_graph_replay(inter: int) -> None:
    ops = _make_operands(inter, seed=999 + inter)
    st = _state(4, inter, shared=True, locality_mode="none")
    layer = st.layer(
        ops["fc1_w"],
        ops["fc1_s"],
        ops["fc2_w"],
        ops["fc2_s"],
        ops["out1_gate"],
        ops["out1"],
        ops["out2"],
        ops["router"],
        ops["gscale"],
        shared_weights=ops["shared"],
    )
    x = torch.randn((4, HIDDEN), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((4, HIDDEN), device="cuda", dtype=torch.bfloat16)
    ml4_moe_engine(x, layer, out=out)  # warm-up compile before capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ml4_moe_engine(x, layer, out=out)
    for seed in range(5):
        gen = torch.Generator(device="cuda").manual_seed(500 + seed)
        x.copy_(torch.randn((4, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        ref = _stock_ml4_moe(x, ops, inter, include_shared=True)
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=0.5)
