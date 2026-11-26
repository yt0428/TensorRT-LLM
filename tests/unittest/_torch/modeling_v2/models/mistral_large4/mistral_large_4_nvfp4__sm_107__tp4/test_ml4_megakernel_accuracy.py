# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 (VR200) GPU accuracy verification harness for the ML4 TP16 B1 M=4 megakernel.

Validates strictly-lossless equivalence against exact PyTorch/ML4 reference math on real VR200
hardware across all 4 heterogeneous MoE layer types:
  - Plain (RoPE owner, shared top-k, 1-kernel launch, 32-CTA TCGen05 MXFP8 Wq)
  - RoPE Owner Leader (2-kernel PDL split: ml4LeaderFrontKernel cluster=(4,1,1) -> ml4LayerKernel)
  - NoPE Owner Leader
  - NoPE Layer-35 KV-Sharer Leader
plus:
  - Hardware SMID -> uGPU Die-Aware probe bijection (106 + 106 SMs)
  - 2D SWIZZLE_128B CUtensorMap encoding for tpb=32 and tpb=64
  - Full execution verifying x_out, res_out, captured_hidden_states, 656B fp8_ds_mla KV,
    132B indexer K cache, exact Top-2048 selection (L<=2048, L=2049/2051 edge, L=64K/128K kModeCand,
    and heavy exact-tie kModeTies multi-level fallback!)
  - Cross-layer PDL chaining (4-layer heterogeneous chain, 4-copy schedule) + CUDA Graph capture & replay
    (5/5 bit-exact).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4 import (
    _megakernel as mk,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4 import (
    weights as ml4_weights,
)


def _is_sm107_vr200() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (props.major, props.minor) == (10, 7) and props.multi_processor_count == 212


_SKIP_REASON = "Requires NVIDIA Rubin VR200 (SM107, 212 SMs) with built libth_ml4_megakernel.so"


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    bits = x.contiguous().view(torch.int32)
    exp = (bits & 0x7F800000) + torch.where(
        (bits & 0x007FFFFF) != 0,
        torch.tensor(0x00800000, dtype=torch.int32, device=x.device),
        torch.tensor(0, dtype=torch.int32, device=x.device),
    )
    return exp.to(torch.int32).view(torch.float32)


def _make_fp8_pb_wo(n: int, k: int, device: torch.device, scale_std: float = 0.04):
    """Create synthetic FP8_PB_WO weights (e4m3 [N, K]) and R128c4 K32 UE8M0 scales."""
    w_raw = (torch.randn((n, k), device=device, dtype=torch.float32) * scale_std).clamp(
        -440.0, 440.0
    )
    w_blk = w_raw.view(n // 128, 128, k // 128, 128)
    amax = w_blk.abs().amax(dim=(1, 3)).clamp(min=1e-6)
    s_128 = _pow2_ceil(amax / 448.0)
    w_q = (w_blk / s_128[:, None, :, None]).reshape(n, k).to(torch.float8_e4m3fn)
    exp_u8 = ((s_128.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    exp_k32 = exp_u8.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
    cols_k32 = k // 32
    s_r128c4 = (
        exp_k32.reshape(n // 128, 4, 32, cols_k32 // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(-1)
    )
    w_deq = w_q.float() * s_128.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return w_q, s_r128c4, w_deq


def _fp8_pb_wo_ref(x: torch.Tensor, w_deq: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    xb = x.float().view(m, k // 128, 128)
    amax = xb.abs().amax(dim=-1).clamp(min=1e-12)
    xs = _pow2_ceil(amax * (1.0 / 448.0))
    xq = (xb / xs[:, :, None]).to(torch.float8_e4m3fn).float() * xs[:, :, None]
    return (xq.reshape(m, k) @ w_deq.t()).to(torch.bfloat16)


def _quant_e2m1_nvfp4(x: torch.Tensor, gscale: float):
    orig_shape = x.shape
    k = orig_shape[-1]
    xg = x.float().reshape(-1, k // 16, 16)
    vmax = xg.abs().amax(dim=-1)
    sf = (gscale * (vmax / 6.0)).to(torch.float8_e4m3fn)
    sff = sf.float()
    step = torch.where(vmax > 0, sff / gscale, torch.ones_like(sff))
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    q_norm = (xg / step.unsqueeze(-1)).clamp(-6.0, 6.0)
    sign = q_norm.sign()
    mag = q_norm.abs().unsqueeze(-1)
    idx = (mag - vals).abs().argmin(dim=-1)
    deq = (sign * vals[idx] * step.unsqueeze(-1)).reshape(orig_shape)
    nib = ((q_norm < 0).to(torch.uint8) << 3) | idx.to(torch.uint8)
    nib = nib.reshape(*orig_shape[:-1], k // 2, 2)
    packed = nib[..., 0] | (nib[..., 1] << 4)
    return packed.contiguous(), sf.reshape(*orig_shape[:-1], k // 16).contiguous(), deq


def _build_synthetic_layer(
    ops,
    device: torch.device,
    leader: bool,
    nope: bool,
    kv_owner: bool,
    tpb: int,
    L: int,
    die_aware: bool,
    smid_map: torch.Tensor | None,
    n_die0: int,
    seed: int = 123,
    force_ties: bool = False,
):
    """Build one complete synthetic ML4 TP16 MoE layer + exact reference inputs/outputs."""
    torch.manual_seed(seed)
    T, H, HEADS, KV_LORA, V_DIM, EXPERTS, MOE_I = 4, 6144, 4, 512, 128, 128, 384
    NOPE_D = 128 if nope else 64
    RMS_EPS = 1e-5
    IDX_EPS = 1e-6

    res_in = (torch.randn((T, H), device=device, dtype=torch.float32) * 0.25).to(torch.bfloat16)
    gamma0 = (1.0 + 0.05 * torch.randn((H,), device=device)).to(torch.bfloat16)
    x_in = (
        res_in.float()
        * torch.rsqrt((res_in.float() ** 2).mean(dim=-1, keepdim=True) + RMS_EPS)
        * gamma0.float()
    ).to(torch.bfloat16)
    positions = torch.arange(L - T, L, dtype=torch.int32, device=device)
    max_pos = max(L + 16, 4096)
    angles = (
        torch.arange(max_pos, dtype=torch.float32, device=device)[:, None]
        * (10000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64.0))[None, :]
    )
    rope_tbl = torch.cat(
        [angles.cos().to(torch.bfloat16).float(), angles.sin().to(torch.bfloat16).float()], dim=1
    ).contiguous()

    wq_q, wq_s_r, wq_deq = _make_fp8_pb_wo(512, H, device, 0.03)
    wo_q, wo_s_r, wo_deq = _make_fp8_pb_wo(H, 512, device, 0.03)
    gu_q, gu_s_r, gu_deq = _make_fp8_pb_wo(2 * MOE_I, H, device, 0.03)
    dn_q, dn_s_r, dn_deq = _make_fp8_pb_wo(H, MOE_I, device, 0.03)

    ssmax_s = (0.1 + 0.05 * torch.randn((HEADS,), device=device)).to(torch.bfloat16)
    ssmax_b = (1.0 + 0.05 * torch.randn((HEADS,), device=device)).to(torch.bfloat16)
    wkv_a = (torch.randn((KV_LORA if nope else 576, H), device=device) * 0.02).to(torch.bfloat16)
    kv_norm = (1.0 + 0.05 * torch.randn((KV_LORA,), device=device)).to(torch.bfloat16)
    k_b = (torch.randn((HEADS, NOPE_D, KV_LORA), device=device) * 0.03).to(torch.bfloat16)
    v_b = (torch.randn((HEADS, V_DIM, KV_LORA), device=device) * 0.03).to(torch.bfloat16)
    gamma_post = (1.0 + 0.05 * torch.randn((H,), device=device)).to(torch.bfloat16)
    gamma_next = (1.0 + 0.05 * torch.randn((H,), device=device)).to(torch.bfloat16)
    router = (torch.randn((EXPERTS, H), device=device) * 0.04).to(torch.bfloat16)

    isc1 = 0.25
    isc2 = 0.25
    ws2_13 = torch.full((EXPERTS,), 0.25, dtype=torch.float32, device=device)
    ws2_2 = torch.full((EXPERTS,), 0.25, dtype=torch.float32, device=device)
    fc1_raw = torch.randn((EXPERTS, 2 * MOE_I, H), device=device) * 0.02
    fc2_raw = torch.randn((EXPERTS, H, MOE_I), device=device) * 0.02
    fc1_c_raw, fc1_s_raw, fc1_deq = _quant_e2m1_nvfp4(fc1_raw, 1.0 / 0.25)
    fc2_c_raw, fc2_s_raw, fc2_deq = _quant_e2m1_nvfp4(fc2_raw, 1.0 / 0.25)
    fc1_perm = ml4_weights._fc1_perm(2 * MOE_I, device)
    fc2_perm = ml4_weights._block32_perm(H, device)
    fc1_w_lay = fc1_c_raw[:, fc1_perm, :].contiguous()
    fc1_s_p = fc1_s_raw[:, fc1_perm, :].view(torch.uint8)
    fc1_s_lay = (
        fc1_s_p.reshape(EXPERTS, (2 * MOE_I) // 128, 4, 32, (H // 16) // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(EXPERTS, 2 * MOE_I, H // 16)
        .view(torch.float8_e4m3fn)
        .contiguous()
    )
    fc2_w_lay = fc2_c_raw[:, fc2_perm, :].contiguous()
    fc2_s_p = fc2_s_raw[:, fc2_perm, :].view(torch.uint8)
    fc2_s_lay = (
        fc2_s_p.reshape(EXPERTS, H // 128, 4, 32, (MOE_I // 16) // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(EXPERTS, H, MOE_I // 16)
        .view(torch.float8_e4m3fn)
        .contiguous()
    )

    ix_q_norm_w = (1.0 + 0.05 * torch.randn((H,), device=device)).to(torch.bfloat16)
    ix_q_norm_b = (0.02 * torch.randn((H,), device=device)).to(torch.bfloat16)
    ix_wq_q, ix_wq_s_r, ix_wq_deq = _make_fp8_pb_wo(4096, H, device, 0.03)
    ix_wk_wp = (torch.randn((160, H), device=device) * 0.03).to(torch.bfloat16)
    ix_kn_w = (1.0 + 0.05 * torch.randn((128,), device=device)).float()
    ix_kn_b = (0.02 * torch.randn((128,), device=device)).float()
    ix_ss_s = (0.02 * torch.randn((32,), device=device)).to(torch.bfloat16)
    ix_ss_b = (1.0 + 0.02 * torch.randn((32,), device=device)).to(torch.bfloat16)

    num_blocks = (L + tpb - 1) // tpb + 4
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device)
    kv_pool = torch.zeros((num_blocks, 1, tpb, 656), dtype=torch.uint8, device=device)
    cached_c = (torch.randn((L, KV_LORA), device=device) * 0.3).to(torch.bfloat16)
    cached_r = (
        torch.zeros((L, 64), device=device, dtype=torch.bfloat16)
        if nope
        else (torch.randn((L, 64), device=device) * 0.3).to(torch.bfloat16)
    )
    ct_v = cached_c.float().view(L, 4, 128)
    sc_v = _pow2_ceil((ct_v.abs().amax(dim=-1) / 448.0).clamp(min=1e-4))
    codes_v = (ct_v / sc_v[:, :, None]).reshape(L, 512).to(torch.float8_e4m3fn).view(torch.uint8)
    kv_rows = kv_pool.view(num_blocks * tpb, 656)
    kv_rows[:L, :512] = codes_v
    kv_rows[:L, 512:528] = sc_v.view(torch.uint8).reshape(L, 16)
    kv_rows[:L, 528:656] = cached_r.view(torch.uint8).reshape(L, 128)

    idx_pool = torch.zeros((num_blocks, tpb, 1, 132), dtype=torch.uint8, device=device)
    idx_pool_flat = idx_pool.view(num_blocks, tpb * 132)
    if force_ties:
        # Constant keys -> forces > 4096 identical logits in 1 bin -> exercises Level 1/2 fallback + kModeTies!
        cached_ik = torch.ones((L, 128), device=device) * 0.25
    else:
        cached_ik = torch.randn((L, 128), device=device) * 0.3
    ik_sc = _pow2_ceil(cached_ik.abs().amax(dim=-1).clamp(min=1e-12) * (1.0 / 448.0))
    ik_codes = (cached_ik / ik_sc[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    full_blks = L // tpb
    if full_blks > 0:
        idx_pool_flat[:full_blks, : tpb * 128] = ik_codes[: full_blks * tpb].reshape(
            full_blks, tpb * 128
        )
        idx_pool_flat[:full_blks, tpb * 128 :] = (
            ik_sc[: full_blks * tpb].view(torch.uint8).reshape(full_blks, tpb * 4)
        )
    rem_ik = L % tpb
    if rem_ik > 0:
        idx_pool_flat[full_blks, : rem_ik * 128] = ik_codes[full_blks * tpb :].reshape(rem_ik * 128)
        idx_pool_flat[full_blks, tpb * 128 : tpb * 128 + rem_ik * 4] = (
            ik_sc[full_blks * tpb :].view(torch.uint8).reshape(rem_ik * 4)
        )

    # Weight packing: TCGen05 MXFP8 tiles for Wq, WGu, WD, IdxWq; warp FP8 for WO.
    wq_p = mk.pack_mxf8_tiles(wq_q, wq_s_r)
    wo_p, wo_s = mk.pack_fp8_pb_wo(wo_q, wo_s_r)
    gu_p = mk.pack_mxf8_tiles(gu_q, gu_s_r)
    dn_p = mk.pack_mxf8_tiles(dn_q, dn_s_r)
    ix_wq_p = mk.pack_mxf8_tiles(ix_wq_q, ix_wq_s_r)
    kv_parts = ([wkv_a] if kv_owner else []) + ([ix_wk_wp] if leader else [])
    w_kva = (
        mk.pack_bf16(torch.cat(kv_parts, dim=0))
        if kv_parts
        else torch.empty(1, dtype=torch.uint8, device=device)
    )
    w_kb = mk.pack_bf16(k_b.transpose(1, 2).contiguous().reshape(HEADS * KV_LORA, NOPE_D))
    w_vb = mk.pack_bf16(v_b.contiguous().reshape(HEADS * V_DIM, KV_LORA))
    w_r = mk.pack_bf16(router)
    w13, w2 = mk.repack_nvfp4_moe(fc1_w_lay, fc1_s_lay, fc2_w_lay, fc2_s_lay)
    alpha1 = torch.cat([isc1 * ws2_13, torch.tensor([1.0 / isc1], device=device)]).contiguous()
    alpha2 = torch.cat([isc2 * ws2_2, torch.tensor([1.0 / isc2], device=device)]).contiguous()

    copies = 4
    arena = torch.full((copies * mk._ARENA_BYTES // 4,), -0.0, dtype=torch.float32, device=device)
    idx_arena = torch.full(
        (copies * mk._IDX_ARENA_BYTES // 4,), -0.0, dtype=torch.float32, device=device
    )
    idx_ar_u8 = idx_arena.view(torch.uint8).view(copies, mk._IDX_ARENA_BYTES)
    idx_ar_u8[:, mk._IDX_ARENA_HIST0_OFF :].zero_()
    idx_logits = torch.zeros(
        (copies * (mk._IDX_LOGITS_BYTES // 4),), dtype=torch.float32, device=device
    )
    idx_counters = torch.zeros(copies * mk._NUM_COUNTERS, dtype=torch.int32, device=device)
    idx_clear_meta = torch.zeros(copies * T * 2, dtype=torch.int32, device=device)
    ugpu_state = torch.zeros(20, dtype=torch.int32, device=device)

    part_o = torch.full((T * 32 * HEADS * KV_LORA,), -0.0, dtype=torch.float32, device=device)
    part_stats = torch.full((T * 32 * HEADS * 2,), -0.0, dtype=torch.float32, device=device)
    fc2out = torch.full((mk._MAX_SLOTS * T * H // 2,), -0.0, dtype=torch.float32, device=device)
    counters = torch.zeros(copies * mk._NUM_COUNTERS, dtype=torch.int32, device=device)
    err = torch.zeros(16 + 5 * mk._CTAS, dtype=torch.int32, pin_memory=True)

    # Zero peer ranks 1..15 in H1/H2/IdxHist; set rank 0 to empty sentinel (-0.0 / 0xFFFFFFFF)!
    h1 = torch.zeros((copies, mk._HOP_BYTES // 4), dtype=torch.float32, device=device)
    h2 = torch.zeros((copies, mk._HOP_BYTES // 4), dtype=torch.float32, device=device)
    tk = torch.full((copies, mk._TK_BYTES // 4), -0.0, dtype=torch.float32, device=device)
    ih = torch.zeros((copies, 3, T, 16, mk._IDX_HIST_WORDS), dtype=torch.int32, device=device)
    ib = torch.full((copies, mk._IDX_BND_BYTES // 4), -1, dtype=torch.int32, device=device)
    h1.view(copies, 16, H // 32, T, 16)[:, 0].fill_(-0.0)
    h2.view(copies, T, 16, H // 2)[:, :, 0].fill_(-0.0)
    ih[:, :, :, 0, :].fill_(-1)

    topk_tensor = torch.full((T, 2048), -1, dtype=torch.int32, device=device)
    if not leader:
        for t in range(T):
            nv = min(L - (T - 1 - t), 2048)
            topk_tensor[t, :nv] = torch.arange(nv, dtype=torch.int32, device=device)

    cap_buf = torch.zeros((T, H), dtype=torch.bfloat16, device=device)
    idx_rows = (num_blocks * tpb * 132) // 128
    idx_tmap = ops.ml4_megakernel_indexer_tmap(idx_pool, idx_rows, tpb).to(device)

    flags = (
        mk._FLAG_HOLD_EXIT
        | mk._FLAG_EVICT_FIRST
        | mk._FLAG_MOE_L2
        | mk._FLAG_X_PRIME
        | mk._FLAG_H_PRIME
        | mk._FLAG_HOP1_LINES
        | mk._FLAG_TEST_RANK0_CLEAR
    )
    if leader:
        flags |= mk._FLAG_LEADER | mk._FLAG_SKIP_FRONT
    if nope:
        flags |= mk._FLAG_NOPE
    if kv_owner:
        flags |= mk._FLAG_KV_OWNER
    if die_aware and smid_map is not None:
        flags |= mk._FLAG_DIE_AWARE

    roles, role_smem, role_issue = mk.build_roles(
        leader=leader, nope=nope, kv_owner=kv_owner, n_die0=n_die0
    )
    tables = torch.cat([roles.reshape(-1), role_smem.reshape(-1), role_issue.reshape(-1)])
    wt_offset = len(mk._FIELDS) + tables.numel() // 2
    p = torch.zeros(wt_offset + mk._CTAS * mk._MAX_ROLES * 3, dtype=torch.int64)
    p[len(mk._FIELDS) : wt_offset] = tables.view(torch.int64)

    def put(name: str, val):
        p[mk._P[name]] = val.data_ptr() if isinstance(val, torch.Tensor) else int(val)

    for k_name, v in (
        ("Magic", mk._MAGIC),
        ("KvBase", kv_pool),
        ("StrideRows", tpb),
        ("LayerRows", 0),
        ("MaxBlocks", num_blocks),
        ("TokensPerBlock", tpb),
        ("Rope", rope_tbl),
        ("Wq", wq_p),
        ("SsmaxS", ssmax_s),
        ("SsmaxB", ssmax_b),
        ("WkvA", w_kva),
        ("Kb", w_kb),
        ("Vb", w_vb),
        ("WO", wo_p),
        ("WOScale", wo_s),
        ("WR", w_r),
        ("WGu", gu_p),
        ("WD", dn_p),
        ("GammaKv", kv_norm),
        ("GammaPost", gamma_post),
        ("GammaNext", gamma_next),
        ("W13", w13),
        ("W2", w2),
        ("Alpha1", alpha1),
        ("Alpha2", alpha2),
        ("Arena", arena),
        ("PartO", part_o),
        ("PartStats", part_stats),
        ("Fc2out", fc2out),
        ("H1Mc", h1),
        ("H1Uc", h1),
        ("H2Mc", h2),
        ("H2Uc", h2),
        ("TkMc", tk),
        ("TkUc", tk),
        ("IdxHistMc", ih),
        ("IdxHistUc", ih),
        ("IdxBndMc", ib),
        ("IdxBndUc", ib),
        ("HopBytes", mk._HOP_BYTES),
        ("Rank", 0),
        ("Counters", counters),
        ("Err", err),
        ("MaxRoles", mk._MAX_ROLES),
        ("TimeoutNs", 10_000_000_000),
        ("Flags", flags),
        ("Copy", 0),
        ("NextCopy", 1),
        ("PrevCopy", 3),
        ("NumCopies", copies),
        ("IdxCopy", 0),
        ("NextIdxCopy", 1),
        ("IdxArena", idx_arena),
        ("IdxLogits", idx_logits),
        ("IdxCounters", idx_counters),
        ("IdxClearMeta", idx_clear_meta),
        ("SmidMap", smid_map if (die_aware and smid_map is not None) else 0),
        ("NDie0", n_die0),
        ("RmsEps", mk._float_to_u32(RMS_EPS)),
        ("IdxEps", mk._float_to_u32(IDX_EPS)),
        ("CaptureOut", cap_buf),
        ("CaptureStride", H),
        ("IdxQNormW", ix_q_norm_w),
        ("IdxQNormB", ix_q_norm_b),
        ("IdxWq", ix_wq_p),
        ("IdxKnW", ix_kn_w),
        ("IdxKnB", ix_kn_b),
        ("IdxSsmaxS", ix_ss_s),
        ("IdxSsmaxB", ix_ss_b),
        ("IdxPool", idx_pool),
        ("IdxBlockStride", tpb * 132),
        ("IdxTmap", idx_tmap),
        ("IdxBlocks", block_table),
        ("TopkOut", topk_tensor),
        ("UgpuState", ugpu_state),
    ):
        put(k_name, v)

    params = p.to(device)
    ops.ml4_megakernel_weight_table(params, wt_offset)

    # --- Exact PyTorch Reference for rank=0 (peer ranks 1..15 = 0) ---
    q_lin = _fp8_pb_wo_ref(x_in, wq_deq).view(T, HEADS, 128)
    lp = torch.log(positions.float() + 1.0).to(torch.bfloat16).float()
    ss = (
        (
            (lp[:, None] * ssmax_s[None, :].float()).to(torch.bfloat16).float()
            + ssmax_b[None, :].float()
        )
        .to(torch.bfloat16)
        .float()
    )
    q_sc = (q_lin.float() * ss[:, :, None]).to(torch.bfloat16)
    fq_ref = torch.zeros((T, HEADS, 576), device=device, dtype=torch.bfloat16)
    fq_ref[:, :, :512] = torch.einsum("bhk,hkd->bhd", q_sc[:, :, :NOPE_D].float(), k_b.float()).to(
        torch.bfloat16
    )
    if not nope:
        qp = q_sc[:, :, 64:].float().view(T, HEADS, 32, 2)
        c = rope_tbl[positions, :32][:, None, :]
        s = rope_tbl[positions, 32:][:, None, :]
        fq_ref[:, :, 512:] = (
            torch.stack([qp[..., 0] * c - qp[..., 1] * s, qp[..., 0] * s + qp[..., 1] * c], dim=-1)
            .view(T, HEADS, 64)
            .to(torch.bfloat16)
        )

    kv_ref_c = cached_c.clone()
    kv_ref_r = cached_r.clone()
    if kv_owner:
        kv_raw = (x_in.float() @ wkv_a.float().t()).to(torch.bfloat16)
        ckv = kv_raw[:, :KV_LORA].float()
        ckv_n = (
            ckv * torch.rsqrt((ckv**2).mean(dim=-1, keepdim=True) + RMS_EPS) * kv_norm.float()
        ).to(torch.bfloat16)
        ct = ckv_n.float().view(T, 4, 128)
        sc = _pow2_ceil((ct.abs().amax(dim=-1) / 448.0).clamp(min=1e-4))
        kv_ref_c[L - T : L] = (
            ((ct / sc[:, :, None]).to(torch.float8_e4m3fn).float() * sc[:, :, None])
            .view(T, 512)
            .to(torch.bfloat16)
        )
        if not nope:
            kp = kv_raw[:, 512:].float().view(T, 32, 2)
            c = rope_tbl[positions, :32]
            s = rope_tbl[positions, 32:]
            kv_ref_r[L - T : L] = (
                torch.stack(
                    [kp[..., 0] * c - kp[..., 1] * s, kp[..., 0] * s + kp[..., 1] * c], dim=-1
                )
                .view(T, 64)
                .to(torch.bfloat16)
            )
        else:
            kv_ref_r[L - T : L].zero_()
    ct_all = kv_ref_c.float().view(L, 4, 128)
    sc_all = _pow2_ceil((ct_all.abs().amax(dim=-1) / 448.0).clamp(min=1e-4))
    kv_deq_c = (
        (ct_all / sc_all[:, :, None]).to(torch.float8_e4m3fn).float() * sc_all[:, :, None]
    ).view(L, 512)
    kv_deq = torch.cat([kv_deq_c, kv_ref_r.float()], dim=-1)

    ref_topk = topk_tensor.clone()
    if leader:
        if L <= 2048:
            for t in range(T):
                nv = L - (T - 1 - t)
                ref_topk[t, :nv] = torch.arange(nv, dtype=torch.int32, device=device)
        else:
            n_tiles_r0 = ((L + 127) // 128) // 16
            n_keys_r0 = min(n_tiles_r0 * 128, L)
            qn = F.layer_norm(
                x_in.float(), (H,), ix_q_norm_w.float(), ix_q_norm_b.float(), eps=IDX_EPS
            ).to(torch.bfloat16)
            iq = _fp8_pb_wo_ref(qn, ix_wq_deq).view(T, 32, 128)
            iss = (
                (
                    (lp[:, None] * ix_ss_s[None, :].float()).to(torch.bfloat16).float()
                    + ix_ss_b[None, :].float()
                )
                .to(torch.bfloat16)
                .float()
            )
            iq = (iq.float() * iss[:, :, None]).to(torch.bfloat16).float()
            if not nope:
                qp = iq[:, :, :64].view(T, 32, 32, 2)
                c = rope_tbl[positions, :32][:, None, :]
                s = rope_tbl[positions, 32:][:, None, :]
                iq[:, :, :64] = (
                    torch.stack(
                        [qp[..., 0] * c - qp[..., 1] * s, qp[..., 0] * s + qp[..., 1] * c], dim=-1
                    )
                    .view(T, 32, 64)
                    .to(torch.bfloat16)
                    .float()
                )
            qs = _pow2_ceil(iq.abs().amax(dim=-1).clamp(min=1e-12) * (1.0 / 448.0))
            iq8 = (iq / qs[:, :, None]).to(torch.float8_e4m3fn).float()
            kw = (x_in.float() @ ix_wk_wp.float().t()).to(torch.bfloat16)
            iw = kw[:, 128:].float() * qs * (128.0**-0.5 * 32.0**-0.5)
            k_all = cached_ik[:n_keys_r0].clone()
            ks_all = _pow2_ceil(k_all.abs().amax(dim=-1).clamp(min=1e-12) * (1.0 / 448.0))
            k8_all = (k_all / ks_all[:, None]).to(torch.float8_e4m3fn).float()
            scores = torch.einsum("thd,kd->tkh", iq8, k8_all).clamp(min=0.0)
            logits_ix = (scores * iw[:, None, :]).sum(dim=-1) * ks_all[None, :]
            for t in range(T):
                row_win = L - (T - 1 - t)
                if row_win <= 2048:
                    ref_topk[t, :row_win] = torch.arange(row_win, dtype=torch.int32, device=device)
                elif force_ties:
                    # Larger token index breaks exact ties (glmDsaHistTopK contract!).
                    lim = min(n_keys_r0, row_win)
                    ref_topk[t, :2048] = torch.arange(
                        lim - 2048, lim, dtype=torch.int32, device=device
                    )
                else:
                    lim = min(n_keys_r0, row_win)
                    vals = logits_ix[t, :lim]
                    k_take = min(2048, lim)
                    ref_topk[t, :k_take] = torch.topk(vals, k_take, largest=True).indices.to(
                        torch.int32
                    )

    lat_ref = torch.zeros((T, HEADS, KV_LORA), device=device, dtype=torch.bfloat16)
    for t in range(T):
        idx = ref_topk[t]
        valid = idx >= 0
        sel = kv_deq[idx[valid].long()]
        s = (fq_ref[t].float() @ sel.t()) * (128.0**-0.5)
        p_attn = torch.softmax(s, dim=-1)
        lat_ref[t] = (p_attn @ sel[:, :KV_LORA]).to(torch.bfloat16)
    attn_out_ref = (
        torch.einsum("thd,hvd->thv", lat_ref.float(), v_b.float())
        .reshape(T, 512)
        .to(torch.bfloat16)
    )
    partial1_ref = _fp8_pb_wo_ref(attn_out_ref, wo_deq)

    res1_ref = (partial1_ref.float() + res_in.float()).to(torch.bfloat16)
    h_ref = (
        res1_ref.float()
        * torch.rsqrt((res1_ref.float() ** 2).mean(dim=-1, keepdim=True) + RMS_EPS)
        * gamma_post.float()
    ).to(torch.bfloat16)

    sh_gu = _fp8_pb_wo_ref(h_ref, gu_deq)
    sh_act = (F.silu(sh_gu[:, :MOE_I].float()) * sh_gu[:, MOE_I:].float()).to(torch.bfloat16)
    sh_out_ref = _fp8_pb_wo_ref(sh_act, dn_deq)

    r_logits = h_ref.float() @ router.float().t()
    top_v, top_i = torch.topk(r_logits, 4, dim=-1)
    top_w = torch.softmax(top_v, dim=-1).to(torch.bfloat16).float()
    _, _, h_nvfp4_deq = _quant_e2m1_nvfp4(h_ref, 1.0 / isc1)
    routed_ref = torch.zeros((T, H), device=device, dtype=torch.float32)
    for t in range(T):
        for k_i in range(4):
            e = int(top_i[t, k_i])
            up = h_nvfp4_deq[t] @ fc1_deq[e, :MOE_I].t()
            gate = h_nvfp4_deq[t] @ fc1_deq[e, MOE_I:].t()
            act = F.silu(gate) * up
            _, _, act_q = _quant_e2m1_nvfp4(act, 1.0 / isc2)
            out_e = (act_q @ fc2_deq[e].t()).to(torch.bfloat16).float()
            routed_ref[t] += top_w[t, k_i] * out_e

    mlp_ref = (routed_ref.to(torch.bfloat16).float() + sh_out_ref.float()).to(torch.bfloat16)
    res2_ref = (mlp_ref.float() + res1_ref.float()).to(torch.bfloat16)
    x_out_ref = (
        res2_ref.float()
        * torch.rsqrt((res2_ref.float() ** 2).mean(dim=-1, keepdim=True) + RMS_EPS)
        * gamma_next.float()
    ).to(torch.bfloat16)

    mutable = [
        arena,
        idx_arena,
        idx_logits,
        idx_counters,
        idx_clear_meta,
        ugpu_state,
        part_o,
        part_stats,
        fc2out,
        counters,
        h1,
        h2,
        tk,
        ih,
        ib,
        err,
        cap_buf,
    ]
    if smid_map is not None:
        mutable.append(smid_map)
    kv_cache = [kv_pool, idx_pool]
    kv_len_t = torch.tensor([L], dtype=torch.int32, device=device)

    keep_alive = [
        wq_p,
        wo_p,
        wo_s,
        gu_p,
        dn_p,
        ix_wq_p,
        w_kva,
        w_kb,
        w_vb,
        w_r,
        w13,
        w2,
        alpha1,
        alpha2,
        ssmax_s,
        ssmax_b,
        kv_norm,
        gamma_post,
        gamma_next,
        rope_tbl,
        ix_q_norm_w,
        ix_q_norm_b,
        ix_kn_w,
        ix_kn_b,
        ix_ss_s,
        ix_ss_b,
        idx_tmap,
        block_table,
    ]
    return dict(
        params=params,
        x_in=x_in,
        res_in=res_in,
        topk=topk_tensor,
        block_table=block_table,
        kv_len=kv_len_t,
        positions=positions,
        mutable=mutable,
        kv_cache=kv_cache,
        cap_buf=cap_buf,
        x_ref=x_out_ref,
        res_ref=res2_ref,
        ref_topk=ref_topk,
        keep_alive=keep_alive,
        h1=h1,
        h2=h2,
        ih=ih,
        ib=ib,
        tk=tk,
        arena=arena,
        idx_arena=idx_arena,
        idx_logits=idx_logits,
        idx_counters=idx_counters,
        idx_clear_meta=idx_clear_meta,
        ugpu_state=ugpu_state,
        counters=counters,
        part_o=part_o,
        part_stats=part_stats,
        fc2out=fc2out,
        leader=leader,
    )


@pytest.mark.skipif(not _is_sm107_vr200(), reason=_SKIP_REASON)
def test_sm107_native_info_and_sm_die_probe():
    """Verify compiled libth_ml4_megakernel.so ABI info and hardware SM -> die probe on VR200."""
    ops = mk._load_native_ops()
    info = ops.ml4_megakernel_info()
    (
        magic,
        num_params,
        arena_bytes,
        clear_units,
        smem_bytes,
        num_counters,
        hop_bytes,
        tk_bytes,
        idx_hist_bytes,
        max_keys,
        idx_bnd_bytes,
        idx_arena_bytes,
        idx_logits_bytes,
        idx_smem_bytes,
    ) = info
    assert magic == mk._MAGIC
    assert num_params == len(mk._FIELDS)
    assert arena_bytes == mk._ARENA_BYTES
    assert clear_units == mk._CLEAR_UNITS
    assert smem_bytes == mk._SMEM_BYTES
    assert num_counters == mk._NUM_COUNTERS
    assert hop_bytes == mk._HOP_BYTES
    assert tk_bytes == mk._TK_BYTES
    assert idx_hist_bytes == mk._IDX_HIST_BYTES
    assert max_keys == mk._MAX_KEYS
    assert idx_bnd_bytes == mk._IDX_BND_BYTES
    assert idx_arena_bytes == mk._IDX_ARENA_BYTES
    assert idx_logits_bytes == mk._IDX_LOGITS_BYTES
    assert idx_smem_bytes == mk._IDX_SMEM_BYTES

    dev = torch.device("cuda", torch.cuda.current_device())
    sm_map, n_die0 = mk._probe_hardware_sm_die_map(ops, dev)
    assert sm_map is not None, "Expected driver-attested Die-Aware SMID bijection on dual-die VR200"
    assert n_die0 == 106
    valid = sm_map[sm_map >= 0]
    assert int(valid.numel()) == 212
    assert set(valid.cpu().tolist()) == set(range(212))
    sample = torch.zeros((4 * 1024 * 1024,), dtype=torch.uint8, device=dev)
    placed = mk._try_split_die_copy(sample, True)
    assert mk._pointer_locality_domain(placed.data_ptr()) == 0
    assert mk._pointer_locality_domain(placed.data_ptr() + 2 * 1024 * 1024) == 1


@pytest.mark.skipif(not _is_sm107_vr200(), reason=_SKIP_REASON)
@pytest.mark.parametrize("tpb", [32, 64])
def test_sm107_indexer_tmap_encoding(tpb: int):
    ops = mk._load_native_ops()
    dev = torch.device("cuda", torch.cuda.current_device())
    num_blocks = 128
    pool = torch.zeros((num_blocks, tpb, 1, 132), dtype=torch.uint8, device=dev)
    rows = (num_blocks * tpb * 132) // 128
    tmap = ops.ml4_megakernel_indexer_tmap(pool, rows, tpb)
    assert tuple(tmap.shape) == (128,) and tmap.dtype == torch.uint8 and tmap.device.type == "cpu"


@pytest.mark.skipif(not _is_sm107_vr200(), reason=_SKIP_REASON)
@pytest.mark.parametrize(
    "case_name,leader,nope,kv_owner,tpb,L,die_aware,force_ties",
    [
        ("plain_rope_owner_tpb32_L512", False, False, True, 32, 512, False, False),
        ("plain_rope_owner_tpb32_L2048_die", False, False, True, 32, 2048, True, False),
        ("plain_rope_owner_tpb64_L512_die", False, False, True, 64, 512, True, False),
        ("plain_rope_owner_tpb64_L2048_nodie", False, False, True, 64, 2048, False, False),
        ("leader_rope_owner_L512_die", True, False, True, 32, 512, True, False),
        ("leader_rope_owner_L2048_die", True, False, True, 32, 2048, True, False),
        ("leader_rope_owner_tpb64_L2048_die", True, False, True, 64, 2048, True, False),
        ("leader_nope_owner_L512_nodie", True, True, True, 32, 512, False, False),
        ("leader_nope_owner_L2048_die", True, True, True, 32, 2048, True, False),
        ("leader_nope_sharer_L512_die", True, True, False, 32, 512, True, False),
        ("leader_nope_sharer_L2048_die", True, True, False, 64, 2048, True, False),
        ("leader_rope_owner_L64K_cand_die", True, False, True, 32, 4096 * 16, True, False),
        ("leader_rope_owner_L64K_cand_nodie", True, False, True, 32, 4096 * 16, False, False),
        ("leader_rope_owner_L64K_cand_tpb64", True, False, True, 64, 4096 * 16, True, False),
        ("leader_nope_owner_L64K_cand_die", True, True, True, 32, 4096 * 16, True, False),
        ("leader_nope_owner_L64K_cand_tpb64", True, True, True, 64, 4096 * 16, True, False),
        ("leader_nope_sharer_L64K_cand_die", True, True, False, 32, 4096 * 16, True, False),
        ("leader_nope_sharer_L64K_cand_tpb64", True, True, False, 64, 4096 * 16, True, False),
        ("leader_rope_owner_L128K_cand_die", True, False, True, 32, 8192 * 16, True, False),
        ("leader_rope_owner_L128K_cand_tpb64", True, False, True, 64, 8192 * 16, True, False),
        ("leader_nope_owner_L128K_cand_die", True, True, True, 32, 8192 * 16, True, False),
        ("leader_nope_sharer_L128K_cand_die", True, True, False, 32, 8192 * 16, True, False),
        ("leader_rope_owner_L128K_ties_fallback", True, False, True, 32, 8192 * 16, True, True),
        ("leader_rope_owner_L370K_cand_die", True, False, True, 32, 23680 * 16, True, False),
    ],
)
def test_sm107_single_gpu_full_layer_accuracy(
    case_name: str,
    leader: bool,
    nope: bool,
    kv_owner: bool,
    tpb: int,
    L: int,
    die_aware: bool,
    force_ties: bool,
):
    """Execute megakernel end-to-end across all layer classes, contexts, and tie modes."""
    ops = mk._load_native_ops()
    dev = torch.device("cuda", torch.cuda.current_device())
    sm_map, n_die0 = (
        mk._probe_hardware_sm_die_map(ops, dev) if die_aware else (None, mk._DEFAULT_NDIE0)
    )
    bundle = _build_synthetic_layer(
        ops,
        dev,
        leader=leader,
        nope=nope,
        kv_owner=kv_owner,
        tpb=tpb,
        L=L,
        die_aware=die_aware,
        smid_map=sm_map,
        n_die0=n_die0,
        seed=123 + (L % 997),
        force_ties=force_ties,
    )
    x_out, res_out = ops.ml4_megakernel(
        bundle["params"],
        bundle["x_in"],
        bundle["res_in"],
        bundle["topk"],
        bundle["block_table"],
        bundle["kv_len"],
        bundle["positions"],
        bundle["mutable"],
        bundle["kv_cache"],
        True,
        leader,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(x_out).all() and torch.isfinite(res_out).all()

    cos_x = F.cosine_similarity(
        x_out.float().reshape(-1), bundle["x_ref"].float().reshape(-1), dim=0
    ).item()
    cos_res = F.cosine_similarity(
        res_out.float().reshape(-1), bundle["res_ref"].float().reshape(-1), dim=0
    ).item()
    assert cos_res >= 0.999, f"{case_name}: res_out cosine={cos_res:.6f} < 0.999"
    assert cos_x >= 0.999, f"{case_name}: x_out cosine={cos_x:.6f} < 0.999"
    assert torch.equal(bundle["cap_buf"], res_out), f"{case_name}: inline DSpark capture mismatch"

    if leader and L > 2048:
        for t in range(4):
            got_set = set(bundle["topk"][t].cpu().tolist())
            ref_set = set(bundle["ref_topk"][t].cpu().tolist())
            overlap = len(got_set & ref_set)
            min_ok = 2048 if force_ties else 2040
            assert overlap >= min_ok, f"{case_name} tok {t}: topk overlap={overlap}/2048"


@pytest.mark.skipif(not _is_sm107_vr200(), reason=_SKIP_REASON)
@pytest.mark.parametrize("leader,L", [(False, 512), (True, 512), (True, 4096 * 16)])
def test_sm107_cuda_graph_capture_and_replay(leader: bool, L: int):
    """Verify CUDA Graph capture & 5x bit-exact replay on VR200 for Plain and Leader layers."""
    ops = mk._load_native_ops()
    dev = torch.device("cuda", torch.cuda.current_device())
    sm_map, n_die0 = mk._probe_hardware_sm_die_map(ops, dev)
    bundle = _build_synthetic_layer(
        ops,
        dev,
        leader=leader,
        nope=False,
        kv_owner=True,
        tpb=32,
        L=L,
        die_aware=sm_map is not None,
        smid_map=sm_map,
        n_die0=n_die0,
        seed=777,
    )

    def reset_rank0_slots():
        bundle["h1"].zero_()
        bundle["h1"].view(4, 16, 6144 // 32, 4, 16)[:, 0].fill_(-0.0)
        bundle["h2"].zero_()
        bundle["h2"].view(4, 4, 16, 6144 // 2)[:, :, 0].fill_(-0.0)
        bundle["tk"].fill_(-0.0)
        bundle["ih"].zero_()
        bundle["ih"][:, :, :, 0, :].fill_(-1)
        bundle["ib"].fill_(-1)
        bundle["arena"].fill_(-0.0)
        bundle["idx_arena"].fill_(-0.0)
        bundle["idx_arena"].view(torch.uint8).view(4, mk._IDX_ARENA_BYTES)[
            :, mk._IDX_ARENA_HIST0_OFF :
        ].zero_()
        bundle["counters"].zero_()
        bundle["idx_counters"].zero_()
        bundle["idx_clear_meta"].zero_()

    x_eager, r_eager = ops.ml4_megakernel(
        bundle["params"],
        bundle["x_in"],
        bundle["res_in"],
        bundle["topk"],
        bundle["block_table"],
        bundle["kv_len"],
        bundle["positions"],
        bundle["mutable"],
        bundle["kv_cache"],
        True,
        leader,
    )
    torch.cuda.synchronize()
    x_eager = x_eager.clone()
    r_eager = r_eager.clone()

    reset_rank0_slots()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x_g, r_g = ops.ml4_megakernel(
            bundle["params"],
            bundle["x_in"],
            bundle["res_in"],
            bundle["topk"],
            bundle["block_table"],
            bundle["kv_len"],
            bundle["positions"],
            bundle["mutable"],
            bundle["kv_cache"],
            True,
            leader,
        )
    for _ in range(5):
        reset_rank0_slots()
        g.replay()
        torch.cuda.synchronize()
        assert torch.allclose(x_g.float(), x_eager.float(), atol=1e-2, rtol=1e-2)
        assert torch.allclose(r_g.float(), r_eager.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not _is_sm107_vr200(), reason=_SKIP_REASON)
@pytest.mark.parametrize("L", [512, 2048, 4096 * 16])
def test_sm107_chained_pdl_4layer_heterogeneous_graph(L: int):
    """Verify 4-layer PDL chain (Plain->Leader RoPE->NoPE Owner->NoPE Sharer) 5/5 bit-exact in CUDA Graph."""
    ops = mk._load_native_ops()
    dev = torch.device("cuda", torch.cuda.current_device())
    sm_map, n_die0 = mk._probe_hardware_sm_die_map(ops, dev)
    da = sm_map is not None
    chain_specs = [
        (False, False, True),
        (True, False, True),
        (True, True, True),
        (True, True, False),
    ]
    base_b = _build_synthetic_layer(
        ops,
        dev,
        leader=True,
        nope=False,
        kv_owner=True,
        tpb=32,
        L=L,
        die_aware=da,
        smid_map=sm_map,
        n_die0=n_die0,
        seed=999,
    )
    layers = []
    idx_ord = 0
    for idx, (ld, np_, ko) in enumerate(chain_specs):
        lb = _build_synthetic_layer(
            ops,
            dev,
            leader=ld,
            nope=np_,
            kv_owner=ko,
            tpb=32,
            L=L,
            die_aware=da,
            smid_map=sm_map,
            n_die0=n_die0,
            seed=999 + idx,
        )
        p_host = lb["params"].cpu()
        p_host[mk._P["Arena"]] = base_b["arena"].data_ptr()
        p_host[mk._P["IdxArena"]] = base_b["idx_arena"].data_ptr()
        p_host[mk._P["IdxLogits"]] = base_b["idx_logits"].data_ptr()
        p_host[mk._P["IdxCounters"]] = base_b["idx_counters"].data_ptr()
        p_host[mk._P["IdxClearMeta"]] = base_b["idx_clear_meta"].data_ptr()
        p_host[mk._P["UgpuState"]] = base_b["ugpu_state"].data_ptr()
        p_host[mk._P["PartO"]] = base_b["part_o"].data_ptr()
        p_host[mk._P["PartStats"]] = base_b["part_stats"].data_ptr()
        p_host[mk._P["Fc2out"]] = base_b["fc2out"].data_ptr()
        p_host[mk._P["Counters"]] = base_b["counters"].data_ptr()
        p_host[mk._P["H1Mc"]] = base_b["h1"].data_ptr()
        p_host[mk._P["H1Uc"]] = base_b["h1"].data_ptr()
        p_host[mk._P["H2Mc"]] = base_b["h2"].data_ptr()
        p_host[mk._P["H2Uc"]] = base_b["h2"].data_ptr()
        p_host[mk._P["TkMc"]] = base_b["tk"].data_ptr()
        p_host[mk._P["TkUc"]] = base_b["tk"].data_ptr()
        p_host[mk._P["IdxHistMc"]] = base_b["ih"].data_ptr()
        p_host[mk._P["IdxHistUc"]] = base_b["ih"].data_ptr()
        p_host[mk._P["IdxBndMc"]] = base_b["ib"].data_ptr()
        p_host[mk._P["IdxBndUc"]] = base_b["ib"].data_ptr()
        p_host[mk._P["Copy"]] = idx
        p_host[mk._P["NextCopy"]] = (idx + 1) % 4
        p_host[mk._P["PrevCopy"]] = (idx - 1) % 4
        p_host[mk._P["NumCopies"]] = 4
        if ld:
            p_host[mk._P["IdxCopy"]] = idx_ord % 3
            p_host[mk._P["NextIdxCopy"]] = (idx_ord + 1) % 3
            idx_ord += 1
        fl = int(p_host[mk._P["Flags"]])
        if idx > 0:
            fl |= mk._FLAG_EARLY_X
        if idx < 3:
            fl |= mk._FLAG_NEXT_CHAINED
        p_host[mk._P["Flags"]] = fl
        lb["params_chain"] = p_host.to(dev)
        layers.append(lb)

    def reset_chain():
        base_b["h1"].zero_()
        base_b["h1"].view(4, 16, 6144 // 32, 4, 16)[:, 0].fill_(-0.0)
        base_b["h2"].zero_()
        base_b["h2"].view(4, 4, 16, 6144 // 2)[:, :, 0].fill_(-0.0)
        base_b["tk"].fill_(-0.0)
        base_b["ih"].zero_()
        base_b["ih"][:, :, :, 0, :].fill_(-1)
        base_b["ib"].fill_(-1)
        base_b["arena"].fill_(-0.0)
        base_b["idx_arena"].fill_(-0.0)
        base_b["idx_arena"].view(torch.uint8).view(4, mk._IDX_ARENA_BYTES)[
            :, mk._IDX_ARENA_HIST0_OFF :
        ].zero_()
        base_b["counters"].zero_()
        base_b["idx_counters"].zero_()
        base_b["idx_clear_meta"].zero_()

    def run_4chain(x0, r0):
        x, r = x0, r0
        for lb in layers:
            x, r = ops.ml4_megakernel(
                lb["params_chain"],
                x,
                r,
                lb["topk"],
                lb["block_table"],
                lb["kv_len"],
                lb["positions"],
                base_b["mutable"],
                lb["kv_cache"],
                True,
                lb["leader"],
            )
        return x, r

    reset_chain()
    x_e, r_e = run_4chain(layers[0]["x_in"], layers[0]["res_in"])
    torch.cuda.synchronize()
    x_e, r_e = x_e.clone(), r_e.clone()
    assert torch.isfinite(x_e).all() and torch.isfinite(r_e).all()

    reset_chain()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x_g, r_g = run_4chain(layers[0]["x_in"], layers[0]["res_in"])
    for _ in range(5):
        reset_chain()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(x_g, x_e) and torch.equal(r_g, r_e)
