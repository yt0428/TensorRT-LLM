// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Lean main-layer stages included inside namespace ml4 by ml4Megakernel.cu.
// Zero indexer code inside ml4LayerKernel -> tiny SASS footprint & 0B stack spills across all stages.

// ----------------------------------------------------------------------------------------------------------------
// Cross-layer PDL helpers: wait on previous copy's Clear counter, poll x' & xSsq from contiguous row-major kArX
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void waitPrevCopyClear(Ctx const& ctx)
{
    if (threadIdx.x == 0)
    {
        uint32_t const* c
            = reinterpret_cast<uint32_t const*>(ctx.p[kPCounters]) + ctx.prevCopy * kNumCounters + kCClear;
        uint64_t t0 = 0;
        uint32_t v;
        while ((v = ldAcquire(c)) < static_cast<uint32_t>(kCountedClear))
        {
            spinCheck(ctx, t0, 99, kCClear, static_cast<int>(v));
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void pollXRcp(Ctx const& ctx, float* sRcp, int code)
{
    int const tid = static_cast<int>(threadIdx.x);
    if (tid < kTok)
    {
        uint4 const v = poll4(ctx, arena<float const>(ctx, kArXSsq) + 4 * tid, code);
        float const s
            = (((0.f + __uint_as_float(v.x)) + __uint_as_float(v.y)) + __uint_as_float(v.z)) + __uint_as_float(v.w);
        sRcp[tid] = rsqrtf(s / static_cast<float>(kHidden) + ctx.rmsEps);
    }
    __syncthreads();
}

__device__ __forceinline__ float hopRcp(Ctx const& ctx, int t, int code)
{
    uint4 const all = poll4(ctx, arena<float const>(ctx, kArSsq1) + 4 * t, code);
    float const s
        = (((0.f + __uint_as_float(all.x)) + __uint_as_float(all.y)) + __uint_as_float(all.z)) + __uint_as_float(all.w);
    return rsqrtf(s / static_cast<float>(kHidden) + ctx.rmsEps);
}

// ----------------------------------------------------------------------------------------------------------------
// Shared TCGen05 MXFP8 Single-N-Tile Helper (128 output rows x nKTiles*128 K cols, 3-slot TMA ring, 0 spill!)
// Uses ring barriers barRing0[0..2], barFree0[0..2], barMma, and barAct (barriers 24..31).
// Returns TMEM D[0..8) ready after waitBar(barMma) (caller drains with warps 4..7 via tmemLd8).
// ----------------------------------------------------------------------------------------------------------------
template <int kRing, bool kPollAct, bool kLayerNorm>
__device__ __forceinline__ void runMxf8Tile(Ctx const& ctx, uint8_t const* wTileBase, uint8_t const* actBase,
    int actLdCols, float const* sRcp, float const* sLnMean, float const* sLnRstd, __nv_bfloat16 const* qnW,
    __nv_bfloat16 const* qnB, int kTile0, int nKTiles, uint32_t smemOff, uint32_t tmem, uint32_t barRing0,
    uint32_t barFree0, uint32_t barMma, uint32_t barAct, int code)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    // SMEM layout at smemOff: [kRing * 17408 B 1024B-aligned ring] | [nKTiles * 1024 B smemB] | [nKTiles * 128 B
    // smemSfb]
    uint32_t const sAOff = smemOff;
    uint32_t const sBOff = sAOff + kRing * kMxf8SlotStride;
    uint32_t const sSfbOff = sBOff + static_cast<uint32_t>(nKTiles) * 1024u;
    uint8_t* smemB = ctx.smem + sBOff;
    uint8_t* smemSfb = ctx.smem + sSfbOff;
    uint32_t const aSmem = ctx.smemBase + sAOff;
    uint32_t const bSmem = ctx.smemBase + sBOff;
    uint32_t const sfbSmem = ctx.smemBase + sSfbOff;

    if (tid < 32)
    {
        if (tid < kRing)
        {
            mbarReinit(barRing0 + 8 * tid, 1);
            mbarReinit(barFree0 + 8 * tid, 1);
        }
        else if (tid == kRing)
        {
            mbarReinit(barMma, 1);
        }
        else if (tid == kRing + 1)
        {
            mbarReinit(barAct, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    bool const evictFirst = (ctx.flags & kFlagEvictFirst) != 0;
    uint64_t const policy = evictFirst ? policyEvictFirst() : 0ull;

    if (warp == 0)
    {
        // Producer warp: stream nKTiles MXFP8 pieces (16,896 B each) into 1024B-aligned kRing-slot ring.
        if (lane == 0)
        {
            for (int ki = 0; ki < nKTiles; ++ki)
            {
                int const st = ki % kRing;
                if (ki >= kRing)
                {
                    waitBar(ctx, barFree0 + 8 * st, static_cast<uint32_t>(((ki / kRing) - 1) & 1), code);
                }
                uint32_t const dst = aSmem + static_cast<uint32_t>(st) * kMxf8SlotStride;
                uint8_t const* src = wTileBase + static_cast<size_t>(kTile0 + ki) * kMxf8PieceBytes;
                uint32_t const bar = barRing0 + 8 * st;
                mbarExpectTx(bar, kMxf8PieceBytes);
                if (evictFirst)
                {
                    bulkCopyHint(dst, src, kMxf8PieceBytes, bar, policy);
                }
                else
                {
                    bulkCopy(dst, src, kMxf8PieceBytes, bar);
                }
            }
        }
    }
    else if (warp == 1)
    {
        // Consumer MMA warp: wait for activations, then stream TCGen05 MXFP8 MMAs overlapped with TMA ring.
        mbarWaitSpin(barAct, 0);
        tcFenceAfter();
        if (electOne())
        {
            for (int ki = 0; ki < nKTiles; ++ki)
            {
                utccp1Rows8(
                    tmem + kTmemSfb + 4u * static_cast<uint32_t>(ki), sfbSmem + 128u * static_cast<uint32_t>(ki));
            }
        }
        __syncwarp();
        for (int ki = 0; ki < nKTiles; ++ki)
        {
            int const st = ki % kRing;
            waitBar(ctx, barRing0 + 8 * st, static_cast<uint32_t>((ki / kRing) & 1), code + 1);
            tcFenceAfter();
            if (electOne())
            {
                uint32_t const aSlot = aSmem + static_cast<uint32_t>(st) * kMxf8SlotStride;
                uint32_t const sfaTmem = tmem + kTmemSfa + 4u * static_cast<uint32_t>(st);
                uint32_t const sfbTmem = tmem + kTmemSfb + 4u * static_cast<uint32_t>(ki);
                utccp1(sfaTmem, aSlot + kMxf8WBytes);
                uint32_t const bTile = bSmem + 1024u * static_cast<uint32_t>(ki);
#pragma unroll
                for (int kb = 0; kb < 4; ++kb)
                {
                    mmaMxf8(tmem + kTmemD, aSlot + 32u * static_cast<uint32_t>(kb),
                        bTile + 32u * static_cast<uint32_t>(kb), sfaTmem, sfbTmem, static_cast<uint32_t>(kb),
                        (ki > 0 || kb > 0) ? 1u : 0u);
                }
                if (ki + kRing < nKTiles)
                {
                    commit1(barFree0 + 8 * st);
                }
            }
            __syncwarp();
        }
        if (electOne())
        {
            commit1(barMma);
        }
        __syncwarp();
    }
    else if (warp >= 4)
    {
        quantActMxf8Warps47<kPollAct, kLayerNorm>(
            ctx, actBase, actLdCols, sRcp, sLnMean, sLnRstd, qnW, qnB, kTile0, nKTiles, smemB, smemSfb, code + 2);
        fenceProxyAsyncSmem();
        asm volatile("bar.sync 2, 128;" ::: "memory");
        if (tid == 128)
        {
            mbarArrive(barAct);
        }
    }
    waitBar(ctx, barMma, 0, code + 3);
    __syncthreads();
    tcFenceAfter();
}

// ----------------------------------------------------------------------------------------------------------------
// Stage 1A (Plain layers): 64-CTA TCGen05 MXFP8 Wq (4 heads x 16 K-splits = 3 K-tiles / 50.7 KB per CTA!)
// Plus head K-reduce + SSMax + Q-RoPE/NoPE on split 0 of each head (4 CTAs).
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageWqPlain(
    Ctx const& ctx, int unit, uint32_t wOff, __nv_bfloat16 const* x, int const* positions)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const head = unit / kWqPlainSplits;
    int const kSplit = unit % kWqPlainSplits;
    int const kTile0 = kSplit * kWqPlainKPerCta;
    bool const early = (ctx.flags & kFlagEarlyX) != 0;

    float* sRcp = reinterpret_cast<float*>(ctx.smem + kSmemSmall + kSmallScalars);
    uint8_t const* actBase = nullptr;
    if (early)
    {
        waitPrevCopyClear(ctx);
        pollXRcp(ctx, sRcp, 101);
        actBase = arena<uint8_t const>(ctx, kArX);
    }
    else
    {
        actBase = reinterpret_cast<uint8_t const*>(x);
    }

    uint32_t const tmem = *reinterpret_cast<uint32_t const*>(ctx.smem + kSmemSmall + kSmallScalars + 4 * kScalarTmem);
    uint32_t const barRing0 = ctx.smemBase + kSmemSmall + kSmallBars + kBarMoeRing0 * 8;
    uint32_t const barFree0 = ctx.smemBase + kSmemSmall + kSmallBars + kBarMoeFree0 * 8;
    uint32_t const barMma = ctx.smemBase + kSmemSmall + kSmallBars + kBarMoeMma * 8;
    uint32_t const barAct = ctx.smemBase + kSmemSmall + kSmallBars + kBarAux * 8;
    uint8_t const* wHead
        = ptr<uint8_t const>(ctx, kPWq) + static_cast<size_t>(head) * kHiddenBlocksFp8 * kMxf8PieceBytes;

    if (early)
    {
        runMxf8Tile<3, true, false>(ctx, wHead, actBase, kHidden, sRcp, nullptr, nullptr, nullptr, nullptr, kTile0,
            kWqPlainKPerCta, wOff, tmem, barRing0, barFree0, barMma, barAct, 100);
    }
    else
    {
        runMxf8Tile<3, false, false>(ctx, wHead, actBase, kHidden, nullptr, nullptr, nullptr, nullptr, nullptr, kTile0,
            kWqPlainKPerCta, wOff, tmem, barRing0, barFree0, barMma, barAct, 100);
    }

    // Drain [128 dims, 4 tokens] from TMEM on warps 4..7 (token-contiguous uint4 -> 1-wave pollStrided<7>!).
    uint8_t* partBase = arena<uint8_t>(ctx, kArWqPart) + static_cast<size_t>(head) * kWqPlainSplits * 128 * 16;
    if (warp >= 4)
    {
        int const q = warp - 4;
        int const dim = 32 * q + lane;
        float v[8];
        tmemLd8(tmem + kTmemD + (static_cast<uint32_t>(32 * q) << 16), v);
        if (kSplit != 0)
        {
            st128(partBase + (static_cast<size_t>(kSplit) * 128 + dim) * 16,
                make_uint4(lamport(__float_as_uint(v[0])), lamport(__float_as_uint(v[1])),
                    lamport(__float_as_uint(v[2])), lamport(__float_as_uint(v[3]))));
        }
        else
        {
            // Split 0 polls all 7 remote splits in ONE parallel wave (0.25 us vs 10.8 us serialized poll1!).
            uint4 pv[kWqPlainSplits - 1];
            pollStrided<kWqPlainSplits - 1>(ctx, pv, partBase + (128 + dim) * 16, 128 * 16, 102);
            float sum[kTok] = {v[0], v[1], v[2], v[3]};
#pragma unroll
            for (int s = 0; s < kWqPlainSplits - 1; ++s)
            {
                sum[0] += __uint_as_float(pv[s].x);
                sum[1] += __uint_as_float(pv[s].y);
                sum[2] += __uint_as_float(pv[s].z);
                sum[3] += __uint_as_float(pv[s].w);
            }
            float const ssS = __bfloat162float(__ldg(ptr<__nv_bfloat16 const>(ctx, kPSsmaxS) + head));
            float const ssB = __bfloat162float(__ldg(ptr<__nv_bfloat16 const>(ctx, kPSsmaxB) + head));
            bool const isNope = (ctx.flags & kFlagNope) != 0;
#pragma unroll
            for (int t = 0; t < kTok; ++t)
            {
                int const pos = __ldg(positions + t);
                float const sc = ssmaxScaleBf16(pos, ssS, ssB);
                float const val = bf16Round(bf16Round(sum[t]) * sc);
                float const partner = __shfl_xor_sync(0xffffffffu, val, 1);
                if ((lane & 1) == 0)
                {
                    float const v0 = val;
                    float const v1 = partner;
                    if (!isNope && dim >= kNopeDim)
                    {
                        int const pairIdx = (dim - kNopeDim) >> 1;
                        float const* cs = ptr<float const>(ctx, kPRope) + static_cast<size_t>(pos) * 64;
                        float const c = __ldg(cs + pairIdx);
                        float const s = __ldg(cs + 32 + pairIdx);
                        uint32_t const w = lamport(packBf16(v0 * c - v1 * s, v0 * s + v1 * c));
                        st32(arena<uint8_t>(ctx, kArFusedQ)
                                + ((t * kHeads + head) * kLatDim + kKvLora + (dim - kNopeDim)) * 2,
                            w);
                    }
                    else
                    {
                        st32(arena<uint8_t>(ctx, kArQNope) + ((t * kHeads + head) * kQkDim + dim) * 2,
                            lamport(packBf16(v0, v1)));
                        if (isNope && dim >= kNopeDim)
                        {
                            st32(arena<uint8_t>(ctx, kArFusedQ)
                                    + ((t * kHeads + head) * kLatDim + kKvLora + (dim - kNopeDim)) * 2,
                                0u);
                        }
                    }
                }
            }
        }
    }
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
}

// ----------------------------------------------------------------------------------------------------------------
// Stage 1B (Plain layers): WkvA (BF16 [576, 6144], reading contiguous row-major kArX or x)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageWkvA(Ctx const& ctx, int tile, uint32_t wOff, uint32_t bar, __nv_bfloat16 const* x)
{
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    constexpr int kPer = kHiddenChunksBf16 / kWarps; // 12 chunks/warp
    bool const early = (ctx.flags & kFlagEarlyX) != 0;
    uint4 b[kPer][2];
    float* sRcp = reinterpret_cast<float*>(ctx.smem + kSmemSmall + kSmallScalars);
    if (early)
    {
        waitPrevCopyClear(ctx);
        pollB<kPer>(ctx, b, arena<uint8_t const>(ctx, kArX), kHidden, warp * kPer, 103);
        pollXRcp(ctx, sRcp, 104);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < kPer; ++i)
        {
            loadB(b[i], x, kHidden, warp * kPer + i);
        }
    }
    waitBar(ctx, bar, 0, 105);

    float part[4][4] = {};
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
        mmaChunkBf16(part[i & 3], ctx.smemBase + wOff + (warp * kPer + i) * kBf16Block, b[i]);
    }
    float acc[1][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
    {
        acc[0][r] = (part[0][r] + part[1][r]) + (part[2][r] + part[3][r]);
    }
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 4096);
    reduceWarps<1>(ctx, acc, out);
    storeTileBf16(out, arena<uint8_t>(ctx, kArKvRaw), kLatDim, 16 * tile, early ? sRcp : nullptr);
}

// ----------------------------------------------------------------------------------------------------------------
// Stage 1C: Kb Absorption (q_nope @ k_b -> kArFusedQ[0..511]); supports RoPE (K=64) and NoPE (K=128)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageKb(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    int const lane = static_cast<int>(threadIdx.x) & 31;
    int const head = unit / kKbUnitsPerHead;
    int const sub = unit % kKbUnitsPerHead;
    bool const isNope = (ctx.flags & kFlagNope) != 0;
    uint8_t const* qSrc = arena<uint8_t const>(ctx, kArQNope) + head * kQkDim * 2;

    if (!isNope)
    {
        uint4 b[1][2];
        if (warp < 4)
        {
            pollB<1>(ctx, b, qSrc, kHeads * kQkDim, 0, 106);
        }
        waitBar(ctx, bar, 0, 107);
        if (warp < 4)
        {
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            mmaChunkBf16(acc, ctx.smemBase + wOff + warp * kBf16Block, b[0]);
            float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed) + warp * 64;
            if ((lane & 3) < 2)
            {
                int const g = lane >> 2;
                int const c = lane & 1;
                out[g * kTok + 2 * c] = acc[0];
                out[g * kTok + 2 * c + 1] = acc[1];
                out[(g + 8) * kTok + 2 * c] = acc[2];
                out[(g + 8) * kTok + 2 * c + 1] = acc[3];
            }
            __syncwarp();
            int const tok = lane / 8;
            int const pr = lane % 8;
            uint32_t const w = lamport(packBf16(out[(2 * pr) * kTok + tok], out[(2 * pr + 1) * kTok + tok]));
            st32(arena<uint8_t>(ctx, kArFusedQ) + ((tok * kHeads + head) * kLatDim + 64 * sub + 16 * warp + 2 * pr) * 2,
                w);
        }
    }
    else
    {
        int const tileIdx = warp & 3;
        int const chunkIdx = warp >> 2;
        uint4 b[1][2];
        pollB<1>(ctx, b, qSrc, kHeads * kQkDim, chunkIdx, 106);
        waitBar(ctx, bar, 0, 107);

        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        mmaChunkBf16(acc, ctx.smemBase + wOff + (tileIdx * 2 + chunkIdx) * kBf16Block, b[0]);
        float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed);
        if ((lane & 3) < 2)
        {
            int const g = lane >> 2;
            int const c = lane & 1;
            float* dst = red + (chunkIdx * 4 + tileIdx) * 64;
            dst[g * kTok + 2 * c] = acc[0];
            dst[g * kTok + 2 * c + 1] = acc[1];
            dst[(g + 8) * kTok + 2 * c] = acc[2];
            dst[(g + 8) * kTok + 2 * c + 1] = acc[3];
        }
        __syncthreads();
        if (threadIdx.x < 4 * kTok * 8)
        {
            int const ti = static_cast<int>(threadIdx.x) / 32;
            int const rem = static_cast<int>(threadIdx.x) % 32;
            int const tok = rem / 8;
            int const pr = rem % 8;
            float const* s0 = red + ti * 64;
            float const* s1 = red + (4 + ti) * 64;
            float const v0 = s0[(2 * pr) * kTok + tok] + s1[(2 * pr) * kTok + tok];
            float const v1 = s0[(2 * pr + 1) * kTok + tok] + s1[(2 * pr + 1) * kTok + tok];
            st32(arena<uint8_t>(ctx, kArFusedQ) + ((tok * kHeads + head) * kLatDim + 64 * sub + 16 * ti + 2 * pr) * 2,
                lamport(packBf16(v0, v1)));
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Stage 1D: KvAppend (kv_a RMSNorm + fp8_ds_mla 656B 4x128 quant + RoPE/NoPE)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageKvAppend(Ctx const& ctx, int const* blockTable, int const* positions)
{
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    int const lane = static_cast<int>(threadIdx.x) & 31;
    if (warp >= kTok)
    {
        return;
    }
    int const tok = warp;
    bool const isNope = (ctx.flags & kFlagNope) != 0;
    int const pos = __ldg(positions + tok);
    int const page = __ldg(blockTable + (pos >> ctx.tpbShift));
    int64_t const poolRow
        = static_cast<int64_t>(page) * ctx.p[kPStrideRows] + (pos & (ctx.tpb - 1)) + ctx.p[kPLayerRows];
    uint8_t* dstPool = ptr<uint8_t>(ctx, kPKvBase) + poolRow * kKvRowBytes;
    uint8_t* dstArena = arena<uint8_t>(ctx, kArKv) + tok * kKvRowBytes;

    uint8_t const* raw = arena<uint8_t const>(ctx, kArKvRaw) + tok * kLatDim * 2;
    uint8_t const* cSrc0 = raw + 32 * lane;
    uint8_t const* cSrc1 = raw + 32 * lane + 16;
    uint8_t const* kPeSrc = raw + kKvLora * 2 + 4 * lane;

    __nv_bfloat16 const* gamma = ptr<__nv_bfloat16 const>(ctx, kPGammaKv);
    uint4 const gm0 = __ldg(reinterpret_cast<uint4 const*>(gamma + 16 * lane));
    uint4 const gm1 = __ldg(reinterpret_cast<uint4 const*>(gamma + 16 * lane + 8));

    float cosv = 1.f;
    float sinv = 0.f;
    if (!isNope)
    {
        float const* cs = ptr<float const>(ctx, kPRope) + static_cast<size_t>(pos) * 64;
        cosv = __ldg(cs + lane);
        sinv = __ldg(cs + 32 + lane);
    }

    uint4 v0 = ldPoll128(cSrc0);
    uint4 v1 = ldPoll128(cSrc1);
    uint32_t vk = isNope ? 0u : ldPoll32(kPeSrc);
    {
        uint64_t t0 = 0;
        while (!warpAll(full4(v0) && full4(v1) && (isNope || vk != kEmpty)))
        {
            spinCheck(ctx, t0, 108, tok, lane);
            if (!full4(v0))
            {
                v0 = ldPoll128(cSrc0);
            }
            if (!full4(v1))
            {
                v1 = ldPoll128(cSrc1);
            }
            if (!isNope && vk == kEmpty)
            {
                vk = ldPoll32(kPeSrc);
            }
        }
    }

    float lat[16];
    uint32_t const vw[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float ssq = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        float const lo = bf16Lo(vw[j]);
        float const hi = bf16Hi(vw[j]);
        lat[2 * j] = lo;
        lat[2 * j + 1] = hi;
        ssq += lo * lo + hi * hi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
    }
    float const rstd = rsqrtf(ssq / static_cast<float>(kKvLora) + ctx.rmsEps);

    uint32_t const gw[8] = {gm0.x, gm0.y, gm0.z, gm0.w, gm1.x, gm1.y, gm1.z, gm1.w};
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        lat[2 * j] = bf16Round((lat[2 * j] * rstd) * bf16Lo(gw[j]));
        lat[2 * j + 1] = bf16Round((lat[2 * j + 1] * rstd) * bf16Hi(gw[j]));
        amax = fmaxf(amax, fmaxf(fabsf(lat[2 * j]), fabsf(lat[2 * j + 1])));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    float const ratio = fmaxf(__fdiv_rn(amax, 448.0f), 1.0e-4f);
    float const scale = pow2Ceil(ratio);
    float const inv = pow2Inv(scale);

    uint32_t codes[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        uint32_t const lo = toE4m3x2(__fmul_rn(lat[4 * q], inv), __fmul_rn(lat[4 * q + 1], inv));
        uint32_t const hi = toE4m3x2(__fmul_rn(lat[4 * q + 2], inv), __fmul_rn(lat[4 * q + 3], inv));
        codes[q] = lo | (hi << 16);
    }
    st128(dstPool + 16 * lane, make_uint4(codes[0], codes[1], codes[2], codes[3]));
    st128(dstArena + 16 * lane, make_uint4(lamport(codes[0]), lamport(codes[1]), lamport(codes[2]), lamport(codes[3])));

    if ((lane & 7) == 0)
    {
        int const tileIdx = lane >> 3;
        uint32_t const sb = __float_as_uint(scale);
        st32(dstPool + kKvScaleOff + 4 * tileIdx, sb);
        st32(dstArena + kKvScaleOff + 4 * tileIdx, lamport(sb));
    }

    uint32_t rPool = 0u;
    if (!isNope)
    {
        float const k0 = bf16Lo(vk);
        float const k1 = bf16Hi(vk);
        rPool = packBf16(k0 * cosv - k1 * sinv, k0 * sinv + k1 * cosv);
    }
    st32(dstPool + kKvRopeOff + 4 * lane, rPool);
    st32(dstArena + kKvRopeOff + 4 * lane, lamport(rPool));
}

// ----------------------------------------------------------------------------------------------------------------
// Stage 3A: Sparse MLA Decode on fp8_ds_mla 656B KV (4 tok x 32 blk = 128 CTAs, 64 keys/CTA)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageAttn(
    Ctx const& ctx, int unit, uint32_t wOff, int const* kvLenPtr, int const* blockTable, int const* topk)
{
    int const t = unit / kAttnBlocks;
    int const blk = unit % kAttnBlocks;
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid >> 5;
    int const lane = tid & 31;

    uint8_t* sKv = ctx.smem + wOff;
    uint8_t* sQ = sKv + kAttnKeys * kKvSmemRowBytes;
    uint8_t* sP = sQ + 8 * kQSmemRowBytes;
    float* sHalfS = reinterpret_cast<float*>(sP + 4 * 8 * kPRowBytes);
    float* sRedMax = sHalfS + 256;
    float* sRedSum = sRedMax + 32;
    float* sMaxScale = sRedSum + 32;
    int* sFreshAll = reinterpret_cast<int*>(sMaxScale + 4);
    uint32_t const barGather = ctx.smemBase + kSmemSmall + kSmallBars + (kBarAttn0 + warp) * 8;

    auto issueOproj = [&]()
    {
        if (tid == 0)
        {
            int* sRoles = reinterpret_cast<int*>(ctx.smem + kSmemSmall + kSmallRoles);
#pragma unroll 1
            for (int j = 0; j < kMaxRoles && sRoles[j] != 0; ++j)
            {
                if (sRoles[2 * kMaxRoles + j] == -4)
                {
                    issueUnitWeights(ctx, j, ctx.smemBase + kSmemSmall + kSmallBars + 8 * j);
                    sRoles[2 * kMaxRoles + j] = -3;
                }
            }
        }
    };

    int const L = __ldg(kvLenPtr);
    int const numValid = min(L - (kTok - 1 - t), kTopK);
    if (blk * kAttnKeys >= numValid)
    {
        issueOproj();
        float* partO = ptr<float>(ctx, kPPartO) + static_cast<size_t>(t * kAttnBlocks + blk) * kHeads * kKvLora;
        for (int i = tid; i < kHeads * kKvLora / 2; i += kThreads)
        {
            st64(partO + 2 * i, make_uint2(0u, 0u));
        }
        if (tid < kHeads)
        {
            st64(ptr<float>(ctx, kPPartStats) + ((t * kAttnBlocks + blk) * kHeads + tid) * 2,
                make_uint2(lamport(__float_as_uint(-INFINITY)), 0u));
        }
        return;
    }

    int const slot = blk * kAttnKeys + warp * 8 + lane;
    int P = -1;
    bool const isLeader = (ctx.flags & kFlagLeader) != 0;
    if (isLeader)
    {
        if (L <= kTopK)
        {
            P = (lane < 8 && slot < numValid) ? slot : -1;
        }
        else if (lane < 8)
        {
            // Leader topk lives in idxTkUc[ctx.idxCopy] (written by ml4LeaderFrontKernel).
            uint32_t const* tkSrc = reinterpret_cast<uint32_t const*>(
                                        ptr<uint8_t const>(ctx, kPTkUc) + static_cast<size_t>(ctx.idxCopy) * kTkBytes)
                + t * kTopK + slot;
            P = static_cast<int>(poll1(ctx, tkSrc, 170));
        }
        if (lane < 8)
        {
            const_cast<int*>(topk)[t * kTopK + slot] = P;
        }
    }
    else if (lane < 8)
    {
        P = __ldcg(topk + t * kTopK + slot);
    }

    bool const kvOwner = (ctx.flags & kFlagKvOwner) != 0;
    bool valid = false;
    bool fresh = false;
    int freshIdx = 0;
    int64_t row = 0;
    if (lane < 8)
    {
        int const block = P >> ctx.tpbShift;
        int page = -1;
        if (P >= 0 && block < static_cast<int>(ctx.p[kPMaxBlocks]))
        {
            page = __ldg(blockTable + block);
        }
        valid = slot < numValid && page >= 0;
        fresh = kvOwner && valid && P >= L - kTok;
        freshIdx = P - (L - kTok);
        row = valid ? static_cast<int64_t>(page) * ctx.p[kPStrideRows] + (P & (ctx.tpb - 1)) + ctx.p[kPLayerRows] : 0;
    }
    uint32_t const validMask = __ballot_sync(0xffffffffu, valid);
    uint32_t const freshMask = __ballot_sync(0xffffffffu, fresh);
    uint8_t const* kvBase = ptr<uint8_t const>(ctx, kPKvBase);
    if (lane == 0)
    {
        mbarExpectTx(barGather, static_cast<uint32_t>(__popc(validMask & ~freshMask)) * kKvRowBytes);
    }
    __syncwarp();
    if (lane < 8 && !valid)
    {
        uint4* z = reinterpret_cast<uint4*>(sKv + (warp * 8 + lane) * kKvSmemRowBytes);
#pragma unroll 4
        for (int i = 0; i < kKvRowBytes / 16; ++i)
        {
            z[i] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    if (valid && !fresh)
    {
        fenceProxyAsyncSmem();
        bulkCopyHint(smemU32(sKv + (warp * 8 + lane) * kKvSmemRowBytes), kvBase + row * kKvRowBytes, kKvRowBytes,
            barGather, policyEvictFirst());
    }
    issueOproj();

    {
        constexpr int kQVecsPerHead = kQRowBytes / 16;
        for (int i = kHeads * kQVecsPerHead + tid; i < 8 * kQVecsPerHead; i += kThreads)
        {
            *reinterpret_cast<uint4*>(sQ + (i / kQVecsPerHead) * kQSmemRowBytes + 16 * (i % kQVecsPerHead))
                = make_uint4(0u, 0u, 0u, 0u);
        }
        uint8_t const* qArena = arena<uint8_t const>(ctx, kArFusedQ) + t * kHeads * kQRowBytes;
        for (int i = tid; i < kHeads * kQVecsPerHead; i += kThreads)
        {
            uint4 const qv = poll4(ctx, qArena + 16 * i, 109);
            int const h = i / kQVecsPerHead;
            int const v = i % kQVecsPerHead;
            if (v < 64)
            {
                int const grp = v >> 2;
                int const sub = v & 3;
                uint32_t const p0 = bf16x2ToHalf2(qv.x);
                uint32_t const p1 = bf16x2ToHalf2(qv.y);
                uint32_t const p2 = bf16x2ToHalf2(qv.z);
                uint32_t const p3 = bf16x2ToHalf2(qv.w);
                uint8_t* qDst = sQ + h * kQSmemRowBytes + grp * 64 + sub * 8;
                *reinterpret_cast<uint2*>(qDst) = make_uint2(p0, p2);
                *reinterpret_cast<uint2*>(qDst + 32) = make_uint2(p1, p3);
            }
            else
            {
                *reinterpret_cast<uint4*>(sQ + h * kQSmemRowBytes + 16 * v) = qv;
            }
        }
        int* sFresh = sFreshAll + warp * 16;
        if (fresh)
        {
            int const r = __popc(freshMask & ((1u << lane) - 1u));
            sFresh[2 * r] = lane;
            sFresh[2 * r + 1] = freshIdx;
        }
        __syncwarp();
        int const nFresh = __popc(freshMask);
        constexpr int kKvVecs = kKvRowBytes / 16;
        uint8_t const* kvArena = arena<uint8_t const>(ctx, kArKv);
        for (int r = 0; r < nFresh; ++r)
        {
            int const dstKey = warp * 8 + sFresh[2 * r];
            uint8_t const* srcRow = kvArena + sFresh[2 * r + 1] * kKvRowBytes;
            for (int v = lane; v < kKvVecs; v += 32)
            {
                *reinterpret_cast<uint4*>(sKv + dstKey * kKvSmemRowBytes + 16 * v) = poll4(ctx, srcRow + 16 * v, 110);
            }
        }
    }
    waitBar(ctx, barGather, 0, 111);
    __syncthreads();

    uint32_t* sValidBits = reinterpret_cast<uint32_t*>(sFreshAll);
    if (lane == 0)
    {
        sValidBits[warp] = validMask;
    }
    __syncthreads();

    int const wg = warp >> 2;
    int const wLocal = warp & 3;
    int const kBase = 16 * wLocal;
    int const g = lane >> 2;
    int const c = lane & 3;
    bool const isNope = (ctx.flags & kFlagNope) != 0;

    uint32_t const kAddrBase
        = smemU32(sKv + (kBase + (lane & 7) + 8 * ((lane >> 3) & 1)) * kKvSmemRowBytes + 16 * (lane >> 4));
    uint32_t const qAddrBase = smemU32(sQ + (lane & 7) * kQSmemRowBytes + 16 * ((lane >> 3) & 1));

    float qkAcc[4] = {0.f, 0.f, 0.f, 0.f};
    int const st0 = wg * 2;
    for (int st = st0; st < st0 + 2; ++st)
    {
        float stAcc0[4] = {0.f, 0.f, 0.f, 0.f};
        float stAcc1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kd = 0; kd < 4; ++kd)
        {
            uint32_t a0, a1, a2, a3;
            ldmatrixX4(kAddrBase + 128 * st + 32 * kd, a0, a1, a2, a3);
            uint32_t b0, b1, b2, b3;
            int const qOff = (4 * st + kd) * 64;
            ldmatrixX2(qAddrBase + qOff, b0, b1);
            ldmatrixX2(qAddrBase + qOff + 32, b2, b3);
            mmaF16(stAcc0, e4m3x2ToHalf2(a0), e4m3x2ToHalf2(a1), e4m3x2ToHalf2(a2), e4m3x2ToHalf2(a3), b0, b1);
            mmaF16(stAcc1, e4m3x2ToHalf2(a0 >> 16), e4m3x2ToHalf2(a1 >> 16), e4m3x2ToHalf2(a2 >> 16),
                e4m3x2ToHalf2(a3 >> 16), b2, b3);
        }
        float const s0 = *reinterpret_cast<float const*>(sKv + (kBase + g) * kKvSmemRowBytes + kKvScaleOff + 4 * st);
        float const s1
            = *reinterpret_cast<float const*>(sKv + (kBase + g + 8) * kKvSmemRowBytes + kKvScaleOff + 4 * st);
        qkAcc[0] = fmaf(stAcc0[0] + stAcc1[0], s0, qkAcc[0]);
        qkAcc[1] = fmaf(stAcc0[1] + stAcc1[1], s0, qkAcc[1]);
        qkAcc[2] = fmaf(stAcc0[2] + stAcc1[2], s1, qkAcc[2]);
        qkAcc[3] = fmaf(stAcc0[3] + stAcc1[3], s1, qkAcc[3]);
    }
    if (wg == 1 && !isNope)
    {
#pragma unroll
        for (int kd = 0; kd < 4; ++kd)
        {
            uint32_t a0, a1, a2, a3, b0, b1;
            ldmatrixX4(kAddrBase + kKvRopeOff + 32 * kd, a0, a1, a2, a3);
            ldmatrixX2(qAddrBase + kKvLora * 2 + 32 * kd, b0, b1);
            mmaBf16(qkAcc, a0, a1, a2, a3, b0, b1);
        }
    }

    if (wg == 0 && c < 2)
    {
        sHalfS[(kBase + g) * kHeads + 2 * c] = qkAcc[0];
        sHalfS[(kBase + g) * kHeads + 2 * c + 1] = qkAcc[1];
        sHalfS[(kBase + g + 8) * kHeads + 2 * c] = qkAcc[2];
        sHalfS[(kBase + g + 8) * kHeads + 2 * c + 1] = qkAcc[3];
    }
    __syncthreads();
    if (wg == 1 && c < 2)
    {
        auto isKeyValid = [&](int k) -> bool { return ((sValidBits[k >> 3] >> (k & 7)) & 1u) != 0u; };
        bool const v0 = isKeyValid(kBase + g);
        bool const v1 = isKeyValid(kBase + g + 8);
        sHalfS[(kBase + g) * kHeads + 2 * c]
            = v0 ? (sHalfS[(kBase + g) * kHeads + 2 * c] + qkAcc[0]) * kSoftmaxScaleLog2 : -INFINITY;
        sHalfS[(kBase + g) * kHeads + 2 * c + 1]
            = v0 ? (sHalfS[(kBase + g) * kHeads + 2 * c + 1] + qkAcc[1]) * kSoftmaxScaleLog2 : -INFINITY;
        sHalfS[(kBase + g + 8) * kHeads + 2 * c]
            = v1 ? (sHalfS[(kBase + g + 8) * kHeads + 2 * c] + qkAcc[2]) * kSoftmaxScaleLog2 : -INFINITY;
        sHalfS[(kBase + g + 8) * kHeads + 2 * c + 1]
            = v1 ? (sHalfS[(kBase + g + 8) * kHeads + 2 * c + 1] + qkAcc[3]) * kSoftmaxScaleLog2 : -INFINITY;
    }
    __syncthreads();

    {
        int const k = tid >> 2;
        int const h = tid & 3;
        int const pSlot = (k & ~31) | ((k & 8) << 1) | ((k & 16) >> 1) | (k & 7);
        float const score = sHalfS[k * kHeads + h];
        float m = score;
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
        if ((lane >> 2) == 0)
        {
            sRedMax[warp * kHeads + h] = m;
        }
        bool const kv = ((sValidBits[k >> 3] >> (k & 7)) & 1u) != 0u;
        float const ks = kv ? *reinterpret_cast<float const*>(sKv + k * kKvSmemRowBytes + kKvScaleOff + 4 * h) : 0.f;
        float ms = ks;
        ms = fmaxf(ms, __shfl_xor_sync(0xffffffffu, ms, 4));
        ms = fmaxf(ms, __shfl_xor_sync(0xffffffffu, ms, 8));
        ms = fmaxf(ms, __shfl_xor_sync(0xffffffffu, ms, 16));
        if ((lane >> 2) == 0)
        {
            sRedSum[warp * kHeads + h] = ms;
        }
        __syncthreads();

        float ctaMax = -INFINITY;
        float ctaMaxS = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            ctaMax = fmaxf(ctaMax, sRedMax[w * kHeads + h]);
            ctaMaxS = fmaxf(ctaMaxS, sRedSum[w * kHeads + h]);
        }
        if (k == 0)
        {
            sMaxScale[h] = ctaMaxS;
        }
        __syncthreads();

        float const offMax = (ctaMax == -INFINITY) ? 0.f : ctaMax;
        float const p = (ctaMax == -INFINITY) ? 0.f : exp2f(score - offMax);
        float sm = p;
        sm += __shfl_xor_sync(0xffffffffu, sm, 4);
        sm += __shfl_xor_sync(0xffffffffu, sm, 8);
        sm += __shfl_xor_sync(0xffffffffu, sm, 16);
        if ((lane >> 2) == 0)
        {
            sRedSum[warp * kHeads + h] = sm;
        }
#pragma unroll
        for (int st = 0; st < 4; ++st)
        {
            float const msSt = sMaxScale[st];
            float const ksSt
                = kv ? *reinterpret_cast<float const*>(sKv + k * kKvSmemRowBytes + kKvScaleOff + 4 * st) : 0.f;
            float const normScale = msSt > 0.f ? (ksSt / msSt) : 0.f;
            __half* pDst = reinterpret_cast<__half*>(sP + (st * 8 + h) * kPRowBytes);
            pDst[pSlot] = __float2half_rn(p * normScale);
        }
        if (h == 0)
        {
            __half const hz = __float2half_rn(0.f);
#pragma unroll
            for (int st = 0; st < 4; ++st)
            {
#pragma unroll
                for (int dh = kHeads; dh < 8; ++dh)
                {
                    reinterpret_cast<__half*>(sP + (st * 8 + dh) * kPRowBytes)[pSlot] = hz;
                }
            }
        }
        __syncthreads();

        if (tid < kHeads)
        {
            float totalSum = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                totalSum += sRedSum[w * kHeads + tid];
            }
            st64(ptr<float>(ctx, kPPartStats) + ((t * kAttnBlocks + blk) * kHeads + tid) * 2,
                make_uint2(lamport(__float_as_uint(ctaMax)), lamport(__float_as_uint(totalSum))));
        }
    }

    float* partO = ptr<float>(ctx, kPPartO) + static_cast<size_t>(t * kAttnBlocks + blk) * kHeads * kKvLora;
    int const st = warp >> 1;
    int const half = warp & 1;
    float const stripeMaxS = sMaxScale[st];
    uint32_t const pAddr = smemU32(sP + (st * 8 + (lane & 7)) * kPRowBytes + 16 * ((lane >> 3) & 1));
    uint32_t const vAddr = smemU32(sKv + lane * kKvSmemRowBytes + 128 * st + 64 * half);
    uint32_t pb[2][4];
#pragma unroll
    for (int kStep = 0; kStep < 2; ++kStep)
    {
        ldmatrixX2(pAddr + 64 * kStep, pb[kStep][0], pb[kStep][1]);
        ldmatrixX2(pAddr + 64 * kStep + 32, pb[kStep][2], pb[kStep][3]);
    }
#pragma unroll 1
    for (int bi0 = 0; bi0 < 4; bi0 += 2)
    {
        float ob2[2][4] = {};
#pragma unroll
        for (int kStep = 0; kStep < 2; ++kStep)
        {
#pragma unroll
            for (int u = 0; u < 2; ++u)
            {
                uint32_t r0, r1, r2, r3;
                ldmatrixX4Trans(vAddr + 32 * kStep * kKvSmemRowBytes + 16 * (bi0 + u), r0, r1, r2, r3);
                uint32_t const a0 = __byte_perm(r0, r1, 0x6420);
                uint32_t const a1 = __byte_perm(r0, r1, 0x7531);
                uint32_t const a2 = __byte_perm(r2, r3, 0x6420);
                uint32_t const a3 = __byte_perm(r2, r3, 0x7531);
                mmaF16(ob2[u], e4m3x2ToHalf2(a0), e4m3x2ToHalf2(a1), e4m3x2ToHalf2(a2), e4m3x2ToHalf2(a3), pb[kStep][0],
                    pb[kStep][1]);
                mmaF16(ob2[u], e4m3x2ToHalf2(a0 >> 16), e4m3x2ToHalf2(a1 >> 16), e4m3x2ToHalf2(a2 >> 16),
                    e4m3x2ToHalf2(a3 >> 16), pb[kStep][2], pb[kStep][3]);
            }
        }
        if (c < 2)
        {
#pragma unroll
            for (int u = 0; u < 2; ++u)
            {
                int const col = 128 * st + 64 * half + 16 * (bi0 + u) + 2 * g;
                st64(partO + (2 * c) * kKvLora + col,
                    make_uint2(lamport(__float_as_uint(ob2[u][0] * stripeMaxS)),
                        lamport(__float_as_uint(ob2[u][2] * stripeMaxS))));
                st64(partO + (2 * c + 1) * kKvLora + col,
                    make_uint2(lamport(__float_as_uint(ob2[u][1] * stripeMaxS)),
                        lamport(__float_as_uint(ob2[u][3] * stripeMaxS))));
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Stage 3B: Merge 32 Sparse MLA partial blocks per (tok, head) -> kArO
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageMerge(Ctx const& ctx, int unit)
{
    int const t = unit / kHeads;
    int const h = unit % kHeads;
    int const tid = static_cast<int>(threadIdx.x);
    float* sMax = reinterpret_cast<float*>(ctx.smem + kSmemRed);
    float* sSum = sMax + kAttnBlocks;

    float* stPtr = ptr<float>(ctx, kPPartStats) + ((t * kAttnBlocks + tid) * kHeads + h) * 2;
    if (tid < kAttnBlocks)
    {
        uint2 st = ldPoll64(stPtr);
        uint64_t t0 = 0;
        while (!full2(st))
        {
            spinCheck(ctx, t0, 112, unit, tid);
            st = ldPoll64(stPtr);
        }
        sMax[tid] = __uint_as_float(st.x);
        sSum[tid] = __uint_as_float(st.y);
    }

    int const d = 2 * tid;
    uint8_t* oBase = reinterpret_cast<uint8_t*>(
        ptr<float>(ctx, kPPartO) + (static_cast<size_t>(t * kAttnBlocks) * kHeads + h) * kKvLora + d);
    constexpr int kStride = kHeads * kKvLora * 4;
    uint2 o[kAttnBlocks];
#pragma unroll
    for (int b = 0; b < kAttnBlocks; ++b)
    {
        o[b] = ldPoll64(oBase + b * kStride);
    }
    {
        uint64_t t0 = 0;
        while (true)
        {
            bool ok = true;
#pragma unroll
            for (int b = 0; b < kAttnBlocks; ++b)
            {
                ok &= full2(o[b]);
            }
            if (warpAll(ok))
            {
                break;
            }
            spinCheck(ctx, t0, 113, unit, tid);
#pragma unroll
            for (int b = 0; b < kAttnBlocks; ++b)
            {
                if (!full2(o[b]))
                {
                    o[b] = ldPoll64(oBase + b * kStride);
                }
            }
        }
    }
    __syncthreads();

    float gMax = -INFINITY;
#pragma unroll
    for (int b = 0; b < kAttnBlocks; ++b)
    {
        gMax = fmaxf(gMax, sMax[b]);
    }
    float gSum = 0.f;
    float acc0 = 0.f;
    float acc1 = 0.f;
    if (gMax != -INFINITY)
    {
#pragma unroll
        for (int b = 0; b < kAttnBlocks; ++b)
        {
            float const mb = sMax[b];
            if (mb != -INFINITY)
            {
                float const w = exp2f(mb - gMax);
                gSum += sSum[b] * w;
                acc0 = fmaf(__uint_as_float(o[b].x), w, acc0);
                acc1 = fmaf(__uint_as_float(o[b].y), w, acc1);
            }
        }
    }
    float const inv = gSum > 0.f ? (1.0f / gSum) : 0.f;
    st32(arena<uint8_t>(ctx, kArO) + ((t * kHeads + h) * kKvLora + d) * 2, lamport(packBf16(acc0 * inv, acc1 * inv)));

    if (tid < kAttnBlocks)
    {
        st64(stPtr, make_uint2(kEmpty, kEmpty));
    }
#pragma unroll
    for (int b = 0; b < kAttnBlocks; ++b)
    {
        st64(oBase + b * kStride, make_uint2(kEmpty, kEmpty));
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Stage 3C & 3D: Vup & Oproj -> MNNVL Multicast H1
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageVup(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    int const head = unit / kVupTilesPerHead;
    int const sub = unit % kVupTilesPerHead;
    uint4 b[1][2];
    pollB<1>(ctx, b, arena<uint8_t const>(ctx, kArO) + head * kKvLora * 2, kHeads * kKvLora, warp, 114);
    waitBar(ctx, bar, 0, 115);

    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
    mmaChunkBf16(acc[0], ctx.smemBase + wOff + warp * kBf16Block, b[0]);
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 4096);
    reduceWarps<1>(ctx, acc, out);
    storeTileBf16(out, arena<uint8_t>(ctx, kArAttnOut), kHeads * kVDim, head * kVDim + 16 * sub);
}

__device__ __forceinline__ size_t hop1Off(int t, int r, int c)
{
    return ((static_cast<size_t>(r) * (kHidden / 32) + c / 32) * kTok + t) * 64 + (c % 32) * 2;
}

__device__ __noinline__ void stageOproj(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    uint4 b[2][2];
    if (warp < kOprojBlocksFp8)
    {
        pollB<2>(ctx, b, arena<uint8_t const>(ctx, kArAttnOut), kHeads * kVDim, 2 * warp, 116);
    }
    waitBar(ctx, bar, 0, 117);

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    if (warp < kOprojBlocksFp8)
    {
        uint4 q8[2];
        float xs0, xs1;
        quant128B(b[0], b[1], 1.f, q8, xs0, xs1);
        float const* woScale = ptr<float const>(ctx, kPWOScale);
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
        {
            int const scaleRow = (2 * unit + ti) / 8;
            float const ws = __ldg(woScale + scaleRow * kOprojBlocksFp8 + warp);
            mmaChunkFp8PbWo(acc[ti], ctx.smemBase + wOff + (ti * kOprojBlocksFp8 + warp) * kFp8Block, q8, ws, xs0, xs1);
        }
    }
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 4096);
    reduceWarps<2, kOprojBlocksFp8>(ctx, acc, out);

    if (threadIdx.x < kTok * 4)
    {
        int const tok = static_cast<int>(threadIdx.x) / 4;
        int const v = static_cast<int>(threadIdx.x) % 4;
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            int const r = 8 * v + 2 * j;
            w[j] = lamport(packBf16(out[r * kTok + tok], out[(r + 1) * kTok + tok]));
        }
        uint8_t* mc = ptr<uint8_t>(ctx, kPH1Mc) + static_cast<size_t>(ctx.parity) * kHopBytes;
        st128(mc + hop1Off(tok, ctx.rank, 32 * unit + 8 * v), make_uint4(w[0], w[1], w[2], w[3]));
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Stage Clear: empties ONLY next copy's lean main Arena [0..kArenaClearBytes) (37 x 16 KiB units = 592 KiB!)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageClear(Ctx const& ctx, int unit)
{
    int const other = ctx.nextCopy;
    int const begin = unit * kClearUnitBytes;
    int const end = min(begin + kClearUnitBytes, kArenaClearBytes);
    uint8_t* base = ptr<uint8_t>(ctx, kPArena) + static_cast<size_t>(other) * kArenaBytes;
    uint4 const e = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
    uint64_t const policy = policyEvictFirst();
    for (int off = begin + 16 * static_cast<int>(threadIdx.x); off < end; off += 16 * kThreads)
    {
        st128Hint(base + off, e, policy);
    }
    signalCounter(ctx.counters + kCClear);
}

// ----------------------------------------------------------------------------------------------------------------
// Stage H1 & H2 (0-spill!): 16-rank Weak-MC AllReduce + Residual + RMSNorm + NVFP4 Quant / Contiguous Early x'
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void pollReduce4Wave(
    Ctx const& ctx, uint8_t const* base, int stride, int rBase, float (&halfSum)[8], int code)
{
    uint4 v[4];
    pollStrided<4>(ctx, v, base + static_cast<size_t>(stride) * rBase, stride, code);
#pragma unroll
    for (int r = 0; r < 4; ++r)
    {
        halfSum[0] += bf16Lo(v[r].x);
        halfSum[1] += bf16Hi(v[r].x);
        halfSum[2] += bf16Lo(v[r].y);
        halfSum[3] += bf16Hi(v[r].y);
        halfSum[4] += bf16Lo(v[r].z);
        halfSum[5] += bf16Hi(v[r].z);
        halfSum[6] += bf16Lo(v[r].w);
        halfSum[7] += bf16Hi(v[r].w);
    }
}

__device__ __forceinline__ void pollReduce8(
    Ctx const& ctx, uint8_t const* buf, int tok, int col, float (&sum)[8], int code, bool first)
{
    // Four 4-rank waves keep peak vector registers at 16 (vs 32) for 0B spill while preserving exact 8+8 order.
    uint8_t const* base
        = first ? (buf + hop1Off(tok, 0, col)) : (buf + (static_cast<size_t>(tok * kRanks) * kHidden + col) * 2);
    int const stride = first ? static_cast<int>(hop1Off(tok, 1, col) - hop1Off(tok, 0, col)) : (kHidden * 2);
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        sum[j] = 0.f;
    }
    pollReduce4Wave(ctx, base, stride, 0, sum, code);
    pollReduce4Wave(ctx, base, stride, 4, sum, code);
    float hi[8] = {};
    pollReduce4Wave(ctx, base, stride, 8, hi, code);
    pollReduce4Wave(ctx, base, stride, 12, hi, code);
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        sum[j] = (0.f + sum[j]) + hi[j];
    }
}

__device__ __forceinline__ void hopSelfClear(Ctx const& ctx, int t, int col)
{
    uint4 const e = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
    uint64_t const policy = policyEvictFirst();
    uint8_t* const h1 = ptr<uint8_t>(ctx, kPH1Uc) + static_cast<size_t>(ctx.parity) * kHopBytes;
    uint8_t* const h2 = ptr<uint8_t>(ctx, kPH2Uc) + static_cast<size_t>(ctx.prevCopy) * kHopBytes;
    int const nR = (ctx.flags & kFlagTestRank0Clear) != 0 ? 1 : kRanks;
    for (int r = 0; r < nR; ++r)
    {
        st128Hint(h1 + hop1Off(t, r, col), e, policy);
        st128Hint(h2 + (static_cast<size_t>(t * kRanks + r) * kHidden + col) * 2, e, policy);
    }
}

__device__ __noinline__ void stageHop(
    Ctx const& ctx, int unit, bool second, __nv_bfloat16 const* residualIn, __nv_bfloat16* xOut, __nv_bfloat16* resOut)
{
    int const t = unit / 4;
    int const s4 = unit % 4;
    int const tid = static_cast<int>(threadIdx.x);
    float* sc = reinterpret_cast<float*>(ctx.smem + kSmemSmall + kSmallScalars);
    uint4* sRes = reinterpret_cast<uint4*>(ctx.smem + kSmemRes);
    bool const active = tid < kColsPerHopUnit / 8;
    int const col = kColsPerHopUnit * s4 + 8 * tid;
    uint8_t const* buf = (second ? ptr<uint8_t const>(ctx, kPH2Uc) : ptr<uint8_t const>(ctx, kPH1Uc))
        + static_cast<size_t>(ctx.parity) * kHopBytes;
    bool const xOnly = second && (ctx.flags & kFlagNextChained) != 0;
    uint8_t* const nextArena = ptr<uint8_t>(ctx, kPArena) + static_cast<size_t>(ctx.nextCopy) * kArenaBytes;

    if (active && second && (ctx.flags & kFlagHopSelfClear) != 0)
    {
        hopSelfClear(ctx, t, col);
    }

    uint4 rPacked = make_uint4(0u, 0u, 0u, 0u);
    float ssq = 0.f;
    if (active)
    {
        float sum[8];
        pollReduce8(ctx, buf, t, col, sum, second ? 120 : 118, !second);
        uint4 const resPrev
            = second ? sRes[tid] : *reinterpret_cast<uint4 const*>(residualIn + static_cast<size_t>(t) * kHidden + col);
        uint4 const gam = __ldg(reinterpret_cast<uint4 const*>(
            (second ? ptr<__nv_bfloat16 const>(ctx, kPGammaNext) : ptr<__nv_bfloat16 const>(ctx, kPGammaPost)) + col));
        uint32_t const rp[4] = {resPrev.x, resPrev.y, resPrev.z, resPrev.w};
        uint32_t const gw[4] = {gam.x, gam.y, gam.z, gam.w};
        uint32_t rw[4];
        uint32_t pr[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float const r0 = bf16Round(bf16Round(sum[2 * j]) + bf16Lo(rp[j]));
            float const r1 = bf16Round(bf16Round(sum[2 * j + 1]) + bf16Hi(rp[j]));
            ssq += bf16Round(r0 * r0) + bf16Round(r1 * r1);
            rw[j] = packBf16(r0, r1);
            pr[j] = lamport(packBf16(bf16Round(r0 * bf16Lo(gw[j])), bf16Round(r1 * bf16Hi(gw[j]))));
        }
        rPacked = make_uint4(rw[0], rw[1], rw[2], rw[3]);
        if (!second)
        {
            st128(arena<uint8_t>(ctx, kArHp) + (static_cast<size_t>(t) * kHidden + col) * 2,
                make_uint4(pr[0], pr[1], pr[2], pr[3]));
        }
        else
        {
            // Contiguous row-major early x' = res' * gamma_next for next layer's TCGen05 MXFP8 Wq/IdxQ & BF16 WkvA!
            st128(nextArena + kArX + (static_cast<size_t>(t) * kHidden + col) * 2,
                make_uint4(pr[0], pr[1], pr[2], pr[3]));
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
    }
    int const warp = tid >> 5;
    if ((tid & 31) == 0)
    {
        sc[warp] = ssq;
    }
    __syncthreads();

    if (tid == 0)
    {
        float part = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            part += sc[w];
        }
        if (second)
        {
            st32(nextArena + kArXSsq + (t * 4 + s4) * 4, lamport(__float_as_uint(part)));
        }
        if (!xOnly)
        {
            float* ssqBuf = arena<float>(ctx, second ? kArSsq2 : kArSsq1) + t * 4;
            st32(ssqBuf + s4, lamport(__float_as_uint(part)));
            uint4 const all = poll4(ctx, ssqBuf, second ? 121 : 119);
            sc[8] = __uint_as_float(all.x);
            sc[9] = __uint_as_float(all.y);
            sc[10] = __uint_as_float(all.z);
            sc[11] = __uint_as_float(all.w);
        }
    }
    __syncthreads();

    if (xOnly)
    {
        if (active)
        {
            *reinterpret_cast<uint4*>(resOut + static_cast<size_t>(t) * kHidden + col) = rPacked;
            if (ctx.p[kPCaptureOut] != 0)
            {
                __nv_bfloat16* cap
                    = ptr<__nv_bfloat16>(ctx, kPCaptureOut) + static_cast<size_t>(t) * ctx.p[kPCaptureStride] + col;
                *reinterpret_cast<uint4*>(cap) = rPacked;
            }
        }
        return;
    }

    float const S = (((0.f + sc[8]) + sc[9]) + sc[10]) + sc[11];
    float const rcp = rsqrtf(S / static_cast<float>(kHidden) + ctx.rmsEps);
    if (active)
    {
        uint4 const gam = __ldg(reinterpret_cast<uint4 const*>(
            (second ? ptr<__nv_bfloat16 const>(ctx, kPGammaNext) : ptr<__nv_bfloat16 const>(ctx, kPGammaPost)) + col));
        uint32_t const rw[4] = {rPacked.x, rPacked.y, rPacked.z, rPacked.w};
        uint32_t const gw[4] = {gam.x, gam.y, gam.z, gam.w};
        float hv[8];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            hv[2 * j] = bf16Round((bf16Lo(rw[j]) * rcp) * bf16Lo(gw[j]));
            hv[2 * j + 1] = bf16Round((bf16Hi(rw[j]) * rcp) * bf16Hi(gw[j]));
        }
        uint4 const hPacked = make_uint4(
            packBf16(hv[0], hv[1]), packBf16(hv[2], hv[3]), packBf16(hv[4], hv[5]), packBf16(hv[6], hv[7]));
        if (second)
        {
            *reinterpret_cast<uint4*>(xOut + static_cast<size_t>(t) * kHidden + col) = hPacked;
            *reinterpret_cast<uint4*>(resOut + static_cast<size_t>(t) * kHidden + col) = rPacked;
            if (ctx.p[kPCaptureOut] != 0)
            {
                __nv_bfloat16* cap
                    = ptr<__nv_bfloat16>(ctx, kPCaptureOut) + static_cast<size_t>(t) * ctx.p[kPCaptureStride] + col;
                *reinterpret_cast<uint4*>(cap) = rPacked;
            }
        }
        else
        {
            sRes[tid] = rPacked;
            st128(arena<uint8_t>(ctx, kArH) + (static_cast<size_t>(t) * kHidden + col) * 2,
                make_uint4(lamport(hPacked.x), lamport(hPacked.y), lamport(hPacked.z), lamport(hPacked.w)));
            float const inScale = __ldg(ptr<float const>(ctx, kPAlpha1) + kExperts);
            float vmax = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                vmax = fmaxf(vmax, fabsf(hv[j]));
            }
            vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, 1));
            float const sfValue = inScale * (vmax * rcpApproxFtz(6.f));
            uint32_t const sf = toE4m3(sfValue);
            float const sff = static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3 const*>(&sf));
            float const outScale = vmax != 0.f ? rcpApproxFtz(sff * rcpApproxFtz(inScale)) : 0.f;
            uint32_t packed = 0u;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                packed |= toE2m1x2(hv[2 * j] * outScale, hv[2 * j + 1] * outScale) << (8 * j);
            }
            st32(arena<uint8_t>(ctx, kArHfp4) + t * (kHidden / 2) + col / 2, lamport(packed));
            uint32_t const s1 = __shfl_down_sync(0xffffffffu, sf, 2);
            uint32_t const s2 = __shfl_down_sync(0xffffffffu, sf, 4);
            uint32_t const s3 = __shfl_down_sync(0xffffffffu, sf, 6);
            if ((tid & 7) == 0)
            {
                st32(arena<uint8_t>(ctx, kArHsf) + t * (kHidden / 16) + col / 16,
                    lamport(sf | (s1 << 8) | (s2 << 16) | (s3 << 24)));
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Stage 4A: Router GEMV (BF16 [128, 6144], 32 units) & Exact ML4 Softmax Top-4 Renorm Routing
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageRouterGemv(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar)
{
    int const et = unit / 4;
    int const kq = unit % 4;
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    constexpr int kPer = 24 / kWarps;
    uint4 b[kPer][2];
    pollB<kPer>(ctx, b, arena<uint8_t const>(ctx, kArHp) + kColsPerHopUnit * kq * 2, kHidden, warp * kPer, 122);
    waitBar(ctx, bar, 0, 123);

    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int i = 0; i < kPer; ++i)
    {
        mmaChunkBf16(acc[0], ctx.smemBase + wOff + (warp * kPer + i) * kBf16Block, b[i]);
    }
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 4096);
    reduceWarps<1>(ctx, acc, out);
    if (threadIdx.x < 16 * kTok)
    {
        int const row = static_cast<int>(threadIdx.x) / kTok;
        int const tok = static_cast<int>(threadIdx.x) % kTok;
        st32(arena<float>(ctx, kArLogit) + (kq * kTok + tok) * kExperts + 16 * et + row,
            lamport(__float_as_uint(out[row * kTok + tok])));
    }
}

__device__ __forceinline__ void routeTokens(Ctx const& ctx, int* smemMoe, bool publish)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int* sTopExpert = reinterpret_cast<int*>(ctx.smem + kSmemRed);
    float* sTopWeight = reinterpret_cast<float*>(ctx.smem + kSmemRed + 64);
    uint32_t* sMask = reinterpret_cast<uint32_t*>(ctx.smem + kSmemRed + 128);
    uint32_t* sTokMask = reinterpret_cast<uint32_t*>(ctx.smem + kSmemRed + 256);
    if (tid < kExperts)
    {
        sTokMask[tid] = 0u;
    }

    if (warp < kTok)
    {
        int const t = warp;
        float const* lp = arena<float const>(ctx, kArLogit);
        uint4 ssq = (lane == 0) ? ldPoll128(arena<float const>(ctx, kArSsq1) + 4 * t) : make_uint4(0u, 0u, 0u, 0u);
        uint32_t lw[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
#pragma unroll
            for (int kq = 0; kq < 4; ++kq)
            {
                lw[i][kq] = ldPoll32(lp + (kq * kTok + t) * kExperts + 32 * i + lane);
            }
        }
        uint64_t t0 = 0;
        while (true)
        {
            bool ok = (lane != 0) || full4(ssq);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int kq = 0; kq < 4; ++kq)
                {
                    ok &= (lw[i][kq] != kEmpty);
                }
            }
            if (warpAll(ok))
            {
                break;
            }
            spinCheck(ctx, t0, 124, t, lane);
            if (lane == 0 && !full4(ssq))
            {
                ssq = ldPoll128(arena<float const>(ctx, kArSsq1) + 4 * t);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int kq = 0; kq < 4; ++kq)
                {
                    if (lw[i][kq] == kEmpty)
                    {
                        lw[i][kq] = ldPoll32(lp + (kq * kTok + t) * kExperts + 32 * i + lane);
                    }
                }
            }
        }

        float rcpT = 1.f;
        if (lane == 0)
        {
            float const s = (((0.f + __uint_as_float(ssq.x)) + __uint_as_float(ssq.y)) + __uint_as_float(ssq.z))
                + __uint_as_float(ssq.w);
            rcpT = rsqrtf(s / static_cast<float>(kHidden) + ctx.rmsEps);
            smemMoe[kMoeRcp + t] = __float_as_int(rcpT);
        }
        rcpT = __shfl_sync(0xffffffffu, rcpT, 0);

        float logits[4];
        uint32_t keys[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float const raw = ((__uint_as_float(lw[i][0]) + __uint_as_float(lw[i][1]))
                                  + (__uint_as_float(lw[i][2]) + __uint_as_float(lw[i][3])))
                * rcpT;
            logits[i] = raw;
            keys[i] = orderKey(raw);
        }

        uint32_t taken = 0u;
        float topLogit = -INFINITY;
        int topExpert = 0;
#pragma unroll
        for (int k = 0; k < kMoeTopK; ++k)
        {
            uint32_t best = 0u;
            int bi = 0;
#pragma unroll
            for (int i = 3; i >= 0; --i)
            {
                if (!(taken & (1u << i)) && keys[i] >= best)
                {
                    best = keys[i];
                    bi = i;
                }
            }
            uint32_t const wmax = __reduce_max_sync(0xffffffffu, best);
            uint32_t const e = static_cast<uint32_t>(32 * bi + lane);
            uint32_t const win = __reduce_min_sync(0xffffffffu, best == wmax ? e : 0xffffu);
            float const l0 = (bi == 0) ? logits[0] : (bi == 1) ? logits[1] : (bi == 2) ? logits[2] : logits[3];
            float const wl = __shfl_sync(0xffffffffu, l0, static_cast<int>(win & 31u));
            if (static_cast<uint32_t>(lane) == (win & 31u))
            {
                taken |= 1u << (win >> 5);
            }
            if (lane == k)
            {
                topLogit = wl;
                topExpert = static_cast<int>(win);
            }
        }

        float const tl = lane < kMoeTopK ? topLogit : -INFINITY;
        float mmax = tl;
        mmax = fmaxf(mmax, __shfl_xor_sync(0xffffffffu, mmax, 1));
        mmax = fmaxf(mmax, __shfl_xor_sync(0xffffffffu, mmax, 2));
        float const ex = lane < kMoeTopK ? expf(tl - mmax) : 0.f;
        float esum = ex;
        esum += __shfl_xor_sync(0xffffffffu, esum, 1);
        esum += __shfl_xor_sync(0xffffffffu, esum, 2);
        if (lane < kMoeTopK)
        {
            sTopExpert[t * kMoeTopK + lane] = topExpert;
            sTopWeight[t * kMoeTopK + lane] = bf16Round(ex / esum);
            atomicOr(&sTokMask[topExpert], 1u << t);
        }
    }
    __syncthreads();

    int* sSlotOfExpert = reinterpret_cast<int*>(ctx.smem + kSmemRed + 1024);
    uint8_t* sSlotMask = reinterpret_cast<uint8_t*>(smemMoe + kMoeSlotMask);
    if (tid < kExperts)
    {
        uint32_t const tm = sTokMask[tid];
        uint32_t const sel = __ballot_sync(0xffffffffu, tm != 0u);
        if (lane == 0)
        {
            sMask[warp] = sel;
        }
    }
    __syncthreads();

    if (tid < kExperts)
    {
        uint32_t const tm = sTokMask[tid];
        uint32_t const sel = sMask[warp];
        int slot = __popc(sel & ((1u << lane) - 1u));
        int e0 = 0;
        int total = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w)
        {
            int const c = __popc(sMask[w]);
            slot += w < warp ? c : 0;
            e0 += w < 2 ? c : 0;
            total += c;
        }
        if (tm != 0u)
        {
            smemMoe[kMoeSlotExpert + slot] = tid;
            sSlotMask[slot] = static_cast<uint8_t>(tm);
            sSlotOfExpert[tid] = slot;
        }
        if (tid == 0)
        {
            smemMoe[kMoeE] = total;
            smemMoe[kMoeE0] = e0;
            smemMoe[kMoeE1] = total - e0;
        }
    }
    __syncthreads();

    if (tid < kTok * kMoeTopK)
    {
        smemMoe[kMoeSlotOfTk + tid] = sSlotOfExpert[sTopExpert[tid]];
        smemMoe[kMoeWeight + tid] = __float_as_int(sTopWeight[tid]);
    }
    __syncthreads();

    if (publish && tid < kTok * kMoeTopK)
    {
        int* route = arena<int>(ctx, kArRoute);
        int const E = smemMoe[kMoeE];
        if (tid < E)
        {
            st32(route + kRouteSlotExpert + tid, lamport(static_cast<uint32_t>(smemMoe[kMoeSlotExpert + tid])));
        }
        st32(route + kRouteSlotOfTk + tid, lamport(static_cast<uint32_t>(smemMoe[kMoeSlotOfTk + tid])));
        st32(route + kRouteWeight + tid, lamport(static_cast<uint32_t>(smemMoe[kMoeWeight + tid])));
        if (tid == 0)
        {
            st32(route + kRouteE, lamport(static_cast<uint32_t>(E)));
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Stage 4B: 12-CTA Dedicated Parallel TCGen05 MXFP8 Shared Expert (SharedGu + SwiGLU + 4-Tile SharedDn -> sOut)
// Reads exact kArH (zero HPrime ULP drift -> bit-exact Shared!), runs 100% in parallel with 200 Routed-MoE CTAs!
// Unit u in 0..11: u=0..5 on Die 0 (gate tiles 0..2 x 2 splits, dn tiles 0..23); u=6..11 on Die 1 (up tiles 3..5,
// dn 24..47).
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageShared(Ctx const& ctx, int unit, uint32_t wOff)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const guTile = unit / kSharedGuSplits;                     // 0..2 gate (Die 0), 3..5 up (Die 1)
    int const kq = unit % kSharedGuSplits;                         // 0..1 (24 K-tiles per half)
    constexpr int kGuKPerCta = kHiddenBlocksFp8 / kSharedGuSplits; // 24
    constexpr int kDnTilesPerCta = kOwnerCols / 128;               // 4
    uint32_t const tmem = *reinterpret_cast<uint32_t const*>(ctx.smem + kSmemSmall + kSmallScalars + 4 * kScalarTmem);
    uint32_t const barRing0 = ctx.smemBase + kSmemSmall + kSmallBars + kBarMoeRing0 * 8;
    uint32_t const barFree0 = ctx.smemBase + kSmemSmall + kSmallBars + kBarMoeFree0 * 8;
    uint32_t const barMma = ctx.smemBase + kSmemSmall + kSmallBars + kBarMoeMma * 8;
    uint32_t const barAct = ctx.smemBase + kSmemSmall + kSmallBars + kBarAux * 8;

    // 1. SharedGu: 128 rows x 24 K-tiles via TCGen05 MXFP8 reading exact kArH (zero rcp mul, bit-exact!).
    uint8_t const* wGu
        = ptr<uint8_t const>(ctx, kPWGu) + static_cast<size_t>(guTile) * kHiddenBlocksFp8 * kMxf8PieceBytes;
    runMxf8Tile<3, true, false>(ctx, wGu, arena<uint8_t const>(ctx, kArH), kHidden, nullptr, nullptr, nullptr, nullptr,
        nullptr, kGuKPerCta * kq, kGuKPerCta, wOff, tmem, barRing0, barFree0, barMma, barAct, 126);

    if (warp >= 4)
    {
        int const q = warp - 4;
        int const row = 32 * q + lane;
        float v[8];
        tmemLd8(tmem + kTmemD + (static_cast<uint32_t>(32 * q) << 16), v);
        uint8_t* shDst = arena<uint8_t>(ctx, kArSh) + static_cast<size_t>(guTile * kSharedGuSplits + kq) * 128 * 16;
        st128(shDst + row * 16,
            make_uint4(lamport(__float_as_uint(v[0])), lamport(__float_as_uint(v[1])), lamport(__float_as_uint(v[2])),
                lamport(__float_as_uint(v[3]))));
    }
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();

    // 2. Vectorized 1-wave poll of all 6 x 2 kArSh partials via pollStrided<2> -> SwiGLU -> BF16 sAct[4][384]!
    __nv_bfloat16* sAct = reinterpret_cast<__nv_bfloat16*>(ctx.smem + kSmemRed);
    uint8_t const* shBase = arena<uint8_t const>(ctx, kArSh);
    for (int f = tid; f < kSharedInter; f += kThreads)
    {
        int const tileIdx = f >> 7; // 0..2
        int const row = f & 127;
        uint4 gv[kSharedGuSplits], uv[kSharedGuSplits];
        pollStrided<kSharedGuSplits>(
            ctx, gv, shBase + (static_cast<size_t>(tileIdx * kSharedGuSplits) * 128 + row) * 16, 128 * 16, 128);
        pollStrided<kSharedGuSplits>(
            ctx, uv, shBase + (static_cast<size_t>((3 + tileIdx) * kSharedGuSplits) * 128 + row) * 16, 128 * 16, 128);
        float gSum[kTok] = {}, uSum[kTok] = {};
#pragma unroll
        for (int q = 0; q < kSharedGuSplits; ++q)
        {
            gSum[0] += __uint_as_float(gv[q].x);
            gSum[1] += __uint_as_float(gv[q].y);
            gSum[2] += __uint_as_float(gv[q].z);
            gSum[3] += __uint_as_float(gv[q].w);
            uSum[0] += __uint_as_float(uv[q].x);
            uSum[1] += __uint_as_float(uv[q].y);
            uSum[2] += __uint_as_float(uv[q].z);
            uSum[3] += __uint_as_float(uv[q].w);
        }
#pragma unroll
        for (int t = 0; t < kTok; ++t)
        {
            float const gb = bf16Round(gSum[t]);
            float const ub = bf16Round(uSum[t]);
            float const sig = 1.f / (1.f + __expf(-gb));
            sAct[t * kSharedInter + f] = __float2bfloat16_rn(gb * sig * ub);
        }
    }
    __syncthreads();

    // 3. Quantize sAct[4][384] ONCE into smemB (3 KB) + smemSfb (384 B), then run ALL 4 N-tiles (512 cols) in 1 ring!
    // TMEM D[0..32) holds all 4 N-tiles simultaneously (4 x 8 = 32 cols <= 64 cols!).
    constexpr uint32_t kDnSmemBOff = 3u * kMxf8SlotStride;
    constexpr uint32_t kDnSmemSfbOff = kDnSmemBOff + kSharedDnBlocks * 1024u;
    uint8_t* dnB = ctx.smem + wOff + kDnSmemBOff;
    uint8_t* dnSfb = ctx.smem + wOff + kDnSmemSfbOff;
    uint32_t const aSmem = ctx.smemBase + wOff;
    uint32_t const bSmem = ctx.smemBase + wOff + kDnSmemBOff;
    uint32_t const sfbSmem = ctx.smemBase + wOff + kDnSmemSfbOff;

    if (tid < 32)
    {
        if (tid < 3)
        {
            mbarReinit(barRing0 + 8 * tid, 1);
            mbarReinit(barFree0 + 8 * tid, 1);
        }
        else if (tid == 3)
        {
            mbarReinit(barMma, 1);
        }
        else if (tid == 4)
        {
            mbarReinit(barAct, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    bool const evictFirst = (ctx.flags & kFlagEvictFirst) != 0;
    uint64_t const policy = evictFirst ? policyEvictFirst() : 0ull;
    // 4 N-tiles (4*unit .. 4*unit+3) x 3 K-blocks = 12 contiguous MXFP8 pieces (202 KB).
    constexpr int kDnPieces = kDnTilesPerCta * kSharedDnBlocks; // 12
    uint8_t const* wDnBase = ptr<uint8_t const>(ctx, kPWD)
        + static_cast<size_t>(kDnTilesPerCta * unit) * kSharedDnBlocks * kMxf8PieceBytes;

    if (warp == 0)
    {
        if (lane == 0)
        {
            for (int p = 0; p < kDnPieces; ++p)
            {
                int const st = p % 3;
                if (p >= 3)
                {
                    waitBar(ctx, barFree0 + 8 * st, static_cast<uint32_t>(((p / 3) - 1) & 1), 129);
                }
                uint32_t const dst = aSmem + static_cast<uint32_t>(st) * kMxf8SlotStride;
                uint8_t const* src = wDnBase + static_cast<size_t>(p) * kMxf8PieceBytes;
                uint32_t const bar = barRing0 + 8 * st;
                mbarExpectTx(bar, kMxf8PieceBytes);
                if (evictFirst)
                {
                    bulkCopyHint(dst, src, kMxf8PieceBytes, bar, policy);
                }
                else
                {
                    bulkCopy(dst, src, kMxf8PieceBytes, bar);
                }
            }
        }
    }
    else if (warp == 1)
    {
        mbarWaitSpin(barAct, 0);
        tcFenceAfter();
        if (electOne())
        {
            for (int ki = 0; ki < kSharedDnBlocks; ++ki)
            {
                utccp1Rows8(
                    tmem + kTmemSfb + 4u * static_cast<uint32_t>(ki), sfbSmem + 128u * static_cast<uint32_t>(ki));
            }
        }
        __syncwarp();
        for (int p = 0; p < kDnPieces; ++p)
        {
            int const ti = p / kSharedDnBlocks; // 0..3 -> D cols [8*ti .. 8*ti+8)
            int const ki = p % kSharedDnBlocks; // 0..2
            int const st = p % 3;
            waitBar(ctx, barRing0 + 8 * st, static_cast<uint32_t>((p / 3) & 1), 129);
            tcFenceAfter();
            if (electOne())
            {
                uint32_t const aSlot = aSmem + static_cast<uint32_t>(st) * kMxf8SlotStride;
                uint32_t const sfaTmem = tmem + kTmemSfa + 4u * static_cast<uint32_t>(st);
                uint32_t const sfbTmem = tmem + kTmemSfb + 4u * static_cast<uint32_t>(ki);
                utccp1(sfaTmem, aSlot + kMxf8WBytes);
                uint32_t const bTile = bSmem + 1024u * static_cast<uint32_t>(ki);
#pragma unroll
                for (int kb = 0; kb < 4; ++kb)
                {
                    mmaMxf8(tmem + kTmemD + 8u * static_cast<uint32_t>(ti), aSlot + 32u * static_cast<uint32_t>(kb),
                        bTile + 32u * static_cast<uint32_t>(kb), sfaTmem, sfbTmem, static_cast<uint32_t>(kb),
                        (ki > 0 || kb > 0) ? 1u : 0u);
                }
                if (p + 3 < kDnPieces)
                {
                    commit1(barFree0 + 8 * st);
                }
            }
            __syncwarp();
        }
        if (electOne())
        {
            commit1(barMma);
        }
        __syncwarp();
    }
    else if (warp >= 4)
    {
        // Quantize sAct[4][384] from SMEM (fast, 48 items across 128 threads = 1 iter!).
        int const qzTid = tid - 128;
        for (int v = qzTid; v < kSharedDnBlocks * 32; v += 128)
        {
            if (v < kSharedDnBlocks * 8)
            {
                *reinterpret_cast<uint4*>(dnSfb + 16 * v) = make_uint4(0u, 0u, 0u, 0u);
            }
            *reinterpret_cast<uint4*>(dnB + (v >> 5) * 1024 + 512 + 16 * (v & 31)) = make_uint4(0u, 0u, 0u, 0u);
        }
        asm volatile("bar.sync 2, 128;" ::: "memory");
        int const totalItems = kSharedDnBlocks * kTok * 4; // 48
        bool const active = qzTid < totalItems;
        int const ki = active ? (qzTid >> 4) : 0;
        int const tok = active ? ((qzTid >> 2) & 3) : 0;
        int const k32 = active ? (qzTid & 3) : 0;
        uint4 raw[4];
        float amax = 0.f;
        if (active)
        {
            uint8_t const* rp = reinterpret_cast<uint8_t const*>(sAct + tok * kSharedInter + ki * 128 + k32 * 32);
#pragma unroll
            for (int z = 0; z < 4; ++z)
            {
                raw[z] = *reinterpret_cast<uint4 const*>(rp + 16 * z);
                amax = fmaxf(amax, fmaxf(fabsf(bf16Lo(raw[z].x)), fabsf(bf16Hi(raw[z].x))));
                amax = fmaxf(amax, fmaxf(fabsf(bf16Lo(raw[z].y)), fabsf(bf16Hi(raw[z].y))));
                amax = fmaxf(amax, fmaxf(fabsf(bf16Lo(raw[z].z)), fabsf(bf16Hi(raw[z].z))));
                amax = fmaxf(amax, fmaxf(fabsf(bf16Lo(raw[z].w)), fabsf(bf16Hi(raw[z].w))));
            }
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        if (active)
        {
            float const scale = ue8m0Scale(amax);
            float const inv = pow2Inv(scale);
            uint32_t q[8];
#pragma unroll
            for (int z = 0; z < 4; ++z)
            {
                q[2 * z] = toE4m3x2(__fmul_rn(bf16Lo(raw[z].x), inv), __fmul_rn(bf16Hi(raw[z].x), inv))
                    | (toE4m3x2(__fmul_rn(bf16Lo(raw[z].y), inv), __fmul_rn(bf16Hi(raw[z].y), inv)) << 16);
                q[2 * z + 1] = toE4m3x2(__fmul_rn(bf16Lo(raw[z].z), inv), __fmul_rn(bf16Hi(raw[z].z), inv))
                    | (toE4m3x2(__fmul_rn(bf16Lo(raw[z].w), inv), __fmul_rn(bf16Hi(raw[z].w), inv)) << 16);
            }
            int const c0 = k32 * 2;
            uint8_t* bRow = dnB + ki * 1024 + tok * 128;
            *reinterpret_cast<uint4*>(bRow + 16 * (c0 ^ (tok & 7))) = make_uint4(q[0], q[1], q[2], q[3]);
            *reinterpret_cast<uint4*>(bRow + 16 * ((c0 + 1) ^ (tok & 7))) = make_uint4(q[4], q[5], q[6], q[7]);
            dnSfb[ki * 128 + tok * 16 + k32] = ue8m0ExpByte(scale);
        }
        fenceProxyAsyncSmem();
        asm volatile("bar.sync 2, 128;" ::: "memory");
        if (tid == 128)
        {
            mbarArrive(barAct);
        }
    }

    waitBar(ctx, barMma, 0, 129);
    __syncthreads();
    tcFenceAfter();

    // Drain all kDnTilesPerCta N-tiles (256 cols x 4 tok = 2,048 B) into sOut at kSmemRed [0..2048) for stageOwner!
    __nv_bfloat16* sOut = reinterpret_cast<__nv_bfloat16*>(ctx.smem + kSmemRed);
    if (warp >= 4)
    {
        int const q = warp - 4;
#pragma unroll
        for (int grp = 0; grp < kDnTilesPerCta / 2; ++grp)
        {
            float vt[2][8];
            tmemLd2x8(tmem + kTmemD + 16u * static_cast<uint32_t>(grp) + (static_cast<uint32_t>(32 * q) << 16), vt);
#pragma unroll
            for (int ti = 0; ti < 2; ++ti)
            {
                int const col = 128 * (2 * grp + ti) + 32 * q + lane;
#pragma unroll
                for (int t = 0; t < kTok; ++t)
                {
                    sOut[t * kOwnerCols + col] = __float2bfloat16_rn(vt[ti][t]);
                }
            }
        }
    }
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
}

// ----------------------------------------------------------------------------------------------------------------
// Stage 4C: 188-CTA Zero-Stall Overlapped TMA+TCGen05 NVFP4 Routed MoE (Inline Fused SwiGLU+NVFP4 Requant!)
// Unit rCta in [0, 188): 0..93 on Die 0, 94..187 on Die 1 (with H1 CTAs mapped to highest local idx 86..93).
// Eliminates cross-CTA kArP1 gather storm, redundant 24x SwiGLU/NVFP4, and per-piece __syncthreads()!
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ int cutStart(int localCta, int nCtas, int nPieces)
{
    return (localCta * nPieces) / nCtas;
}

__device__ __noinline__ void issueFc1Piece(Ctx const& ctx, int expert, int ti, int kp, uint32_t slotAddr, uint32_t bar)
{
    uint8_t const* src = ptr<uint8_t const>(ctx, kPW13)
        + ((static_cast<size_t>(expert) * kFc1TilesPerExpert + ti) * kFc1KPieces + kp) * kPieceBytes;
    mbarExpectTx(bar, kPieceBytes);
    bool const evictFirst = (ctx.flags & kFlagEvictFirst) != 0;
    copyPieces(slotAddr, src, kPieceBytes, bar, evictFirst);
}

__device__ __noinline__ void issueFc2Piece(Ctx const& ctx, int expert, int pIdx, uint32_t slotAddr, uint32_t bar)
{
    uint8_t const* src
        = ptr<uint8_t const>(ctx, kPW2) + (static_cast<size_t>(expert) * kFc2PiecesPerSlot + pIdx) * kPieceBytes;
    mbarExpectTx(bar, kPieceBytes);
    bool const evictFirst = (ctx.flags & kFlagEvictFirst) != 0;
    copyPieces(slotAddr, src, kPieceBytes, bar, evictFirst);
}

__device__ __noinline__ void stageMoe(Ctx const& ctx, int rCta, uint32_t wOff)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    constexpr uint32_t kRegion2 = kRingSlots * kPieceBytes;
    uint8_t* region2 = ctx.smem + wOff + kRegion2;
    uint32_t const hb = ctx.smemBase + wOff + kRegion2;
    uint8_t* hsf = region2 + kHfp4SmemBytes;
    uint8_t* fc2B = hsf + kHsfBytes;
    uint8_t* fc2Sfb = fc2B + kFc2BSmemBytes;
    uint32_t const barRing0 = ctx.smemBase + kSmemSmall + kSmallBars + kBarMoeRing0 * 8;
    uint32_t const barFree0 = ctx.smemBase + kSmemSmall + kSmallBars + kBarMoeFree0 * 8;
    uint32_t const barMma = ctx.smemBase + kSmemSmall + kSmallBars + kBarMoeMma * 8;
    uint32_t const tmem = *reinterpret_cast<uint32_t const*>(ctx.smem + kSmemSmall + kSmallScalars + 4 * kScalarTmem);
    int* smemMoe = reinterpret_cast<int*>(ctx.smem + kSmemSmall + kSmallMoe);

    fenceProxyAsyncSmem();
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();

    // 1. Exact ML4 Softmax Top-4 Renorm routing (rCta == 0 publishes kArRoute for the 24 Owner CTAs).
    routeTokens(ctx, smemMoe, rCta == 0);
    int const E = smemMoe[kMoeE];
    int const E0 = smemMoe[kMoeE0];
    int const E1 = smemMoe[kMoeE1];

    // 2. Die-Aware partitioning across 188 Routed CTAs (94 Die 0 + 94 Die 1).
    // Enable dieSplit whenever both dies have active experts and max(E0, E1) <= 8 (<= 48 FC1 tiles <= 94 CTAs/die!).
    bool const dieSplit = (ctx.flags & kFlagDieAware) != 0 && E0 > 0 && E1 > 0 && E0 <= 8 && E1 <= 8;
    int localCta, nLocalCtas, slotBase, nLocalExperts;
    if (dieSplit)
    {
        if (rCta < kRoutedDie0)
        {
            localCta = rCta;
            nLocalCtas = kRoutedDie0;
            slotBase = 0;
            nLocalExperts = E0;
        }
        else
        {
            localCta = rCta - kRoutedDie0;
            nLocalCtas = kRoutedCtas - kRoutedDie0;
            slotBase = E0;
            nLocalExperts = E1;
        }
    }
    else
    {
        localCta = rCta;
        nLocalCtas = kRoutedCtas;
        slotBase = 0;
        nLocalExperts = E;
    }

    // Tile-aligned FC1 assignment:
    //   nTiles = 6 * nLocalExperts <= nLocalCtas.
    //   nFull = max(0, 2 * nTiles - nLocalCtas) tiles run full 8 K-pieces in 1 CTA (zero kArP1 traffic!).
    //   nSplit = nTiles - nFull tiles run 2 K-half CTAs (kh=0: 4 pieces -> kArP1; kh=1: 4 pieces + fused
    //   SwiGLU+NVFP4!).
    int const nTiles = kFc1TilesPerExpert * nLocalExperts;
    int const nFull = max(0, 2 * nTiles - nLocalCtas);
    int const nSplit = nTiles - nFull;
    int fc1Slot = -1;
    int fc1Ti = 0;
    int fc1K0 = 0;
    int fc1NK = 0;
    int fc1Role = 0; // 0 = none, 1 = kh0 (write kArP1), 2 = kh1 (poll kArP1 + SwiGLU+NVFP4), 3 = full (SwiGLU+NVFP4)
    if (localCta < nFull)
    {
        fc1Slot = slotBase + localCta / kFc1TilesPerExpert;
        fc1Ti = localCta % kFc1TilesPerExpert;
        fc1K0 = 0;
        fc1NK = kFc1KPieces; // 8
        fc1Role = 3;
    }
    else if (localCta < nFull + nSplit)
    {
        int const gt = nFull + (localCta - nFull);
        fc1Slot = slotBase + gt / kFc1TilesPerExpert;
        fc1Ti = gt % kFc1TilesPerExpert;
        fc1K0 = 0;
        fc1NK = kFc1KPieces / 2; // 4
        fc1Role = 1;
    }
    else if (localCta < nFull + 2 * nSplit)
    {
        int const gt = nFull + (localCta - nFull - nSplit);
        fc1Slot = slotBase + gt / kFc1TilesPerExpert;
        fc1Ti = gt % kFc1TilesPerExpert;
        fc1K0 = kFc1KPieces / 2; // 4
        fc1NK = kFc1KPieces / 2; // 4
        fc1Role = 2;
    }

    // FC2 range: 24 * nLocalExperts pieces partitioned in REVERSE localCta order so idle FC1 CTAs take the heaviest
    // FC2 slots first while nFull (8-piece FC1) CTAs sit at the tail with minimal FC2 load!
    int const revCta = nLocalCtas - 1 - localCta;
    int const f2a = slotBase * kFc2PiecesPerSlot + cutStart(revCta, nLocalCtas, kFc2PiecesPerSlot * nLocalExperts);
    int const f2b = slotBase * kFc2PiecesPerSlot + cutStart(revCta + 1, nLocalCtas, kFc2PiecesPerSlot * nLocalExperts);

    // 3. Re-init ring barriers and immediately issue initial FC1 TMA pieces while polling Hfp4/Hsf!
    if (tid < 32)
    {
        if (tid < kRingSlots)
        {
            mbarReinit(barRing0 + 8 * tid, 1);
            mbarReinit(barFree0 + 8 * tid, 1);
        }
        else if (tid == kRingSlots)
        {
            mbarReinit(barMma, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    int const fc1Expert = (fc1Slot >= 0) ? smemMoe[kMoeSlotExpert + fc1Slot] : 0;
    if (tid == 0 && fc1NK > 0)
    {
        for (int l = 0; l < kRingSlots && l < fc1NK; ++l)
        {
            issueFc1Piece(ctx, fc1Expert, fc1Ti, fc1K0 + l, ctx.smemBase + wOff + l * kPieceBytes, barRing0 + 8 * l);
        }
        if (fc1NK > kRingSlots && (ctx.flags & kFlagMoeL2) != 0)
        {
            uint64_t const pol = policyEvictFirst();
            for (int l = kRingSlots; l < fc1NK; ++l)
            {
                uint8_t const* src = ptr<uint8_t const>(ctx, kPW13)
                    + ((static_cast<size_t>(fc1Expert) * kFc1TilesPerExpert + fc1Ti) * kFc1KPieces + fc1K0 + l)
                        * kPieceBytes;
                prefetchBulkL2Hint(src, kPieceBytes, pol);
            }
        }
    }

    // 4. Poll NVFP4(h) & scales in ONE parallel pollStrided<3> wave; zero ONLY unused rows 4..7 (disjoint, -2 syncs!).
    if (fc1NK > 0)
    {
        for (int v = tid; v < kHfp4Atoms * 16; v += kThreads)
        {
            int const atom = v >> 4;
            int const rem = v & 15; // 16 x 16B = 256B = rows 4..7 of this 512B atom
            *reinterpret_cast<uint4*>(region2 + atom * 512 + 256 + 16 * rem) = make_uint4(0u, 0u, 0u, 0u);
        }
        uint8_t const* hSrc = arena<uint8_t const>(ctx, kArHfp4);
        constexpr int kHVecE = (kTok * (kHidden / 2)) / 16; // 768 = 3 * 256
        constexpr int kHVecS = kHsfBytes / 16;              // 96
        uint4 hv[3];
        pollStrided<3>(ctx, hv, hSrc + 16 * tid, 16 * kThreads, 130);
#pragma unroll
        for (int it = 0; it < 3; ++it)
        {
            int const j = tid + it * kThreads;
            int const t = j / 192;
            int const c16 = j % 192;
            *reinterpret_cast<uint4*>(region2 + (c16 >> 2) * 512 + t * 64 + (((c16 & 3) ^ ((t >> 1) & 3)) << 4))
                = hv[it];
        }
        if (tid < kHVecS)
        {
            *reinterpret_cast<uint4*>(hsf + 16 * tid) = poll4(ctx, hSrc + kHVecE * 16 + 16 * tid, 131);
        }
        __syncthreads();

        uint8_t* stage = ctx.smem + kSmemRed + 4096;
        for (int v = tid; v < 48 * 8; v += kThreads)
        {
            int const pair = v >> 3;
            int const row = v & 7;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (row < kTok)
            {
                uint32_t const* sf = reinterpret_cast<uint32_t const*>(hsf + row * 384) + 2 * pair;
                val = make_uint4(sf[0], 0u, sf[1], 0u);
            }
            *reinterpret_cast<uint4*>(stage + 16 * v) = val;
        }
        fenceProxyAsyncSmem();
        __syncthreads();
        if (warp == 1)
        {
            tcFenceAfter();
            if (electOne())
            {
#pragma unroll
                for (int pair = 0; pair < 48; ++pair)
                {
                    utccp1Rows8(tmem + kTmemSfb + 4 * pair, smemU32(stage + 128 * pair));
                }
            }
            __syncwarp();
        }
        __syncthreads();

        // 5. Overlapped FC1 TMA + TCGen05 MMA loop (ZERO __syncthreads() across all 4 or 8 K-pieces!).
        if (warp == 0)
        {
            if (lane == 0)
            {
                for (int p = kRingSlots; p < fc1NK; ++p)
                {
                    int const rs = p % kRingSlots;
                    waitBar(ctx, barFree0 + 8 * rs, static_cast<uint32_t>(((p / kRingSlots) - 1) & 1), 132);
                    issueFc1Piece(
                        ctx, fc1Expert, fc1Ti, fc1K0 + p, ctx.smemBase + wOff + rs * kPieceBytes, barRing0 + 8 * rs);
                }
            }
        }
        else if (warp == 1)
        {
            for (int p = 0; p < fc1NK; ++p)
            {
                int const rs = p % kRingSlots;
                int const kp = fc1K0 + p;
                waitBar(ctx, barRing0 + 8 * rs, static_cast<uint32_t>((p / kRingSlots) & 1), 132);
                tcFenceAfter();
                if (electOne())
                {
                    uint32_t const base = ctx.smemBase + wOff + rs * kPieceBytes;
                    uint32_t const sfa = tmem + kTmemSfa + 48u * static_cast<uint32_t>(rs);
                    uint32_t const sfb = tmem + kTmemSfb + 24u * static_cast<uint32_t>(kp);
                    uint32_t const hbq = hb + 6u * static_cast<uint32_t>(kp) * 512u;
#pragma unroll
                    for (int st = 0; st < 12; ++st)
                    {
                        utccp1(sfa + 4 * st, base + kPieceW + 512 * st);
                    }
#pragma unroll
                    for (int st = 0; st < 12; ++st)
                    {
                        int const kb = st >> 1;
                        int const ks = st & 1;
                        mmaNvf4(tmem + kTmemD, base + kb * 8192 + 32 * ks, hbq + kb * 512 + 32 * ks, sfa + 4 * st,
                            sfb + 2 * st, (p > 0 || st > 0) ? 1u : 0u);
                    }
                    if (p + kRingSlots < fc1NK)
                    {
                        commit1(barFree0 + 8 * rs);
                    }
                }
                __syncwarp();
            }
            if (electOne())
            {
                commit1(barMma);
            }
            __syncwarp();
        }
        waitBar(ctx, barMma, 0, 133);
        __syncthreads();
        tcFenceAfter();

        // 6. FC1 TMEM drain + inline register-only SwiGLU + 16-elem NVFP4 requant directly to kArFc2B/Sfb!
        uint8_t const tokenMask = reinterpret_cast<uint8_t const*>(smemMoe + kMoeSlotMask)[fc1Slot];
        uint8_t* p1Tile
            = arena<uint8_t>(ctx, kArP1) + (static_cast<size_t>(fc1Slot) * kFc1TilesPerExpert + fc1Ti) * kTok * 128 * 4;
        if (warp < 4)
        {
            float v[8];
            tmemLd8(tmem + kTmemD + (static_cast<uint32_t>(32 * warp) << 16), v);
            int const feat = 2 * (lane & 7) + (lane >> 4); // 0..15 within warp's 16-feature block
            int const gate = (lane >> 3) & 1;              // 0 = up, 1 = gate (lane ^ 8 is partner!)
            if (fc1Role == 1)
            {
#pragma unroll
                for (int t = 0; t < kTok; ++t)
                {
                    if (((tokenMask >> t) & 1u) != 0u)
                    {
                        st32(p1Tile + 4 * (t * 128 + 32 * warp + 2 * feat + gate), lamport(__float_as_uint(v[t])));
                    }
                }
            }
            else
            {
                float const alpha1 = __ldg(ptr<float const>(ctx, kPAlpha1) + fc1Expert);
                float const fc2InScale = __ldg(ptr<float const>(ctx, kPAlpha2) + kExperts);
                uint8_t* dstB = arena<uint8_t>(ctx, kArFc2B) + static_cast<size_t>(fc1Slot) * kFc2BSmemBytes;
                uint8_t* dstS = arena<uint8_t>(ctx, kArFc2Sfb) + static_cast<size_t>(fc1Slot) * kFc2SfbArenaBytes;
                int const f0 = 64 * fc1Ti + 16 * warp; // 16-aligned feature block
                int const kb = f0 >> 7;                // 0..2
                int const bInAtom = (f0 & 127) >> 1;   // 0, 8, 16, 24, 32, 40, 48, 56
                int const blkInKb = (f0 & 127) >> 4;   // 0..7
                int const halfStep = blkInKb >> 2;     // 0 or 1
                int const bOff = blkInKb & 3;          // 0..3
#pragma unroll
                for (int t = 0; t < kTok; ++t)
                {
                    if (((tokenMask >> t) & 1u) == 0u)
                    {
                        continue;
                    }
                    float val = v[t];
                    if (fc1Role == 2)
                    {
                        val += __uint_as_float(poll1(ctx, p1Tile + 4 * (t * 128 + 32 * warp + 2 * feat + gate), 134));
                    }
                    // Partner (lane ^ 8) holds gate for up lanes (and up for gate lanes).
                    float const other = __shfl_xor_sync(0xffffffffu, val, 8);
                    float const uVal = (gate == 0 ? val : other) * alpha1;
                    float const gVal = (gate == 0 ? other : val) * alpha1;
                    float const sig = 1.f / (1.f + __expf(-gVal));
                    float const act = (gVal * sig) * uVal;

                    // All 16 features of this NVFP4 block live in lanes {0..7, 16..23} (and duplicated in
                    // {8..15, 24..31}).
                    float vmax = fabsf(act);
                    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, 1));
                    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, 2));
                    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, 4));
                    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, 16));
                    float const sfVal = fc2InScale * (vmax * rcpApproxFtz(6.f));
                    uint32_t const sf = toE4m3(sfVal);
                    float const sff = static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3 const*>(&sf));
                    float const outScale = vmax != 0.f ? rcpApproxFtz(sff * rcpApproxFtz(fc2InScale)) : 0.f;
                    float const qv = act * outScale;
                    // Even feature = lane 0..7; odd feature = lane 16..23 (lane ^ 16!).
                    float const qvOdd = __shfl_xor_sync(0xffffffffu, qv, 16);
                    uint32_t const b0 = toE2m1x2(qv, qvOdd);
                    uint32_t const b1 = __shfl_down_sync(0xffffffffu, b0, 1);
                    uint32_t const b2 = __shfl_down_sync(0xffffffffu, b0, 2);
                    uint32_t const b3 = __shfl_down_sync(0xffffffffu, b0, 3);
                    uint32_t const w4 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                    uint32_t const w4Hi = __shfl_down_sync(0xffffffffu, w4, 4);
                    if (lane == 0)
                    {
                        int const swC = (bInAtom >> 4) ^ ((t >> 1) & 3);
                        st64(dstB + kb * 512 + t * 64 + (swC << 4) + (bInAtom & 15), make_uint2(w4, w4Hi));
                        dstS[kb * 128 + t * 16 + halfStep * 8 + bOff] = static_cast<uint8_t>(sf);
                    }
                }
            }
        }
        if (fc1Role >= 2)
        {
            signalCounter(ctx.counters + kCFc2Ready0 + fc1Slot);
        }
        tcFenceBefore();
        __syncthreads();
        tcFenceAfter();
    }

    // 7. FC2 execution: re-init ring barriers, prefetch FC2 pieces while polling 1-word kCFc2Ready[slot]!
    int const nFc2 = f2b - f2a;
    if (nFc2 <= 0)
    {
        return;
    }
    if (tid < 32)
    {
        if (tid < kRingSlots)
        {
            mbarReinit(barRing0 + 8 * tid, 1);
            mbarReinit(barFree0 + 8 * tid, 1);
        }
        else if (tid == kRingSlots)
        {
            mbarReinit(barMma, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    // Issue initial FC2 TMA pieces immediately (single elected thread for uniform TMA/mbarrier issue)!
    if (tid == 0)
    {
        for (int l = 0; l < kRingSlots && l < nFc2; ++l)
        {
            int const j = f2a + l;
            int const slot = j / kFc2PiecesPerSlot;
            int const pIdx = j % kFc2PiecesPerSlot;
            issueFc2Piece(
                ctx, smemMoe[kMoeSlotExpert + slot], pIdx, ctx.smemBase + wOff + l * kPieceBytes, barRing0 + 8 * l);
        }
    }

    uint32_t mmaPhase = 0u;
    int p = 0;
    for (int j = f2a; j < f2b;)
    {
        int const slot = j / kFc2PiecesPerSlot;
        int const jEnd = min(f2b, (slot + 1) * kFc2PiecesPerSlot);
        int const expert = smemMoe[kMoeSlotExpert + slot];
        uint8_t const mask = reinterpret_cast<uint8_t const*>(smemMoe + kMoeSlotMask)[slot];
        float const alpha2 = __ldg(ptr<float const>(ctx, kPAlpha2) + expert);

        // Single-counter wait (6 tiles ready) -> coalesced L1/L2 load of 1,536 B fc2B + 384 B fc2Sfb (~100 ns!).
        waitCounterPtr(ctx, ctx.counters + kCFc2Ready0 + slot, static_cast<uint32_t>(kFc1TilesPerExpert), 134);
        uint8_t const* srcB = arena<uint8_t const>(ctx, kArFc2B) + static_cast<size_t>(slot) * kFc2BSmemBytes;
        uint8_t const* srcS = arena<uint8_t const>(ctx, kArFc2Sfb) + static_cast<size_t>(slot) * kFc2SfbArenaBytes;
        if (tid < kFc2BSmemBytes / 16)         // 96 threads copy 1,536 B (zeroing inactive token rows 4..7 too!)
        {
            int const rInKb = (tid & 31) >> 2; // 0..7
            bool const liveRow = (rInKb < kTok) && (((mask >> rInKb) & 1u) != 0u);
            *reinterpret_cast<uint4*>(fc2B + 16 * tid)
                = liveRow ? *reinterpret_cast<uint4 const*>(srcB + 16 * tid) : make_uint4(0u, 0u, 0u, 0u);
        }
        else if (tid >= 128 && tid < 128 + kFc2SfbSmemBytes / 16) // 24 threads copy 384 B SFB
        {
            int const sIdx = tid - 128;
            int const rInKb = sIdx & 7;
            bool const liveRow = (rInKb < kTok) && (((mask >> rInKb) & 1u) != 0u);
            *reinterpret_cast<uint4*>(fc2Sfb + 16 * sIdx)
                = liveRow ? *reinterpret_cast<uint4 const*>(srcS + 16 * sIdx) : make_uint4(0u, 0u, 0u, 0u);
        }
        fenceProxyAsyncSmem();
        __syncthreads();

        // Copy this slot's 3 K-block SFB atoms into TMEM ONCE per slot (reused across all FC2 pieces of this slot!).
        if (warp == 1)
        {
            tcFenceAfter();
            if (electOne())
            {
#pragma unroll
                for (int kb = 0; kb < kFc2KBlocks; ++kb)
                {
                    utccp1Rows8(tmem + kTmemSfb2 + 4 * kb, smemU32(fc2Sfb + 128 * kb));
                }
            }
            __syncwarp();
        }

        for (; j < jEnd; ++j, ++p)
        {
            int const pIdx = j % kFc2PiecesPerSlot;
            int const rs = p % kRingSlots;
            uint32_t const base = ctx.smemBase + wOff + rs * kPieceBytes;
            waitBar(ctx, barRing0 + 8 * rs, static_cast<uint32_t>((p / kRingSlots) & 1), 135);
            if (warp == 1)
            {
                tcFenceAfter();
                if (electOne())
                {
                    uint32_t const sfa = tmem + kTmemSfa + 48u * static_cast<uint32_t>(rs);
#pragma unroll
                    for (int st = 0; st < 12; ++st)
                    {
                        utccp1(sfa + 4 * st, base + kPieceW + 512 * st);
                    }
#pragma unroll
                    for (int ti = 0; ti < kFc2TilesPerPiece; ++ti)
                    {
#pragma unroll
                        for (int ks = 0; ks < 2 * kFc2KBlocks; ++ks)
                        {
                            int const kb = ks >> 1;
                            int const sub = ks & 1;
                            mmaNvf4(tmem + kTmemD + 8 * ti, base + (kFc2KBlocks * ti + kb) * 8192 + 32 * sub,
                                smemU32(fc2B) + kb * 512 + 32 * sub, sfa + 4 * (6 * ti + ks), tmem + kTmemSfb2 + 2 * ks,
                                ks > 0 ? 1u : 0u);
                        }
                    }
                    commit1(barMma);
                }
                __syncwarp();
            }
            waitBar(ctx, barMma, mmaPhase, 136);
            mmaPhase ^= 1u;
            tcFenceAfter();
            if (tid == 0 && p + kRingSlots < nFc2)
            {
                int const jNext = f2a + p + kRingSlots;
                int const sNext = jNext / kFc2PiecesPerSlot;
                issueFc2Piece(ctx, smemMoe[kMoeSlotExpert + sNext], jNext % kFc2PiecesPerSlot, base, barRing0 + 8 * rs);
            }
            if (warp < 4)
            {
                uint8_t* outRows = reinterpret_cast<uint8_t*>(
                    ptr<__nv_bfloat16>(ctx, kPFc2out) + static_cast<size_t>(slot) * kTok * kHidden);
                float vt[2][8];
                tmemLd2x8(tmem + kTmemD + (static_cast<uint32_t>(32 * warp) << 16), vt);
#pragma unroll
                for (int t = 0; t < kTok; ++t)
                {
                    if (((mask >> t) & 1u) == 0u)
                    {
                        continue;
                    }
#pragma unroll
                    for (int ti = 0; ti < 2; ++ti)
                    {
                        int const col = 128 * (2 * pIdx + ti) + 32 * warp + 4 * (lane & 7);
                        float const x0 = vt[ti][t] * alpha2;
                        float const x1 = __shfl_down_sync(0xffffffffu, x0, 8);
                        float const x2 = __shfl_down_sync(0xffffffffu, x0, 16);
                        float const x3 = __shfl_down_sync(0xffffffffu, x0, 24);
                        if (lane < 8)
                        {
                            st64(outRows + (static_cast<size_t>(t) * kHidden + col) * 2,
                                make_uint2(lamport(packBf16(x0, x1)), lamport(packBf16(x2, x3))));
                        }
                    }
                }
            }
            tcFenceBefore();
            __syncthreads();
            tcFenceAfter();
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Stage 4D: 12-CTA Owner (512 cols/CTA, 256/256 threads active, 0-spill! Top-4 Combine + Shared Add -> MNNVL H2)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void stageOwner(Ctx const& ctx, int J)
{
    int const tid = static_cast<int>(threadIdx.x);
    __nv_bfloat16 const* sOut = reinterpret_cast<__nv_bfloat16 const*>(ctx.smem + kSmemRed); // [4][512] = 4,096 B
    int* sSlots = reinterpret_cast<int*>(ctx.smem + kSmemRed + 4096);                        // [16]
    float* sWts = reinterpret_cast<float*>(ctx.smem + kSmemRed + 4160);                      // [16]

    // Poll published routing table from kArRoute (32 ints = 128 B, 100% L2 hit).
    if (tid < kTok * kMoeTopK)
    {
        int const* route = arena<int const>(ctx, kArRoute);
        sSlots[tid] = static_cast<int>(poll1(ctx, route + kRouteSlotOfTk + tid, 137));
        sWts[tid] = __uint_as_float(poll1(ctx, route + kRouteWeight + tid, 137));
    }
    if (tid == 128)
    {
        uint32_t const* c = ctx.counters + kCClear;
        uint64_t t0 = 0;
        uint32_t v;
        while ((v = ldAcquire(c)) < static_cast<uint32_t>(kCountedClear))
        {
            spinCheck(ctx, t0, 138, kCClear, static_cast<int>(v));
        }
    }
    __syncthreads();

    // 4 tokens x (kOwnerCols / 8) vectors of 8 BF16 columns = 128 active threads (256 cols per Owner CTA!).
    uint4 shard = make_uint4(0u, 0u, 0u, 0u);
    int tok = 0;
    int col = 0;
    int slots[kMoeTopK] = {0, 0, 0, 0};
    if (tid < kTok * (kOwnerCols / 8))
    {
        tok = tid / (kOwnerCols / 8);
        int const v = tid % (kOwnerCols / 8);
        col = kOwnerCols * J + 8 * v;
        float wts[kMoeTopK];
#pragma unroll
        for (int k = 0; k < kMoeTopK; ++k)
        {
            slots[k] = sSlots[tok * kMoeTopK + k];
            wts[k] = sWts[tok * kMoeTopK + k];
        }
        uint8_t* fc2 = reinterpret_cast<uint8_t*>(ptr<__nv_bfloat16>(ctx, kPFc2out));
        uint4 o[kMoeTopK];
#pragma unroll
        for (int k = 0; k < kMoeTopK; ++k)
        {
            o[k] = ldPoll128(fc2 + ((static_cast<size_t>(slots[k]) * kTok + tok) * kHidden + col) * 2);
        }
        uint64_t t0 = 0;
        while (true)
        {
            bool ok = full4(o[0]) & full4(o[1]) & full4(o[2]) & full4(o[3]);
            if (warpAll(ok))
            {
                break;
            }
            spinCheck(ctx, t0, 139, J, tok);
#pragma unroll
            for (int k = 0; k < kMoeTopK; ++k)
            {
                if (!full4(o[k]))
                {
                    o[k] = ldPoll128(fc2 + ((static_cast<size_t>(slots[k]) * kTok + tok) * kHidden + col) * 2);
                }
            }
        }

        float routed[8] = {};
#pragma unroll
        for (int k = 0; k < kMoeTopK; ++k)
        {
            uint32_t const ow[4] = {o[k].x, o[k].y, o[k].z, o[k].w};
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                routed[2 * j] = fmaf(wts[k], bf16Lo(ow[j]), routed[2 * j]);
                routed[2 * j + 1] = fmaf(wts[k], bf16Hi(ow[j]), routed[2 * j + 1]);
            }
        }
        uint32_t pk[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float const s0 = __bfloat162float(sOut[tok * kOwnerCols + 8 * v + 2 * j]);
            float const s1 = __bfloat162float(sOut[tok * kOwnerCols + 8 * v + 2 * j + 1]);
            pk[j] = lamport(packBf16(s0 + bf16Round(routed[2 * j]), s1 + bf16Round(routed[2 * j + 1])));
        }
        shard = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
    __syncthreads();
    if (tid < kTok * (kOwnerCols / 8))
    {
        size_t const off = (static_cast<size_t>(tok * kRanks + ctx.rank) * kHidden + col) * 2;
        st128(ptr<uint8_t>(ctx, kPH2Mc) + static_cast<size_t>(ctx.parity) * kHopBytes + off, shard);
        uint8_t* fc2 = reinterpret_cast<uint8_t*>(ptr<__nv_bfloat16>(ctx, kPFc2out));
        uint4 const e = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
#pragma unroll
        for (int k = 0; k < kMoeTopK; ++k)
        {
            st128(fc2 + ((static_cast<size_t>(slots[k]) * kTok + tok) * kHidden + col) * 2, e);
        }
    }
}
