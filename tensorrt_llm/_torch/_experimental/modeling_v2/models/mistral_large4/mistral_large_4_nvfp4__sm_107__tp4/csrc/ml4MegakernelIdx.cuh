// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Lean Leader Front-End Kernel (ml4LeaderFrontKernel: 176 CTAs = 44 clusters of 4, 132 KiB SMEM, 0 spills).
// Included inside namespace ml4 by ml4Megakernel.cu.
//
// Parallel T=0 front-end on Leader layers (43 layers, 100% SM co-resident on 212 SMs):
//   - Clusters 0..31  (128 CTAs): TCGen05 MXFP8 IdxQ (32 heads x 4 K-splits, fused 2-pass LN using Cluster 43
//     stats, cluster per-token st.async K-reduce, SSMax + RoPE/NoPE + exact 128-dim head amax + E4M3 -> idxArena;
//     skipped when L <= 2048).
//   - Clusters 32..35 (16 CTAs):  TCGen05 MXFP8 Wq (4 heads x 4 K-splits, cluster st.async K-reduce + SSMax + RoPE)
//     followed immediately by 2x Kb absorption -> kArFusedQ.
//   - Clusters 36..41 (24 CTAs):  Double-buffered streaming WkvA (18 CTAs x 2 tiles) + IdxKw (5 CTAs x 2 tiles)
//     + IdxK (1 CTA -> idxPool & kIArKNew).
//   - Cluster 42      (4 CTAs):   KvAppend (CTA 0 -> kvPool & kArKv).
//   - Cluster 43      (4 CTAs):   Dedicated 256-thread 2-pass LayerNorm stats (1 CTA per token -> kIArLnMean/Rstd).
// Then for L > 2048:
//   - IdxScore (up to 176 CTAs, prioritizing early-finishing Clusters 32..43): double-buffered TCGen05 mmaF8N128,
//     writes FP32 logits -> idxLogits & accumulates Level-0 SMEM hist -> idxArena hist0.
//     Clusters 4..43 EXIT immediately (freeing 160 SMs for PDL ml4LayerKernel).
//   - IdxTopK (ONLY Clusters 0..3 = 16 CTAs, 1 cluster of 4 CTAs per token row):
//     Coarse(64 groups = 4 KB) + Fine(32 bins = 1 KB) 16-rank MNNVL exchange + 1-level kModeCand boundary exchange
//     & local SMEM resolveBoundary256 on CTA 3 (while CTAs 0..2 clear nextIdxCopy in parallel),
//     with exact Level-1/Level-2 cluster-reduced fallback when binAll > 4096.

// ----------------------------------------------------------------------------------------------------------------
// Control SMEM offsets inside ml4LeaderFrontKernel (from kIdxSmemCtrl = 131,072, total 4,096 B)
// ----------------------------------------------------------------------------------------------------------------
constexpr int kIdxCtrlBars = 0;  // 24 barriers x 8 B = 192 B
constexpr int kIdxBarRing0 = 0;  // 0..3: 4 TMA ring barriers
constexpr int kIdxBarFree0 = 4;  // 4..7: 4 MMA-free ring barriers
constexpr int kIdxBarMma = 8;    // 8: MMA final commit
constexpr int kIdxBarAct = 9;    // 9: activation quant ready
constexpr int kIdxBarRed = 10;   // 10: cluster st.async K-reduce barrier (IdxQ / Wq)
constexpr int kIdxBarHist0 = 11; // 11..13: cluster st.async hist barriers (lvl 0..2)
constexpr int kIdxBarCnt = 14;   // 14: cluster st.async (nAbove, nBnd) count barrier
constexpr int kIdxBarK0 = 15;    // 15..18: IdxScore K0/K1 + Mma0/Mma1
constexpr int kIdxBarK1 = 16;
constexpr int kIdxBarMma0 = 17;
constexpr int kIdxBarMma1 = 18;
constexpr int kIdxBarKb = 19; // 19: Kb weight prefetch barrier
constexpr int kIdxNumBars = 20;

constexpr int kIdxCtrlScalars = 256; // 64 floats = 256 B (tmem, rcp[4], lnMean[4], lnRstd[4], claimedCluster)
constexpr int kIdxScTmem = 0;
constexpr int kIdxScClaim = 1;
constexpr int kIdxScRcp = 4;          // [4]
constexpr int kIdxScLnMean = 8;       // [4]
constexpr int kIdxScLnRstd = 12;      // [4]
constexpr int kIdxCtrlParams = 512;   // 80 int64 = 640 B
constexpr int kIdxCtrlCtx = 1152;     // 128 B
constexpr int kIdxCtrlScratch = 1280; // 2816 B scratch (reductions, peer counts, cut)
static_assert(kIdxCtrlParams + kPNumParams * 8 <= kIdxCtrlCtx, "idx front params fit");
static_assert(kIdxCtrlScratch + 2816 <= kIdxSmemCtrlBytes, "idx front ctrl fits");

// Modes for IdxTopK (identical to glmDsaHistTopK).
constexpr int kTopkModeFull = 0;
constexpr int kTopkModeTies = 1;
constexpr int kTopkModeCand = 2;
constexpr int kTopkModeNext = 3;

// ----------------------------------------------------------------------------------------------------------------
// 256-thread Block Scan & Local Cut Helpers (for resolveBoundary & IdxTopK)
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t blockExclusiveSuffix256(uint32_t v, uint32_t* ws, uint32_t& total)
{
    int const lane = static_cast<int>(threadIdx.x) & 31;
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    uint32_t x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        uint32_t const y = __shfl_down_sync(0xffffffffu, x, o);
        if (lane + o < 32)
        {
            x += y;
        }
    }
    if (lane == 0)
    {
        ws[warp] = x;
    }
    __syncthreads();
    uint32_t higher = x - v;
    uint32_t tot = 0u;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
    {
        uint32_t const ww = ws[w];
        tot += ww;
        if (w > warp)
        {
            higher += ww;
        }
    }
    total = tot;
    __syncthreads();
    return higher;
}

__device__ __forceinline__ uint32_t blockExclusiveScan256(uint32_t v, uint32_t* ws, uint32_t& total)
{
    int const lane = static_cast<int>(threadIdx.x) & 31;
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    uint32_t x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        uint32_t const y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o)
        {
            x += y;
        }
    }
    if (lane == 31)
    {
        ws[warp] = x;
    }
    __syncthreads();
    uint32_t pre = x - v;
    uint32_t tot = 0u;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
    {
        uint32_t const ww = ws[w];
        if (w < warp)
        {
            pre += ww;
        }
        tot += ww;
    }
    total = tot;
    __syncthreads();
    return pre;
}

// Find cut bin of a local histogram (nbins <= 2048, up to 8 bins per thread across 256 threads).
// Writes cut[0] = bin, cut[1] = aboveAll, cut[2] = binCount.
__device__ __forceinline__ void findLocalCut256(
    uint32_t const* h, int nbins, uint32_t need, uint32_t* ws, uint32_t* cut)
{
    int const tid = static_cast<int>(threadIdx.x);
    uint32_t c8[8];
    uint32_t sum = 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        int const b = 8 * tid + j;
        c8[j] = b < nbins ? h[b] : 0u;
        sum += c8[j];
    }
    uint32_t total = 0u;
    uint32_t acc = blockExclusiveSuffix256(sum, ws, total);
#pragma unroll
    for (int j = 7; j >= 0; --j)
    {
        int const b = 8 * tid + j;
        if (b < nbins)
        {
            if (acc < need && need <= acc + c8[j])
            {
                cut[0] = static_cast<uint32_t>(b);
                cut[1] = acc;
                cut[2] = c8[j];
            }
            acc += c8[j];
        }
    }
    __syncthreads();
}

// ----------------------------------------------------------------------------------------------------------------
// Local Exact Boundary Resolver on 256 Threads (kModeCand: nb <= 4096 pairs in SMEM -> exact top-need tokens)
// Tie-breaks identically to glmDsaHistTopK: larger (orderKey, globalPos) first; emits in deterministic pair order.
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void resolveBoundary256(uint32_t const* sKeys, uint32_t const* sToks, int nb, uint32_t need,
    int shift, uint32_t keyTop, uint32_t* rh, uint32_t* ws, uint32_t* cut, uint32_t* small, uint32_t* smallCount,
    int* outTkUc, int baseOut)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const w1 = shift > 11 ? 11 : shift;
    int const sh1 = shift - w1;
    uint32_t const mask1 = (1u << w1) - 1u;
    int const nBins1 = 1 << w1;

    for (int i = tid; i < nBins1; i += kThreads)
    {
        rh[i] = 0u;
    }
    if (tid == 0)
    {
        *smallCount = 0u;
    }
    __syncthreads();

    for (int i = tid; i < nb; i += kThreads)
    {
        atomicAdd(&rh[(sKeys[i] >> sh1) & mask1], 1u);
    }
    __syncthreads();
    findLocalCut256(rh, nBins1, need, ws, cut);
    uint32_t const d1 = cut[0];
    uint32_t const need1 = need - cut[1];
    uint32_t const cnt1 = cut[2];

    uint32_t tKey = 0u;
    uint32_t tTok = 0u;
    if (cnt1 <= static_cast<uint32_t>(kIdxSmallRank))
    {
        for (int i = tid; i < nb; i += kThreads)
        {
            if (((sKeys[i] >> sh1) & mask1) == d1)
            {
                uint32_t const sl = atomicAdd(smallCount, 1u);
                small[sl] = sKeys[i];
                small[kIdxSmallRank + sl] = sToks[i];
            }
        }
        __syncthreads();
        if (tid < static_cast<int>(cnt1))
        {
            uint32_t const k = small[tid];
            uint32_t const tok = small[kIdxSmallRank + tid];
            uint32_t higher = 0u;
            for (uint32_t j = 0; j < cnt1; ++j)
            {
                uint32_t const kj = small[j];
                uint32_t const tj = small[kIdxSmallRank + j];
                higher += (kj > k || (kj == k && tj > tok)) ? 1u : 0u;
            }
            if (higher == need1 - 1u)
            {
                cut[0] = k;
                cut[1] = tok;
            }
        }
        __syncthreads();
        tKey = cut[0];
        tTok = cut[1];
    }
    else
    {
        uint32_t needR = need1;
        uint32_t prefix = d1;
        uint32_t tie = cnt1;
        if (sh1 > 0)
        {
            int const nBins2 = 1 << sh1;
            uint32_t const mask2 = static_cast<uint32_t>(nBins2 - 1);
            for (int i = tid; i < nBins2; i += kThreads)
            {
                rh[i] = 0u;
            }
            __syncthreads();
            for (int i = tid; i < nb; i += kThreads)
            {
                if (((sKeys[i] >> sh1) & mask1) == d1)
                {
                    atomicAdd(&rh[sKeys[i] & mask2], 1u);
                }
            }
            __syncthreads();
            findLocalCut256(rh, nBins2, needR, ws, cut);
            prefix = (d1 << sh1) | cut[0];
            needR -= cut[1];
            tie = cut[2];
        }
        uint32_t const kLow = prefix;
        uint32_t const lowMask = (1u << shift) - 1u;
        uint32_t tokPrefix = 0u;
        if (needR != tie)
        {
            int rem = 21;
            while (rem > 0)
            {
                int const width = rem > 11 ? 11 : rem;
                int const sh = rem - width;
                int const nbW = 1 << width;
                uint32_t const matchMask = (1u << (21 - rem)) - 1u;
                for (int i = tid; i < nbW; i += kThreads)
                {
                    rh[i] = 0u;
                }
                __syncthreads();
                for (int i = tid; i < nb; i += kThreads)
                {
                    if ((sKeys[i] & lowMask) == kLow && ((sToks[i] >> rem) & matchMask) == tokPrefix)
                    {
                        atomicAdd(&rh[(sToks[i] >> sh) & static_cast<uint32_t>(nbW - 1)], 1u);
                    }
                }
                __syncthreads();
                findLocalCut256(rh, nbW, needR, ws, cut);
                tokPrefix = (tokPrefix << width) | cut[0];
                needR -= cut[1];
                rem = sh;
            }
        }
        tKey = keyTop | kLow;
        tTok = tokPrefix;
    }

    // Deterministic emit in pair order (strictly bit-identical across all 16 ranks!).
    // Process in chunks of 256 pairs so each thread uses 1 bit per chunk with blockExclusiveScan256.
    int emitted = 0;
    for (int off = 0; off < nb; off += kThreads)
    {
        int const idx = off + tid;
        bool const win = idx < nb && (sKeys[idx] > tKey || (sKeys[idx] == tKey && sToks[idx] >= tTok));
        uint32_t chunkTot = 0u;
        uint32_t const pos = blockExclusiveScan256(win ? 1u : 0u, ws, chunkTot);
        if (win)
        {
            st32(outTkUc + baseOut + emitted + static_cast<int>(pos), lamport(sToks[idx]));
        }
        emitted += static_cast<int>(chunkTot);
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Front-End Helper: Indexer K (Stage 2A, identical lossless numerics, writing to idxArena & idxPool)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void frontStageIdxK(Ctx const& ctx, uint8_t* idxArena, int const* positions)
{
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    int const lane = static_cast<int>(threadIdx.x) & 31;
    if (warp >= kTok)
    {
        return;
    }
    int const t = warp;
    bool const isNope = (ctx.flags & kFlagNope) != 0;
    int const pos = __ldg(positions + t);
    int const blk = __ldg(ptr<int const>(ctx, kPIdxBlocks) + (pos >> ctx.tpbShift));

    float4 const lw = __ldg(reinterpret_cast<float4 const*>(ptr<float const>(ctx, kPIdxKnW)) + lane);
    float4 const lb = __ldg(reinterpret_cast<float4 const*>(ptr<float const>(ctx, kPIdxKnB)) + lane);
    float2 cs = make_float2(1.f, 1.f);
    float2 sn = make_float2(0.f, 0.f);
    if (!isNope && lane < 16)
    {
        float const* row = ptr<float const>(ctx, kPRope) + static_cast<size_t>(pos) * 64;
        cs = __ldg(reinterpret_cast<float2 const*>(row) + lane);
        sn = __ldg(reinterpret_cast<float2 const*>(row + 32) + lane);
    }

    uint8_t const* src = idxArena + kIArKw + (t * kIdxKwCols + 4 * lane) * 2;
    uint2 v = ldPoll64(src);
    {
        uint64_t t0 = 0;
        while (!warpAll(full2(v)))
        {
            spinCheck(ctx, t0, 150, t, lane);
            if (!full2(v))
            {
                v = ldPoll64(src);
            }
        }
    }

    float const x[4] = {bf16Lo(v.x), bf16Hi(v.x), bf16Lo(v.y), bf16Hi(v.y)};
    float sum = (x[0] + x[1]) + (x[2] + x[3]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    float const mean = sum / static_cast<float>(kIdxDim);
    float kc[4];
    float var = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        kc[i] = x[i] - mean;
        var = fmaf(kc[i], kc[i], var);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        var += __shfl_xor_sync(0xffffffffu, var, off);
    }
    float const rstd = rsqrtf(var / static_cast<float>(kIdxDim) + ctx.idxEps);
    float const w[4] = {lw.x, lw.y, lw.z, lw.w};
    float const b[4] = {lb.x, lb.y, lb.z, lb.w};
    float k[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        k[i] = bf16Round(fmaf(kc[i] * rstd, w[i], b[i]));
    }
    if (!isNope && lane < 16)
    {
        float const r0 = bf16Round(fmaf(k[0], cs.x, __fmul_rn(-k[1], sn.x)));
        float const r1 = bf16Round(fmaf(k[1], cs.x, __fmul_rn(k[0], sn.x)));
        float const r2 = bf16Round(fmaf(k[2], cs.y, __fmul_rn(-k[3], sn.y)));
        float const r3 = bf16Round(fmaf(k[3], cs.y, __fmul_rn(k[2], sn.y)));
        k[0] = r0;
        k[1] = r1;
        k[2] = r2;
        k[3] = r3;
    }

    float amax = fmaxf(fmaxf(fabsf(k[0]), fabsf(k[1])), fmaxf(fabsf(k[2]), fabsf(k[3])));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    float const scale = ue8m0Scale(amax);
    float const inv = pow2Inv(scale);
    uint32_t const word = toE4m3x2(__fmul_rn(k[0], inv), __fmul_rn(k[1], inv))
        | (toE4m3x2(__fmul_rn(k[2], inv), __fmul_rn(k[3], inv)) << 16);

    int const slotInBlk = pos & (ctx.tpb - 1);
    uint8_t* page = ptr<uint8_t>(ctx, kPIdxPool) + static_cast<size_t>(blk) * ctx.p[kPIdxBlockStride];
    st32(page + slotInBlk * kIdxDim + 4 * lane, word);
    st32(idxArena + kIArKNew + t * kIdxDim + 4 * lane, lamport(word));
    if (lane == 0)
    {
        uint32_t const sb = __float_as_uint(scale);
        st32(page + ctx.tpb * kIdxDim + slotInBlk * 4, sb);
        st32(idxArena + kIArKNewS + 4 * t, lamport(sb));
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Front-End Helper: Streaming 2-Stage Double-Buffered BF16 WkvA / IdxKw (2 tiles/CTA, fits in 98 KB SMEM!)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void frontStageWkvATwoTiles(Ctx const& ctx, int tile0, int nTiles, uint8_t* dstBase, int dstLd,
    int dstRowBase, __nv_bfloat16 const* x, float const* sRcp, bool early)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    // Stream 96 BF16 chunks in 4 batches of 24 chunks (49,152 B/batch) in a 2-slot ring (98,304 B <= 128 KB).
    constexpr int kBatchChunks = 24;
    constexpr int kBatchBytes = kBatchChunks * kBf16Block;     // 49,152 B
    constexpr int kBatches = kHiddenChunksBf16 / kBatchChunks; // 4
    uint32_t const bar0 = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarRing0 * 8;
    uint8_t const* wBase = ptr<uint8_t const>(ctx, kPWkvA);
    bool const evictFirst = (ctx.flags & kFlagEvictFirst) != 0;

    // Load or poll this warp's 12 BF16 chunks of x once and reuse across both output tiles!
    constexpr int kPer = kHiddenChunksBf16 / kWarps; // 12 chunks/warp (3 per batch)
    uint4 b[kPer][2];
    if (early)
    {
        pollB<kPer>(ctx, b, arena<uint8_t const>(ctx, kArX), kHidden, warp * kPer, 103);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < kPer; ++i)
        {
            loadB(b[i], x, kHidden, warp * kPer + i);
        }
    }

    for (int ti = 0; ti < nTiles; ++ti)
    {
        int const tile = tile0 + ti;
        if (tid < 32)
        {
            if (tid < 2)
            {
                mbarReinit(bar0 + 8 * tid, 1);
            }
            __syncwarp();
            if (tid == 0)
            {
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            }
        }
        __syncthreads();

        auto issueBatch = [&](int bi)
        {
            int const slot = bi & 1;
            uint32_t const dst = ctx.smemBase + static_cast<uint32_t>(slot) * kBatchBytes;
            uint8_t const* src
                = wBase + (static_cast<size_t>(tile) * kHiddenChunksBf16 + bi * kBatchChunks) * kBf16Block;
            mbarExpectTx(bar0 + 8 * slot, kBatchBytes);
            copyPieces(dst, src, kBatchBytes, bar0 + 8 * slot, evictFirst);
        };

        if (tid == 0)
        {
            issueBatch(0);
            issueBatch(1);
        }

        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int bi = 0; bi < kBatches; ++bi)
        {
            int const slot = bi & 1;
            uint32_t const ph = static_cast<uint32_t>((bi >> 1) & 1);
            waitBar(ctx, bar0 + 8 * slot, ph, 105);
            // Warp w owns global chunks [12*w .. 12*w+11]. Batch bi covers [24*bi .. 24*bi+23] = warps 2*bi and 2*bi+1!
            int const w0 = 2 * bi;
            if (warp == w0 || warp == w0 + 1)
            {
                int const baseInBatch = (warp - w0) * kPer;
                uint32_t const sSlot = ctx.smemBase + static_cast<uint32_t>(slot) * kBatchBytes;
#pragma unroll
                for (int i = 0; i < kPer; ++i)
                {
                    mmaChunkBf16(acc, sSlot + (baseInBatch + i) * kBf16Block, b[i]);
                }
            }
            __syncthreads();
            if (tid == 0 && bi + 2 < kBatches)
            {
                issueBatch(bi + 2);
            }
        }

        // Reduce 8 warps via kIdxCtrlScratch (8*16*4 = 512 floats = 2,048 B).
        float* red = reinterpret_cast<float*>(ctx.smem + kIdxSmemCtrl + kIdxCtrlScratch);
        if ((lane & 3) < 2)
        {
            int const cl = ((lane >> 2) << 1) | (lane & 1);
            *reinterpret_cast<float4*>(red + (warp * 16 + cl) * 4) = make_float4(acc[0], acc[1], acc[2], acc[3]);
        }
        __syncthreads();
        float* out = red + 512; // 64 floats
        if (tid < 64)
        {
            int const cl = tid >> 2;
            int const r = tid & 3;
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                s += red[(w * 16 + cl) * 4 + r];
            }
            int const g = cl >> 1;
            int const c = cl & 1;
            int const row = g + (r >= 2 ? 8 : 0);
            int const tok = 2 * c + (r & 1);
            out[row * kTok + tok] = s;
        }
        __syncthreads();
        storeTileBf16(out, dstBase, dstLd, 16 * (dstRowBase + ti), early ? sRcp : nullptr);
        __syncthreads();
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Front-End Helper: Cluster TCGen05 MXFP8 IdxQ (Clusters 0..15, 2 heads/cluster = 4 K-splits/head, fused 2-pass LN)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void frontStageIdxQ(Ctx const& ctx, int cid, int crank, uint8_t* idxArena,
    __nv_bfloat16 const* x, float const* sRcp, bool early, int const* positions)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    uint32_t const tmem
        = *reinterpret_cast<uint32_t const*>(ctx.smem + kIdxSmemCtrl + kIdxCtrlScalars + 4 * kIdxScTmem);
    int* sClaim = reinterpret_cast<int*>(ctx.smem + kIdxSmemCtrl + kIdxCtrlScalars + 4 * kIdxScClaim);
    float* sLnMean = reinterpret_cast<float*>(ctx.smem + kIdxSmemCtrl + kIdxCtrlScalars + 4 * kIdxScLnMean);
    float* sLnRstd = reinterpret_cast<float*>(ctx.smem + kIdxSmemCtrl + kIdxCtrlScalars + 4 * kIdxScLnRstd);
    uint32_t const barRing0 = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarRing0 * 8;
    uint32_t const barFree0 = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarFree0 * 8;
    uint32_t const barMma = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarMma * 8;
    uint32_t const barAct = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarAct * 8;
    uint32_t const barRed = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarRed * 8;

    // Optional Paradigm-B Die-Aware head claim (32 clusters -> 16 Die-0 heads 0..15, 16 Die-1 heads 16..31).
    int claimedCid = cid;
    if ((ctx.flags & kFlagDieAware) != 0 && ctx.p[kPSmidMap] != 0 && ctx.p[kPUgpuState] != 0)
    {
        if (crank == 0 && tid == 0)
        {
            int const* smMap = reinterpret_cast<int const*>(ctx.p[kPSmidMap]);
            int const lcta = __ldg(smMap + getSmid());
            int const die = (lcta >= 0 && lcta < ctx.nDie0) ? 0 : 1;
            uint32_t* st = reinterpret_cast<uint32_t*>(ctx.p[kPUgpuState]);
            uint32_t const enter = atomicAdd(st, 1u);
            uint32_t const slot = (enter >> 5) & 3u;
            if ((enter & 31u) == 31u)
            {
                uint32_t const clr = (slot + 2u) & 3u;
                stReleaseU32(st + 4 + clr * 4 + 0, 0u);
                stReleaseU32(st + 4 + clr * 4 + 1, 0u);
            }
            uint32_t const tOwn = atomicAdd(st + 4 + slot * 4 + die, 1u);
            int cClaim;
            if (tOwn < 16u)
            {
                cClaim = die * 16 + static_cast<int>(tOwn);
            }
            else
            {
                int const other = 1 - die;
                uint32_t const tOther = atomicAdd(st + 4 + slot * 4 + other, 1u);
                cClaim = other * 16 + static_cast<int>(tOther);
            }
            *sClaim = cClaim;
            uint32_t const cAddr = smemU32(sClaim);
#pragma unroll
            for (int p = 1; p < kIdxClusterSize; ++p)
            {
                uint32_t const rem = peerAddr(cAddr, p);
                asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(rem), "r"(cClaim) : "memory");
            }
        }
        clusterArriveRelease();
        clusterWait();
        __syncthreads();
        claimedCid = *sClaim;
    }

    int const head = claimedCid; // 0..31 (1 cluster of 4 CTAs per head)
    int const subRank = crank;   // 0..3 (K-split AND owned output token!)
    int const kTile0 = subRank * 12;

    uint8_t const* actBase = early ? arena<uint8_t const>(ctx, kArX) : reinterpret_cast<uint8_t const*>(x);
    uint8_t const* wHead
        = ptr<uint8_t const>(ctx, kPIdxWq) + static_cast<size_t>(head) * kHiddenBlocksFp8 * kMxf8PieceBytes;
    __nv_bfloat16 const* qnW = ptr<__nv_bfloat16 const>(ctx, kPIdxQNormW);
    __nv_bfloat16 const* qnB = ptr<__nv_bfloat16 const>(ctx, kPIdxQNormB);

    // SMEM layout in kIdxSmemMain (128 KB):
    //   [0 .. 69,632): 4-slot 1024B-aligned MXFP8 TMA ring (4 x 17,408 B)
    //   [69,632 .. 81,920): 12 x 1024 B smemB
    //   [81,920 .. 83,456): 12 x 128 B smemSfb (utccp1Rows8)
    //   [83,456 .. 85,504): 2,048 B cluster K-reduce mailbox (4 splits x 128 floats for token subRank!)
    constexpr uint32_t kRing = 4;
    constexpr uint32_t kSmemBOff = kRing * kMxf8SlotStride;
    constexpr uint32_t kSmemSfbOff = kSmemBOff + 12u * 1024u;
    constexpr uint32_t kSmemMbOff = kSmemSfbOff + 12u * 128u;
    static_assert(kSmemMbOff + 2048u <= kIdxSmemMain, "IdxQ SMEM fits");

    uint8_t* smemB = ctx.smem + kSmemBOff;
    uint8_t* smemSfb = ctx.smem + kSmemSfbOff;
    float* sMailbox = reinterpret_cast<float*>(ctx.smem + kSmemMbOff);
    uint32_t const aSmem = ctx.smemBase;
    uint32_t const bSmem = ctx.smemBase + kSmemBOff;
    uint32_t const sfbSmem = ctx.smemBase + kSmemSfbOff;

    bool const evictFirst = (ctx.flags & kFlagEvictFirst) != 0;
    uint64_t const policy = evictFirst ? policyEvictFirst() : 0ull;

    if (warp == 0)
    {
        // Stream all 12 weight+scale tiles into 1024B-aligned 4-slot TMA ring while warps 4..7 compute LN + quant!
        if (lane == 0)
        {
            for (int ki = 0; ki < 12; ++ki)
            {
                int const st = ki & 3;
                if (ki >= 4)
                {
                    waitBar(ctx, barFree0 + 8 * st, static_cast<uint32_t>(((ki >> 2) - 1) & 1), 153);
                }
                uint32_t const dst = aSmem + static_cast<uint32_t>(st) * kMxf8SlotStride;
                uint8_t const* src = wHead + static_cast<size_t>(kTile0 + ki) * kMxf8PieceBytes;
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
            for (int ki = 0; ki < 12; ++ki)
            {
                utccp1Rows8(tmem + 48u + 4u * static_cast<uint32_t>(ki), sfbSmem + 128u * static_cast<uint32_t>(ki));
            }
        }
        __syncwarp();
        for (int ki = 0; ki < 12; ++ki)
        {
            int const st = ki & 3;
            waitBar(ctx, barRing0 + 8 * st, static_cast<uint32_t>((ki >> 2) & 1), 153);
            tcFenceAfter();
            if (electOne())
            {
                uint32_t const aSlot = aSmem + static_cast<uint32_t>(st) * kMxf8SlotStride;
                uint32_t const sfaTmem = tmem + 16u + 4u * static_cast<uint32_t>(st);
                uint32_t const sfbTmem = tmem + 48u + 4u * static_cast<uint32_t>(ki);
                utccp1(sfaTmem, aSlot + kMxf8WBytes);
                uint32_t const bTile = bSmem + 1024u * static_cast<uint32_t>(ki);
#pragma unroll
                for (int kb = 0; kb < 4; ++kb)
                {
                    mmaMxf8(tmem + 0u, aSlot + 32u * static_cast<uint32_t>(kb), bTile + 32u * static_cast<uint32_t>(kb),
                        sfaTmem, sfbTmem, static_cast<uint32_t>(kb), (ki > 0 || kb > 0) ? 1u : 0u);
                }
                if (ki + 4 < 12)
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
        // Poll exact 2-pass LayerNorm (mean[4], rstd[4]) published by dedicated Cluster 43 (~0.7 us vs 4.5 us x 128
        // CTAs!).
        if (tid == 128)
        {
            uint4 const mv = poll4(ctx, idxArena + kIArLnMean, 151);
            uint4 const rv = poll4(ctx, idxArena + kIArLnRstd, 151);
            *reinterpret_cast<uint4*>(sLnMean) = mv;
            *reinterpret_cast<uint4*>(sLnRstd) = rv;
        }
        asm volatile("bar.sync 2, 128;" ::: "memory");

        // All 128 threads quantize ONLY this CTA's 12 K-tiles into SMEM (with fused LayerNorm)!
        if (early)
        {
            quantActMxf8Warps47<true, true>(
                ctx, actBase, kHidden, sRcp, sLnMean, sLnRstd, qnW, qnB, kTile0, 12, smemB, smemSfb, 152);
        }
        else
        {
            quantActMxf8Warps47<false, true>(
                ctx, actBase, kHidden, nullptr, sLnMean, sLnRstd, qnW, qnB, kTile0, 12, smemB, smemSfb, 152);
        }
        fenceProxyAsyncSmem();
        asm volatile("bar.sync 2, 128;" ::: "memory");
        if (tid == 128)
        {
            mbarArrive(barAct);
        }
    }

    waitBar(ctx, barMma, 0, 153);
    __syncthreads();
    tcFenceAfter();

    // Cluster K-reduce: route token t's 128 dims to peer (half*4 + t) via st.async.shared::cluster!
    if (warp >= 4)
    {
        int const q = warp - 4;
        int const dim = 32 * q + lane;
        float v[8];
        tmemLd8(tmem + 0u + (static_cast<uint32_t>(32 * q) << 16), v);
#pragma unroll
        for (int t = 0; t < kTok; ++t)
        {
            if (t == subRank)
            {
                sMailbox[subRank * 128 + dim] = v[t];
            }
            else
            {
                float const v1 = __shfl_down_sync(0xffffffffu, v[t], 1);
                float const v2 = __shfl_down_sync(0xffffffffu, v[t], 2);
                float const v3 = __shfl_down_sync(0xffffffffu, v[t], 3);
                if ((lane & 3) == 0)
                {
                    int const peer = t;
                    uint32_t const dst = peerAddr(smemU32(sMailbox + subRank * 128 + dim), peer);
                    pushClusterV4(dst,
                        make_uint4(
                            __float_as_uint(v[t]), __float_as_uint(v1), __float_as_uint(v2), __float_as_uint(v3)),
                        peerAddr(barRed, peer));
                }
            }
        }
    }
    mbarWaitSpin(barRed, 0);
    __syncthreads();

    // Epilogue for (head, tok = subRank): ALL 128 dims live in warps 4..7! Zero kArIdxAmax global poll!
    if (warp >= 4)
    {
        int const q = warp - 4;
        int const dim = 32 * q + lane;
        int const tok = subRank;
        float const sum
            = (sMailbox[0 * 128 + dim] + sMailbox[1 * 128 + dim]) + (sMailbox[2 * 128 + dim] + sMailbox[3 * 128 + dim]);

        int const pos = __ldg(positions + tok);
        bool const isNope = (ctx.flags & kFlagNope) != 0;
        float const ssS = __bfloat162float(__ldg(ptr<__nv_bfloat16 const>(ctx, kPIdxSsmaxS) + head));
        float const ssB = __bfloat162float(__ldg(ptr<__nv_bfloat16 const>(ctx, kPIdxSsmaxB) + head));
        float const ss = ssmaxScaleBf16(pos, ssS, ssB);

        float val = bf16Round(sum) * ss;
        if (!isNope && dim < 64)
        {
            val = bf16RoundFtz(val);
            float const partner = __shfl_xor_sync(0xffffffffu, val, 1);
            float const* cs = ptr<float const>(ctx, kPRope) + static_cast<size_t>(pos) * 64;
            float const c = __ldg(cs + (dim >> 1));
            float const s = __ldg(cs + 32 + (dim >> 1));
            if ((dim & 1) == 0)
            {
                val = bf16Round(fmaf(val, c, __fmul_rn(-partner, s)));
            }
            else
            {
                val = bf16Round(fmaf(val, c, __fmul_rn(partner, s)));
            }
        }
        else
        {
            val = bf16Round(val);
        }

        // Exact 128-dim head amax across warps 4..7 via 4 floats in kIdxCtrlScratch!
        float wMax = fabsf(val);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            wMax = fmaxf(wMax, __shfl_xor_sync(0xffffffffu, wMax, off));
        }
        float* sWMax = reinterpret_cast<float*>(ctx.smem + kIdxSmemCtrl + kIdxCtrlScratch);
        if (lane == 0)
        {
            sWMax[q] = wMax;
        }
        asm volatile("bar.sync 2, 128;" ::: "memory");
        float const headAmax = fmaxf(fmaxf(sWMax[0], sWMax[1]), fmaxf(sWMax[2], sWMax[3]));
        float const qScale = ue8m0Scale(headAmax);
        float const inv = pow2Inv(qScale);

        float const qNext = __shfl_down_sync(0xffffffffu, val, 1);
        uint32_t const two = toE4m3x2(__fmul_rn(val, inv), __fmul_rn(qNext, inv));
        uint32_t const twoNext = __shfl_down_sync(0xffffffffu, two, 2);
        if ((lane & 3) == 0)
        {
            st32(idxArena + kIArQ8 + (tok * kIdxHeads + head) * kIdxDim + dim, lamport(two | (twoNext << 16)));
        }
        if (dim == 0)
        {
            uint8_t const* kwPtr = idxArena + kIArKw + (tok * kIdxKwCols + kIdxDim + (head & ~1)) * 2;
            uint32_t const wr = poll1(ctx, kwPtr, 155);
            float const wraw = (head & 1) != 0 ? bf16Hi(wr) : bf16Lo(wr);
            float const weff = __fmul_rn(__fmul_rn(wraw, qScale), kIdxWeightScale);
            st32(idxArena + kIArW + (tok * kIdxHeads + head) * 4, lamport(__float_as_uint(weff)));
        }
    }
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
}

// ----------------------------------------------------------------------------------------------------------------
// Front-End Helper: Cluster TCGen05 MXFP8 Wq + 2x Kb (Clusters 32..35: 1 cluster of 4 CTAs per Q head!)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void frontStageWqAndKb(
    Ctx const& ctx, int head, int crank, __nv_bfloat16 const* x, float const* sRcp, bool early, int const* positions)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    uint32_t const tmem
        = *reinterpret_cast<uint32_t const*>(ctx.smem + kIdxSmemCtrl + kIdxCtrlScalars + 4 * kIdxScTmem);
    uint32_t const barRing0 = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarRing0 * 8;
    uint32_t const barFree0 = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarFree0 * 8;
    uint32_t const barMma = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarMma * 8;
    uint32_t const barAct = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarAct * 8;
    uint32_t const barRed = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarRed * 8;
    uint32_t const barKb = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarKb * 8;

    bool const isNope = (ctx.flags & kFlagNope) != 0;
    int const kbUnit0 = head * kKbUnitsPerHead + 2 * crank;
    int const kbBytes = (isNope ? 2 : 1) * 4 * kBf16Block; // 8,192 or 16,384 B per Kb unit

    // SMEM layout (118.8 KB <= 131 KB):
    //   [0 .. 69,632): 4-slot 1024B-aligned MXFP8 TMA ring
    //   [69,632 .. 81,920): 12 x 1024 B smemB
    //   [81,920 .. 83,456): 12 x 128 B smemSfb (utccp1Rows8)
    //   [83,456 .. 85,504): 2,048 B cluster K-reduce mailbox (4 splits x 32 rows x 4 tok)
    //   [86,016 .. 118,784): 2 contiguous Kb units (16 KB RoPE or 32 KB NoPE, prefetched RIGHT NOW at T=0!)
    constexpr uint32_t kSmemBOff = 4u * kMxf8SlotStride;
    constexpr uint32_t kSmemSfbOff = kSmemBOff + 12u * 1024u;
    constexpr uint32_t kSmemMbOff = kSmemSfbOff + 12u * 128u;
    constexpr uint32_t kSmemKbOff = 86016u;
    static_assert(kSmemMbOff + 2048u <= kSmemKbOff && kSmemKbOff + 32768u <= kIdxSmemMain, "Wq+2xKb SMEM fits");

    if (tid == 0)
    {
        uint8_t const* kbSrc = ptr<uint8_t const>(ctx, kPKb) + static_cast<size_t>(kbUnit0) * kbBytes;
        mbarExpectTx(barKb, static_cast<uint32_t>(2 * kbBytes));
        copyPieces(ctx.smemBase + kSmemKbOff, kbSrc, static_cast<uint32_t>(2 * kbBytes), barKb,
            (ctx.flags & kFlagEvictFirst) != 0);
    }

    uint8_t const* actBase = early ? arena<uint8_t const>(ctx, kArX) : reinterpret_cast<uint8_t const*>(x);
    uint8_t const* wHead
        = ptr<uint8_t const>(ctx, kPWq) + static_cast<size_t>(head) * kHiddenBlocksFp8 * kMxf8PieceBytes;
    uint8_t* smemB = ctx.smem + kSmemBOff;
    uint8_t* smemSfb = ctx.smem + kSmemSfbOff;
    float* sMailbox = reinterpret_cast<float*>(ctx.smem + kSmemMbOff);
    int const kTile0 = crank * 12;
    bool const evictFirst = (ctx.flags & kFlagEvictFirst) != 0;
    uint64_t const policy = evictFirst ? policyEvictFirst() : 0ull;

    if (warp == 0)
    {
        if (lane == 0)
        {
            for (int ki = 0; ki < 12; ++ki)
            {
                int const st = ki & 3;
                if (ki >= 4)
                {
                    waitBar(ctx, barFree0 + 8 * st, static_cast<uint32_t>(((ki >> 2) - 1) & 1), 100);
                }
                uint32_t const dst = ctx.smemBase + static_cast<uint32_t>(st) * kMxf8SlotStride;
                uint8_t const* src = wHead + static_cast<size_t>(kTile0 + ki) * kMxf8PieceBytes;
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
            for (int ki = 0; ki < 12; ++ki)
            {
                utccp1Rows8(tmem + 48u + 4u * static_cast<uint32_t>(ki), ctx.smemBase + kSmemSfbOff + 128u * ki);
            }
        }
        __syncwarp();
        for (int ki = 0; ki < 12; ++ki)
        {
            int const st = ki & 3;
            waitBar(ctx, barRing0 + 8 * st, static_cast<uint32_t>((ki >> 2) & 1), 100);
            tcFenceAfter();
            if (electOne())
            {
                uint32_t const aSlot = ctx.smemBase + static_cast<uint32_t>(st) * kMxf8SlotStride;
                uint32_t const sfaTmem = tmem + 16u + 4u * static_cast<uint32_t>(st);
                uint32_t const sfbTmem = tmem + 48u + 4u * static_cast<uint32_t>(ki);
                utccp1(sfaTmem, aSlot + kMxf8WBytes);
                uint32_t const bTile = ctx.smemBase + kSmemBOff + 1024u * ki;
#pragma unroll
                for (int kb = 0; kb < 4; ++kb)
                {
                    mmaMxf8(tmem + 0u, aSlot + 32u * static_cast<uint32_t>(kb), bTile + 32u * static_cast<uint32_t>(kb),
                        sfaTmem, sfbTmem, static_cast<uint32_t>(kb), (ki > 0 || kb > 0) ? 1u : 0u);
                }
                if (ki + 4 < 12)
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
        if (early)
        {
            quantActMxf8Warps47<true, false>(
                ctx, actBase, kHidden, sRcp, nullptr, nullptr, nullptr, nullptr, kTile0, 12, smemB, smemSfb, 100);
        }
        else
        {
            quantActMxf8Warps47<false, false>(
                ctx, actBase, kHidden, nullptr, nullptr, nullptr, nullptr, nullptr, kTile0, 12, smemB, smemSfb, 100);
        }
        fenceProxyAsyncSmem();
        asm volatile("bar.sync 2, 128;" ::: "memory");
        if (tid == 128)
        {
            mbarArrive(barAct);
        }
    }

    waitBar(ctx, barMma, 0, 102);
    __syncthreads();
    tcFenceAfter();

    // Cluster K-reduce: each CTA owner q in 0..3 owns 32 rows [32*q .. 32*q+31] x 4 tokens (128 floats = 512 B/peer)!
    if (warp >= 4)
    {
        int const q = warp - 4;
        int const dim = 32 * q + lane;
        int const owner = q;   // 0..3
        int const rLoc = lane; // 0..31
        float v[8];
        tmemLd8(tmem + 0u + (static_cast<uint32_t>(32 * q) << 16), v);
        uint4 const pack
            = make_uint4(__float_as_uint(v[0]), __float_as_uint(v[1]), __float_as_uint(v[2]), __float_as_uint(v[3]));
        if (owner == crank)
        {
            *reinterpret_cast<uint4*>(sMailbox + (crank * 32 + rLoc) * 4) = pack;
        }
        else
        {
            uint32_t const dst = peerAddr(smemU32(sMailbox + (crank * 32 + rLoc) * 4), owner);
            pushClusterV4(dst, pack, peerAddr(barRed, owner));
        }
        (void) dim;
    }
    mbarWaitSpin(barRed, 0);
    __syncthreads();

    // 64 threads reduce 32 rows x 4 tokens, apply SSMax + Q-RoPE/NoPE -> kArQNope & kArFusedQ[512..575]!
    if (tid < kTok * 16)
    {
        int const tok = tid / 16;
        int const pr = tid % 16;
        int const d0 = crank * 32 + 2 * pr;
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int p = 0; p < 4; ++p)
        {
            s0 += sMailbox[(p * 32 + 2 * pr) * 4 + tok];
            s1 += sMailbox[(p * 32 + 2 * pr + 1) * 4 + tok];
        }
        int const pos = __ldg(positions + tok);
        float const ssS = __bfloat162float(__ldg(ptr<__nv_bfloat16 const>(ctx, kPSsmaxS) + head));
        float const ssB = __bfloat162float(__ldg(ptr<__nv_bfloat16 const>(ctx, kPSsmaxB) + head));
        float const sc = ssmaxScaleBf16(pos, ssS, ssB);
        float const v0 = bf16Round(bf16Round(s0) * sc);
        float const v1 = bf16Round(bf16Round(s1) * sc);
        if (!isNope && d0 >= kNopeDim)
        {
            int const pairIdx = (d0 - kNopeDim) >> 1;
            float const* cs = ptr<float const>(ctx, kPRope) + static_cast<size_t>(pos) * 64;
            float const c = __ldg(cs + pairIdx);
            float const s = __ldg(cs + 32 + pairIdx);
            st32(arena<uint8_t>(ctx, kArFusedQ) + ((tok * kHeads + head) * kLatDim + kKvLora + (d0 - kNopeDim)) * 2,
                lamport(packBf16(v0 * c - v1 * s, v0 * s + v1 * c)));
        }
        else
        {
            st32(arena<uint8_t>(ctx, kArQNope) + ((tok * kHeads + head) * kQkDim + d0) * 2, lamport(packBf16(v0, v1)));
            if (isNope && d0 >= kNopeDim)
            {
                st32(arena<uint8_t>(ctx, kArFusedQ) + ((tok * kHeads + head) * kLatDim + kKvLora + (d0 - kNopeDim)) * 2,
                    0u);
            }
        }
    }
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();

    // Immediately execute both Kb units (2*crank and 2*crank+1) using prefetched Kb weights in SMEM!
    uint8_t const* qSrc = arena<uint8_t const>(ctx, kArQNope) + head * kQkDim * 2;
    float* red = reinterpret_cast<float*>(ctx.smem + kIdxSmemCtrl + kIdxCtrlScratch);
    if (!isNope)
    {
        uint4 b[1][2];
        if (warp < 4)
        {
            pollB<1>(ctx, b, qSrc, kHeads * kQkDim, 0, 106);
        }
        waitBar(ctx, barKb, 0, 107);
        if (warp < 4)
        {
#pragma unroll
            for (int kbi = 0; kbi < 2; ++kbi)
            {
                int const sub = 2 * crank + kbi;
                uint32_t const wKb = ctx.smemBase + kSmemKbOff + static_cast<uint32_t>(kbi * kbBytes);
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                mmaChunkBf16(acc, wKb + warp * kBf16Block, b[0]);
                float* out = red + warp * 64;
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
                st32(arena<uint8_t>(ctx, kArFusedQ)
                        + ((tok * kHeads + head) * kLatDim + 64 * sub + 16 * warp + 2 * pr) * 2,
                    w);
            }
        }
    }
    else
    {
        int const tileIdx = warp & 3;
        int const chunkIdx = warp >> 2;
        uint4 b[1][2];
        pollB<1>(ctx, b, qSrc, kHeads * kQkDim, chunkIdx, 106);
        waitBar(ctx, barKb, 0, 107);
        for (int kbi = 0; kbi < 2; ++kbi)
        {
            int const sub = 2 * crank + kbi;
            uint32_t const wKb = ctx.smemBase + kSmemKbOff + static_cast<uint32_t>(kbi * kbBytes);
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            mmaChunkBf16(acc, wKb + (tileIdx * 2 + chunkIdx) * kBf16Block, b[0]);
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
            if (tid < 4 * kTok * 8)
            {
                int const ti = tid / 32;
                int const rem = tid % 32;
                int const tok = rem / 8;
                int const pr = rem % 8;
                float const* s0 = red + ti * 64;
                float const* s1 = red + (4 + ti) * 64;
                float const v0 = s0[(2 * pr) * kTok + tok] + s1[(2 * pr) * kTok + tok];
                float const v1 = s0[(2 * pr + 1) * kTok + tok] + s1[(2 * pr + 1) * kTok + tok];
                st32(arena<uint8_t>(ctx, kArFusedQ)
                        + ((tok * kHeads + head) * kLatDim + 64 * sub + 16 * ti + 2 * pr) * 2,
                    lamport(packBf16(v0, v1)));
            }
            __syncthreads();
        }
    }
    __syncthreads();
}

// ----------------------------------------------------------------------------------------------------------------
// Front-End Helper: IdxScore (TCGen05 mmaF8N128, double-buffered K tiles -> idxLogits & Level-0 hist0)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void frontIssueKTile(
    Ctx const& ctx, uint32_t a0, int slot, int globalTile, int nPagesTotal, uint32_t bar)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const pagesPerTile = 128 >> ctx.tpbShift;
    int const pageBytesK = ctx.tpb * kIdxDim;
    int const pageBytesS = ctx.tpb * 4;
    if (tid == 0)
    {
        int const p0 = globalTile * pagesPerTile;
        int const nValidPages = max(0, min(pagesPerTile, nPagesTotal - p0));
        int64_t const stride = ctx.p[kPIdxBlockStride];
        uint64_t const policy = policyEvictFirst();
        fenceProxyAsyncSmem();
        mbarExpectTx(bar, static_cast<uint32_t>(nValidPages) * static_cast<uint32_t>(pageBytesK + pageBytesS));
        for (int l = 0; l < nValidPages; ++l)
        {
            int const blk = __ldg(ptr<int const>(ctx, kPIdxBlocks) + p0 + l);
            int const y = static_cast<int>(static_cast<int64_t>(blk) * (stride / 128));
            uint8_t const* scales = ptr<uint8_t const>(ctx, kPIdxPool) + static_cast<size_t>(blk) * stride + pageBytesK;
            tmaLoad2dHint(
                a0 + (slot == 0 ? 0u : 16384u) + l * pageBytesK, ptr<void const>(ctx, kPIdxTmap), 0, y, bar, policy);
            bulkCopyHint(
                a0 + 49152u + (slot * 128 + l * ctx.tpb) * 4, scales, static_cast<uint32_t>(pageBytesS), bar, policy);
        }
    }
}

__device__ __noinline__ void frontStageIdxScore(Ctx const& ctx, uint8_t* idxArena, float* idxLogits,
    uint32_t* idxCounters, int tileStartRank, int myTileOff, int myTiles, int nPagesTotal, int L, int const* positions)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    // SMEM layout (83.5 KB <= 128 KB):
    //   [0..16384) K0 | [16384..32768) K1 | [32768..49152) Q8 | [49152..50176) Ks | [50176..50688) W | [51200..83968)
    //   sHist[4][2048]
    constexpr uint32_t kOffQ8 = 32768u;
    constexpr uint32_t kOffKs = 49152u;
    constexpr uint32_t kOffW = 50176u;
    constexpr uint32_t kOffHist = 51200u;
    uint8_t* s0 = ctx.smem;
    uint32_t const a0 = ctx.smemBase;
    float* sKs = reinterpret_cast<float*>(s0 + kOffKs);
    float* sW = reinterpret_cast<float*>(s0 + kOffW);
    uint32_t* sHist = reinterpret_cast<uint32_t*>(s0 + kOffHist);

    uint32_t const barK0 = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarK0 * 8;
    uint32_t const barK1 = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarK1 * 8;
    uint32_t const barMma0 = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarMma0 * 8;
    uint32_t const barMma1 = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarMma1 * 8;
    uint32_t const tmem
        = *reinterpret_cast<uint32_t const*>(ctx.smem + kIdxSmemCtrl + kIdxCtrlScalars + 4 * kIdxScTmem);

    for (int i = tid; i < kTok * kIdxHistBins / 4; i += kThreads)
    {
        *reinterpret_cast<uint4*>(sHist + 4 * i) = make_uint4(0u, 0u, 0u, 0u);
    }

    // Issue first 2 K-tiles immediately (overlaps with Q8/W poll!).
    frontIssueKTile(ctx, a0, 0, tileStartRank + myTileOff, nPagesTotal, barK0);
    if (myTiles > 1)
    {
        frontIssueKTile(ctx, a0, 1, tileStartRank + myTileOff + 1, nPagesTotal, barK1);
    }

    // Poll Q8 & W from idxArena.
    {
        uint8_t const* qSrc = idxArena + kIArQ8;
        uint8_t const* wSrc = idxArena + kIArW;
        uint4 qv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            qv[i] = ldPoll128(qSrc + 16 * (tid + kThreads * i));
        }
        uint32_t wv = tid < kTok * kIdxHeads ? ldPoll32(wSrc + 4 * tid) : 0u;
        uint64_t t0 = 0;
        while (true)
        {
            bool ok = (tid >= kTok * kIdxHeads || wv != kEmpty) && full4(qv[0]) && full4(qv[1]) && full4(qv[2])
                && full4(qv[3]);
            if (warpAll(ok))
            {
                break;
            }
            spinCheck(ctx, t0, 161, tid, 0);
            if (tid < kTok * kIdxHeads && wv == kEmpty)
            {
                wv = ldPoll32(wSrc + 4 * tid);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                if (!full4(qv[i]))
                {
                    qv[i] = ldPoll128(qSrc + 16 * (tid + kThreads * i));
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int const c = tid + kThreads * i;
            int const row = c >> 3;
            *reinterpret_cast<uint4*>(s0 + kOffQ8 + row * 128 + 16 * ((c & 7) ^ (row & 7))) = qv[i];
        }
        if (tid < kTok * kIdxHeads)
        {
            sW[tid] = __uint_as_float(wv);
        }
    }
    fenceProxyAsyncSmem();
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();

    auto patchAndIssue = [&](int ti)
    {
        int const sb = ti & 1;
        uint32_t const barK = sb == 0 ? barK0 : barK1;
        uint32_t const barMma = sb == 0 ? barMma0 : barMma1;
        uint32_t const kPh = static_cast<uint32_t>((ti >> 1) & 1);
        waitBar(ctx, barK, kPh, 162);

        int const gTile = tileStartRank + myTileOff + ti;
        int const keyBase = gTile * 128;
        if (tid < kTok * 8)
        {
            int const ft = tid >> 3;
            int const ch = tid & 7;
            int const fpos = __ldg(positions + ft);
            int const lk = fpos - keyBase;
            if (lk >= 0 && lk < 128)
            {
                uint4 const kv = poll4(ctx, idxArena + kIArKNew + ft * kIdxDim + 16 * ch, 163);
                *reinterpret_cast<uint4*>(s0 + (sb == 0 ? 0u : 16384u) + lk * 128 + 16 * (ch ^ (lk & 7))) = kv;
                if (ch == 0)
                {
                    sKs[sb * 128 + lk] = __uint_as_float(poll1(ctx, idxArena + kIArKNewS + 4 * ft, 164));
                }
            }
        }
        fenceProxyAsyncSmem();
        tcFenceBefore();
        __syncthreads();
        tcFenceAfter();

        if (warp == 1)
        {
            if (electOne())
            {
                uint32_t const aK = a0 + (sb == 0 ? 0u : 16384u);
                uint32_t const aQ = a0 + kOffQ8;
                uint32_t const dBase = tmem + 128u * static_cast<uint32_t>(sb);
#pragma unroll
                for (int ks = 0; ks < 4; ++ks)
                {
                    mmaF8N128(dBase, aK + 32u * ks, aQ + 32u * ks, ks > 0 ? 1u : 0u);
                }
                commit1(barMma);
            }
            __syncwarp();
        }
    };

    patchAndIssue(0);
    for (int ti = 0; ti < myTiles; ++ti)
    {
        int const sb = ti & 1;
        uint32_t const barMma = sb == 0 ? barMma0 : barMma1;
        uint32_t const mPh = static_cast<uint32_t>((ti >> 1) & 1);
        int const gTile = tileStartRank + myTileOff + ti;
        int const keyBase = gTile * 128;
        int const localBase = (myTileOff + ti) * 128;

        if (ti + 1 < myTiles)
        {
            patchAndIssue(ti + 1);
        }
        waitBar(ctx, barMma, mPh, 165);
        tcFenceAfter();

        int const q = warp & 3;
        int const rowPair = warp >> 2;
        int const lk = 32 * q + lane;
        int const globalPos = keyBase + lk;
        float const ks = sKs[sb * 128 + lk];
        uint32_t const tBase = tmem + 128u * static_cast<uint32_t>(sb) + (static_cast<uint32_t>(32 * q) << 16);

#pragma unroll
        for (int dr = 0; dr < 2; ++dr)
        {
            int const r = 2 * rowPair + dr;
            uint32_t reg[32];
            tmemLd32(tBase + 32u * static_cast<uint32_t>(r), reg);
            int const rowLimit = L - (kTok - 1 - r);
            float lgt = -INFINITY;
            if (globalPos < rowLimit)
            {
                float acc = 0.f;
#pragma unroll
                for (int h = 0; h < kIdxHeads; ++h)
                {
                    acc = fmaf(fmaxf(__uint_as_float(reg[h]), 0.f), sW[r * kIdxHeads + h], acc);
                }
                lgt = acc * ks;
                redsAdd(a0 + kOffHist + (r * kIdxHistBins + (orderKey(lgt) >> 21)) * 4, 1u);
            }
            idxLogits[static_cast<size_t>(r) * kIdxMaxShard + localBase + lk] = lgt;
        }
        tcFenceBefore();
        __syncthreads();
        if (ti + 2 < myTiles)
        {
            frontIssueKTile(ctx, a0, sb, tileStartRank + myTileOff + ti + 2, nPagesTotal, sb == 0 ? barK0 : barK1);
        }
    }

    // Flush non-zero Level-0 histogram bins to rank-local idxArena hist0 and signal kCIdxScoreDone.
    uint32_t* localHist0 = reinterpret_cast<uint32_t*>(idxArena + kIArHist0);
    for (int t = 0; t < kTok; ++t)
    {
        for (int b = tid; b < kIdxHistBins; b += kThreads)
        {
            uint32_t const cnt = sHist[t * kIdxHistBins + b];
            if (cnt != 0u)
            {
                redRelaxedAdd(localHist0 + t * kIdxHistBins + b, cnt);
            }
        }
    }
    signalCounter(idxCounters + kCIdxScoreDone);
}

// ----------------------------------------------------------------------------------------------------------------
// Front-End Helper: Clear nextIdxCopy for row t (parallel across CTAs 0..6 while CTA 7 resolves kModeCand -> 0 us!)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void clearNextIdxCopyRow(
    Ctx const& ctx, int row, int ctaIdx, int ctaCount, uint8_t* nextIdxArena, uint32_t const* clearMeta)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const nextCopy = ctx.nextIdxCopy;
    uint32_t const metaLvl = clearMeta[row * 2 + 0];
    uint32_t const metaBnd = clearMeta[row * 2 + 1];
    uint4 const eLam = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
    uint4 const eHist = make_uint4(kEmptyHist, kEmptyHist, kEmptyHist, kEmptyHist);
    uint64_t const policy = policyEvictFirst();
    int const nRanksClear = (ctx.flags & kFlagTestRank0Clear) != 0 ? 1 : kRanks;

    // 1. Clear idxTkUc[nextCopy][row] (2048 u32 = 512 uint4).
    uint8_t* tkBase
        = ptr<uint8_t>(ctx, kPTkUc) + static_cast<size_t>(nextCopy) * kTkBytes + static_cast<size_t>(row) * kTopK * 4;
    for (int i = ctaIdx * kThreads + tid; i < kTopK / 4; i += ctaCount * kThreads)
    {
        st128Hint(tkBase + 16 * i, eLam, policy);
    }

    // 2. Clear idxHistUc[nextCopy][0..metaLvl-1][row][0..nRanksClear-1] (1088 words = 272 uint4 per rank).
    constexpr int kHistVecsPerRank = kIdxHistWords / 4; // 272
    for (uint32_t lvl = 0; lvl < metaLvl; ++lvl)
    {
        uint8_t* hBase = ptr<uint8_t>(ctx, kPIdxHistUc) + static_cast<size_t>(nextCopy) * kIdxHistBytes
            + static_cast<size_t>(lvl) * kIdxHistLevelBytes + static_cast<size_t>(row) * kRanks * kIdxHistWords * 4;
        for (int i = ctaIdx * kThreads + tid; i < nRanksClear * kHistVecsPerRank; i += ctaCount * kThreads)
        {
            st128Hint(hBase + 16 * i, eHist, policy);
        }
    }

    // 3. Clear idxBndUc[nextCopy][row][0..2*metaBnd).
    int const bndVecs = static_cast<int>((2u * metaBnd + 3u) / 4u);
    if (bndVecs > 0)
    {
        uint8_t* bBase = ptr<uint8_t>(ctx, kPIdxBndUc) + static_cast<size_t>(nextCopy) * kIdxBndBytes
            + static_cast<size_t>(row) * kIdxBndCap * 8;
        for (int i = ctaIdx * kThreads + tid; i < bndVecs; i += ctaCount * kThreads)
        {
            st128Hint(bBase + 16 * i, eHist, policy);
        }
    }

    // 4. Clear nextIdxArena: hist0 slice for this row (zeroed), plus Kw/KNew/Q8/W once on row == 0.
    uint4 const z = make_uint4(0u, 0u, 0u, 0u);
    uint8_t* h0Dst = nextIdxArena + kIArHist0 + static_cast<size_t>(row) * kIdxHistBins * 4;
    for (int i = ctaIdx * kThreads + tid; i < kIdxHistBins / 4; i += ctaCount * kThreads)
    {
        st128Hint(h0Dst + 16 * i, z, policy);
    }
    if (row == 0)
    {
        for (int i = ctaIdx * kThreads + tid; i < kIArHist0 / 16; i += ctaCount * kThreads)
        {
            st128Hint(nextIdxArena + 16 * i, eLam, policy);
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Front-End Helper: 1-Level Coarse+Fine MNNVL + kModeCand Global Top-2048 (Clusters 0..3: 1 cluster of 8 CTAs/row!)
// ----------------------------------------------------------------------------------------------------------------
__device__ __noinline__ void frontStageIdxTopK(Ctx const& ctx, int row, int c, uint8_t* idxArena, uint8_t* nextIdxArena,
    float const* rowLogits, uint32_t const* idxCounters, int nScoreCtas, int tileStartRank, int nTilesRank, int L)
{
    int const tid = static_cast<int>(threadIdx.x);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    uint32_t* clearMetaAll = ptr<uint32_t>(ctx, kPIdxClearMeta);
    uint32_t* clearMetaCur = clearMetaAll + ctx.idxCopy * (kTok * 2);
    uint32_t const* clearMetaNext = clearMetaAll + ctx.nextIdxCopy * (kTok * 2);

    // Wait for all active scoring CTAs to finish writing idxLogits and localHist0.
    if (nScoreCtas > 0)
    {
        waitCounterPtr(ctx, idxCounters + kCIdxScoreDone, static_cast<uint32_t>(nScoreCtas), 171);
    }

    int const rowWin = L - (kTok - 1 - row);
    if (rowWin <= kTopK)
    {
        // Edge case L in {2049, 2050, 2051}: this row has <= 2048 keys -> write identity locally, zero MNNVL!
        int* tkUc = reinterpret_cast<int*>(ptr<uint8_t>(ctx, kPTkUc) + static_cast<size_t>(ctx.idxCopy) * kTkBytes)
            + row * kTopK;
        for (int s = c * kThreads + tid; s < kTopK; s += kIdxClusterSize * kThreads)
        {
            int const val = s < rowWin ? s : -1;
            st32(tkUc + s, lamport(static_cast<uint32_t>(val)));
        }
        clearNextIdxCopyRow(ctx, row, c, kIdxClusterSize, nextIdxArena, clearMetaNext);
        if (c == 0 && tid == 0)
        {
            clearMetaCur[row * 2 + 0] = 0u;
            clearMetaCur[row * 2 + 1] = 0u;
        }
        return;
    }

    // SMEM layout inside kIdxSmemMain (128 KB):
    //   [0      .. 8192)   sHist[2048]     (local hist for lvl 1/2 fallback OR resolveBoundary rh[2048])
    //   [8192   .. 16384)  sSlots[8][256]  (cluster push slots for lvl 1/2 fallback)
    //   [16384  .. 20480)  sCoarse[16][64] (4 KB coarse group totals)
    //   [20480  .. 22528)  sFine[16][32]   (2 KB crossing group bins)
    //   [22528  .. 23040)  sGsum[2][64]    (512 B group sums)
    //   [23040  .. 23552)  sCut[32] + sPeerCnt[8][4] + sSmall[128]
    //   [24576  .. 40960)  sBndKeys[4096]  (16 KB resolver keys)
    //   [40960  .. 57344)  sBndToks[4096]  (16 KB resolver toks)
    //   [57344  .. 65536)  sWs[2048]       (8 KB scan / staging scratch)
    uint32_t* sHist = reinterpret_cast<uint32_t*>(ctx.smem + 0);
    uint32_t* sSlots = reinterpret_cast<uint32_t*>(ctx.smem + 8192);
    uint32_t* sCoarse = reinterpret_cast<uint32_t*>(ctx.smem + 16384);
    uint32_t* sFine = reinterpret_cast<uint32_t*>(ctx.smem + 20480);
    uint32_t* sGsum = reinterpret_cast<uint32_t*>(ctx.smem + 22528);
    uint32_t* sCut = reinterpret_cast<uint32_t*>(ctx.smem + 23040);
    uint32_t* sPeerCnt = sCut + 32;   // [8][4] = 32 words
    uint32_t* sSmall = sPeerCnt + 32; // [128]
    uint32_t* sSmallCnt = sSmall + 128;
    uint32_t* sBndKeys = reinterpret_cast<uint32_t*>(ctx.smem + 24576);
    uint32_t* sBndToks = reinterpret_cast<uint32_t*>(ctx.smem + 40960);
    uint32_t* sWs = reinterpret_cast<uint32_t*>(ctx.smem + 57344);

    uint32_t const barHist0 = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarHist0 * 8;
    uint32_t const barCnt = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars + kIdxBarCnt * 8;

    int const nKeysRank = nTilesRank * 128;
    int const cKeyStart = (c * nTilesRank / kIdxClusterSize) * 128;
    int const cKeyEnd = ((c + 1) * nTilesRank / kIdxClusterSize) * 128;
    int const rankBasePos = tileStartRank * 128;

    uint32_t need = static_cast<uint32_t>(kTopK);
    uint32_t aboveAllTot = 0u;
    uint32_t lowTot = 0u;
    uint32_t prefix = 0u;
    int lvl = 0;
    int mode = kTopkModeNext;
    uint32_t binAllFinal = 0u;
    uint32_t bndRankOffFinal = 0u;
    uint32_t tiesOffFinal = 0u;
    uint32_t takeRankFinal = 0u;

    for (;; ++lvl)
    {
        int const nBits = (lvl == 2) ? 10 : 11;
        int const nBins = 1 << nBits;
        int const shift = (lvl == 0) ? 21 : (lvl == 1) ? 10 : 0;
        int const prevShift = (lvl == 1) ? 21 : 10;
        uint32_t const binMask = static_cast<uint32_t>(nBins - 1);
        int const perCta = nBins / kIdxClusterSize;      // 256 or 128
        int const groupsPerCta = perCta / kIdxGroupBins; // 8 or 4

        size_t const lvlOff
            = static_cast<size_t>(ctx.idxCopy) * kIdxHistBytes + static_cast<size_t>(lvl) * kIdxHistLevelBytes;
        uint32_t* histMc = reinterpret_cast<uint32_t*>(ptr<uint8_t>(ctx, kPIdxHistMc) + lvlOff)
            + (row * kRanks + ctx.rank) * kIdxHistWords;
        uint32_t const* histUc = reinterpret_cast<uint32_t const*>(ptr<uint8_t const>(ctx, kPIdxHistUc) + lvlOff)
            + row * kRanks * kIdxHistWords;

        // Step A: Build & Multicast this rank's 1/4 slice of Level-lvl histogram (512 or 256 bins = 16 or 8
        // groups/CTA).
        if (lvl > 0)
        {
            // Level 1/2 fallback (only when binAll > 4096): build local hist in SMEM and cluster-reduce via st.async!
            for (int b = tid; b < nBins; b += kThreads)
            {
                sHist[b] = 0u;
            }
            __syncthreads();
            for (int k = cKeyStart + tid; k < cKeyEnd; k += kThreads)
            {
                float const lgt = rowLogits[k];
                if (lgt != -INFINITY)
                {
                    uint32_t const ok = orderKey(lgt);
                    if ((ok >> prevShift) == prefix)
                    {
                        redsAdd(smemU32(sHist + ((ok >> shift) & binMask)), 1u);
                    }
                }
            }
            __syncthreads();
            int const vecs = perCta / 4;
            for (int i = tid; i < kIdxClusterSize * vecs; i += kThreads)
            {
                int const q = i / vecs;
                if (q != c)
                {
                    int const k = i - q * vecs;
                    pushClusterV4(peerAddr(smemU32(sSlots + c * perCta + 4 * k), q),
                        *reinterpret_cast<uint4 const*>(sHist + q * perCta + 4 * k), peerAddr(barHist0 + 8 * lvl, q));
                }
            }
            mbarWaitSpin(barHist0 + 8 * lvl, 0);
            __syncthreads();
        }

        // Pack u16x2 fine words (capped at kHistSat = 0xfffe) + 32-bin coarse group totals and multicast via st128!
        uint32_t const* localHist0 = reinterpret_cast<uint32_t const*>(idxArena + kIArHist0) + row * kIdxHistBins;
        for (int bOff = 0; bOff < perCta; bOff += kThreads)
        {
            uint32_t myBinVal;
            if (lvl == 0)
            {
                myBinVal = localHist0[c * perCta + bOff + tid];
            }
            else
            {
                myBinVal = sHist[c * perCta + bOff + tid];
#pragma unroll
                for (int q = 0; q < kIdxClusterSize; ++q)
                {
                    myBinVal += (q != c) ? sSlots[q * perCta + bOff + tid] : 0u;
                }
            }
            uint32_t const groupTot = __reduce_add_sync(0xffffffffu, myBinVal);
            if (lane == 0)
            {
                sWs[(bOff >> 5) + warp] = groupTot;
            }
            uint32_t const w0
                = min(myBinVal, kHistSat) | (min(__shfl_down_sync(0xffffffffu, myBinVal, 1), kHistSat) << 16);
            uint32_t const w1 = __shfl_down_sync(0xffffffffu, w0, 2);
            uint32_t const w2 = __shfl_down_sync(0xffffffffu, w0, 4);
            uint32_t const w3 = __shfl_down_sync(0xffffffffu, w0, 6);
            if ((lane & 7) == 0)
            {
                st128(histMc + (c * perCta + bOff + tid) / 2, make_uint4(w0, w1, w2, w3));
            }
        }
        __syncthreads();
        if (tid < groupsPerCta / 4)
        {
            st128(histMc + kIdxFineWords + c * groupsPerCta + 4 * tid, *reinterpret_cast<uint4 const*>(sWs + 4 * tid));
        }

        // Step B: Coarse (4 KB) + Fine (1 KB) MNNVL Poll & Exact Global Cut (IDENTICAL on all 8 CTAs of row!).
        int const nGroups = nBins / kIdxGroupBins; // 64 or 32
        int const vecsPerRank = nGroups / 4;       // 16 or 8
        if (tid < 2 * kIdxMaxGroups)
        {
            sGsum[tid] = 0u;
        }
        if (tid == 0)
        {
            sCut[0] = kEmptyHist;
            sCut[4] = kEmptyHist;
        }
        __syncthreads();

        // 1. Poll 4 KB coarse group totals across all 16 ranks (1 single ldPoll128 per thread!).
        if (tid < kRanks * vecsPerRank)
        {
            int const r = tid / vecsPerRank;
            int const vi = tid % vecsPerRank;
            uint4 const v
                = poll4Hist(ctx, histUc + static_cast<size_t>(r) * kIdxHistWords + kIdxFineWords + 4 * vi, 174 + lvl);
            uint32_t const vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                atomicAdd(&sGsum[4 * vi + j], vv[j]);
                if (r < ctx.rank)
                {
                    atomicAdd(&sGsum[kIdxMaxGroups + 4 * vi + j], vv[j]);
                }
            }
            *reinterpret_cast<uint4*>(sCoarse + r * kIdxMaxGroups + 4 * vi) = v;
        }
        __syncthreads();

        // 2. Warp 0 finds crossing group g in 5 shuffles (~30 ns).
        if (tid < 32)
        {
            uint32_t gAll[2], gLow[2];
#pragma unroll
            for (int k = 0; k < 2; ++k)
            {
                int const g = 2 * tid + k;
                gAll[k] = g < nGroups ? sGsum[g] : 0u;
                gLow[k] = g < nGroups ? sGsum[kIdxMaxGroups + g] : 0u;
            }
            uint32_t xa = gAll[0] + gAll[1];
            uint32_t xl = gLow[0] + gLow[1];
#pragma unroll
            for (int o = 1; o < 32; o <<= 1)
            {
                uint32_t const ya = __shfl_down_sync(0xffffffffu, xa, o);
                uint32_t const yl = __shfl_down_sync(0xffffffffu, xl, o);
                if (tid + o < 32)
                {
                    xa += ya;
                    xl += yl;
                }
            }
            uint32_t const aAbove1 = xa - gAll[0] - gAll[1];
            uint32_t const lAbove1 = xl - gLow[0] - gLow[1];
            int which = -1;
            if (aAbove1 < need && need <= aAbove1 + gAll[1])
            {
                which = 1;
            }
            else if (aAbove1 + gAll[1] < need && need <= aAbove1 + gAll[1] + gAll[0])
            {
                which = 0;
            }
            if (which >= 0)
            {
                sCut[4] = static_cast<uint32_t>(2 * tid + which);
                sCut[5] = which == 1 ? aAbove1 : (aAbove1 + gAll[1]);
                sCut[6] = which == 1 ? lAbove1 : (lAbove1 + gLow[1]);
            }
        }
        __syncthreads();

        int const g = static_cast<int>(sCut[4]);
        uint32_t const needInGroup = need - sCut[5];

        // 3. Poll ONLY the 32 bins of crossing group g across all 16 ranks (1 KB = 64 threads x 1 ldPoll128!).
        if (tid < kRanks * 4)
        {
            int const r = tid >> 2;
            int const vi = tid & 3;
            uint4 const v = poll4Hist(
                ctx, histUc + static_cast<size_t>(r) * kIdxHistWords + (kIdxGroupBins / 2) * g + 4 * vi, 178 + lvl);
            uint32_t const ww[4] = {v.x, v.y, v.z, v.w};
            uint32_t x[8];
            uint32_t others = 0u;
            bool sat = false;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                x[2 * j] = ww[j] & 0xffffu;
                x[2 * j + 1] = ww[j] >> 16;
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                if (x[j] == kHistSat)
                {
                    sat = true;
                }
                else
                {
                    others += x[j];
                }
            }
            others += __shfl_xor_sync(0xffffffffu, others, 1);
            others += __shfl_xor_sync(0xffffffffu, others, 2);
            if (sat)
            {
                uint32_t const total = sCoarse[r * kIdxMaxGroups + g];
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    if (x[j] == kHistSat)
                    {
                        x[j] = total - others;
                    }
                }
            }
            *reinterpret_cast<uint4*>(sFine + r * kIdxGroupBins + 8 * vi) = make_uint4(x[0], x[1], x[2], x[3]);
            *reinterpret_cast<uint4*>(sFine + r * kIdxGroupBins + 8 * vi + 4) = make_uint4(x[4], x[5], x[6], x[7]);
        }
        __syncthreads();

        // 4. Warp 0 finds exact cutBin within group g & rank offsets.
        if (tid < 32)
        {
            uint32_t all = 0u, low = 0u;
#pragma unroll
            for (int r = 0; r < kRanks; ++r)
            {
                uint32_t const xv = sFine[r * kIdxGroupBins + tid];
                all += xv;
                low += r < ctx.rank ? xv : 0u;
            }
            uint32_t xa = all, xl = low;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1)
            {
                uint32_t const ya = __shfl_down_sync(0xffffffffu, xa, o);
                uint32_t const yl = __shfl_down_sync(0xffffffffu, xl, o);
                if (tid + o < 32)
                {
                    xa += ya;
                    xl += yl;
                }
            }
            uint32_t const aAbove = xa - all;
            uint32_t const lAbove = xl - low;
            uint32_t const holder = __ballot_sync(0xffffffffu, aAbove < needInGroup && needInGroup <= aAbove + all);
            if (holder != 0u)
            {
                int const w = __ffs(static_cast<int>(holder)) - 1;
                uint32_t const aAboveW = __shfl_sync(0xffffffffu, aAbove, w);
                uint32_t const allW = __shfl_sync(0xffffffffu, all, w);
                uint32_t const lAboveW = __shfl_sync(0xffffffffu, lAbove, w);
                uint32_t const bc = tid < kRanks ? sFine[tid * kIdxGroupBins + w] : 0u;
                uint32_t const bndRankOff = __reduce_add_sync(0xffffffffu, tid < ctx.rank ? bc : 0u);
                // Ties broken by higher rank / larger token (matching glmDsaHistTopK).
                uint32_t const needBin = needInGroup - aAboveW;
                uint32_t xs = bc;
#pragma unroll
                for (int o = 1; o < kRanks; o <<= 1)
                {
                    uint32_t const y = __shfl_down_sync(0xffffffffu, xs, o);
                    if (tid + o < kRanks)
                    {
                        xs += y;
                    }
                }
                int const higher = static_cast<int>(xs - bc);
                int const take = max(0, min(static_cast<int>(needBin) - higher, static_cast<int>(bc)));
                uint32_t const tiesOff
                    = __reduce_add_sync(0xffffffffu, tid < ctx.rank ? static_cast<uint32_t>(take) : 0u);
                uint32_t const takeRank = __shfl_sync(0xffffffffu, static_cast<uint32_t>(take), ctx.rank);
                if (tid == 0)
                {
                    sCut[0] = static_cast<uint32_t>(g * kIdxGroupBins + w);
                    sCut[1] = sCut[5] + aAboveW;
                    sCut[2] = allW;
                    sCut[3] = sCut[6] + lAboveW;
                    sCut[7] = bndRankOff;
                    sCut[8] = tiesOff;
                    sCut[9] = takeRank;
                }
            }
        }
        __syncthreads();

        uint32_t const bin = sCut[0];
        uint32_t const aboveAll = sCut[1];
        uint32_t const binAll = sCut[2];
        uint32_t const aboveLow = sCut[3];
        aboveAllTot += aboveAll;
        lowTot += aboveLow;
        need -= aboveAll;
        prefix = (lvl == 0) ? bin : ((prefix << nBits) | bin);

        if (need == binAll)
        {
            mode = kTopkModeFull;
        }
        else if (lvl == kIdxHistLevels - 1)
        {
            mode = kTopkModeTies;
        }
        else if (binAll <= static_cast<uint32_t>(kIdxBndCap))
        {
            mode = kTopkModeCand;
        }
        else
        {
            mode = kTopkModeNext;
        }
        if (mode != kTopkModeNext)
        {
            binAllFinal = binAll;
            bndRankOffFinal = sCut[7];
            tiesOffFinal = sCut[8];
            takeRankFinal = sCut[9];
            break;
        }
    }

    int const levelsUsed = lvl + 1;
    int const shiftF = (lvl == 0) ? 21 : (lvl == 1) ? 10 : 0;

    // Step C: Count this CTA's (nAboveCta, nBndCta) over its 1/8 slice of rowLogits, exchange across 8-CTA cluster!
    uint32_t cntAbove = 0u, cntBnd = 0u;
    for (int k = cKeyStart + tid; k < cKeyEnd; k += kThreads)
    {
        float const lgt = rowLogits[k];
        if (lgt != -INFINITY)
        {
            uint32_t const s = orderKey(lgt) >> shiftF;
            cntAbove += (s > prefix) ? 1u : 0u;
            cntBnd += (s == prefix) ? 1u : 0u;
        }
    }
    uint32_t nAboveCta = 0u, nBndCta = 0u;
    (void) blockExclusiveScan256(cntAbove, sWs, nAboveCta);
    (void) blockExclusiveScan256(cntBnd, sWs + 32, nBndCta);

    if (tid == 0)
    {
        uint4 const cPack = make_uint4(nAboveCta, nBndCta, 0u, 0u);
        *reinterpret_cast<uint4*>(sPeerCnt + 4 * c) = cPack;
#pragma unroll
        for (int q = 0; q < kIdxClusterSize; ++q)
        {
            if (q != c)
            {
                pushClusterV4(peerAddr(smemU32(sPeerCnt + 4 * c), q), cPack, peerAddr(barCnt, q));
            }
        }
    }
    mbarWaitSpin(barCnt, 0);
    __syncthreads();

    // Compute exact deterministic (outOffC, bndOffC, takeC) across the 8 CTAs in this cluster.
    uint32_t outOffC = lowTot;
    uint32_t bndOffC = bndRankOffFinal;
    uint32_t takeC = 0u;
    if (mode == kTopkModeTies)
    {
        outOffC += tiesOffFinal;
        // Higher CTAs (larger tokens) take ties first!
        int higherBnd = 0;
        for (int q = c + 1; q < kIdxClusterSize; ++q)
        {
            higherBnd += static_cast<int>(sPeerCnt[4 * q + 1]);
        }
        takeC = static_cast<uint32_t>(
            max(0, min(static_cast<int>(takeRankFinal) - higherBnd, static_cast<int>(nBndCta))));
        for (int q = 0; q < c; ++q)
        {
            int hq = 0;
            for (int r = q + 1; r < kIdxClusterSize; ++r)
            {
                hq += static_cast<int>(sPeerCnt[4 * r + 1]);
            }
            uint32_t const tq = static_cast<uint32_t>(
                max(0, min(static_cast<int>(takeRankFinal) - hq, static_cast<int>(sPeerCnt[4 * q + 1]))));
            outOffC += sPeerCnt[4 * q + 0] + tq;
        }
    }
    else
    {
        if (mode == kTopkModeFull)
        {
            outOffC += bndRankOffFinal;
            takeC = nBndCta;
        }
        for (int q = 0; q < c; ++q)
        {
            outOffC += sPeerCnt[4 * q + 0] + (mode == kTopkModeFull ? sPeerCnt[4 * q + 1] : 0u);
            bndOffC += sPeerCnt[4 * q + 1];
        }
    }

    // Emit selected tokens to idxTkMc & (if kModeCand) boundary (key, pos) pairs to idxBndMc!
    uint32_t* tkMc
        = reinterpret_cast<uint32_t*>(ptr<uint8_t>(ctx, kPTkMc) + static_cast<size_t>(ctx.idxCopy) * kTkBytes)
        + row * kTopK;
    uint32_t* bndMc
        = reinterpret_cast<uint32_t*>(ptr<uint8_t>(ctx, kPIdxBndMc) + static_cast<size_t>(ctx.idxCopy) * kIdxBndBytes)
        + row * kIdxBndCap * 2;
    uint32_t const firstTaken = (mode == kTopkModeTies) ? (nBndCta - takeC) : 0u;
    uint32_t outCursor = outOffC;
    uint32_t bndCursor = bndOffC;
    uint32_t bndSeen = 0u;

    for (int baseK = cKeyStart; baseK < cKeyEnd; baseK += kThreads)
    {
        int const k = baseK + tid;
        bool isSel = false;
        bool isBnd = false;
        uint32_t ok = 0u;
        if (k < cKeyEnd)
        {
            float const lgt = rowLogits[k];
            if (lgt != -INFINITY)
            {
                ok = orderKey(lgt);
                uint32_t const s = ok >> shiftF;
                isSel = (s > prefix) || (mode == kTopkModeFull && s == prefix);
                isBnd = (s == prefix);
            }
        }
        uint32_t bndChunkTot = 0u;
        uint32_t const bndLocal = blockExclusiveScan256(isBnd ? 1u : 0u, sWs + 32, bndChunkTot);
        if (mode == kTopkModeTies && isBnd && (bndSeen + bndLocal >= firstTaken))
        {
            isSel = true;
        }
        uint32_t selChunkTot = 0u;
        uint32_t const selLocal = blockExclusiveScan256(isSel ? 1u : 0u, sWs, selChunkTot);
        uint32_t const gPos = static_cast<uint32_t>(rankBasePos + k);
        if (isSel)
        {
            st32(tkMc + outCursor + selLocal, lamport(gPos));
        }
        else if (mode == kTopkModeCand && isBnd)
        {
            st64(bndMc + 2 * (bndCursor + bndLocal), make_uint2(ok, gPos));
        }
        outCursor += selChunkTot;
        bndCursor += bndChunkTot;
        bndSeen += bndChunkTot;
    }

    // Step D: Resolver CTA (c == 7 in kModeCand) resolves boundary pairs locally while CTAs 0..6 clear nextIdxCopy!
    bool const isResolver = (mode == kTopkModeCand) && (c == kIdxClusterSize - 1);
    if (isResolver)
    {
        int const nb = static_cast<int>(binAllFinal);
        uint32_t const* bndUc = reinterpret_cast<uint32_t const*>(ptr<uint8_t const>(ctx, kPIdxBndUc)
                                    + static_cast<size_t>(ctx.idxCopy) * kIdxBndBytes)
            + row * kIdxBndCap * 2;
        int const vecs = (2 * nb + 3) / 4;
        for (int vi = tid; vi < vecs; vi += kThreads)
        {
            int const valid = min(4, 2 * nb - 4 * vi);
            uint4 const v = poll4HistValid(ctx, bndUc + 4 * vi, valid, 182);
            sBndKeys[2 * vi] = v.x;
            sBndToks[2 * vi] = v.y;
            if (valid > 2)
            {
                sBndKeys[2 * vi + 1] = v.z;
                sBndToks[2 * vi + 1] = v.w;
            }
        }
        __syncthreads();
        int* tkUc = reinterpret_cast<int*>(ptr<uint8_t>(ctx, kPTkUc) + static_cast<size_t>(ctx.idxCopy) * kTkBytes)
            + row * kTopK;
        resolveBoundary256(sBndKeys, sBndToks, nb, need, shiftF, prefix << shiftF, sHist, sWs, sCut, sSmall, sSmallCnt,
            tkUc, static_cast<int>(aboveAllTot));
    }
    else
    {
        int const clearCtas = (mode == kTopkModeCand) ? (kIdxClusterSize - 1) : kIdxClusterSize;
        clearNextIdxCopyRow(ctx, row, c, clearCtas, nextIdxArena, clearMetaNext);
    }

    if (c == 0 && tid == 0)
    {
        clearMetaCur[row * 2 + 0] = static_cast<uint32_t>(levelsUsed);
        clearMetaCur[row * 2 + 1] = (mode == kTopkModeCand) ? binAllFinal : 0u;
    }
    (void) nKeysRank;
}

// ----------------------------------------------------------------------------------------------------------------
// Leader Front-End Kernel (176 CTAs = 44 clusters of 4, 256 threads, 132 KiB SMEM)
// ----------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads, 1)
    ml4LeaderFrontKernel(int64_t const* __restrict__ params, __nv_bfloat16 const* __restrict__ x,
        int const* __restrict__ blockTable, int const* __restrict__ kvLen, int const* __restrict__ positions)
{
    extern __shared__ __align__(1024) uint8_t smemRaw[];
    int const tid = static_cast<int>(threadIdx.x);
    int const cta = static_cast<int>(blockIdx.x);
    int const cid = cta / kIdxClusterSize;   // 0..23
    int const crank = cta % kIdxClusterSize; // 0..7

    int64_t* sParams = reinterpret_cast<int64_t*>(smemRaw + kIdxSmemCtrl + kIdxCtrlParams);
    if (tid < kPNumParams)
    {
        sParams[tid] = __ldg(params + tid);
    }
    __syncthreads();

    Ctx& ctx = *reinterpret_cast<Ctx*>(smemRaw + kIdxSmemCtrl + kIdxCtrlCtx);
    if (tid == 0)
    {
        ctx.p = sParams;
        ctx.smem = smemRaw;
        ctx.smemBase = smemU32(smemRaw);
        ctx.parity = static_cast<int>(sParams[kPCopy]);
        ctx.nextCopy = static_cast<int>(sParams[kPNextCopy]);
        ctx.prevCopy = static_cast<int>(sParams[kPPrevCopy]);
        ctx.idxCopy = static_cast<int>(sParams[kPIdxCopy]);
        ctx.nextIdxCopy = static_cast<int>(sParams[kPNextIdxCopy]);
        ctx.numCopies = static_cast<int>(sParams[kPNumCopies]);
        ctx.rank = static_cast<int>(sParams[kPRank]);
        ctx.cta = cta;
        ctx.nDie0 = static_cast<int>(sParams[kPNDie0]);
        ctx.flags = sParams[kPFlags];
        ctx.counters = reinterpret_cast<uint32_t*>(sParams[kPCounters]) + ctx.parity * kNumCounters;
        ctx.err = reinterpret_cast<int*>(sParams[kPErr]);
        ctx.timeoutNs = static_cast<uint64_t>(sParams[kPTimeoutNs]);
        ctx.arena = reinterpret_cast<uint8_t*>(sParams[kPArena]) + static_cast<size_t>(ctx.parity) * kArenaBytes;
        ctx.tpb = static_cast<int>(sParams[kPTokensPerBlock]);
        ctx.tpbShift = ctx.tpb == 64 ? 6 : 5;
        ctx.rmsEps = __uint_as_float(static_cast<uint32_t>(sParams[kPRmsEps]));
        ctx.idxEps = __uint_as_float(static_cast<uint32_t>(sParams[kPIdxEps]));
    }
    __syncthreads();

    // Initialize ALL cluster & TMA barriers and allocate 256 TMEM cols before cluster sync!
    uint32_t const barBase = ctx.smemBase + kIdxSmemCtrl + kIdxCtrlBars;
    if (tid < 32)
    {
        if (tid < kIdxNumBars)
        {
            mbarInit(barBase + 8 * tid, 1);
        }
        __syncwarp();
        if (tid == 0)
        {
            // Expect-tx for cluster st.async barriers initialized ONCE before clusterWait (cluster=(4,1,1) -> 3 peers)!
            if (cid < 32)
            {
                // IdxQ: 3 remote splits push 128 floats = 512 B each -> 1,536 B!
                mbarExpectTx(barBase + 8 * kIdxBarRed, 3u * 128u * 4u);
            }
            else if (cid < 36)
            {
                // Wq: 3 remote splits push 32 rows x 4 tok = 512 B each -> 1,536 B!
                mbarExpectTx(barBase + 8 * kIdxBarRed, 3u * 32u * kTok * 4u);
            }
            if (cid < 4)
            {
                // IdxTopK: lvl 1 (2048/4=512 u32), lvl 2 (1024/4=256 u32) x 3 peers, and count exchange (3 x 16 B)!
                mbarExpectTx(barBase + 8 * (kIdxBarHist0 + 1), 3u * 512u * 4u);
                mbarExpectTx(barBase + 8 * (kIdxBarHist0 + 2), 3u * 256u * 4u);
                mbarExpectTx(barBase + 8 * kIdxBarCnt, 3u * 16u);
            }
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    else if (tid >= 32 && tid < 64)
    {
        tmemAlloc(smemU32(smemRaw + kIdxSmemCtrl + kIdxCtrlScalars + 4 * kIdxScTmem), kIdxTmemCols);
    }

    // Zero nextIdxCopy's idxCounters (64 words = 256 B).
    uint32_t* idxCountersCur = ptr<uint32_t>(ctx, kPIdxCounters) + ctx.idxCopy * kNumCounters;
    uint32_t* idxCountersNext = ptr<uint32_t>(ctx, kPIdxCounters) + ctx.nextIdxCopy * kNumCounters;
    if (cta == 0 && tid < kNumCounters)
    {
        idxCountersNext[tid] = 0u;
    }

    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
    clusterArriveRelaxed();
    clusterWait();

    int const L = __ldg(kvLen);
    bool const early = (ctx.flags & kFlagEarlyX) != 0;
    bool const kvOwner = (ctx.flags & kFlagKvOwner) != 0;
    bool const isNope = (ctx.flags & kFlagNope) != 0;
    uint8_t* idxArenaCur = ptr<uint8_t>(ctx, kPIdxArena) + static_cast<size_t>(ctx.idxCopy) * kIdxArenaBytes;
    uint8_t* idxArenaNext = ptr<uint8_t>(ctx, kPIdxArena) + static_cast<size_t>(ctx.nextIdxCopy) * kIdxArenaBytes;
    float* idxLogitsCur = ptr<float>(ctx, kPIdxLogits) + static_cast<size_t>(ctx.idxCopy) * kTok * kIdxMaxShard;

    // Cross-layer PDL overlap: if early, wait prevCopy main-arena clear and poll xSsq BEFORE griddepcontrol.wait!
    float* sRcp = reinterpret_cast<float*>(smemRaw + kIdxSmemCtrl + kIdxCtrlScalars + 4 * kIdxScRcp);
    if (!early)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    else
    {
        // Only CTAs that read kArX need waitPrevCopyClear & pollXRcp (Clusters 0..41 & 43 when L>2048, 32..41 when
        // L<=2048).
        bool const needsX = (cid >= 32 && cid < 42) || ((cid < 32 || cid == 43) && L > kTopK);
        if (needsX)
        {
            waitPrevCopyClear(ctx);
            pollXRcp(ctx, sRcp, 101);
        }
    }

    // --- Phase 1: T=0 Parallel Front-End across 44 clusters of 4 (176 CTAs, 100% co-resident on 212 SMs!) ---
    if (cid < 32)
    {
        if (L > kTopK)
        {
            frontStageIdxQ(ctx, cid, crank, idxArenaCur, x, sRcp, early, positions);
        }
        else if (cid < kTok)
        {
            // Keep nextIdxCopy completely clean across L>2048 -> L<=2048 -> L>2048 transitions using idle clusters!
            uint32_t* clearMetaAll = ptr<uint32_t>(ctx, kPIdxClearMeta);
            uint32_t* clearMetaCur = clearMetaAll + ctx.idxCopy * (kTok * 2);
            uint32_t const* clearMetaNext = clearMetaAll + ctx.nextIdxCopy * (kTok * 2);
            clearNextIdxCopyRow(ctx, cid, crank, kIdxClusterSize, idxArenaNext, clearMetaNext);
            if (crank == 0 && tid == 0)
            {
                clearMetaCur[cid * 2 + 0] = 0u;
                clearMetaCur[cid * 2 + 1] = 0u;
            }
        }
    }
    else if (cid < 36)
    {
        frontStageWqAndKb(ctx, cid - 32, crank, x, sRcp, early, positions);
    }
    else if (cid < 42)
    {
        int const u = (cid - 36) * kIdxClusterSize + crank; // 0..23
        int const kvTiles = kvOwner ? (isNope ? kWkvATilesNope : kWkvATilesRope) : 0;
        int const kvCtas = kvTiles / 2;                     // 18 RoPE, 16 NoPE, 0 Sharer
        if (u < kvCtas)
        {
            frontStageWkvATwoTiles(ctx, 2 * u, 2, arena<uint8_t>(ctx, kArKvRaw), kLatDim, 2 * u, x, sRcp, early);
        }
        else if (u >= 18 && u < 23)
        {
            int const kwIdx = u - 18; // 0..4 -> 2 tiles each = 10 tiles (160 cols)
            frontStageWkvATwoTiles(
                ctx, kvTiles + 2 * kwIdx, 2, idxArenaCur + kIArKw, kIdxKwCols, 2 * kwIdx, x, sRcp, early);
        }
        else if (u == 23)
        {
            frontStageIdxK(ctx, idxArenaCur, positions);
        }
    }
    else if (cid == 42)
    {
        if (crank == 0 && kvOwner)
        {
            stageKvAppend(ctx, blockTable, positions);
        }
    }
    else if (cid == 43 && L > kTopK)
    {
        // Dedicated 4-CTA (1 CTA/token, 256 threads = 24 BF16 elems/thread!) 2-pass LayerNorm stats (~0.7 us!).
        int const tok = crank;
        float const rcpT = early ? sRcp[tok] : 1.f;
        uint8_t const* rowBase = (early ? arena<uint8_t const>(ctx, kArX) : reinterpret_cast<uint8_t const*>(x))
            + static_cast<size_t>(tok) * kHidden * 2;
        uint4 rv[3];
        if (early)
        {
            pollStrided<3>(ctx, rv, rowBase + 16 * tid, 16 * kThreads, 151);
        }
        else
        {
#pragma unroll
            for (int r = 0; r < 3; ++r)
            {
                rv[r] = ldcg128(rowBase + (r * kThreads + tid) * 16);
            }
        }
        float vals[24];
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < 3; ++r)
        {
            uint32_t const ww[4] = {rv[r].x, rv[r].y, rv[r].z, rv[r].w};
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float v0 = bf16Lo(ww[j]);
                float v1 = bf16Hi(ww[j]);
                if (early)
                {
                    v0 = bf16Round(v0 * rcpT);
                    v1 = bf16Round(v1 * rcpT);
                }
                vals[r * 8 + 2 * j] = v0;
                vals[r * 8 + 2 * j + 1] = v1;
                sum += v0 + v1;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            sum += __shfl_xor_sync(0xffffffffu, sum, off);
        }
        float* sRed = reinterpret_cast<float*>(smemRaw + kIdxSmemCtrl + kIdxCtrlScratch);
        if ((tid & 31) == 0)
        {
            sRed[tid >> 5] = sum;
        }
        __syncthreads();
        float totSum = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            totSum += sRed[w];
        }
        float const mean = totSum / static_cast<float>(kHidden);
        float var = 0.f;
#pragma unroll
        for (int i = 0; i < 24; ++i)
        {
            float const d = vals[i] - mean;
            var = fmaf(d, d, var);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            var += __shfl_xor_sync(0xffffffffu, var, off);
        }
        if ((tid & 31) == 0)
        {
            sRed[tid >> 5] = var;
        }
        __syncthreads();
        if (tid == 0)
        {
            float totVar = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                totVar += sRed[w];
            }
            float const rstd = rsqrtf(totVar / static_cast<float>(kHidden) + ctx.idxEps);
            st32(idxArenaCur + kIArLnMean + 4 * tok, lamport(__float_as_uint(mean)));
            st32(idxArenaCur + kIArLnRstd + 4 * tok, lamport(__float_as_uint(rstd)));
        }
    }

    if (early)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    // --- Phase 2 & 3 (L > 2048): IdxScore -> Early Exit for 160 CTAs -> IdxTopK on 16 CTAs (Clusters 0..3) ---
    if (L > kTopK)
    {
        int const nPagesTotal = (L + ctx.tpb - 1) >> ctx.tpbShift;
        int const nTilesTotal = (L + 127) >> 7;
        int const tileStartRank = (nTilesTotal * ctx.rank) / kRanks;
        int const tileEndRank = (nTilesTotal * (ctx.rank + 1)) / kRanks;
        int const nTilesRank = tileEndRank - tileStartRank;
        int const nScoreCtas = min(nTilesRank, kIdxFrontCtas);

        // Prioritize early-finishing Clusters 32..43 (scoreOrd 0..47) before Clusters 0..31 (scoreOrd 48..175)!
        int const scoreOrd
            = (cid >= 32) ? ((cid - 32) * kIdxClusterSize + crank) : (48 + cid * kIdxClusterSize + crank);
        if (scoreOrd < nScoreCtas)
        {
            int const myStart = (scoreOrd * nTilesRank) / nScoreCtas;
            int const myEnd = ((scoreOrd + 1) * nTilesRank) / nScoreCtas;
            frontStageIdxScore(ctx, idxArenaCur, idxLogitsCur, idxCountersCur, tileStartRank, myStart, myEnd - myStart,
                nPagesTotal, L, positions);
        }

        // Release TMEM before launching dependents / running IdxTopK (IdxTopK uses zero TMEM!).
        tcFenceBefore();
        __syncthreads();
        if (tid < 32)
        {
            tcFenceAfter();
            uint32_t const tmem
                = *reinterpret_cast<uint32_t const*>(smemRaw + kIdxSmemCtrl + kIdxCtrlScalars + 4 * kIdxScTmem);
            tmemDealloc(tmem, kIdxTmemCols);
        }

        // Trigger PDL for ml4LayerKernel right after IdxScore! Clusters 4..23 exit immediately (160 SMs freed!).
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

        if (cid < kTok)
        {
            frontStageIdxTopK(ctx, cid, crank, idxArenaCur, idxArenaNext,
                idxLogitsCur + static_cast<size_t>(cid) * kIdxMaxShard, idxCountersCur, nScoreCtas, tileStartRank,
                nTilesRank, L);
        }
    }
    else
    {
        tcFenceBefore();
        __syncthreads();
        if (tid < 32)
        {
            tcFenceAfter();
            uint32_t const tmem
                = *reinterpret_cast<uint32_t const*>(smemRaw + kIdxSmemCtrl + kIdxCtrlScalars + 4 * kIdxScTmem);
            tmemDealloc(tmem, kIdxTmemCols);
        }
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}
