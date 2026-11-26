// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Mistral Large 4 (SM107 / VR200) TP16 B1 M=4 persistent megakernel.
// Two lean specialized kernels chained via PDL on Leader layers (43 layers) and single-kernel on Plain layers (28):
//   1. ml4LeaderFrontKernel (176 CTAs, cluster=(4,1,1), 132 KiB SMEM): TCGen05 MXFP8 IdxQ (fused 2-pass LN + cluster
//      per-token K-reduce + SSMax + RoPE + E4M3) & Wq+Kb (cluster K-reduce), double-buffered WkvA/IdxKw +
//      IdxK/KvAppend, TCGen05 FP8 IdxScore, and 1-level Coarse+Fine MNNVL + kModeCand Global Top-2048 (with exact
//      multi-level fallback).
//   2. ml4LayerKernel (212 CTAs, 223 KiB SMEM, zero indexer code): TCGen05 MXFP8 Wq (64 CTAs on Plain), exact 656B
//      fp8_ds_mla sparse MLA, 37-unit Clear, 0-spill 16-rank MNNVL AR+RMSNorm+NVFP4 H1/H2, 24-CTA dedicated TCGen05
//      MXFP8 Shared+Owner parallel with 188-CTA zero-stall overlapped TMA+TCGen05 NVFP4 Routed MoE (inline fused
//      SwiGLU+NVFP4 requant eliminating cross-CTA kArP1 gather storms), Die-Aware uGPU placement, and DSpark capture.

#include "mistral_large4/mistral_large_4_nvfp4__sm_107__tp4/csrc/ml4Megakernel.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#if !defined(ML4_SMEM_MAIN)
#define ML4_SMEM_MAIN 200704
#endif
#if !defined(ML4_SETUP_MODE)
#define ML4_SETUP_MODE 1
#endif
#if !defined(ML4_LEAN)
#define ML4_LEAN 1
#endif

ML4_MEGA_NS_BEGIN

namespace kernels::mistral_large4::megakernel
{

namespace ml4
{

// ----------------------------------------------------------------------------------------------------------------
// Model and hardware geometry (Mistral Large 4 NVFP4, VR200 SM107, TP16, B1 M=4)
// ----------------------------------------------------------------------------------------------------------------
constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kCtas = 212;
constexpr int kDefaultNDie0 = 106;
constexpr int kIdxFrontCtas = 176; // 44 clusters of 4 CTAs (100% co-resident on 212-SM VR200)
constexpr int kIdxClusterSize = 4;
constexpr int kTok = 4;
constexpr int kHidden = 6144;
constexpr int kRanks = 16;
constexpr int kHeads = 4;
constexpr int kQkDim = 128;
constexpr int kNopeDim = 64;
constexpr int kRopeDim = 64;
constexpr int kKvLora = 512;
constexpr int kLatDim = kKvLora + kRopeDim; // 576
constexpr int kVDim = 128;
constexpr int kTopK = 2048;

// fp8_ds_mla latent KV row: 512 e4m3 + 4 fp32 scales + 64 bf16 rope = 656 B.
constexpr int kKvRowBytes = 656;
constexpr int kKvScaleOff = 512;
constexpr int kKvRopeOff = 528;
constexpr int kKvSmemRowBytes = 672;
constexpr int kQRowBytes = kLatDim * 2; // 1152 B
constexpr int kQSmemRowBytes = 1168;
constexpr int kAttnBlocks = 32;
constexpr int kAttnKeys = kTopK / kAttnBlocks; // 64
constexpr int kPRowBytes = kAttnKeys * 2 + 16; // 144 B
constexpr float kSoftmaxScale = 0.08838834764831845f;
constexpr float kLog2E = 1.44269504f;
constexpr float kSoftmaxScaleLog2 = kSoftmaxScale * kLog2E;

// Weight chunk / TCGen05 MXFP8 piece geometry.
constexpr int kBf16Block = 2048;                           // 16 rows x 64 BF16
constexpr int kFp8Block = 2048;                            // 16 rows x 128 E4M3 (Oproj)
constexpr int kMxf8WBytes = 128 * 128;                     // 16,384 B (SWIZZLE_128B E4M3)
constexpr int kMxf8SBytes = 512;                           // R128c4 K32 UE8M0 atom
constexpr int kMxf8PieceBytes = kMxf8WBytes + kMxf8SBytes; // 16,896 B (global weight piece)
constexpr int kMxf8SlotStride
    = (kMxf8PieceBytes + 1023) & ~1023;                    // 17,408 B (1024B-aligned SMEM ring slot for SWIZZLE_128B)
constexpr int kHiddenChunksBf16 = kHidden / 64;            // 96
constexpr int kHiddenBlocksFp8 = kHidden / 128;            // 48
constexpr int kWkvATilesRope = kLatDim / 16;               // 36
constexpr int kWkvATilesNope = kKvLora / 16;               // 32
constexpr int kKbUnitsPerHead = kKvLora / 64;              // 8
constexpr int kVupTilesPerHead = kVDim / 16;               // 8
constexpr int kVupChunks = kKvLora / 64;                   // 8
constexpr int kOprojBlocksFp8 = (kHeads * kVDim) / 128;    // 4
constexpr int kColsPerHopUnit = kHidden / 4;               // 1536
constexpr int kWqPlainSplits = 8; // 4 heads x 8 K-splits = 32 CTAs (6 K-tiles/CTA, 1-wave pollStrided<7>)
constexpr int kWqPlainKPerCta = kHiddenBlocksFp8 / kWqPlainSplits; // 6

// MoE & Shared Expert geometry (TP16: 128 experts, top-4 softmax renorm, I_moe = 384, I_shared = 384).
constexpr int kExperts = 128;
constexpr int kMoeTopK = 4;
constexpr int kMaxSlots = kTok * kMoeTopK; // 16
constexpr int kMoeInter = 384;
constexpr int kSharedInter = 384;
constexpr int kSharedCtas = 24; // 12 Die-0 + 12 Die-1 dedicated Shared + Owner CTAs (11.8 us hidden behind MoE!)
constexpr int kSharedHalfCtas = 12;
constexpr int kRoutedCtas = kCtas - kSharedCtas;             // 188 (94 Die-0 + 94 Die-1)
constexpr int kRoutedDie0 = kDefaultNDie0 - kSharedHalfCtas; // 94
constexpr int kSharedGuSplits = 4;                           // 6 x 4 = 24 CTAs (12 K-tiles/CTA)
constexpr int kSharedDnBlocks = kSharedInter / 128;          // 3
constexpr int kOwnerCols = kHidden / kSharedCtas;            // 256 cols per Owner CTA (2 x 128)

// Routed MoE NVFP4 pieces: identical 55,296 B piece size for both FC1 and FC2.
constexpr int kFc1TilesPerExpert = (2 * kMoeInter) / 128;              // 6
constexpr int kFc1KPieces = kHidden / 768;                             // 8
constexpr int kFc2TilesPerPiece = 2;                                   // 256 rows per piece
constexpr int kFc2KBlocks = kMoeInter / 128;                           // 3
constexpr int kFc2PiecesPerSlot = kHidden / (128 * kFc2TilesPerPiece); // 24
constexpr int kPieceW = 6 * 128 * 64;                                  // 49,152 B
constexpr int kPieceS = 12 * 512;                                      // 6,144 B
constexpr int kPieceBytes = kPieceW + kPieceS;                         // 55,296 B
constexpr int kRingSlots = 3;
constexpr int kHfp4Atoms = kHidden / 128;                              // 48
constexpr int kHfp4SmemBytes = kHfp4Atoms * 512;                       // 24,576 B
constexpr int kHsfBytes = kTok * (kHidden / 16);                       // 1,536 B
constexpr int kFc2BSmemBytes = kFc2KBlocks * 512;                      // 1,536 B
constexpr int kFc2SfbSmemBytes = kFc2KBlocks * 128;                    // 384 B (utccp1Rows8 layout)
constexpr int kFc2SfbArenaBytes = 512;                                 // 128B-aligned per slot
constexpr int kMoeRegion2Bytes = kHfp4SmemBytes + kHsfBytes + kFc2BSmemBytes + kFc2SfbArenaBytes; // 28,160 B
constexpr int kMoeSmemBytes = kRingSlots * kPieceBytes + kMoeRegion2Bytes;                        // 194,048 B

// DSA Indexer constants (32 heads x 128 dim, key-sharded 1/16 across 16 ranks).
constexpr int kIdxHeads = 32;
constexpr int kIdxDim = 128;
constexpr int kIdxKwCols = kIdxDim + kIdxHeads; // 160
constexpr int kIdxHistBins = 2048;
constexpr int kIdxGroupBins = 32;
constexpr int kIdxMaxGroups = kIdxHistBins / kIdxGroupBins;  // 64
constexpr int kIdxFineWords = kIdxHistBins / 2;              // 1024 u32 words (u16x2 bins)
constexpr int kIdxHistWords = kIdxFineWords + kIdxMaxGroups; // 1088 u32 words per (lvl, tok, rank)
constexpr int kIdxHistLevels = 3;
constexpr int kIdxBndCap = 4096;                             // max boundary pairs for 1-level kModeCand completion
constexpr int kIdxSmallRank = 64;
constexpr float kIdxWeightScale = 0.015625f;                 // 128^-0.5 * 32^-0.5
constexpr int kIdxMaxKeys = 2 << 20;                         // 2,097,152 tokens supported
constexpr int kIdxMaxShard = kIdxMaxKeys / kRanks;           // 131,072 keys per rank

constexpr uint32_t kEmpty = 0x80000000u;                     // FP32 -0.0 Lamport empty sentinel
constexpr uint32_t kEmptyHist = 0xffffffffu;                 // histogram & boundary pair empty sentinel
constexpr uint32_t kHistSat = 0xfffeu;                       // u16 saturation cap (never forms 0xffffffff)
constexpr int64_t kMagic = 0x4d4c345f4d4b3035LL;             // "ML4_MK05" ABI magic
constexpr int kMaxRoles = 16;

// MNNVL multicast buffers per copy.
constexpr int kHopBytes = kTok * kRanks * kHidden * 2;                // 786,432 B
constexpr int kTkBytes = kTok * kTopK * 4;                            // 32,768 B
constexpr int kIdxHistLevelBytes = kTok * kRanks * kIdxHistWords * 4; // 278,528 B
constexpr int kIdxHistBytes = kIdxHistLevels * kIdxHistLevelBytes;    // 835,584 B
constexpr int kIdxBndBytes = kTok * kIdxBndCap * 2 * 4;               // 131,072 B

// Separate rank-local Indexer Arena (per idxCopy) & Logits scratch (per idxCopy, never cleared!).
constexpr int kIArKw = 0;                                // BF16 [4][160] = 1,280 B
constexpr int kIArKNew = kIArKw + 1280;                  // E4M3 [4][128] = 512 B
constexpr int kIArKNewS = kIArKNew + 512;                // FP32 [4] + LnMean[4] @+16 + LnRstd[4] @+32 (padded 128 B)
constexpr int kIArLnMean = kIArKNewS + 16;               // FP32 [4] = 16 B (produced by Cluster 43)
constexpr int kIArLnRstd = kIArKNewS + 32;               // FP32 [4] = 16 B (produced by Cluster 43)
constexpr int kIArQ8 = kIArKNewS + 128;                  // E4M3 [4][32][128] = 16,384 B
constexpr int kIArW = kIArQ8 + 16384;                    // FP32 [4][32] = 512 B
constexpr int kIArHist0 = kIArW + 512;                   // u32 [4][2048] = 32,768 B (zeroed)
constexpr int kIdxArenaBytes = kIArHist0 + 32768;        // 51,584 B (~50 KiB!)
constexpr int kIdxLogitsBytes = kTok * kIdxMaxShard * 4; // 2,097,152 B (2 MiB/copy, never cleared)

static_assert((kIArHist0 & 127) == 0 && (kIdxArenaBytes & 127) == 0, "128B idx arena alignment");

// ----------------------------------------------------------------------------------------------------------------
// Main 4-Copy Arena layout (lean — zero indexer buffers, 6.6x smaller kArP1 => only 37 Clear units!)
// ----------------------------------------------------------------------------------------------------------------
constexpr int kArQNope = 0;                                                   // BF16 [4][4][128] = 4,096
constexpr int kArFusedQ = kArQNope + 4096;                                    // BF16 [4][4][576] = 18,432
constexpr int kArKvRaw = kArFusedQ + 18432;                                   // BF16 [4][576] = 4,608
constexpr int kArKv = kArKvRaw + 4608;                                        // u8 [4][656] padded = 2,688
constexpr int kArO = kArKv + 2688;                                            // BF16 [4][4][512] = 16,384
constexpr int kArAttnOut = kArO + 16384;                                      // BF16 [4][512] = 4,096
constexpr int kArWqPart = kArAttnOut + 4096;                                  // FP32 [4][16][4][128] = 131,072
constexpr int kArH = kArWqPart + 131072;                                      // BF16 [4][6144] = 49,152
constexpr int kArHp = kArH + 49152;                                           // BF16 [4][6144] = 49,152
constexpr int kArHfp4 = kArHp + 49152;                                        // E2M1 [4][3072] = 12,288
constexpr int kArHsf = kArHfp4 + 12288;                                       // E4M3 [4][384] = 1,536
constexpr int kArSsq1 = kArHsf + 1536;                                        // FP32 [4][4] padded = 128
constexpr int kArSsq2 = kArSsq1 + 128;                                        // FP32 [4][4] padded = 128
constexpr int kArLogit = kArSsq2 + 128;                                       // FP32 [4][4][128] = 8,192
constexpr int kArRoute = kArLogit + 8192;                                     // int32 [128] = 512
constexpr int kArSh = kArRoute + 512;                                         // FP32 [6][4][4][128] = 49,152
constexpr int kArP1 = kArSh + 49152;                                          // FP32 [16][6][4][128] = 196,608
constexpr int kArX = kArP1 + kMaxSlots * kFc1TilesPerExpert * kTok * 128 * 4; // BF16 [4][6144] = 49,152
constexpr int kArXSsq = kArX + 49152;                                         // FP32 [4][4] padded = 128
constexpr int kArenaClearBytes = kArXSsq + 128;                               // 597,504 B (37 x 16 KiB units!)
// Non-cleared tail (guarded by kCFc2Ready0..15 counters zeroed in prologue):
constexpr int kArFc2B = kArenaClearBytes;                              // u8 [16][1536] = 24,576
constexpr int kArFc2Sfb = kArFc2B + kMaxSlots * kFc2BSmemBytes;        // u8 [16][512] = 8,192
constexpr int kArenaBytes = kArFc2Sfb + kMaxSlots * kFc2SfbArenaBytes; // 630,272 B (~615 KiB)

static_assert(kArQNope == 0 && kArWqPart == 50304 && kArH == 181376 && kArP1 == 351616 && kArenaClearBytes == 597504
        && kArenaBytes == 630272,
    "main arena offset contract");
static_assert(kArHfp4 + 12288 == kArHsf, "kArHsf must immediately follow kArHfp4");
static_assert((kArWqPart & 127) == 0 && (kArP1 & 127) == 0 && (kArX & 127) == 0 && (kArFc2B & 127) == 0
        && (kArenaBytes & 127) == 0,
    "128B arena alignment");

// Clear units (16 KiB each): covers main Arena [0..kArenaClearBytes) in 37 units.
constexpr int kClearUnitBytes = 16384;
constexpr int kClearUnits = (kArenaClearBytes + kClearUnitBytes - 1) / kClearUnitBytes; // 37
constexpr int kCountedClear = kClearUnits;
static_assert(kClearUnits == 37 && kClearUnits <= kCtas, "37 clear units");

// ----------------------------------------------------------------------------------------------------------------
// Shared memory budgets
// ----------------------------------------------------------------------------------------------------------------
constexpr int kSmemMain = ML4_SMEM_MAIN; // 200,704 B
constexpr int kSmemRed = kSmemMain;
constexpr int kSmemRedBytes = 16384;
constexpr int kSmemRes = kSmemRed + kSmemRedBytes;
constexpr int kSmemResBytes = 3072;
constexpr int kSmemSmall = kSmemRes + kSmemResBytes;
constexpr int kSmemSmallBytes = 3072;
constexpr int kSmemBytes = kSmemSmall + kSmemSmallBytes; // 223,232 B
constexpr int kAttnOff = 131072;
constexpr int kAttnSmemBytes = kAttnKeys * kKvSmemRowBytes + 8 * kQSmemRowBytes + 4 * 8 * kPRowBytes + 1024 + 1024;

// ml4LeaderFrontKernel SMEM budget: 128 KiB main + 4 KiB control = 135,168 B (132 KiB).
constexpr int kIdxSmemMain = 131072;
constexpr int kIdxSmemCtrl = kIdxSmemMain;
constexpr int kIdxSmemCtrlBytes = 4096;
constexpr int kIdxSmemBytes = kIdxSmemCtrl + kIdxSmemCtrlBytes; // 135,168 B

static_assert(kSmemMain == 200704 && kSmemBytes == 223232 && kSmemBytes <= 232448, "main SMEM contract");
static_assert(kAttnOff + kAttnSmemBytes <= kSmemMain, "stageAttn fits above 128 KiB");
static_assert(kMoeSmemBytes <= kSmemMain, "stageMoe fits inside kSmemMain");
static_assert(kHiddenChunksBf16 * kBf16Block <= kSmemMain, "WkvA tile fits inside kSmemMain");
static_assert(kIdxSmemBytes == 135168 && kIdxSmemBytes <= 232448, "idx front SMEM contract");

// Small control area offsets in ml4LayerKernel (from kSmemSmall).
constexpr int kSmallBars = 0;                           // 32 barriers x 8 B = 256 B
constexpr int kBarAttn0 = kMaxRoles;                    // 16..23: 8 gather barriers
constexpr int kBarMoeRing0 = kMaxRoles + 8;             // 24..26: 3 MoE/MXFP8 ring TMA barriers
constexpr int kBarMoeFree0 = kBarMoeRing0 + kRingSlots; // 27..29: 3 ring MMA-free barriers
constexpr int kBarMoeMma = kBarMoeFree0 + kRingSlots;   // 30: MMA final commit barrier
constexpr int kBarAux = kBarMoeMma + 1;                 // 31: activation ready / ring slot 3 barrier
constexpr int kNumBars = kBarAux + 1;                   // 32
static_assert(kNumBars == 32 && kNumBars * 8 <= 256, "barrier slots fit in 256 B");

constexpr int kSmallRoles = 256;    // 3 * kMaxRoles int32 = 192 B
constexpr int kSmallScalars = 448;  // 80 floats = 320 B (slot 72 holds TMEM base addr)
constexpr int kScalarTmem = 72;
constexpr int kSmallMoe = 768;      // 128 int32 = 512 B
constexpr int kSmallParams = 1280;  // 80 int64 = 640 B
constexpr int kSmallCtx = 1920;     // 128 B
constexpr int kSmallWeights = 2048; // 16 * 3 * 8 = 384 B
static_assert(kSmallWeights + kMaxRoles * 3 * 8 <= kSmemSmallBytes, "small area fits in 3072 B");

// MoE work list offsets (int32 words at kSmallMoe).
constexpr int kMoeE = 0;
constexpr int kMoeE0 = 1;
constexpr int kMoeE1 = 2;
constexpr int kMoeRcp = 12;        // [4]
constexpr int kMoeSlotOfTk = 16;   // [16]
constexpr int kMoeSlotExpert = 32; // [16]
constexpr int kMoeWeight = 48;     // [16]
constexpr int kMoeSlotMask = 64;   // [16] uint8_t (4 words)
static_assert(kMoeSlotMask + 4 <= 128, "MoE work list fits in kSmallMoe");

// Route block in arena (int32 words at kArRoute).
constexpr int kRouteE = 0;
constexpr int kRouteSlotExpert = 4; // [16]
constexpr int kRouteSlotOfTk = 20;  // [16]
constexpr int kRouteWeight = 36;    // [16]

// ----------------------------------------------------------------------------------------------------------------
// Stage codes (role entry = (stage << 16) | unit)
// ----------------------------------------------------------------------------------------------------------------
enum Stage : int
{
    kStNone = 0,
    kStWq = 1,          // Plain: 64-CTA TCGen05 MXFP8 wq + head K-reduce + SSMax + Q-RoPE
    kStWkvA = 2,        // Plain: BF16 wkv_a (36 tiles)
    kStKb = 3,          // Plain: BF16 q_nope @ k_b
    kStKvAppend = 4,    // Plain: kv_a RMSNorm + fp8_ds_mla 656B quant + RoPE
    kStAttn = 8,        // Sparse MLA decode on fp8_ds_mla 656B (4 tok x 32 blk x 64 keys)
    kStMerge = 9,       // Merge 32 partial blocks per (tok, head) -> kArO
    kStVup = 10,        // BF16 lat_out @ v_b_t -> kArAttnOut
    kStOproj = 11,      // FP8_PB_WO attn_out @ wo^T -> MNNVL H1 multicast
    kStClear = 12,      // Clear next copy's lean main Arena (37 x 16 KiB units)
    kStH1 = 13,         // 16-rank AR#1 + residual + ffn_norm RMSNorm + NVFP4 quant + early h'
    kStRouterGemv = 14, // BF16 router GEMV (128 experts across 4 K quarters)
    kStShared = 15,     // 24-CTA dedicated TCGen05 MXFP8 SharedGu + SwiGLU + SharedDn -> sOut
    kStMoe = 16,        // 188-CTA Die-Aware overlapped TMA+TCGen05 NVFP4 Routed MoE (fused SwiGLU+NVFP4)
    kStOwner = 18,      // 24-CTA Top-4 combine + shared add (256 cols/CTA) -> MNNVL H2 multicast
    kStH2 = 19,         // 16-rank AR#2 + residual + next RMSNorm + early x' + DSpark capture
};

// ----------------------------------------------------------------------------------------------------------------
// Counters (u32, next copy zeroed in kernel prologue; 64 words = 256 B per copy)
// ----------------------------------------------------------------------------------------------------------------
enum Counter : int
{
    kCClear = 0,        // completed main Arena clear units (37)
    kCMoeDone = 1,      // CTAs finished with stageMoe / stageOwner (212 for HoldExit)
    kCIdxScoreDone = 2, // ml4LeaderFrontKernel: active scoring CTAs finished
    kCFc2Ready0 = 16,   // 16..31: per-slot FC1->SwiGLU+NVFP4 ready count (target = 6 tiles)
    kNumCounters = 64,  // 256-byte aligned per copy
};

// ----------------------------------------------------------------------------------------------------------------
// Parameter block (int64 words, filled by host in _megakernel.py; must fit <= 80 words in kSmallParams)
// ----------------------------------------------------------------------------------------------------------------
enum Param : int
{
    kPMagic = 0,
    kPKvBase,
    kPStrideRows,
    kPLayerRows,
    kPMaxBlocks,
    kPTokensPerBlock,
    kPRope,
    kPWq,
    kPSsmaxS,
    kPSsmaxB,
    kPWkvA,
    kPKb,
    kPVb,
    kPWO,
    kPWOScale,
    kPWR,
    kPWGu,
    kPWD,
    kPGammaKv,
    kPGammaPost,
    kPGammaNext,
    kPW13,
    kPW2,
    kPAlpha1,
    kPAlpha2,
    kPArena,
    kPPartO,
    kPPartStats,
    kPFc2out,
    kPH1Mc,
    kPH1Uc,
    kPH2Mc,
    kPH2Uc,
    kPTkMc,
    kPTkUc,
    kPIdxHistMc,
    kPIdxHistUc,
    kPIdxBndMc,
    kPIdxBndUc,
    kPHopBytes,
    kPRank,
    kPCounters,
    kPErr,
    kPMaxRoles,
    kPTimeoutNs,
    kPFlags,
    kPCopy,
    kPNextCopy,
    kPPrevCopy,
    kPNumCopies,
    kPIdxCopy,
    kPNextIdxCopy,
    kPIdxArena,
    kPIdxLogits,
    kPIdxCounters,
    kPIdxClearMeta,
    kPSmidMap,
    kPNDie0,
    kPRmsEps,
    kPIdxEps,
    kPCaptureOut,
    kPCaptureStride,
    kPIdxQNormW,
    kPIdxQNormB,
    kPIdxWq,
    kPIdxKnW,
    kPIdxKnB,
    kPIdxSsmaxS,
    kPIdxSsmaxB,
    kPIdxPool,
    kPIdxBlockStride,
    kPIdxTmap,
    kPIdxBlocks,
    kPTopkOut,
    kPUgpuState,
    kPNumParams,
};

static_assert(kSmallParams + kPNumParams * 8 <= kSmallCtx, "Param block fits in kSmallParams");

// ----------------------------------------------------------------------------------------------------------------
// Runtime and compile-time flags (kPFlags)
// ----------------------------------------------------------------------------------------------------------------
enum Flag : int64_t
{
    kFlagHoldExit = 1ll << 0,        // hold CTA exit until all 212 CTAs finish MoE/Owner
    kFlagNextChained = 1ll << 1,     // next layer reads x' from arena (H2 skips ssq exchange & xOut)
    kFlagEvictFirst = 1ll << 2,      // L2 evict-first hint on one-shot weight bulk copies
    kFlagMoeL2 = 1ll << 4,           // prefetch MoE pieces beyond ring into L2
    kFlagEarlyX = 1ll << 5,          // previous layer is megakernel: front-end polls x' before PDL wait
    kFlagLeader = 1ll << 6,          // full-indexer leader layer (43 MoE layers)
    kFlagNope = 1ll << 7,            // NoPE layer (i % 4 == 3: 18 layers; k_b K=128, no RoPE)
    kFlagKvOwner = 1ll << 8,         // owns latent KV (62 MoE layers; false on 9 Layer-35 sharers)
    kFlagDieAware = 1ll << 9,        // use SMID->logical_cta Die-Aware uGPU mapping and die-split MoE
    kFlagHopSelfClear = 1ll << 10,   // H2 self-clears H1 and previous H2 multicast slots
    kFlagXPrime = 1ll << 11,         // H2 publishes unscaled x' = res' * gamma_next + xSsq
    kFlagHPrime = 1ll << 12,         // H1 publishes unscaled h' = res * gamma_post before ssq exchange
    kFlagHop1Lines = 1ll << 13,      // hop 1 packed as [rank][oproj_unit][tok][32 cols]
    kFlagSkipFront = 1ll << 16,      // ml4LeaderFrontKernel already produced kArFusedQ/kArKv/idxTk
    kFlagTestRank0Clear = 1ll << 17, // 1-GPU fake-peer microbench only: clear only rank 0 MNNVL slots
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

// ----------------------------------------------------------------------------------------------------------------
// PTX helpers (SM100 / SM107a)
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smemU32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t getSmid()
{
    uint32_t id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

__device__ __forceinline__ uint64_t globalTimer()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint32_t peerAddr(uint32_t addr, int rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
    return r;
}

__device__ __forceinline__ void clusterArriveRelaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterArriveRelease()
{
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterWait()
{
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void pushClusterV4(uint32_t remote, uint4 v, uint32_t remoteBar)
{
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.u32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(remote),
        "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(remoteBar)
        : "memory");
}

__device__ __forceinline__ void mbarInval(uint32_t bar)
{
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mbarInit(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarReinit(uint32_t bar, uint32_t count)
{
    mbarInval(bar);
    mbarInit(bar, count);
}

__device__ __forceinline__ void mbarExpectTx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarArrive(uint32_t bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ bool mbarTryWait(uint32_t bar, uint32_t parity)
{
    uint32_t ok;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(ok)
        : "r"(bar), "r"(parity)
        : "memory");
    return ok != 0;
}

__device__ __forceinline__ bool mbarTestWait(uint32_t bar, uint32_t parity)
{
    uint32_t ok;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(ok)
        : "r"(bar), "r"(parity)
        : "memory");
    return ok != 0;
}

__device__ __forceinline__ void mbarWaitSpin(uint32_t bar, uint32_t parity)
{
    while (!mbarTestWait(bar, parity))
    {
    }
}

__device__ __forceinline__ void bulkCopy(uint32_t dst, void const* src, uint32_t bytes, uint32_t bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}

__device__ __forceinline__ void bulkCopyHint(
    uint32_t dst, void const* src, uint32_t bytes, uint32_t bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
            "r"(dst),
        "l"(src), "r"(bytes), "r"(bar), "l"(policy)
        : "memory");
}

__device__ __forceinline__ void prefetchBulkL2Hint(void const* src, uint32_t bytes, uint64_t policy)
{
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(src), "r"(bytes), "l"(policy)
                 : "memory");
}

__device__ __forceinline__ uint64_t policyEvictFirst()
{
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}

__device__ __forceinline__ void fenceProxyAsyncSmem()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ uint32_t ldAcquire(uint32_t const* p)
{
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void stReleaseU32(uint32_t* p, uint32_t v)
{
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void redReleaseAdd(uint32_t* p, uint32_t v)
{
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void redRelaxedAdd(uint32_t* p, uint32_t v)
{
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint4 ldcg128(void const* p)
{
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ldPoll128(void const* p)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ldPoll64(void const* p)
{
    uint2 v;
    asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ldPoll32(void const* p)
{
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void st128(void* p, uint4 v)
{
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ void st128Hint(void* p, uint4 v, uint64_t policy)
{
    asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
                 "r"(v.w), "l"(policy)
                 : "memory");
}

__device__ __forceinline__ void st64(void* p, uint2 v)
{
    asm volatile("st.global.v2.b32 [%0], {%1, %2};" ::"l"(p), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ void st32(void* p, uint32_t v)
{
    asm volatile("st.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint4 lds128(uint32_t addr)
{
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

__device__ __forceinline__ void redsAdd(uint32_t addr, uint32_t v)
{
    asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}

__device__ __forceinline__ void mmaBf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mmaF16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mmaE4m3(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrixX4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrixX4Trans(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrixX2(uint32_t addr, uint32_t& r0, uint32_t& r1)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ uint32_t e4m3x2ToHalf2(uint32_t pair)
{
    __half2_raw const h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair & 0xffffu), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

__device__ __forceinline__ uint32_t bf16x2ToHalf2(uint32_t bf)
{
    __half2 const h2 = __floats2half2_rn(__uint_as_float(bf << 16), __uint_as_float(bf & 0xffff0000u));
    return *reinterpret_cast<uint32_t const*>(&h2);
}

__device__ __forceinline__ uint8_t toE4m3(float x)
{
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint32_t toE4m3x2(float lo, float hi)
{
    return static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float rcpApproxFtz(float x)
{
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t toE2m1x2(float lo, float hi)
{
    uint32_t v;
    asm("{\n"
        ".reg .b8 b;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        "mov.b32 %0, {b, b, b, b};\n"
        "}\n"
        : "=r"(v)
        : "f"(lo), "f"(hi));
    return v & 0xffu;
}

__device__ __forceinline__ float bf16Round(float x)
{
    uint16_t h;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return __uint_as_float(static_cast<uint32_t>(h) << 16);
}

__device__ __forceinline__ float bf16RoundFtz(float x)
{
    float const r = bf16Round(x);
    uint32_t const bits = __float_as_uint(r);
    uint32_t const flushed = (bits & 0x7f800000u) == 0u ? (bits & 0x80000000u) : bits;
    return __uint_as_float(flushed);
}

__device__ __forceinline__ uint32_t packBf16(float lo, float hi)
{
    __nv_bfloat162 const v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t const*>(&v);
}

__device__ __forceinline__ float bf16Lo(uint32_t v)
{
    return __uint_as_float(v << 16);
}

__device__ __forceinline__ float bf16Hi(uint32_t v)
{
    return __uint_as_float(v & 0xffff0000u);
}

__device__ __forceinline__ float pow2Ceil(float x)
{
    uint32_t const bits = __float_as_uint(x);
    uint32_t const exp = (bits & 0x7f800000u) + ((bits & 0x007fffffu) != 0u ? 0x00800000u : 0u);
    return __uint_as_float(exp);
}

__device__ __forceinline__ float pow2Inv(float s)
{
    uint32_t const expBits = __float_as_uint(s) & 0x7f800000u;
    return __uint_as_float(0x7f000000u - expBits);
}

__device__ __forceinline__ float ue8m0Scale(float amax)
{
    float const ratio = __fmul_rn(fmaxf(amax, 1.0e-12f), 1.0f / 448.0f);
    return pow2Ceil(ratio);
}

__device__ __forceinline__ uint8_t ue8m0ExpByte(float scale)
{
    uint32_t const e = (__float_as_uint(scale) >> 23) & 0xffu;
    return static_cast<uint8_t>(e > 254u ? 254u : e);
}

__device__ __forceinline__ float ssmaxScaleBf16(int pos, float s, float b)
{
    float const lp = bf16Round(logf(static_cast<float>(pos) + 1.0f));
    float const sb = bf16Round(s);
    float const bb = bf16Round(b);
    return bf16Round(bf16Round(lp * sb) + bb);
}

__device__ __forceinline__ uint32_t orderKey(float f)
{
    uint32_t const u = __float_as_uint(f);
    uint32_t const k = u ^ ((u >> 31) != 0u ? 0xffffffffu : 0x80000000u);
    return k < 0xffffffffu ? k : 0xfffffffeu;
}

// ----------------------------------------------------------------------------------------------------------------
// TCGen05 NVFP4, MXFP8 (UE8M0 block scale), and FP8 MMA helpers (SM100 / SM107a)
// ----------------------------------------------------------------------------------------------------------------
constexpr uint32_t kIdescNvf4 = (1u << 7) | (1u << 10) | ((8u >> 3) << 17) | ((128u >> 4) << 24); // M=128, N=8
// MXFP8: E4M3 x E4M3, UE8M0 scale (bit 23 = 1), M=128 (bit 27 = 1), N=8 (bit 17 = 1), K=32 (bit 31 = 0).
constexpr uint32_t kIdescMxf8Base = (1u << 17) | (1u << 23) | (1u << 27);
constexpr uint64_t kDescSw64 = (1ull << 16) | (static_cast<uint64_t>(512 >> 4) << 32) | (1ull << 46) | (4ull << 61);
constexpr uint64_t kDescSf = (static_cast<uint64_t>(128 >> 4) << 32) | (1ull << 46);
constexpr uint64_t kDescSw128 = (1ull << 16) | (static_cast<uint64_t>(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
constexpr uint32_t kIdescF8N128 = (1u << 4) | ((128u >> 3) << 17) | ((128u >> 4) << 24);
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kIdxTmemCols = 256;
constexpr uint32_t kTmemD = 0;      // [0, 64): FC1 8 cols, FC2 2 tiles x 8 = 16 cols, MXFP8 8 cols
constexpr uint32_t kTmemSfa = 64;   // [64, 208): 3 ring slots x 48 cols (or MXFP8 4 ring slots x 4 cols)
constexpr uint32_t kTmemSfb = 256;  // [256, 448): FC1 token scales (96 K steps x 2 cols) / MXFP8 SFB (12 x 4)
constexpr uint32_t kTmemSfb2 = 448; // [448, 460): FC2 token scales (6 K steps x 2 cols)

__device__ __forceinline__ void tcFenceBefore()
{
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcFenceAfter()
{
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmemAlloc(uint32_t dst, uint32_t cols)
{
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst), "r"(cols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmemDealloc(uint32_t taddr, uint32_t cols)
{
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(cols) : "memory");
}

__device__ __forceinline__ bool electOne()
{
    uint32_t e;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xffffffff;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(e));
    return e != 0;
}

__device__ __forceinline__ void utccp1(uint32_t taddr, uint32_t src)
{
    uint64_t const sdesc = kDescSf | ((src & 0x3ffffu) >> 4);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}

__device__ __forceinline__ void utccp1Rows8(uint32_t taddr, uint32_t src)
{
    uint64_t const sdesc = (1ull << 46) | ((src & 0x3ffffu) >> 4);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}

__device__ __forceinline__ void mmaNvf4(uint32_t d, uint32_t a, uint32_t b, uint32_t sfa, uint32_t sfb, uint32_t acc)
{
    uint64_t const da = kDescSw64 | ((a & 0x3ffffu) >> 4);
    uint64_t const db = kDescSw64 | ((b & 0x3ffffu) >> 4);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\n"
        "}\n" ::"r"(d),
        "l"(da), "l"(db), "r"(kIdescNvf4), "r"(sfa), "r"(sfb), "r"(acc)
        : "memory");
}

__device__ __forceinline__ void mmaMxf8(
    uint32_t d, uint32_t a, uint32_t b, uint32_t sfa, uint32_t sfb, uint32_t kb, uint32_t acc)
{
    uint64_t const da = kDescSw128 | ((a & 0x3ffffu) >> 4);
    uint64_t const db = kDescSw128 | ((b & 0x3ffffu) >> 4);
    uint32_t const idesc = kIdescMxf8Base | (kb << 4) | (kb << 29);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], p;\n"
        "}\n" ::"r"(d),
        "l"(da), "l"(db), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc)
        : "memory");
}

__device__ __forceinline__ void mmaF8N128(uint32_t d, uint32_t a, uint32_t b, uint32_t acc)
{
    uint64_t const da = kDescSw128 | ((a & 0x3ffffu) >> 4);
    uint64_t const db = kDescSw128 | ((b & 0x3ffffu) >> 4);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
        "}\n" ::"r"(d),
        "l"(da), "l"(db), "r"(kIdescF8N128), "r"(acc)
        : "memory");
}

__device__ __forceinline__ void commit1(uint32_t bar)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void tmaLoad2dHint(
    uint32_t dst, void const* tmap, int x, int y, uint32_t bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
        "l"(tmap), "r"(x), "r"(y), "r"(bar), "l"(policy)
        : "memory");
}

__device__ __forceinline__ void tmemLd8(uint32_t taddr, float (&v)[8])
{
    uint32_t r[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(taddr)
                 : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        v[i] = __uint_as_float(r[i]);
    }
}

__device__ __forceinline__ void tmemLd2x8(uint32_t taddr, float (&v)[2][8])
{
    uint32_t r[16];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
        "%15}, [%16];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(taddr)
        : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            v[i][j] = __uint_as_float(r[8 * i + j]);
        }
    }
}

__device__ __forceinline__ void tmemLd32(uint32_t taddr, uint32_t (&r)[32])
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr)
        : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// ----------------------------------------------------------------------------------------------------------------
// Kernel Context (fits inside kSmallCtx = 128 B)
// ----------------------------------------------------------------------------------------------------------------
struct Ctx
{
    int64_t const* p;
    uint8_t* smem;
    uint32_t smemBase;
    uint32_t* counters;
    int* err;
    uint64_t timeoutNs;
    uint8_t* arena;
    int parity;
    int nextCopy;
    int prevCopy;
    int idxCopy;
    int nextIdxCopy;
    int rank;
    int cta;
    int nDie0;
    int64_t flags;
    int numCopies;
    int tpb;
    int tpbShift;
    float rmsEps;
    float idxEps;
};

static_assert(sizeof(Ctx) <= 128, "Ctx exceeds its 128-byte shared-memory slot");

template <typename T>
__device__ __forceinline__ T* ptr(Ctx const& ctx, int idx)
{
    return reinterpret_cast<T*>(ctx.p[idx]);
}

template <typename T>
__device__ __forceinline__ T* arena(Ctx const& ctx, int off)
{
    return reinterpret_cast<T*>(ctx.arena + off);
}

__device__ __noinline__ void fail(Ctx const& ctx, int code, int a, int b)
{
    int volatile* err = ctx.err;
    if (err != nullptr)
    {
        err[1] = ctx.cta;
        err[2] = static_cast<int>(threadIdx.x);
        err[3] = a;
        err[4] = b;
        err[0] = code;
        __threadfence_system();
    }
    __trap();
}

__device__ __forceinline__ void spinCheck(Ctx const& ctx, uint64_t& t0, int code, int a, int b)
{
#if ML4_LEAN
    (void) ctx;
    (void) t0;
    (void) code;
    (void) a;
    (void) b;
    return;
#else
    if (t0 == 0)
    {
        t0 = globalTimer();
    }
    else if ((++t0 & 0xffull) == 0)
    {
        if (globalTimer() - t0 > ctx.timeoutNs)
        {
            fail(ctx, code, a, b);
        }
    }
#endif
}

__device__ __forceinline__ bool warpAll(bool ok)
{
    return __all_sync(0xffffffffu, ok);
}

// ----------------------------------------------------------------------------------------------------------------
// Data-as-Flag primitives
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t lamport(uint32_t w)
{
    return w == kEmpty ? 0u : w;
}

__device__ __forceinline__ bool full4(uint4 v)
{
    return (v.x != kEmpty) & (v.y != kEmpty) & (v.z != kEmpty) & (v.w != kEmpty);
}

__device__ __forceinline__ bool full2(uint2 v)
{
    return (v.x != kEmpty) & (v.y != kEmpty);
}

__device__ __forceinline__ uint4 poll4(Ctx const& ctx, void const* p, int code)
{
    uint4 v = ldPoll128(p);
    uint64_t t0 = 0;
    while (!full4(v))
    {
        spinCheck(ctx, t0, code, 0, 0);
        v = ldPoll128(p);
    }
    return v;
}

__device__ __forceinline__ uint32_t poll1(Ctx const& ctx, void const* p, int code)
{
    uint32_t v = ldPoll32(p);
    uint64_t t0 = 0;
    while (v == kEmpty)
    {
        spinCheck(ctx, t0, code, 0, 0);
        v = ldPoll32(p);
    }
    return v;
}

__device__ __forceinline__ bool full4Hist(uint4 v)
{
    return (v.x != kEmptyHist) & (v.y != kEmptyHist) & (v.z != kEmptyHist) & (v.w != kEmptyHist);
}

__device__ __forceinline__ uint4 poll4Hist(Ctx const& ctx, void const* p, int code)
{
    uint4 v = ldPoll128(p);
    uint64_t t0 = 0;
    while (!full4Hist(v))
    {
        spinCheck(ctx, t0, code, 0, 0);
        v = ldPoll128(p);
    }
    return v;
}

__device__ __forceinline__ uint4 poll4HistValid(Ctx const& ctx, void const* p, int valid, int code)
{
    auto anyEmp = [valid](uint4 v) -> bool
    {
        return (valid > 0 && v.x == kEmptyHist) || (valid > 1 && v.y == kEmptyHist) || (valid > 2 && v.z == kEmptyHist)
            || (valid > 3 && v.w == kEmptyHist);
    };
    uint4 v = ldPoll128(p);
    uint64_t t0 = 0;
    while (anyEmp(v))
    {
        spinCheck(ctx, t0, code, valid, 0);
        v = ldPoll128(p);
    }
    return v;
}

template <int kN>
__device__ __forceinline__ void pollStrided(Ctx const& ctx, uint4 (&v)[kN], uint8_t const* src, int stride, int code)
{
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        v[i] = ldPoll128(src + static_cast<size_t>(stride) * i);
    }
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            ok &= full4(v[i]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, stride, 0);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!full4(v[i]))
            {
                v[i] = ldPoll128(src + static_cast<size_t>(stride) * i);
            }
        }
    }
}

__device__ __forceinline__ void signalCounter(uint32_t* ctrPtr)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        redReleaseAdd(ctrPtr, 1u);
    }
}

__device__ __forceinline__ void waitCounterPtr(Ctx const& ctx, uint32_t const* ctrPtr, uint32_t target, int code)
{
    if (threadIdx.x == 0)
    {
        uint64_t t0 = 0;
        uint32_t v;
        while ((v = ldAcquire(ctrPtr)) < target)
        {
            spinCheck(ctx, t0, code, static_cast<int>(target), static_cast<int>(v));
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void waitBar(Ctx const& ctx, uint32_t bar, uint32_t parity, int code)
{
    if (!mbarTryWait(bar, parity))
    {
        uint64_t t0 = 0;
        while (!mbarTryWait(bar, parity))
        {
            spinCheck(ctx, t0, code, static_cast<int>(bar), static_cast<int>(parity));
        }
    }
}

__device__ __forceinline__ void copyPieces(
    uint32_t dst, uint8_t const* src, uint32_t bytes, uint32_t bar, bool evictFirst)
{
    constexpr uint32_t kPiece = 1u << 19;
    uint64_t const policy = evictFirst ? policyEvictFirst() : 0ull;
    for (uint32_t off = 0; off < bytes; off += kPiece)
    {
        uint32_t const n = bytes - off < kPiece ? bytes - off : kPiece;
        if (evictFirst)
        {
            bulkCopyHint(dst + off, src + off, n, bar, policy);
        }
        else
        {
            bulkCopy(dst + off, src + off, n, bar);
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// GEMM Building Blocks (BF16 m16n8k16, FP8_PB_WO m16n8k32 for Oproj, and TCGen05 MXFP8 helpers)
// ----------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void loadB(uint4 (&b)[2], T const* gB, int ldb, int chunk)
{
    int const lane = threadIdx.x & 31;
    if (lane < kTok * 4)
    {
        T const* src = gB + static_cast<size_t>(lane >> 2) * ldb + chunk * 64 + 16 * (lane & 3);
        b[0] = ldcg128(src);
        b[1] = ldcg128(src + 8);
    }
    else
    {
        b[0] = make_uint4(0u, 0u, 0u, 0u);
        b[1] = make_uint4(0u, 0u, 0u, 0u);
    }
}

template <int kN>
__device__ __forceinline__ void pollB(
    Ctx const& ctx, uint4 (&b)[kN][2], uint8_t const* gB, int ld, int chunk0, int code)
{
    int const lane = threadIdx.x & 31;
    bool const live = lane < kTok * 4;
    uint8_t const* src = gB + ((lane >> 2) * ld + chunk0 * 64 + 16 * (lane & 3)) * 2;
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        b[i][0] = live ? ldPoll128(src + 128 * i) : make_uint4(0u, 0u, 0u, 0u);
        b[i][1] = live ? ldPoll128(src + 128 * i + 16) : make_uint4(0u, 0u, 0u, 0u);
    }
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            ok &= full4(b[i][0]) & full4(b[i][1]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, chunk0, 0);
        if (live)
        {
#pragma unroll
            for (int i = 0; i < kN; ++i)
            {
                if (!full4(b[i][0]))
                {
                    b[i][0] = ldPoll128(src + 128 * i);
                }
                if (!full4(b[i][1]))
                {
                    b[i][1] = ldPoll128(src + 128 * i + 16);
                }
            }
        }
    }
}

__device__ __forceinline__ float amaxUint4Bf16(uint4 v, float rcpTok)
{
    float m = fmaxf(fabsf(bf16Round(bf16Lo(v.x) * rcpTok)), fabsf(bf16Round(bf16Hi(v.x) * rcpTok)));
    m = fmaxf(m, fmaxf(fabsf(bf16Round(bf16Lo(v.y) * rcpTok)), fabsf(bf16Round(bf16Hi(v.y) * rcpTok))));
    m = fmaxf(m, fmaxf(fabsf(bf16Round(bf16Lo(v.z) * rcpTok)), fabsf(bf16Round(bf16Hi(v.z) * rcpTok))));
    m = fmaxf(m, fmaxf(fabsf(bf16Round(bf16Lo(v.w) * rcpTok)), fabsf(bf16Round(bf16Hi(v.w) * rcpTok))));
    return m;
}

__device__ __forceinline__ uint32_t quantPairBf16(uint32_t loWord, uint32_t hiWord, float rcpTok, float inv)
{
    uint32_t const lo = toE4m3x2(
        __fmul_rn(bf16Round(bf16Lo(loWord) * rcpTok), inv), __fmul_rn(bf16Round(bf16Hi(loWord) * rcpTok), inv));
    uint32_t const hi = toE4m3x2(
        __fmul_rn(bf16Round(bf16Lo(hiWord) * rcpTok), inv), __fmul_rn(bf16Round(bf16Hi(hiWord) * rcpTok), inv));
    return lo | (hi << 16);
}

__device__ __forceinline__ void quant128B(
    uint4 const (&bEven)[2], uint4 const (&bOdd)[2], float rcpTok, uint4 (&q8)[2], float& xs0, float& xs1)
{
    int const lane = threadIdx.x & 31;
    bool const live = lane < kTok * 4;
    float amax = 0.f;
    if (live)
    {
        amax = fmaxf(amaxUint4Bf16(bEven[0], rcpTok), amaxUint4Bf16(bEven[1], rcpTok));
        amax = fmaxf(amax, fmaxf(amaxUint4Bf16(bOdd[0], rcpTok), amaxUint4Bf16(bOdd[1], rcpTok)));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    float const xs = live ? ue8m0Scale(amax) : 0.f;
    float const inv = live ? pow2Inv(xs) : 0.f;
    if (live)
    {
        q8[0] = make_uint4(quantPairBf16(bEven[0].x, bEven[0].y, rcpTok, inv),
            quantPairBf16(bEven[0].z, bEven[0].w, rcpTok, inv), quantPairBf16(bEven[1].x, bEven[1].y, rcpTok, inv),
            quantPairBf16(bEven[1].z, bEven[1].w, rcpTok, inv));
        q8[1] = make_uint4(quantPairBf16(bOdd[0].x, bOdd[0].y, rcpTok, inv),
            quantPairBf16(bOdd[0].z, bOdd[0].w, rcpTok, inv), quantPairBf16(bOdd[1].x, bOdd[1].y, rcpTok, inv),
            quantPairBf16(bOdd[1].z, bOdd[1].w, rcpTok, inv));
    }
    else
    {
        q8[0] = make_uint4(0u, 0u, 0u, 0u);
        q8[1] = make_uint4(0u, 0u, 0u, 0u);
    }
    int const c = lane & 3;
    int const t0 = (2 * c) < kTok ? (2 * c) : 0;
    int const t1 = (2 * c + 1) < kTok ? (2 * c + 1) : 0;
    xs0 = __shfl_sync(0xffffffffu, xs, t0 * 4);
    xs1 = __shfl_sync(0xffffffffu, xs, t1 * 4);
}

// Quantize contiguous BF16 activation slice [kTok, nKTiles * 128] (starting at kTile0 * 128) into TCGen05 MXFP8
// SWIZZLE_128B smemB ([nKTiles][8][128]) + R128c4 smemSfb ([nKTiles][512]) using warps 4..7 (128 threads, 0 spill!).
// Supports raw global BF16 (poll=false), arena Lamport BF16 (poll=true, with optional sRcp multiply), and optional
// fused LayerNorm (lnMean/lnRstd + qnW/qnB).
template <bool kPoll, bool kLayerNorm>
__device__ __forceinline__ void quantActMxf8Warps47(Ctx const& ctx, uint8_t const* srcBase, int srcLdCols,
    float const* sRcp, float const* sLnMean, float const* sLnRstd, __nv_bfloat16 const* qnW, __nv_bfloat16 const* qnB,
    int kTile0, int nKTiles, uint8_t* smemB, uint8_t* smemSfb, int code)
{
    int const qzTid = static_cast<int>(threadIdx.x) - 128; // 0..127
    // Zero rows 4..7 of smemB (8*128 = 1024 B/tile) and all of smemSfb (128 B/tile for utccp1Rows8).
    for (int v = qzTid; v < nKTiles * 32; v += 128)
    {
        if (v < nKTiles * 8)
        {
            *reinterpret_cast<uint4*>(smemSfb + 16 * v) = make_uint4(0u, 0u, 0u, 0u);
        }
        int const ki = v >> 5;
        int const rOff = 512 + 16 * (v & 31); // rows 4..7 = bytes [512..1024)
        *reinterpret_cast<uint4*>(smemB + ki * 1024 + rOff) = make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("bar.sync 2, 128;" ::: "memory");
    // 4 consecutive lanes hold the 4 K32 sub-blocks of one (ki, tok) K128 block.
    // Round loop bound up to warp multiple so __shfl_xor_sync(0xffffffffu, ...) always converges across all 32 lanes.
    int const totalItems = nKTiles * kTok * 4;
    int const roundedItems = (totalItems + 31) & ~31;
    for (int item = qzTid; item < roundedItems; item += 128)
    {
        bool const active = item < totalItems;
        int const ki = active ? (item >> 4) : 0;
        int const tok = active ? ((item >> 2) & 3) : 0;
        int const k32 = active ? (item & 3) : 0;
        int const col0 = (kTile0 + ki) * 128 + k32 * 32;
        uint4 raw[4];
        float amax = 0.f;
        if (active)
        {
            uint8_t const* rowPtr = srcBase + (static_cast<size_t>(tok) * srcLdCols + col0) * 2;
            if constexpr (kPoll)
            {
                pollStrided<4>(ctx, raw, rowPtr, 16, code);
            }
            else
            {
#pragma unroll
                for (int z = 0; z < 4; ++z)
                {
                    raw[z] = ldcg128(rowPtr + 16 * z);
                }
            }
            float const rcpTok = (sRcp != nullptr) ? sRcp[tok] : 1.f;
            float const meanT = kLayerNorm ? sLnMean[tok] : 0.f;
            float const rstdT = kLayerNorm ? sLnRstd[tok] : 0.f;

            auto prepWord = [&](uint32_t& w, int z, int e)
            {
                float v0 = bf16Lo(w);
                float v1 = bf16Hi(w);
                if (sRcp != nullptr)
                {
                    v0 = bf16Round(v0 * rcpTok);
                    v1 = bf16Round(v1 * rcpTok);
                }
                if constexpr (kLayerNorm)
                {
                    int const c = col0 + 8 * z + 2 * e;
                    uint32_t const gw = *reinterpret_cast<uint32_t const*>(qnW + c);
                    uint32_t const gb = *reinterpret_cast<uint32_t const*>(qnB + c);
                    v0 = bf16Round(fmaf((v0 - meanT) * rstdT, bf16Lo(gw), bf16Lo(gb)));
                    v1 = bf16Round(fmaf((v1 - meanT) * rstdT, bf16Hi(gw), bf16Hi(gb)));
                }
                amax = fmaxf(amax, fmaxf(fabsf(v0), fabsf(v1)));
                w = packBf16(v0, v1);
            };
#pragma unroll
            for (int z = 0; z < 4; ++z)
            {
                prepWord(raw[z].x, z, 0);
                prepWord(raw[z].y, z, 1);
                prepWord(raw[z].z, z, 2);
                prepWord(raw[z].w, z, 3);
            }
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        if (active)
        {
            float const scale = ue8m0Scale(amax);
            float const inv = pow2Inv(scale);
            uint8_t const expByte = ue8m0ExpByte(scale);

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
            uint8_t* bRow = smemB + ki * 1024 + tok * 128;
            *reinterpret_cast<uint4*>(bRow + 16 * (c0 ^ (tok & 7))) = make_uint4(q[0], q[1], q[2], q[3]);
            *reinterpret_cast<uint4*>(bRow + 16 * ((c0 + 1) ^ (tok & 7))) = make_uint4(q[4], q[5], q[6], q[7]);
            smemSfb[ki * 128 + tok * 16 + k32] = expByte;
        }
    }
}

// BF16 MMA: acc += A(tile, chunk64) x B(chunk64).
__device__ __forceinline__ void mmaChunkBf16(float (&acc)[4], uint32_t aBlock, uint4 const (&b)[2])
{
    int const lane = threadIdx.x & 31;
    uint32_t const a = aBlock + lane * 16;
    uint4 const p0 = lds128(a);
    uint4 const p1 = lds128(a + 512);
    uint4 const p2 = lds128(a + 1024);
    uint4 const p3 = lds128(a + 1536);
    mmaBf16(acc, p0.x, p2.x, p0.y, p2.y, b[0].x, b[0].y);
    mmaBf16(acc, p0.z, p2.z, p0.w, p2.w, b[0].z, b[0].w);
    mmaBf16(acc, p1.x, p3.x, p1.y, p3.y, b[1].x, b[1].y);
    mmaBf16(acc, p1.z, p3.z, p1.w, p3.w, b[1].z, b[1].w);
}

// FP8_PB_WO MMA (used by stageOproj).
__device__ __forceinline__ void mmaChunkFp8PbWo(
    float (&acc)[4], uint32_t aBlock, uint4 const (&q8)[2], float ws, float xs0, float xs1)
{
    int const lane = threadIdx.x & 31;
    uint32_t const a = aBlock + lane * 16;
    uint4 const p0 = lds128(a);
    uint4 const p1 = lds128(a + 512);
    uint4 const p2 = lds128(a + 1024);
    uint4 const p3 = lds128(a + 1536);
    float bacc[4] = {0.f, 0.f, 0.f, 0.f};
    mmaE4m3(bacc, p0.x, p2.x, p0.y, p2.y, q8[0].x, q8[0].y);
    mmaE4m3(bacc, p0.z, p2.z, p0.w, p2.w, q8[0].z, q8[0].w);
    mmaE4m3(bacc, p1.x, p3.x, p1.y, p3.y, q8[1].x, q8[1].y);
    mmaE4m3(bacc, p1.z, p3.z, p1.w, p3.w, q8[1].z, q8[1].w);
    float const sc0 = xs0 * ws;
    float const sc1 = xs1 * ws;
    acc[0] = fmaf(bacc[0], sc0, acc[0]);
    acc[1] = fmaf(bacc[1], sc1, acc[1]);
    acc[2] = fmaf(bacc[2], sc0, acc[2]);
    acc[3] = fmaf(bacc[3], sc1, acc[3]);
}

template <int kTiles, int kNumWarps = kWarps>
__device__ __forceinline__ void reduceWarps(Ctx const& ctx, float const (&acc)[kTiles][4], float* out)
{
    float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed);
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    if (warp < kNumWarps && (lane & 3) < 2)
    {
        int const compactLane = ((lane >> 2) << 1) | (lane & 1);
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
        {
            *reinterpret_cast<float4*>(red + ((warp * kTiles + t) * 16 + compactLane) * 4)
                = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
        }
    }
    __syncthreads();
    for (int i = static_cast<int>(threadIdx.x); i < kTiles * 16 * 4; i += kThreads)
    {
        int const t = i / 64;
        int const cl = (i / 4) % 16;
        int const r = i % 4;
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < kNumWarps; ++w)
        {
            s += red[((w * kTiles + t) * 16 + cl) * 4 + r];
        }
        int const g = cl >> 1;
        int const c = cl & 1;
        int const row = 16 * t + g + (r >= 2 ? 8 : 0);
        int const tok = 2 * c + (r & 1);
        out[row * kTok + tok] = s;
    }
    __syncthreads();
}

__device__ __forceinline__ void storeTileBf16(
    float const* out, uint8_t* dst, int ld, int row0, float const* scale = nullptr)
{
    if (threadIdx.x < kTok * 8)
    {
        int const tok = static_cast<int>(threadIdx.x) / 8;
        int const pr = static_cast<int>(threadIdx.x) % 8;
        float const sc = scale != nullptr ? scale[tok] : 1.f;
        uint32_t const w = packBf16(out[(2 * pr) * kTok + tok] * sc, out[(2 * pr + 1) * kTok + tok] * sc);
        st32(dst + (tok * ld + row0 + 2 * pr) * 2, lamport(w));
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Static unit weight descriptors (precomputed into per-CTA table by ml4WeightTableKernel)
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void unitWeights(
    Ctx const& ctx, int stage, int unit, uint8_t const*& src0, uint32_t& n0, uint8_t const*& src1, uint32_t& n1)
{
    src0 = nullptr;
    src1 = nullptr;
    n0 = 0;
    n1 = 0;
    bool const isNope = (ctx.flags & kFlagNope) != 0;
    switch (stage)
    {
    case kStWkvA:
        src0 = ptr<uint8_t const>(ctx, kPWkvA) + static_cast<size_t>(unit) * kHiddenChunksBf16 * kBf16Block;
        n0 = kHiddenChunksBf16 * kBf16Block; // 196,608 B
        break;
    case kStKb:
    {
        int const chunks = isNope ? 2 : 1;
        src0 = ptr<uint8_t const>(ctx, kPKb) + static_cast<size_t>(unit) * 4 * chunks * kBf16Block;
        n0 = static_cast<uint32_t>(4 * chunks * kBf16Block); // 8,192 B (RoPE) or 16,384 B (NoPE)
        break;
    }
    case kStVup:
        src0 = ptr<uint8_t const>(ctx, kPVb) + static_cast<size_t>(unit) * kVupChunks * kBf16Block;
        n0 = kVupChunks * kBf16Block; // 16,384 B
        break;
    case kStOproj:
        src0 = ptr<uint8_t const>(ctx, kPWO) + static_cast<size_t>(unit) * 2 * kOprojBlocksFp8 * kFp8Block;
        n0 = 2 * kOprojBlocksFp8 * kFp8Block; // 16,384 B
        break;
    case kStRouterGemv:
    {
        int const et = unit / 4;
        int const kq = unit % 4;
        src0 = ptr<uint8_t const>(ctx, kPWR) + (static_cast<size_t>(et) * kHiddenChunksBf16 + 24 * kq) * kBf16Block;
        n0 = 24 * kBf16Block; // 49,152 B
        break;
    }
    default: break;
    }
}

__device__ __noinline__ void issueUnitWeights(Ctx const& ctx, int j, uint32_t bar)
{
    int const* roles = reinterpret_cast<int const*>(ctx.smem + kSmemSmall + kSmallRoles);
    int64_t const* w = reinterpret_cast<int64_t const*>(ctx.smem + kSmemSmall + kSmallWeights) + 3 * j;
    uint8_t const* src0 = reinterpret_cast<uint8_t const*>(w[0]);
    uint8_t const* src1 = reinterpret_cast<uint8_t const*>(w[1]);
    uint32_t const n0 = static_cast<uint32_t>(w[2]);
    uint32_t const n1 = static_cast<uint32_t>(static_cast<uint64_t>(w[2]) >> 32);
    if (n0 + n1 == 0)
    {
        return;
    }
    uint32_t const dst = ctx.smemBase + static_cast<uint32_t>(roles[kMaxRoles + j]);
    fenceProxyAsyncSmem();
    mbarExpectTx(bar, n0 + n1);
    bool const evictFirst = (ctx.flags & kFlagEvictFirst) != 0;
    copyPieces(dst, src0, n0, bar, evictFirst);
    if (n1 != 0)
    {
        copyPieces(dst + n0, src1, n1, bar, evictFirst);
    }
}

__global__ void ml4WeightTableKernel(int64_t const* params, int64_t* out)
{
    int const c = static_cast<int>(blockIdx.x);
    int const j = static_cast<int>(threadIdx.x);
    Ctx ctx{};
    ctx.p = params;
    ctx.flags = params[kPFlags];
    int const role = reinterpret_cast<int const*>(params + kPNumParams)[c * kMaxRoles + j];
    uint8_t const* src0 = nullptr;
    uint8_t const* src1 = nullptr;
    uint32_t n0 = 0;
    uint32_t n1 = 0;
    if (role != 0)
    {
        unitWeights(ctx, role >> 16, role & 0xffff, src0, n0, src1, n1);
    }
    int64_t* o = out + (static_cast<int64_t>(c) * kMaxRoles + j) * 3;
    o[0] = reinterpret_cast<int64_t>(src0);
    o[1] = reinterpret_cast<int64_t>(src1);
    o[2] = static_cast<int64_t>(n0) | (static_cast<int64_t>(n1) << 32);
}

__global__ void __launch_bounds__(kThreads, 1) ml4SmDieProbeKernel(
    uint32_t const* __restrict__ die0Buf, uint32_t const* __restrict__ die1Buf, int64_t* __restrict__ out)
{
    extern __shared__ uint8_t probeSmem[];
    if (threadIdx.x != 0)
    {
        return;
    }
    (void) probeSmem;
    uint32_t const sm = getSmid();
    int const cta = static_cast<int>(blockIdx.x);
    auto timeCold = [&](uint32_t const* base) -> int64_t
    {
        uint32_t idx = static_cast<uint32_t>(cta * 64);
        uint64_t t0, t1;
        constexpr int kProbes = 32;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0)::"memory");
        for (int i = 0; i < kProbes; ++i)
        {
            uint32_t val;
            asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(val) : "l"(base + (idx & 0x7ffffu)) : "memory");
            idx = (idx + 64u) ^ (val & 1u);
        }
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.eq.u32 p, %1, 0xdeadbeef;\n"
            "@p trap;\n"
            "mov.u64 %0, %%clock64;\n"
            "}\n"
            : "=l"(t1)
            : "r"(idx)
            : "memory");
        return static_cast<int64_t>(t1 - t0) / kProbes;
    };
    out[cta * 3 + 0] = static_cast<int64_t>(sm);
    out[cta * 3 + 1] = timeCold(die0Buf);
    out[cta * 3 + 2] = timeCold(die1Buf);
}

// clang-format off
#include "mistral_large4/mistral_large_4_nvfp4__sm_107__tp4/csrc/ml4MegakernelStages.cuh"
#include "mistral_large4/mistral_large_4_nvfp4__sm_107__tp4/csrc/ml4MegakernelIdx.cuh"

// clang-format on

// ----------------------------------------------------------------------------------------------------------------
// Lean Persistent Main Layer Kernel (212 CTAs, zero indexer code!)
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void runUnit(Ctx const& c, int stage, int unit, uint32_t off, uint32_t bar,
    __nv_bfloat16 const* x, __nv_bfloat16 const* residual, __nv_bfloat16* xOut, __nv_bfloat16* resOut, int const* topk,
    int const* blockTable, int const* kvLen, int const* positions)
{
    switch (stage)
    {
    case kStWq: stageWqPlain(c, unit, off, x, positions); break;
    case kStWkvA: stageWkvA(c, unit, off, bar, x); break;
    case kStKb: stageKb(c, unit, off, bar); break;
    case kStKvAppend: stageKvAppend(c, blockTable, positions); break;
    case kStAttn: stageAttn(c, unit, off, kvLen, blockTable, topk); break;
    case kStMerge: stageMerge(c, unit); break;
    case kStVup: stageVup(c, unit, off, bar); break;
    case kStOproj: stageOproj(c, unit, off, bar); break;
    case kStClear: stageClear(c, unit); break;
    case kStH1: stageHop(c, unit, false, residual, xOut, resOut); break;
    case kStRouterGemv: stageRouterGemv(c, unit, off, bar); break;
    case kStShared: stageShared(c, unit, off); break;
    case kStMoe: stageMoe(c, unit, off); break;
    case kStOwner: stageOwner(c, unit); break;
    case kStH2: stageHop(c, unit, true, residual, xOut, resOut); break;
    default: break;
    }
}

__global__ void __launch_bounds__(kThreads, 1) ml4LayerKernel(int64_t const* __restrict__ params,
    __nv_bfloat16 const* __restrict__ x, __nv_bfloat16 const* __restrict__ residual, __nv_bfloat16* __restrict__ xOut,
    __nv_bfloat16* __restrict__ resOut, int const* __restrict__ topk, int const* __restrict__ blockTable,
    int const* __restrict__ kvLen, int const* __restrict__ positions)
{
    extern __shared__ __align__(1024) uint8_t smemRaw[];
    int const tid = static_cast<int>(threadIdx.x);

    int64_t* sParams = reinterpret_cast<int64_t*>(smemRaw + kSmemSmall + kSmallParams);
    int* sRoles = reinterpret_cast<int*>(smemRaw + kSmemSmall + kSmallRoles);
    int64_t* sWeights = reinterpret_cast<int64_t*>(smemRaw + kSmemSmall + kSmallWeights);
    if (tid < kPNumParams)
    {
        sParams[tid] = __ldg(params + tid);
    }
    __syncthreads();

    int logicalCta = static_cast<int>(blockIdx.x);
    if ((sParams[kPFlags] & kFlagDieAware) != 0 && sParams[kPSmidMap] != 0)
    {
        int const* smMap = reinterpret_cast<int const*>(sParams[kPSmidMap]);
        int const mapped = __ldg(smMap + getSmid());
        if (mapped >= 0 && mapped < kCtas)
        {
            logicalCta = mapped;
        }
    }

    int const* tables = reinterpret_cast<int const*>(params + kPNumParams);
    int64_t const* wt
        = params + kPNumParams + 3 * kCtas * kMaxRoles / 2 + static_cast<size_t>(logicalCta) * 3 * kMaxRoles;
    if (tid < 3 * kMaxRoles)
    {
        int const tbl = tid / kMaxRoles;
        int const idx = tid % kMaxRoles;
        sRoles[tid] = __ldg(tables + (tbl * kCtas + logicalCta) * kMaxRoles + idx);
        sWeights[tid] = __ldg(wt + tid);
    }

    Ctx& ctx = *reinterpret_cast<Ctx*>(smemRaw + kSmemSmall + kSmallCtx);
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
        ctx.cta = logicalCta;
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

    if (sParams[kPMagic] != kMagic || sParams[kPHopBytes] != kHopBytes || sParams[kPMaxRoles] != kMaxRoles
        || (ctx.smemBase & 1023u) != 0)
    {
        __trap();
    }

    if (tid < 32)
    {
        if (tid < kNumBars)
        {
            mbarInit(ctx.smemBase + kSmemSmall + kSmallBars + 8 * tid, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        __syncwarp();
        if (tid == 0)
        {
#pragma unroll 1
            for (int i = 0; i < kMaxRoles && sRoles[i] != 0; ++i)
            {
                if (sRoles[2 * kMaxRoles + i] == -1)
                {
                    issueUnitWeights(ctx, i, ctx.smemBase + kSmemSmall + kSmallBars + 8 * i);
                }
            }
        }
    }
    else if (tid >= 32 && tid < 64)
    {
        tmemAlloc(smemU32(smemRaw + kSmemSmall + kSmallScalars + 4 * kScalarTmem), kTmemCols);
    }

    if (tid < 128)
    {
        reinterpret_cast<int*>(smemRaw + kSmemSmall + kSmallMoe)[tid] = 0;
    }

    // Zero next launch copy's counters (64 words including kCFc2Ready0..15).
    {
        uint32_t* next = reinterpret_cast<uint32_t*>(sParams[kPCounters]) + ctx.nextCopy * kNumCounters;
        for (int i = logicalCta * kThreads + tid; i < kNumCounters; i += kCtas * kThreads)
        {
            next[i] = 0u;
        }
    }
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();

    int nRoles = 0;
#pragma unroll 1
    while (nRoles < kMaxRoles && sRoles[nRoles] != 0)
    {
        ++nRoles;
    }

    // Cross-layer & intra-layer PDL overlap:
    //  - Plain layers (kFlagEarlyX): Wq/WkvA start before griddepcontrol.wait polling kArX.
    //  - Leader layers (kFlagSkipFront): stageAttn starts before griddepcontrol.wait polling kArFusedQ/kArKv/idxTkUc
    //    while ml4LeaderFrontKernel's 4 resolver CTAs finish kModeCand!
    int const st0 = nRoles > 0 ? (sRoles[0] >> 16) : kStNone;
    bool const skipFront = (ctx.flags & kFlagSkipFront) != 0;
    bool const early = (skipFront && st0 == kStAttn)
        || ((ctx.flags & kFlagEarlyX) != 0 && !skipFront && (st0 == kStWq || st0 == kStWkvA));
    auto dependencyWait = [&]()
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    };
    if (!early)
    {
        dependencyWait();
    }

    for (int i = 0; i < nRoles; ++i)
    {
        if (early && i == 1)
        {
            dependencyWait();
        }
        int const role = sRoles[i];
        int const stage = role >> 16;
        int const unit = role & 0xffff;
        uint32_t const off = static_cast<uint32_t>(sRoles[kMaxRoles + i]);
        uint32_t const bar = ctx.smemBase + kSmemSmall + kSmallBars + 8 * i;
        runUnit(ctx, stage, unit, off, bar, x, residual, xOut, resOut, topk, blockTable, kvLen, positions);
        fenceProxyAsyncSmem();
        __syncthreads();
        if ((stage == kStMoe || stage == kStOwner) && (ctx.flags & kFlagHoldExit) != 0 && tid == 0)
        {
            atomicAdd(ctx.counters + kCMoeDone, 1u);
        }
        if (tid == 0)
        {
            bool const attnNext = i + 1 < nRoles && (sRoles[i + 1] >> 16) == kStAttn;
#pragma unroll 1
            for (int j = i + 1; j < nRoles; ++j)
            {
                if (sRoles[2 * kMaxRoles + j] == i)
                {
                    if (attnNext && (sRoles[j] >> 16) == kStOproj)
                    {
                        sRoles[2 * kMaxRoles + j] = -4;
                    }
                    else
                    {
                        issueUnitWeights(ctx, j, ctx.smemBase + kSmemSmall + kSmallBars + 8 * j);
                    }
                }
            }
        }
    }

    if (early && nRoles == 1)
    {
        dependencyWait();
    }
    tcFenceBefore();
    __syncthreads();
    if (tid < 32)
    {
        tcFenceAfter();
        uint32_t const tmem
            = *reinterpret_cast<uint32_t const*>(smemRaw + kSmemSmall + kSmallScalars + 4 * kScalarTmem);
        tmemDealloc(tmem, kTmemCols);
    }
    if ((ctx.flags & kFlagHoldExit) != 0)
    {
        waitCounterPtr(ctx, ctx.counters + kCMoeDone, static_cast<uint32_t>(kCtas), 150);
    }
}

#else

__global__ void ml4LayerKernel(int64_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    __nv_bfloat16*, int const*, int const*, int const*, int const*)
{
}

__global__ void ml4LeaderFrontKernel(int64_t const*, __nv_bfloat16 const*, int const*, int const*, int const*) {}

__global__ void ml4WeightTableKernel(int64_t const*, int64_t*) {}

__global__ void ml4SmDieProbeKernel(uint32_t const*, uint32_t const*, int64_t*) {}

#endif

} // namespace ml4

// ----------------------------------------------------------------------------------------------------------------
// Host API exported to ml4MegakernelOp.cpp
// ----------------------------------------------------------------------------------------------------------------
int ml4LayerSmemBytes()
{
    return ml4::kSmemBytes;
}

int ml4LayerIdxSmemBytes()
{
    return ml4::kIdxSmemBytes;
}

int ml4LayerNumParams()
{
    return ml4::kPNumParams;
}

int64_t ml4LayerMagic()
{
    return ml4::kMagic;
}

int ml4LayerArenaBytes()
{
    return ml4::kArenaBytes;
}

int ml4LayerClearUnits()
{
    return ml4::kClearUnits;
}

int ml4LayerNumCounters()
{
    return ml4::kNumCounters;
}

int ml4LayerHopBytes()
{
    return ml4::kHopBytes;
}

int ml4LayerTkBytes()
{
    return ml4::kTkBytes;
}

int ml4LayerIdxHistBytes()
{
    return ml4::kIdxHistBytes;
}

int ml4LayerIdxBndBytes()
{
    return ml4::kIdxBndBytes;
}

int ml4LayerIdxArenaBytes()
{
    return ml4::kIdxArenaBytes;
}

int ml4LayerIdxLogitsBytes()
{
    return ml4::kIdxLogitsBytes;
}

int ml4LayerMaxKeys()
{
    return ml4::kIdxMaxKeys;
}

int ml4LayerIdxTmap(void* out, void* pool, unsigned long long numRows, int tokensPerBlock)
{
    cuuint64_t dims[2] = {128, static_cast<cuuint64_t>(numRows)};
    cuuint64_t strides[1] = {128};
    cuuint32_t box[2] = {128, static_cast<cuuint32_t>(tokensPerBlock)};
    cuuint32_t elem[2] = {1, 1};
    CUresult const r = cuTensorMapEncodeTiled(static_cast<CUtensorMap*>(out), CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, pool,
        dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return static_cast<int>(r);
}

int ml4LayerWeightTable(void const* params, void* out, void* stream)
{
    ml4::ml4WeightTableKernel<<<ml4::kCtas, ml4::kMaxRoles, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t const*>(params), static_cast<int64_t*>(out));
    return static_cast<int>(cudaGetLastError());
}

int ml4LayerSmDieProbe(void const* die0Ptr, void const* die1Ptr, void* outCycles, void* stream)
{
#if defined(PLACEHOLDER_KERNELS)
    (void) die0Ptr;
    (void) die1Ptr;
    (void) outCycles;
    (void) stream;
    return static_cast<int>(cudaErrorNotSupported);
#else
    constexpr int kProbeSmem = 196608;
    cudaFuncSetAttribute(ml4::ml4SmDieProbeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kProbeSmem);
    ml4::ml4SmDieProbeKernel<<<ml4::kCtas, ml4::kThreads, kProbeSmem, static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint32_t const*>(die0Ptr), static_cast<uint32_t const*>(die1Ptr), static_cast<int64_t*>(outCycles));
    return static_cast<int>(cudaGetLastError());
#endif
}

int ml4LayerInit()
{
#if defined(PLACEHOLDER_KERNELS)
    return static_cast<int>(cudaErrorNotSupported);
#else
    int dev = 0;
    int optInBytes = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optInBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    auto configKernel = [&](void const* fn, int smemBytes) -> cudaError_t
    {
        if (smemBytes > optInBytes)
        {
            return cudaFuncSetAttribute(
                fn, cudaFuncAttributeSharedMemoryMode, cudaSharedMemoryModePreferOversizedSharedMemory);
        }
        cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
        if (e != cudaSuccess)
        {
            return e;
        }
        return cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    };
    cudaError_t e = configKernel(reinterpret_cast<void const*>(ml4::ml4LayerKernel), ml4::kSmemBytes);
    if (e != cudaSuccess)
    {
        return static_cast<int>(e);
    }
    e = configKernel(reinterpret_cast<void const*>(ml4::ml4LeaderFrontKernel), ml4::kIdxSmemBytes);
    if (e != cudaSuccess)
    {
        return static_cast<int>(e);
    }
    e = cudaFuncSetAttribute(ml4::ml4LeaderFrontKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    return static_cast<int>(e);
#endif
}

int ml4LayerLaunch(void const* params, void const* x, void const* residual, void* xOut, void* resOut, void const* topk,
    void const* blockTable, void const* kvLen, void const* positions, void* stream, int pdl, int isLeader)
{
#if defined(PLACEHOLDER_KERNELS)
    (void) params;
    (void) x;
    (void) residual;
    (void) xOut;
    (void) resOut;
    (void) topk;
    (void) blockTable;
    (void) kvLen;
    (void) positions;
    (void) stream;
    (void) pdl;
    (void) isLeader;
    return static_cast<int>(cudaErrorNotSupported);
#else
    if (isLeader != 0)
    {
        cudaLaunchConfig_t fcfg{};
        fcfg.gridDim = dim3(ml4::kIdxFrontCtas);
        fcfg.blockDim = dim3(ml4::kThreads);
        fcfg.dynamicSmemBytes = ml4::kIdxSmemBytes;
        fcfg.stream = static_cast<cudaStream_t>(stream);
        cudaLaunchAttribute fattrs[3]{};
        fattrs[0].id = cudaLaunchAttributeClusterDimension;
        fattrs[0].val.clusterDim.x = ml4::kIdxClusterSize;
        fattrs[0].val.clusterDim.y = 1;
        fattrs[0].val.clusterDim.z = 1;
        fattrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        fattrs[1].val.programmaticStreamSerializationAllowed = pdl != 0 ? 1 : 0;
        fattrs[2].id = cudaLaunchAttributePreferredSharedMemoryCarveout;
        fattrs[2].val.sharedMemCarveout = 100;
        fcfg.attrs = fattrs;
        fcfg.numAttrs = 3;
        cudaError_t const fe = cudaLaunchKernelEx(&fcfg, ml4::ml4LeaderFrontKernel, static_cast<int64_t const*>(params),
            static_cast<__nv_bfloat16 const*>(x), static_cast<int const*>(blockTable), static_cast<int const*>(kvLen),
            static_cast<int const*>(positions));
        if (fe != cudaSuccess)
        {
            return static_cast<int>(fe);
        }
    }

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(ml4::kCtas);
    config.blockDim = dim3(ml4::kThreads);
    config.dynamicSmemBytes = ml4::kSmemBytes;
    config.stream = static_cast<cudaStream_t>(stream);
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl != 0 ? 1 : 0;
    attrs[1].id = cudaLaunchAttributePreferredSharedMemoryCarveout;
    attrs[1].val.sharedMemCarveout = 100;
    config.attrs = attrs;
    config.numAttrs = 2;
    cudaError_t const e = cudaLaunchKernelEx(&config, ml4::ml4LayerKernel, static_cast<int64_t const*>(params),
        static_cast<__nv_bfloat16 const*>(x), static_cast<__nv_bfloat16 const*>(residual),
        static_cast<__nv_bfloat16*>(xOut), static_cast<__nv_bfloat16*>(resOut), static_cast<int const*>(topk),
        static_cast<int const*>(blockTable), static_cast<int const*>(kvLen), static_cast<int const*>(positions));
    return static_cast<int>(e);
#endif
}

} // namespace kernels::mistral_large4::megakernel

ML4_MEGA_NS_END
