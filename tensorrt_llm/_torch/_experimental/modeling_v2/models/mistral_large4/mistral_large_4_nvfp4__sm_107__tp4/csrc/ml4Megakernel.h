// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(TRTLLM_MV2_SCOPE)
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#define ML4_MEGA_NS_BEGIN TRTLLM_MV2_NAMESPACE_BEGIN
#define ML4_MEGA_NS_END TRTLLM_MV2_NAMESPACE_END
#define ML4_MEGA_NS TRTLLM_MV2_NS
#else
#include "tensorrt_llm/common/config.h"
#define ML4_MEGA_NS_BEGIN                                                                                              \
    TRTLLM_NAMESPACE_BEGIN                                                                                             \
    namespace modeling_v2::mistral_large4::sm_107_tp4                                                                  \
    {
#define ML4_MEGA_NS_END                                                                                                \
    }                                                                                                                  \
    TRTLLM_NAMESPACE_END
#define ML4_MEGA_NS ::tensorrt_llm::modeling_v2::mistral_large4::sm_107_tp4
#endif

ML4_MEGA_NS_BEGIN

namespace kernels::mistral_large4::megakernel
{
int ml4LayerSmemBytes();
int ml4LayerIdxSmemBytes();
int ml4LayerNumParams();
int64_t ml4LayerMagic();
int ml4LayerArenaBytes();
int ml4LayerClearUnits();
int ml4LayerNumCounters();
int ml4LayerHopBytes();
int ml4LayerTkBytes();
int ml4LayerIdxHistBytes();
int ml4LayerIdxBndBytes();
int ml4LayerIdxArenaBytes();
int ml4LayerIdxLogitsBytes();
int ml4LayerMaxKeys();
int ml4LayerIdxTmap(void* out, void* pool, unsigned long long numRows, int tokensPerBlock);
int ml4LayerWeightTable(void const* params, void* out, void* stream);
int ml4LayerSmDieProbe(void const* die0Ptr, void const* die1Ptr, void* outCycles, void* stream);
int ml4LayerInit();
int ml4LayerLaunch(void const* params, void const* x, void const* residual, void* xOut, void* resOut, void const* topk,
    void const* blockTable, void const* kvLen, void const* positions, void* stream, int pdl, int isLeader);
} // namespace kernels::mistral_large4::megakernel

ML4_MEGA_NS_END
