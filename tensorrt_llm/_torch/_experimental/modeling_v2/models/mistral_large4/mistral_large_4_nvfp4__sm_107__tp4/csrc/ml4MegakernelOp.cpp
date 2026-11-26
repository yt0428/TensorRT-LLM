// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mistral_large4/mistral_large_4_nvfp4__sm_107__tp4/csrc/ml4Megakernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <torch/library.h>
#include <tuple>
#include <vector>

#if defined(TRTLLM_MV2_TORCH_NS)
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#define ML4_MEGA_LIBRARY(m) TRTLLM_MV2_LIBRARY_FRAGMENT(m)
#define ML4_MEGA_LIBRARY_IMPL(k, m) TRTLLM_MV2_LIBRARY_IMPL(k, m)
#else
#define ML4_MEGA_LIBRARY(m) TORCH_LIBRARY(ml4_megakernel, m)
#define ML4_MEGA_LIBRARY_IMPL(k, m) TORCH_LIBRARY_IMPL(ml4_megakernel, k, m)
#endif

namespace mega = ML4_MEGA_NS::kernels::mistral_large4::megakernel;

ML4_MEGA_NS_BEGIN

namespace torch_ext
{
namespace
{
constexpr int64_t kRows = 4;
constexpr int64_t kHidden = 6144;
constexpr int64_t kTopK = 2048;
constexpr int64_t kCtas = 212;
constexpr int64_t kMaxRoles = 16;
constexpr int64_t kWeightWords = kCtas * kMaxRoles * 3;
constexpr int64_t kRoleWordsInt64 = kWeightWords / 2;

void checkMegakernelDevice(at::Tensor const& tensor)
{
    CHECK_TH_CUDA(tensor);
    auto const* props = at::cuda::getDeviceProperties(tensor.get_device());
    TORCH_CHECK(props->major == 10 && props->minor == 7 && props->multiProcessorCount == kCtas,
        "ml4_megakernel requires SM107 with 212 SMs");
}

void checkPackedParams(at::Tensor const& params)
{
    CHECK_TH_CUDA(params);
    TORCH_CHECK(params.scalar_type() == at::kLong && params.dim() == 1 && params.is_contiguous()
            && params.numel() == mega::ml4LayerNumParams() + kRoleWordsInt64 + kWeightWords,
        "ml4_megakernel: params must contain the int64 parameter block, role tables, and weight table");
}

void rejectMegakernelFunctionalization(c10::OperatorHandle const&, torch::jit::Stack*)
{
    TORCH_CHECK(
        false, "ml4_megakernel uses packed device pointers and multicast state; functionalization is unsupported");
}
} // namespace

std::vector<int64_t> ml4MegakernelInfo()
{
    return {mega::ml4LayerMagic(), mega::ml4LayerNumParams(), mega::ml4LayerArenaBytes(), mega::ml4LayerClearUnits(),
        mega::ml4LayerSmemBytes(), mega::ml4LayerNumCounters(), mega::ml4LayerHopBytes(), mega::ml4LayerTkBytes(),
        mega::ml4LayerIdxHistBytes(), mega::ml4LayerMaxKeys(), mega::ml4LayerIdxBndBytes(),
        mega::ml4LayerIdxArenaBytes(), mega::ml4LayerIdxLogitsBytes(), mega::ml4LayerIdxSmemBytes()};
}

at::Tensor ml4MegakernelIndexerTmap(at::Tensor const& pool, int64_t rows, int64_t tokensPerBlock)
{
    checkMegakernelDevice(pool);
    TORCH_CHECK(
        tokensPerBlock == 32 || tokensPerBlock == 64, "ml4_megakernel_indexer_tmap: tokens_per_block must be 32 or 64");
    int64_t const availableBytes = pool.storage().nbytes() - pool.storage_offset() * pool.element_size();
    TORCH_CHECK(rows > 0 && rows <= availableBytes / 128,
        "ml4_megakernel_indexer_tmap: pool storage must contain rows of 128 bytes");
    c10::cuda::CUDAGuard const deviceGuard(pool.device());
    auto out = at::empty({128}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    CUtensorMap tensorMap{};
    auto const result = mega::ml4LayerIdxTmap(&tensorMap, pool.data_ptr(), rows, static_cast<int>(tokensPerBlock));
    TORCH_CHECK(result == CUDA_SUCCESS, "ml4_megakernel_indexer_tmap: CUDA driver error ", result);
    std::copy_n(reinterpret_cast<uint8_t const*>(&tensorMap), sizeof(tensorMap), out.data_ptr<uint8_t>());
    return out;
}

void ml4MegakernelWeightTable(at::Tensor& params, int64_t offset)
{
    checkMegakernelDevice(params);
    checkPackedParams(params);
    TORCH_CHECK(offset == mega::ml4LayerNumParams() + kRoleWordsInt64,
        "ml4_megakernel_weight_table: invalid weight table offset");
    c10::cuda::CUDAGuard const deviceGuard(params.device());
    C10_CUDA_CHECK(static_cast<cudaError_t>(mega::ml4LayerInit()));
    C10_CUDA_CHECK(static_cast<cudaError_t>(mega::ml4LayerWeightTable(
        params.data_ptr(), params.data_ptr<int64_t>() + offset, at::cuda::getCurrentCUDAStream(params.get_device()))));
}

at::Tensor ml4MegakernelSmDieProbe(at::Tensor const& die0Buf, at::Tensor const& die1Buf)
{
    checkMegakernelDevice(die0Buf);
    TORCH_CHECK(
        die1Buf.device() == die0Buf.device() && die0Buf.nbytes() >= 256 * 1024 && die1Buf.nbytes() >= 256 * 1024,
        "ml4_megakernel_sm_die_probe: buffers must reside on the same SM107 device with at least 256 KiB each");
    c10::cuda::CUDAGuard const deviceGuard(die0Buf.device());
    auto out = at::zeros({kCtas, 3}, at::TensorOptions().dtype(at::kLong).device(die0Buf.device()));
    C10_CUDA_CHECK(static_cast<cudaError_t>(mega::ml4LayerSmDieProbe(
        die0Buf.data_ptr(), die1Buf.data_ptr(), out.data_ptr(), at::cuda::getCurrentCUDAStream(die0Buf.get_device()))));
    return out;
}

std::tuple<at::Tensor, at::Tensor> ml4Megakernel(at::Tensor const& params, at::Tensor const& x,
    at::Tensor const& residual, at::Tensor& topk, at::Tensor const& blockTable, at::Tensor const& kvLens,
    at::Tensor const& positions, std::vector<at::Tensor> const& state, std::vector<at::Tensor> const& kvCache, bool pdl,
    bool isLeader)
{
    checkMegakernelDevice(x);
    checkPackedParams(params);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) == kRows && x.size(1) == kHidden
            && x.is_contiguous(),
        "ml4_megakernel: x must be contiguous BF16 [4, 6144]");
    TORCH_CHECK(residual.device() == x.device() && residual.scalar_type() == x.scalar_type()
            && residual.sizes() == x.sizes() && residual.is_contiguous() && params.device() == x.device(),
        "ml4_megakernel: residual and params must match the input device and residual shape");
    TORCH_CHECK(topk.device() == x.device() && topk.scalar_type() == at::kInt && topk.dim() == 2
            && topk.size(0) == kRows && topk.size(1) == kTopK && topk.is_contiguous(),
        "ml4_megakernel: topk must be contiguous int32 [4, 2048]");
    TORCH_CHECK(blockTable.device() == x.device() && blockTable.scalar_type() == at::kInt && blockTable.dim() == 1
            && blockTable.numel() > 0 && blockTable.is_contiguous(),
        "ml4_megakernel: block_table must be a contiguous int32 row");
    TORCH_CHECK(kvLens.device() == x.device() && kvLens.scalar_type() == at::kInt && kvLens.numel() == 1
            && kvLens.is_contiguous(),
        "ml4_megakernel: kv_lens must contain one int32 request length");
    TORCH_CHECK(positions.device() == x.device() && positions.scalar_type() == at::kInt && positions.dim() == 1
            && positions.numel() == kRows && positions.is_contiguous(),
        "ml4_megakernel: positions must be contiguous int32 [4]");
    TORCH_CHECK(!state.empty() && !kvCache.empty(), "ml4_megakernel: mutable state and KV pools must be supplied");
    for (auto const& tensor : state)
    {
        TORCH_CHECK(tensor.device() == x.device() || (tensor.is_cpu() && tensor.is_pinned()),
            "ml4_megakernel: state must be on the input device or pinned host memory");
    }
    for (auto const& tensor : kvCache)
    {
        TORCH_CHECK(tensor.device() == x.device(), "ml4_megakernel: KV pools must be on the input device");
    }
    c10::cuda::CUDAGuard const deviceGuard(x.device());
    auto xOut = at::empty_like(x);
    auto resOut = at::empty_like(residual);
    C10_CUDA_CHECK(static_cast<cudaError_t>(mega::ml4LayerLaunch(params.data_ptr(), x.data_ptr(), residual.data_ptr(),
        xOut.data_ptr(), resOut.data_ptr(), topk.data_ptr(), blockTable.data_ptr(), kvLens.data_ptr(),
        positions.data_ptr(), at::cuda::getCurrentCUDAStream(x.get_device()), pdl ? 1 : 0, isLeader ? 1 : 0)));
    return {xOut, resOut};
}
} // namespace torch_ext

ML4_MEGA_NS_END

ML4_MEGA_LIBRARY(m)
{
    m.def("ml4_megakernel_info() -> int[]");
    m.def("ml4_megakernel_indexer_tmap(Tensor pool, int rows, int tokens_per_block) -> Tensor");
    m.def("ml4_megakernel_weight_table(Tensor(a!) params, int offset) -> ()");
    m.def("ml4_megakernel_sm_die_probe(Tensor die0_buf, Tensor die1_buf) -> Tensor");
    m.def(
        "ml4_megakernel(Tensor params, Tensor x, Tensor residual, Tensor(a!) topk, Tensor block_table, "
        "Tensor kv_lens, Tensor positions, Tensor(b!)[] state, Tensor(c!)[] kv_cache, bool pdl, bool is_leader=False) "
        "-> (Tensor, Tensor)");
}

ML4_MEGA_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl("ml4_megakernel_info", &ML4_MEGA_NS::torch_ext::ml4MegakernelInfo);
}

ML4_MEGA_LIBRARY_IMPL(CUDA, m)
{
    m.impl("ml4_megakernel_indexer_tmap", &ML4_MEGA_NS::torch_ext::ml4MegakernelIndexerTmap);
    m.impl("ml4_megakernel_weight_table", &ML4_MEGA_NS::torch_ext::ml4MegakernelWeightTable);
    m.impl("ml4_megakernel_sm_die_probe", &ML4_MEGA_NS::torch_ext::ml4MegakernelSmDieProbe);
    m.impl("ml4_megakernel", &ML4_MEGA_NS::torch_ext::ml4Megakernel);
}

ML4_MEGA_LIBRARY_IMPL(Functionalize, m)
{
    m.impl("ml4_megakernel",
        torch::CppFunction::makeFromBoxedFunction<&ML4_MEGA_NS::torch_ext::rejectMegakernelFunctionalization>());
    m.impl("ml4_megakernel_weight_table",
        torch::CppFunction::makeFromBoxedFunction<&ML4_MEGA_NS::torch_ext::rejectMegakernelFunctionalization>());
}
