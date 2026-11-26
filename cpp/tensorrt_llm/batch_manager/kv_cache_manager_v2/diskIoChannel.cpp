/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kv_cache_manager_v2/diskIoChannel.h"
#include "kv_cache_manager_v2/exceptions.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <chrono>
#include <cstring>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

struct HostFnArgs
{
    DiskIoEngine* engine;
    DiskIoTicket ticket;
};

CUDA_CB void activateHostFn(void* userData) noexcept
{
    std::unique_ptr<HostFnArgs> const args(static_cast<HostFnArgs*>(userData));
    args->engine->activate(args->ticket);
}

CUDA_CB void waitHostFn(void* userData) noexcept
{
    std::unique_ptr<HostFnArgs> const args(static_cast<HostFnArgs*>(userData));
    args->engine->wait(args->ticket);
}

//! Whether this context accepts cuStreamWaitValue32 on host-mapped memory. Tried rather than read
//! from a device attribute, because the attribute names moved between CUDA versions.
bool probeStreamMemOps(CUdeviceptr word)
{
    CUstream stream = nullptr;
    if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS)
    {
        return false;
    }
    // The word holds 0, so a wait for >= 0 is already satisfied.
    bool const supported = cuStreamWaitValue32(stream, word, 0, CU_STREAM_WAIT_VALUE_GEQ) == CUDA_SUCCESS
        && cuStreamSynchronize(stream) == CUDA_SUCCESS;
    cuStreamDestroy(stream);
    return supported;
}

constexpr std::chrono::milliseconds kDestroyDrainTimeout{60'000};

} // namespace

DiskIoChannel::DiskIoChannel(DiskIoEngine::Options const& options)
{
    size_t const bytes = sizeof(uint32_t) * options.ringSize;
    void* words = nullptr;
    cuCheck(cuMemHostAlloc(&words, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
    std::memset(words, 0, bytes);
    mWords = static_cast<uint32_t*>(words);
    try
    {
        cuCheck(cuMemHostGetDevicePointer(&mWordsDevice, words, 0));
        mStreamMemOps = probeStreamMemOps(mWordsDevice);
        if (!mStreamMemOps)
        {
            TLLM_LOG_WARNING(
                "KV cache manager v2 disk I/O: CUDA stream memory operations are unavailable, so streams wait for "
                "disk I/O in host functions, which serializes those waits.");
        }
        mEngine = std::make_unique<DiskIoEngine>(options, mWords);
    }
    catch (...)
    {
        cuMemFreeHost(words);
        mWords = nullptr;
        throw;
    }
}

DiskIoChannel::~DiskIoChannel()
{
    destroy();
}

DiskIoTicket DiskIoChannel::enqueue(CUstream stream, DiskIoKind kind, std::vector<DiskIoOp> ops, bool fence)
{
    TLLM_CHECK_WITH_INFO(mEngine != nullptr, "DiskIoChannel used after destroy()");
    DiskIoPriority const priority
        = (kind == DiskIoKind::kRead || fence) ? DiskIoPriority::kRequired : DiskIoPriority::kBackground;
    DiskIoTicket const ticket = mEngine->submit(kind, std::move(ops), /*activate=*/false, priority);

    auto activateArgs = std::make_unique<HostFnArgs>(HostFnArgs{mEngine.get(), ticket});
    if (CUresult const result = cuLaunchHostFunc(stream, activateHostFn, activateArgs.get()); result != CUDA_SUCCESS)
    {
        // The batch must still complete, or drain() would wait for it forever. The stream is broken at
        // this point, so ordering against it no longer means anything.
        mEngine->activate(ticket);
        mEngine->wait(ticket);
        cuCheck(result);
    }
    activateArgs.release();

    if (!fence)
    {
        return ticket;
    }
    if (mStreamMemOps)
    {
        CUdeviceptr const word = mWordsDevice + sizeof(uint32_t) * (ticket & (mEngine->ringSize() - 1));
        cuCheck(cuStreamWaitValue32(stream, word, ticket, CU_STREAM_WAIT_VALUE_GEQ));
    }
    else
    {
        auto waitArgs = std::make_unique<HostFnArgs>(HostFnArgs{mEngine.get(), ticket});
        cuCheck(cuLaunchHostFunc(stream, waitHostFn, waitArgs.get()));
        waitArgs.release();
    }
    return ticket;
}

void DiskIoChannel::destroy()
{
    if (mEngine == nullptr)
    {
        return;
    }
    if (!mEngine->drainFor(kDestroyDrainTimeout))
    {
        TLLM_LOG_ERROR(
            "KV cache manager v2 disk I/O: batches still pending at shutdown; leaking the I/O engine and its "
            "completion words instead of freeing them under a pending wait.");
        static_cast<void>(mEngine.release());
        mWords = nullptr;
        return;
    }
    // Every batch has published its ticket, but a stream may not have executed its wait on the word yet.
    // The words must outlive those waits.
    CUresult const synced = cuCtxSynchronize();
    mEngine.reset();
    if (synced == CUDA_SUCCESS)
    {
        cuMemFreeHost(mWords);
    }
    // Otherwise the context is gone or broken, and its host allocations with it.
    mWords = nullptr;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
