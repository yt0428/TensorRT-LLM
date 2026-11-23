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

#include "kv_cache_manager_v2/copyEngine.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/diskIoChannel.h"
#include "kv_cache_manager_v2/exceptions.h"

// Reuse existing copy implementations (no Python round-trip).
#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"

#include "tensorrt_llm/common/assert.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// dispatchCopy<DstTier, SrcTier> — template that maps CacheTier pair to
// address types and the underlying copy function via if-constexpr.
// Replaces 7 near-identical static dispatcher functions.
// ---------------------------------------------------------------------------

template <CacheTier Tier>
using TierAddr = std::conditional_t<Tier == CacheTier::DISK, DiskAddress, MemAddress>;

template <CacheTier DstTier, CacheTier SrcTier>
static void dispatchCopy(std::vector<CopyTask> const& tasks, size_t numBytes, CUstream stream)
{
    using DstAddr = TierAddr<DstTier>;
    using SrcAddr = TierAddr<SrcTier>;

    std::vector<Task<DstAddr, SrcAddr>> t;
    t.reserve(tasks.size());
    for (auto const& task : tasks)
        t.push_back({std::get<DstAddr>(task.dst), std::get<SrcAddr>(task.src)});

    constexpr auto G = CacheTier::GPU_MEM;
    constexpr auto H = CacheTier::HOST_MEM;
    constexpr auto D = CacheTier::DISK;

    // clang-format off
    if constexpr      (DstTier == G && SrcTier == G) cuCheck(copyDeviceToDevice(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == G && SrcTier == H) cuCheck(copyHostToDevice(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == H && SrcTier == G) cuCheck(copyDeviceToHost(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == H && SrcTier == H) cuCheck(copyHostToHost(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == D && SrcTier == D) cuCheck(copyDiskToDisk(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == H && SrcTier == D) cuCheck(copyDiskToHost(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == D && SrcTier == H) cuCheck(copyHostToDisk(std::move(t), static_cast<ssize_t>(numBytes), stream));
    // clang-format on
}

// ---------------------------------------------------------------------------
// CopyEngine
// ---------------------------------------------------------------------------

CopyEngine::CopyEngine(StagingBufferManager* pageStagingManager, DiskIoChannel* diskIo)
    : mPageStagingManager(pageStagingManager)
    , mDiskIo(diskIo)
{
}

StagingBufferManager& CopyEngine::pageStagingManager() const
{
    TLLM_CHECK_WITH_INFO(mPageStagingManager != nullptr, "CopyEngine requires page staging for this transfer");
    return *mPageStagingManager;
}

DiskIoTicket CopyEngine::submitDisk(CacheTier dstTier, CacheTier srcTier, size_t numBytes,
    std::vector<CopyTask> const& tasks, size_t begin, size_t end, CUstream stream, bool fence)
{
    TLLM_CHECK_WITH_INFO(mDiskIo != nullptr, "CopyEngine requires the disk I/O channel for this transfer");
    constexpr auto H = CacheTier::HOST_MEM;
    constexpr auto D = CacheTier::DISK;
    DiskIoKind kind{};
    if (dstTier == D && srcTier == H)
        kind = DiskIoKind::kWrite;
    else if (dstTier == H && srcTier == D)
        kind = DiskIoKind::kRead;
    else if (dstTier == D && srcTier == D)
        kind = DiskIoKind::kCopy;
    else
        throw std::invalid_argument("CopyEngine::submitDisk: not a host/disk tier pair");

    std::vector<DiskIoOp> ops;
    ops.reserve(end - begin);
    for (size_t i = begin; i < end; ++i)
    {
        auto const& task = tasks[i];
        DiskIoOp op;
        op.size = numBytes;
        switch (kind)
        {
        case DiskIoKind::kWrite:
        {
            auto const dst = std::get<DiskAddress>(task.dst);
            op.fd = dst.fd;
            op.offset = dst.pos;
            op.buffer = reinterpret_cast<std::byte*>(std::get<MemAddress>(task.src));
            break;
        }
        case DiskIoKind::kRead:
        {
            auto const src = std::get<DiskAddress>(task.src);
            op.fd = src.fd;
            op.offset = src.pos;
            op.buffer = reinterpret_cast<std::byte*>(std::get<MemAddress>(task.dst));
            break;
        }
        case DiskIoKind::kCopy:
        {
            auto const dst = std::get<DiskAddress>(task.dst);
            auto const src = std::get<DiskAddress>(task.src);
            op.fd = dst.fd;
            op.offset = dst.pos;
            op.srcFd = src.fd;
            op.srcOffset = src.pos;
            break;
        }
        }
        ops.push_back(op);
    }
    return mDiskIo->enqueue(stream, kind, std::move(ops), fence);
}

// Two-hop transfer via host staging buffer (e.g., GPU→Disk or Disk→GPU).
// SrcTier → MidTier (staging) → DstTier.  MidTier is the staging tier (HOST_MEM).
template <CacheTier DstTier, CacheTier MidTier, CacheTier SrcTier>
void CopyEngine::twoHopTransfer(
    size_t numBytes, std::vector<CopyTask> const& tasks, CUstream stream, std::vector<DiskIoTicket>* taskTickets)
{
    constexpr auto D = CacheTier::DISK;
    auto& manager = pageStagingManager();
    size_t remaining = tasks.size();
    size_t offset = 0;

    while (remaining > 0)
    {
        size_t const maxSize = remaining > std::numeric_limits<size_t>::max() / numBytes
            ? std::numeric_limits<size_t>::max()
            : numBytes * remaining;
        // O_DIRECT needs the staging pages aligned, and the disk file is opened that way only when numBytes
        // is a multiple of kDiskIoAlignment. The size granularity must be a multiple of the alignment.
        size_t const alignment = numBytes % kDiskIoAlignment == 0 ? kDiskIoAlignment : 1;
        StagingBuffer buf = manager.acquire(numBytes, maxSize, numBytes, alignment, stream);
        MemAddress addr = buf.address();
        size_t n = buf.size() / numBytes;
        TLLM_CHECK_DEBUG(n > 0 && n <= remaining);
        DiskIoTicket ticket = kNoDiskIoTicket;

        // First hop: src → staging
        {
            std::vector<CopyTask> hop1;
            hop1.reserve(n);
            for (size_t i = 0; i < n; ++i)
                hop1.push_back({Address{addr + numBytes * i}, tasks[offset + i].src});
            if (SrcTier == D && mDiskIo != nullptr)
                ticket = submitDisk(MidTier, SrcTier, numBytes, hop1, 0, n, *buf.stream());
            else
                dispatchCopy<MidTier, SrcTier>(hop1, numBytes, *buf.stream());
        }

        // Second hop: staging → dst
        {
            std::vector<CopyTask> hop2;
            hop2.reserve(n);
            for (size_t i = 0; i < n; ++i)
                hop2.push_back({tasks[offset + i].dst, Address{addr + numBytes * i}});
            if (DstTier == D && mDiskIo != nullptr)
                ticket = submitDisk(DstTier, MidTier, numBytes, hop2, 0, n, *buf.stream());
            else
                dispatchCopy<DstTier, MidTier>(hop2, numBytes, *buf.stream());
        }

        if (taskTickets != nullptr && ticket != kNoDiskIoTicket)
        {
            taskTickets->insert(taskTickets->end(), n, ticket);
        }
        offset += n;
        remaining -= n;
    }
}

void CopyEngine::transfer(CacheTier dstTier, CacheTier srcTier, size_t numBytes, std::vector<CopyTask> const& tasks,
    CUstream stream, std::vector<DiskIoTicket>* taskTickets, bool fenceDisk)
{
    // Nothing to copy. Also guards the two-hop path against a div-by-zero on
    // `buf.size() / numBytes` when numBytes == 0.
    if (tasks.empty() || numBytes == 0)
    {
        return;
    }

    constexpr auto G = CacheTier::GPU_MEM;
    constexpr auto H = CacheTier::HOST_MEM;
    constexpr auto D = CacheTier::DISK;

    bool const hostDiskPair = (dstTier == D || srcTier == D) && dstTier != G && srcTier != G;
    if (hostDiskPair && mDiskIo != nullptr)
    {
        DiskIoTicket const ticket = submitDisk(dstTier, srcTier, numBytes, tasks, 0, tasks.size(), stream, fenceDisk);
        if (taskTickets != nullptr)
        {
            taskTickets->insert(taskTickets->end(), tasks.size(), ticket);
        }
        return;
    }

    if (dstTier == G && srcTier == G)
        dispatchCopy<G, G>(tasks, numBytes, stream);
    else if (dstTier == G && srcTier == H)
        dispatchCopy<G, H>(tasks, numBytes, stream);
    else if (dstTier == G && srcTier == D)
        twoHopTransfer<G, H, D>(numBytes, tasks, stream, taskTickets);
    else if (dstTier == H && srcTier == G)
        dispatchCopy<H, G>(tasks, numBytes, stream);
    else if (dstTier == H && srcTier == H)
        dispatchCopy<H, H>(tasks, numBytes, stream);
    else if (dstTier == H && srcTier == D)
        dispatchCopy<H, D>(tasks, numBytes, stream);
    else if (dstTier == D && srcTier == G)
        twoHopTransfer<D, H, G>(numBytes, tasks, stream, taskTickets);
    else if (dstTier == D && srcTier == H)
        dispatchCopy<D, H>(tasks, numBytes, stream);
    else if (dstTier == D && srcTier == D)
        dispatchCopy<D, D>(tasks, numBytes, stream);
    else
        throw std::invalid_argument("CopyEngine::transfer: unsupported tier combination");
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
