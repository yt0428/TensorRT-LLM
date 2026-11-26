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

#pragma once

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/diskIoEngine.h"
#include "kv_cache_manager_v2/stagingBuffer.h"

#include <cstddef>
#include <cuda.h>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class DiskIoChannel;

struct CopyTask
{
    Address dst;
    Address src;
};

// ---------------------------------------------------------------------------
// CopyEngine — dispatches bulk transfers between cache tiers.
//
// Single-hop memory pairs call the appropriate copy function from kvCacheManagerV2Utils. Pairs that
// touch disk are queued on the disk I/O channel, ordered with the stream. Two-hop pairs (GPU↔Disk)
// route through a pinned-host page ring borrowed from StorageManager.
// ---------------------------------------------------------------------------
class CopyEngine
{
public:
    // Both are non-owning and may be null when no cold tier needs them.
    CopyEngine(StagingBufferManager* pageStagingManager, DiskIoChannel* diskIo);
    ~CopyEngine() = default;

    CopyEngine(CopyEngine const&) = delete;
    CopyEngine& operator=(CopyEngine const&) = delete;

    // Transfer numBytes per task. Tasks must all share the same source and destination tiers.
    // When the transfer touches disk and taskTickets is non-null, appends the disk ticket of each task, in
    // task order; a batch that fails still completes on the stream, and only its ticket reports the failure.
    // Without fenceDisk, a host/disk transfer leaves `stream` free to run ahead of the I/O (see
    // DiskIoChannel::enqueue); two-hop transfers always fence, since GPU work consumes or produces them.
    void transfer(CacheTier dstTier, CacheTier srcTier, size_t numBytes, std::vector<CopyTask> const& tasks,
        CUstream stream, std::vector<DiskIoTicket>* taskTickets = nullptr, bool fenceDisk = true);

private:
    [[nodiscard]] StagingBufferManager& pageStagingManager() const;

    // Queues tasks[begin, end) of a host/disk pair on the disk channel and returns its ticket.
    DiskIoTicket submitDisk(CacheTier dstTier, CacheTier srcTier, size_t numBytes, std::vector<CopyTask> const& tasks,
        size_t begin, size_t end, CUstream stream, bool fence = true);

    template <CacheTier DstTier, CacheTier MidTier, CacheTier SrcTier>
    void twoHopTransfer(
        size_t numBytes, std::vector<CopyTask> const& tasks, CUstream stream, std::vector<DiskIoTicket>* taskTickets);

    // Non-owning; null when every cold tier is GPU or host memory.
    StagingBufferManager* mPageStagingManager;
    // Non-owning; null without a disk tier.
    DiskIoChannel* mDiskIo;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
