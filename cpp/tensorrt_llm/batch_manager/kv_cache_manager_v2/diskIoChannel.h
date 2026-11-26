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

#include "kv_cache_manager_v2/diskIoEngine.h"

#include <cuda.h>
#include <memory>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Orders disk I/O with CUDA streams.
//!
//! enqueue(stream, ...) submits a batch to the engine inactive and activates it from a host function
//! once `stream` reaches that point, so the I/O starts only after the work the stream already waits
//! for: the source page, the destination slot and the staging range. Later work on `stream` then waits
//! until the batch completes. The host function only pushes the batch onto a queue; the I/O runs on
//! the engine's workers, so the driver's callback thread is never held by disk latency.
//!
//! A stream parked on a disk wait also stalls unrelated streams that share its hardware connection
//! (measured on VR200 with the default 8 connections), so the fence is for the rare transfers that
//! GPU work consumes directly; transfers between host and disk go unfenced.
//!
//! The completion wait is cuStreamWaitValue32 on host-mapped words that the engine updates. Where
//! stream memory operations are unavailable, a host function blocks until the batch completes
//! instead; that is still correct, but it holds the callback thread for the duration of the I/O.
//!
//! A batch completes even when it fails, so a stream never hangs on a disk error. Callers learn about
//! the failure through the ticket.
class DiskIoChannel
{
public:
    //! Needs a current CUDA context.
    explicit DiskIoChannel(DiskIoEngine::Options const& options);
    ~DiskIoChannel();

    DiskIoChannel(DiskIoChannel const&) = delete;
    DiskIoChannel& operator=(DiskIoChannel const&) = delete;

    //! Queues `ops` behind the work already on `stream`. With `fence`, the work queued on `stream`
    //! afterwards also waits for them; without it, nothing on the GPU waits, and the caller orders
    //! every later use of the buffers and slots through the ticket instead.
    DiskIoTicket enqueue(CUstream stream, DiskIoKind kind, std::vector<DiskIoOp> ops, bool fence = true);

    [[nodiscard]] DiskIoEngine& engine() const
    {
        return *mEngine;
    }

    [[nodiscard]] bool usesStreamMemOps() const noexcept
    {
        return mStreamMemOps;
    }

    //! Waits for every queued batch and for the device to pass every wait on the completion words,
    //! then releases the workers and the words. Idempotent. If the queued batches do not complete in
    //! time (a stream that never reaches its activation), the engine and the words are deliberately
    //! leaked rather than freed under a pending wait.
    void destroy();

private:
    uint32_t* mWords = nullptr;
    CUdeviceptr mWordsDevice = 0;
    bool mStreamMemOps = false;
    std::unique_ptr<DiskIoEngine> mEngine;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
