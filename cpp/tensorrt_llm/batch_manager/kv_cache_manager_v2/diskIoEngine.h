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

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Address and size alignment that O_DIRECT needs on the filesystems KVCM2 targets. 4 KiB covers both
//! 512-byte and 4 KiB logical block sizes.
inline constexpr size_t kDiskIoAlignment = 4096;

enum class DiskIoKind : uint8_t
{
    //! (fd, offset) -> buffer.
    kRead,
    //! buffer -> (fd, offset).
    kWrite,
    //! (srcFd, srcOffset) -> (fd, offset), through a worker-owned aligned bounce buffer.
    kCopy,
};

//! One page-sized transfer. A batch holds any number of them; workers take them one at a time, so a
//! batch spreads over every free worker.
struct DiskIoOp
{
    int fd = -1;
    int64_t offset = 0;
    //! Host memory the operation reads into or writes from. Unused by kCopy.
    std::byte* buffer = nullptr;
    size_t size = 0;
    //! kCopy only.
    int srcFd = -1;
    int64_t srcOffset = 0;
};

//! Identifies one submitted batch. Never 0, so 0 can mean "no ticket".
using DiskIoTicket = uint32_t;
inline constexpr DiskIoTicket kNoDiskIoTicket = 0;

enum class DiskIoStatus : uint8_t
{
    kPending,
    kOk,
    kFailed,
};

enum class DiskIoPriority : uint8_t
{
    //! Latency-critical transfers: reads, and fenced writes that a GPU stream or read waits on.
    kRequired,
    //! Best-effort background transfers such as unfenced host-reserve writes.
    kBackground,
};

//! Counters since the last DiskIoEngine::takeStats(). Busy times are summed over workers, so they can
//! exceed the wall time of the window.
struct DiskIoStats
{
    uint64_t readBytes = 0;
    uint64_t writeBytes = 0;
    //! Subset of writeBytes from kRequired (fenced) write and copy operations.
    uint64_t requiredWriteBytes = 0;
    uint64_t readOps = 0;
    uint64_t writeOps = 0;
    uint64_t readBusyNs = 0;
    uint64_t writeBusyNs = 0;
    uint64_t readErrors = 0;
    uint64_t writeErrors = 0;
    //! Read batches completed, and their activation-to-completion latency.
    uint64_t readBatches = 0;
    uint64_t readBatchLatencyNsSum = 0;
    uint64_t readBatchLatencyNsMax = 0;
    //! Operations queued but not yet started, at the time of the call and at their peak.
    uint64_t queuedReadOps = 0;
    uint64_t queuedWriteOps = 0;
    uint64_t peakQueuedReadOps = 0;
    uint64_t peakQueuedWriteOps = 0;
    //! Bytes of write and copy batches submitted but not completed, at the time of the call and at their peak.
    uint64_t pendingWriteBytes = 0;
    uint64_t peakPendingWriteBytes = 0;
};

//! Multi-threaded disk I/O with separate read and write queues.
//!
//! A batch is submitted inactive or active. Only active batches run; activate() may be called from any
//! thread, including a CUDA host function, and only takes a mutex. That split lets a caller submit on
//! the CPU and start the I/O when a CUDA stream reaches a point that makes the buffers ready.
//!
//! On completion the engine stores the batch's ticket into completionWords[ticket % ringSize] with
//! release ordering. Those words may live in host-mapped memory that a GPU stream waits on; a ticket
//! is published only after every operation of its batch has returned. A batch that fails still
//! completes, with status kFailed: the word tells a waiter that the buffers are no longer in use, not
//! that the data is good.
//!
//! Workers prefer reads, except numWritePreferringThreads of them, which prefer writes. Every worker
//! falls back to the other queue when its own is empty.
//!
//! Thread safety: every public method may be called concurrently.
class DiskIoEngine
{
public:
    struct Options
    {
        int numThreads = 8;
        int numWritePreferringThreads = 1;
        //! Batches that may be in flight at once. A power of two; submit() blocks while all are in use.
        uint32_t ringSize = 4096;
    };

    //! completionWords must hold options.ringSize words and outlive the engine. The engine never reads
    //! them; their initial value only has to be below every ticket a waiter waits for, which 0 is.
    DiskIoEngine(Options const& options, uint32_t* completionWords);

    //! Drains every submitted batch, then joins the workers. Every inactive batch must be activated
    //! eventually, or this blocks.
    ~DiskIoEngine();

    DiskIoEngine(DiskIoEngine const&) = delete;
    DiskIoEngine& operator=(DiskIoEngine const&) = delete;

    //! Queues a batch and returns its ticket. A batch with no operations completes when it is activated.
    DiskIoTicket submit(
        DiskIoKind kind, std::vector<DiskIoOp> ops, bool activate, DiskIoPriority priority = DiskIoPriority::kRequired);

    //! Makes a batch submitted inactive runnable. Activating twice, or an unknown ticket, terminates.
    void activate(DiskIoTicket ticket) noexcept;

    //! Status of a ticket this engine issued. Tickets whose ring entry was reused report their final
    //! status, which is kept for failures.
    [[nodiscard]] DiskIoStatus query(DiskIoTicket ticket) const;

    //! Blocks until the ticket's batch completes and returns kOk or kFailed.
    DiskIoStatus wait(DiskIoTicket ticket) const;

    //! errno of the first failed operation of a completed batch, or 0.
    [[nodiscard]] int error(DiskIoTicket ticket) const;

    //! Blocks until every batch submitted so far completes.
    void drain() const;

    //! drain() with a time limit. Returns whether everything completed.
    [[nodiscard]] bool drainFor(std::chrono::milliseconds timeout) const;

    [[nodiscard]] uint64_t pendingWriteBytes() const;

    //! Returns the counters and resets them; the queue depths in the result are current values.
    DiskIoStats takeStats();

    [[nodiscard]] uint32_t ringSize() const noexcept
    {
        return mRingSize;
    }

    //! Makes the next `count` operations of `kind` fail with errnoValue without touching the file.
    //! For tests.
    void injectFaults(DiskIoKind kind, int count, int errnoValue);

    //! While paused, workers start no new operation; queued batches wait. For tests that need two
    //! batches queued before either starts.
    void setPausedForTest(bool paused);

private:
    struct Batch
    {
        DiskIoTicket ticket = kNoDiskIoTicket;
        DiskIoKind kind = DiskIoKind::kRead;
        DiskIoPriority priority = DiskIoPriority::kRequired;
        std::vector<DiskIoOp> ops;
        size_t nextOp = 0;
        size_t remainingOps = 0;
        uint64_t bytes = 0;
        int error = 0;
        bool active = false;
        bool done = true;
        int64_t activateNs = 0;
    };

    [[nodiscard]] Batch& entry(DiskIoTicket ticket) noexcept
    {
        return mRing[ticket & (mRingSize - 1)];
    }

    [[nodiscard]] Batch const& entry(DiskIoTicket ticket) const noexcept
    {
        return mRing[ticket & (mRingSize - 1)];
    }

    [[nodiscard]] static bool isWriteLike(DiskIoKind kind) noexcept
    {
        return kind != DiskIoKind::kRead;
    }

    [[nodiscard]] bool hasWorkLocked() const noexcept
    {
        return !mReadQueue.empty() || !mRequiredWriteQueue.empty() || !mBackgroundWriteQueue.empty();
    }

    [[nodiscard]] std::deque<DiskIoTicket>& selectQueueLocked(bool preferWrites);
    [[nodiscard]] DiskIoStatus statusLocked(DiskIoTicket ticket) const;
    void enqueueLocked(Batch& batch);
    void completeLocked(Batch& batch);
    void workerLoop(bool preferWrites);
    int runOp(DiskIoKind kind, DiskIoOp const& op, std::vector<std::byte>& bounce, std::byte*& bounceAligned);

    static constexpr int64_t kRequiredWriteMaxWaitNs = 10'000'000;

    uint32_t const mRingSize;
    uint32_t* const mCompletionWords;

    mutable std::mutex mMutex;
    std::condition_variable mWorkCv;
    mutable std::condition_variable mDoneCv;

    std::vector<Batch> mRing;
    DiskIoTicket mNextTicket = 1;
    uint64_t mNumIncomplete = 0;
    std::deque<DiskIoTicket> mReadQueue;
    std::deque<DiskIoTicket> mRequiredWriteQueue;
    std::deque<DiskIoTicket> mBackgroundWriteQueue;
    bool mServedAgedWriteLast = false;
    std::unordered_set<DiskIoTicket> mFailedTickets;
    DiskIoStats mStats;
    int mFaultsLeft[3] = {0, 0, 0};
    int mFaultErrno[3] = {0, 0, 0};
    bool mPaused = false;
    bool mStop = false;

    std::vector<std::thread> mWorkers;
};

//! Reads or writes `size` bytes at `offset`, retrying on EINTR and short transfers. Returns 0, or the errno of
//! the failure; a read that reaches end of file before `size` bytes fails with EIO.
int diskReadAll(int fd, int64_t offset, std::byte* data, size_t size) noexcept;
int diskWriteAll(int fd, int64_t offset, std::byte const* data, size_t size) noexcept;

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
