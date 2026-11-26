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

#include "kv_cache_manager_v2/diskIoEngine.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <exception>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

int64_t steadyClockNanos() noexcept
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

size_t kindIndex(DiskIoKind kind) noexcept
{
    return static_cast<size_t>(kind);
}

} // namespace

int diskReadAll(int fd, int64_t offset, std::byte* data, size_t size) noexcept
{
    size_t done = 0;
    while (done < size)
    {
        ssize_t const n
            = ::pread(fd, data + done, size - done, static_cast<off_t>(offset + static_cast<int64_t>(done)));
        if (n > 0)
        {
            done += static_cast<size_t>(n);
            continue;
        }
        if (n == 0)
        {
            // End of file before the page was complete: the slot lies outside the file.
            return EIO;
        }
        if (errno == EINTR)
        {
            continue;
        }
        return errno;
    }
    return 0;
}

int diskWriteAll(int fd, int64_t offset, std::byte const* data, size_t size) noexcept
{
    size_t done = 0;
    while (done < size)
    {
        ssize_t const n
            = ::pwrite(fd, data + done, size - done, static_cast<off_t>(offset + static_cast<int64_t>(done)));
        if (n > 0)
        {
            done += static_cast<size_t>(n);
            continue;
        }
        if (n == 0)
        {
            return EIO;
        }
        if (errno == EINTR)
        {
            continue;
        }
        return errno;
    }
    return 0;
}

DiskIoEngine::DiskIoEngine(Options const& options, uint32_t* completionWords)
    : mRingSize(options.ringSize)
    , mCompletionWords(completionWords)
{
    if (options.numThreads < 1)
    {
        throw std::invalid_argument("DiskIoEngine: numThreads must be at least 1");
    }
    if (options.numWritePreferringThreads < 0 || options.numWritePreferringThreads > options.numThreads)
    {
        throw std::invalid_argument("DiskIoEngine: numWritePreferringThreads must be within [0, numThreads]");
    }
    if (mRingSize < 2 || (mRingSize & (mRingSize - 1)) != 0)
    {
        throw std::invalid_argument("DiskIoEngine: ringSize must be a power of two of at least 2");
    }
    if (mCompletionWords == nullptr)
    {
        throw std::invalid_argument("DiskIoEngine: completionWords must not be null");
    }
    mRing.resize(mRingSize);
    mWorkers.reserve(static_cast<size_t>(options.numThreads));
    try
    {
        for (int i = 0; i < options.numThreads; ++i)
        {
            bool const preferWrites = i < options.numWritePreferringThreads;
            mWorkers.emplace_back([this, preferWrites]() { workerLoop(preferWrites); });
            pthread_setname_np(mWorkers.back().native_handle(), "kvcm2-disk-io");
        }
    }
    catch (...)
    {
        {
            std::lock_guard<std::mutex> const lock(mMutex);
            mStop = true;
        }
        mWorkCv.notify_all();
        for (auto& worker : mWorkers)
        {
            worker.join();
        }
        throw;
    }
}

DiskIoEngine::~DiskIoEngine()
{
    drain();
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        mStop = true;
    }
    mWorkCv.notify_all();
    for (auto& worker : mWorkers)
    {
        worker.join();
    }
}

DiskIoTicket DiskIoEngine::submit(DiskIoKind kind, std::vector<DiskIoOp> ops, bool activate, DiskIoPriority priority)
{
    uint64_t bytes = 0;
    for (auto const& op : ops)
    {
        bytes += op.size;
    }
    std::unique_lock<std::mutex> lock(mMutex);
    if (mStop)
    {
        throw std::logic_error("DiskIoEngine: submit after shutdown");
    }
    // The ring entry is free once its previous batch completed. Waiting here bounds the batches in
    // flight; it only blocks when ringSize batches are outstanding at once.
    mDoneCv.wait(lock, [this]() { return entry(mNextTicket).done; });
    DiskIoTicket const ticket = mNextTicket;
    mNextTicket = mNextTicket + 1 == kNoDiskIoTicket ? 1 : mNextTicket + 1;

    Batch& batch = entry(ticket);
    batch.ticket = ticket;
    batch.kind = kind;
    batch.priority = priority;
    batch.ops = std::move(ops);
    batch.nextOp = 0;
    batch.remainingOps = batch.ops.size();
    batch.bytes = bytes;
    batch.error = 0;
    batch.active = false;
    batch.done = false;
    batch.activateNs = 0;
    ++mNumIncomplete;
    if (isWriteLike(kind))
    {
        mStats.pendingWriteBytes += bytes;
        mStats.peakPendingWriteBytes = std::max(mStats.peakPendingWriteBytes, mStats.pendingWriteBytes);
    }
    if (activate)
    {
        enqueueLocked(batch);
    }
    return ticket;
}

void DiskIoEngine::activate(DiskIoTicket ticket) noexcept
{
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        Batch& batch = entry(ticket);
        if (batch.ticket != ticket || batch.active || batch.done)
        {
            // An activation that does not match an inactive batch means the stream bookkeeping is
            // broken; running the wrong batch would move data into buffers that are not ready.
            std::terminate();
        }
        enqueueLocked(batch);
    }
}

void DiskIoEngine::enqueueLocked(Batch& batch)
{
    batch.active = true;
    batch.activateNs = steadyClockNanos();
    if (batch.ops.empty())
    {
        completeLocked(batch);
        return;
    }
    if (batch.kind == DiskIoKind::kRead)
    {
        mReadQueue.push_back(batch.ticket);
        mStats.queuedReadOps += batch.ops.size();
        mStats.peakQueuedReadOps = std::max(mStats.peakQueuedReadOps, mStats.queuedReadOps);
    }
    else
    {
        auto& writeQueue = batch.priority == DiskIoPriority::kRequired ? mRequiredWriteQueue : mBackgroundWriteQueue;
        writeQueue.push_back(batch.ticket);
        mStats.queuedWriteOps += batch.ops.size();
        mStats.peakQueuedWriteOps = std::max(mStats.peakQueuedWriteOps, mStats.queuedWriteOps);
    }
    mWorkCv.notify_all();
}

void DiskIoEngine::completeLocked(Batch& batch)
{
    batch.done = true;
    if (isWriteLike(batch.kind))
    {
        mStats.pendingWriteBytes -= batch.bytes;
    }
    else
    {
        uint64_t const latency = static_cast<uint64_t>(std::max<int64_t>(0, steadyClockNanos() - batch.activateNs));
        ++mStats.readBatches;
        mStats.readBatchLatencyNsSum += latency;
        mStats.readBatchLatencyNsMax = std::max(mStats.readBatchLatencyNsMax, latency);
    }
    if (batch.error != 0)
    {
        mFailedTickets.insert(batch.ticket);
    }
    std::vector<DiskIoOp>().swap(batch.ops);
    --mNumIncomplete;
    __atomic_store_n(&mCompletionWords[batch.ticket & (mRingSize - 1)], batch.ticket, __ATOMIC_RELEASE);
    mDoneCv.notify_all();
}

DiskIoStatus DiskIoEngine::statusLocked(DiskIoTicket ticket) const
{
    Batch const& batch = entry(ticket);
    if (batch.ticket == ticket && !batch.done)
    {
        return DiskIoStatus::kPending;
    }
    // Either this batch completed, or its entry was reused, which happens only after it completed.
    return mFailedTickets.count(ticket) != 0 ? DiskIoStatus::kFailed : DiskIoStatus::kOk;
}

DiskIoStatus DiskIoEngine::query(DiskIoTicket ticket) const
{
    std::lock_guard<std::mutex> const lock(mMutex);
    return statusLocked(ticket);
}

DiskIoStatus DiskIoEngine::wait(DiskIoTicket ticket) const
{
    std::unique_lock<std::mutex> lock(mMutex);
    mDoneCv.wait(lock, [this, ticket]() { return statusLocked(ticket) != DiskIoStatus::kPending; });
    return statusLocked(ticket);
}

int DiskIoEngine::error(DiskIoTicket ticket) const
{
    std::lock_guard<std::mutex> const lock(mMutex);
    Batch const& batch = entry(ticket);
    if (batch.ticket == ticket)
    {
        return batch.done ? batch.error : 0;
    }
    // The entry was reused, so only the fact of the failure is left.
    return mFailedTickets.count(ticket) != 0 ? EIO : 0;
}

void DiskIoEngine::drain() const
{
    std::unique_lock<std::mutex> lock(mMutex);
    mDoneCv.wait(lock, [this]() { return mNumIncomplete == 0; });
}

bool DiskIoEngine::drainFor(std::chrono::milliseconds timeout) const
{
    std::unique_lock<std::mutex> lock(mMutex);
    return mDoneCv.wait_for(lock, timeout, [this]() { return mNumIncomplete == 0; });
}

uint64_t DiskIoEngine::pendingWriteBytes() const
{
    std::lock_guard<std::mutex> const lock(mMutex);
    return mStats.pendingWriteBytes;
}

DiskIoStats DiskIoEngine::takeStats()
{
    std::lock_guard<std::mutex> const lock(mMutex);
    DiskIoStats const result = mStats;
    DiskIoStats fresh;
    fresh.queuedReadOps = mStats.queuedReadOps;
    fresh.queuedWriteOps = mStats.queuedWriteOps;
    fresh.peakQueuedReadOps = mStats.queuedReadOps;
    fresh.peakQueuedWriteOps = mStats.queuedWriteOps;
    fresh.pendingWriteBytes = mStats.pendingWriteBytes;
    fresh.peakPendingWriteBytes = mStats.pendingWriteBytes;
    mStats = fresh;
    return result;
}

void DiskIoEngine::injectFaults(DiskIoKind kind, int count, int errnoValue)
{
    if (count < 0 || (count > 0 && errnoValue == 0))
    {
        throw std::invalid_argument("DiskIoEngine::injectFaults: needs a non-negative count and a non-zero errno");
    }
    std::lock_guard<std::mutex> const lock(mMutex);
    mFaultsLeft[kindIndex(kind)] = count;
    mFaultErrno[kindIndex(kind)] = errnoValue;
}

void DiskIoEngine::setPausedForTest(bool paused)
{
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        mPaused = paused;
    }
    mWorkCv.notify_all();
}

int DiskIoEngine::runOp(DiskIoKind kind, DiskIoOp const& op, std::vector<std::byte>& bounce, std::byte*& bounceAligned)
{
    switch (kind)
    {
    case DiskIoKind::kRead: return diskReadAll(op.fd, op.offset, op.buffer, op.size);
    case DiskIoKind::kWrite: return diskWriteAll(op.fd, op.offset, op.buffer, op.size);
    case DiskIoKind::kCopy:
    {
        size_t const needed = op.size + kDiskIoAlignment;
        if (bounce.size() < needed)
        {
            bounce.assign(needed, std::byte{0});
            auto const address = reinterpret_cast<uintptr_t>(bounce.data());
            bounceAligned = reinterpret_cast<std::byte*>((address + kDiskIoAlignment - 1) & ~(kDiskIoAlignment - 1));
        }
        if (int const err = diskReadAll(op.srcFd, op.srcOffset, bounceAligned, op.size); err != 0)
        {
            return err;
        }
        return diskWriteAll(op.fd, op.offset, bounceAligned, op.size);
    }
    }
    return EINVAL;
}

std::deque<DiskIoTicket>& DiskIoEngine::selectQueueLocked(bool preferWrites)
{
    if (preferWrites)
    {
        if (!mRequiredWriteQueue.empty())
        {
            return mRequiredWriteQueue;
        }
        if (!mBackgroundWriteQueue.empty())
        {
            return mBackgroundWriteQueue;
        }
        return mReadQueue;
    }
    if (!mReadQueue.empty())
    {
        if (!mRequiredWriteQueue.empty() && !mServedAgedWriteLast)
        {
            int64_t const waitNs = steadyClockNanos() - entry(mRequiredWriteQueue.front()).activateNs;
            if (waitNs >= kRequiredWriteMaxWaitNs)
            {
                mServedAgedWriteLast = true;
                return mRequiredWriteQueue;
            }
        }
        mServedAgedWriteLast = false;
        return mReadQueue;
    }
    mServedAgedWriteLast = false;
    if (!mRequiredWriteQueue.empty())
    {
        return mRequiredWriteQueue;
    }
    return mBackgroundWriteQueue;
}

void DiskIoEngine::workerLoop(bool preferWrites)
{
    std::vector<std::byte> bounce;
    std::byte* bounceAligned = nullptr;
    std::unique_lock<std::mutex> lock(mMutex);
    while (true)
    {
        mWorkCv.wait(lock, [this]() { return mStop || (!mPaused && hasWorkLocked()); });
        if (mPaused || !hasWorkLocked())
        {
            return; // mStop, and nothing this worker may start.
        }
        auto& queue = selectQueueLocked(preferWrites);
        Batch& batch = entry(queue.front());
        size_t const opIndex = batch.nextOp++;
        if (batch.nextOp == batch.ops.size())
        {
            queue.pop_front();
        }
        DiskIoKind const kind = batch.kind;
        DiskIoPriority const priority = batch.priority;
        DiskIoOp const op = batch.ops[opIndex];
        bool const isRead = kind == DiskIoKind::kRead;
        --(isRead ? mStats.queuedReadOps : mStats.queuedWriteOps);
        int fault = 0;
        if (mFaultsLeft[kindIndex(kind)] > 0)
        {
            --mFaultsLeft[kindIndex(kind)];
            fault = mFaultErrno[kindIndex(kind)];
        }
        lock.unlock();

        // The batch's ring entry is not reused before the batch completes, which needs this
        // operation, so `batch` stays valid while the lock is dropped.
        int64_t const startNs = steadyClockNanos();
        int const err = fault != 0 ? fault : runOp(kind, op, bounce, bounceAligned);
        uint64_t const busyNs = static_cast<uint64_t>(std::max<int64_t>(0, steadyClockNanos() - startNs));

        lock.lock();
        if (isRead)
        {
            ++mStats.readOps;
            mStats.readBusyNs += busyNs;
            mStats.readBytes += err == 0 ? op.size : 0;
            mStats.readErrors += err == 0 ? 0 : 1;
        }
        else
        {
            ++mStats.writeOps;
            mStats.writeBusyNs += busyNs;
            mStats.writeBytes += err == 0 ? op.size : 0;
            if (err == 0 && priority == DiskIoPriority::kRequired)
            {
                mStats.requiredWriteBytes += op.size;
            }
            mStats.writeErrors += err == 0 ? 0 : 1;
        }
        if (err != 0 && batch.error == 0)
        {
            batch.error = err;
        }
        if (--batch.remainingOps == 0)
        {
            completeLocked(batch);
        }
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
