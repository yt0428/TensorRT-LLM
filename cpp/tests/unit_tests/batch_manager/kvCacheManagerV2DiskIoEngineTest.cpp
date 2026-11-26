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

// CPU-only tests of the KVCM2 disk I/O engine: data movement, activation, queue priority, completion
// words, error reporting and shutdown. The CUDA ordering built on top of it is tested with the cache.

#include "kv_cache_manager_v2/diskIoEngine.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace kv = tensorrt_llm::batch_manager::kv_cache_manager_v2;

namespace
{

constexpr size_t kPage = 3 * kv::kDiskIoAlignment;

struct FreeDeleter
{
    void operator()(std::byte* ptr) const noexcept
    {
        std::free(ptr);
    }
};

using AlignedBuffer = std::unique_ptr<std::byte, FreeDeleter>;

AlignedBuffer alignedBuffer(size_t size)
{
    void* ptr = nullptr;
    EXPECT_EQ(posix_memalign(&ptr, kv::kDiskIoAlignment, size), 0);
    std::memset(ptr, 0, size);
    return AlignedBuffer(static_cast<std::byte*>(ptr));
}

void fillPattern(std::byte* data, size_t size, uint8_t seed)
{
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<std::byte>(static_cast<uint8_t>(seed + i * 7));
    }
}

//! An unlinked temporary file of a fixed size, opened buffered or with O_DIRECT.
class TempFile
{
public:
    TempFile(size_t size, bool direct)
    {
        std::string path = ::testing::TempDir() + "kvcm2DiskIoXXXXXX";
        int const fd = ::mkstemp(path.data());
        EXPECT_GE(fd, 0) << std::strerror(errno);
        ::unlink(path.c_str());
        EXPECT_EQ(::ftruncate(fd, static_cast<off_t>(size)), 0);
        if (direct)
        {
            // Reopen through /proc so O_DIRECT applies; some filesystems (tmpfs) reject it.
            std::string const procPath = "/proc/self/fd/" + std::to_string(fd);
            mFd = ::open(procPath.c_str(), O_RDWR | O_DIRECT);
            ::close(fd);
        }
        else
        {
            mFd = fd;
        }
    }

    ~TempFile()
    {
        if (mFd >= 0)
        {
            ::close(mFd);
        }
    }

    TempFile(TempFile const&) = delete;
    TempFile& operator=(TempFile const&) = delete;

    [[nodiscard]] int fd() const noexcept
    {
        return mFd;
    }

private:
    int mFd = -1;
};

class DiskIoEngineTest : public ::testing::Test
{
protected:
    std::unique_ptr<kv::DiskIoEngine> makeEngine(int numThreads, int numWritePreferring = 0, uint32_t ringSize = 64)
    {
        mWords.assign(ringSize, 0);
        kv::DiskIoEngine::Options options;
        options.numThreads = numThreads;
        options.numWritePreferringThreads = numWritePreferring;
        options.ringSize = ringSize;
        return std::make_unique<kv::DiskIoEngine>(options, mWords.data());
    }

    std::vector<kv::DiskIoOp> pageOps(int fd, std::byte* base, size_t numPages, size_t firstPage = 0)
    {
        std::vector<kv::DiskIoOp> ops;
        for (size_t i = 0; i < numPages; ++i)
        {
            ops.push_back(kv::DiskIoOp{fd, static_cast<int64_t>((firstPage + i) * kPage), base + i * kPage, kPage});
        }
        return ops;
    }

    std::vector<uint32_t> mWords;
};

void roundTrip(kv::DiskIoEngine& engine, std::vector<uint32_t> const& words, int fd)
{
    constexpr size_t kNumPages = 32;
    auto src = alignedBuffer(kNumPages * kPage);
    auto dst = alignedBuffer(kNumPages * kPage);
    fillPattern(src.get(), kNumPages * kPage, 3);

    std::vector<kv::DiskIoOp> writes;
    std::vector<kv::DiskIoOp> reads;
    for (size_t i = 0; i < kNumPages; ++i)
    {
        // Pages land in reverse order on disk, so a mixed-up offset shows in the comparison.
        auto const offset = static_cast<int64_t>((kNumPages - 1 - i) * kPage);
        writes.push_back(kv::DiskIoOp{fd, offset, src.get() + i * kPage, kPage});
        reads.push_back(kv::DiskIoOp{fd, offset, dst.get() + i * kPage, kPage});
    }
    auto const writeTicket = engine.submit(kv::DiskIoKind::kWrite, std::move(writes), true);
    ASSERT_EQ(engine.wait(writeTicket), kv::DiskIoStatus::kOk);
    auto const readTicket = engine.submit(kv::DiskIoKind::kRead, std::move(reads), true);
    ASSERT_EQ(engine.wait(readTicket), kv::DiskIoStatus::kOk);
    EXPECT_EQ(std::memcmp(src.get(), dst.get(), kNumPages * kPage), 0);
    EXPECT_EQ(words[writeTicket % words.size()], writeTicket);
    EXPECT_EQ(words[readTicket % words.size()], readTicket);
}

} // namespace

TEST_F(DiskIoEngineTest, BufferedRoundTrip)
{
    TempFile file(64 * kPage, /*direct=*/false);
    auto engine = makeEngine(4);
    roundTrip(*engine, mWords, file.fd());
}

TEST_F(DiskIoEngineTest, DirectIoRoundTrip)
{
    TempFile file(64 * kPage, /*direct=*/true);
    if (file.fd() < 0)
    {
        GTEST_SKIP() << "O_DIRECT is not supported under " << ::testing::TempDir();
    }
    auto engine = makeEngine(4);
    roundTrip(*engine, mWords, file.fd());
}

TEST_F(DiskIoEngineTest, InactiveBatchWaitsForActivation)
{
    TempFile file(4 * kPage, false);
    auto engine = makeEngine(2);
    auto src = alignedBuffer(kPage);
    fillPattern(src.get(), kPage, 11);

    auto const ticket = engine->submit(kv::DiskIoKind::kWrite, pageOps(file.fd(), src.get(), 1), false);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(engine->query(ticket), kv::DiskIoStatus::kPending);
    EXPECT_EQ(mWords[ticket % mWords.size()], 0u);
    EXPECT_EQ(engine->pendingWriteBytes(), kPage);

    std::thread activator([&]() { engine->activate(ticket); });
    EXPECT_EQ(engine->wait(ticket), kv::DiskIoStatus::kOk);
    activator.join();
    EXPECT_EQ(engine->pendingWriteBytes(), 0u);
    // The peak outlives the write until takeStats() starts the next window at the current level.
    EXPECT_EQ(engine->takeStats().peakPendingWriteBytes, kPage);
    EXPECT_EQ(engine->takeStats().peakPendingWriteBytes, 0u);

    auto check = alignedBuffer(kPage);
    ASSERT_EQ(kv::diskReadAll(file.fd(), 0, check.get(), kPage), 0);
    EXPECT_EQ(std::memcmp(src.get(), check.get(), kPage), 0);
}

TEST_F(DiskIoEngineTest, EmptyBatchCompletesOnActivation)
{
    auto engine = makeEngine(1);
    auto const ticket = engine->submit(kv::DiskIoKind::kRead, {}, false);
    EXPECT_EQ(engine->query(ticket), kv::DiskIoStatus::kPending);
    engine->activate(ticket);
    EXPECT_EQ(engine->query(ticket), kv::DiskIoStatus::kOk);
    EXPECT_EQ(mWords[ticket % mWords.size()], ticket);
}

// With one read-preferring worker and both queues non-empty, the read runs first and sees the data
// from before the write; with one write-preferring worker it sees the write.
TEST_F(DiskIoEngineTest, QueuePreference)
{
    for (int numWritePreferring : {0, 1})
    {
        TempFile file(2 * kPage, false);
        auto engine = makeEngine(1, numWritePreferring);
        auto oldData = alignedBuffer(kPage);
        auto newData = alignedBuffer(kPage);
        auto readBack = alignedBuffer(kPage);
        fillPattern(oldData.get(), kPage, 1);
        fillPattern(newData.get(), kPage, 2);
        ASSERT_EQ(kv::diskWriteAll(file.fd(), 0, oldData.get(), kPage), 0);

        engine->setPausedForTest(true);
        auto const write = engine->submit(kv::DiskIoKind::kWrite, pageOps(file.fd(), newData.get(), 1), true);
        auto const read = engine->submit(kv::DiskIoKind::kRead, pageOps(file.fd(), readBack.get(), 1), true);
        engine->setPausedForTest(false);
        ASSERT_EQ(engine->wait(read), kv::DiskIoStatus::kOk);
        ASSERT_EQ(engine->wait(write), kv::DiskIoStatus::kOk);
        auto const& expected = numWritePreferring == 0 ? oldData : newData;
        EXPECT_EQ(std::memcmp(expected.get(), readBack.get(), kPage), 0) << "write-preferring=" << numWritePreferring;
    }
}

// The caller orders a read after a write by activating the read only once the write completed.
TEST_F(DiskIoEngineTest, ActivationOrdersWriteBeforeRead)
{
    TempFile file(2 * kPage, false);
    auto engine = makeEngine(4);
    auto newData = alignedBuffer(kPage);
    auto readBack = alignedBuffer(kPage);
    fillPattern(newData.get(), kPage, 5);

    auto const write = engine->submit(kv::DiskIoKind::kWrite, pageOps(file.fd(), newData.get(), 1), false);
    auto const read = engine->submit(kv::DiskIoKind::kRead, pageOps(file.fd(), readBack.get(), 1), false);
    engine->activate(write);
    ASSERT_EQ(engine->wait(write), kv::DiskIoStatus::kOk);
    engine->activate(read);
    ASSERT_EQ(engine->wait(read), kv::DiskIoStatus::kOk);
    EXPECT_EQ(std::memcmp(newData.get(), readBack.get(), kPage), 0);
}

TEST_F(DiskIoEngineTest, ManyBatchesPublishTheirTickets)
{
    constexpr size_t kNumBatches = 200;
    constexpr size_t kPagesPerBatch = 3;
    TempFile file(kNumBatches * kPagesPerBatch * kPage, false);
    auto engine = makeEngine(8, 1, 64);
    auto data = alignedBuffer(kNumBatches * kPagesPerBatch * kPage);
    fillPattern(data.get(), kNumBatches * kPagesPerBatch * kPage, 9);

    std::vector<kv::DiskIoTicket> tickets;
    for (size_t b = 0; b < kNumBatches; ++b)
    {
        auto* base = data.get() + b * kPagesPerBatch * kPage;
        tickets.push_back(
            engine->submit(kv::DiskIoKind::kWrite, pageOps(file.fd(), base, kPagesPerBatch, b * kPagesPerBatch), true));
    }
    engine->drain();
    for (auto const ticket : tickets)
    {
        EXPECT_EQ(engine->query(ticket), kv::DiskIoStatus::kOk);
    }
    // The last ring's worth of tickets are still in their words.
    for (size_t b = kNumBatches - mWords.size(); b < kNumBatches; ++b)
    {
        EXPECT_EQ(mWords[tickets[b] % mWords.size()], tickets[b]);
    }
    auto readBack = alignedBuffer(kNumBatches * kPagesPerBatch * kPage);
    ASSERT_EQ(kv::diskReadAll(file.fd(), 0, readBack.get(), kNumBatches * kPagesPerBatch * kPage), 0);
    EXPECT_EQ(std::memcmp(data.get(), readBack.get(), kNumBatches * kPagesPerBatch * kPage), 0);

    auto const stats = engine->takeStats();
    EXPECT_EQ(stats.writeOps, kNumBatches * kPagesPerBatch);
    EXPECT_EQ(stats.writeBytes, kNumBatches * kPagesPerBatch * kPage);
    EXPECT_EQ(stats.writeErrors, 0u);
    EXPECT_EQ(stats.pendingWriteBytes, 0u);
    EXPECT_EQ(engine->takeStats().writeOps, 0u);
}

TEST_F(DiskIoEngineTest, SubmitBlocksWhileTheRingIsFull)
{
    auto engine = makeEngine(1, 0, 2);
    auto const first = engine->submit(kv::DiskIoKind::kRead, {}, false);
    auto const second = engine->submit(kv::DiskIoKind::kRead, {}, false);
    std::atomic<bool> submitted{false};
    kv::DiskIoTicket third = kv::kNoDiskIoTicket;
    std::thread submitter(
        [&]()
        {
            third = engine->submit(kv::DiskIoKind::kRead, {}, true);
            submitted = true;
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(submitted.load());
    engine->activate(first);
    submitter.join();
    EXPECT_TRUE(submitted.load());
    EXPECT_EQ(engine->query(third), kv::DiskIoStatus::kOk);
    engine->activate(second);
}

TEST_F(DiskIoEngineTest, FailuresAreReportedAndKept)
{
    TempFile file(2 * kPage, false);
    auto engine = makeEngine(2, 0, 4);
    auto buffer = alignedBuffer(4 * kPage);

    // A bad descriptor.
    auto const badFd = engine->submit(kv::DiskIoKind::kRead, pageOps(-1, buffer.get(), 1), true);
    EXPECT_EQ(engine->wait(badFd), kv::DiskIoStatus::kFailed);
    EXPECT_EQ(engine->error(badFd), EBADF);

    // A read past the end of the file is an error, not a short success.
    auto const pastEnd = engine->submit(kv::DiskIoKind::kRead, pageOps(file.fd(), buffer.get(), 1, 5), true);
    EXPECT_EQ(engine->wait(pastEnd), kv::DiskIoStatus::kFailed);
    EXPECT_EQ(engine->error(pastEnd), EIO);

    // One injected failure fails its whole batch and only that batch.
    engine->injectFaults(kv::DiskIoKind::kWrite, 1, ENOSPC);
    auto const injected = engine->submit(kv::DiskIoKind::kWrite, pageOps(file.fd(), buffer.get(), 2), true);
    EXPECT_EQ(engine->wait(injected), kv::DiskIoStatus::kFailed);
    EXPECT_EQ(engine->error(injected), ENOSPC);
    auto const good = engine->submit(kv::DiskIoKind::kWrite, pageOps(file.fd(), buffer.get(), 2), true);
    EXPECT_EQ(engine->wait(good), kv::DiskIoStatus::kOk);
    // A failed batch still publishes its ticket: the buffers are free again.
    EXPECT_EQ(mWords[injected % mWords.size()], injected);

    // Statuses survive the reuse of their ring entries.
    for (int i = 0; i < 8; ++i)
    {
        engine->wait(engine->submit(kv::DiskIoKind::kRead, {}, true));
    }
    EXPECT_EQ(engine->query(badFd), kv::DiskIoStatus::kFailed);
    EXPECT_EQ(engine->query(injected), kv::DiskIoStatus::kFailed);
    EXPECT_EQ(engine->query(good), kv::DiskIoStatus::kOk);

    auto const stats = engine->takeStats();
    EXPECT_EQ(stats.readErrors, 2u);
    EXPECT_EQ(stats.writeErrors, 1u);
    EXPECT_EQ(stats.writeBytes, 3 * kPage);
}

TEST_F(DiskIoEngineTest, CopyMovesDataBetweenFiles)
{
    TempFile src(4 * kPage, false);
    TempFile dst(4 * kPage, false);
    auto engine = makeEngine(2);
    auto data = alignedBuffer(kPage);
    fillPattern(data.get(), kPage, 21);
    ASSERT_EQ(kv::diskWriteAll(src.fd(), 2 * kPage, data.get(), kPage), 0);

    kv::DiskIoOp op;
    op.fd = dst.fd();
    op.offset = static_cast<int64_t>(kPage);
    op.size = kPage;
    op.srcFd = src.fd();
    op.srcOffset = static_cast<int64_t>(2 * kPage);
    auto const ticket = engine->submit(kv::DiskIoKind::kCopy, {op}, true);
    ASSERT_EQ(engine->wait(ticket), kv::DiskIoStatus::kOk);

    auto check = alignedBuffer(kPage);
    ASSERT_EQ(kv::diskReadAll(dst.fd(), kPage, check.get(), kPage), 0);
    EXPECT_EQ(std::memcmp(data.get(), check.get(), kPage), 0);
}

TEST_F(DiskIoEngineTest, DestructionDrainsLateActivations)
{
    TempFile file(8 * kPage, false);
    auto engine = makeEngine(2);
    auto data = alignedBuffer(8 * kPage);
    fillPattern(data.get(), 8 * kPage, 4);
    auto const active = engine->submit(kv::DiskIoKind::kWrite, pageOps(file.fd(), data.get(), 4), true);
    auto const late = engine->submit(kv::DiskIoKind::kWrite, pageOps(file.fd(), data.get() + 4 * kPage, 4, 4), false);
    std::thread activator(
        [&]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            engine->activate(late);
        });
    // Waits for the late activation, then for both batches.
    engine->drain();
    activator.join();
    engine.reset();
    EXPECT_EQ(mWords[active % mWords.size()], active);
    EXPECT_EQ(mWords[late % mWords.size()], late);

    auto check = alignedBuffer(8 * kPage);
    ASSERT_EQ(kv::diskReadAll(file.fd(), 0, check.get(), 8 * kPage), 0);
    EXPECT_EQ(std::memcmp(data.get(), check.get(), 8 * kPage), 0);
}

TEST_F(DiskIoEngineTest, RejectsBadOptions)
{
    std::vector<uint32_t> words(8, 0);
    kv::DiskIoEngine::Options options;
    options.numThreads = 0;
    EXPECT_THROW(kv::DiskIoEngine(options, words.data()), std::invalid_argument);
    options.numThreads = 1;
    options.ringSize = 6;
    EXPECT_THROW(kv::DiskIoEngine(options, words.data()), std::invalid_argument);
    options.ringSize = 8;
    EXPECT_THROW(kv::DiskIoEngine(options, nullptr), std::invalid_argument);
}

TEST_F(DiskIoEngineTest, RequiredWritesPreemptBackgroundWrites)
{
    TempFile file(2 * kPage, false);
    auto engine = makeEngine(1, /*numWritePreferring=*/1);
    auto backgroundData = alignedBuffer(kPage);
    auto requiredData = alignedBuffer(kPage);
    fillPattern(backgroundData.get(), kPage, 10);
    fillPattern(requiredData.get(), kPage, 20);

    engine->setPausedForTest(true);
    auto const bg = engine->submit(
        kv::DiskIoKind::kWrite, pageOps(file.fd(), backgroundData.get(), 1), true, kv::DiskIoPriority::kBackground);
    auto const req = engine->submit(
        kv::DiskIoKind::kWrite, pageOps(file.fd(), requiredData.get(), 1), true, kv::DiskIoPriority::kRequired);
    engine->setPausedForTest(false);
    ASSERT_EQ(engine->wait(req), kv::DiskIoStatus::kOk);
    ASSERT_EQ(engine->wait(bg), kv::DiskIoStatus::kOk);

    // Required ran first and background overwrote the same slot afterwards.
    auto check = alignedBuffer(kPage);
    ASSERT_EQ(kv::diskReadAll(file.fd(), 0, check.get(), kPage), 0);
    EXPECT_EQ(std::memcmp(backgroundData.get(), check.get(), kPage), 0);

    auto const stats = engine->takeStats();
    EXPECT_EQ(stats.writeBytes, 2 * kPage);
    EXPECT_EQ(stats.requiredWriteBytes, kPage);
}

TEST_F(DiskIoEngineTest, AgedRequiredWriteRunsAheadOfReads)
{
    TempFile file(2 * kPage, false);
    auto engine = makeEngine(1, /*numWritePreferring=*/0);
    auto oldData = alignedBuffer(kPage);
    auto newData = alignedBuffer(kPage);
    auto readBack = alignedBuffer(kPage);
    fillPattern(oldData.get(), kPage, 31);
    fillPattern(newData.get(), kPage, 42);
    ASSERT_EQ(kv::diskWriteAll(file.fd(), 0, oldData.get(), kPage), 0);

    engine->setPausedForTest(true);
    auto const write = engine->submit(
        kv::DiskIoKind::kWrite, pageOps(file.fd(), newData.get(), 1), true, kv::DiskIoPriority::kRequired);
    // Let the required write wait past kRequiredWriteMaxWaitNs (10 ms) before unpausing.
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    auto const read = engine->submit(kv::DiskIoKind::kRead, pageOps(file.fd(), readBack.get(), 1), true);
    engine->setPausedForTest(false);
    ASSERT_EQ(engine->wait(write), kv::DiskIoStatus::kOk);
    ASSERT_EQ(engine->wait(read), kv::DiskIoStatus::kOk);

    // The aged required write ran ahead of the read on the read-preferring worker.
    EXPECT_EQ(std::memcmp(newData.get(), readBack.get(), kPage), 0);
}
