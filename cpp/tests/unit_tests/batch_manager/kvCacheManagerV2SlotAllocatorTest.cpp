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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/cudaEvent.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

std::vector<Slot> allocateSlots(SlotAllocator& allocator, std::size_t count)
{
    std::vector<Slot> slots;
    slots.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        slots.push_back(allocator.allocate());
    }
    return slots;
}

// Release slots[first, last). Slot's implicit move leaves the source's slot id
// intact (std::optional<SlotId> is trivially copyable), so hand ownership over
// with setSlot(), which resets the source and makes double release impossible.
void releaseSlots(
    SlotAllocator& allocator, std::vector<Slot>& slots, std::size_t first = 0, std::size_t last = SIZE_MAX)
{
    for (std::size_t i = first; i < std::min(last, slots.size()); ++i)
    {
        Slot slot;
        slot.setSlot(slots[i]);
        allocator.release(std::move(slot));
    }
}

// Pool rebalance shrinks a pool group whose new size is still above the slot-ID
// high-water mark. Regression for NVBug 6225866: finishShrink() used to compute
// the expected overflow count as numActiveSlots - targetCapacity, which goes
// negative here, so the count never matched and the shrink threw instead of
// completing. Mirrors TestSlotAllocatorShrink.test_shrink_underused_pool in
// tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py, which
// only covers the Python SlotAllocator.
TEST(KvCacheManagerV2SlotAllocatorTest, ShrinkUnderusedPool)
{
    SlotAllocator allocator{SlotCount{184064}};
    auto slots = allocateSlots(allocator, 2048);
    releaseSlots(allocator, slots);
    EXPECT_EQ(allocator.numActiveSlots(), 2048);

    allocator.prepareForShrink(SlotCount{122624});
    EXPECT_EQ(allocator.numOverflowSlots(), 0);

    EXPECT_TRUE(allocator.finishShrink());
    EXPECT_EQ(allocator.numSlots(), 122624);
    EXPECT_EQ(allocator.numActiveSlots(), 2048);
    EXPECT_FALSE(allocator.shrinkInProgress());
}

// The non-trivial migration path: every ID is issued, half are released, and the
// pool shrinks to half. The released overflow-range slots must be reclaimed by
// finishShrink(). Mirrors test_shrink_touched_pool.
TEST(KvCacheManagerV2SlotAllocatorTest, ShrinkTouchedPool)
{
    SlotAllocator allocator{SlotCount{16}};
    auto slots = allocateSlots(allocator, 16);
    releaseSlots(allocator, slots, 8);
    EXPECT_EQ(allocator.numActiveSlots(), 16);

    allocator.prepareForShrink(SlotCount{8});
    EXPECT_EQ(allocator.numOverflowSlots(), 8);

    EXPECT_TRUE(allocator.finishShrink());
    EXPECT_EQ(allocator.numSlots(), 8);
    EXPECT_EQ(allocator.numActiveSlots(), 8);
    EXPECT_FALSE(allocator.shrinkInProgress());

    // Leave the allocator quiescent so the debug-build destructor checks pass.
    releaseSlots(allocator, slots, 0, 8);
}

// SlotId is signed, so an upper-bound-only test admits negative ids, which convert to a huge
// size_t and index far outside the pool.
TEST(KvCacheManagerV2SlotAllocatorTest, RejectsSlotIdsOutsideThePool)
{
    SlotAllocator allocator{SlotCount{8}};

    EXPECT_FALSE(allocator.isValidSlotId(SlotId{-1}));
    EXPECT_FALSE(allocator.isValidSlotId(SlotId{std::numeric_limits<SlotCount>::min()}));
    EXPECT_TRUE(allocator.isValidSlotId(SlotId{0}));
    EXPECT_TRUE(allocator.isValidSlotId(SlotId{7}));
    EXPECT_FALSE(allocator.isValidSlotId(SlotId{8}));
}

// Events recorded on mStream stay pending until openGate(): the stream starts with a host function
// that spins until the gate opens.
class KvCacheManagerV2SlotAllocatorEventTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaLaunchHostFunc(mStream, &waitForGate, &mGateOpen), cudaSuccess);
    }

    void TearDown() override
    {
        openGate();
        EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    CachedCudaEvent pendingEvent() const
    {
        return CachedCudaEvent{reinterpret_cast<CudaStream>(mStream)};
    }

    // Lets the stream run and waits until every event recorded on it has completed.
    void completeEvents()
    {
        openGate();
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    }

private:
    void openGate()
    {
        mGateOpen.store(true);
    }

    static void CUDART_CB waitForGate(void* gate)
    {
        auto const& open = *static_cast<std::atomic<bool> const*>(gate);
        while (!open.load())
        {
            std::this_thread::yield();
        }
    }

    cudaStream_t mStream{};
    std::atomic<bool> mGateOpen{false};
};

// A released slot whose event is still pending goes out after every never-issued id, and then only
// together with that event, so its next user still waits for the previous one.
TEST_F(KvCacheManagerV2SlotAllocatorEventTest, PendingSlotIsTakenLastWithItsEvent)
{
    SlotAllocator allocator{SlotCount{3}};
    auto slots = allocateSlots(allocator, 1);
    slots[0].readyEvent = pendingEvent();
    CUevent const pending = slots[0].readyEvent.handle();
    releaseSlots(allocator, slots);

    auto fresh = allocator.allocateMultiple(SlotCount{2});
    ASSERT_EQ(fresh.size(), 2U);
    EXPECT_EQ(fresh[0].slotId(), SlotId{1});
    EXPECT_EQ(fresh[1].slotId(), SlotId{2});

    auto reused = allocateSlots(allocator, 1);
    EXPECT_EQ(reused[0].slotId(), SlotId{0});
    EXPECT_EQ(reused[0].readyEvent.handle(), pending);
    EXPECT_FALSE(reused[0].readyEvent.queryComplete());

    completeEvents();
    releaseSlots(allocator, fresh);
    releaseSlots(allocator, reused);
    allocator.synchronize();
}

// release() leaves the event alone; the next allocation finds it complete, retires it and reuses the
// slot ahead of a never-issued id.
TEST_F(KvCacheManagerV2SlotAllocatorEventTest, AllocationRetiresEventsOfReleasedSlots)
{
    SlotAllocator allocator{SlotCount{2}};
    auto slots = allocateSlots(allocator, 1);
    slots[0].readyEvent = pendingEvent();
    // Shares the event's payload, so it observes when the allocator retires the event.
    CachedCudaEvent const observer = slots[0].readyEvent;
    releaseSlots(allocator, slots);
    completeEvents();
    EXPECT_FALSE(observer.isClosed());

    auto reused = allocateSlots(allocator, 1);
    EXPECT_EQ(reused[0].slotId(), SlotId{0});
    EXPECT_TRUE(reused[0].readyEvent.isClosed());
    EXPECT_TRUE(observer.isClosed());

    releaseSlots(allocator, reused);
    allocator.synchronize();
}

// allocateMultiple() scrubs once for the batch and still hands out every completed recycled slot,
// oldest first, before any never-issued id.
TEST_F(KvCacheManagerV2SlotAllocatorEventTest, AllocateMultipleReusesCompletedSlotsFirst)
{
    SlotAllocator allocator{SlotCount{8}};
    auto slots = allocateSlots(allocator, 4);
    for (auto& slot : slots)
    {
        slot.readyEvent = pendingEvent();
    }
    releaseSlots(allocator, slots);
    completeEvents();

    auto reused = allocator.allocateMultiple(SlotCount{4});
    ASSERT_EQ(reused.size(), 4U);
    for (std::size_t i = 0; i < reused.size(); ++i)
    {
        EXPECT_EQ(reused[i].slotId(), SlotId{static_cast<SlotCount>(i)});
        EXPECT_TRUE(reused[i].readyEvent.isClosed());
    }

    releaseSlots(allocator, reused);
    allocator.synchronize();
}

} // namespace
