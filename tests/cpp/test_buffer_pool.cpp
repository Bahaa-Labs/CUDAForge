#include "cudaforge/memory/block_allocator.h"
#include "cudaforge/memory/buffer_pool.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

namespace cudaforge::memory::test {

class BufferPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        allocator_ = std::make_unique<BlockAllocator>(0.05);
        pool_ = std::make_unique<BufferPool>(*allocator_, 500 * 1024 * 1024); // 500MB cache limit
    }

    void TearDown() override {
        pool_.reset();
        allocator_.reset();
    }

    std::unique_ptr<BlockAllocator> allocator_;
    std::unique_ptr<BufferPool> pool_;
};

// =============================================================================
// Test 1: Pool Hit vs Miss Reuse Verification
// =============================================================================
TEST_F(BufferPoolTest, PoolAcquireReleaseReuse) {
    const size_t request_bytes = 1000 * 1024; // ~1MB -> rounds up to 1024KB bucket

    // First acquire: Should be a pool miss
    void* ptr1 = pool_->acquire(request_bytes, MemoryCategory::WORKSPACE);
    ASSERT_NE(ptr1, nullptr);

    BufferPoolStats stats1 = pool_->get_stats();
    EXPECT_EQ(stats1.pool_misses, 1u);
    EXPECT_EQ(stats1.pool_hits, 0u);

    // Release buffer back to pool cache
    pool_->release(ptr1);

    BufferPoolStats stats_rel = pool_->get_stats();
    EXPECT_EQ(stats_rel.active_bytes, 0u);
    EXPECT_GT(stats_rel.free_cached_bytes, 0u);

    // Second acquire of same size: Should hit pool and reuse ptr1 address
    void* ptr2 = pool_->acquire(request_bytes, MemoryCategory::WORKSPACE);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_EQ(ptr1, ptr2) << "BufferPool failed to reuse cached memory address!";

    BufferPoolStats stats2 = pool_->get_stats();
    EXPECT_EQ(stats2.pool_hits, 1u);

    pool_->release(ptr2);
}

// =============================================================================
// Test 2: Internal and External Fragmentation Calculations
// =============================================================================
TEST_F(BufferPoolTest, FragmentationMetricsCalculation) {
    // Request 300KB -> rounds up to 512KB power of 2 bucket
    const size_t request_bytes = 300 * 1024;
    void* ptr = pool_->acquire(request_bytes, MemoryCategory::ACTIVATIONS);

    BufferPoolStats stats = pool_->get_stats();
    
    // Internal Fragmentation = 1.0 - (300KB / 512KB) = ~0.414 (41.4%)
    EXPECT_GT(stats.internal_fragmentation_ratio, 0.30);
    EXPECT_LT(stats.internal_fragmentation_ratio, 0.50);

    pool_->release(ptr);
}

// =============================================================================
// Test 3: RAII ScopedBuffer Auto-Release
// =============================================================================
TEST_F(BufferPoolTest, ScopedBufferAutoRelease) {
    {
        // 2049 bytes rounds up to the next power-of-2 bucket (4096 bytes)
        ScopedBuffer buffer = pool_->acquire_scoped(2049, MemoryCategory::WORKSPACE);
        ASSERT_NE(buffer.get(), nullptr);
        EXPECT_EQ(pool_->get_stats().active_bytes, 4096u);
    }

    // After exiting scope, buffer is auto-released back to free cached bytes
    EXPECT_EQ(pool_->get_stats().active_bytes, 0u);
    EXPECT_EQ(pool_->get_stats().free_cached_bytes, 4096u);
}

// =============================================================================
// Test 4: Emergency Purge & OOM Recovery Path
// =============================================================================
TEST_F(BufferPoolTest, LowLevelOOMRecoveryPurge) {
    // Fill free cache with cached allocations
    void* ptr1 = pool_->acquire(10 * 1024 * 1024, MemoryCategory::WORKSPACE); // 10MB
    pool_->release(ptr1);

    EXPECT_GT(pool_->get_stats().free_cached_bytes, 0u);

    // Explicitly purge free cache
    size_t purged = pool_->purge_free_buffers();
    EXPECT_GE(purged, 10 * 1024 * 1024);
    EXPECT_EQ(pool_->get_stats().free_cached_bytes, 0u);
}

} // namespace cudaforge::memory::test