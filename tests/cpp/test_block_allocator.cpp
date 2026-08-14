#include "cudaforge/memory/block_allocator.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <numeric>

namespace cudaforge::memory::test {

class BlockAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize allocator with 5% VRAM safety headroom
        allocator_ = std::make_unique<BlockAllocator>(0.05);
    }

    void TearDown() override {
        allocator_.reset();
    }

    std::unique_ptr<BlockAllocator> allocator_;
};

// =============================================================================
// Test 1: Basic Allocation, 256-Byte Alignment, & GPU Read/Write Verification
// =============================================================================
TEST_F(BlockAllocatorTest, AllocationAlignmentAndDeviceAccess) {
    const size_t element_count = 1024;
    const size_t request_bytes = element_count * sizeof(float); // 4096 bytes

    float* d_ptr = allocator_->allocate_typed<float>(element_count, MemoryCategory::WORKSPACE);
    ASSERT_NE(d_ptr, nullptr);

    // Verify 256-byte alignment (lower 8 bits of pointer address must be zero)
    uintptr_t address = reinterpret_cast<uintptr_t>(d_ptr);
    EXPECT_EQ(address % 256, 0u) << "Pointer " << d_ptr << " is not aligned to 256 bytes!";

    // Prepare host pattern data
    std::vector<float> host_src(element_count);
    std::iota(host_src.begin(), host_src.end(), 1.0f);

    // Write host data to allocated device memory
    cudaError_t err = cudaMemcpy(d_ptr, host_src.data(), request_bytes, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy HostToDevice failed: " << cudaGetErrorString(err);

    // Read device memory back to host
    std::vector<float> host_dst(element_count, 0.0f);
    err = cudaMemcpy(host_dst.data(), d_ptr, request_bytes, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy DeviceToHost failed: " << cudaGetErrorString(err);

    // Assert data integrity
    EXPECT_EQ(host_src, host_dst);

    // Clean up
    allocator_->free(d_ptr);
    EXPECT_EQ(allocator_->get_total_allocated_bytes(), 0u);
}

// =============================================================================
// Test 2: Memory Category Breakdown Tracking
// =============================================================================
TEST_F(BlockAllocatorTest, MemoryCategoryTracking) {
    const size_t weights_bytes = 1024 * 1024;     // 1 MB
    const size_t act_bytes = 2 * 1024 * 1024;      // 2 MB
    const size_t kv_bytes = 4 * 1024 * 1024;       // 4 MB

    void* p_weights = allocator_->allocate(weights_bytes, MemoryCategory::WEIGHTS);
    void* p_act = allocator_->allocate(act_bytes, MemoryCategory::ACTIVATIONS);
    void* p_kv = allocator_->allocate(kv_bytes, MemoryCategory::KV_CACHE);

    EXPECT_GE(allocator_->get_category_bytes(MemoryCategory::WEIGHTS), weights_bytes);
    EXPECT_GE(allocator_->get_category_bytes(MemoryCategory::ACTIVATIONS), act_bytes);
    EXPECT_GE(allocator_->get_category_bytes(MemoryCategory::KV_CACHE), kv_bytes);
    EXPECT_EQ(allocator_->get_category_bytes(MemoryCategory::WORKSPACE), 0u);

    const size_t total_expected = allocator_->get_category_bytes(MemoryCategory::WEIGHTS) +
                                  allocator_->get_category_bytes(MemoryCategory::ACTIVATIONS) +
                                  allocator_->get_category_bytes(MemoryCategory::KV_CACHE);

    EXPECT_EQ(allocator_->get_total_allocated_bytes(), total_expected);

    // Deallocate selectively and verify category updates
    allocator_->free(p_weights);
    EXPECT_EQ(allocator_->get_category_bytes(MemoryCategory::WEIGHTS), 0u);
    EXPECT_GT(allocator_->get_total_allocated_bytes(), 0u);

    allocator_->free(p_act);
    allocator_->free(p_kv);
    EXPECT_EQ(allocator_->get_total_allocated_bytes(), 0u);
}

// =============================================================================
// Test 3: Stream-Ordered Execution Test
// =============================================================================
TEST_F(BlockAllocatorTest, StreamOrderedAllocAndFree) {
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    ASSERT_EQ(err, cudaSuccess);

    const size_t size = 512 * 1024; // 512 KB
    void* d_ptr = allocator_->allocate(size, MemoryCategory::WORKSPACE, stream);
    ASSERT_NE(d_ptr, nullptr);

    // Asynchronously deallocate on stream
    allocator_->free(d_ptr, stream);

    // Synchronize stream to ensure GPU operations complete cleanly
    err = cudaStreamSynchronize(stream);
    EXPECT_EQ(err, cudaSuccess);

    cudaStreamDestroy(stream);
}

// =============================================================================
// Test 4: Peak Metric Allocation Tracking & Peak Reset
// =============================================================================
TEST_F(BlockAllocatorTest, PeakAllocationTrackingAndReset) {
    void* p1 = allocator_->allocate(10 * 1024 * 1024, MemoryCategory::WORKSPACE); // 10 MB
    size_t peak1 = allocator_->get_peak_allocated_bytes();
    EXPECT_GE(peak1, 10 * 1024 * 1024);

    void* p2 = allocator_->allocate(20 * 1024 * 1024, MemoryCategory::WORKSPACE); // +20 MB -> 30 MB
    size_t peak2 = allocator_->get_peak_allocated_bytes();
    EXPECT_GE(peak2, 30 * 1024 * 1024);

    // Free all pointers
    allocator_->free(p1);
    allocator_->free(p2);

    // Allocated bytes drop to 0, but peak remains at highest level
    EXPECT_EQ(allocator_->get_total_allocated_bytes(), 0u);
    EXPECT_EQ(allocator_->get_peak_allocated_bytes(), peak2);

    // Reset peak metric
    allocator_->reset_peak_stats();
    EXPECT_EQ(allocator_->get_peak_allocated_bytes(), 0u);
}

// =============================================================================
// Test 5: Pre-Allocation OOM Protection Safeguard
// =============================================================================
TEST_F(BlockAllocatorTest, OutOfMemorySafeguardThrowsException) {
    // Attempt allocation exceeding physical VRAM capacity (e.g., 64 GB on a 10 GB GPU)
    const size_t oversized_request = static_cast<size_t>(64) * 1024 * 1024 * 1024;

    EXPECT_THROW(
        {
            [[maybe_unused]] void* ptr = allocator_->allocate(oversized_request, MemoryCategory::ACTIVATIONS);
        },
        OutOfMemoryException
    );
}

} // namespace cudaforge::memory::test