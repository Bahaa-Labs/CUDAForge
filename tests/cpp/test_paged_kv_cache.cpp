#include "cudaforge/memory/block_allocator.h"
#include "cudaforge/memory/paged_kv_cache.h"
#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>

namespace cudaforge::memory::test {

class PagedKVCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        allocator_ = std::make_unique<BlockAllocator>(0.05);

        config_.num_layers = 2;
        config_.num_kv_heads = 2;
        config_.head_dim = 64;
        config_.block_size = 16;
        config_.element_size_bytes = 2;
        config_.total_gpu_blocks = 32;
        config_.max_supported_batch_size = 16;
        config_.max_blocks_per_sequence = 32;

        kv_cache_ = std::make_unique<PagedKVCache>(config_, *allocator_);
    }

    void TearDown() override {
        kv_cache_.reset();
        allocator_.reset();
    }

    KVCacheConfig config_;
    std::unique_ptr<BlockAllocator> allocator_;
    std::unique_ptr<PagedKVCache> kv_cache_;
};

// =============================================================================
// Test: GPU Batch Metadata Preparation & Device Sync
// =============================================================================
TEST_F(PagedKVCacheTest, PrepareBatchMetadataStagesGPUTransfers) {
    std::vector<int32_t> prompt_a(32, 1); // 2 blocks
    std::vector<int32_t> prompt_b(48, 2); // 3 blocks

    kv_cache_->register_sequence_with_prefix(100, prompt_a);
    kv_cache_->register_sequence_with_prefix(200, prompt_b);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    std::vector<uint64_t> active_batch = {100, 200};
    
    // Prepare metadata for CUDA kernel consumption
    KVCacheBatchDescriptor desc = kv_cache_->prepare_batch_metadata(active_batch, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_EQ(desc.batch_size, 2);
    EXPECT_EQ(desc.max_blocks_per_seq, 3);
    EXPECT_NE(desc.block_tables, nullptr);
    EXPECT_NE(desc.context_lens, nullptr);
    EXPECT_NE(desc.key_pool_ptr, nullptr);
    EXPECT_NE(desc.val_pool_ptr, nullptr);

    // Verify GPU memory context length contents by copying back to host
    std::vector<int32_t> h_ctx_lens(2);
    ASSERT_EQ(cudaMemcpy(h_ctx_lens.data(), desc.context_lens, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(h_ctx_lens[0], 32); // Sequence 100 context length
    EXPECT_EQ(h_ctx_lens[1], 48); // Sequence 200 context length

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    kv_cache_->unregister_sequence(100);
    kv_cache_->unregister_sequence(200);
}

} // namespace cudaforge::memory::test