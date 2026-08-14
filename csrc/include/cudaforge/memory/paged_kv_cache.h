#pragma once

#include "cudaforge/memory/block_allocator.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <list>
#include <memory>
#include <string>
#include <stdexcept>
#include <span>

namespace cudaforge::memory {

/*
    Helper CUDA error check macro for internal runtime calls.
 */
#define CUDA_CHECK_METADATA(call)                                                \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(                                           \
                std::string("CUDA Error in PagedKVCache: ") +                  \
                cudaGetErrorString(err) + " at " + __FILE__ + ":" +             \
                std::to_string(__LINE__));                                      \
        }                                                                       \
    } while (0)

class OutOfPagesException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct KVCacheConfig {
    uint32_t num_layers{32};
    uint32_t num_kv_heads{8};
    uint32_t head_dim{128};
    uint32_t block_size{16}; 
    size_t element_size_bytes{2}; 
    size_t total_gpu_blocks{1024}; 
    bool enable_prefix_caching{true};
    size_t max_supported_batch_size{256};
    size_t max_blocks_per_sequence{512};
};

struct PhysicalBlock {
    int32_t block_id{-1};
    uint32_t ref_count{0};
    uint32_t num_tokens_filled{0};
    uint64_t hash_value{0};
    bool is_cached{false};
    std::list<int32_t>::iterator lru_iterator;
};

struct PrefixTreeNode {
    uint64_t hash_value{0};
    int32_t physical_block_id{-1};
    std::unordered_map<uint64_t, std::shared_ptr<PrefixTreeNode>> children;
};

struct KVCacheStats {
    size_t total_blocks{0};
    size_t free_blocks{0};
    size_t allocated_blocks{0};
    size_t cached_prefix_blocks{0};
    size_t total_memory_bytes{0};
    size_t used_memory_bytes{0};
    double pool_utilization_ratio{0.0};
    uint32_t active_sequences_count{0};
    uint64_t prefix_cache_hits{0};
    uint64_t prefix_cache_misses{0};
    uint64_t lru_evictions_count{0};
};

/*
    Plain C-compatible Batch Descriptor struct passed directly to CUDA device kernels.
 */
struct KVCacheBatchDescriptor {
    const int32_t* block_tables{nullptr};  // GPU pointer: [batch_size, max_blocks_per_seq]
    const int32_t* context_lens{nullptr};  // GPU pointer: [batch_size]
    int32_t batch_size{0};
    int32_t max_blocks_per_seq{0};
    int32_t block_size{0};
    int32_t num_kv_heads{0};
    int32_t head_dim{0};
    size_t bytes_per_block{0};
    size_t layer_block_bytes{0};
    const void* key_pool_ptr{nullptr};
    const void* val_pool_ptr{nullptr};
};

/*
    Virtual Memory Page Table Manager with GPU Metadata Staging.
 */
class PagedKVCache {
public:
    PagedKVCache(const KVCacheConfig& config, BlockAllocator& allocator);
    ~PagedKVCache();

    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&) = delete;
    PagedKVCache& operator=(PagedKVCache&&) = delete;

    uint32_t register_sequence_with_prefix(uint64_t sequence_id, std::span<const int32_t> prompt_tokens);
    void append_tokens(uint64_t sequence_id, uint32_t num_tokens = 1);
    void unregister_sequence(uint64_t sequence_id);
    void fork_sequence(uint64_t parent_sequence_id, uint64_t child_sequence_id);

    /**
     * @brief Asynchronously prepares and stages GPU metadata structures for a batch of sequences.
     * @param active_sequence_ids Sequence IDs included in current forward step.
     * @param stream CUDA stream for non-blocking H2D transfers.
     * @return KVCacheBatchDescriptor with valid device pointers ready for kernel launch.
     */
    KVCacheBatchDescriptor prepare_batch_metadata(
        const std::vector<uint64_t>& active_sequence_ids, 
        cudaStream_t stream = 0
    );

    [[nodiscard]] const std::vector<int32_t>& get_block_table(uint64_t sequence_id) const;
    [[nodiscard]] void* get_key_block_ptr(int32_t block_id, uint32_t layer_idx) const;
    [[nodiscard]] void* get_value_block_ptr(int32_t block_id, uint32_t layer_idx) const;

    [[nodiscard]] const KVCacheConfig& get_config() const noexcept { return config_; }
    [[nodiscard]] KVCacheStats get_stats() const;

private:
    [[nodiscard]] static uint64_t compute_block_hash(std::span<const int32_t> tokens, uint64_t parent_hash = 0) noexcept;
    
    int32_t allocate_physical_block();
    void free_physical_block(int32_t block_id);
    bool evict_lru_block();
    void touch_lru(int32_t block_id);

    KVCacheConfig config_;
    BlockAllocator& allocator_;

    size_t bytes_per_block_{0};
    size_t layer_block_bytes_{0};

    void* d_key_pool_{nullptr};
    void* d_val_pool_{nullptr};

    // Staging Buffers for GPU Metadata (Pinned Host Memory & Device Memory)
    int32_t* h_pinned_block_tables_{nullptr};
    int32_t* h_pinned_context_lens_{nullptr};
    int32_t* d_block_tables_{nullptr};
    int32_t* d_context_lens_{nullptr};

    mutable std::mutex mutex_;

    std::vector<PhysicalBlock> physical_blocks_;
    std::vector<int32_t> free_block_stack_;
    std::list<int32_t> lru_list_;

    std::shared_ptr<PrefixTreeNode> prefix_tree_root_;
    std::unordered_map<uint64_t, std::shared_ptr<PrefixTreeNode>> hash_to_node_map_;
    std::unordered_map<uint64_t, std::vector<int32_t>> page_tables_;

    uint64_t prefix_cache_hits_{0};
    uint64_t prefix_cache_misses_{0};
    uint64_t lru_evictions_count_{0};
};

} // namespace cudaforge::memory