#include "cudaforge/memory/paged_kv_cache.h"
#include <algorithm>
#include <cstring>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

namespace cudaforge::memory {

PagedKVCache::PagedKVCache(const KVCacheConfig &config,
                           BlockAllocator &allocator)
    : config_(config), allocator_(allocator) {

  layer_block_bytes_ = static_cast<size_t>(config_.num_kv_heads) *
                       config_.block_size * config_.head_dim *
                       config_.element_size_bytes;

  bytes_per_block_ = config_.num_layers * layer_block_bytes_;
  const size_t total_pool_bytes = config_.total_gpu_blocks * bytes_per_block_;

  d_key_pool_ = allocator_.allocate(total_pool_bytes, MemoryCategory::KV_CACHE);
  d_val_pool_ = allocator_.allocate(total_pool_bytes, MemoryCategory::KV_CACHE);

  // Allocate Host Paging Buffers (Page-Locked Pinned Memory for fast PCIe DMA)
  const size_t max_table_elements =
      config_.max_supported_batch_size * config_.max_blocks_per_sequence;
  CUDA_CHECK_METADATA(cudaHostAlloc(
      reinterpret_cast<void **>(&h_pinned_block_tables_),
      max_table_elements * sizeof(int32_t), cudaHostAllocWriteCombined));
  CUDA_CHECK_METADATA(
      cudaHostAlloc(reinterpret_cast<void **>(&h_pinned_context_lens_),
                    config_.max_supported_batch_size * sizeof(int32_t),
                    cudaHostAllocDefault));

  // Allocate Device Staging Memory for CUDA Kernel Consumption
  CUDA_CHECK_METADATA(cudaMalloc(reinterpret_cast<void **>(&d_block_tables_),
                                 max_table_elements * sizeof(int32_t)));
  CUDA_CHECK_METADATA(
      cudaMalloc(reinterpret_cast<void **>(&d_context_lens_),
                 config_.max_supported_batch_size * sizeof(int32_t)));

  physical_blocks_.resize(config_.total_gpu_blocks);
  free_block_stack_.reserve(config_.total_gpu_blocks);

  for (int32_t i = static_cast<int32_t>(config_.total_gpu_blocks) - 1; i >= 0;
       --i) {
    physical_blocks_[i].block_id = i;
    physical_blocks_[i].ref_count = 0;
    physical_blocks_[i].num_tokens_filled = 0;
    physical_blocks_[i].hash_value = 0;
    physical_blocks_[i].is_cached = false;
    physical_blocks_[i].lru_iterator = lru_list_.end();
    free_block_stack_.push_back(i);
  }

  prefix_tree_root_ = std::make_shared<PrefixTreeNode>();

  spdlog::info("PagedKVCache Initialized with GPU Metadata Staging Engine:");
  spdlog::info("  Max Batch Size:       {}", config_.max_supported_batch_size);
  spdlog::info("  Max Blocks per Seq:   {}", config_.max_blocks_per_sequence);
}

PagedKVCache::~PagedKVCache() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (h_pinned_block_tables_)
    cudaFreeHost(h_pinned_block_tables_);
  if (h_pinned_context_lens_)
    cudaFreeHost(h_pinned_context_lens_);
  if (d_block_tables_)
    cudaFree(d_block_tables_);
  if (d_context_lens_)
    cudaFree(d_context_lens_);

  if (d_key_pool_)
    allocator_.free(d_key_pool_);
  if (d_val_pool_)
    allocator_.free(d_val_pool_);
}

uint64_t PagedKVCache::compute_block_hash(std::span<const int32_t> tokens,
                                          uint64_t parent_hash) noexcept {
  uint64_t hash = parent_hash ^ 0x9e3779b97f4a7c15ULL;
  for (int32_t token : tokens) {
    hash ^=
        std::hash<int32_t>{}(token) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  return hash;
}

void PagedKVCache::touch_lru(int32_t block_id) {
  PhysicalBlock &block = physical_blocks_[block_id];
  if (block.is_cached) {
    if (block.lru_iterator != lru_list_.end()) {
      lru_list_.erase(block.lru_iterator);
    }
    lru_list_.push_front(block_id);
    block.lru_iterator = lru_list_.begin();
  }
}

bool PagedKVCache::evict_lru_block() {
  if (lru_list_.empty())
    return false;

  for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
    int32_t block_id = *it;
    PhysicalBlock &block = physical_blocks_[block_id];

    if (block.ref_count == 0) {
      hash_to_node_map_.erase(block.hash_value);
      lru_list_.erase(std::next(it).base());

      block.is_cached = false;
      block.hash_value = 0;
      block.num_tokens_filled = 0;
      block.lru_iterator = lru_list_.end();

      free_block_stack_.push_back(block_id);
      lru_evictions_count_++;
      return true;
    }
  }
  return false;
}

int32_t PagedKVCache::allocate_physical_block() {
  if (free_block_stack_.empty()) {
    if (!evict_lru_block()) {
      throw OutOfPagesException(
          fmt::format("OutOfPagesException: Physical KV Cache pool exhausted! "
                      "All {} blocks in active use.",
                      config_.total_gpu_blocks));
    }
  }

  int32_t block_id = free_block_stack_.back();
  free_block_stack_.pop_back();

  PhysicalBlock &block = physical_blocks_[block_id];
  block.ref_count = 1;
  block.num_tokens_filled = 0;
  block.is_cached = false;
  return block_id;
}

void PagedKVCache::free_physical_block(int32_t block_id) {
  if (block_id < 0 || static_cast<size_t>(block_id) >= config_.total_gpu_blocks)
    return;

  PhysicalBlock &block = physical_blocks_[block_id];
  if (block.ref_count == 0)
    return;

  block.ref_count--;
  if (block.ref_count == 0) {
    if (config_.enable_prefix_caching && block.hash_value != 0) {
      block.is_cached = true;
      lru_list_.push_front(block_id);
      block.lru_iterator = lru_list_.begin();
    } else {
      block.num_tokens_filled = 0;
      free_block_stack_.push_back(block_id);
    }
  }
}

uint32_t PagedKVCache::register_sequence_with_prefix(
    uint64_t sequence_id, std::span<const int32_t> prompt_tokens) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (page_tables_.find(sequence_id) != page_tables_.end())
    return 0;

  const uint32_t total_tokens = static_cast<uint32_t>(prompt_tokens.size());
  const uint32_t full_blocks = total_tokens / config_.block_size;

  std::vector<int32_t> block_table;
  uint32_t matched_tokens = 0;
  uint64_t current_parent_hash = 0;
  auto current_tree_node = prefix_tree_root_;

  if (config_.enable_prefix_caching) {
    for (uint32_t b = 0; b < full_blocks; ++b) {
      auto block_tokens =
          prompt_tokens.subspan(b * config_.block_size, config_.block_size);
      uint64_t block_hash =
          compute_block_hash(block_tokens, current_parent_hash);

      auto child_it = current_tree_node->children.find(block_hash);
      if (child_it != current_tree_node->children.end()) {
        int32_t cached_block_id = child_it->second->physical_block_id;
        PhysicalBlock &cached_block = physical_blocks_[cached_block_id];

        cached_block.ref_count++;
        if (cached_block.is_cached) {
          cached_block.is_cached = false;
          if (cached_block.lru_iterator != lru_list_.end()) {
            lru_list_.erase(cached_block.lru_iterator);
            cached_block.lru_iterator = lru_list_.end();
          }
        }

        block_table.push_back(cached_block_id);
        matched_tokens += config_.block_size;
        prefix_cache_hits_++;

        current_parent_hash = block_hash;
        current_tree_node = child_it->second;
      } else {
        break;
      }
    }
  }

  uint32_t remaining_tokens = total_tokens - matched_tokens;
  uint32_t un_matched_full_blocks = remaining_tokens / config_.block_size;
  uint32_t tail_tokens = remaining_tokens % config_.block_size;
  uint32_t start_block_idx = matched_tokens / config_.block_size;

  for (uint32_t b = 0; b < un_matched_full_blocks; ++b) {
    uint32_t block_idx = start_block_idx + b;
    auto block_tokens = prompt_tokens.subspan(block_idx * config_.block_size,
                                              config_.block_size);
    uint64_t block_hash = compute_block_hash(block_tokens, current_parent_hash);

    int32_t new_block_id = allocate_physical_block();
    PhysicalBlock &new_block = physical_blocks_[new_block_id];
    new_block.num_tokens_filled = config_.block_size;
    new_block.hash_value = block_hash;

    auto new_tree_node = std::make_shared<PrefixTreeNode>();
    new_tree_node->hash_value = block_hash;
    new_tree_node->physical_block_id = new_block_id;

    current_tree_node->children[block_hash] = new_tree_node;
    hash_to_node_map_[block_hash] = new_tree_node;

    block_table.push_back(new_block_id);
    prefix_cache_misses_++;

    current_parent_hash = block_hash;
    current_tree_node = new_tree_node;
  }

  if (tail_tokens > 0 || block_table.empty()) {
    int32_t tail_block_id = allocate_physical_block();
    physical_blocks_[tail_block_id].num_tokens_filled = tail_tokens;
    block_table.push_back(tail_block_id);
  }

  page_tables_[sequence_id] = std::move(block_table);
  return matched_tokens;
}

void PagedKVCache::append_tokens(uint64_t sequence_id, uint32_t num_tokens) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = page_tables_.find(sequence_id);
  if (it == page_tables_.end()) {
    throw std::runtime_error(
        fmt::format("Sequence ID {} not registered!", sequence_id));
  }

  std::vector<int32_t> &block_table = it->second;
  int32_t last_block_id = block_table.back();
  PhysicalBlock &last_block = physical_blocks_[last_block_id];

  uint32_t available = config_.block_size - last_block.num_tokens_filled;

  if (num_tokens <= available) {
    last_block.num_tokens_filled += num_tokens;
  } else {
    uint32_t remaining = num_tokens - available;
    last_block.num_tokens_filled = config_.block_size;
    uint32_t additional_blocks =
        (remaining + config_.block_size - 1) / config_.block_size;

    for (uint32_t i = 0; i < additional_blocks; ++i) {
      int32_t new_block_id = allocate_physical_block();
      uint32_t added = std::min(remaining, config_.block_size);
      physical_blocks_[new_block_id].num_tokens_filled = added;
      remaining -= added;
      block_table.push_back(new_block_id);
    }
  }
}

void PagedKVCache::unregister_sequence(uint64_t sequence_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = page_tables_.find(sequence_id);
  if (it == page_tables_.end())
    return;

  for (int32_t block_id : it->second) {
    free_physical_block(block_id);
  }
  page_tables_.erase(it);
}

void PagedKVCache::fork_sequence(uint64_t parent_sequence_id,
                                 uint64_t child_sequence_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto parent_it = page_tables_.find(parent_sequence_id);
  if (parent_it == page_tables_.end()) {
    throw std::runtime_error(
        fmt::format("Parent Sequence ID {} not found!", parent_sequence_id));
  }

  const auto &parent_blocks = parent_it->second;
  for (int32_t block_id : parent_blocks) {
    physical_blocks_[block_id].ref_count++;
  }
  page_tables_[child_sequence_id] = parent_blocks;
}

KVCacheBatchDescriptor PagedKVCache::prepare_batch_metadata(
    const std::vector<uint64_t> &active_sequence_ids, cudaStream_t stream) {

  std::lock_guard<std::mutex> lock(mutex_);

  const size_t batch_size = active_sequence_ids.size();
  if (batch_size > config_.max_supported_batch_size) {
    throw std::out_of_range(
        "Requested batch size exceeds max_supported_batch_size");
  }

  size_t max_blocks_in_batch = 0;
  for (uint64_t seq_id : active_sequence_ids) {
    auto it = page_tables_.find(seq_id);
    if (it == page_tables_.end()) {
      throw std::runtime_error(fmt::format(
          "prepare_batch_metadata: Sequence ID {} not found!", seq_id));
    }
    max_blocks_in_batch = std::max(max_blocks_in_batch, it->second.size());
  }

  if (max_blocks_in_batch > config_.max_blocks_per_sequence) {
    throw std::out_of_range(
        "Sequence blocks count exceeds max_blocks_per_sequence limit");
  }

  // 1. Flatten host-side block tables and compute context lengths in pinned
  // memory
  std::memset(h_pinned_block_tables_, -1,
              batch_size * max_blocks_in_batch * sizeof(int32_t));

  for (size_t i = 0; i < batch_size; ++i) {
    uint64_t seq_id = active_sequence_ids[i];
    const auto &blocks = page_tables_[seq_id];

    // Fill row i of block table
    for (size_t b = 0; b < blocks.size(); ++b) {
      h_pinned_block_tables_[i * max_blocks_in_batch + b] = blocks[b];
    }

    // Total context length calculation
    uint32_t full_tokens =
        static_cast<uint32_t>(blocks.size() - 1) * config_.block_size;
    uint32_t last_block_tokens =
        physical_blocks_[blocks.back()].num_tokens_filled;
    h_pinned_context_lens_[i] =
        static_cast<int32_t>(full_tokens + last_block_tokens);
  }

  // 2. Asynchronously copy metadata from Pinned Host Memory -> Device Staging
  // Buffers
  const size_t bytes_block_table =
      batch_size * max_blocks_in_batch * sizeof(int32_t);
  const size_t bytes_context_lens = batch_size * sizeof(int32_t);

  CUDA_CHECK_METADATA(cudaMemcpyAsync(d_block_tables_, h_pinned_block_tables_,
                                      bytes_block_table, cudaMemcpyHostToDevice,
                                      stream));

  CUDA_CHECK_METADATA(cudaMemcpyAsync(d_context_lens_, h_pinned_context_lens_,
                                      bytes_context_lens,
                                      cudaMemcpyHostToDevice, stream));

  // 3. Assemble and return device descriptor struct
  KVCacheBatchDescriptor desc;
  desc.block_tables = d_block_tables_;
  desc.context_lens = d_context_lens_;
  desc.batch_size = static_cast<int32_t>(batch_size);
  desc.max_blocks_per_seq = static_cast<int32_t>(max_blocks_in_batch);
  desc.block_size = static_cast<int32_t>(config_.block_size);
  desc.num_kv_heads = static_cast<int32_t>(config_.num_kv_heads);
  desc.head_dim = static_cast<int32_t>(config_.head_dim);
  desc.bytes_per_block = bytes_per_block_;
  desc.layer_block_bytes = layer_block_bytes_;
  desc.key_pool_ptr = d_key_pool_;
  desc.val_pool_ptr = d_val_pool_;

  return desc;
}

const std::vector<int32_t> &
PagedKVCache::get_block_table(uint64_t sequence_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = page_tables_.find(sequence_id);
  if (it == page_tables_.end()) {
    throw std::runtime_error(
        fmt::format("Sequence ID {} not found!", sequence_id));
  }
  return it->second;
}

void *PagedKVCache::get_key_block_ptr(int32_t block_id,
                                      uint32_t layer_idx) const {
  size_t block_offset = (static_cast<size_t>(block_id) * bytes_per_block_) +
                        (layer_idx * layer_block_bytes_);
  return static_cast<uint8_t *>(d_key_pool_) + block_offset;
}

void *PagedKVCache::get_value_block_ptr(int32_t block_id,
                                        uint32_t layer_idx) const {
  size_t block_offset = (static_cast<size_t>(block_id) * bytes_per_block_) +
                        (layer_idx * layer_block_bytes_);
  return static_cast<uint8_t *>(d_val_pool_) + block_offset;
}

KVCacheStats PagedKVCache::get_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);

  KVCacheStats stats;
  stats.total_blocks = config_.total_gpu_blocks;
  stats.free_blocks = free_block_stack_.size();
  stats.cached_prefix_blocks = lru_list_.size();
  stats.allocated_blocks =
      stats.total_blocks - stats.free_blocks - stats.cached_prefix_blocks;
  stats.total_memory_bytes = stats.total_blocks * bytes_per_block_ * 2;
  stats.used_memory_bytes =
      (stats.allocated_blocks + stats.cached_prefix_blocks) * bytes_per_block_ *
      2;
  stats.pool_utilization_ratio =
      static_cast<double>(stats.total_blocks - stats.free_blocks) /
      static_cast<double>(stats.total_blocks);
  stats.active_sequences_count = page_tables_.size();
  stats.prefix_cache_hits = prefix_cache_hits_;
  stats.prefix_cache_misses = prefix_cache_misses_;
  stats.lru_evictions_count = lru_evictions_count_;

  return stats;
}

} // namespace cudaforge::memory