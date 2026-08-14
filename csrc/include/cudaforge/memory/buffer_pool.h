#pragma once

#include "cudaforge/memory/block_allocator.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cudaforge::memory {

/*
    Exception raised when buffer allocation fails even after multi-stage OOM
   purging.
 */
class BufferPoolOOMException : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/*
    Internal metadata descriptor for pooled GPU memory buffers.
 */
struct PooledBufferNode {
  void *ptr{nullptr};
  size_t capacity_bytes{0};
  size_t requested_bytes{0};
  MemoryCategory category{MemoryCategory::WORKSPACE};
  bool in_use{false};
  uint64_t last_used_frame{0};
};

/*
    Diagnostic telemetry snapshot for the buffer pool.
 */
struct BufferPoolStats {
  size_t total_pooled_bytes{0};
  size_t active_bytes{0};
  size_t free_cached_bytes{0};
  size_t peak_pooled_bytes{0};
  size_t total_requested_bytes{0};

  // Fragmentation metrics [0.0 - 1.0]
  double internal_fragmentation_ratio{0.0};
  double external_fragmentation_ratio{0.0};

  uint64_t total_allocations{0};
  uint64_t pool_hits{0};
  uint64_t pool_misses{0};
  uint32_t oom_purge_recoveries{0};
};

/*
    Forward declaration of RAII ScopedBuffer wrapper.
 */
class ScopedBuffer;

/*
    Stream-aware, bucketed GPU buffer pool with fragmentation metrics & OOM
   recovery.
 */
class BufferPool {
public:
  /**
   * @brief Construct buffer pool bound to a parent BlockAllocator instance.
   * @param allocator Reference to lower-level GPU BlockAllocator.
   * @param max_cached_bytes Max VRAM bytes allowed to stay cached in free
   * buckets (default 2GB).
   */
  explicit BufferPool(BlockAllocator &allocator,
                      size_t max_cached_bytes = 2ULL * 1024 * 1024 * 1024);
  ~BufferPool();

  // Non-copyable, non-movable
  BufferPool(const BufferPool &) = delete;
  BufferPool &operator=(const BufferPool &) = delete;
  BufferPool(BufferPool &&) = delete;
  BufferPool &operator=(BufferPool &&) = delete;

  /**
   * @brief Acquires a reusable buffer from pool or allocates new VRAM via
   * BlockAllocator. Executes emergency purging if BlockAllocator throws an OOM
   * exception.
   * @param bytes Payload size requested.
   * @param category Memory classification.
   * @param stream CUDA stream context.
   * @return Device pointer to allocated/reused memory.
   */
  [[nodiscard]] void *
  acquire(size_t bytes, MemoryCategory category = MemoryCategory::WORKSPACE,
          cudaStream_t stream = nullptr);

  /**
   * @brief Releases an active buffer back into the free cache pool.
   * @param ptr Device pointer to return to pool.
   * @param stream CUDA stream context.
   */
  void release(void *ptr, cudaStream_t stream = nullptr);

  /**
   * @brief Flushes unused cached free buffers back to BlockAllocator to free
   * physical VRAM.
   * @param stream Optional CUDA stream to synchronize before releasing.
   * @return Number of bytes purged back to driver.
   */
  size_t purge_free_buffers(cudaStream_t stream = nullptr);

  /**
   * @brief Creates an RAII ScopedBuffer container that auto-releases upon going
   * out of scope.
   */
  [[nodiscard]] ScopedBuffer
  acquire_scoped(size_t bytes,
                 MemoryCategory category = MemoryCategory::WORKSPACE,
                 cudaStream_t stream = nullptr);

  /**
   * @brief Generates current metric & fragmentation statistics.
   */
  [[nodiscard]] BufferPoolStats get_stats() const;

  /**
   * @brief Resets peak VRAM metrics and allocation counters.
   */
  void reset_stats();

  /**
   * @brief Prints diagnostic buffer pool breakdown via spdlog.
   */
  void dump_pool_state() const;

private:
  [[nodiscard]] static size_t calculate_bucket_size(size_t bytes) noexcept;
  [[nodiscard]] bool attempt_allocation(size_t bucket_bytes,
                                        size_t requested_bytes,
                                        MemoryCategory category,
                                        cudaStream_t stream, void **out_ptr);

  BlockAllocator &allocator_;
  size_t max_cached_bytes_;

  mutable std::mutex mutex_;
  std::unordered_map<void *, PooledBufferNode> pool_nodes_;

  // Bucket sizes -> list of free device pointers
  std::unordered_map<size_t, std::vector<void *>> free_buckets_;

  size_t total_pooled_bytes_{0};
  size_t active_bytes_{0};
  size_t free_cached_bytes_{0};
  size_t peak_pooled_bytes_{0};
  size_t total_requested_bytes_{0};

  uint64_t total_allocations_{0};
  uint64_t pool_hits_{0};
  uint64_t pool_misses_{0};
  uint32_t oom_purge_recoveries_{0};
  uint64_t frame_counter_{0};
};

/*
    RAII Guard for managing automatic BufferPool acquisitions and releases.
 */
class ScopedBuffer {
public:
  ScopedBuffer() = default;
  ScopedBuffer(BufferPool *pool, void *ptr, size_t bytes,
               cudaStream_t stream = nullptr)
      : pool_(pool), ptr_(ptr), bytes_(bytes), stream_(stream) {}

  ~ScopedBuffer() { release(); }

  // Move-only semantics
  ScopedBuffer(const ScopedBuffer &) = delete;
  ScopedBuffer &operator=(const ScopedBuffer &) = delete;

  ScopedBuffer(ScopedBuffer &&other) noexcept
      : pool_(other.pool_), ptr_(other.ptr_), bytes_(other.bytes_),
        stream_(other.stream_) {
    other.pool_ = nullptr;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
    other.stream_ = nullptr;
  }

  ScopedBuffer &operator=(ScopedBuffer &&other) noexcept {
    if (this != &other) {
      release();
      pool_ = other.pool_;
      ptr_ = other.ptr_;
      bytes_ = other.bytes_;
      stream_ = other.stream_;
      other.pool_ = nullptr;
      other.ptr_ = nullptr;
      other.bytes_ = 0;
      other.stream_ = nullptr;
    }
    return *this;
  }

  void release() {
    if (pool_ && ptr_) {
      pool_->release(ptr_, stream_);
      pool_ = nullptr;
      ptr_ = nullptr;
      bytes_ = 0;
    }
  }

  [[nodiscard]] void *get() const noexcept { return ptr_; }

  template <typename T> [[nodiscard]] T *get_typed() const noexcept {
    return static_cast<T *>(ptr_);
  }

  [[nodiscard]] size_t size_bytes() const noexcept { return bytes_; }

private:
  BufferPool *pool_{nullptr};
  void *ptr_{nullptr};
  size_t bytes_{0};
  cudaStream_t stream_{nullptr};
};

} // namespace cudaforge::memory