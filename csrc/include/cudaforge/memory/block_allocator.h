#pragma once

#include <array>
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

namespace cudaforge::memory {

/*
    Categorization of GPU memory allocations for granular VRAM tracking.
 */
enum class MemoryCategory : uint8_t {
  WEIGHTS = 0,
  ACTIVATIONS = 1,
  WORKSPACE = 2,
  KV_CACHE = 3,
  NUM_CATEGORIES = 4
};

/*
    Helper to convert MemoryCategory enum to string representation.
 */
[[nodiscard]] constexpr std::string_view
category_to_string(MemoryCategory category) noexcept {
  switch (category) {
  case MemoryCategory::WEIGHTS:
    return "WEIGHTS";
  case MemoryCategory::ACTIVATIONS:
    return "ACTIVATIONS";
  case MemoryCategory::WORKSPACE:
    return "WORKSPACE";
  case MemoryCategory::KV_CACHE:
    return "KV_CACHE";
  default:
    return "UNKNOWN";
  }
}

/*
    Exception thrown when an allocation request exceeds the configured VRAM
   safety budget.
 */
class OutOfMemoryException : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/*
    Metadata tracking block stored with active GPU allocations.
 */
struct BlockHandle {
  void *ptr{nullptr};
  size_t size_bytes{0};
  MemoryCategory category{MemoryCategory::WORKSPACE};
  cudaStream_t stream{nullptr};
};

/*
    Consolidated snapshot of VRAM state and category breakdowns.
 */
struct MemoryStats {
  size_t total_vram_bytes{0};
  size_t free_vram_bytes{0};
  size_t currently_allocated_bytes{0};
  size_t peak_allocated_bytes{0};
  size_t safety_limit_bytes{0};
  std::array<size_t, static_cast<size_t>(MemoryCategory::NUM_CATEGORIES)>
      category_bytes{};
};

/*
    Thread-safe low-level GPU memory tracker and stream-ordered allocator.
 */
class BlockAllocator {
public:
  /**
   * @brief Construct allocator with default or custom VRAM headroom limit.
   * @param vram_headroom_fraction Percentage of VRAM reserved for CUDA driver
   * overhead (default 5% -> 95% ceiling).
   */
  explicit BlockAllocator(double vram_headroom_fraction = 0.05);
  ~BlockAllocator();

  // Non-copyable, non-movable
  BlockAllocator(const BlockAllocator &) = delete;
  BlockAllocator &operator=(const BlockAllocator &) = delete;
  BlockAllocator(BlockAllocator &&) = delete;
  BlockAllocator &operator=(BlockAllocator &&) = delete;

  /**
   * @brief Allocates stream-ordered asynchronous memory aligned to 256 bytes.
   * @param bytes Number of bytes to allocate.
   * @param category Memory classification for tracker context.
   * @param stream CUDA stream associated with execution context (default
   * nullptr / default stream).
   * @return Raw GPU void pointer to memory block.
   * @throws OutOfMemoryException if budget or physical VRAM is exceeded.
   */
  [[nodiscard]] void *allocate(size_t bytes, MemoryCategory category,
                               cudaStream_t stream = nullptr);

  /**
   * @brief Strongly typed allocator wrapper for element arrays.
   */
  template <typename T>
  [[nodiscard]] T *allocate_typed(size_t count, MemoryCategory category,
                                  cudaStream_t stream = nullptr) {
    return static_cast<T *>(allocate(count * sizeof(T), category, stream));
  }

  /**
   * @brief Deallocates memory asynchronously on the specified CUDA stream.
   * @param ptr Device pointer to release.
   * @param stream CUDA stream associated with deallocation.
   */
  void free(void *ptr, cudaStream_t stream = nullptr);

  /**
   * @brief Queries physical VRAM from CUDA driver and updates internal stats
   * snapshot.
   */
  [[nodiscard]] MemoryStats get_stats() const;

  /**
   * @brief Returns current allocated bytes for a given memory category.
   */
  [[nodiscard]] size_t get_category_bytes(MemoryCategory category) const;

  /**
   * @brief Returns current total allocated bytes across all categories.
   */
  [[nodiscard]] size_t get_total_allocated_bytes() const;

  /**
   * @brief Returns the historical peak allocated bytes since reset.
   */
  [[nodiscard]] size_t get_peak_allocated_bytes() const;

  /**
   * @brief Resets the peak allocated metric to current allocated levels.
   */
  void reset_peak_stats();

  /**
   * @brief Prints structured memory map details via spdlog.
   */
  void dump_memory_map() const;

private:
  [[nodiscard]] static size_t align_bytes(size_t bytes,
                                          size_t alignment = 256) noexcept;
  void verify_cuda_error(cudaError_t err, std::string_view action_desc) const;

  mutable std::mutex mutex_;
  std::unordered_map<void *, BlockHandle> active_allocations_;

  size_t total_vram_bytes_{0};
  size_t vram_safety_threshold_bytes_{0};
  size_t total_allocated_bytes_{0};
  size_t peak_allocated_bytes_{0};

  std::array<size_t, static_cast<size_t>(MemoryCategory::NUM_CATEGORIES)>
      category_allocated_bytes_{};
};

} // namespace cudaforge::memory