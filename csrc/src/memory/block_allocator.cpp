#include "cudaforge/memory/block_allocator.h"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <algorithm>

namespace cudaforge::memory {

BlockAllocator::BlockAllocator(double vram_headroom_fraction) {
    size_t free_vram = 0;
    cudaError_t err = cudaMemGetInfo(&free_vram, &total_vram_bytes_);
    if (err != cudaSuccess) {
        spdlog::critical("BlockAllocator: Failed to query CUDA driver memory: {}", cudaGetErrorString(err));
        throw std::runtime_error("CUDA Driver query failed during BlockAllocator initialization.");
    }

    // Apply safety headroom threshold (e.g., 95% maximum usable VRAM)
    double usable_fraction = std::clamp(1.0 - vram_headroom_fraction, 0.50, 0.99);
    vram_safety_threshold_bytes_ = static_cast<size_t>(static_cast<double>(total_vram_bytes_) * usable_fraction);

    spdlog::info(
        "BlockAllocator initialized | Total VRAM: {:.2f} GB | Safety Limit: {:.2f} GB (Headroom: {:.1f}%)",
        static_cast<double>(total_vram_bytes_) / (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(vram_safety_threshold_bytes_) / (1024.0 * 1024.0 * 1024.0),
        vram_headroom_fraction * 100.0
    );
}

BlockAllocator::~BlockAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_allocations_.empty()) {
        spdlog::warn(
            "BlockAllocator destructed with {} active unfreed allocations totaling {:.2f} MB!",
            active_allocations_.size(),
            static_cast<double>(total_allocated_bytes_) / (1024.0 * 1024.0)
        );
    }
}

size_t BlockAllocator::align_bytes(size_t bytes, size_t alignment) noexcept {
    return (bytes + alignment - 1) & ~(alignment - 1);
}

void BlockAllocator::verify_cuda_error(cudaError_t err, std::string_view action_desc) const {
    if (err != cudaSuccess) {
        std::string err_msg = fmt::format(
            "CUDA Error during '{}': {} (code {})", action_desc, cudaGetErrorString(err), static_cast<int>(err)
        );
        spdlog::error("{}", err_msg);
        throw std::runtime_error(err_msg);
    }
}

void* BlockAllocator::allocate(size_t bytes, MemoryCategory category, cudaStream_t stream) {
    if (bytes == 0) {
        return nullptr;
    }

    const size_t aligned_bytes = align_bytes(bytes, 256);

    std::unique_lock<std::mutex> lock(mutex_);

    // Check pre-allocation budget against safety threshold
    const size_t projected_allocated = total_allocated_bytes_ + aligned_bytes;
    if (projected_allocated > vram_safety_threshold_bytes_) {
        size_t actual_free_bytes = 0;
        size_t actual_total_bytes = 0;
        cudaMemGetInfo(&actual_free_bytes, &actual_total_bytes);

        std::string oom_detail = fmt::format(
            "CUDAForge OOM Safeguard Triggered!\n"
            "  Requested:              {:.2f} MB (Aligned: {:.2f} MB)\n"
            "  Category:               {}\n"
            "  Currently Allocated:    {:.2f} MB / Safety Limit: {:.2f} MB\n"
            "  Physical VRAM Free:     {:.2f} MB / Total: {:.2f} MB\n"
            "  Category Breakdown:\n"
            "    - WEIGHTS:            {:.2f} MB\n"
            "    - ACTIVATIONS:        {:.2f} MB\n"
            "    - WORKSPACE:          {:.2f} MB\n"
            "    - KV_CACHE:           {:.2f} MB",
            static_cast<double>(bytes) / (1024.0 * 1024.0),
            static_cast<double>(aligned_bytes) / (1024.0 * 1024.0),
            category_to_string(category),
            static_cast<double>(total_allocated_bytes_) / (1024.0 * 1024.0),
            static_cast<double>(vram_safety_threshold_bytes_) / (1024.0 * 1024.0),
            static_cast<double>(actual_free_bytes) / (1024.0 * 1024.0),
            static_cast<double>(actual_total_bytes) / (1024.0 * 1024.0),
            static_cast<double>(category_allocated_bytes_[static_cast<size_t>(MemoryCategory::WEIGHTS)]) / (1024.0 * 1024.0),
            static_cast<double>(category_allocated_bytes_[static_cast<size_t>(MemoryCategory::ACTIVATIONS)]) / (1024.0 * 1024.0),
            static_cast<double>(category_allocated_bytes_[static_cast<size_t>(MemoryCategory::WORKSPACE)]) / (1024.0 * 1024.0),
            static_cast<double>(category_allocated_bytes_[static_cast<size_t>(MemoryCategory::KV_CACHE)]) / (1024.0 * 1024.0)
        );

        spdlog::critical("{}", oom_detail);
        throw OutOfMemoryException(oom_detail);
    }

    lock.unlock(); // Release lock while making driver call to prevent blocking other threads

    void* d_ptr = nullptr;
    cudaError_t err = cudaMallocAsync(&d_ptr, aligned_bytes, stream);

    if (err != cudaSuccess) {
        std::string err_msg = fmt::format(
            "cudaMallocAsync failed for {:.2f} MB in category '{}': {}",
            static_cast<double>(aligned_bytes) / (1024.0 * 1024.0),
            category_to_string(category),
            cudaGetErrorString(err)
        );
        spdlog::error("{}", err_msg);
        throw OutOfMemoryException(err_msg);
    }

    lock.lock();

    // Record handle and update tracking stats
    BlockHandle handle{
        .ptr = d_ptr,
        .size_bytes = aligned_bytes,
        .category = category,
        .stream = stream
    };

    active_allocations_[d_ptr] = handle;
    total_allocated_bytes_ += aligned_bytes;
    category_allocated_bytes_[static_cast<size_t>(category)] += aligned_bytes;
    peak_allocated_bytes_ = std::max(peak_allocated_bytes_, total_allocated_bytes_);

    spdlog::trace(
        "Allocated block {:p} | Size: {:.2f} KB | Category: {}",
        d_ptr, static_cast<double>(aligned_bytes) / 1024.0, category_to_string(category)
    );

    return d_ptr;
}

void BlockAllocator::free(void* ptr, cudaStream_t stream) {
    if (ptr == nullptr) {
        return;
    }

    std::unique_lock<std::mutex> lock(mutex_);

    auto it = active_allocations_.find(ptr);
    if (it == active_allocations_.end()) {
        spdlog::error("BlockAllocator::free attempted to deallocate untracked pointer: {:p}", ptr);
        return;
    }

    BlockHandle handle = it->second;
    active_allocations_.erase(it);

    total_allocated_bytes_ -= handle.size_bytes;
    category_allocated_bytes_[static_cast<size_t>(handle.category)] -= handle.size_bytes;

    lock.unlock(); // Unlock before issuing asynchronous deallocation call

    cudaError_t err = cudaFreeAsync(ptr, stream);
    if (err != cudaSuccess) {
        spdlog::error("cudaFreeAsync failed for pointer {:p}: {}", ptr, cudaGetErrorString(err));
    } else {
        spdlog::trace("Freed block {:p} | Released: {:.2f} KB", ptr, static_cast<double>(handle.size_bytes) / 1024.0);
    }
}

MemoryStats BlockAllocator::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t free_vram = 0;
    size_t total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);

    MemoryStats stats;
    stats.total_vram_bytes = total_vram;
    stats.free_vram_bytes = free_vram;
    stats.currently_allocated_bytes = total_allocated_bytes_;
    stats.peak_allocated_bytes = peak_allocated_bytes_;
    stats.safety_limit_bytes = vram_safety_threshold_bytes_;
    stats.category_bytes = category_allocated_bytes_;

    return stats;
}

size_t BlockAllocator::get_category_bytes(MemoryCategory category) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return category_allocated_bytes_[static_cast<size_t>(category)];
}

size_t BlockAllocator::get_total_allocated_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_bytes_;
}

size_t BlockAllocator::get_peak_allocated_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_allocated_bytes_;
}

void BlockAllocator::reset_peak_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    peak_allocated_bytes_ = total_allocated_bytes_;
}

void BlockAllocator::dump_memory_map() const {
    MemoryStats stats = get_stats();
    
    spdlog::info(" Physical VRAM Total:      {:.2f} GB", static_cast<double>(stats.total_vram_bytes) / (1024.0 * 1024.0 * 1024.0));
    spdlog::info(" Physical VRAM Free:       {:.2f} GB", static_cast<double>(stats.free_vram_bytes) / (1024.0 * 1024.0 * 1024.0));
    spdlog::info(" Allocator Limit Ceiling:  {:.2f} GB", static_cast<double>(stats.safety_limit_bytes) / (1024.0 * 1024.0 * 1024.0));
    spdlog::info(" Active Allocated Total:   {:.2f} MB", static_cast<double>(stats.currently_allocated_bytes) / (1024.0 * 1024.0));
    spdlog::info(" Peak Allocated Historical: {:.2f} MB", static_cast<double>(stats.peak_allocated_bytes) / (1024.0 * 1024.0));
    spdlog::info(" Category Breakdown:");
    spdlog::info("   - WEIGHTS:             {:.2f} MB", static_cast<double>(stats.category_bytes[static_cast<size_t>(MemoryCategory::WEIGHTS)]) / (1024.0 * 1024.0));
    spdlog::info("   - ACTIVATIONS:         {:.2f} MB", static_cast<double>(stats.category_bytes[static_cast<size_t>(MemoryCategory::ACTIVATIONS)]) / (1024.0 * 1024.0));
    spdlog::info("   - WORKSPACE:           {:.2f} MB", static_cast<double>(stats.category_bytes[static_cast<size_t>(MemoryCategory::WORKSPACE)]) / (1024.0 * 1024.0));
    spdlog::info("   - KV_CACHE:            {:.2f} MB", static_cast<double>(stats.category_bytes[static_cast<size_t>(MemoryCategory::KV_CACHE)]) / (1024.0 * 1024.0));
}

} // namespace cudaforge::memory