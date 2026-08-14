#include "cudaforge/memory/buffer_pool.h"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <algorithm>
#include <cmath>

namespace cudaforge::memory {

BufferPool::BufferPool(BlockAllocator& allocator, size_t max_cached_bytes)
    : allocator_(allocator), max_cached_bytes_(max_cached_bytes) {
    spdlog::info("BufferPool initialized | Max Cached Capacity Limit: {:.2f} MB", static_cast<double>(max_cached_bytes_) / (1024.0 * 1024.0));
}

BufferPool::~BufferPool() {
    purge_free_buffers();
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_bytes_ > 0) {
        spdlog::warn("BufferPool destroyed with {:.2f} MB active allocations still unreleased!", static_cast<double>(active_bytes_) / (1024.0 * 1024.0));
    }
}

size_t BufferPool::calculate_bucket_size(size_t bytes) noexcept {
    if (bytes == 0) return 256;
    // Align to power of 2 (minimum 256 bytes)
    size_t power = 256;
    while (power < bytes) {
        power <<= 1;
    }
    return power;
}

bool BufferPool::attempt_allocation(size_t bucket_bytes, size_t requested_bytes, MemoryCategory category, cudaStream_t stream, void** out_ptr) {
    try {
        void* d_ptr = allocator_.allocate(bucket_bytes, category, stream);
        *out_ptr = d_ptr;
        
        PooledBufferNode node{
            .ptr = d_ptr,
            .capacity_bytes = bucket_bytes,
            .requested_bytes = requested_bytes,
            .category = category,
            .in_use = true,
            .last_used_frame = ++frame_counter_
        };

        pool_nodes_[d_ptr] = node;
        total_pooled_bytes_ += bucket_bytes;
        active_bytes_ += bucket_bytes;
        total_requested_bytes_ += requested_bytes;
        peak_pooled_bytes_ = std::max(peak_pooled_bytes_, total_pooled_bytes_);
        return true;
    } catch (const OutOfMemoryException& e) {
        *out_ptr = nullptr;
        return false;
    }
}

void* BufferPool::acquire(size_t bytes, MemoryCategory category, cudaStream_t stream) {
    if (bytes == 0) {
        return nullptr;
    }

    const size_t bucket_size = calculate_bucket_size(bytes);

    std::unique_lock<std::mutex> lock(mutex_);
    total_allocations_++;

    // 1. Search for a cached free buffer in the bucket
    auto it = free_buckets_.find(bucket_size);
    if (it != free_buckets_.end() && !it->second.empty()) {
        void* d_ptr = it->second.back();
        it->second.pop_back();

        auto node_it = pool_nodes_.find(d_ptr);
        node_it->second.in_use = true;
        node_it->second.requested_bytes = bytes;
        node_it->second.last_used_frame = ++frame_counter_;

        free_cached_bytes_ -= bucket_size;
        active_bytes_ += bucket_size;
        total_requested_bytes_ += bytes;
        pool_hits_++;

        spdlog::trace("BufferPool Hit | Address: {:p} | Bucket: {} KB", d_ptr, bucket_size / 1024);
        return d_ptr;
    }

    // 2. Pool Miss - Attempt Allocation via BlockAllocator
    pool_misses_++;
    void* d_ptr = nullptr;

    if (attempt_allocation(bucket_size, bytes, category, stream, &d_ptr)) {
        spdlog::trace("BufferPool Miss (Allocated) | Address: {:p} | Bucket: {} KB", d_ptr, bucket_size / 1024);
        return d_ptr;
    }

    // 3. Low-Level OOM Recovery Path
    spdlog::warn("BufferPool OOM Encountered. Initiating Recovery Purge for request size {:.2f} MB...", static_cast<double>(bytes) / (1024.0 * 1024.0));
    
    // Unlock mutex during purge to perform stream synchronizations
    lock.unlock();
    size_t purged_bytes = purge_free_buffers(stream);
    lock.lock();

    oom_purge_recoveries_++;

    // Retry allocation after cache purge
    if (attempt_allocation(bucket_size, bytes, category, stream, &d_ptr)) {
        spdlog::info("BufferPool Recovery Successful! Purged {:.2f} MB VRAM to satisfy {:.2f} MB request.",
                     static_cast<double>(purged_bytes) / (1024.0 * 1024.0), static_cast<double>(bytes) / (1024.0 * 1024.0));
        return d_ptr;
    }

    // 4. Exhaustion Failure -> Raise Explicit Exception
    std::string err_msg = fmt::format(
        "BufferPool OOM Failure: Exceeded GPU memory limits even after recovery purge!\n"
        "  Requested Payload:       {:.2f} MB\n"
        "  Bucket Target:          {:.2f} MB\n"
        "  Total Pooled VRAM:      {:.2f} MB\n"
        "  Active In-Use VRAM:     {:.2f} MB\n"
        "  Purged Bytes Freed:     {:.2f} MB",
        static_cast<double>(bytes) / (1024.0 * 1024.0),
        static_cast<double>(bucket_size) / (1024.0 * 1024.0),
        static_cast<double>(total_pooled_bytes_) / (1024.0 * 1024.0),
        static_cast<double>(active_bytes_) / (1024.0 * 1024.0),
        static_cast<double>(purged_bytes) / (1024.0 * 1024.0)
    );

    spdlog::critical("{}", err_msg);
    throw BufferPoolOOMException(err_msg);
}

void BufferPool::release(void* ptr, cudaStream_t stream) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto node_it = pool_nodes_.find(ptr);
    if (node_it == pool_nodes_.end()) {
        spdlog::error("BufferPool::release attempted on untracked pointer: {:p}", ptr);
        return;
    }

    PooledBufferNode& node = node_it->second;
    if (!node.in_use) {
        spdlog::warn("BufferPool::release called on pointer {:p} which is already free!", ptr);
        return;
    }

    node.in_use = false;
    active_bytes_ -= node.capacity_bytes;
    total_requested_bytes_ -= node.requested_bytes;
    free_cached_bytes_ += node.capacity_bytes;

    free_buckets_[node.capacity_bytes].push_back(ptr);

    // If cached VRAM exceeds max_cached_bytes_ threshold, trim oldest cached buffers
    if (free_cached_bytes_ > max_cached_bytes_) {
        spdlog::debug("BufferPool cache limit exceeded ({:.2f} MB > {:.2f} MB). Trimming...",
                      static_cast<double>(free_cached_bytes_) / (1024.0 * 1024.0),
                      static_cast<double>(max_cached_bytes_) / (1024.0 * 1024.0));
    }
}

size_t BufferPool::purge_free_buffers(cudaStream_t stream) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (stream != nullptr) {
        cudaStreamSynchronize(stream);
    }

    size_t bytes_purged = 0;

    for (auto& [bucket_size, ptr_vector] : free_buckets_) {
        for (void* ptr : ptr_vector) {
            auto node_it = pool_nodes_.find(ptr);
            if (node_it != pool_nodes_.end()) {
                size_t cap = node_it->second.capacity_bytes;
                pool_nodes_.erase(node_it);
                
                // Release via BlockAllocator
                allocator_.free(ptr, stream);
                
                bytes_purged += cap;
                total_pooled_bytes_ -= cap;
                free_cached_bytes_ -= cap;
            }
        }
        ptr_vector.clear();
    }

    free_buckets_.clear();

    spdlog::info("BufferPool Purged {} bytes back to BlockAllocator.", bytes_purged);
    return bytes_purged;
}

ScopedBuffer BufferPool::acquire_scoped(size_t bytes, MemoryCategory category, cudaStream_t stream) {
    void* ptr = acquire(bytes, category, stream);
    return ScopedBuffer(this, ptr, bytes, stream);
}

BufferPoolStats BufferPool::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    BufferPoolStats stats;
    stats.total_pooled_bytes = total_pooled_bytes_;
    stats.active_bytes = active_bytes_;
    stats.free_cached_bytes = free_cached_bytes_;
    stats.peak_pooled_bytes = peak_pooled_bytes_;
    stats.total_requested_bytes = total_requested_bytes_;

    stats.total_allocations = total_allocations_;
    stats.pool_hits = pool_hits_;
    stats.pool_misses = pool_misses_;
    stats.oom_purge_recoveries = oom_purge_recoveries_;

    // Calculate Internal Fragmentation
    if (active_bytes_ > 0) {
        stats.internal_fragmentation_ratio = 1.0 - (static_cast<double>(total_requested_bytes_) / static_cast<double>(active_bytes_));
    } else {
        stats.internal_fragmentation_ratio = 0.0;
    }

    // Calculate External Fragmentation in cached free buckets
    if (free_cached_bytes_ > 0) {
        size_t largest_free_block = 0;
        for (const auto& [bucket_size, vec] : free_buckets_) {
            if (!vec.empty()) {
                largest_free_block = std::max(largest_free_block, bucket_size);
            }
        }
        stats.external_fragmentation_ratio = 1.0 - (static_cast<double>(largest_free_block) / static_cast<double>(free_cached_bytes_));
    } else {
        stats.external_fragmentation_ratio = 0.0;
    }

    return stats;
}

void BufferPool::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    peak_pooled_bytes_ = total_pooled_bytes_;
    total_allocations_ = 0;
    pool_hits_ = 0;
    pool_misses_ = 0;
    oom_purge_recoveries_ = 0;
}

void BufferPool::dump_pool_state() const {
    BufferPoolStats stats = get_stats();

    spdlog::info(" Total Pooled Memory:    {:.2f} MB", static_cast<double>(stats.total_pooled_bytes) / (1024.0 * 1024.0));
    spdlog::info(" Active In-Use Memory:   {:.2f} MB", static_cast<double>(stats.active_bytes) / (1024.0 * 1024.0));
    spdlog::info(" Free Cached Memory:     {:.2f} MB", static_cast<double>(stats.free_cached_bytes) / (1024.0 * 1024.0));
    spdlog::info(" Peak Historical Pool:   {:.2f} MB", static_cast<double>(stats.peak_pooled_bytes) / (1024.0 * 1024.0));
    spdlog::info(" Internal Fragmentation: {:.2f}%", stats.internal_fragmentation_ratio * 100.0);
    spdlog::info(" External Fragmentation: {:.2f}%", stats.external_fragmentation_ratio * 100.0);
    spdlog::info(" Allocations Count:      {} (Hits: {}, Misses: {})", stats.total_allocations, stats.pool_hits, stats.pool_misses);
    spdlog::info(" OOM Purge Recoveries:   {}", stats.oom_purge_recoveries);
}

} // namespace cudaforge::memory