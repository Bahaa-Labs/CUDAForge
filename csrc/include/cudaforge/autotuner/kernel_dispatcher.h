#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>
#include <mutex>
#include <iostream>
#include <chrono>

namespace cudaforge::autotuner {

/*
    Unique Key structure representing problem dimensions and type configurations.
 */
struct KernelKey {
    std::string op_name;
    int32_t M{0};
    int32_t N{0};
    int32_t K{0};
    int32_t dtype_code{0}; // 0: FP16, 1: BF16, 2: INT4

    bool operator==(const KernelKey& other) const {
        return op_name == other.op_name &&
               M == other.M &&
               N == other.N &&
               K == other.K &&
               dtype_code == other.dtype_code;
    }
};

struct KernelKeyHash {
    std::size_t operator()(const KernelKey& k) const {
        std::size_t h1 = std::hash<std::string>{}(k.op_name);
        std::size_t h2 = std::hash<int32_t>{}(k.M);
        std::size_t h3 = std::hash<int32_t>{}(k.N);
        std::size_t h4 = std::hash<int32_t>{}(k.K);
        std::size_t h5 = std::hash<int32_t>{}(k.dtype_code);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
    }
};

using KernelFunc = std::function<void(cudaStream_t)>;

/*
    High-Precision Dynamic Autotuner & Dispatcher.
 */
class KernelDispatcher {
public:
    static KernelDispatcher& instance() {
        static KernelDispatcher dispatcher;
        return dispatcher;
    }

    /**
     * @brief Dispatches the best cached kernel or benchmarks candidate functions to elect the fastest variant.
     * 
     * @param key Unique problem shape/dtype descriptor
     * @param candidates Map of variant names to candidate kernel lambdas
     * @param stream CUDA execution stream
     */
    void dispatch_or_autotune(
        const KernelKey& key,
        const std::unordered_map<std::string, KernelFunc>& candidates,
        cudaStream_t stream = 0
    ) {
        std::string winner_name;

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                winner_name = it->second;
            }
        }

        if (winner_name.empty()) {
            winner_name = autotune_candidates(key, candidates, stream);
            std::lock_guard<std::mutex> lock(cache_mutex_);
            cache_[key] = winner_name;
        }

        // Dispatch optimal kernel
        candidates.at(winner_name)(stream);
    }

    /**
     * @brief Clears the active autotuning cache.
     */
    void clear_cache() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_.clear();
    }

private:
    KernelDispatcher() = default;

    std::string autotune_candidates(
        const KernelKey& key,
        const std::unordered_map<std::string, KernelFunc>& candidates,
        cudaStream_t stream
    ) {
        if (candidates.empty()) {
            throw std::runtime_error("Autotuner error: No candidate kernels provided.");
        }

        std::string best_candidate;
        float best_latency_ms = 1e9f;

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        constexpr int WARMUP_RUNS = 3;
        constexpr int BENCHMARK_RUNS = 10;

        for (const auto& [name, func] : candidates) {
            // Warmup iterations
            for (int i = 0; i < WARMUP_RUNS; ++i) {
                func(stream);
            }
            cudaStreamSynchronize(stream);

            // Benchmark iterations using CUDA Events
            cudaEventRecord(start_event, stream);
            for (int i = 0; i < BENCHMARK_RUNS; ++i) {
                func(stream);
            }
            cudaEventRecord(stop_event, stream);
            cudaEventSynchronize(stop_event);

            float total_ms = 0.0f;
            cudaEventElapsedTime(&total_ms, start_event, stop_event);
            float avg_ms = total_ms / BENCHMARK_RUNS;

            if (avg_ms < best_latency_ms) {
                best_latency_ms = avg_ms;
                best_candidate = name;
            }
        }

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);

        return best_candidate;
    }

    std::mutex cache_mutex_;
    std::unordered_map<KernelKey, std::string, KernelKeyHash> cache_;
};

} // namespace cudaforge::autotuner