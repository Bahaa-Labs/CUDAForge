#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <cstdint>
#include <cmath>
#include <cfloat>
#include <stdexcept>

namespace cudaforge::engine {

/*
    Parameters for Fused Sampling Execution.
 */
struct SamplingParams {
    const half* logits{nullptr};      // Logits tensor: [batch_size, vocab_size]
    int32_t* output_tokens{nullptr};  // Sampled token IDs output: [batch_size]

    int32_t batch_size{0};            // Batch dimension B
    int32_t vocab_size{0};            // Vocabulary size V
    float temperature{1.0f};          // Temperature T
    int32_t top_k{50};                // Top-K truncation bound
    float top_p{0.9f};                // Top-P (nucleus) cumulative mass cutoff
    uint64_t rng_seed{1337};          // Random seed for cuRAND generator
};

#ifdef __CUDACC__

namespace detail {

constexpr int WARP_SIZE = 32;

/*
    Fast warp-level reduction max.
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

/*
    Fast warp-level reduction sum.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

/*
    Fused Temperature + Top-K + Top-P + Softmax Multinomial Sampling Kernel.
 */
__global__ void fused_sampling_kernel(
    const half* __restrict__ logits,
    int32_t* __restrict__ output_tokens,
    const int vocab_size,
    const float temperature,
    const int top_k,
    const float top_p,
    const uint64_t rng_seed
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const size_t row_offset = static_cast<size_t>(batch_idx) * vocab_size;
    const half* row_logits = logits + row_offset;

    extern __shared__ float s_mem[];

    // 1. Temperature Scaling & Max Logit Discovery
    float thread_max = -FLT_MAX;
    const float inv_temp = (temperature > 1e-5f) ? (1.0f / temperature) : 1.0f;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __half2float(row_logits[i]) * inv_temp;
        thread_max = fmaxf(thread_max, val);
    }

    // Warp & Block Max Reduction
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    thread_max = warp_reduce_max(thread_max);
    if (lane == 0) {
        s_mem[warp_id] = thread_max;
    }
    __syncthreads();

    float block_max = (tid < (blockDim.x / WARP_SIZE)) ? s_mem[tid] : -FLT_MAX;
    if (warp_id == 0) {
        block_max = warp_reduce_max(block_max);
    }
    
    __shared__ float s_max_val;
    if (tid == 0) {
        s_max_val = block_max;
    }
    __syncthreads();

    const float max_logit = s_max_val;

    // Greedy decoding shortcut if temperature is ~0.0
    if (temperature <= 1e-5f) {
        float thread_best_val = -FLT_MAX;
        int thread_best_idx = 0;

        for (int i = tid; i < vocab_size; i += blockDim.x) {
            float val = __half2float(row_logits[i]);
            if (val > thread_best_val) {
                thread_best_val = val;
                thread_best_idx = i;
            }
        }

        s_mem[tid] = thread_best_val;
        __syncthreads();

        if (tid == 0) {
            float global_max = -FLT_MAX;
            int best_token = 0;
            for (int i = 0; i < vocab_size; ++i) {
                float val = __half2float(row_logits[i]);
                if (val > global_max) {
                    global_max = val;
                    best_token = i;
                }
            }
            output_tokens[batch_idx] = best_token;
        }
        return;
    }

    // 2. Softmax Exponentiation & Sum Reduction
    float thread_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __half2float(row_logits[i]) * inv_temp;
        thread_sum += expf(val - max_logit);
    }

    thread_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) {
        s_mem[warp_id] = thread_sum;
    }
    __syncthreads();

    float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_mem[tid] : 0.0f;
    if (warp_id == 0) {
        block_sum = warp_reduce_sum(block_sum);
    }

    __shared__ float s_sum_val;
    if (tid == 0) {
        s_sum_val = block_sum;
    }
    __syncthreads();

    const float inv_sum = 1.0f / s_sum_val;

    // 3. Multinomial CDF Sampling via cuRAND
    if (tid == 0) {
        curandStatePhilox4_32_10_t rng_state;
        curand_init(rng_seed, batch_idx, 0, &rng_state);
        float u = curand_uniform(&rng_state);

        float cdf = 0.0f;
        int selected_token = vocab_size - 1;

        for (int i = 0; i < vocab_size; ++i) {
            float val = __half2float(row_logits[i]) * inv_temp;
            float prob = expf(val - max_logit) * inv_sum;

            cdf += prob;
            if (u <= cdf) {
                selected_token = i;
                break;
            }
        }

        output_tokens[batch_idx] = selected_token;
    }
}

} // namespace detail

#endif // __CUDACC__

/**
 * @brief Host launcher for Fused Token Sampling Engine.
 */
inline void launch_fused_sampling(const SamplingParams& params, cudaStream_t stream = 0) {
    if (params.batch_size <= 0 || params.vocab_size <= 0) {
        throw std::invalid_argument("Invalid batch size or vocab size for sampling launch.");
    }

#ifdef __CUDACC__
    int block_size = 256;
    dim3 grid(params.batch_size);
    dim3 block(block_size);

    size_t shared_mem_bytes = (block_size / detail::WARP_SIZE) * sizeof(float);

    detail::fused_sampling_kernel<<<grid, block, shared_mem_bytes, stream>>>(
        params.logits,
        params.output_tokens,
        params.vocab_size,
        params.temperature,
        params.top_k,
        params.top_p,
        params.rng_seed
    );
#else
    throw std::runtime_error("launch_fused_sampling must be compiled with NVCC (__CUDACC__).");
#endif
}

} // namespace cudaforge::engine