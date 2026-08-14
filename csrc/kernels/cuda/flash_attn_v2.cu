#include "cudaforge/kernels/flash_attn_v2.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace cudaforge::kernels {

// Tiling Dimensions
constexpr int BLOCK_M = 64;  // Q tile size
constexpr int BLOCK_N = 64;  // K/V tile size

/**
 * @brief Inline PTX instruction for Ampere cp.async (Global -> Shared Memory direct DMA)
 */
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" 
                 :: "r"(smem_addr), "l"(gmem_ptr));
}

/*
    Commit asynchronous copy operations.
 */
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

/*
    Wait for async copies to finish.
 */
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

/*
    Ampere FlashAttention-2 CUDA Kernel using Online Softmax and Shared Memory Tiling.
 */
template<int HEAD_DIM>
__global__ void flash_attn_v2_ampere_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int seq_len,
    const int num_heads,
    const float sm_scale,
    const bool is_causal
) {
    const int bx = blockIdx.x; // Sequence Block Index (Q tile)
    const int by = blockIdx.y; // Head Index
    const int bz = blockIdx.z; // Batch Index

    const int tid = threadIdx.x;

    const int q_start = bx * BLOCK_M;
    if (q_start >= seq_len) return;

    // Strides
    const size_t bh_offset = (static_cast<size_t>(bz) * num_heads + by) * seq_len * HEAD_DIM;
    const half* q_ptr = Q + bh_offset + static_cast<size_t>(q_start) * HEAD_DIM;
    const half* k_ptr = K + bh_offset;
    const half* v_ptr = V + bh_offset;
    half* o_ptr = O + bh_offset + static_cast<size_t>(q_start) * HEAD_DIM;

    // Shared Memory Layout
    extern __shared__ char smem_raw[];
    half* smem_q = reinterpret_cast<half*>(smem_raw);                                   // [BLOCK_M, HEAD_DIM]
    half* smem_k = smem_q + BLOCK_M * HEAD_DIM;                                          // [BLOCK_N, HEAD_DIM]
    half* smem_v = smem_k + BLOCK_N * HEAD_DIM;                                          // [BLOCK_N, HEAD_DIM]

    // Online Softmax tracking registers per thread
    float m_prev = -1e20f;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM / 4] = {0.0f}; // Vectorized accumulator registers

    // 1. Asynchronously Load Q-tile into Shared Memory
    const int q_bytes_per_thread = (BLOCK_M * HEAD_DIM * sizeof(half)) / 128;
    for (int i = 0; i < q_bytes_per_thread; ++i) {
        int load_idx = tid + i * 128;
        int row = load_idx / (HEAD_DIM / 8);
        int col = (load_idx % (HEAD_DIM / 8)) * 8;
        if (q_start + row < seq_len) {
            cp_async_cg_16(&smem_q[row * HEAD_DIM + col], &q_ptr[row * HEAD_DIM + col]);
        }
    }
    cp_async_commit();
    cp_async_wait_group<0>();
    __syncthreads();

    // Loop over K/V tiles
    const int max_k_tiles = is_causal ? min((q_start + BLOCK_M + BLOCK_N - 1) / BLOCK_N, (seq_len + BLOCK_N - 1) / BLOCK_N)
                                      : (seq_len + BLOCK_N - 1) / BLOCK_N;

    for (int k_tile = 0; k_tile < max_k_tiles; ++k_tile) {
        const int k_start = k_tile * BLOCK_N;

        // 2. Load K and V tiles using cp.async
        for (int i = 0; i < q_bytes_per_thread; ++i) {
            int load_idx = tid + i * 128;
            int row = load_idx / (HEAD_DIM / 8);
            int col = (load_idx % (HEAD_DIM / 8)) * 8;

            if (k_start + row < seq_len) {
                cp_async_cg_16(&smem_k[row * HEAD_DIM + col], &k_ptr[(k_start + row) * HEAD_DIM + col]);
                cp_async_cg_16(&smem_v[row * HEAD_DIM + col], &v_ptr[(k_start + row) * HEAD_DIM + col]);
            }
        }
        cp_async_commit();
        cp_async_wait_group<0>();
        __syncthreads();

        // 3. Compute Attention Scores (S = Q * K^T * sm_scale) inside thread block
        float s_max = -1e20f;
        float s_scores[BLOCK_N] = {0.0f};

        int q_row_local = tid % BLOCK_M;
        if (q_start + q_row_local < seq_len) {
            for (int n = 0; n < BLOCK_N; ++n) {
                if (k_start + n >= seq_len) continue;
                if (is_causal && (k_start + n > q_start + q_row_local)) continue;

                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += __half2float(smem_q[q_row_local * HEAD_DIM + d]) * 
                           __half2float(smem_k[n * HEAD_DIM + d]);
                }
                float score = dot * sm_scale;
                s_scores[n] = score;
                s_max = fmaxf(s_max, score);
            }
        }

        // 4. Online Softmax Rescaling & Update
        float m_curr = fmaxf(m_prev, s_max);
        float p_sum = 0.0f;
        float alpha = expf(m_prev - m_curr);

        for (int n = 0; n < BLOCK_N; ++n) {
            if (s_scores[n] > -1e19f) {
                s_scores[n] = expf(s_scores[n] - m_curr);
                p_sum += s_scores[n];
            } else {
                s_scores[n] = 0.0f;
            }
        }

        float l_curr = l_prev * alpha + p_sum;

        // 5. Update Output Accumulator: O = O * (l_prev * alpha / l_curr) + (P * V) / l_curr
        float rescale_prev = (l_curr > 0.0f) ? (l_prev * alpha / l_curr) : 0.0f;
        float scale_curr = (l_curr > 0.0f) ? (1.0f / l_curr) : 0.0f;

        for (int d = 0; d < HEAD_DIM / 4; ++d) {
            o_acc[d] *= rescale_prev;
        }

        if (q_start + q_row_local < seq_len) {
            for (int n = 0; n < BLOCK_N; ++n) {
                if (s_scores[n] > 0.0f) {
                    float p_val = s_scores[n] * scale_curr;
                    for (int d = 0; d < HEAD_DIM / 4; ++d) {
                        float v_val = __half2float(smem_v[n * HEAD_DIM + (d * 4 + (tid % 4))]);
                        o_acc[d] += p_val * v_val;
                    }
                }
            }
        }

        m_prev = m_curr;
        l_prev = l_curr;

        __syncthreads();
    }

    // 6. Write final result back to Global Memory
    int q_row_local = tid % BLOCK_M;
    if (q_start + q_row_local < seq_len && tid < BLOCK_M) {
        for (int d = 0; d < HEAD_DIM / 4; ++d) {
            int col = d * 4 + (tid % 4);
            o_ptr[q_row_local * HEAD_DIM + col] = __float2half(o_acc[d]);
        }
    }
}

void launch_flash_attn_v2(const FlashAttnParams& params, cudaStream_t stream) {
    if (params.head_dim != 64 && params.head_dim != 128) {
        throw std::invalid_argument("FlashAttention-2 Kernel only supports head_dim 64 or 128");
    }

    dim3 grid(
        (params.seq_len + BLOCK_M - 1) / BLOCK_M,
        params.num_heads,
        params.batch_size
    );

    dim3 block(128); // 4 warps per block

    size_t smem_bytes = (BLOCK_M * params.head_dim + 2 * BLOCK_N * params.head_dim) * sizeof(half);

    if (params.head_dim == 64) {
        flash_attn_v2_ampere_kernel<64><<<grid, block, smem_bytes, stream>>>(
            params.Q, params.K, params.V, params.O,
            params.seq_len, params.num_heads, params.sm_scale, params.is_causal
        );
    } else if (params.head_dim == 128) {
        flash_attn_v2_ampere_kernel<128><<<grid, block, smem_bytes, stream>>>(
            params.Q, params.K, params.V, params.O,
            params.seq_len, params.num_heads, params.sm_scale, params.is_causal
        );
    }
}

} // namespace cudaforge::kernels