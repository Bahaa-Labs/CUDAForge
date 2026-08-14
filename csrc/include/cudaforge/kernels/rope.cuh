#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace cudaforge::kernels {

/*
    Applies Rotary Position Embedding (RoPE) to a single Q/K vector pair in register/shared memory.
 */
template <int HEAD_DIM>
__device__ __forceinline__ void apply_rope_vector(
    half* __restrict__ vec,
    const float* __restrict__ cos_ptr,
    const float* __restrict__ sin_ptr,
    const int tid
) {
    constexpr int HALF_DIM = HEAD_DIM / 2;
    
    #pragma unroll
    for (int i = tid; i < HALF_DIM; i += blockDim.x) {
        float x1 = __half2float(vec[i]);
        float x2 = __half2float(vec[i + HALF_DIM]);

        float c = cos_ptr[i];
        float s = sin_ptr[i];

        vec[i]            = __float2half(x1 * c - x2 * s);
        vec[i + HALF_DIM] = __float2half(x1 * s + x2 * c);
    }
}

/*
    CUDA Kernel for Rotary Position Embeddings on Query and Key Tensors.
 */
template <int HEAD_DIM>
__global__ void rope_kernel(
    half* __restrict__ Q,             // [batch_size, seq_len, num_q_heads, head_dim]
    half* __restrict__ K,             // [batch_size, seq_len, num_kv_heads, head_dim]
    const float* __restrict__ cos,    // [seq_len, head_dim / 2]
    const float* __restrict__ sin,    // [seq_len, head_dim / 2]
    const int num_q_heads,
    const int num_kv_heads,
    const int seq_len
) {
    const int token_idx = blockIdx.x; // Token index in combined batch*seq dimension
    const int head_idx  = blockIdx.y; // Head index
    const int is_key    = blockIdx.z; // 0 for Query, 1 for Key

    const int tid = threadIdx.x;
    const int pos = token_idx % seq_len;

    const float* cos_row = cos + pos * (HEAD_DIM / 2);
    const float* sin_row = sin + pos * (HEAD_DIM / 2);

    if (is_key == 0) {
        if (head_idx >= num_q_heads) return;
        size_t offset = (static_cast<size_t>(token_idx) * num_q_heads + head_idx) * HEAD_DIM;
        apply_rope_vector<HEAD_DIM>(Q + offset, cos_row, sin_row, tid);
    } else {
        if (head_idx >= num_kv_heads) return;
        size_t offset = (static_cast<size_t>(token_idx) * num_kv_heads + head_idx) * HEAD_DIM;
        apply_rope_vector<HEAD_DIM>(K + offset, cos_row, sin_row, tid);
    }
}

} // namespace cudaforge::kernels