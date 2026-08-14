#include "cudaforge/kernels/fused_norm.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cudaforge::kernels {

constexpr int WARP_SIZE = 32;

/*
    Warp-level reduction sum using shuffle instructions.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

/*
    Block-level reduction sum across warps.
 */
__device__ __forceinline__ float block_reduce_sum(float val, float* shared_mem) {
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    const int num_warps = blockDim.x / WARP_SIZE;
    val = (tid < num_warps) ? shared_mem[tid] : 0.0f;

    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

/*
    Fused Residual Addition + RMSNorm CUDA Kernel with 128-bit Vectorized Memory Access.
 */
__global__ void fused_rmsnorm_kernel(
    const half* __restrict__ input,
    half* __restrict__ residual,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int hidden_dim,
    const float epsilon
) {
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const size_t row_offset = static_cast<size_t>(row_idx) * hidden_dim;
    const half* in_row = input + row_offset;
    half* res_row = residual + row_offset;
    half* out_row = output + row_offset;

    extern __shared__ float s_mem[];

    // 128-bit Vectorized Memory Transfers (8 halfs per uint4)
    constexpr int VEC_SIZE = 8;
    const int vec_cols = hidden_dim / VEC_SIZE;

    float thread_sq_sum = 0.0f;

    // Phase 1: Read input & residual, accumulate sum of squares, write back updated residual
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        const uint4* in_vec_ptr = reinterpret_cast<const uint4*>(in_row) + i;
        uint4* res_vec_ptr = reinterpret_cast<uint4*>(res_row) + i;

        uint4 in_v = *in_vec_ptr;
        uint4 res_v = *res_vec_ptr;

        const half2* in_h2 = reinterpret_cast<const half2*>(&in_v);
        half2* res_h2 = reinterpret_cast<half2*>(&res_v);

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 in_f2 = __half22float2(in_h2[k]);
            float2 res_f2 = __half22float2(res_h2[k]);

            float2 sum_f2;
            sum_f2.x = in_f2.x + res_f2.x;
            sum_f2.y = in_f2.y + res_f2.y;

            res_h2[k] = __float22half2_rn(sum_f2);

            thread_sq_sum += sum_f2.x * sum_f2.x + sum_f2.y * sum_f2.y;
        }

        *res_vec_ptr = res_v; // Store updated residual back to HBM
    }

    // Phase 2: Compute global sum of squares across row
    float total_sq_sum = block_reduce_sum(thread_sq_sum, s_mem);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        float mean_sq = total_sq_sum / static_cast<float>(hidden_dim);
        s_inv_rms = rsqrtf(mean_sq + epsilon);
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;

    // Phase 3: Normalize and write output using 128-bit transfers
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        const uint4* res_vec_ptr = reinterpret_cast<const uint4*>(res_row) + i;
        const uint4* w_vec_ptr = reinterpret_cast<const uint4*>(weight) + i;
        uint4* out_vec_ptr = reinterpret_cast<uint4*>(out_row) + i;

        uint4 res_v = *res_vec_ptr;
        uint4 w_v = *w_vec_ptr;
        uint4 out_v;

        const half2* res_h2 = reinterpret_cast<const half2*>(&res_v);
        const half2* w_h2 = reinterpret_cast<const half2*>(&w_v);
        half2* out_h2 = reinterpret_cast<half2*>(&out_v);

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 res_f2 = __half22float2(res_h2[k]);
            float2 w_f2 = __half22float2(w_h2[k]);

            float2 norm_f2;
            norm_f2.x = res_f2.x * inv_rms * w_f2.x;
            norm_f2.y = res_f2.y * inv_rms * w_f2.y;

            out_h2[k] = __float22half2_rn(norm_f2);
        }

        *out_vec_ptr = out_v;
    }
}

void launch_fused_rmsnorm(const FusedRMSNormParams& params, cudaStream_t stream) {
    if (params.hidden_dim % 8 != 0) {
        throw std::invalid_argument("fused_rmsnorm requires hidden_dim to be a multiple of 8 for 128-bit vectorization.");
    }

    int block_size = 256;
    if (params.hidden_dim / 8 < block_size) {
        block_size = std::max(32, (params.hidden_dim / 8 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    }

    dim3 grid(params.num_tokens);
    dim3 block(block_size);

    size_t shared_mem_bytes = (block_size / WARP_SIZE) * sizeof(float);

    fused_rmsnorm_kernel<<<grid, block, shared_mem_bytes, stream>>>(
        params.input,
        params.residual,
        params.weight,
        params.output,
        params.hidden_dim,
        params.epsilon
    );
}

} // namespace cudaforge::kernels