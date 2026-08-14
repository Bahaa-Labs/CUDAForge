#include "cudaforge/kernels/dequantize.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <stdexcept>

namespace cudaforge::kernels {

using namespace nvcuda;

// WMMA Tile Dimensions for Ampere FP16 Tensor Cores
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

/*
    Fast register-level unpacking of 8 INT4 values from a 32-bit register into 4 half2 registers.
 */
__device__ __forceinline__ void unpack_int4_to_fp16_v8(
    uint32_t val,
    half* out_fp16,
    half scale,
    half zero
) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t nibble = (val >> (i * 4)) & 0x0F;
        float fval = static_cast<float>(nibble);
        half hval = __float2half(fval);
        
        // Apply scaling and zero point: (W - Z) * S
        out_fp16[i] = __hmul(__hsub(hval, zero), scale);
    }
}

/*
    Fused INT4 Dequantization Tensor Core GEMM Kernel using WMMA Primitives.
 */
__global__ void int4_gemm_dequant_kernel(
    const half* __restrict__ A,
    const uint32_t* __restrict__ B_quant,
    const half* __restrict__ scales,
    const half* __restrict__ zeros,
    half* __restrict__ C,
    int M, int N, int K,
    int group_size
) {
    // Warp and Block coordinates
    const int warp_row = (blockIdx.y * blockDim.y + threadIdx.y);
    const int warp_col = (blockIdx.x * blockDim.x + threadIdx.x);

    const int m_offset = warp_row * WMMA_M;
    const int n_offset = warp_col * WMMA_N;

    if (m_offset >= M || n_offset >= N) return;

    // WMMA Accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Temp register buffer for unpacked FP16 weights (16 elements per WMMA tile step)
    half unpacked_w[WMMA_K * WMMA_N];

    // Loop over K dimension in WMMA_K chunks
    for (int k_idx = 0; k_idx < K; k_idx += WMMA_K) {
        // 1. Load Activation Fragment A [WMMA_M, WMMA_K]
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, A + m_offset * K + k_idx, K);

        // 2. Unpack and Dequantize Weight Fragment B directly in registers
        const int group_idx = k_idx / group_size;
        
        // Thread 0 in warp handles register unpacking for the tile
        const int lane_id = threadIdx.x % 32;
        
        for (int row = 0; row < WMMA_K; ++row) {
            int current_k = k_idx + row;
            // B_quant is stored as [K / 8, N]
            int packed_k_idx = current_k / 8;
            int nibble_shift = (current_k % 8) * 4;

            for (int col = lane_id; col < WMMA_N; col += 32) {
                int col_idx = n_offset + col;
                uint32_t packed_val = B_quant[packed_k_idx * N + col_idx];
                uint32_t nibble = (packed_val >> nibble_shift) & 0x0F;

                half scale = scales[group_idx * N + col_idx];
                half zero = (zeros != nullptr) ? zeros[group_idx * N + col_idx] : __float2half(0.0f);

                float fval = static_cast<float>(nibble);
                unpacked_w[row * WMMA_N + col] = __hmul(__hsub(__float2half(fval), zero), scale);
            }
        }
        __syncwarp();

        // 3. Load unpacked FP16 weights into WMMA matrix_b fragment
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, unpacked_w, WMMA_N);

        // 4. Tensor Core Multiply-Accumulate
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Convert accumulator to FP16 and store to global memory C
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    for (size_t i = 0; i < acc_frag.num_elements; ++i) {
        c_frag.x[i] = __float2half(acc_frag.x[i]);
    }

    wmma::store_matrix_sync(C + m_offset * N + n_offset, c_frag, N, wmma::mem_row_major);
}

void launch_int4_gemm_dequant(const Int4GemmParams& params, cudaStream_t stream) {
    if (params.K % 16 != 0 || params.N % 16 != 0 || params.M % 16 != 0) {
        throw std::invalid_argument("INT4 GEMM dimensions (M, N, K) must be multiples of 16.");
    }

    dim3 threads_per_block(4, 4); // 16 warps = 512 threads per block
    dim3 num_blocks(
        (params.N + (WMMA_N * threads_per_block.x) - 1) / (WMMA_N * threads_per_block.x),
        (params.M + (WMMA_M * threads_per_block.y) - 1) / (WMMA_M * threads_per_block.y)
    );

    int4_gemm_dequant_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        params.A,
        params.B_quant,
        params.scales,
        params.zeros,
        params.C,
        params.M,
        params.N,
        params.K,
        params.group_size
    );
}

} // namespace cudaforge::kernels