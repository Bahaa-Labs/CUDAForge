#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace cudaforge::kernels {

/*
    Configuration parameters for Fused INT4 GEMM with register dequantization.
 */
struct Int4GemmParams {
    const half* A{nullptr};            // Activation Matrix A [M, K] in FP16
    const uint32_t* B_quant{nullptr};  // Packed INT4 Weight Matrix B [K / 8, N]
    const half* scales{nullptr};       // Per-group scaling factors [K / group_size, N] in FP16
    const half* zeros{nullptr};        // Optional zero-points [K / group_size, N] in FP16 (nullptr for symmetric)
    half* C{nullptr};                  // Output Matrix C [M, N] in FP16

    int32_t M{0};                      // Batch * SeqLen dimension
    int32_t N{0};                      // Out features / Hidden dimension
    int32_t K{0};                      // In features / Input dimension
    int32_t group_size{128};           // Quantization group size (e.g., 64, 128)
};

/*
    Host launcher for Fused INT4 Dequantization GEMM.
 */
void launch_int4_gemm_dequant(const Int4GemmParams& params, cudaStream_t stream = 0);

} // namespace cudaforge::kernels