#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace cudaforge::kernels {

/*
    Parameters for Fused Residual Addition + RMSNorm launch.
 */
struct FusedRMSNormParams {
    const half* input{nullptr};       // Input tensor: [num_tokens, hidden_dim]
    half* residual{nullptr};          // In-out residual tensor: [num_tokens, hidden_dim]
    const half* weight{nullptr};        // RMSNorm scale weights: [hidden_dim]
    half* output{nullptr};            // Output normalized tensor: [num_tokens, hidden_dim]

    int32_t num_tokens{0};            // Total tokens in batch (batch_size * seq_len)
    int32_t hidden_dim{0};            // Hidden dimension size D
    float epsilon{1e-6f};             // RMSNorm variance epsilon
};

/*
    Host launcher for Fused Residual + RMSNorm CUDA Kernel.
 */
void launch_fused_rmsnorm(const FusedRMSNormParams& params, cudaStream_t stream = 0);

} // namespace cudaforge::kernels