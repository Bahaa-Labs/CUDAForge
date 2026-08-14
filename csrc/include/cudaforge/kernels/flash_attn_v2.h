#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace cudaforge::kernels {

/*
    Configuration parameters for CUDA FlashAttention-2 Kernel.
 */
struct FlashAttnParams {
    const half* Q{nullptr}; // Shape: [batch_size, num_heads, seq_len, head_dim]
    const half* K{nullptr}; // Shape: [batch_size, num_heads, seq_len, head_dim]
    const half* V{nullptr}; // Shape: [batch_size, num_heads, seq_len, head_dim]
    half* O{nullptr};       // Shape: [batch_size, num_heads, seq_len, head_dim]

    int32_t batch_size{0};
    int32_t num_heads{0};
    int32_t seq_len{0};
    int32_t head_dim{0};    // Supported: 64, 128

    float sm_scale{1.0f};   // Softmax scaling factor 1 / sqrt(head_dim)
    bool is_causal{false};  // Causal attention mask switch
};

/*
    Host launcher for FlashAttention-2 CUDA Kernel.
 */
void launch_flash_attn_v2(const FlashAttnParams& params, cudaStream_t stream = 0);

} // namespace cudaforge::kernels