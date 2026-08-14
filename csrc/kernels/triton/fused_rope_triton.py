import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rope_kernel(
    Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    num_q_heads, num_kv_heads, seq_len,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    is_key = tl.program_id(2)

    batch_id = token_idx // seq_len
    seq_id = token_idx % seq_len

    offs_d = tl.arange(0, BLOCK_DIM)
    mask = offs_d < HALF_DIM

    # Load Cosine and Sine vectors
    cos_ptrs = Cos_ptr + seq_id * stride_cos_s + offs_d * stride_cos_d
    sin_ptrs = Sin_ptr + seq_id * stride_sin_s + offs_d * stride_sin_d

    cos = tl.load(cos_ptrs, mask=mask, other=0.0)
    sin = tl.load(sin_ptrs, mask=mask, other=0.0)

    if is_key == 0:
        if head_idx < num_q_heads:
            q_first_ptrs = Q_ptr + batch_id * stride_qb + seq_id * stride_qs + head_idx * stride_qh + offs_d * stride_qd
            q_second_ptrs = Q_ptr + batch_id * stride_qb + seq_id * stride_qs + head_idx * stride_qh + (offs_d + HALF_DIM) * stride_qd

            x1 = tl.load(q_first_ptrs, mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(q_second_ptrs, mask=mask, other=0.0).to(tl.float32)

            o1 = x1 * cos - x2 * sin
            o2 = x1 * sin + x2 * cos

            tl.store(q_first_ptrs, o1.to(tl.float16), mask=mask)
            tl.store(q_second_ptrs, o2.to(tl.float16), mask=mask)
    else:
        if head_idx < num_kv_heads:
            k_first_ptrs = K_ptr + batch_id * stride_kb + seq_id * stride_ks + head_idx * stride_kh + offs_d * stride_kd
            k_second_ptrs = K_ptr + batch_id * stride_kb + seq_id * stride_ks + head_idx * stride_kh + (offs_d + HALF_DIM) * stride_kd

            x1 = tl.load(k_first_ptrs, mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(k_second_ptrs, mask=mask, other=0.0).to(tl.float32)

            o1 = x1 * cos - x2 * sin
            o2 = x1 * sin + x2 * cos

            tl.store(k_first_ptrs, o1.to(tl.float16), mask=mask)
            tl.store(k_second_ptrs, o2.to(tl.float16), mask=mask)


def fused_rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) in-place using Triton.

    Args:
        q: [batch_size, seq_len, num_q_heads, head_dim] tensor (fp16)
        k: [batch_size, seq_len, num_kv_heads, head_dim] tensor (fp16)
        cos: [seq_len, head_dim // 2] tensor (fp32 or fp16)
        sin: [seq_len, head_dim // 2] tensor (fp32 or fp16)
    """
    batch_size, seq_len, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape

    half_dim = head_dim // 2
    block_dim = triton.next_power_of_2(half_dim)

    total_tokens = batch_size * seq_len
    max_heads = max(num_q_heads, num_kv_heads)

    grid = (total_tokens, max_heads, 2)

    _fused_rope_kernel[grid](
        q, k, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        num_q_heads, num_kv_heads, seq_len,
        HEAD_DIM=head_dim,
        HALF_DIM=half_dim,
        BLOCK_DIM=block_dim,
        num_warps=4,
    )

    return q, k