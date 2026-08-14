import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_v2_fwd_kernel(
    Q,
    K,
    V,
    Out,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # Offset pointers for current batch and head
    Q_ptr = Q + off_z * stride_qb + off_h * stride_qh
    K_ptr = K + off_z * stride_kb + off_h * stride_kh
    V_ptr = V + off_z * stride_vb + off_h * stride_vh
    O_ptr = Out + off_z * stride_ob + off_h * stride_oh

    # Block offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q tile
    q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Initialize Online Softmax accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Scale Q
    q = (q * sm_scale).to(tl.float16)

    # Loop boundaries for K/V tiles
    end_n = N_CTX if not IS_CAUSAL else min(N_CTX, (start_m + 1) * BLOCK_M)

    for start_n in range(0, end_n, BLOCK_N):
        curr_offs_n = start_n + offs_n

        # Load K tile
        k_ptrs = K_ptr + (
            curr_offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        )
        k = tl.load(k_ptrs, mask=curr_offs_n[None, :] < N_CTX, other=0.0)

        # Q * K^T GEMM
        qk = tl.dot(q, k)

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= curr_offs_n[None, :], qk, -float("inf"))

        # Online Softmax updates
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Rescale accumulator
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load V tile
        v_ptrs = V_ptr + (
            curr_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=curr_offs_n[:, None] < N_CTX, other=0.0)

        # P * V GEMM
        p = p.to(tl.float16)
        acc += tl.dot(p, v)

        m_i = m_ij

    # Final normalization
    acc = acc / l_i[:, None]

    # Store Output tile
    o_ptrs = O_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


def flash_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """Triton FlashAttention-2 Wrapper function.

    Args:
        q: [Batch, Heads, SeqLen, HeadDim] tensor in fp16
        k: [Batch, Heads, SeqLen, HeadDim] tensor in fp16
        v: [Batch, Heads, SeqLen, HeadDim] tensor in fp16
        causal: Apply causal attention mask
        sm_scale: Softmax scale factor (defaults to 1 / sqrt(head_dim))
    """
    Z, H, N_CTX, D_HEAD = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (D_HEAD**0.5)

    out = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(N_CTX, BLOCK_M), H, Z)

    _flash_attn_v2_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        Z,
        H,
        N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=D_HEAD,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )

    return out
