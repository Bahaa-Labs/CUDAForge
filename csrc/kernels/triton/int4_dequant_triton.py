import torch
import triton
import triton.language as tl


@triton.jit
def _int4_dequant_gemm_kernel(
    A_ptr,
    B_quant_ptr,
    Scales_ptr,
    Zeros_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_sm,
    stride_sn,
    stride_cm,
    stride_cn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # 1. Load Activation A chunk [BLOCK_M, BLOCK_K] (FP16)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(
            a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0
        )

        # 2. Load Scales for current group [BLOCK_N]
        group_idx = (k * BLOCK_K) // GROUP_SIZE
        scale_ptrs = Scales_ptr + group_idx * stride_sm + offs_n[None, :] * stride_sn
        scales = tl.load(scale_ptrs, mask=offs_n[None, :] < N, other=1.0)

        # 3. Load Packed INT4 B weights: 8 elements per uint32 along K dimension
        packed_k_offs = offs_k // 8
        b_ptrs = B_quant_ptr + (
            packed_k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )
        b_packed = tl.load(
            b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0
        )

        # 4. Unpack INT4 values directly in registers
        shift = (offs_k % 8) * 4
        b_int4 = (b_packed >> shift[:, None]) & 0x0F

        # Dequantize to FP32 for arithmetic accuracy, then downcast to FP16 to match A
        b_fp16 = (b_int4.to(tl.float32) * scales).to(tl.float16)

        # 5. Execute Tensor Core Matrix Multiply Accumulate (FP16 x FP16 -> FP32)
        acc += tl.dot(a, b_fp16)

    # Store FP16 result to output C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


def int4_gemm_dequant_triton(
    a: torch.Tensor,
    b_quant: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Triton Fused INT4 Weight Dequantization GEMM Execution Launcher.

    Args:
        a: Activation tensor [M, K] (float16)
        b_quant: Packed INT4 weights [K // 8, N] (int32 / uint32)
        scales: Scale factors [K // group_size, N] (float16)
        zeros: Optional Zero points
        group_size: Quantization group size
    """
    M, K = a.shape
    _, N = b_quant.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _int4_dequant_gemm_kernel[grid](
        a,
        b_quant,
        scales,
        zeros,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_quant.stride(0),
        b_quant.stride(1),
        scales.stride(0),
        scales.stride(1),
        c.stride(0),
        c.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )

    return c
