import pytest
import torch
from cudaforge.kernels.int4_dequant_triton import int4_gemm_dequant_triton


def pack_int4_weights(w_int4: torch.Tensor) -> torch.Tensor:
    """Packs an [K, N] INT4 weight tensor into [K // 8, N] int32 tensor."""
    K, N = w_int4.shape
    assert K % 8 == 0
    w_int4 = w_int4.to(torch.int32) & 0x0F

    packed = torch.zeros((K // 8, N), dtype=torch.int32, device=w_int4.device)
    for i in range(8):
        packed |= w_int4[i::8, :] << (i * 4)
    return packed


@pytest.mark.parametrize("M", [16, 64])
@pytest.mark.parametrize("N", [128, 256])
@pytest.mark.parametrize("K", [128, 512])
@pytest.mark.parametrize("group_size", [128])
def test_int4_gemm_dequant_parity(M, N, K, group_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    torch.manual_seed(42)

    # Generate random activations
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")

    # Generate random INT4 weights (values from 0 to 15)
    w_int4 = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int32)
    scales = (
        torch.randn((K // group_size, N), dtype=torch.float16, device="cuda").abs()
        * 0.1
    )

    # Pack weights
    b_quant = pack_int4_weights(w_int4)

    # Reference Python Dequantized GEMM
    w_dequant = torch.zeros((K, N), dtype=torch.float16, device="cuda")
    for k in range(K):
        group_idx = k // group_size
        w_dequant[k, :] = w_int4[k, :].to(torch.float16) * scales[group_idx, :]

    c_ref = torch.matmul(a, w_dequant)

    # Kernel execution
    c_triton = int4_gemm_dequant_triton(a, b_quant, scales, group_size=group_size)

    # Numerical verification
    torch.testing.assert_close(c_triton, c_ref, atol=1e-2, rtol=1e-2)
