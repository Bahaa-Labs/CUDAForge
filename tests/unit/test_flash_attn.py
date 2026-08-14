import pytest
import torch
from csrc.kernels.triton.flash_attn_triton import flash_attn_triton


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("seq_len", [128, 256])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_triton_against_sdpa(
    batch_size, num_heads, seq_len, head_dim, causal
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    q = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.float16, device="cuda"
    )
    k = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.float16, device="cuda"
    )
    v = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.float16, device="cuda"
    )

    # Triton Output
    out_triton = flash_attn_triton(q, k, v, causal=causal)

    # Reference PyTorch Scaled Dot-Product Attention
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal
    )

    # Verify numerical precision parity
    torch.testing.assert_close(out_triton, out_ref, atol=1e-2, rtol=1e-2)
