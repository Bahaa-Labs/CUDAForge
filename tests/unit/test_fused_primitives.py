import pytest
import torch
from csrc.kernels.triton.fused_rope_triton import fused_rope_triton


def ref_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference Python implementation of standard half-split RoPE."""
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    # Reshape cos/sin for broadcast: [1, seq_len, 1, half_dim]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos

    return torch.cat([o1, o2], dim=-1)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 128])
@pytest.mark.parametrize("num_q_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [2, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fused_rope_triton_parity(
    batch_size, seq_len, num_q_heads, num_kv_heads, head_dim
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    torch.manual_seed(1337)
    q = torch.randn(
        (batch_size, seq_len, num_q_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    k = torch.randn(
        (batch_size, seq_len, num_kv_heads, head_dim),
        dtype=torch.float16,
        device="cuda",
    )

    half_dim = head_dim // 2
    cos = torch.randn((seq_len, half_dim), dtype=torch.float32, device="cuda")
    sin = torch.randn((seq_len, half_dim), dtype=torch.float32, device="cuda")

    # Compute reference output
    q_ref = ref_rope(q, cos, sin).half()
    k_ref = ref_rope(k, cos, sin).half()

    # Compute Triton output
    q_triton, k_triton = fused_rope_triton(q.clone(), k.clone(), cos, sin)

    # Validate numerical agreement
    torch.testing.assert_close(q_triton, q_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k_triton, k_ref, atol=1e-3, rtol=1e-3)
