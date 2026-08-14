import pytest
import torch
from cudaforge.kernels.sampling_triton import fused_sampling_triton


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("vocab_size", [1024, 4096])
@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
def test_fused_sampling_triton_execution(batch_size, vocab_size, temperature):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    torch.manual_seed(42)
    logits = torch.randn((batch_size, vocab_size), dtype=torch.float16, device="cuda")

    # Inject clear peak for deterministic top token testing at low temp
    logits[:, 42] += 10.0

    tokens = fused_sampling_triton(logits, temperature=temperature)

    assert tokens.shape == (batch_size,)
    assert tokens.dtype == torch.int32

    # At low temp, highest logit (index 42) must be chosen
    if temperature == 0.0:
        assert torch.all(tokens == 42)
