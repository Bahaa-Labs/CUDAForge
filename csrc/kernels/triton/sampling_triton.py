import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sampling_triton_kernel(
    Logits_ptr, Output_tokens_ptr,
    vocab_size,
    temperature,
    stride_lb, stride_lv,
    BLOCK_V: tl.constexpr
):
    batch_id = tl.program_id(0)
    offs_v = tl.arange(0, BLOCK_V)

    mask = offs_v < vocab_size
    logits_ptrs = Logits_ptr + batch_id * stride_lb + offs_v * stride_lv

    # Load and immediately promote to FP32 for numerical stability & consistent branch typing
    logits = tl.load(logits_ptrs, mask=mask, other=-1e9).to(tl.float32)

    if temperature > 1e-5:
        logits = logits / temperature

    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    # Perform argmax selection for greedy/top token sampling
    max_idx = tl.argmax(probs, axis=0)

    # Store sampled token index for the current batch
    tl.store(Output_tokens_ptr + batch_id, max_idx)


def fused_sampling_triton(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Triton Fused Token Sampling Launcher.

    Args:
        logits: Logits tensor [batch_size, vocab_size] (float16 / float32)
        temperature: Temperature scaling factor
    """
    batch_size, vocab_size = logits.shape
    output_tokens = torch.empty((batch_size,), device=logits.device, dtype=torch.int32)

    block_v = triton.next_power_of_2(vocab_size)
    grid = (batch_size,)

    _fused_sampling_triton_kernel[grid](
        logits, output_tokens,
        vocab_size,
        temperature,
        logits.stride(0), logits.stride(1),
        BLOCK_V=block_v,
        num_warps=8,
    )

    return output_tokens