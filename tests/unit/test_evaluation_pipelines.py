import math
import pytest
import torch

from eval_platform.pipelines import (
    AccuracyEvaluator,
    LatencyEvaluator,
)


def test_perplexity_computation():
    logits = torch.randn(2, 8, 100, dtype=torch.float32)
    targets = torch.randint(0, 100, (2, 8), dtype=torch.int64)

    res = AccuracyEvaluator.compute_perplexity(logits, targets)
    assert res.total_tokens_evaluated == 16
    assert res.mean_perplexity > 1.0
    assert not math.isnan(res.mean_cross_entropy)


def test_correctness_and_rouge_l():
    gen_seqs = [[101, 200, 300, 102], [101, 500, 102]]
    ref_seqs = [[101, 200, 300, 102], [101, 400, 102]]

    res = AccuracyEvaluator.evaluate_correctness(gen_seqs, ref_seqs)
    assert res.exact_match == 0.5  # 1 out of 2 exact match
    assert res.rouge_l_f1 > 0.6
    assert res.total_samples == 2


def test_logit_drift_evaluation():
    fp16_logits = torch.randn(4, 16, 512, dtype=torch.float32)
    # Add small Gaussian noise to simulate quantization drift
    quant_logits = fp16_logits + torch.randn_like(fp16_logits) * 0.05

    drift = AccuracyEvaluator.compare_numerical_drift(fp16_logits, quant_logits)
    assert drift.cosine_similarity > 0.98
    assert drift.snr_db > 20.0
    assert drift.mse < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for latency profiler test")
def test_latency_evaluator_profiling():
    evaluator = LatencyEvaluator(warmup_runs=1)

    def mock_step(step_idx: int):
        a = torch.randn(128, 128, device="cuda")
        _ = torch.matmul(a, a)

    profile = evaluator.profile_generation_loop(
        step_fn=mock_step, prompt_len=128, gen_len=5, batch_size=1
    )

    assert profile.ttft_ms > 0.0
    assert profile.itl_mean_ms > 0.0
    assert profile.itl_p50_ms > 0.0
    assert profile.generation_tps > 0.0
    assert len(profile.per_token_latencies_ms) == 4