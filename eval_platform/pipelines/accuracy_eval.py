"""
Measures perplexity, generation correctness (Exact Match, ROUGE-L), and logit-level
numerical drift between FP16 baseline models and INT8/INT4 quantized variants.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Dataclasses & Evaluation Schemas
# ============================================================================


@dataclass
class PerplexityResult:
    mean_perplexity: float
    mean_cross_entropy: float
    total_tokens_evaluated: int


@dataclass
class GenerationCorrectnessResult:
    exact_match: float
    rouge_l_f1: float
    token_accuracy: float
    total_samples: int


@dataclass
class LogitDriftResult:
    mse: float
    rmse: float
    cosine_similarity: float
    kl_divergence: float
    snr_db: float
    max_absolute_error: float


@dataclass
class AccuracyResult:
    """
    Unified summary container expected by the evaluation platform orchestrator.
    """

    task_name: str
    metric_score: float
    num_samples: int
    perplexity_result: Optional[PerplexityResult] = None
    correctness_result: Optional[GenerationCorrectnessResult] = None
    drift_result: Optional[LogitDriftResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Core Evaluator Implementation
# ============================================================================


class AccuracyEvaluator:
    """
    Evaluator for quantifying perplexity, correctness, and FP16 vs INT4 quantization drift.
    """

    def __init__(self, task_name: str = "accuracy_benchmark"):
        self.task_name = task_name

    def evaluate(
        self,
        generated_sequences: Optional[List[List[int]]] = None,
        reference_sequences: Optional[List[List[int]]] = None,
        logits: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
    ) -> AccuracyResult:
        """
        Unified evaluation entrypoint for the Orchestrator pipeline.
        """
        perplexity_res = None
        correctness_res = None
        primary_metric = 0.0
        num_samples = 0

        if logits is not None and target_ids is not None:
            perplexity_res = self.compute_perplexity(logits, target_ids)
            primary_metric = perplexity_res.mean_perplexity
            num_samples = perplexity_res.total_tokens_evaluated

        if generated_sequences is not None and reference_sequences is not None:
            correctness_res = self.evaluate_correctness(
                generated_sequences, reference_sequences
            )
            if primary_metric == 0.0:
                primary_metric = correctness_res.exact_match
            num_samples = correctness_res.total_samples

        return AccuracyResult(
            task_name=self.task_name,
            metric_score=primary_metric,
            num_samples=num_samples,
            perplexity_result=perplexity_res,
            correctness_result=correctness_res,
        )

    @staticmethod
    def compute_perplexity(
        logits: torch.Tensor, target_ids: torch.Tensor, ignore_index: int = -100
    ) -> PerplexityResult:
        """
        Computes perplexity over token prediction logits.
        """
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = target_ids.view(-1)

        loss = F.cross_entropy(
            flat_logits, flat_targets, ignore_index=ignore_index, reduction="sum"
        )
        non_padding_mask = flat_targets != ignore_index
        num_valid_tokens = int(non_padding_mask.sum().item())

        if num_valid_tokens == 0:
            return PerplexityResult(
                mean_perplexity=float("nan"),
                mean_cross_entropy=float("nan"),
                total_tokens_evaluated=0,
            )

        mean_ce = (loss / num_valid_tokens).item()
        perplexity = math.exp(mean_ce)

        return PerplexityResult(
            mean_perplexity=perplexity,
            mean_cross_entropy=mean_ce,
            total_tokens_evaluated=num_valid_tokens,
        )

    @staticmethod
    def evaluate_correctness(
        generated_sequences: List[List[int]],
        reference_sequences: List[List[int]],
    ) -> GenerationCorrectnessResult:
        """
        Evaluates Exact Match (EM) and ROUGE-L F1 scores across token ID sequences.
        """
        assert len(generated_sequences) == len(
            reference_sequences
        ), "Mismatched sample counts."
        total_samples = len(generated_sequences)
        if total_samples == 0:
            return GenerationCorrectnessResult(0.0, 0.0, 0.0, 0)

        exact_matches = 0
        rouge_f1_sum = 0.0
        token_acc_sum = 0.0

        for gen, ref in zip(generated_sequences, reference_sequences):
            if gen == ref:
                exact_matches += 1

            min_len = min(len(gen), len(ref))
            if min_len > 0:
                matches = sum(1 for i in range(min_len) if gen[i] == ref[i])
                token_acc_sum += matches / float(max(len(gen), len(ref)))

            lcs_length = AccuracyEvaluator._lcs_length(gen, ref)
            if lcs_length > 0 and len(gen) > 0 and len(ref) > 0:
                prec = lcs_length / float(len(gen))
                rec = lcs_length / float(len(ref))
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            else:
                f1 = 0.0
            rouge_f1_sum += f1

        return GenerationCorrectnessResult(
            exact_match=exact_matches / float(total_samples),
            rouge_l_f1=rouge_f1_sum / float(total_samples),
            token_accuracy=token_acc_sum / float(total_samples),
            total_samples=total_samples,
        )

    @staticmethod
    def _lcs_length(seq1: List[int], seq2: List[int]) -> int:
        """Computes length of Longest Common Subsequence via Dynamic Programming."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    @staticmethod
    def compare_numerical_drift(
        fp16_logits: torch.Tensor, quantized_logits: torch.Tensor
    ) -> LogitDriftResult:
        """
        Measures numerical drift and KL divergence between full-precision baseline
        and quantized (INT8/INT4) logits.
        """
        assert (
            fp16_logits.shape == quantized_logits.shape
        ), "Shape mismatch in logit evaluation."

        fp16 = fp16_logits.detach().to(torch.float64)
        quant = quantized_logits.detach().to(torch.float64)

        diff = fp16 - quant
        mse = float(torch.mean(diff**2).item())
        rmse = math.sqrt(mse)
        max_err = float(torch.max(torch.abs(diff)).item())

        flat_fp16 = fp16.view(-1)
        flat_quant = quant.view(-1)
        cos_sim = float(
            torch.dot(flat_fp16, flat_quant)
            / (torch.norm(flat_fp16) * torch.norm(flat_quant) + 1e-12)
        )

        p_prob = F.softmax(fp16, dim=-1)
        q_prob = F.softmax(quant, dim=-1)
        kl_div = float(F.kl_div(q_prob.log(), p_prob, reduction="batchmean").item())

        signal_power = torch.mean(fp16**2)
        noise_power = torch.mean(diff**2)
        snr_db = 10.0 * math.log10(
            (signal_power / (noise_power + 1e-12)).item() + 1e-12
        )

        return LogitDriftResult(
            mse=mse,
            rmse=rmse,
            cosine_similarity=cos_sim,
            kl_divergence=kl_div,
            snr_db=snr_db,
            max_absolute_error=max_err,
        )
