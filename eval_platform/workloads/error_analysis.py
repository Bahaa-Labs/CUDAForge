"""
Analyzes generation discrepancies, exact match failure modes, and output
divergence between baseline unquantized and quantized model checkpoints.
"""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from eval_platform.workloads.dataset_loader import EvaluationSample


@dataclass
class SampleErrorDiagnostic:
    sample_id: str
    task_type: str
    prompt: str
    expected_reference: str
    baseline_output: str
    quantized_output: str
    is_baseline_correct: bool
    is_quantized_correct: bool
    quantization_divergence: bool
    error_category: str  # "EXACT_MATCH_FAIL", "CODE_SYNTAX_ERR", "REASONING_DRIFT", "QUANT_DIVERGENCE"


@dataclass
class ErrorAnalysisReport:
    total_samples: int
    baseline_accuracy: float
    quantized_accuracy: float
    quantization_degradation_rate: float
    category_counts: Dict[str, int]
    failed_sample_ids: List[str]
    sample_diagnostics: List[SampleErrorDiagnostic]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ErrorAnalyzer:
    """
    Diagnostic analyzer for categorizing generation failures and quantization quality losses.
    """

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalizes output strings for robust comparison."""
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @classmethod
    def evaluate_exact_match(cls, prediction: str, reference: str) -> bool:
        """Determines if prediction matches reference string."""
        return cls.normalize_text(prediction) == cls.normalize_text(reference)

    @classmethod
    def analyze_quantization_drift(
        cls,
        samples: List[EvaluationSample],
        baseline_outputs: List[str],
        quantized_outputs: List[str],
    ) -> ErrorAnalysisReport:
        """
        Compares baseline vs quantized inference outputs across evaluation samples.
        """
        if len(samples) != len(baseline_outputs) or len(samples) != len(quantized_outputs):
            raise ValueError("Sample and output list lengths must match exactly.")

        total_samples = len(samples)
        if total_samples == 0:
            raise ValueError("Cannot perform error analysis on empty sample set.")

        baseline_correct_count = 0
        quantized_correct_count = 0
        category_counts: Dict[str, int] = {
            "EXACT_MATCH_FAIL": 0,
            "QUANT_DIVERGENCE": 0,
            "REASONING_DRIFT": 0,
            "NONE_PASSED": 0,
        }
        failed_ids: List[str] = []
        diagnostics: List[SampleErrorDiagnostic] = []

        for sample, base_out, quant_out in zip(samples, baseline_outputs, quantized_outputs):
            base_correct = cls.evaluate_exact_match(base_out, sample.target_reference)
            quant_correct = cls.evaluate_exact_match(quant_out, sample.target_reference)

            if base_correct:
                baseline_correct_count += 1
            if quant_correct:
                quantized_correct_count += 1

            divergence = cls.normalize_text(base_out) != cls.normalize_text(quant_out)

            # Determine primary error category
            if base_correct and not quant_correct:
                err_cat = "QUANT_DIVERGENCE"
                failed_ids.append(sample.sample_id)
            elif not base_correct and not quant_correct:
                err_cat = "NONE_PASSED"
                failed_ids.append(sample.sample_id)
            elif not base_correct and quant_correct:
                err_cat = "REASONING_DRIFT"
            else:
                err_cat = "NONE"

            if err_cat in category_counts:
                category_counts[err_cat] += 1

            diagnostics.append(
                SampleErrorDiagnostic(
                    sample_id=sample.sample_id,
                    task_type=sample.task_type,
                    prompt=sample.prompt,
                    expected_reference=sample.target_reference,
                    baseline_output=base_out,
                    quantized_output=quant_out,
                    is_baseline_correct=base_correct,
                    is_quantized_correct=quant_correct,
                    quantization_divergence=divergence,
                    error_category=err_cat,
                )
            )

        base_acc = baseline_correct_count / total_samples
        quant_acc = quantized_correct_count / total_samples
        degradation = base_acc - quant_acc

        return ErrorAnalysisReport(
            total_samples=total_samples,
            baseline_accuracy=round(base_acc, 4),
            quantized_accuracy=round(quant_acc, 4),
            quantization_degradation_rate=round(degradation, 4),
            category_counts=category_counts,
            failed_sample_ids=failed_ids,
            sample_diagnostics=diagnostics,
        )