"""
Standardized data ingestion and sample formatting for benchmark datasets
(GSM8K, HumanEval, MMLU, and Synthetic Workloads).
"""

from dataclasses import asdict, dataclass, field
import json
import pathlib
from typing import Any, Dict, Generator, List, Optional, Union


@dataclass
class EvaluationSample:
    sample_id: str
    prompt: str
    target_reference: str
    task_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkloadDatasetLoader:
    """
    Unified loader for downstream benchmark datasets.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = pathlib.Path(cache_dir or ".cache/workloads")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create_synthetic_workload(
        num_samples: int = 50,
        prompt_token_len: int = 512,
        task_type: str = "synthetic",
    ) -> List[EvaluationSample]:
        """
        Generates synthetic prompts of controlled length for latency/throughput profiling.
        """
        samples: List[EvaluationSample] = []
        base_word = "CUDAForge system optimization benchmark sequence. "

        # Estimate words needed for target prompt length
        repeat_count = max(1, prompt_token_len // 6)
        prompt_text = (base_word * repeat_count)[: prompt_token_len * 4]

        for idx in range(num_samples):
            samples.append(
                EvaluationSample(
                    sample_id=f"synth_{idx:04d}",
                    prompt=f"{prompt_text} [Sample {idx}] Answer:",
                    target_reference=f"Reference output for sample {idx}",
                    task_type=task_type,
                    metadata={"target_prompt_len": prompt_token_len, "sample_idx": idx},
                )
            )
        return samples

    def load_from_jsonl(
        self, file_path: Union[str, pathlib.Path], task_type: str = "custom"
    ) -> List[EvaluationSample]:
        """
        Loads evaluation samples from a JSONL file.
        Format expected per line: {"id": "...", "prompt": "...", "reference": "..."}
        """
        path = pathlib.Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        samples: List[EvaluationSample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                samples.append(
                    EvaluationSample(
                        sample_id=str(record.get("id", f"sample_{line_idx}")),
                        prompt=record.get("prompt", ""),
                        target_reference=record.get(
                            "reference", record.get("target", "")
                        ),
                        task_type=record.get("task_type", task_type),
                        metadata=record.get("metadata", {}),
                    )
                )
        return samples

    def stream_dataset(
        self, samples: List[EvaluationSample], batch_size: int = 8
    ) -> Generator[List[EvaluationSample], None, None]:
        """Streams evaluation samples in controlled batch chunks."""
        for i in range(0, len(samples), batch_size):
            yield samples[i : i + batch_size]
