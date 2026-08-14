"""
Records git commits, hardware state, seeds, configurations, and evaluation outputs
for 100% reproducible benchmark runs.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
import pathlib
import platform
import random
import subprocess
import sys
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch

from eval_platform.tracking.artifact_store import ArtifactManifest, ArtifactStore

logger = logging.getLogger("eval_platform.tracking")


@dataclass
class GitEnvironmentState:
    commit_hash: str
    branch_name: str
    is_dirty: bool
    commit_timestamp: str


@dataclass
class HardwareEnvironmentState:
    platform: str
    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    cudnn_version: Optional[int]
    gpu_device_name: str
    gpu_count: int
    total_vram_gb: float
    cpu_count: int


@dataclass
class ExperimentRunRecord:
    run_id: str
    experiment_name: str
    seed: int
    status: str  # "RUNNING", "SUCCESS", "FAILED"
    start_time: str
    end_time: Optional[str] = None
    git_state: Optional[GitEnvironmentState] = None
    hardware_state: Optional[HardwareEnvironmentState] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[ArtifactManifest] = field(default_factory=list)
    error_log: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        return res


class ExperimentLogger:
    """
    Logger for orchestrating run state, environment provenance, and artifact indexing.
    """

    def __init__(self, experiment_name: str, store_root: str = ".artifacts/store"):
        self.experiment_name = experiment_name
        self.store = ArtifactStore(store_root=store_root)
        self.current_run: Optional[ExperimentRunRecord] = None

    @staticmethod
    def set_global_seed(seed: int = 42) -> None:
        """Sets global random seeds across random, numpy, PyTorch, and CUDA."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def capture_git_state() -> GitEnvironmentState:
        """Captures version control state using system Git calls."""
        try:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("ascii")
                .strip()
            )
            branch_name = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("ascii")
                .strip()
            )
            dirty_check = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
                )
                .decode("ascii")
                .strip()
            )
            commit_time = (
                subprocess.check_output(
                    ["git", "log", "-1", "--format=%cd", "--date=iso"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("ascii")
                .strip()
            )

            return GitEnvironmentState(
                commit_hash=commit_hash,
                branch_name=branch_name,
                is_dirty=len(dirty_check) > 0,
                commit_timestamp=commit_time,
            )
        except Exception:
            return GitEnvironmentState(
                commit_hash="UNKNOWN",
                branch_name="UNKNOWN",
                is_dirty=True,
                commit_timestamp="UNKNOWN",
            )

    @staticmethod
    def capture_hardware_state() -> HardwareEnvironmentState:
        """Captures hardware and PyTorch runtime metadata."""
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A (CPU Only)"
        gpu_count = torch.cuda.device_count() if cuda_avail else 0
        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024.0**3)
            if cuda_avail
            else 0.0
        )

        return HardwareEnvironmentState(
            platform=platform.platform(),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version() if cuda_avail else None,
            gpu_device_name=gpu_name,
            gpu_count=gpu_count,
            total_vram_gb=round(vram_gb, 2),
            cpu_count=os.cpu_count() or 1,
        )

    @contextmanager
    def start_run(
        self, run_name: Optional[str] = None, seed: int = 42
    ) -> Generator[ExperimentRunRecord, None, None]:
        """
        Context manager for lifecycle tracking of a benchmark run.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        prefix = run_name if run_name else "run"
        run_id = f"{prefix}_{timestamp}_{seed}"

        self.set_global_seed(seed)

        self.current_run = ExperimentRunRecord(
            run_id=run_id,
            experiment_name=self.experiment_name,
            seed=seed,
            status="RUNNING",
            start_time=datetime.utcnow().isoformat(),
            git_state=self.capture_git_state(),
            hardware_state=self.capture_hardware_state(),
        )

        try:
            yield self.current_run
            self.current_run.status = "SUCCESS"
        except Exception as err:
            self.current_run.status = "FAILED"
            self.current_run.error_log = str(err)
            raise err
        finally:
            self.current_run.end_time = datetime.utcnow().isoformat()
            self._flush_run_manifest()

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Logs hyperparameters or evaluation configurations."""
        if self.current_run is None:
            raise RuntimeError(
                "No active run. Call start_run() before logging parameters."
            )
        self.current_run.parameters.update(params)

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Logs scalar evaluation metrics."""
        if self.current_run is None:
            raise RuntimeError(
                "No active run. Call start_run() before logging metrics."
            )
        self.current_run.metrics.update(metrics)

    def log_artifact(
        self, source_path: Union[str, pathlib.Path], content_type: str = "raw"
    ) -> ArtifactManifest:
        """Registers and stores a local artifact under the active run."""
        if self.current_run is None:
            raise RuntimeError(
                "No active run. Call start_run() before logging artifacts."
            )

        manifest = self.store.store_file(
            source_path=source_path,
            run_id=self.current_run.run_id,
            content_type=content_type,
        )
        self.current_run.artifacts.append(manifest)
        return manifest

    def _flush_run_manifest(self) -> None:
        """Persists the complete run record metadata to disk."""
        if self.current_run is not None:
            self.store.store_json(
                data=self.current_run.to_dict(),
                run_id=self.current_run.run_id,
                filename="run_manifest.json",
            )
