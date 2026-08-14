"""
Manages content-addressable storage, checksum validation (SHA-256), and file system
persistence for experiment logs, metrics, and evaluation artifacts.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class ArtifactManifest:
    artifact_id: str
    filename: str
    sha256_hash: str
    size_bytes: int
    content_type: str
    created_at: str
    relative_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArtifactStore:
    """
    Content-addressable file store for persisting benchmark outputs and metadata.
    """

    def __init__(self, store_root: str = ".artifacts/store"):
        self.store_root = pathlib.Path(store_root).resolve()
        self.store_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(file_path: pathlib.Path) -> str:
        """Computes SHA-256 hash of a local file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def store_file(
        self,
        source_path: Union[str, pathlib.Path],
        run_id: str,
        content_type: str = "raw",
    ) -> ArtifactManifest:
        """
        Copies a file into the run's artifact directory and indexes its hash.
        """
        src = pathlib.Path(source_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source file for artifact does not exist: {src}")

        run_dir = self.store_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        dest = (run_dir / src.name).resolve()

        # Avoid shutil.SameFileError if source is already inside target run_dir
        if src != dest:
            shutil.copy2(src, dest)

        sha256_hash = self.compute_sha256(dest)
        size_bytes = dest.stat().st_size
        rel_path = str(dest.relative_to(self.store_root))

        manifest = ArtifactManifest(
            artifact_id=f"art_{sha256_hash[:12]}",
            filename=src.name,
            sha256_hash=sha256_hash,
            size_bytes=size_bytes,
            content_type=content_type,
            created_at=datetime.utcnow().isoformat(),
            relative_path=rel_path,
        )

        return manifest

    def store_json(
        self, data: Dict[str, Any], run_id: str, filename: str
    ) -> ArtifactManifest:
        """Serializes and stores a dictionary as a JSON artifact."""
        if not filename.endswith(".json"):
            filename += ".json"

        run_dir = self.store_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        target_path = run_dir / filename

        with open(target_path, "w") as f:
            json.dump(data, f, indent=2)

        return self.store_file(target_path, run_id, content_type="application/json")

    def store_pt_tensor(
        self, tensor: torch.Tensor, run_id: str, filename: str
    ) -> ArtifactManifest:
        """Saves a PyTorch tensor artifact."""
        if not filename.endswith(".pt"):
            filename += ".pt"

        run_dir = self.store_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        target_path = run_dir / filename

        torch.save(tensor.detach().cpu(), target_path)

        return self.store_file(
            target_path, run_id, content_type="application/octet-stream"
        )
