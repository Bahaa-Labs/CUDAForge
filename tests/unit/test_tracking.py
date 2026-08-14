import json
import pathlib
import pytest
import torch

from eval_platform.tracking import (
    ArtifactStore,
    ExperimentLogger,
)


def test_artifact_store_and_sha256(tmp_path):
    store = ArtifactStore(store_root=str(tmp_path))
    data = {"metric_key": "val_123", "score": 0.99}

    manifest = store.store_json(data=data, run_id="run_test_001", filename="metrics.json")

    assert manifest.filename == "metrics.json"
    assert manifest.sha256_hash is not None
    assert len(manifest.sha256_hash) == 64
    assert manifest.size_bytes > 0


def test_experiment_logger_lifecycle(tmp_path):
    logger = ExperimentLogger(
        experiment_name="unit_test_experiment", store_root=str(tmp_path)
    )

    with logger.start_run(run_name="test_run", seed=1234) as run:
        logger.log_parameters({"batch_size": 16, "precision": "int8"})
        logger.log_metrics({"throughput_tps": 450.5, "p99_latency_ms": 12.4})

        # Save dummy artifact file
        artifact_path = tmp_path / "temp_trace.txt"
        artifact_path.write_text("dummy kernel execution trace data")
        logger.log_artifact(source_path=artifact_path, content_type="text/plain")

    assert run.status == "SUCCESS"
    assert run.seed == 1234
    assert run.parameters["batch_size"] == 16
    assert run.metrics["throughput_tps"] == 450.5
    assert len(run.artifacts) == 1

    # Verify run manifest JSON persisted to disk
    manifest_path = tmp_path / run.run_id / "run_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r") as f:
        persisted_data = json.load(f)
    assert persisted_data["run_id"] == run.run_id
    assert persisted_data["status"] == "SUCCESS"