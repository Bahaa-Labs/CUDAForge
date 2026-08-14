from fastapi.testclient import TestClient
import pytest

from serving.api.v1.metrics import EngineMetricsTracker
from serving.app import app


@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client


def test_liveness_probe(test_client: TestClient):
    response = test_client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


def test_readiness_probe(test_client: TestClient):
    response = test_client.get("/v1/health/ready")
    assert response.status_code in (200, 530)
    data = response.json()
    assert "cuda_available" in data
    assert "semaphore_initialized" in data
    assert "active_in_flight_requests" in data


def test_engine_metrics_tracker_computations():
    EngineMetricsTracker.set_batch_size(8)
    EngineMetricsTracker.update_kv_cache_usage(
        allocated_bytes=750.0, total_bytes=1000.0
    )
    EngineMetricsTracker.record_tokens_generated(count=150, elapsed_seconds=3.0)
    EngineMetricsTracker.record_request_duration(0.025)

    telemetry = EngineMetricsTracker.refresh_cuda_hardware_telemetry()
    assert "allocated" in telemetry
    assert "reserved" in telemetry
    assert "fragmentation" in telemetry


def test_prometheus_metrics_endpoint(test_client: TestClient):
    response = test_client.get("/v1/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    body = response.text
    assert "cudaforge_active_batch_size" in body
    assert "cudaforge_kv_cache_usage_percent" in body
    assert "cudaforge_token_generation_speed_tps" in body
    assert "cudaforge_vram_fragmentation_ratio" in body
    assert "cudaforge_request_latency_seconds" in body
