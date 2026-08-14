from fastapi.testclient import TestClient
import pytest

from serving.app import app


def test_health_check_endpoint():
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_sec" in data
        assert "max_concurrency" in data


def test_generate_non_streaming():
    with TestClient(app) as client:
        payload = {
            "prompt": "Test CUDA acceleration engine",
            "max_tokens": 10,
            "stream": False,
        }
        response = client.post("/v1/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert data["prompt"] == payload["prompt"]
        assert len(data["generated_text"]) > 0
        assert data["tokens_generated"] > 0


def test_generate_streaming_sse():
    with TestClient(app) as client:
        payload = {
            "prompt": "Test streaming SSE tokens",
            "max_tokens": 5,
            "stream": True,
        }
        response = client.post("/v1/generate", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Read first chunk from response body stream
        content = response.text
        assert "data: " in content
        assert "token_id" in content


def test_invalid_empty_prompt_validation():
    with TestClient(app) as client:
        payload = {
            "prompt": "   ",  # Whitespace only
            "max_tokens": 10,
        }
        response = client.post("/v1/generate", json=payload)
        assert response.status_code == 422  # Unprocessable Entity
