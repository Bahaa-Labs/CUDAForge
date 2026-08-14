"""
Provides structured liveness and readiness probe endpoints for upstream load
balancers and Kubernetes orchestrators.
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("cudaforge.serving.health")

router = APIRouter(tags=["Health & Readiness"])


# ============================================================================
# Response Schemas
# ============================================================================


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    timestamp: float = Field(
        ..., description="UNIX epoch timestamp of check execution."
    )
    uptime_seconds: float = Field(..., description="Engine process uptime in seconds.")
    version: str = Field(default="1.0.0", description="CUDAForge service version.")


class ReadinessResponse(BaseModel):
    status: str = Field(..., example="ready")
    cuda_available: bool = Field(..., description="CUDA GPU runtime availability.")
    semaphore_initialized: bool = Field(..., description="Concurrency state readiness.")
    active_in_flight_requests: int = Field(
        ..., description="Current in-flight request count."
    )


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="Basic ping endpoint indicating whether the web server process is alive.",
)
async def liveness_probe(request: Request) -> HealthResponse:
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = time.time() - start_time

    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        uptime_seconds=round(uptime, 2),
        version="1.0.0",
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Detailed readiness check validating engine concurrency primitives and CUDA status.",
)
async def readiness_probe(request: Request) -> JSONResponse:
    import torch

    semaphore_ready = hasattr(request.app.state, "concurrency_semaphore") and (
        request.app.state.concurrency_semaphore is not None
    )
    cuda_ready = torch.cuda.is_available()

    in_flight = 0
    if semaphore_ready:
        in_flight = max(0, 32 - request.app.state.concurrency_semaphore._value)

    is_ready = semaphore_ready

    payload = ReadinessResponse(
        status="ready" if is_ready else "degraded",
        cuda_available=cuda_ready,
        semaphore_initialized=semaphore_ready,
        active_in_flight_requests=in_flight,
    )

    status_code = (
        status.HTTP_200_OK if is_ready else status.HTTP_530_SERVICE_OFFLINE_ANOTHER
    )

    return JSONResponse(status_code=status_code, content=payload.model_dump())
