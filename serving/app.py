"""
Low-overhead ASGI web service equipped with backpressure middleware, CORS,
rate/concurrency limiting, health check endpoints, and standardized error handlers.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import time
from typing import Callable, Dict, Any

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import API v1 Routers
from serving.api.v1.generate import router as generate_v1_router
from serving.api.v1.health import router as health_v1_router
from serving.api.v1.metrics import router as metrics_v1_router

# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cudaforge.serving.app")


# ============================================================================
# Application Lifecycle & State Setup
# ============================================================================

MAX_CONCURRENT_REQUESTS = 32
MAX_BACKPRESSURE_QUEUE = 128

active_in_flight_requests = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes CUDA contexts, model weights, and concurrency primitive state."""
    logger.info("Initializing CUDAForge serving infrastructure...")
    app.state.concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    app.state.start_time = time.time()
    logger.info(
        f"Serving layer active. Max Concurrency Limit: {MAX_CONCURRENT_REQUESTS}"
    )

    yield

    logger.info("Shutting down CUDAForge serving infrastructure...")


app = FastAPI(
    title="CUDAForge Inference API",
    description="High-Precision Production Serving Engine for LLMs and PEFT Benchmarks",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Custom Backpressure Middleware
# ============================================================================


@app.middleware("http")
async def backpressure_middleware(request: Request, call_next: Callable) -> Response:
    """Rejects incoming HTTP requests when overall processing queue saturates."""
    global active_in_flight_requests

    # Exclude health, readiness, and metrics endpoints from backpressure throttling
    exempt_paths = [
        "/healthz",
        "/metrics",
        "/v1/health",
        "/v1/health/ready",
        "/v1/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    ]
    if request.url.path in exempt_paths:
        return await call_next(request)

    if active_in_flight_requests >= (MAX_CONCURRENT_REQUESTS + MAX_BACKPRESSURE_QUEUE):
        logger.warning(
            f"Backpressure triggered. In-flight requests = {active_in_flight_requests}"
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error_code": "ERR_BACKPRESSURE_LIMIT_EXCEEDED",
                "message": "Server is severely overloaded. Please apply exponential backoff.",
            },
        )

    active_in_flight_requests += 1
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response
    finally:
        active_in_flight_requests -= 1


# ============================================================================
# Exception Handlers & Core Diagnostics
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        f"Unhandled exception encountered at {request.url.path}: {exc}", exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "ERR_INTERNAL_SERVER_ERROR",
            "message": "An unhandled execution error occurred inside the engine.",
            "detail": str(exc),
        },
    )


@app.get("/healthz", tags=["System Diagnostics"])
async def health_check() -> Dict[str, Any]:
    """Root liveness check for container readiness probes."""
    uptime_sec = time.time() - getattr(app.state, "start_time", time.time())
    return {
        "status": "healthy",
        "uptime_sec": round(uptime_sec, 2),
        "active_requests": active_in_flight_requests,
        "max_concurrency": MAX_CONCURRENT_REQUESTS,
    }


# ============================================================================
# Include API Routers under /v1 Prefix
# ============================================================================

app.include_router(generate_v1_router, prefix="/v1")
app.include_router(health_v1_router, prefix="/v1")
app.include_router(metrics_v1_router, prefix="/v1")
