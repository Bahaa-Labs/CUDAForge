"""
Contains ASGI middleware components for request interception and backpressure control.
"""

import logging
import time
from typing import Awaitable, Callable, Set

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("cudaforge.serving.middleware")


class BackpressureMiddleware:
    def __init__(self, max_concurrent: int = 32, max_queue: int = 128):
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.active_in_flight_requests = 0
        self.exempt_paths: Set[str] = {
            "/healthz",
            "/v1/health",
            "/v1/health/ready",
            "/metrics",
            "/v1/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }

    async def __call__(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        capacity_limit = self.max_concurrent + self.max_queue
        if self.active_in_flight_requests >= capacity_limit:
            logger.warning("Engine overloaded. Shedding load.")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error_code": "ERR_BACKPRESSURE_LIMIT_EXCEEDED",
                    "message": "CUDAForge engine queue saturated. Retry with exponential backoff.",
                },
                headers={"Retry-After": "5"},
            )

        self.active_in_flight_requests += 1
        start_time = time.perf_counter()
        try:
            response = await call_next(request)
            process_time_ms = (time.perf_counter() - start_time) * 1000.0
            response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
            return response
        finally:
            self.active_in_flight_requests -= 1


backpressure_middleware = BackpressureMiddleware()