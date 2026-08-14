"""
Provides asynchronous Server-Sent Events (SSE) streaming and batch generation
routes with Pydantic V2 schema validation and structured error handling.
"""

import asyncio
from dataclasses import asdict
from datetime import datetime
import json
import logging
import time
from typing import AsyncGenerator, Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("cudaforge.serving.generate")

router = APIRouter(tags=["Inference Generation"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input context prompt text.")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability.")
    repetition_penalty: float = Field(default=1.0, ge=0.8, le=2.0, description="Repetition penalty factor.")
    stream: bool = Field(default=True, description="Enable Server-Sent Events (SSE) streaming.")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Optional stop tokens.")

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt text cannot consist solely of whitespace.")
        return v


class TokenStreamChunk(BaseModel):
    token_id: int
    text: str
    logprob: Optional[float] = None
    step_latency_ms: float
    is_final: bool = False


class GenerationResponse(BaseModel):
    request_id: str
    prompt: str
    generated_text: str
    finish_reason: str
    tokens_generated: int
    total_latency_ms: float
    time_to_first_token_ms: float


# ============================================================================
# Token Generation Stream Engine
# ============================================================================

async def mock_cuda_token_generator(
    request: GenerationRequest, request_id: str
) -> AsyncGenerator[str, None]:
    """
    Simulates low-overhead CUDA engine execution yield loop with high-precision timing.
    Replace inner loop step with actual model pipeline generation iterator.
    """
    start_time = time.perf_counter()
    ttft_measured = False
    ttft_ms = 0.0

    dummy_tokens = request.prompt.split()[:5] + ["is", "optimized", "by", "CUDAForge", "engine."]
    total_steps = min(request.max_tokens, len(dummy_tokens))

    try:
        for idx in range(total_steps):
            step_start = time.perf_counter()
            
            # Simulate CUDA kernel execution delay per token
            await asyncio.sleep(0.015)  # ~15ms per token
            
            step_latency_ms = (time.perf_counter() - step_start) * 1000.0
            
            if not ttft_measured:
                ttft_ms = (time.perf_counter() - start_time) * 1000.0
                ttft_measured = True

            token_text = dummy_tokens[idx] + " "
            is_final = (idx == total_steps - 1)

            chunk = TokenStreamChunk(
                token_id=1000 + idx,
                text=token_text,
                logprob=-0.05,
                step_latency_ms=round(step_latency_ms, 2),
                is_final=is_final,
            )

            # Format as SSE event payload
            sse_data = f"data: {chunk.model_dump_json()}\n\n"
            yield sse_data

            if is_final:
                break

    except asyncio.CancelledError:
        logger.warning(f"Client disconnected early from streaming request_id={request_id}")
        raise
    except Exception as err:
        logger.error(f"Execution error during token generation for request_id={request_id}: {err}")
        error_payload = json.dumps({"error_code": "ERR_INTERNAL_EXECUTION", "detail": str(err)})
        yield f"event: error\ndata: {error_payload}\n\n"


# ============================================================================
# API Routes
# ============================================================================

@router.post(
    "/generate",
    response_model=None,
    summary="Generate Text / Stream Tokens",
    description="Main entrypoint for autoregressive LLM token generation with streaming support.",
)
async def generate_tokens(
    req: GenerationRequest,
    raw_request: Request,
):
    request_id = f"req_{int(time.time() * 1000)}"

    # Safely retrieve or lazily initialize the semaphore (for TestClient / direct test runs)
    concurrency_semaphore: asyncio.Semaphore = getattr(
        raw_request.app.state,
        "concurrency_semaphore",
        None,
    )
    if concurrency_semaphore is None:
        concurrency_semaphore = asyncio.Semaphore(32)
        raw_request.app.state.concurrency_semaphore = concurrency_semaphore

    # Acquire concurrency lock with timeout protection
    try:
        await asyncio.wait_for(concurrency_semaphore.acquire(), timeout=5.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "ERR_QUEUE_FULL",
                "message": "Model execution concurrency queue is saturated. Try again shortly.",
            },
        )

    try:
        if req.stream:
            async def streaming_wrapper():
                try:
                    async for event in mock_cuda_token_generator(req, request_id):
                        yield event
                finally:
                    concurrency_semaphore.release()

            return StreamingResponse(
                streaming_wrapper(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Synchronous non-streaming execution path
            start_time = time.perf_counter()
            generated_chunks = []
            async for event in mock_cuda_token_generator(req, request_id):
                if event.startswith("data: "):
                    raw_json = event[6:].strip()
                    if raw_json:
                        chunk_dict = json.loads(raw_json)
                        generated_chunks.append(chunk_dict["text"])

            total_latency = (time.perf_counter() - start_time) * 1000.0
            full_text = "".join(generated_chunks)

            concurrency_semaphore.release()

            return GenerationResponse(
                request_id=request_id,
                prompt=req.prompt,
                generated_text=full_text,
                finish_reason="stop",
                tokens_generated=len(generated_chunks),
                total_latency_ms=round(total_latency, 2),
                time_to_first_token_ms=15.0,
            )

    except Exception as err:
        concurrency_semaphore.release()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "ERR_INTERNAL", "message": str(err)},
        )