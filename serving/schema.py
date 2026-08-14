"""
Centralized Pydantic V2 models for strict request validation, response serialization,
and OpenAPI documentation mapping.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerationRequest(BaseModel):
    """
    Client request payload for LLM inference generation.
    Supports both batch execution and Server-Sent Events (SSE) streaming.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "prompt": "Explain the architecture of a Retrieval-Augmented Generation system.",
                "max_tokens": 512,
                "temperature": 0.2,
                "stream": True,
            }
        },
    )

    prompt: str = Field(..., min_length=1, description="Input context prompt text.")
    max_tokens: int = Field(
        default=256, ge=1, le=4096, description="Maximum tokens to generate."
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature."
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability."
    )
    repetition_penalty: float = Field(
        default=1.0, ge=0.8, le=2.0, description="Repetition penalty factor."
    )
    stream: bool = Field(
        default=True, description="Enable Server-Sent Events (SSE) streaming."
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None, description="Optional stop tokens."
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError(
                "Prompt text cannot consist solely of whitespace or be empty."
            )
        return value


class TokenStreamChunk(BaseModel):
    """
    Individual token payload yielded during SSE streaming.
    """

    model_config = ConfigDict(
        frozen=True
    )  # Immutability for safety during async yields

    token_id: int = Field(..., description="Vocabulary ID of the generated token.")
    text: str = Field(..., description="Decoded string representation of the token.")
    logprob: Optional[float] = Field(
        default=None, description="Log probability of the token."
    )
    step_latency_ms: float = Field(
        ..., description="Wall-clock time taken to generate this specific token."
    )
    is_final: bool = Field(
        default=False,
        description="Flag indicating if this is the final token in the sequence.",
    )


class GenerationResponse(BaseModel):
    """
    Synchronous response payload for non-streaming generation requests.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(
        ..., description="Unique trace ID for the inference request."
    )
    prompt: str = Field(..., description="The original input prompt.")
    generated_text: str = Field(
        ..., description="The complete, concatenated output text."
    )
    finish_reason: str = Field(
        ..., description="Reason for generation halt (e.g., 'stop', 'length')."
    )
    tokens_generated: int = Field(..., description="Total number of tokens yielded.")
    total_latency_ms: float = Field(
        ..., description="End-to-end request processing time."
    )
    time_to_first_token_ms: float = Field(
        ..., description="Latency until the first token was generated (TTFT)."
    )
