"""
Provides clean Pythonic interfaces (both synchronous and asynchronous) over
the native C++/CUDA runtime bindings (_C extension module). Supports zero-copy
tensor memory sharing, prefix-cached paged KV caching, continuous batching,
and high-throughput streaming inference.
"""
from __future__ import annotations
import asyncio
import ctypes
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Set, Union
import torch

# Configure module-level logger
logger = logging.getLogger("cudaforge.engine")

# -----------------------------------------------------------------------------
# C++ Extension Loading & Fallback Guard
# -----------------------------------------------------------------------------
try:
    from cudaforge import _C
    _HAS_NATIVE_EXTENSION = True
except ImportError:
    try:
        import _C
        _HAS_NATIVE_EXTENSION = True
    except ImportError:
        _HAS_NATIVE_EXTENSION = False
        logger.warning(
            "CUDAForge C++ native extension (_C) not found. "
            "Please compile extensions using CMake or `pip install -e .`"
        )


# -----------------------------------------------------------------------------
# High-Level Configuration & Data Structures
# -----------------------------------------------------------------------------
@dataclass
class SamplingParams:
    """Generation and sampling hyper-parameters for a request."""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 128
    priority: int = 0
    stop_token_ids: Set[int] = field(default_factory=set)
    ignore_eos: bool = False


@dataclass
class EngineConfig:
    """Global configuration parameters for the CUDAForge Engine runtime."""
    # Model Architecture Specs
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 32000
    eos_token_id: int = 2
    element_size_bytes: int = 2  # FP16 / BF16

    # KV Cache Parameters
    block_size: int = 16
    total_gpu_blocks: int = 1024
    enable_prefix_caching: bool = True
    max_blocks_per_sequence: int = 512

    # Continuous Scheduler Parameters
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192

    def to_native_kv_config(self) -> "_C.KVCacheConfig":
        """Converts Python EngineConfig to C++ KVCacheConfig struct."""
        cfg = _C.KVCacheConfig()
        cfg.num_layers = self.num_layers
        cfg.num_kv_heads = self.num_kv_heads
        cfg.head_dim = self.head_dim
        cfg.block_size = self.block_size
        cfg.element_size_bytes = self.element_size_bytes
        cfg.total_gpu_blocks = self.total_gpu_blocks
        cfg.enable_prefix_caching = self.enable_prefix_caching
        cfg.max_supported_batch_size = self.max_num_seqs
        cfg.max_blocks_per_sequence = self.max_blocks_per_sequence
        return cfg


@dataclass
class GenerationOutput:
    """Represents a generated output token step for a given request."""
    request_id: int
    token_id: int
    is_final: bool = False
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# Zero-Copy Tensor Utility
# -----------------------------------------------------------------------------
def ptr_to_torch_tensor(
    ptr: int,
    shape: torch.Size,
    dtype: torch.dtype = torch.float16,
    device_id: int = 0
) -> torch.Tensor:
    """
    Wraps a raw CUDA physical pointer from C++ into a PyTorch Tensor
    without copying memory.
    """
    if ptr == 0:
        raise ValueError("Cannot wrap a NULL CUDA memory pointer into PyTorch Tensor.")
    
    # Calculate total elements and size
    numel = shape.numel()
    element_size = torch.tensor([], dtype=dtype).element_size()
    
    # Create PyTorch tensor directly from memory address
    ctx = ctypes.c_void_p(ptr)
    tensor = torch.frombuffer(
        (ctx.value + i * element_size for i in range(numel)),
        dtype=dtype,
        count=numel
    ).view(shape)
    
    return tensor.to(f"cuda:{device_id}")


# -----------------------------------------------------------------------------
# Synchronous Core LLM Engine
# -----------------------------------------------------------------------------
class LLMEngine:
    """
    Synchronous Core Inference Engine managing continuous batching,
    PagedKV Cache memory, and native C++ execution kernels.
    """

    def __init__(self, config: EngineConfig):
        if not _HAS_NATIVE_EXTENSION:
            raise RuntimeError("Cannot initialize LLMEngine: Native _C extension module unavailable.")

        self.config = config
        self._request_counter: int = 0
        self._active_requests: Dict[int, SamplingParams] = {}

        # 1. Initialize Native C++ Memory Arena & KV Cache
        self.kv_config = self.config.to_native_kv_config()
        self.allocator = _C.BlockAllocator(
            self.config.total_gpu_blocks,
            self.config.block_size
        )
        self.kv_cache = _C.PagedKVCache(self.kv_config, self.allocator)

        # 2. Initialize Native Scheduler & Model Runner
        self.batcher = _C.ContinuousBatcher(
            self.config.max_num_seqs,
            self.config.max_num_batched_tokens
        )
        self.model_runner = _C.ModelRunner(
            self.batcher,
            self.config.vocab_size,
            self.config.eos_token_id
        )

        logger.info(
            f"CUDAForge LLMEngine initialized successfully. "
            f"GPU Blocks: {self.config.total_gpu_blocks}, Block Size: {self.config.block_size}"
        )

    def add_request(
        self,
        prompt_tokens: List[int],
        sampling_params: Optional[SamplingParams] = None
    ) -> int:
        """
        Registers a new prompt request with prefix caching and pushes it
        into the C++ continuous batcher.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        self._request_counter += 1
        req_id = self._request_counter

        # 1. Register sequence tokens into PagedKVCache (performs prefix hash matching)
        matched_prefix_tokens = self.kv_cache.register_sequence_with_prefix(
            req_id, prompt_tokens
        )
        logger.debug(f"Req ID {req_id}: Prefix Cache Hit on {matched_prefix_tokens} tokens.")

        # 2. Instantiate C++ Request Object and queue into Batcher
        cpp_request = _C.Request(
            req_id,
            prompt_tokens,
            sampling_params.max_tokens,
            sampling_params.priority
        )
        self.batcher.add_request(cpp_request)
        self._active_requests[req_id] = sampling_params

        return req_id

    def step(self) -> List[GenerationOutput]:
        """
        Executes a single forward engine step in C++ and returns generated tokens.
        """
        # 1. Trigger C++ Model Runner Step
        self.model_runner.step()

        # 2. Pop output tokens from C++ stream buffer
        stream_buffer = self.model_runner.get_stream_buffer()
        raw_outputs = stream_buffer.pop_all()

        results: List[GenerationOutput] = []
        finished_req_ids: List[int] = []

        for out in raw_outputs:
            req_id = out.request_id
            token_id = out.token_id
            is_final = out.is_final

            sampling_params = self._active_requests.get(req_id)
            if sampling_params:
                # Check for custom stop token matching
                if token_id in sampling_params.stop_token_ids:
                    is_final = True

            # Update KV cache for decode steps
            if not is_final:
                self.kv_cache.append_tokens(req_id, 1)

            results.append(GenerationOutput(
                request_id=req_id,
                token_id=token_id,
                is_final=is_final
            ))

            if is_final:
                finished_req_ids.append(req_id)

        # 3. Clean up finished sequences in C++ Engine and KV Cache
        if finished_req_ids:
            self.batcher.update_requests_state(finished_req_ids)
            for req_id in finished_req_ids:
                self.kv_cache.unregister_sequence(req_id)
                self._active_requests.pop(req_id, None)

        return results

    def cancel_request(self, request_id: int) -> None:
        """Cancels an active request and frees allocated memory blocks."""
        self.model_runner.cancel_request(request_id)
        self.kv_cache.unregister_sequence(request_id)
        self._active_requests.pop(request_id, None)
        logger.info(f"Request {request_id} cancelled successfully.")

    def get_stats(self) -> _C.KVCacheStats:
        """Returns physical KV cache and memory pool runtime metrics."""
        return self.kv_cache.get_stats()

    def get_key_buffer_tensor(self, block_id: int, layer_idx: int) -> torch.Tensor:
        """Zero-copy accessor retrieving PyTorch Tensor view over C++ Key block pointer."""
        raw_ptr = self.kv_cache.get_key_block_ptr(block_id, layer_idx)
        shape = torch.Size([
            self.config.num_kv_heads,
            self.config.block_size,
            self.config.head_dim
        ])
        return ptr_to_torch_tensor(raw_ptr, shape)

    def get_value_buffer_tensor(self, block_id: int, layer_idx: int) -> torch.Tensor:
        """Zero-copy accessor retrieving PyTorch Tensor view over C++ Value block pointer."""
        raw_ptr = self.kv_cache.get_value_block_ptr(block_id, layer_idx)
        shape = torch.Size([
            self.config.num_kv_heads,
            self.config.block_size,
            self.config.head_dim
        ])
        return ptr_to_torch_tensor(raw_ptr, shape)


# -----------------------------------------------------------------------------
# Asynchronous High-Throughput Serving Engine
# -----------------------------------------------------------------------------
class AsyncLLMEngine:
    """
    Asynchronous Wrapper providing non-blocking async generator streaming outputs.
    Ideal for integration into web servers (FastAPI, gRPC).
    """

    def __init__(self, engine_config: EngineConfig):
        self.engine = LLMEngine(engine_config)
        self._request_queues: Dict[int, asyncio.Queue] = {}
        self._loop_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

    async def start(self) -> None:
        """Starts background step processing loop."""
        if not self._is_running:
            self._is_running = True
            self._loop_task = asyncio.create_task(self._background_engine_loop())
            logger.info("AsyncLLMEngine processing loop started.")

    async def shutdown(self) -> None:
        """Gracefully stops engine processing loop."""
        if self._is_running:
            self._is_running = False
            if self._loop_task:
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except asyncio.CancelledError:
                    pass
            logger.info("AsyncLLMEngine shutdown complete.")

    async def _background_engine_loop(self) -> None:
        """Background worker continuously invoking steps on synchronous core engine."""
        while self._is_running:
            try:
                if self.engine.batcher.get_running_count() > 0 or self.engine.batcher.get_pending_count() > 0:
                    outputs = self.engine.step()
                    for out in outputs:
                        queue = self._request_queues.get(out.request_id)
                        if queue:
                            await queue.put(out)
                            if out.is_final:
                                self._request_queues.pop(out.request_id, None)
                    # Yield control briefly to prevent event loop starvation
                    await asyncio.sleep(0.0001)
                else:
                    # Idle sleep when no active requests exist
                    await asyncio.sleep(0.005)
            except Exception as e:
                logger.error(f"Error in AsyncLLMEngine background loop: {e}", exc_info=True)
                await asyncio.sleep(0.01)

    async def generate(
        self,
        prompt_tokens: List[int],
        sampling_params: Optional[SamplingParams] = None
    ) -> AsyncGenerator[GenerationOutput, None]:
        """
        Submits request and yields generated tokens as an asynchronous stream.
        """
        if not self._is_running:
            await self.start()

        queue: asyncio.Queue = asyncio.Queue()
        req_id = self.engine.add_request(prompt_tokens, sampling_params)
        self._request_queues[req_id] = queue

        try:
            while True:
                output: GenerationOutput = await queue.get()
                yield output
                queue.task_done()
                if output.is_final:
                    break
        except asyncio.CancelledError:
            self.engine.cancel_request(req_id)
            self._request_queues.pop(req_id, None)
            raise

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()