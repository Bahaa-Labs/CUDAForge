import asyncio
from unittest.mock import MagicMock, patch
import pytest
import torch

from cudaforge.engine import (
    AsyncLLMEngine,
    EngineConfig,
    GenerationOutput,
    LLMEngine,
    SamplingParams,
)


@pytest.fixture
def sample_config():
    return EngineConfig(
        num_layers=4,
        num_kv_heads=2,
        head_dim=64,
        vocab_size=1000,
        eos_token_id=2,
        block_size=16,
        total_gpu_blocks=64,
        max_num_seqs=16,
        max_num_batched_tokens=512,
    )


def test_engine_config_native_conversion(sample_config):
    """Verify EngineConfig maps correctly to C++ KVCacheConfig properties."""
    with patch("cudaforge.engine._C") as mock_C:
        mock_kv_cfg = MagicMock()
        mock_C.KVCacheConfig.return_value = mock_kv_cfg

        native_cfg = sample_config.to_native_kv_config()

        assert native_cfg.num_layers == 4
        assert native_cfg.num_kv_heads == 2
        assert native_cfg.head_dim == 64
        assert native_cfg.block_size == 16
        assert native_cfg.total_gpu_blocks == 64


def test_llm_engine_request_lifecycle(sample_config):
    """Test registering request and stepping through engine execution."""
    with patch("cudaforge.engine._C") as mock_C:
        # Mock C++ native components
        mock_kv = MagicMock()
        mock_kv.register_sequence_with_prefix.return_value = 0
        mock_C.PagedKVCache.return_value = mock_kv

        mock_batcher = MagicMock()
        mock_C.ContinuousBatcher.return_value = mock_batcher

        mock_runner = MagicMock()
        mock_C.ModelRunner.return_value = mock_runner

        # Mock C++ stream buffer output
        mock_stream = MagicMock()
        mock_output = MagicMock()
        mock_output.request_id = 1
        mock_output.token_id = 42
        mock_output.is_final = False
        mock_stream.pop_all.return_value = [mock_output]
        mock_runner.get_stream_buffer.return_value = mock_stream

        # Instantiate engine
        engine = LLMEngine(sample_config)

        # 1. Add Request
        prompt = [101, 202, 303]
        req_id = engine.add_request(prompt)
        assert req_id == 1

        # 2. Execute Step
        outputs = engine.step()
        assert len(outputs) == 1
        assert outputs[0].request_id == 1
        assert outputs[0].token_id == 42
        assert outputs[0].is_final is False

        # Verify C++ bindings were called correctly
        mock_kv.register_sequence_with_prefix.assert_called_once_with(1, prompt)
        mock_runner.step.assert_called_once()
        mock_kv.append_tokens.assert_called_once_with(1, 1)


@pytest.mark.asyncio
async def test_async_llm_engine_streaming(sample_config):
    """Test non-blocking AsyncLLMEngine streaming interface."""
    with patch("cudaforge.engine._C") as mock_C:
        mock_kv = MagicMock()
        mock_kv.register_sequence_with_prefix.return_value = 0
        mock_C.PagedKVCache.return_value = mock_kv

        mock_batcher = MagicMock()
        # Allow running count to stay 1 for 2 steps, then terminate with 0
        mock_batcher.get_running_count.side_effect = [1, 1, 0]
        mock_batcher.get_pending_count.return_value = 0
        mock_C.ContinuousBatcher.return_value = mock_batcher

        mock_runner = MagicMock()
        mock_C.ModelRunner.return_value = mock_runner

        # Setup 2 generation steps: 1 token then final EOS, followed by empty list
        out1 = MagicMock(request_id=1, token_id=50, is_final=False)
        out2 = MagicMock(request_id=1, token_id=2, is_final=True)

        mock_stream = MagicMock()
        mock_stream.pop_all.side_effect = [[out1], [out2], []]
        mock_runner.get_stream_buffer.return_value = mock_stream

        async with AsyncLLMEngine(sample_config) as async_engine:
            tokens_received = []
            async for output in async_engine.generate([1, 2, 3]):
                tokens_received.append(output.token_id)
                if output.is_final:
                    break

            assert tokens_received == [50, 2]
