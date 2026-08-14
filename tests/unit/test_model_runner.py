import pytest
import torch
from cudaforge import ContinuousBatcher, Request, ModelRunner


def test_model_runner_streaming_and_cancellation():
    batcher = ContinuousBatcher(max_num_seqs=4, max_num_batched_tokens=512)
    runner = ModelRunner(batcher=batcher, vocab_size=1000, eos_token_id=999)

    # Request 1: Normal generation (5 tokens)
    req1 = Request(req_id=1, prompt=[10, 20], max_tokens=5)
    # Request 2: Cancelled mid-flight
    req2 = Request(req_id=2, prompt=[30, 40], max_tokens=10)

    batcher.add_request(req1)
    batcher.add_request(req2)

    # Step 1: Prefill
    runner.step()

    # Mid-flight cancellation of request 2
    runner.cancel_request(2)
    assert runner.is_cancelled(2)

    # Step 2: Decode step
    runner.step()

    # Verify Stream Output Buffer
    stream_buffer = runner.get_stream_buffer()
    tokens = stream_buffer.pop_all()

    req1_tokens = [t for t in tokens if t.request_id == 1]
    req2_tokens = [t for t in tokens if t.request_id == 2]

    assert len(req1_tokens) > 0
    # Req 2 should have stopped producing tokens after cancellation
    assert len(req2_tokens) <= 1
