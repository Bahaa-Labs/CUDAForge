import pytest
from cudaforge import ContinuousBatcher, Request, RequestState


def test_continuous_batcher_priority_and_budget():
    # Max 2 sequences, max 300 total batched tokens
    batcher = ContinuousBatcher(max_num_seqs=2, max_num_batched_tokens=300)

    req1 = Request(
        req_id=1, prompt=[101, 102] * 50, max_tokens=10, priority=1
    )  # 100 tokens
    req2 = Request(
        req_id=2, prompt=[201, 202] * 100, max_tokens=10, priority=5
    )  # 200 tokens (High Priority)
    req3 = Request(
        req_id=3, prompt=[301, 302] * 20, max_tokens=10, priority=2
    )  # 40 tokens

    assert batcher.add_request(req1)
    assert batcher.add_request(req2)
    assert batcher.add_request(req3)

    assert batcher.get_pending_count() == 3

    # Step 1: Schedule
    # Priority order: Req 2 (prio 5, 200 tokens), Req 3 (prio 2, 40 tokens).
    # Total tokens = 240 <= 300, Seqs = 2 <= 2. Req 1 stays in waiting queue.
    step1 = batcher.schedule_step()

    assert len(step1.prefill_requests) == 2
    assert step1.prefill_requests[0].id == 2
    assert step1.prefill_requests[1].id == 3
    assert step1.total_batched_tokens == 240
    assert batcher.get_running_count() == 2
    assert batcher.get_pending_count() == 1


def test_continuous_batcher_cancel():
    batcher = ContinuousBatcher(max_num_seqs=4, max_num_batched_tokens=1024)
    req = Request(req_id=99, prompt=[1, 2, 3], max_tokens=20)

    batcher.add_request(req)
    assert batcher.get_pending_count() == 1

    assert batcher.cancel_request(99)
    step = batcher.schedule_step()
    assert len(step.prefill_requests) == 0
