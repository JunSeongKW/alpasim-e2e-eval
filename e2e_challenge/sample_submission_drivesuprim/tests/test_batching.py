from __future__ import annotations

import threading

import pytest
from drivesuprim_challenge.batching import InferenceMicroBatcher


def test_two_concurrent_requests_share_one_batch() -> None:
    calls: list[list[int]] = []
    batcher = InferenceMicroBatcher(
        lambda requests: calls.append(requests) or [value * 10 for value in requests],
        max_batch_size=2,
        batch_wait_s=0.05,
    )
    barrier = threading.Barrier(3)
    results: dict[int, int] = {}

    def submit(value: int) -> None:
        barrier.wait()
        results[value] = batcher.submit(value)

    threads = [threading.Thread(target=submit, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    batcher.close()

    assert len(calls) == 1
    assert sorted(calls[0]) == [1, 2]
    assert results == {1: 10, 2: 20}


def test_single_request_runs_after_batch_window() -> None:
    calls: list[list[int]] = []
    batcher = InferenceMicroBatcher(
        lambda requests: calls.append(requests) or requests,
        max_batch_size=2,
        batch_wait_s=0.001,
    )

    assert batcher.submit(7) == 7
    batcher.close()
    assert calls == [[7]]


def test_batch_error_is_returned_to_every_request() -> None:
    def fail(requests: list[int]) -> list[int]:
        raise ValueError(f"bad batch {requests}")

    batcher = InferenceMicroBatcher(
        fail,
        max_batch_size=2,
        batch_wait_s=0.05,
    )
    barrier = threading.Barrier(3)
    errors: list[str] = []

    def submit(value: int) -> None:
        barrier.wait()
        try:
            batcher.submit(value)
        except ValueError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=submit, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    batcher.close()

    assert len(errors) == 2
    assert all(error.startswith("bad batch") for error in errors)


def test_closed_batcher_rejects_new_requests() -> None:
    batcher = InferenceMicroBatcher(lambda requests: requests)
    batcher.close()

    with pytest.raises(RuntimeError, match="closed"):
        batcher.submit(1)

