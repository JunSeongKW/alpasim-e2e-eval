"""Small synchronous micro-batcher for concurrent inference requests."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass
class _PendingRequest(Generic[RequestT, ResultT]):
    request: RequestT
    done: threading.Event = field(default_factory=threading.Event)
    result: ResultT | None = None
    error: Exception | None = None


class InferenceMicroBatcher(Generic[RequestT, ResultT]):
    """Combine nearby synchronous requests into one bounded inference batch."""

    def __init__(
        self,
        predict_batch: Callable[[list[RequestT]], list[ResultT]],
        *,
        max_batch_size: int = 2,
        batch_wait_s: float = 0.005,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if batch_wait_s < 0:
            raise ValueError("batch_wait_s must be non-negative")
        self._predict_batch = predict_batch
        self._max_batch_size = max_batch_size
        self._batch_wait_s = batch_wait_s
        self._pending: deque[_PendingRequest[RequestT, ResultT]] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="drivesuprim-inference-batcher",
            daemon=True,
        )
        self._worker.start()

    def submit(self, request: RequestT) -> ResultT:
        pending = _PendingRequest[RequestT, ResultT](request=request)
        with self._condition:
            if self._closed:
                raise RuntimeError("inference batcher is closed")
            self._pending.append(pending)
            self._condition.notify()
        pending.done.wait()
        if pending.error is not None:
            raise pending.error
        return pending.result  # type: ignore[return-value]

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join()

    def _run(self) -> None:
        while True:
            batch = self._next_batch()
            if batch is None:
                return
            try:
                results = self._predict_batch([item.request for item in batch])
                if len(results) != len(batch):
                    raise RuntimeError(
                        "batch predictor returned "
                        f"{len(results)} results for {len(batch)} requests"
                    )
            except Exception as exc:  # noqa: BLE001 - propagate batch failures to callers
                for item in batch:
                    item.error = exc
                    item.done.set()
                continue
            for item, result in zip(batch, results, strict=True):
                item.result = result
                item.done.set()

    def _next_batch(
        self,
    ) -> list[_PendingRequest[RequestT, ResultT]] | None:
        with self._condition:
            self._condition.wait_for(lambda: self._pending or self._closed)
            if not self._pending:
                return None

            deadline = time.monotonic() + self._batch_wait_s
            while len(self._pending) < self._max_batch_size and not self._closed:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    break
                self._condition.wait(timeout=remaining_s)

            size = min(len(self._pending), self._max_batch_size)
            return [self._pending.popleft() for _ in range(size)]
