"""Bounded asynchronous admission and microbatching for synchronous RAG models.

One dedicated executor thread runs at most one model batch at a time. Client
cancellation/deadlines do not cancel native model work: active reservations stay
counted until that batch finishes, preventing cancellation from bypassing capacity.
"""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


class RetrievalOverloaded(RuntimeError):
    """The bounded retrieval queue and active batch are at capacity."""


class RetrievalTimedOut(TimeoutError):
    """The request exceeded its end-to-end queue/model deadline."""


class RetrievalClosed(RuntimeError):
    """The scheduler has not started or is shutting down."""


@dataclass(eq=False)
class _Request:
    course: str
    query: str
    future: asyncio.Future
    deadline: float
    active: bool = False


class RetrievalScheduler:
    def __init__(self, service: Any, capacity=40, batch_size=8, wait_ms=8, timeout=60):
        if not 1 <= capacity <= 10000 or not 1 <= batch_size <= min(capacity, 128):
            raise ValueError('Require 1 <= batch_size <= min(capacity, 128) and capacity <= 10000')
        if not 0 <= wait_ms <= 1000 or not 0 < timeout <= 3600:
            raise ValueError('Require 0 <= wait_ms <= 1000 and 0 < timeout <= 3600')
        self.service = service
        self.capacity = capacity
        self.batch_size = batch_size
        self.wait_seconds = wait_ms / 1000
        self.timeout = timeout
        self._queue: deque[_Request] = deque()
        self._requests: set[_Request] = set()
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closing = False

    @property
    def outstanding(self) -> int:
        """Queued plus active reservations, including abandoned active model work."""
        return len(self._requests)

    @property
    def queued(self) -> int:
        return len(self._queue)

    async def start(self):
        if self._closing:
            raise RetrievalClosed('Retrieval scheduler is closed')
        if self._worker is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='teacher-retrieval')
            self._worker = asyncio.create_task(self._dispatch(), name='teacher-retrieval-dispatch')

    async def search(self, course: str, query: str) -> list[dict]:
        if self._closing or self._worker is None or self._worker.done():
            raise RetrievalClosed('Retrieval scheduler is unavailable')
        # No awaits between admission check and reservation; one event loop owns state.
        self._prune()
        if self.outstanding >= self.capacity:
            raise RetrievalOverloaded('Retrieval capacity reached')
        loop = asyncio.get_running_loop()
        request = _Request(course, query, loop.create_future(), loop.time() + self.timeout)
        self._requests.add(request)
        self._queue.append(request)
        self._wake.set()
        try:
            async with asyncio.timeout_at(request.deadline):
                return await request.future
        except TimeoutError as exc:
            raise RetrievalTimedOut('Retrieval deadline exceeded') from exc
        finally:
            if not request.future.done():
                request.future.cancel()
            if not request.active:
                self._discard_queued(request)

    def _discard_queued(self, request):
        try:
            self._queue.remove(request)
        except ValueError:
            pass
        self._requests.discard(request)

    def _prune(self):
        now = asyncio.get_running_loop().time()
        for request in tuple(self._queue):
            if request.future.done() or request.deadline <= now:
                self._discard_queued(request)
                if not request.future.done():
                    request.future.set_exception(RetrievalTimedOut('Retrieval deadline exceeded'))

    def _run_batch(self, requests):
        method = getattr(self.service, 'search_many', None)
        if callable(method):
            results = method(requests)
        else:
            # Injected legacy test/services still run only on our dedicated worker.
            results = [self.service.search(course, query) for course, query in requests]
        if (
            not isinstance(results, list)
            or len(results) != len(requests)
            or any(not isinstance(result, list) for result in results)
        ):
            raise RuntimeError('Retrieval provider returned an invalid batch')
        return results

    async def _dispatch(self):
        loop = asyncio.get_running_loop()
        try:
            while not self._closing:
                await self._wake.wait()
                self._wake.clear()
                self._prune()
                if not self._queue:
                    continue
                cutoff = loop.time() + self.wait_seconds
                while len(self._queue) < self.batch_size and not self._closing:
                    remaining = min(cutoff, self._queue[0].deadline) - loop.time() if self._queue else 0
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                        self._wake.clear()
                    except TimeoutError:
                        break
                    self._prune()
                self._prune()
                if self._closing:
                    break
                batch = []
                while self._queue and len(batch) < self.batch_size:
                    request = self._queue.popleft()
                    request.active = True
                    batch.append(request)
                if not batch:
                    continue
                try:
                    results = await loop.run_in_executor(
                        self._executor,
                        self._run_batch,
                        [(r.course, r.query) for r in batch],
                    )
                except Exception as exc:
                    for request in batch:
                        if not request.future.done():
                            request.future.set_exception(exc)
                else:
                    for request, result in zip(batch, results):
                        if not request.future.done():
                            if request.deadline <= loop.time():
                                request.future.set_exception(RetrievalTimedOut('Retrieval deadline exceeded'))
                            else:
                                request.future.set_result(result)
                finally:
                    for request in batch:
                        self._requests.discard(request)
                if self._queue:
                    self._wake.set()
        finally:
            for request in tuple(self._requests):
                if not request.future.done():
                    request.future.set_exception(RetrievalClosed('Retrieval scheduler stopped'))
            self._queue.clear()
            self._requests.clear()

    async def close(self):
        """Reject new/queued work and drain the running model batch; never close Qdrant here."""
        if self._close_task is None:
            self._closing = True
            for request in tuple(self._queue):
                self._discard_queued(request)
                if not request.future.done():
                    request.future.set_exception(RetrievalClosed('Retrieval scheduler is shutting down'))
            self._wake.set()
            self._close_task = asyncio.create_task(self._shutdown())
        # A caller canceling lifespan cleanup must not cancel native work ownership.
        await asyncio.shield(self._close_task)

    async def _shutdown(self):
        if self._worker is not None:
            await self._worker
        if self._executor is not None:
            # _dispatch has joined its only in-flight future, so this is nonblocking.
            self._executor.shutdown(wait=True, cancel_futures=True)
