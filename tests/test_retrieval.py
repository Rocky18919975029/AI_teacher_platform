"""Microbatch admission, cancellation, deadlines and graceful native-work draining."""

import asyncio
import threading

import pytest

from teacher.retrieval import (
    RetrievalClosed,
    RetrievalOverloaded,
    RetrievalScheduler,
    RetrievalTimedOut,
)


class Service:
    def __init__(self):
        self.batches = []
        self.threads = []

    def search_many(self, requests):
        self.batches.append(requests)
        self.threads.append(threading.current_thread().name)
        return [[{'course': course, 'query': query}] for course, query in requests]


class BlockingService(Service):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def search_many(self, requests):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError('test failed to release service')
        result = super().search_many(requests)
        self.finished.set()
        return result


async def eventually(predicate, timeout=1):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


async def test_40_simultaneous_requests_aggregate_without_lost_or_mixed_results():
    service = Service()
    scheduler = RetrievalScheduler(service, capacity=40, batch_size=8, wait_ms=8)
    await scheduler.start()
    try:
        requests = [(f'course-{i % 4}', f'question-{i}') for i in range(40)]
        results = await asyncio.wait_for(
            asyncio.gather(
                *(scheduler.search(course, query) for course, query in requests),
            ),
            timeout=2,
        )
        assert results == [[{'course': course, 'query': query}] for course, query in requests]
        assert len(service.batches) == 5
        assert all(len(batch) == 8 for batch in service.batches)
        assert len(set(service.threads)) == 1
        assert service.threads[0].startswith('teacher-retrieval')
        assert scheduler.outstanding == scheduler.queued == 0
    finally:
        await scheduler.close()


async def test_microbatch_wait_collects_later_arrivals_without_waiting_full_deadline():
    service = Service()
    scheduler = RetrievalScheduler(service, capacity=8, batch_size=8, wait_ms=30)
    await scheduler.start()
    try:
        first = asyncio.create_task(scheduler.search('a', 'one'))
        await asyncio.sleep(0.005)
        second = asyncio.create_task(scheduler.search('b', 'two'))
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        assert service.batches == [[('a', 'one'), ('b', 'two')]]
    finally:
        await scheduler.close()


async def test_overload_rejects_fast_and_canceling_active_request_keeps_capacity_reserved():
    service = BlockingService()
    scheduler = RetrievalScheduler(service, capacity=2, batch_size=1, wait_ms=0)
    await scheduler.start()
    active = asyncio.create_task(scheduler.search('a', 'active'))
    queued = None
    try:
        await eventually(service.started.is_set)
        queued = asyncio.create_task(scheduler.search('b', 'queued'))
        await eventually(lambda: scheduler.outstanding == 2)
        with pytest.raises(RetrievalOverloaded):
            await asyncio.wait_for(scheduler.search('c', 'excess'), timeout=0.1)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert scheduler.outstanding == 2 and scheduler.queued == 1
        with pytest.raises(RetrievalOverloaded):
            await scheduler.search('c', 'canceled-active-still-costs-work')
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert scheduler.outstanding == 1 and scheduler.queued == 0
        service.release.set()
        await eventually(lambda: scheduler.outstanding == 0)
        assert service.batches == [[('a', 'active')]]
    finally:
        service.release.set()
        if queued is not None:
            queued.cancel()
        await scheduler.close()


async def test_queued_deadlines_are_removed_and_stalled_active_work_stays_bounded():
    service = BlockingService()
    scheduler = RetrievalScheduler(service, capacity=2, batch_size=1, wait_ms=0, timeout=0.05)
    await scheduler.start()
    try:
        active = asyncio.create_task(scheduler.search('a', 'active'))
        await eventually(service.started.is_set)
        queued = asyncio.create_task(scheduler.search('b', 'expired-in-queue'))
        results = await asyncio.gather(active, queued, return_exceptions=True)
        assert all(isinstance(result, RetrievalTimedOut) for result in results)
        assert scheduler.outstanding == 1 and scheduler.queued == 0
        # More timed-out clients cannot submit additional native work behind the stall.
        for _ in range(2):
            with pytest.raises(RetrievalTimedOut):
                await scheduler.search('c', 'queued-then-expired')
            assert scheduler.outstanding == 1 and scheduler.queued == 0
        service.release.set()
        await eventually(lambda: scheduler.outstanding == 0)
        assert service.batches == [[('a', 'active')]]
    finally:
        service.release.set()
        await scheduler.close()


async def test_cancel_during_batch_wait_never_reaches_backend_and_frees_admission():
    service = Service()
    scheduler = RetrievalScheduler(service, capacity=2, batch_size=2, wait_ms=50)
    await scheduler.start()
    # A partial batch waits for another arrival, giving cancellation a queue phase.
    task = asyncio.create_task(scheduler.search('a', 'canceled'))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    try:
        await eventually(lambda: scheduler.outstanding == 0)
        assert not service.batches
        assert await scheduler.search('b', 'replacement') == [{'course': 'b', 'query': 'replacement'}]
    finally:
        await scheduler.close()


async def test_shutdown_rejects_queued_work_and_drains_active_before_returning():
    service = BlockingService()
    scheduler = RetrievalScheduler(service, capacity=2, batch_size=1, wait_ms=0)
    await scheduler.start()
    active = asyncio.create_task(scheduler.search('a', 'active'))
    closing = None
    try:
        await eventually(service.started.is_set)
        queued = asyncio.create_task(scheduler.search('b', 'queued'))
        await eventually(lambda: scheduler.queued == 1)
        closing = asyncio.create_task(scheduler.close())
        with pytest.raises(RetrievalClosed):
            await queued
        with pytest.raises(RetrievalClosed):
            await scheduler.search('c', 'new')
        assert not closing.done() and not service.finished.is_set()
        service.release.set()
        await asyncio.wait_for(closing, timeout=1)
        assert service.finished.is_set()
        assert await active == [{'course': 'a', 'query': 'active'}]
        assert scheduler.outstanding == scheduler.queued == 0
        await scheduler.close()
    finally:
        service.release.set()
        await scheduler.close()


async def test_canceling_close_does_not_abandon_native_batch():
    service = BlockingService()
    scheduler = RetrievalScheduler(service, capacity=1, batch_size=1, wait_ms=0)
    await scheduler.start()
    active = asyncio.create_task(scheduler.search('a', 'active'))
    try:
        await eventually(service.started.is_set)
        closing = asyncio.create_task(scheduler.close())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not service.finished.is_set()
        service.release.set()
        await scheduler.close()
        assert service.finished.is_set()
        assert await active == [{'course': 'a', 'query': 'active'}]
    finally:
        service.release.set()
        await scheduler.close()


@pytest.mark.parametrize('failure', ['exception', 'count', 'type'])
async def test_bad_batch_fails_all_futures_without_partial_results_and_worker_recovers(failure):
    class BrokenOnce(Service):
        def search_many(self, requests):
            if not self.batches:
                self.batches.append(requests)
                if failure == 'exception':
                    raise RuntimeError('backend failed')
                if failure == 'count':
                    return [[]]
                return ['wrong'] * len(requests)
            return super().search_many(requests)

    scheduler = RetrievalScheduler(BrokenOnce(), capacity=4, batch_size=4, wait_ms=20)
    await scheduler.start()
    try:
        results = await asyncio.gather(
            *(scheduler.search(str(i), 'q') for i in range(4)), return_exceptions=True
        )
        assert all(isinstance(result, RuntimeError) for result in results)
        assert scheduler.outstanding == 0
        assert await scheduler.search('safe', 'again') == [{'course': 'safe', 'query': 'again'}]
    finally:
        await scheduler.close()


async def test_legacy_single_search_runs_on_dedicated_executor():
    class Legacy:
        def search(self, course, query):
            assert threading.current_thread().name.startswith('teacher-retrieval')
            return [{'course': course, 'query': query}]

    scheduler = RetrievalScheduler(Legacy())
    with pytest.raises(RetrievalClosed):
        await scheduler.search('a', 'not-started')
    await scheduler.start()
    await scheduler.start()
    assert await scheduler.search('a', 'q') == [{'course': 'a', 'query': 'q'}]
    await scheduler.close()
    with pytest.raises(RetrievalClosed):
        await scheduler.start()
