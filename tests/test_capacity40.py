"""40-user application acceptance with deterministic fakes, not GPU benchmarks."""

import asyncio
import json
import threading

import httpx
import pytest

from teacher.api import create_app
from teacher.config import Settings


def key(index):
    return f'capacity-user-{index:02d}-' + 'x' * 32


def headers(index):
    return {'Authorization': f'Bearer {key(index)}'}


class ScopedRag:
    def __init__(self):
        self.calls = []
        self.block = False
        self.release = threading.Event()

    def search_many(self, requests):
        if self.block and not self.release.wait(10):
            raise TimeoutError('Test did not release retrieval')
        self.calls.extend(requests)
        return [
            [
                {
                    'id': 'chunk',
                    'document_id': course,
                    'title': course,
                    'page': 1,
                    'text': f'Course material for {course}.',
                    'score': 0.9,
                }
            ]
            for course, _ in requests
        ]

    def health(self):
        return True

    def close(self):
        self.release.set()


class BarrierInference:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.finished = 0
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, messages):
        self.active += 1
        self.peak = max(self.active, self.peak)
        if self.active == 40:
            self.all_started.set()
        try:
            yield 'Course answer '
            await self.release.wait()
            yield '[S1].'
        finally:
            self.active -= 1
            self.finished += 1

    async def health(self):
        return True

    async def close(self):
        self.release.set()


@pytest.fixture
async def classroom(tmp_path):
    cfg = Settings(
        _env_file=None,
        data_dir=tmp_path,
        api_keys=[{'key': key(i), 'subject': f'user-{i}', 'courses': [f'course-{i}']} for i in range(41)],
    )
    assert cfg.max_concurrent_chats == 40
    rag, inference = ScopedRag(), BarrierInference()
    app = create_app(cfg, rag, inference)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            yield app, client, rag, inference


async def request(client, index, mode='chat', **extras):
    payload = {'course_id': f'course-{index}'}
    payload['prompt' if mode == 'chat' else 'query'] = f'Explain course {index}'
    payload.update(extras)
    path = '/api/chat/stream' if mode == 'chat' else '/api/search'
    return await client.post(path, headers=headers(index), json=payload)


def done(response):
    assert response.status_code == 200
    events = [block for block in response.text.strip().split('\n\n')]
    assert not any(block.startswith('event: error') for block in events)
    assert events[-1].startswith('event: done')
    return json.loads(events[-1].split('data: ', 1)[1])


async def wait_until(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.005)


async def assert_capacity_restored(app):
    for _ in range(40):
        await asyncio.wait_for(app.state.capacity.acquire(), timeout=1)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(app.state.capacity.acquire(), timeout=0.02)
    finally:
        for _ in range(40):
            app.state.capacity.release()


async def test_40_distinct_streams_course_isolation_41st_overload_then_recovery(classroom):
    app, client, rag, inference = classroom
    tasks = [asyncio.create_task(request(client, i)) for i in range(40)]
    try:
        await asyncio.wait_for(inference.all_started.wait(), timeout=10)
        assert inference.active == inference.peak == 40
        assert app.state.capacity.locked()
        assert len(rag.calls) == 40
        overload = await request(client, 40)
        assert overload.status_code == 503
        assert overload.headers['Retry-After'] == '2'
        assert inference.active == 40
        assert (await client.get('/health/live')).status_code == 200
        inference.release.set()
        responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
        session_ids = []
        for i, response in enumerate(responses):
            result = done(response)
            assert result['sources'][0]['document_id'] == f'course-{i}'
            session_ids.append(result['session_id'])
            saved = app.state.store.session(result['session_id'], f'user-{i}')
            assert saved['busy'] == 0
            assert len(saved['messages']) == 2
            denied = await client.get(f'/api/sessions/{result["session_id"]}', headers=headers((i + 1) % 40))
            assert denied.status_code == 404
        assert len(set(session_ids)) == 40
        assert inference.active == 0
        assert done(await request(client, 40))['sources'][0]['document_id'] == 'course-40'
        await assert_capacity_restored(app)
    finally:
        inference.release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_40_mixed_chat_and_search_keep_health_responsive(classroom):
    app, client, rag, inference = classroom
    rag.block = True
    inference.release.set()
    tasks = [asyncio.create_task(request(client, i, 'chat' if i % 2 == 0 else 'search')) for i in range(40)]
    try:
        await wait_until(lambda: app.state.retrieval.outstanding == 40)
        assert app.state.capacity.locked()
        # Generous deadline checks event-loop liveness, not a hardware latency SLO.
        live, ready = await asyncio.wait_for(
            asyncio.gather(client.get('/health/live'), client.get('/health/ready')), timeout=2
        )
        assert live.status_code == ready.status_code == 200
        rejected = await request(client, 40, 'search')
        assert rejected.status_code == 503
        rag.release.set()
        responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
        for i, response in enumerate(responses):
            assert response.status_code == 200
            result = done(response) if i % 2 == 0 else response.json()
            assert result['sources'][0]['document_id'] == f'course-{i}'
        assert len(rag.calls) == 40
        assert inference.finished == 20
        assert app.state.retrieval.outstanding == 0
        await assert_capacity_restored(app)
    finally:
        rag.release.set()
        inference.release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_cancel_40_streams_releases_all_sessions_upstreams_and_permits(classroom):
    app, client, _, inference = classroom
    sessions = [app.state.store.create_session(f'user-{i}', f'course-{i}') for i in range(40)]
    tasks = [asyncio.create_task(request(client, i, session_id=sessions[i])) for i in range(40)]
    try:
        await asyncio.wait_for(inference.all_started.wait(), timeout=10)
        for task in tasks:
            task.cancel()
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        assert inference.active == 0
        assert inference.finished == 40
        for i, sid in enumerate(sessions):
            saved = app.state.store.session(sid, f'user-{i}')
            assert saved['busy'] == 0
            assert saved['messages'] == []
        await assert_capacity_restored(app)
        inference.release.set()
        assert done(await request(client, 0, session_id=sessions[0]))['session_id'] == sessions[0]
    finally:
        inference.release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
