"""Concurrency regressions through ASGI and the application's real lifespan."""

import asyncio
import threading
from contextlib import suppress

import httpx
import pytest

from teacher.api import ClosingStreamResponse, create_app
from teacher.config import Settings
from teacher.storage import Store

KEY = 'concurrency-test-key-' * 3
AUTH = {'Authorization': f'Bearer {KEY}'}


class Rag:
    def __init__(self):
        self.searches = 0
        self.closed = False

    def search(self, course, query):
        self.searches += 1
        return [
            {
                'id': 'chunk',
                'document_id': 'doc',
                'title': 'Notes',
                'page': 1,
                'text': 'Gravity attracts masses.',
                'score': 0.9,
            }
        ]

    def health(self):
        return True

    def close(self):
        self.closed = True


class Inference:
    def __init__(self):
        self.started = asyncio.Event()
        self.continue_stream = asyncio.Event()
        self.canceled = asyncio.Event()

    async def stream(self, messages):
        self.started.set()
        try:
            yield 'Gravity '
            await self.continue_stream.wait()
            yield 'attracts masses. [S1]'
        finally:
            self.canceled.set()

    async def health(self):
        return True

    async def close(self):
        pass


def config(path):
    return Settings(
        _env_file=None,
        data_dir=path,
        max_concurrent_chats=1,
        api_keys=[{'key': KEY, 'subject': 'alice', 'role': 'admin', 'courses': ['physics']}],
    )


@pytest.fixture
async def running(tmp_path):
    rag, inference = Rag(), Inference()
    app = create_app(config(tmp_path), rag, inference)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            yield app, client, rag, inference


async def assert_exactly_one_permit(capacity):
    await asyncio.wait_for(capacity.acquire(), timeout=1)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(capacity.acquire(), timeout=0.02)
    finally:
        capacity.release()


async def test_cancel_while_waiting_for_capacity_releases_session(running, monkeypatch):
    app, client, _, _ = running
    sid = app.state.store.create_session('alice', 'physics')
    claimed = asyncio.Event()
    original = app.state.store.claim_session

    def claim(*args):
        result = original(*args)
        if result:
            claimed.set()
        return result

    monkeypatch.setattr(app.state.store, 'claim_session', claim)
    await app.state.capacity.acquire()
    request = asyncio.create_task(
        client.post(
            '/ai-teacher',
            headers=AUTH,
            json={
                'course_id': 'physics',
                'session_id': sid,
                'prompt': 'What is gravity?',
            },
        )
    )
    try:
        await asyncio.wait_for(claimed.wait(), timeout=1)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert app.state.store.session(sid, 'alice')['busy'] == 0
        # The canceled request never owned the held permit and must not release it.
        assert app.state.capacity.locked()
    finally:
        request.cancel()
        with suppress(asyncio.CancelledError):
            await request
        app.state.capacity.release()


async def test_sse_cancellation_closes_upstream_and_releases_session_and_capacity(running):
    app, client, _, inference = running
    sid = app.state.store.create_session('alice', 'physics')
    request = asyncio.create_task(
        client.post(
            '/api/chat/stream',
            headers=AUTH,
            json={
                'course_id': 'physics',
                'session_id': sid,
                'prompt': 'What is gravity?',
            },
        )
    )
    try:
        await asyncio.wait_for(inference.started.wait(), timeout=1)
        assert app.state.store.session(sid, 'alice')['busy'] == 1
        assert app.state.capacity.locked()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await asyncio.wait_for(inference.canceled.wait(), timeout=1)
        assert app.state.store.session(sid, 'alice')['busy'] == 0
        assert app.state.store.session(sid, 'alice')['messages'] == []
        await assert_exactly_one_permit(app.state.capacity)
        # A subsequent request must succeed using the same session and permit.
        inference.continue_stream.set()
        response = await asyncio.wait_for(
            client.post(
                '/ai-teacher',
                headers=AUTH,
                json={
                    'course_id': 'physics',
                    'session_id': sid,
                    'prompt': 'Retry the answer',
                },
            ),
            timeout=1,
        )
        assert response.status_code == 200
        assert len(app.state.store.session(sid, 'alice')['messages']) == 2
        await assert_exactly_one_permit(app.state.capacity)
    finally:
        request.cancel()
        inference.continue_stream.set()
        with suppress(asyncio.CancelledError):
            await request


async def test_cancel_response_start_before_generator_iteration_releases_once(running, monkeypatch):
    app, client, rag, inference = running
    sid = app.state.store.create_session('alice', 'physics')
    response_start = asyncio.Event()
    hold_send = asyncio.Event()
    original = ClosingStreamResponse.stream_response

    async def paused_stream(self, send):
        async def paused_send(message):
            if message['type'] == 'http.response.start':
                response_start.set()
                await hold_send.wait()
            await send(message)

        await original(self, paused_send)

    monkeypatch.setattr(ClosingStreamResponse, 'stream_response', paused_stream)
    request = asyncio.create_task(
        client.post(
            '/api/chat/stream',
            headers=AUTH,
            json={
                'course_id': 'physics',
                'session_id': sid,
                'prompt': 'What is gravity?',
            },
        )
    )
    try:
        await asyncio.wait_for(response_start.wait(), timeout=1)
        assert not inference.started.is_set()
        assert rag.searches == 0
        assert app.state.store.session(sid, 'alice')['busy'] == 1
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert app.state.store.session(sid, 'alice')['busy'] == 0
        assert app.state.store.session(sid, 'alice')['messages'] == []
        await assert_exactly_one_permit(app.state.capacity)
    finally:
        request.cancel()
        hold_send.set()
        with suppress(asyncio.CancelledError):
            await request


async def test_search_rejects_at_capacity_without_queueing_model_work(running):
    app, client, rag, _ = running
    await app.state.capacity.acquire()
    try:
        response = await asyncio.wait_for(
            client.post(
                '/api/search',
                headers=AUTH,
                json={
                    'course_id': 'physics',
                    'query': 'gravity',
                },
            ),
            timeout=1,
        )
        assert response.status_code == 503
        assert response.headers['Retry-After'] == '2'
        assert rag.searches == 0
        assert app.state.capacity.locked()
    finally:
        app.state.capacity.release()
    response = await client.post(
        '/api/search',
        headers=AUTH,
        json={
            'course_id': 'physics',
            'query': 'gravity',
        },
    )
    assert response.status_code == 200
    assert rag.searches == 1


async def test_shutdown_drains_active_ingestion_before_closing_dependencies(tmp_path):
    class BlockingRag(Rag):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.finish = threading.Event()
            self.order = []

        def ingest(self, course, filename, data):
            self.started.set()
            if not self.finish.wait(timeout=5):
                raise TimeoutError('test failed to unblock ingestion')
            assert not self.closed
            self.order.append('ingested')
            return {'document_id': 'doc', 'title': filename, 'chunks': 1}

        def close(self):
            self.order.append('closed')
            super().close()

    rag = BlockingRag()
    app = create_app(config(tmp_path), rag, Inference())
    lifecycle = app.router.lifespan_context(app)
    await lifecycle.__aenter__()
    shutdown = None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post(
                '/api/documents?course_id=physics',
                headers=AUTH,
                files={'file': ('notes.txt', b'Gravity notes')},
            )
            assert response.status_code == 202
            job_id = response.json()['id']
            started = await asyncio.to_thread(rag.started.wait, 3)
            assert started
            shutdown = asyncio.create_task(lifecycle.__aexit__(None, None, None))
            # Shield the shutdown task so the observation timeout cannot cancel it.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(shutdown), timeout=0.05)
            assert not rag.closed
            rag.finish.set()
            await asyncio.wait_for(shutdown, timeout=2)
        assert rag.order == ['ingested', 'closed']
        assert app.state.store.job(job_id)['status'] == 'completed'
        assert list((tmp_path / 'uploads').iterdir()) == []
    finally:
        rag.finish.set()
        if shutdown is None:
            await lifecycle.__aexit__(None, None, None)
        elif not shutdown.done():
            with suppress(asyncio.CancelledError):
                await shutdown


async def test_startup_removes_orphan_uploads_but_preserves_queued_uploads(tmp_path):
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    orphan = upload_dir / ('a' * 32)
    queued = upload_dir / ('b' * 32)
    unrelated = upload_dir / 'operator-note.txt'
    orphan.write_bytes(b'orphan')
    queued.write_bytes(b'queued')
    unrelated.write_bytes(b'preserve operator file')
    store = Store(tmp_path / 'teacher.sqlite3')
    job = store.enqueue('alice', 'physics', 'notes.txt', queued, limit=5)
    # Delay claim so this test isolates startup sweeping from job processing.
    with store.db() as db:
        db.execute('UPDATE jobs SET updated=updated+3600 WHERE id=?', (job['id'],))
    app = create_app(config(tmp_path), Rag(), Inference())
    async with app.router.lifespan_context(app):
        assert not orphan.exists()
        assert queued.is_file()
        assert unrelated.is_file()
