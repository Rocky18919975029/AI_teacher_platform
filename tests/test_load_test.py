"""Verify the real-service acceptance tool cannot silently count failed SSE as success."""

import asyncio
import json

import httpx
import pytest

from scripts.load_test import Student, load_students, main, perform, run_load

SOURCES = [{'id': 'S1', 'document_id': 'doc'}]
STUDENT = Student('k' * 40, 'physics')


def event(name, payload):
    return f'event: {name}\ndata: {json.dumps(payload)}\n\n'.encode()


def good_stream(sid='session-1', sources=None):
    if sources is None:
        sources = SOURCES
    return [
        event('meta', {'session_id': sid}),
        event('sources', {'sources': sources}),
        event('token', {'text': 'Answer [S1].'}),
        event('done', {'session_id': sid, 'text': 'Answer [S1].', 'sources': sources}),
    ]


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0):
        self.chunks = chunks
        self.delay = delay
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self):
        self.closed = True


def response(chunks, delay=0):
    return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Chunks(chunks, delay))


@pytest.mark.parametrize(
    'chunks,expected',
    [
        ([event('error', {'message': 'private upstream detail'})], 'sse_error'),
        (good_stream()[:-1], 'missing_done'),
        ([b'event: token\ndata: {"text": "partial"}'], 'truncated_event'),
        ([b'event: token\ndata: {invalid}\n\n'], 'invalid_event_json'),
        (good_stream(sources=[]), 'ungrounded_fallback'),
        (good_stream() + [event('token', {'text': 'late'})], 'event_after_done'),
        ([event('done', {'text': 'No stream tokens', 'sources': SOURCES})], 'missing_or_inconsistent_answer'),
    ],
)
async def test_incomplete_or_failed_streams_never_pass(chunks, expected):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response(chunks)), base_url='http://test'
    ) as client:
        sample, _ = await perform(client, STUDENT, 'private prompt', 'chat', timeout=1, require_sources=False)
    assert not sample.success
    assert sample.error == expected
    assert 'private' not in json.dumps(sample.__dict__)


async def test_overload_recorded_and_no_response_body_logged():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503, json={'detail': 'private'})),
        base_url='http://test',
    ) as client:
        result = await run_load(client, [STUDENT], ['secret prompt'], rounds=1, warmup_rounds=0)
    assert not result['passed']
    assert result['measured']['errors'] == {'http_503': 1}
    assert result['measured']['ttft_seconds']['samples'] == 0
    assert 'private' not in json.dumps(result)
    assert 'secret' not in json.dumps(result)


async def test_incremental_sse_ttft_precedes_completion_and_transport_closes():
    upstream = response(good_stream(), delay=0.01)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: upstream), base_url='http://test'
    ) as client:
        sample, sid = await perform(client, STUDENT, 'question', 'chat', timeout=1, require_sources=True)
    assert sample.success
    assert sid == 'session-1'
    assert 0 < sample.ttft_seconds < sample.total_seconds
    assert sample.source_count == 1
    assert upstream.is_closed


async def test_40_synchronized_users_warmup_separate_and_sessions_reused():
    active, peak, calls = 0, 0, []
    barrier = asyncio.Barrier(40)

    async def serve(req):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        payload = json.loads(req.content)
        calls.append((req.headers['authorization'], payload, req.url.path))
        await asyncio.wait_for(barrier.wait(), timeout=2)
        await asyncio.sleep(0.001)
        active -= 1
        if req.url.path == '/api/search':
            return httpx.Response(200, json={'sources': SOURCES})
        return response(good_stream(sid=payload['course_id']))

    students = [Student(f'{i:02d}' + 'k' * 38, f'course-{i}') for i in range(40)]
    async with httpx.AsyncClient(transport=httpx.MockTransport(serve), base_url='http://test') as client:
        result = await run_load(
            client,
            students,
            ['question'],
            mode='mixed',
            rounds=2,
            warmup_rounds=1,
            require_sources=True,
            max_p95_ttft=2,
            max_p95_total=2,
        )
    assert peak == 40
    assert result['passed']
    assert result['warmup']['requests'] == 40
    assert result['measured']['requests'] == 80
    assert result['by_kind']['chat']['successes'] == result['by_kind']['search']['successes'] == 40
    assert result['measured']['ttft_seconds']['samples'] == 40
    assert len({auth for auth, _, _ in calls}) == 40
    assert all('session_id' not in data for _, data, _ in calls[:40])
    assert all(
        data.get('session_id') == data['course_id']
        for _, data, path in calls[40:]
        if path == '/api/chat/stream'
    )
    rendered = json.dumps(result)
    assert 'Bearer' not in rendered and 'question' not in rendered and 'course-0' not in rendered


async def test_failed_warmup_fails_even_if_measured_requests_pass():
    calls = 0

    def serve(req):
        nonlocal calls
        calls += 1
        return httpx.Response(503) if calls == 1 else response(good_stream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve), base_url='http://test') as client:
        result = await run_load(client, [STUDENT], ['question'], rounds=1, warmup_rounds=1)
    assert result['measured']['successes'] == 1
    assert result['warmup']['failures'] == 1
    assert result['violations'] == ['warmup_failed']
    assert not result['passed']


async def test_explicit_latency_gate_and_timeout_fail_acceptance():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response(good_stream(), delay=0.01)), base_url='http://test'
    ) as client:
        result = await run_load(
            client,
            [STUDENT],
            ['question'],
            rounds=1,
            warmup_rounds=0,
            max_p95_ttft=0.001,
            max_p95_total=0.001,
        )
        sample, _ = await perform(client, STUDENT, 'question', 'chat', timeout=0.001, require_sources=True)
    assert result['violations'] == [
        'chat_p95_ttft_exceeded_or_unavailable',
        'chat_p95_total_exceeded_or_unavailable',
    ]
    assert not sample.success
    assert sample.error == 'timeout'


async def test_search_source_requirement():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={'sources': []})),
        base_url='http://test',
    ) as client:
        result = await run_load(
            client, [STUDENT], ['question'], mode='search', rounds=1, warmup_rounds=0, require_sources=True
        )
    assert not result['passed']
    assert result['measured']['errors'] == {'no_sources': 1}
    assert result['measured']['ttft_seconds']['p95'] is None


def test_private_distinct_key_inventory_and_shared_identity(tmp_path, monkeypatch):
    path = tmp_path / 'users.json'
    path.write_text(json.dumps([{'key': 'a' * 40, 'course_id': 'physics'}, {'key': 'b' * 40}]))
    path.chmod(0o644)
    with pytest.raises(ValueError, match='private'):
        load_students(path, 2, 'physics')
    path.chmod(0o600)
    students, identity = load_students(path, 2, 'physics')
    assert len(students) == 2
    assert students[1].course_id == 'physics'
    assert identity.startswith('distinct_keys')
    monkeypatch.setenv('TEACHER_API_KEY', 'k' * 40)
    students, identity = load_students(None, 40, 'physics')
    assert len(students) == 40
    assert identity.startswith('shared_key')
    path.write_text(json.dumps([{'key': 'a' * 40}, {'key': 'a' * 40}]))
    with pytest.raises(ValueError, match='distinct'):
        load_students(path, 2, 'physics')


@pytest.mark.parametrize(
    'extra',
    [['--users', '0'], ['--rounds', '0'], ['--timeout', 'nan'], ['--mode', 'search', '--max-p95-ttft', '2']],
)
def test_cli_invalid_workloads_rejected(extra):
    with pytest.raises(SystemExit) as error:
        main(['--base-url', 'http://localhost:8001', *extra])
    assert error.value.code == 2
