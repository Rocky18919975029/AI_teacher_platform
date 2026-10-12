import json
import time

import httpx
import pytest

from teacher.config import Settings
from teacher.inference import Inference
from teacher.storage import Store


def config(tmp_path):
    return Settings(
        _env_file=None, data_dir=tmp_path, api_keys=[{'key': 'a' * 40, 'subject': 'test', 'courses': ['x']}]
    )


@pytest.mark.asyncio
async def test_actual_openai_stream_parser_and_payload(tmp_path):
    client = Inference(config(tmp_path))
    await client.client.aclose()

    def handler(request):
        body = json.loads(request.content)
        assert body['stream'] is True and body['max_tokens'] == 1024
        assert request.headers['authorization'] == 'Bearer EMPTY'
        return httpx.Response(
            200,
            text=': heartbeat\n\ndata: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: {"choices":[]}\n\ndata: [DONE]\n\n',
        )

    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert [t async for t in client.stream([{'role': 'user', 'content': 'hi'}])] == ['Hello']
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('body', ['data: not-json\n\n', 'data: {"choices":[]}\n\n'])
async def test_corrupt_or_truncated_upstream_is_not_success(tmp_path, body):
    client = Inference(config(tmp_path))
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body)))
    with pytest.raises(ValueError):
        _ = [t async for t in client.stream([])]
    await client.close()


def test_queue_recovery_backoff_retry_limit_and_capacity(tmp_path):
    store = Store(tmp_path / 'db.sqlite3')
    job = store.enqueue('admin', 'course', 'file.txt', tmp_path / 'upload', 1)
    with pytest.raises(OverflowError):
        store.enqueue('admin', 'course', 'second.txt', tmp_path / 'other', 1)
    assert store.claim_job()['id'] == job['id']
    store.finish_job(job['id'], error='transient', retry=True)
    assert store.job(job['id'])['status'] == 'pending'
    assert store.claim_job() is None
    with store.db() as db:
        db.execute('UPDATE jobs SET updated=?', (time.time() - 1,))
    store.claim_job()
    store.recover_jobs()
    assert store.job(job['id'])['status'] == 'pending'
    store.claim_job()
    store.finish_job(job['id'], error='transient', retry=True)
    assert store.job(job['id'])['status'] == 'failed'
    assert store.job(job['id'])['attempts'] == 3


def test_sessions_persist_across_reopen_and_turns_atomic(tmp_path):
    store = Store(tmp_path / 'db.sqlite3')
    sid = store.create_session('alice', 'math')
    store.append_turn(sid, 'question', 'answer')
    store.claim_session(sid, 'alice')
    reopened = Store(tmp_path / 'db.sqlite3')
    assert len(reopened.session(sid, 'alice')['messages']) == 2
    with pytest.raises(KeyError):
        reopened.session(sid, 'bob')
    reopened.recover_jobs()
    assert reopened.claim_session(sid, 'alice')
