import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from teacher.api import create_app
from teacher.config import Settings
from teacher.inference import build_messages

ADMIN = 'a' * 40
STUDENT = 'b' * 40
OTHER = 'c' * 40


class FakeRag:
    def __init__(self):
        self.sources = [
            {
                'id': 'chunk',
                'document_id': 'doc',
                'title': 'Lesson',
                'page': 1,
                'text': 'Gravity attracts masses.',
                'score': 0.9,
            }
        ]
        self.calls = []

    def search(self, course, query):
        self.calls.append((course, query))
        return self.sources

    def ingest(self, course, filename, data):
        return {'id': 'doc', 'course_id': course, 'title': filename, 'chunks': 1}

    def documents(self, course):
        return []

    def delete(self, course, document_id):
        pass

    def health(self):
        return True

    def close(self):
        pass


class FakeInference:
    def __init__(self):
        self.calls = []
        self.fail = False

    async def stream(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError('private upstream secret')
        for token in ('Gravity ', 'attracts masses. [S1]'):
            yield token
            await asyncio.sleep(0)

    async def health(self):
        return True

    async def synthesize(self, text, audio_dir):
        raise RuntimeError('TTS unavailable')

    async def close(self):
        pass


@pytest.fixture
def setup(tmp_path):
    cfg = Settings(
        _env_file=None,
        data_dir=tmp_path,
        api_keys=[
            {'key': ADMIN, 'subject': 'admin', 'role': 'admin', 'courses': ['physics']},
            {'key': STUDENT, 'subject': 'alice', 'courses': ['physics']},
            {'key': OTHER, 'subject': 'bob', 'courses': ['chemistry']},
        ],
    )
    rag, llm = FakeRag(), FakeInference()
    app = create_app(cfg, rag, llm)
    with TestClient(app) as client:
        yield client, app, rag, llm


def auth(token=STUDENT):
    return {'Authorization': f'Bearer {token}'}


def chat(client, **kwargs):
    return client.post(
        '/ai-teacher', headers=auth(), json={'prompt': 'What is gravity?', 'course_id': 'physics', **kwargs}
    )


def test_auth_course_scope_and_roles(setup):
    client, _, rag, _ = setup
    assert client.post('/ai-teacher', json={'prompt': 'hello'}).status_code == 401
    assert (
        client.post(
            '/api/search', headers=auth(OTHER), json={'course_id': 'physics', 'query': 'gravity'}
        ).status_code
        == 403
    )
    assert not rag.calls
    assert (
        client.post(
            '/api/documents?course_id=physics', headers=auth(), files={'file': ('x.txt', b'text')}
        ).status_code
        == 403
    )
    assert client.get('/metrics', headers=auth()).status_code == 403
    assert client.get('/metrics', headers=auth(ADMIN)).status_code == 200


def test_grounded_chat_and_isolated_history(setup):
    client, _, rag, llm = setup
    result = chat(client).json()
    assert result['sources'][0]['id'] == 'S1'
    assert rag.calls == [('physics', 'What is gravity?')]
    assert 'Untrusted course reference' in llm.calls[0][-1]['content']
    sid = result['session_id']
    assert client.get(f'/api/sessions/{sid}', headers=auth(OTHER)).status_code == 404
    assert len(client.get(f'/api/sessions/{sid}', headers=auth()).json()['messages']) == 2
    saved = client.get(f'/api/sessions/{sid}', headers=auth()).json()
    assert saved['title'] == 'What is gravity?'
    assert saved['messages'][-1]['sources'] == result['sources']
    assert chat(client, session_id=sid).status_code == 200
    assert len(llm.calls[-1]) == 4
    assert client.delete(f'/api/sessions/{sid}', headers=auth()).status_code == 200
    assert client.get(f'/api/sessions/{sid}', headers=auth()).status_code == 404


def test_empty_retrieval_abstains_without_llm(setup):
    client, _, rag, llm = setup
    rag.sources = []
    result = chat(client).json()
    assert 'could not find supporting' in result['text']
    assert not llm.calls


def test_upstream_failure_no_secret_no_partial_history_releases_capacity(setup):
    client, app, _, llm = setup
    llm.fail = True
    result = chat(client)
    assert result.status_code == 502
    assert 'secret' not in result.text
    sid = client.get('/api/sessions?course_id=physics', headers=auth()).json()[0]['id']
    assert not app.state.store.session(sid, 'alice')['messages']
    llm.fail = False
    assert chat(client, session_id=sid).status_code == 200


def test_sse_events_and_audio_degradation(setup):
    client, _, _, _ = setup
    response = client.post(
        '/api/chat/stream',
        headers=auth(),
        json={'prompt': 'Gravity?', 'course_id': 'physics', 'with_audio': True},
    )
    assert response.status_code == 200
    events = [block for block in response.text.strip().split('\n\n')]
    assert events[0].startswith('event: meta')
    assert any(e.startswith('event: token') for e in events)
    done = json.loads(events[-1].split('data: ', 1)[1])
    assert done['warnings'] and done['audio_url'] is None


def test_ingestion_job_durable_and_private_paths_hidden(setup):
    client, app, _, _ = setup
    response = client.post(
        '/api/documents?course_id=physics',
        headers=auth(ADMIN),
        files={'file': ('notes.txt', b'Gravity attracts masses.')},
    )
    assert response.status_code == 202
    job = response.json()
    assert 'path' not in job and 'owner' not in job
    for _ in range(30):
        job = client.get(f'/api/jobs/{job["id"]}', headers=auth(ADMIN)).json()
        if job['status'] == 'completed':
            break
        time.sleep(0.05)
    assert job['status'] == 'completed'
    assert list((app.state.cfg.data_dir / 'uploads').iterdir()) == []


def test_validation_rate_and_limits(setup):
    client, app, _, _ = setup
    assert chat(client, prompt=' ').status_code == 422
    assert chat(client, prompt='x' * 8001).status_code == 422
    response = client.post(
        '/api/documents?course_id=physics', headers=auth(ADMIN), files={'file': ('bad.exe', b'x')}
    )
    assert response.status_code == 415
    response = client.post(
        '/ai-teacher', headers={**auth(), 'Content-Type': 'application/json'}, content=b'x' * 70000
    )
    assert response.status_code == 413
    app.state.cfg.requests_per_minute = 1
    assert client.get('/api/sessions?course_id=physics', headers=auth()).status_code == 429


def test_busy_session_and_audio_ownership(setup):
    client, app, _, _ = setup
    sid = app.state.store.create_session('alice', 'physics')
    app.state.store.claim_session(sid, 'alice')
    assert chat(client, session_id=sid).status_code == 409
    assert client.delete(f'/api/sessions/{sid}', headers=auth()).status_code == 409
    aid = 'd' * 32
    (app.state.cfg.data_dir / 'audio' / f'{aid}.wav').write_bytes(b'RIFF')
    app.state.store.register_audio(aid, 'alice', 'physics')
    assert client.get(f'/audio/{aid}.wav', headers=auth(OTHER)).status_code == 404
    assert client.get(f'/audio/{aid}.wav', headers=auth()).status_code == 200


def test_config_fail_closed_and_history_budget():
    with pytest.raises(ValueError):
        Settings(_env_file=None, api_keys=[])
    with pytest.raises(ValueError):
        Settings(_env_file=None, api_keys=[{'key': 'short', 'subject': 'a', 'courses': ['x']}])
    messages = build_messages('question', [{'role': 'user', 'content': 'x' * 100}], [], 10)
    assert len(messages) == 2
