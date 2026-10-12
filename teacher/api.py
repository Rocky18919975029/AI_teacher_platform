import asyncio
import hmac
import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from contextlib import aclosing, asynccontextmanager, suppress
from pathlib import Path
from typing import Literal

import anyio
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from teacher.config import Settings
from teacher.inference import Inference, build_messages
from teacher.storage import Store

LOG = logging.getLogger('teacher')
COURSE = r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$'
AUTH = HTTPBearer(auto_error=False)


class ClosingStreamResponse(StreamingResponse):
    """Explicitly close nested generators even on transport-send cancellation."""

    def __init__(self, content, *, release, **kwargs):
        super().__init__(content, **kwargs)
        self.release = release

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    self.release()


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    course_id: str = Field(default='default', pattern=COURSE)
    session_id: str | None = Field(default=None, max_length=36)
    with_audio: bool = False
    voice: Literal['teacher_female', 'teacher_male'] = 'teacher_female'
    rag: bool = True


class SearchRequest(BaseModel):
    course_id: str = Field(pattern=COURSE)
    query: str = Field(min_length=1, max_length=8000)


def public_job(job):
    return {k: v for k, v in job.items() if k not in ('owner', 'path')}


def create_app(settings=None, rag=None, inference=None):
    registry = CollectorRegistry()
    requests = Counter(
        'teacher_http_requests_total', 'HTTP requests', ['method', 'route', 'status'], registry=registry
    )
    duration = Histogram(
        'teacher_http_duration_seconds', 'HTTP response initiation latency', ['route'], registry=registry
    )
    chat_duration = Histogram('teacher_chat_duration_seconds', 'Complete chat latency', registry=registry)
    failures = Counter('teacher_failures_total', 'Dependency failures', ['component'], registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        cfg = settings or Settings()
        if not LOG.handlers:
            LOG.addHandler(logging.StreamHandler())
        LOG.setLevel(logging.INFO)
        LOG.propagate = False
        app.state.cfg = cfg
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        for directory in ('uploads', 'audio'):
            (cfg.data_dir / directory).mkdir(exist_ok=True)
        app.state.store = Store(cfg.data_dir / 'teacher.sqlite3')
        if rag is None:
            from teacher.rag import RagService

            app.state.rag = RagService(cfg)
        else:
            app.state.rag = rag
        app.state.inference = inference or Inference(cfg)
        app.state.capacity = asyncio.Semaphore(cfg.max_concurrent_chats)
        app.state.rate = defaultdict(deque)
        app.state.store.recover_jobs()
        active_uploads = app.state.store.active_uploads()
        for upload in (cfg.data_dir / 'uploads').iterdir():
            if re.fullmatch(r'[0-9a-f]{32}', upload.name) and str(upload) not in active_uploads:
                upload.unlink(missing_ok=True)

        async def process_job(job):
            try:
                result = await run_in_threadpool(
                    app.state.rag.ingest, job['course'], job['filename'], Path(job['path']).read_bytes()
                )
                app.state.store.finish_job(job['id'], result=result)
            except Exception as exc:
                failures.labels('ingestion').inc()
                LOG.warning('ingestion_failed job=%s type=%s', job['id'], type(exc).__name__)
                app.state.store.finish_job(
                    job['id'],
                    error='Ingestion failed; check file format and service health, then retry upload.',
                    retry=not isinstance(exc, ValueError),
                )
            finally:
                if app.state.store.job(job['id'])['status'] in ('completed', 'failed'):
                    Path(job['path']).unlink(missing_ok=True)

        async def jobs():
            while True:
                job = app.state.store.claim_job()
                if job:
                    active = asyncio.create_task(process_job(job))
                    try:
                        await asyncio.shield(active)
                    except asyncio.CancelledError:
                        # A parser/model thread cannot be killed safely. Drain before
                        # closing Qdrant; a forced process kill recovers on restart.
                        await active
                        raise
                else:
                    for aid in app.state.store.expire_audio(time.time() - cfg.audio_ttl_seconds):
                        (cfg.data_dir / 'audio' / f'{aid}.wav').unlink(missing_ok=True)
                    await asyncio.sleep(1)

        worker = asyncio.create_task(jobs())
        try:
            yield
        finally:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
            await app.state.inference.close()
            await run_in_threadpool(app.state.rag.close)

    app = FastAPI(title='AI Teacher Platform', version='2.0.0', lifespan=lifespan)

    @app.middleware('http')
    async def observe(request, call_next):
        rid = uuid.uuid4().hex
        request.state.request_id = rid
        start = time.monotonic()
        response = await call_next(request)
        route = getattr(request.scope.get('route'), 'path', 'unmatched')
        method = (
            request.method
            if request.method in {'GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH'}
            else 'OTHER'
        )
        requests.labels(method, route, str(response.status_code)).inc()
        duration.labels(route).observe(time.monotonic() - start)
        response.headers.update(
            {
                'X-Request-ID': rid,
                'X-Content-Type-Options': 'nosniff',
                'Referrer-Policy': 'no-referrer',
                'X-Frame-Options': 'DENY',
                'Cache-Control': 'no-store',
                'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'",
            }
        )
        LOG.info(
            json.dumps(
                {
                    'request_id': rid,
                    'method': request.method,
                    'route': route,
                    'status': response.status_code,
                    'duration_ms': round((time.monotonic() - start) * 1000),
                }
            )
        )
        return response

    class BodyLimit:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope['type'] != 'http':
                return await self.app(scope, receive, send)
            cfg = scope['app'].state.cfg
            limit = cfg.max_upload_bytes + 65536 if scope['path'] == '/api/documents' else 65536
            size = 0
            chunks = deque()
            try:
                async with asyncio.timeout(30):
                    while True:
                        message = await receive()
                        if message['type'] == 'http.disconnect':
                            return
                        size += len(message.get('body', b''))
                        if size > limit:
                            return await JSONResponse({'detail': 'Request body too large'}, status_code=413)(
                                scope, receive, send
                            )
                        chunks.append(message)
                        if not message.get('more_body', False):
                            break
            except TimeoutError:
                return await JSONResponse({'detail': 'Request body timed out'}, status_code=408)(
                    scope, receive, send
                )

            async def bounded_receive():
                return chunks.popleft() if chunks else await receive()

            await self.app(scope, bounded_receive, send)

    app.add_middleware(BodyLimit)

    async def principal(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(AUTH)):
        token = credentials.credentials if credentials else ''
        matched = None
        for candidate in request.app.state.cfg.api_keys:
            if hmac.compare_digest(token.encode(), candidate.key.get_secret_value().encode()):
                matched = candidate
        if matched is None:
            raise HTTPException(401, 'Valid bearer token required', headers={'WWW-Authenticate': 'Bearer'})
        window = request.app.state.rate[matched.subject]
        now = time.monotonic()
        while window and window[0] < now - 60:
            window.popleft()
        if len(window) >= request.app.state.cfg.requests_per_minute:
            raise HTTPException(429, 'Rate limit reached', headers={'Retry-After': '60'})
        window.append(now)
        return matched

    def authorize(user, course):
        if course not in user.courses and '*' not in user.courses:
            raise HTTPException(403, 'Course access denied')

    def admin(user):
        if user.role != 'admin':
            raise HTTPException(403, 'Administrator role required')

    @app.get('/health/live')
    @app.get('/')
    async def live():
        return {'status': 'ok', 'version': '2.0.0'}

    @app.get('/health/ready')
    async def ready():
        async def rag_health():
            try:
                return await run_in_threadpool(app.state.rag.health)
            except Exception:
                return False

        vector_ok, llm_ok = await asyncio.gather(rag_health(), app.state.inference.health())
        return JSONResponse(
            {'ready': vector_ok and llm_ok, 'qdrant': vector_ok, 'llm': llm_ok},
            status_code=200 if vector_ok and llm_ok else 503,
        )

    @app.get('/metrics')
    async def metrics(user=Depends(principal)):
        admin(user)
        return Response(generate_latest(registry), media_type='text/plain; version=0.0.4')

    async def prepare(req, user):
        authorize(user, req.course_id)
        if not req.prompt.strip():
            raise HTTPException(422, 'Prompt cannot be blank')
        store = app.state.store
        sid = req.session_id or store.create_session(user.subject, req.course_id)
        try:
            session = store.session(sid, user.subject)
        except KeyError:
            raise HTTPException(404, 'Session not found') from None
        if session['course'] != req.course_id:
            raise HTTPException(409, 'Session belongs to a different course')
        if not store.claim_session(sid, user.subject):
            raise HTTPException(409, 'A reply is already running in this session')
        try:
            await asyncio.wait_for(app.state.capacity.acquire(), timeout=0.1)
        except TimeoutError:
            store.release_session(sid)
            raise HTTPException(
                503, 'Chat capacity reached; retry shortly', headers={'Retry-After': '2'}
            ) from None
        except BaseException:
            store.release_session(sid)
            raise
        return sid, session

    def release_chat(sid, session):
        if not session.get('_released'):
            session['_released'] = True
            app.state.store.release_session(sid)
            app.state.capacity.release()

    async def events(req, user, sid, session):
        started = time.monotonic()
        store = app.state.store
        cfg = app.state.cfg
        sources = []
        warnings = []
        try:
            yield 'meta', {'session_id': sid}
            if req.rag:
                try:
                    sources = await run_in_threadpool(app.state.rag.search, req.course_id, req.prompt)
                    sources = [dict(s, id=f'S{i}') for i, s in enumerate(sources, 1)]
                except Exception:
                    failures.labels('retrieval').inc()
                    yield 'error', {'message': 'Course retrieval is unavailable; please retry.'}
                    return
            yield 'sources', {'sources': sources}
            answer = ''
            if req.rag and not sources:
                answer = 'I could not find supporting material in this course. Please upload relevant materials or switch off course grounding for a general explanation.'
                yield 'token', {'text': answer}
            else:
                messages = build_messages(req.prompt, session['messages'], sources, cfg.max_history_chars)
                async with asyncio.timeout(cfg.llm_timeout):
                    async with aclosing(app.state.inference.stream(messages)) as tokens:
                        async for token in tokens:
                            answer += token
                            if len(answer) > 32000:
                                raise ValueError('Response too long')
                            yield 'token', {'text': token}
                if not answer.strip():
                    raise ValueError('Empty model response')
            cited = set(re.findall(r'\[(S\d+)\]', answer))
            available = {s['id'] for s in sources}
            if cited - available:
                warnings.append(
                    'The model emitted unsupported citation IDs; verify against the listed sources.'
                )
            if sources and not cited:
                warnings.append(
                    'The model did not cite its answer; retrieved passages are provided for verification.'
                )
            audio_url = None
            if req.with_audio:
                try:
                    aid = await app.state.inference.synthesize(answer, cfg.data_dir / 'audio')
                    if aid:
                        store.register_audio(aid, user.subject, req.course_id)
                        audio_url = f'/audio/{aid}.wav'
                    else:
                        warnings.append('Speech synthesis is disabled on this server.')
                except Exception:
                    failures.labels('tts').inc()
                    warnings.append('Speech synthesis is unavailable; the text answer was preserved.')
            store.append_turn(
                sid, req.prompt, answer, {'sources': sources, 'audio_url': audio_url, 'warnings': warnings}
            )
            yield (
                'done',
                {
                    'text': answer,
                    'session_id': sid,
                    'sources': sources,
                    'audio_url': audio_url,
                    'warnings': warnings,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failures.labels('generation').inc()
            LOG.warning('generation_failed type=%s', type(exc).__name__)
            yield 'error', {'message': 'Answer generation failed. Please retry.'}
        finally:
            release_chat(sid, session)
            chat_duration.observe(time.monotonic() - started)

    @app.post('/api/chat/stream')
    async def chat_stream(req: ChatRequest, user=Depends(principal)):
        sid, session = await prepare(req, user)

        async def stream():
            async with aclosing(events(req, user, sid, session)) as generator:
                async for event, data in generator:
                    yield f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'

        return ClosingStreamResponse(
            stream(),
            release=lambda: release_chat(sid, session),
            media_type='text/event-stream',
            headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'},
        )

    @app.post('/ai-teacher')
    async def compatibility_chat(req: ChatRequest, user=Depends(principal)):
        sid, session = await prepare(req, user)
        result = None
        error = None
        async for event, data in events(req, user, sid, session):
            if event == 'error':
                error = data['message']
            if event == 'done':
                result = data
        if error:
            raise HTTPException(502, error)
        return result

    @app.get('/api/sessions')
    async def sessions(course_id: str = Query(pattern=COURSE), user=Depends(principal)):
        authorize(user, course_id)
        return app.state.store.sessions(user.subject, course_id)

    @app.get('/api/sessions/{sid}')
    async def session(sid: str, user=Depends(principal)):
        try:
            result = app.state.store.session(sid, user.subject)
        except KeyError:
            raise HTTPException(404, 'Session not found') from None
        authorize(user, result['course'])
        return {k: v for k, v in result.items() if k not in ('owner', 'busy')}

    @app.delete('/api/sessions/{sid}')
    async def delete_session(sid: str, user=Depends(principal)):
        result = await session(sid, user)
        if not app.state.store.delete_session(result['id'], user.subject):
            raise HTTPException(409, 'Session has an active reply')
        return {'deleted': True}

    @app.post('/api/documents', status_code=202)
    async def upload(
        course_id: str = Query(pattern=COURSE), file: UploadFile = File(), user=Depends(principal)
    ):
        admin(user)
        authorize(user, course_id)
        name = Path(file.filename or 'document.txt').name[:200]
        if Path(name).suffix.lower() not in {'.pdf', '.docx', '.txt', '.md'}:
            raise HTTPException(415, 'Supported formats: PDF, DOCX, TXT, MD')
        path = app.state.cfg.data_dir / 'uploads' / uuid.uuid4().hex
        size = 0
        try:
            with path.open('xb') as target:
                while chunk := await file.read(65536):
                    size += len(chunk)
                    if size > app.state.cfg.max_upload_bytes:
                        raise HTTPException(413, 'File too large')
                    target.write(chunk)
            if size == 0:
                raise HTTPException(422, 'Empty file')
            job = app.state.store.enqueue(user.subject, course_id, name, path, app.state.cfg.max_pending_jobs)
        except OverflowError:
            path.unlink(missing_ok=True)
            raise HTTPException(503, 'Ingestion queue is full', headers={'Retry-After': '30'}) from None
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        return public_job(job)

    @app.get('/api/jobs/{jid}')
    async def job(jid: str, user=Depends(principal)):
        admin(user)
        try:
            result = app.state.store.job(jid)
        except KeyError:
            raise HTTPException(404, 'Job not found') from None
        authorize(user, result['course'])
        return public_job(result)

    @app.get('/api/documents')
    async def documents(course_id: str = Query(pattern=COURSE), user=Depends(principal)):
        authorize(user, course_id)
        try:
            return await run_in_threadpool(app.state.rag.documents, course_id)
        except Exception:
            raise HTTPException(503, 'Document store unavailable') from None

    @app.post('/api/search')
    async def search(req: SearchRequest, user=Depends(principal)):
        authorize(user, req.course_id)
        if not req.query.strip():
            raise HTTPException(422, 'Query cannot be blank')
        try:
            await asyncio.wait_for(app.state.capacity.acquire(), timeout=0.1)
        except TimeoutError:
            raise HTTPException(503, 'Retrieval capacity reached', headers={'Retry-After': '2'}) from None
        try:
            sources = await run_in_threadpool(app.state.rag.search, req.course_id, req.query)
            return {'sources': sources}
        except Exception:
            raise HTTPException(503, 'Retrieval unavailable') from None
        finally:
            app.state.capacity.release()

    @app.delete('/api/documents/{document_id}')
    async def delete_document(
        document_id: uuid.UUID, course_id: str = Query(pattern=COURSE), user=Depends(principal)
    ):
        admin(user)
        authorize(user, course_id)
        try:
            await run_in_threadpool(app.state.rag.delete, course_id, str(document_id))
        except Exception:
            raise HTTPException(503, 'Document store unavailable') from None
        return {'deleted': True}

    @app.get('/audio/{filename}')
    async def audio(filename: str, user=Depends(principal)):
        if not re.fullmatch(r'[0-9a-f]{32}\.wav', filename):
            raise HTTPException(404, 'Audio not found')
        entry = app.state.store.get_audio(filename[:-4], user.subject, app.state.cfg.audio_ttl_seconds)
        if not entry:
            raise HTTPException(404, 'Audio not found')
        authorize(user, entry['course'])
        path = app.state.cfg.data_dir / 'audio' / filename
        if not path.is_file():
            raise HTTPException(404, 'Audio expired')
        return FileResponse(path, media_type='audio/wav')

    app.mount(
        '/ui',
        StaticFiles(directory=Path(__file__).resolve().parent.parent / 'frontend', html=True),
        name='ui',
    )
    return app
