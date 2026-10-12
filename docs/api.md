# API contracts

All `/api/*`, `/ai-teacher` and `/audio/*` routes require
`Authorization: Bearer <key>`. `/metrics` requires an admin key. Health and frontend
assets are public. The machine-readable schema is `/openapi.json`. Never pass keys
in URLs. Course IDs contain letters, digits, `_` or `-`, start with an alphanumeric
character and have at most 64 characters. Admin role does not bypass course scopes.

## Chat

`POST /api/chat/stream` accepts:

```json
{"course_id":"metaverse","prompt":"Explain spatial presence using the course material.","rag":true,"with_audio":false,"session_id":null}
```

Omit `session_id` for a new conversation; reuse the returned ID to continue.
Client-provided system/assistant messages are not accepted. Prompts are limited to
8,000 characters and cannot be blank. The `text/event-stream` response has JSON data:

| Event | Payload | Meaning |
| --- | --- | --- |
| `meta` | `{"session_id":"..."}` | Save the conversation ID |
| `sources` | `{"sources":[...]}` | Render plain-text source details |
| `token` | `{"text":"..."}` | Append text incrementally |
| `done` | `text`, `session_id`, `sources`, `audio_url`, `warnings` | Successful completion |
| `error` | `{"message":"..."}` | Incomplete response; offer retry |

An HTTP 200 or closed connection is not success without `done`. Errors after the
stream starts use SSE events. There are no resumable event IDs; reconnect requires
an explicit request. Failed/cancelled partial turns are not stored as complete.
Concurrent replies in the same session return 409.

With `rag: true`, no source means an insufficient-evidence response and a retrieval
failure means an error. `rag: false` skips course grounding. Chat source IDs are
local citation labels (`S1`); document ID, title, page and passage provide provenance.
Citation warnings do not establish factual correctness. Speech failure preserves text.
`POST /ai-teacher` takes the same input and returns the buffered `done` JSON object.

## Documents and retrieval

| Method and path | Input | Access/result |
| --- | --- | --- |
| `POST /api/documents?course_id=...` | Multipart `file` | Scoped admin; 202 with job ID/status |
| `GET /api/jobs/{id}` | Job ID | Scoped admin; job state/result |
| `GET /api/documents?course_id=...` | Course | Course member; document metadata |
| `DELETE /api/documents/{id}?course_id=...` | Document UUID | Scoped admin; deletes that course's vectors |
| `POST /api/search` | `{"course_id":"...","query":"..."}` | Course member; `{"sources":[...]}` |

Jobs are `pending`, `running`, `completed` or `failed`. Successful results contain
`document_id`, `title`, `chunks`, `sha256` and `reused`. Identical bytes, filename,
course and chunk settings reuse a version. Changed content creates a new version;
delete superseded versions explicitly after checking the replacement. Original
upload files are removed on terminal states; retain authoritative files separately.

Search hits contain `id`, `document_id`, `title`, `page`, `text`, `score` and
`retrieval_score`. The threshold uses cosine similarity; final scores may be
cross-encoder scores, not probabilities. Search is dense retrieval with optional
reranking, not hybrid lexical search.

## Sessions and audio

`GET /api/sessions?course_id=...` lists your recent sessions.
`GET /api/sessions/{id}` returns your messages. `DELETE /api/sessions/{id}` removes
the session if no answer is running. Another principal cannot read your session,
even in the same course. Moving a session to another course returns 409.

Fetch `/audio/<id>.wav` with the same bearer header and play a browser Blob URL.
Direct unauthenticated media requests fail. Foreign/expired audio returns 404.
Default audio TTL is one hour; it does not delete conversation transcripts.

## Operations and errors

`GET /health/live` confirms process liveness; `/health/ready` checks Qdrant/vLLM
connectivity and returns 503 on failure. Readiness does not exercise generation or
prove lazy model weights are loaded. Warm up retrieval and chat before rollout.

Admin `/metrics` exposes `teacher_http_requests_total`,
`teacher_http_duration_seconds`, `teacher_chat_duration_seconds` and
`teacher_failures_total`. HTTP latency is measured to response initiation; chat
latency spans the event generator. Labels use route templates, not person/document
names. Scrape requests count against the monitoring key's rate limit.

401: invalid/missing credentials; 403: insufficient role/course; 404: unavailable
or foreign resource; 409: session conflict; 413: too much input; 415: unsupported
type; 422: invalid request; 429: rate cap; 503: saturation/dependency failure.
Honor `Retry-After` and use bounded backoff. Preserve `X-Request-ID` for support,
without logging prompts or credentials.
