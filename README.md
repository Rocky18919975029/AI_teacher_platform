# AI Teacher Platform

A locally hosted teaching assistant with course-grounded answers, streamed text,
traceable source passages and optional Piper speech synthesis. The original
Qwen/vLLM + FastAPI classroom prototype now includes persistent course retrieval,
access boundaries and operational controls.

The application uses **one central API process**, with an optional deployment
bundle for **40 independent RTX 4090 inference hosts**. Its default admission cap
is 40 combined active chats/searches. It does not claim high availability,
security certification or measured production-scale capacity. GPU inference,
real-course retrieval quality and deployment load still need to pass the
[acceptance checklist](docs/operations.md#release-acceptance) on your host.

## Capabilities

- **Retrieval:** instructor-uploaded PDF, DOCX, UTF-8 TXT and Markdown; bounded
  extraction, token-aware overlapping chunks, Sentence Transformers embeddings,
  persistent Qdrant cosine search, CrossEncoder reranking and bounded context.
- **Evidence:** server-enforced course filtering; document IDs, titles and PDF pages
  accompany retrieved passages. Answers use `[S1]` references; unknown or missing
  citation IDs produce warnings. Empty retrieval returns insufficient evidence.
- **Inference:** asynchronous OpenAI-compatible vLLM requests and real SSE token
  streaming, response timeouts, bounded simultaneous chats and session serialization.
- **Classroom concurrency:** retrieval microbatches up to eight questions with an
  8 ms collection window, a dedicated bounded worker, course-isolated vector
  queries, and a separately bounded two-job optional speech stage.
- **State:** SQLite WAL sessions scoped to user and course, a durable upload queue
  with bounded retries, idempotent document versions and source deletion.
- **Access and operations:** scoped instructor/student keys, rate limits,
  authenticated expiring audio, admin-only Prometheus metrics, health probes,
  request IDs and a same-origin browser client without CDN scripts.
- **Speech:** optional Piper subprocess synthesis with a timeout. Voice model and
  CUDA runtime are deployment settings; the default container runs text/RAG only.

Retrieved content is untrusted reference data. Citation checks detect invalid
references, not semantic correctness. Instructors must evaluate course answers.
No model tool execution or autonomous external actions are enabled.

## Technology

| Layer | Implementation |
| --- | --- |
| Serving | Python 3.12, FastAPI, Pydantic Settings, Uvicorn, HTTPX, SSE |
| Inference | Qwen2.5-7B-Instruct/vLLM; optional 40 replicas behind Nginx least-connections routing |
| Retrieval | Sentence Transformers, BGE embeddings, CrossEncoder, Qdrant |
| Documents | pypdf, python-docx; text-based PDFs, no built-in OCR |
| Persistence | SQLite WAL and Qdrant persistent storage |
| Browser | Same-origin JavaScript, incremental streaming, native audio |
| Delivery | uv lockfile, non-root Docker image, Compose, GitHub Actions |
| Observability | Prometheus counters/histograms, JSON request logs, health probes |

## Run locally

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Start a separately managed vLLM endpoint and Qdrant server; weights, CUDA drivers
and model licenses are not bundled here.

```sh
uv sync --locked --extra rag
uv run python scripts/configure.py
```

The setup script creates a private `.env` with independently generated keys and
refuses to overwrite an existing file. Edit its service URLs and course mappings;
give each person a stable, unique `subject` and only their authorized `courses`.
`VLLM_URL` is the full chat-completions endpoint. Configure the generated Qdrant and
inference keys on those services too. `.env.example` lists all settings.

```sh
uv run uvicorn app:app --host 127.0.0.1 --port 9000 --workers 1 --no-access-log
```

Open [http://127.0.0.1:9000/ui/](http://127.0.0.1:9000/ui/), supply your key and course.
FastAPI serves the interface; no separate frontend server is required. Browser keys
remain in memory, not URLs or local storage.

The first RAG request loads the embedding/reranking models. Pin their revision
hashes and warm them before admitting students. English defaults need replacement
and evaluation for predominantly Chinese or multilingual courses. Downloads need
internet initially; local inference can operate after weights are provisioned.

## Container deployment

Set `.env` service URLs, including a `VLLM_URL` reachable **from the container**.
Container `127.0.0.1` is not the inference host. Then:

```sh
docker compose up --build -d
curl --fail http://127.0.0.1:9000/health/live
curl --fail http://127.0.0.1:9000/health/ready
```

Compose starts the API and authenticated Qdrant. Qdrant has no published port; the
API binds to loopback. Put a TLS reverse proxy in front for remote users. Data and
model caches use named volumes. One API worker is required by local job coordination
and process-local rate/admission limits. See [operations](docs/operations.md).

For the 40-machine deployment, use the [cluster runbook](docs/cluster40.md),
`deploy/compose.inference-node.yaml` on each GPU host and the generated central
gateway configuration. It includes fleet-wide health probing and a load-test
procedure; no GPU deployment is performed by generating configuration.

## Upload and evaluate

Set `TEACHER_API_KEY` through your secret manager or a hidden shell prompt. Use an
instructor key for upload and a course-scoped key for retrieval evaluation.

```sh
uv run python scripts/ingest.py course-notes.pdf --course metaverse
uv run python scripts/evaluate_retrieval.py labeled-questions.jsonl --k 5 --minimum-recall 0.8
```

The upload command waits for a durable job and returns a document ID. Label a JSONL
evaluation set with IDs from your deployment:

```json
{"course_id":"metaverse","query":"How are virtual classroom interactions represented?","relevant_document_ids":["REPLACE_WITH_INGESTED_DOCUMENT_UUID"]}
```

Evaluation measures document recall, first-relevant reciprocal rank, hit rate and
request latency over the first `k` returned chunks. `--minimum-recall` is a chosen
gate, not a claimed score. Set server `RAG_TOP_K >= k`; there is no per-query override.
This does not measure answer faithfulness or establish an SLA. See [API contracts](docs/api.md).

## Development and validation

```sh
uv sync --locked
uv run ruff check .
uv run python -m pytest -q
uv run pip-audit --strict --progress-spinner off
```

Tests use deterministic inference/embedding doubles and isolated stores, including
local Qdrant integration. They do not require model downloads or a GPU. Install
`--extra rag` for real model execution. CI runs lint/tests, dependency auditing and
a container build. Real GPU/model and target-host load acceptance remains separate.
The [validation record](docs/validation.md) separates completed checks from pending
deployment acceptance.

For an explicit real embedding/reranker smoke test (downloads weights if uncached):

```sh
RUN_MODEL_SMOKE=1 uv run --locked --extra rag python -m pytest -q tests/test_rag.py -m integration
```

This checks actual model retrieval, course scoping and local Qdrant persistence;
it is not a remote Qdrant/vLLM integration or a course-quality benchmark.

`teacher/` contains configuration, API, state, inference and retrieval modules;
`frontend/` is the browser client; `scripts/` provides operational tools; `docs/`
contains API and deployment instructions.

The `/ai-teacher` JSON route remains but requires authentication and a course ID.
Use `rag: false` for general conversation. Shared demo history is not migrated and
audio is no longer public. Review [migration notes](docs/operations.md#migration-and-rollbacks).
