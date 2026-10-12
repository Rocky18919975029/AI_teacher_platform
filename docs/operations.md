# Operations and acceptance

## Supported topology

Run **one FastAPI/Uvicorn process**, persistent Qdrant and separately provisioned
vLLM inference (one service or the [40-replica gateway](cluster40.md)). SQLite WAL stores sessions/jobs/audio metadata at
`DATA_DIR/teacher.sqlite3`; uploads/audio share that data directory. Qdrant stores
passage text, embeddings and document metadata. This release is not an HA cluster.

Compose runs the API as UID/GID 10001 with a read-only root filesystem, named data
and cache volumes, bounded temporary storage and resource limits. Qdrant has an
API key, internal network and no published host port. Provision vLLM's GPU, model
weights/license and context limit separately. Do not use additional API workers:
rate/admission limits and job coordination assume one process.

Embedding/reranking default to CPU; Linux installs use the official CPU PyTorch
index, keeping the API image separate from GPU vLLM serving. To use CUDA for RAG,
adapt the PyTorch source and image runtime, regenerate the lockfile and validate
the GPU installation before setting `RAG_DEVICE=cuda`. First use downloads models into the cache
volume; set `RAG_EMBEDDING_REVISION` and `RAG_RERANKER_REVISION` to trusted model
commit hashes and warm them before serving. Once cached, test offline restart with
`HF_HUB_OFFLINE=1`. Model changes alter the Qdrant collection suffix; reingest and
evaluate before switching. Old collections remain until explicitly retired.

## Secrets and identity

`scripts/configure.py` creates `.env` with fresh secrets and mode 0600, refusing to
overwrite an existing file. `.env.example` documents settings but its placeholders
must never be deployed. In production, inject keys through your secret store.
Docker/host administrators can inspect environment values and require privileged
trust. Service keys must match their vLLM/Qdrant configurations.

`API_KEYS` is a JSON array of `{key,subject,role,courses}`. Use unique subjects per
person, separate admin/student identities and explicit course scopes. For rotation,
replace a key keeping its subject/course mapping and restart the API in a
maintenance window. Remove a mapping and restart to revoke it. Shared subjects
share sessions. Institution SSO/OIDC and short-lived token issuance are not included.
Use TLS remotely; keep credentials out of URLs, logs, browser storage and analytics.

## Proxy and limits

Compose binds the API to loopback. Terminate HTTPS at your reverse proxy, preserve
`Authorization`, disable buffering for `/api/chat/stream`, set read timeouts above
generation/TTS limits, and cap request bodies just above `MAX_UPLOAD_BYTES` for
multipart overhead. Enforce connection and slow-client limits at the proxy. Do not
cache authenticated API/audio responses. Trust forwarded headers only from known
proxies. Serve `/ui/` and `/api/` on the same origin.

The browser renders responses as text with no runtime CDN script. Keep that property
when extending it; never insert model text as raw HTML. Tune concurrency, per-key
rate limits, queue caps and upload bounds against real hardware. The combined
system prompt, question, history, retrieval and output must fit the LLM context
window. Retrieval token counts use the embedding tokenizer, not necessarily the
generation tokenizer, so budget conservatively.

Default `MAX_CONCURRENT_CHATS=40` is the combined active chat/search admission cap.
The upstream connection pool scales with that cap. Retrieval uses
`RETRIEVAL_BATCH_SIZE=8`, `RETRIEVAL_BATCH_WAIT_MS=8` and `RETRIEVAL_TIMEOUT=60` with
a bounded dedicated worker, not the generic web thread pool. Query embeddings and
reranker pairs batch together while Qdrant filters and contexts remain per course.
Cancelled native model calls retain their capacity until the work actually returns.
Configured admission and simulated capacity are not measurements of GPU throughput.

The default container gives Uvicorn 240 seconds for graceful request drain and
Compose 270 seconds before termination. Retrieval (60 s), inference (90 s) and
optional speech (2 s queue plus 45 s execution) can add up. Increase both drain
windows for larger timeouts or long ingestion/model initialization; native parsing
or model calls cannot necessarily be preempted. Stop new work before maintenance.

## Documents and trust

Only instructors upload/delete materials. PDF/DOCX parsers have byte, character,
page and chunk limits; scanned PDFs need OCR upstream. Treat uploads as trusted
administrative input and isolate the container. This is not a hardened anonymous
public file-processing service.

The queue persists across restart. Interrupted jobs recover with bounded attempts;
transient failures retry up to three attempts with backoff and validation errors
fail directly. Terminal failed jobs can be reuploaded after the cause is corrected.
Content-addressed vector IDs support idempotency. SQLite/Qdrant changes are not a
distributed transaction: monitor failed jobs and reconcile after interrupted restores.

Passages are untrusted user-context data, and the model has no tools. This reduces
the consequences of prompt injection without proving immunity. Curate sources,
test adversarial passages and inspect citations. Citation-ID checking cannot prove
that a source supports a generated claim.

## Optional speech

For local speech, run `uv sync --locked --extra rag --extra tts`. Set
`TTS_ENABLED=true` and provision a trusted `PIPER_BINARY`, matching `PIPER_MODEL`
ONNX file and `PIPER_CONFIG`. Validate CPU with `PIPER_CUDA=false` first. CUDA needs
compatible ONNX Runtime GPU libraries/drivers. The base image omits optional TTS
and GPU runtime; build a deployment-specific image and mount model files read-only.

One configured voice model is used. The legacy `voice` request field does not
switch models. Confirm Piper and voice-model licenses before distribution. Text
survives TTS failure; audio requires authentication and expires after the configured
TTL. Monitor disk use even with expiry configured.

`MAX_CONCURRENT_TTS=2` independently caps synthesis. After `TTS_QUEUE_TIMEOUT=2`
seconds of queueing, a saturated speech stage returns text with a warning. A
40-user text/RAG target does not promise 40 simultaneous audio syntheses.

## Monitoring and retention

Use `/health/live` for liveness and `/health/ready` for dependency connectivity.
Readiness is not model warmup or quality verification. Scrape `/metrics` with a
dedicated admin key; monitor error rate, completed-chat latency and component
failures. Derive alert thresholds and operating limits from measured demand.

JSON request logs record ID, method, route template, status and duration, omitting
prompts, answers and tokens. SQLite retains conversation text until session deletion;
there is no automatic transcript retention policy. Apply institutional retention
rules. Qdrant stores full passages and belongs in your sensitive-data inventory.

## Backup and restore

Retain authoritative documents outside transient upload storage. Take a coordinated
backup in a maintenance window:

1. Stop new traffic, allow active chats to finish and stop the API.
2. Copy its complete data volume, including SQLite and any `-wal`/`-shm` files,
   remaining uploads and audio. A cold copy avoids inconsistent WAL capture; an
   online SQLite backup requires its backup API rather than a raw file copy.
3. Export Qdrant collection snapshots through the authenticated internal API, or
   stop Qdrant and back up its whole volume. Record collections, model revisions,
   app commit, lockfile, configuration schema and backup timestamp.
4. Encrypt/restrict backups and keep secrets separately.

Restore with traffic disabled: recover matching SQLite/Qdrant state and model
revisions, start one worker, then check document counts, known retrieval cases,
course isolation, session ownership and a full streamed chat. Jobs may rerun with
idempotent IDs. Practice restoration on an isolated stack before accepting backups.

`docker compose down` preserves named volumes; adding `--volumes` destroys their
data. Do not delete volumes as an outage troubleshooting step.

## Migration and rollbacks

Old clients must add bearer keys and course IDs; use `rag: false` without indexed
course material. `/ai-teacher` remains but audio now requires authenticated fetch
and clients must handle warnings/errors. Global demo history is not imported.
Retain the old files through migration acceptance and required retention periods.

Before updates, record the image digest and back up both stores. Test schema/model
changes on restored data. Roll back application/configuration and compatible state
together; never point old code at a new schema without a compatibility check.

## Release acceptance

These are target-deployment gates, not claimed results from the development machine:

- Run lint/tests/dependency audit and build from the lockfile. Review image/package
  advisories; pin deployment image digests after verification.
- Exercise real vLLM, embeddings, reranking and optional TTS on target hardware;
  test cold boot and cached offline restart with recorded model revisions.
- Ingest representative documents; test duplicates, changed versions, malformed
  files, deletion and restart during indexing.
- Label held-out course questions, run the retrieval evaluator and calibrate
  thresholds/top-k. Separately review answer faithfulness, citations, insufficient
  evidence and multilingual behavior.
- Check cross-user/course access, revoked keys, foreign audio, adversarial source
  text and all limits. Verify no prompts/keys enter logs.
- Load-test concurrent users against actual inference; record p50/p95 first-token
  and full-response latency, 429/503 rates, CPU/GPU memory, disk growth and cancellation.
- Fail/restart dependencies and restore backups. Assign operating owners, alerts,
  incident contacts and data-retention procedures.

For replicas/HA, first move queue/rate-limit coordination to shared infrastructure,
replace SQLite where needed, externalize files, implement schema migrations and
validate failure modes. More Uvicorn workers alone is unsupported.

CI exports the complete default-container runtime graph including the RAG extra.
For advisory lookup only, its official `torch==<version>+cpu` wheel suffix is
normalized to the upstream version because PyPI does not host those wheel variants.
The installed lockfile remains unchanged. Advisory auditing does not scan OS image
layers or independently audit third-party model weights.

## References

See [uv locked Docker builds](https://docs.astral.sh/uv/guides/integration/docker/),
[FastAPI container guidance](https://fastapi.tiangolo.com/deployment/docker/),
[Qdrant access controls](https://qdrant.tech/documentation/operations/security/) and
[GitHub Actions security](https://docs.github.com/en/actions/reference/security/secure-use)
when adapting or upgrading the deployment.
