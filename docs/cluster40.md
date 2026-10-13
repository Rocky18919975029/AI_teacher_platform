# At least forty concurrent students: capacity and deployment

The requirement is **at least 40 students interacting with web chat and course
retrieval simultaneously**. GPU and machine counts are not fixed. The application
can use an inference replica pool on one multi-GPU LAN host or multiple LAN hosts;
the required hardware is determined by the actual model and workload benchmark.
There is no one-to-one mapping between students and GPUs.

Students receive one institution-provided **HTTPS `/ui/` address** and their
course-scoped key/course. They use the browser chat interface only. Backend,
inference and vector-store addresses, GPU selection and the commands below are
operator configuration, not student setup steps.

One central API owns sessions, authorization, ingestion and batched retrieval;
Nginx routes inference to the replica with the fewest active connections. Each
replica runs Qwen2.5-7B-Instruct/vLLM with its selected GPU. Full bounded conversation
context travels with each request, so no sticky user-to-replica mapping is needed.
These independent replicas do not use cross-host tensor parallelism.

The default admission cap is 40 combined active chat/search requests. It is a
baseline for the concurrent-student target, not a count of registered users or
browser tabs. A replica's `max-num-seqs=4` is a local scheduling setting, not a
capacity guarantee. Higher admission can be configured with `MAX_CONCURRENT_CHATS`
after end-to-end testing. Real performance depends on prompts, retained history,
output length, retrieval, LAN traffic and GPU memory. No real LAN deployment or
hardware capacity measurement has been performed by generating this bundle.

## 1. Prepare identities and the exact inventory

From the repository root, on the central administration host:

```sh
uv run python scripts/configure.py --students 40 --users-output private-users.json --course-id metaverse
cp deploy/nodes.example.json deploy/nodes.local.json
```

The setup command creates private `.env` and load-test identity files with unique
student keys and subjects; it refuses to overwrite files. Keep `private-users.json`
outside version control. Edit the inventory to replace **every** `.example` host
with the actual LAN address. For example, two replicas can share one host with
different ports and GPU selections:

```json
{
  "nodes": [
    {"name":"replica01","url":"http://inference-host.example:8100","bind_host":"192.0.2.10","gpu_device":"0"},
    {"name":"replica02","url":"http://inference-host.example:8101","bind_host":"192.0.2.10","gpu_device":"1"}
  ]
}
```

The hostname/address above are illustrative and must be replaced. **This topology
example does not establish that two GPUs can serve 40 students.** Supply any
nonempty inventory matching the actual machines: one entry per inference replica,
with unique names/endpoints and no duplicate GPU selection on a shared host.
`gpu_device` defaults to `0`; use a GPU index or full NVIDIA `GPU-...` UUID.
Use one consistent hostname/address and device-identifier convention per physical
host: validation cannot detect two aliases for the same machine or a GPU identified
once by index and again by UUID.

URLs accept an HTTP origin with explicit port, no credentials, path, query or
fragment. The supplied inference transport is for the trusted LAN; use a protected
network when needed, and do not expose these bearer-key endpoints publicly. Host
firewalls should allow inference access only from the gateway/health host. Replicas
on different hosts can reuse a port; replicas on the same host need distinct ports.

```sh
uv run python scripts/generate_cluster.py deploy/nodes.local.json
```

This writes `deploy/generated/nginx.conf`, canonical `nodes.json` and one non-secret
`.env` file per configured replica. Each env file includes `VLLM_GPU_DEVICE` and a
distinct `COMPOSE_PROJECT_NAME=teacher-<lowercase-replica-name>`. The generator
refuses an existing output directory; use `--output` to review a replacement bundle
before rollout. No machines are contacted. For an IP
inventory, node binding defaults to that IP; for DNS inventory, explicitly set
`bind_host` to the node's private interface, otherwise binding stays on loopback.

Use the **same strong inference-only `VLLM_API_KEY`** for the API, fleet health
service and all vLLM nodes. Student/admin application keys are separate and must
never be installed as inference credentials. Supply secrets via the environment
or a private env file on each host; they do not appear in generated configuration.
Pin the same `VLLM_MODEL_REVISION` commit hash on every node.

## 2. Start the selected replicas on their assigned hosts

Provision a compatible NVIDIA driver, NVIDIA Container Toolkit and Docker Compose.
The node image is pinned to `vllm/vllm-openai:v0.31.0-cu129`, the upstream CUDA 12.9
artifact. Confirm driver/runtime compatibility before rollout. Copy the repository's
node Compose file and the assigned replicas' env files onto each host. For the
illustrative two-replica, same-host topology, run both commands on that host:

```sh
docker compose --env-file deploy/generated/replica01.env -f deploy/compose.inference-node.yaml up -d
docker compose --env-file deploy/generated/replica02.env -f deploy/compose.inference-node.yaml up -d
```

Export `VLLM_API_KEY` and `VLLM_MODEL_REVISION` through the host's secret/configuration
manager before running it. The command refuses missing required values. Run only
the replicas assigned to the current machine. Their separate Compose project names
prevent one invocation from replacing another replica's container/volume; do not
override them with the same `-p` option or global `COMPOSE_PROJECT_NAME`.

Compose selects the replica's explicit `VLLM_GPU_DEVICE` via `device_ids`.
Each replica uses BF16, tensor-parallel size 1,
`max-num-seqs=4`, `gpu-memory-utilization=0.88`, prefix caching and a 16,384-token
model context. Model weights/cache persist locally. These are starting settings:
check startup KV-cache allocation and representative long prompts on one node
before replicating the configuration. Reduce context/concurrency if profiling fails;
do not interpret the memory fraction as a guarantee that all requested sequences fit.
Request logging is disabled; do not enable debug response logging with real students.

Apply the central API settings in `deploy/cluster.env.example`: history 4,000
characters, retrieved context 1,200 embedding tokens, output 1,024 tokens, and 40
active requests. Characters are not generation tokens, especially for Chinese.
Validate the complete system + question + history + source context + output with
the actual serving tokenizer. A 16k model limit is not permission to fill every
field to its maximum simultaneously. Treat context-limit errors as failed requests
in acceptance, and tune history/context/prompt policy accordingly.

## 3. Start the gateway and central API

Keep the generated bundle at `deploy/generated/` on the gateway host. Set
`GATEWAY_BIND_HOST` to its private interface for remote/container access (default
loopback), `GATEWAY_PORT=8001` and the shared inference key, then:

```sh
docker compose -f deploy/compose.gateway.yaml up --build -d
curl --fail http://127.0.0.1:8001/health
```

Use the configured private address instead of loopback in the health command when
binding only that interface. Point central API `VLLM_URL` to
`http://ACTUAL_GATEWAY_ADDRESS:8001/v1/chat/completions`, reachable from its container,
and retain the single API worker. Keep the student-facing API behind HTTPS.

Nginx uses shared least-connection counters, persistent upstream connections and
passive failure detection (`max_fails=2`, `fail_timeout=10s`). SSE response buffering
and caching are disabled; client disconnects propagate upstream. Connection failures
can try another node, bounded by three attempts/five seconds. The configuration
does **not** enable retries of a POST already sent upstream, and never replays a
partially streamed answer. The application uses an absolute generation deadline;
Nginx's 120-second read timeout is an idle timeout, not a total-response SLA.

The internal fleet-health service probes every configured node's `/health` and
authenticated `/v1/models`, including the expected model ID. Gateway `/health`
returns aggregate counts, never node addresses, prompts or keys. Initial or stale
results fail closed; snapshots older than 30 seconds cannot be ready. All
configured replicas must pass for readiness; there is no degraded-readiness override.

Active probing reports readiness; it does not rewrite Nginx's upstream list.
Passive routing handles connection failures. Wrong model/key nodes make readiness
fail and need operator correction. Readiness signals do not guarantee error-free
routing between probe cycles. The gateway has a 240-second graceful-stop window;
drain the central API before stopping inference/gateway services. DNS names resolve
at Nginx startup: regenerate and reload after inventory/address changes. The central API, gateway and Qdrant
remain single points of failure; this bundle does not introduce HA orchestration.

## 4. Warm up and verify the concurrent-student target

Before admitting students, complete one real GPU response on each replica, warm
embedding/reranker models, ingest course documents and verify fleet/API readiness.
Use 40 separately configured student identities; one shared key exercises only one
subject and one rate-limit bucket. Run the load harness against the **student API**,
not directly against the vLLM gateway, so retrieval and session work are included.
See `uv run python scripts/load_test.py --help` for the complete interface and
choose explicit latency/success gates before running. For example:

```sh
uv run python scripts/load_test.py \
  --base-url http://127.0.0.1:9000 \
  --users-file private-users.json --users 40 --course-id metaverse \
  --mode chat --rounds 3 --warmup-rounds 1 --require-sources \
  --max-p95-ttft 5 --max-p95-total 60 --output load-results-chat.json
```

The 5-second TTFT and 60-second completion limits are **example acceptance gates**,
not measured performance promises. Replace the origin with your actual HTTPS
student API for remote testing. Repeat using `--mode mixed` and `--mode search`,
distinct output filenames and suitable gates. Search has no first-token metric;
omit `--max-p95-ttft` for a search-only run. Use `--prompts-file` for a representative
JSON array of course questions. Existing report files are never overwritten.

Measure at least 40 simultaneous grounded chats and mixed chat/search traffic with
representative prompt/output lengths and multiple rounds. Record p50/p95 time to
first token, completed-response latency, success fraction, 429/503/timeout counts,
CPU/GPU utilization and memory. Empty-evidence fallbacks do not count as successful
inference. Include a node failure, client cancellation, reconnect, course-isolation
checks and long multilingual requests. Preserve model/image revisions and workload
settings alongside the report; omit prompts and credentials from shared results.

`MAX_CONCURRENT_TTS=2` and a two-second speech queue keep optional audio bounded;
saturated speech degrades to text. The baseline target covers 40 text/RAG operations;
speech capacity requires separate validation. CPU retrieval can bottleneck even with idle inference GPUs:
measure it and tune batch size, reranker candidate count, CPU allocation or an
independently validated retrieval GPU service before raising admission limits.
If the requirement grows beyond 40 active operations, raise the API admission cap
and load-test user count together and repeat acceptance on the intended hardware.
Do not infer capacity simply by multiplying replica count by sequence limits.

Application tests that hold 40 fake streams or batch 40 deterministic retrievals
prove control-flow/concurrency behavior only. They do not establish real GPU
throughput. Record target-host results in [the validation record](validation.md)
only after the complete workload has passed.

## Upstream references

Version/behavior references: [vLLM 0.31.0 release images](https://github.com/vllm-project/vllm/releases/tag/v0.31.0),
[vLLM Docker deployment](https://docs.vllm.ai/en/stable/deployment/docker/),
[vLLM API-key environment setting](https://docs.vllm.ai/en/stable/configuration/env_vars/),
[Nginx stable release](https://nginx.org/en/download.html),
[least-connection/passive-failure routing](https://nginx.org/en/docs/http/ngx_http_upstream_module.html)
and [stream buffering/retry semantics](https://nginx.org/en/docs/http/ngx_http_proxy_module.html).
See [Compose GPU device selection](https://docs.docker.com/compose/how-tos/gpu-support/)
and [Compose project-name precedence](https://docs.docker.com/compose/how-tos/environment-variables/envvars/)
for per-replica device isolation and project naming. The chosen CUDA/vLLM stack
still requires validation on the actual installed hardware.
