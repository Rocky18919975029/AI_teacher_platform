# Forty RTX 4090 hosts: deployment and acceptance

This profile targets the provided inventory of **40 machines with one RTX 4090
each**. Each host runs an independent Qwen2.5-7B-Instruct/vLLM replica. One central
API owns sessions, authorization, document ingestion and batched retrieval; Nginx
routes inference to the replica with the fewest active connections. Full bounded
conversation context travels with each request, so no sticky user-to-GPU mapping
is needed. There is no 40-way tensor-parallel collective.

The central service admits 40 combined chat/search operations. A GPU replica's
`max-num-seqs=4` is a local scheduling setting, not a claim of 160-user capacity.
The fleet provides deployment flexibility and failure headroom; real capacity and
latency depend on prompts, retained history, output length, retrieval, networking
and available GPU memory. Forty physical GPUs have not been deployed or benchmarked
by this change; addresses, access and target-host acceptance remain required.

## 1. Prepare identities and the exact inventory

From the repository root, on the central administration host:

```sh
uv run python scripts/configure.py --students 40 --users-output private-users.json --course-id metaverse
cp deploy/nodes.example.json deploy/nodes.local.json
```

The setup command creates private `.env` and load-test identity files with unique
student keys and subjects; it refuses to overwrite files. Keep `private-users.json`
outside version control. Edit the inventory to replace **every** `.example` host
with an actual private/VPN address. One entry looks like this:

```json
{"name":"gpu01","url":"http://10.20.0.11:8000","bind_host":"10.20.0.11"}
```

The address above is illustrative, not a discovered host. Supply exactly 40 distinct
endpoints and names. URLs accept an HTTP origin with explicit port, no credentials,
path, query or fragment. This bundle uses a trusted private/VPN network for inference;
do not send bearer keys/plaintext course content across public networks. Enforce host
firewalls so only the gateway/health host can connect to inference ports.

```sh
uv run python scripts/generate_cluster.py deploy/nodes.local.json
```

This writes `deploy/generated/nginx.conf`, canonical `nodes.json` and 40 non-secret
per-node `.env` files. It refuses an existing output directory; use `--output` to
review a replacement bundle before rollout. No machines are contacted. For an IP
inventory, node binding defaults to that IP; for DNS inventory, explicitly set
`bind_host` to the node's private interface, otherwise binding stays on loopback.

Use the **same strong inference-only `VLLM_API_KEY`** for the API, fleet health
service and all vLLM nodes. Student/admin application keys are separate and must
never be installed as inference credentials. Supply secrets via the environment
or a private env file on each host; they do not appear in generated configuration.
Pin the same `VLLM_MODEL_REVISION` commit hash on every node.

## 2. Start one inference replica per machine

Provision a compatible NVIDIA driver, NVIDIA Container Toolkit and Docker Compose.
The node image is pinned to `vllm/vllm-openai:v0.31.0-cu129`, the upstream CUDA 12.9
artifact. Confirm driver/runtime compatibility before rollout. Copy the repository's
node Compose file and that host's generated env file onto its machine, then use:

```sh
docker compose --env-file deploy/generated/gpu01.env -f deploy/compose.inference-node.yaml up -d
```

Export `VLLM_API_KEY` and `VLLM_MODEL_REVISION` through the host's secret/configuration
manager before running it. The command refuses missing required values. Adjust the
example node name on each host; do not launch all 40 replicas on the central host.

Each replica requests exactly one GPU and uses BF16, tensor-parallel size 1,
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
results fail closed; snapshots older than 30 seconds cannot be ready. All 40
configured replicas must pass for readiness; there is no degraded-readiness override.

Active probing reports readiness; it does not rewrite Nginx's upstream list.
Passive routing handles connection failures. Wrong model/key nodes make readiness
fail and need operator correction. Readiness signals do not guarantee error-free
routing between probe cycles. The gateway has a 240-second graceful-stop window;
drain the central API before stopping inference/gateway services. DNS names resolve at Nginx startup: regenerate
and reload after inventory/address changes. The central API, gateway and Qdrant
remain single points of failure; this bundle does not introduce HA orchestration.

## 4. Warm up and verify the forty-user target

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

Measure both 40 simultaneous grounded chats and mixed chat/search traffic with
representative prompt/output lengths and multiple rounds. Record p50/p95 time to
first token, completed-response latency, success fraction, 429/503/timeout counts,
CPU/GPU utilization and memory. Empty-evidence fallbacks do not count as successful
inference. Include a node failure, client cancellation, reconnect, course-isolation
checks and long multilingual requests. Preserve model/image revisions and workload
settings alongside the report; omit prompts and credentials from shared results.

`MAX_CONCURRENT_TTS=2` and a two-second speech queue keep optional audio bounded;
saturated speech degrades to text. The target is 40 text/RAG operations, not a
40-way speech SLA. CPU retrieval can bottleneck even with idle inference GPUs:
measure it and tune batch size, reranker candidate count, CPU allocation or an
independently validated retrieval GPU service before raising admission limits.

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
[NVIDIA's GPU table](https://developer.nvidia.com/cuda/gpus) lists the RTX 4090 under
compute capability 8.9; the installed CUDA/vLLM stack still requires host validation.
