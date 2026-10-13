# Validation record

Local implementation checks on **2026-10-13**, macOS with Python 3.12. Results apply
to the checked source, not to an already deployed classroom instance.

| Check | Observed result | Scope |
| --- | --- | --- |
| Python regression suite | 150 passed, 1 skipped | API authorization/course isolation, 40-request scheduling and transport, session/stream cancellation and exact permit cleanup, persistent local Qdrant retrieval, document limits/queue recovery, configurable replica/GPU inventory, cluster probes and load-tool behavior using deterministic model doubles |
| Frontend protocol suite | 6 passed | SSE framing/UTF-8/CRLF/EOF, malformed data, cancellation, same-origin authenticated URLs and document ID mapping |
| Browser smoke | Passed | Authentication, send, streamed response, sources and saved conversation with a fake inference backend |
| Python lint | Passed | Ruff checks of repository source and operational scripts |
| Credential setup tool | Passed | Generated config loads; independent service/admin keys and 40 distinct student identities, private config/load inventory mode 0600, existing-file overwrite refused |
| Retrieval evaluator | Passed | Hand-checked document recall, MRR and empty-result scoring |
| Locked runtime dependency advisory audit | No known vulnerabilities reported | Full exported RAG runtime graph for the local platform; not an OS-image scan |
| Deployment/CI manifests | YAML parsed; action commit pins checked against upstream tags | Static validation, not an executed container deployment |

Re-run `uv run python -m pytest -q` and the CI workflow for the current revision's
test count; this record is evidence of the local check, not a permanent guarantee.

The real-model smoke test was attempted but Hugging Face downloads failed with
connection resets in this environment. It remains opt-in via:

```sh
RUN_MODEL_SMOKE=1 uv run --locked --extra rag python -m pytest -q tests/test_rag.py -m integration
```

No real BGE/CrossEncoder quality score, vLLM/GPU benchmark, Piper audio check,
remote Qdrant cluster check, real-model multi-user capacity or classroom rollout is claimed
by this upgrade. Docker was unavailable on the development host, so image build
and Compose startup need the CI/container host. The [operations acceptance gates](operations.md#release-acceptance)
cover these remaining checks, backup restoration and institutional deployment.

## Forty-user application baseline and configurable deployment

The new profile adds tests for 40 simultaneous fake inference streams, mixed
chat/search admission, exact cancellation/permit cleanup, 40 real TCP connections
to a deterministic upstream and five eight-query retrieval batches. These verify
application scheduling and transport behavior using controlled dependencies;
they do not measure real-model throughput or establish a required GPU/machine count.

Cluster configuration accepts a nonempty replica inventory independently of student
count. Replicas can share a host with distinct ports and GPU devices, and generated
Compose project names separate their deployments. Inventory validation rejects
credentials and configuration injection, and generation preserves existing output
directories. Readiness requires all configured replicas and reports aggregate counts
only; stale or failed probes do not indicate readiness. The node and gateway Compose
files have been parsed as YAML; generated Nginx and GPU container runtime acceptance
still require a Docker/NVIDIA target host.

The [capacity/deployment runbook](cluster40.md) supplies a real-service load procedure with
distinct student identities and explicit latency gates. No successful real-model
40-user acceptance report is included. Add measured results only after deployment,
including hardware, model revisions, prompt/output sizes and the actual workload.
Two-replica topology examples are not evidence that two GPUs can serve the target.
The pending acceptance criterion is at least 40 simultaneous student browser
chat/RAG interactions through one HTTPS interface; no latency guarantee or actual
LAN deployment is asserted here.

CI audits upstream PyTorch advisories for its official CPU wheel version after
normalizing the `+cpu` suffix in an audit-only manifest. Lockfile package versions
and hashes remain intact. Successful advisory scans cannot certify security of
unreviewed models, parser inputs or future dependency releases.
