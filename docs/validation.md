# Validation record

Local implementation checks on **2026-10-13**, macOS with Python 3.12. Results apply
to the checked source, not to an already deployed classroom instance.

| Check | Observed result | Scope |
| --- | --- | --- |
| Python regression suite | 45 passed, 1 skipped | API authorization/course isolation, session/stream cancellation and exact permit cleanup, persistent local Qdrant retrieval, document limits and queue recovery using deterministic model doubles |
| Frontend protocol suite | 6 passed | SSE framing/UTF-8/CRLF/EOF, malformed data, cancellation, same-origin authenticated URLs and document ID mapping |
| Browser smoke | Passed | Authentication, send, streamed response, sources and saved conversation with a fake inference backend |
| Python lint | Passed | Ruff checks of repository source and operational scripts |
| Credential setup tool | Passed | Generated config loads; four independent secrets, mode 0600, existing-file overwrite refused |
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
remote Qdrant cluster check, multi-user capacity or classroom rollout is claimed
by this upgrade. Docker was unavailable on the development host, so image build
and Compose startup need the CI/container host. The [operations acceptance gates](operations.md#release-acceptance)
cover these remaining checks, backup restoration and institutional deployment.

CI audits upstream PyTorch advisories for its official CPU wheel version after
normalizing the `+cpu` suffix in an audit-only manifest. Lockfile package versions
and hashes remain intact. Successful advisory scans cannot certify security of
unreviewed models, parser inputs or future dependency releases.
