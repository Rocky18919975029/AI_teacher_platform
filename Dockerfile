# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra rag

FROM python:3.12-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 teacher && useradd --uid 10001 --gid teacher --no-create-home teacher \
    && mkdir -p /app/data /app/model-cache && chown -R teacher:teacher /app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data HF_HOME=/app/model-cache TOKENIZERS_PARALLELISM=false
COPY --from=builder /app/.venv /app/.venv
COPY --chown=teacher:teacher app.py ./
COPY --chown=teacher:teacher teacher/ ./teacher/
COPY --chown=teacher:teacher frontend/ ./frontend/
USER 10001:10001
EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/health/live', timeout=3)"
# One worker is intentional: SQLite job recovery and in-memory admission limits are process-local.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1", "--no-access-log", "--timeout-graceful-shutdown", "240"]
