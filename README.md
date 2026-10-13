# AI Teacher Platform

**English** | [简体中文](README.zh-CN.md)

AI Teacher Platform is a self-hosted AI teaching assistant that combines a large language model (LLM) with retrieval-augmented generation (RAG). Instructors upload course materials to build a course knowledge base, while students ask questions through a web interface and receive answers grounded in those materials, with source references.

The platform uses **Qwen2.5**, **vLLM**, **FastAPI**, and **Qdrant**. It also supports general AI chat without course retrieval and optional text-to-speech.

## Features

- **AI tutoring:** Multi-turn conversations with real-time streamed responses, saved chat history, and the ability to start or stop a conversation.
- **Course knowledge base:** Upload PDF, DOCX, UTF-8 TXT, and Markdown files; the platform extracts, indexes, and searches course content.
- **Source-grounded answers:** Retrieve relevant passages and show document titles, source excerpts, and PDF page numbers when available. If no suitable material is found, the assistant indicates insufficient evidence.
- **Course and user access:** Separate administrator and student keys, course-scoped access, and private conversation histories.
- **Document management:** Administrators can upload, view, and delete course documents directly in the browser.
- **Optional speech:** Read answers aloud using locally hosted Piper TTS (requires additional setup).
- **Multi-user support:** Request limits and batched retrieval help manage classroom traffic.

## Technical Highlights

- **RAG pipeline:** Sentence Transformers embeddings → Qdrant vector search → CrossEncoder reranking → cited LLM responses.
- **Responsive inference:** OpenAI-compatible vLLM integration, server-sent event (SSE) token streaming, and bounded concurrent requests.
- **Local-first architecture:** Self-hosted inference and storage, persistent SQLite conversations, and per-course knowledge isolation.
- **Operational readiness:** Durable document indexing jobs, health checks, access controls, and Prometheus metrics.

## Tech Stack

| Component | Technology |
| --- | --- |
| Language model | Qwen2.5-7B-Instruct |
| LLM serving | vLLM |
| Backend | Python 3.12, FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Retrieval | Sentence Transformers, BGE embeddings, CrossEncoder |
| Vector database | Qdrant |
| Application data | SQLite |
| Optional speech | Piper TTS |
| Deployment | Docker Compose |

## Quick Start

### 1. Prerequisites

You will need:

- **Git**, **Python 3.12**, and [uv](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** with Docker Compose
- A separately running **vLLM** service with access to the Qwen2.5-7B-Instruct model and suitable GPU resources

> Docker Compose starts the application and Qdrant. **It does not start vLLM.** Model weights and GPU drivers are not included in this repository.

### 2. Clone the repository

```bash
git clone --branch codex/production-ai-teacher --single-branch \
  https://github.com/Rocky18919975029/AI_teacher_platform.git
cd AI_teacher_platform
```

### 3. Create your configuration

```bash
uv run python scripts/configure.py
```

This creates a private `.env` file with separate administrator, student, vLLM, and Qdrant credentials. By default, it creates one administrator, one student, and a course named `metaverse`.

To generate credentials for more students **when creating a new `.env`**, run instead:

```bash
uv run python scripts/configure.py --students 40 --course-id metaverse
```

The script will not overwrite an existing `.env`. Find the generated administrator and student access tokens in its `API_KEYS` setting; distribute only the appropriate key to each user.

### 4. Connect a vLLM server

Start a vLLM server separately. For example, after installing vLLM and obtaining the model weights:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --api-key "<VLLM_API_KEY_FROM_ENV>"
```

Edit `.env` so the application uses the correct endpoint and matching key:

```dotenv
VLLM_URL=http://<VLLM_HOST_IP>:8000/v1/chat/completions
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VLLM_API_KEY=<SAME_KEY_USED_BY_VLLM>
```

**Important:** When the application runs in Docker, `VLLM_URL` must be reachable **from inside the container**. Do not use `127.0.0.1` unless vLLM actually runs in the same container. Keep the inference server on a trusted network.

### 5. Start the platform

```bash
docker compose up --build -d
docker compose ps
```

Check the API:

```bash
curl http://127.0.0.1:9000/health/live
curl http://127.0.0.1:9000/health/ready
```

`/health/ready` checks connectivity to the dependent services. It may return `503` if Qdrant or vLLM is unavailable.

Open the web interface:

**http://127.0.0.1:9000/ui/**

Enter a course ID (for example, `metaverse`) and its authorized access token, then select **Connect to classroom**.

## How to Use

### For instructors

1. Sign in with an **administrator** token and the relevant course ID.
2. Open **Course knowledge** in the sidebar.
3. Choose a PDF, DOCX, TXT, or Markdown file and click **Upload & index**.
4. Wait for indexing to complete. The material is now available for course-grounded questions.
5. Use the same panel to view or delete documents.

The default maximum upload size is **10 MB per file**. Scanned PDFs require OCR before uploading; OCR is not built in.

### For students

1. Open the classroom URL provided by the instructor.
2. Enter the course ID and your personal access token.
3. Ask a question in the chat box.
4. Keep **Course grounding** enabled for answers based on uploaded materials, or disable it for general AI chat.
5. Review the cited sources when shown. You can also open previous conversations, start a new one, or stop a response.

Students only need a browser; they do not need to install the model or database.

## Configuration and Optional Features

Common settings are defined in `.env`:

| Setting | Purpose |
| --- | --- |
| `API_KEYS` | Users, roles, credentials, and authorized courses |
| `VLLM_URL`, `VLLM_MODEL`, `VLLM_API_KEY` | LLM endpoint, model, and authentication |
| `QDRANT_API_KEY` | Vector database authentication |
| `MAX_CONCURRENT_CHATS` | Maximum active chat/search requests (default: `40`) |
| `MAX_UPLOAD_BYTES` | Per-file upload limit |
| `TTS_ENABLED` | Enable optional Piper speech synthesis |

The default `40` is a **request admission limit**, not a verified 40-student performance benchmark. Actual capacity depends on your hardware and workload.

For optional speech, install the TTS dependencies, provide a Piper executable and voice model, and set `TTS_ENABLED=true`. See the [operations guide](docs/operations.md#optional-speech). The default Docker image does not include Piper.

To run the backend directly for local development (with vLLM and Qdrant already available):

```bash
uv sync --locked --extra rag
uv run uvicorn app:app --host 127.0.0.1 --port 9000 --workers 1 --no-access-log
```

## API and Documentation

The browser uses the same HTTP API that other clients can integrate with. API requests require `Authorization: Bearer <access-token>`.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/chat/stream` | Stream an AI response |
| `POST /api/search` | Search course content |
| `POST /api/documents?course_id=...` | Upload a document (administrator) |
| `GET /api/documents?course_id=...` | List course documents |
| `GET /api/sessions?course_id=...` | List conversation history |
| `GET /health/ready` | Check service readiness |

- [API reference](docs/api.md)
- [Deployment and operations](docs/operations.md)
- [Multi-user deployment](docs/cluster40.md)

**Deployment notes:** Keep `.env` out of version control. For access from other devices, use an HTTPS reverse proxy; the default Docker setup binds the application to local loopback. AI answers can be incorrect, so verify important information against the original course materials.
