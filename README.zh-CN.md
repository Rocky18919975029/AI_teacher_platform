# AI Teacher Platform

[English](README.md) | **简体中文**

AI Teacher Platform 是一个支持本地部署的 **AI 智能教学平台**，结合大语言模型（LLM）和检索增强生成（RAG），为学生提供基于课程资料的智能问答服务。教师可以上传教材、讲义和参考文档，建立课程知识库；学生则可以通过浏览器提问，获取结合课程内容生成、附带资料来源的回答。

项目使用 **Qwen2.5、vLLM、FastAPI 和 Qdrant** 构建，同时支持不使用课程检索的通用 AI 对话，以及可选的语音播报。

## 功能介绍

- **AI 智能问答：** 支持多轮对话、实时流式输出、历史对话保存，以及新建或停止对话。
- **课程知识库：** 上传 PDF、DOCX、UTF-8 TXT 和 Markdown 文件，自动提取、建立索引并检索课程内容。
- **基于资料回答：** 检索相关段落，在回答中展示文档名称、引用内容，以及可用的 PDF 页码；缺少相关资料时会提示证据不足。
- **用户与课程权限：** 区分管理员和学生访问密钥，按课程隔离访问权限，保护个人对话记录。
- **文档管理：** 管理员可直接通过网页上传、查看和删除课程资料。
- **可选语音播报：** 使用本地 Piper TTS 将回答转换成语音（需要额外配置）。
- **多用户访问：** 通过请求并发限制和检索批处理管理课堂访问负载。

## 技术亮点

- **完整 RAG 检索链路：** Sentence Transformers 向量化 → Qdrant 向量检索 → CrossEncoder 重排序 → 带引用的模型回答。
- **实时交互与并发控制：** 基于 vLLM 提供 OpenAI 兼容的模型推理服务，使用 SSE 流式输出，并限制同时处理的请求数量。
- **本地化与课程隔离：** 推理和数据存储可自行部署，SQLite 持久化对话，不同课程的知识检索互相隔离。
- **面向实际运行的设计：** 持久化文档索引任务、健康检查、访问控制与 Prometheus 监控指标。

## 技术栈

| 组件 | 技术 |
| --- | --- |
| 大语言模型 | Qwen2.5-7B-Instruct |
| 模型推理 | vLLM |
| 后端 | Python 3.12、FastAPI |
| 前端 | HTML、CSS、JavaScript |
| 知识检索 | Sentence Transformers、BGE、CrossEncoder |
| 向量数据库 | Qdrant |
| 业务数据 | SQLite |
| 可选语音合成 | Piper TTS |
| 部署 | Docker Compose |

## 快速开始

### 1. 环境准备

需要提前安装或准备：

- **Git、Python 3.12** 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** 和 Docker Compose
- 可以运行 Qwen2.5-7B-Instruct、具备合适 GPU 资源的独立 **vLLM** 服务

> Docker Compose 会启动教学平台和 Qdrant，**不会自动启动 vLLM**。仓库也不包含模型权重和 GPU 驱动。

### 2. 克隆项目

```bash
git clone --branch codex/production-ai-teacher --single-branch \
  https://github.com/Rocky18919975029/AI_teacher_platform.git
cd AI_teacher_platform
```

### 3. 生成配置文件

```bash
uv run python scripts/configure.py
```

命令会创建私有的 `.env` 文件，自动生成管理员、学生、vLLM 和 Qdrant 的访问密钥。默认创建 **1 个管理员、1 个学生，以及 `metaverse` 课程**。

如果需要为更多学生生成独立账号，请在**首次创建 `.env` 时**改用：

```bash
uv run python scripts/configure.py --students 40 --course-id metaverse
```

脚本不会覆盖已有的 `.env`。管理员和学生的访问密钥保存在 `.env` 的 `API_KEYS` 配置中，应分别发给对应用户。

### 4. 连接 vLLM 推理服务

需要单独安装并启动 vLLM。准备好模型权重后，可以使用以下命令启动：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --api-key "<VLLM_API_KEY_FROM_ENV>"
```

然后编辑 `.env`，填写教学平台能够访问的 vLLM 地址和与推理服务一致的密钥：

```dotenv
VLLM_URL=http://<VLLM_HOST_IP>:8000/v1/chat/completions
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VLLM_API_KEY=<SAME_KEY_USED_BY_VLLM>
```

**注意：** 如果教学平台运行在 Docker 容器中，`VLLM_URL` 必须是**容器内部可访问的地址**，不能直接使用容器的 `127.0.0.1` 访问宿主机上的 vLLM。推理服务应部署在可信网络中。

### 5. 启动教学平台

```bash
docker compose up --build -d
docker compose ps
```

检查服务状态：

```bash
curl http://127.0.0.1:9000/health/live
curl http://127.0.0.1:9000/health/ready
```

`/health/ready` 会检查依赖服务是否可连接。如果 Qdrant 或 vLLM 尚未准备好，可能返回 HTTP `503`。

使用浏览器打开：

**http://127.0.0.1:9000/ui/**

输入课程 ID（例如 `metaverse`）和有权访问该课程的密钥，点击 **Connect to classroom** 即可进入平台。

## 使用指南

### 教师：管理课程资料

1. 使用**管理员密钥**和课程 ID 登录。
2. 打开侧边栏的 **Course knowledge**。
3. 选择 PDF、DOCX、TXT 或 Markdown 文件，点击 **Upload & index**。
4. 等待文档索引完成，课程资料即可用于知识问答。
5. 可以在同一面板查看或删除已上传的文档。

默认单个文件的大小上限为 **10 MB**。扫描版 PDF 需要先使用外部 OCR 工具识别文字，平台暂未内置 OCR。

### 学生：使用 AI 教师

1. 打开教师提供的课程网页地址。
2. 输入课程 ID 和个人访问密钥。
3. 在聊天框中输入问题并发送。
4. 开启 **Course grounding**，让 AI 结合课程资料回答；关闭后则使用通用 AI 对话。
5. 查看回答中显示的引用来源，也可以浏览历史对话、新建会话或停止生成。

学生只需要浏览器，不必自行安装模型或数据库。

## 配置与可选功能

常用设置位于 `.env`：

| 配置项 | 说明 |
| --- | --- |
| `API_KEYS` | 用户身份、角色、密钥和课程权限 |
| `VLLM_URL`、`VLLM_MODEL`、`VLLM_API_KEY` | 模型推理地址、模型名称和认证密钥 |
| `QDRANT_API_KEY` | 向量数据库认证密钥 |
| `MAX_CONCURRENT_CHATS` | 同时处理的聊天/检索请求数量上限（默认 `40`） |
| `MAX_UPLOAD_BYTES` | 单个上传文件的大小限制 |
| `TTS_ENABLED` | 是否启用 Piper 语音合成 |

默认的 `40` 是**请求准入上限**，并不代表已经验证可在任意硬件上支持 40 名学生同时使用。实际容量需要结合 GPU 和具体负载测试。

如需语音播报，需要安装可选的 TTS 依赖、准备 Piper 程序及语音模型，并设置 `TTS_ENABLED=true`。参见[运维文档中的语音配置](docs/operations.md#optional-speech)。默认 Docker 镜像不包含 Piper。

如果需要不使用 Docker 启动后端进行本地开发，请先确保 Qdrant 和 vLLM 已运行，然后执行：

```bash
uv sync --locked --extra rag
uv run uvicorn app:app --host 127.0.0.1 --port 9000 --workers 1 --no-access-log
```

## API 与文档

网页使用的 HTTP API 也可以对接其他客户端。调用 API 时需要在请求头添加 `Authorization: Bearer <access-token>`。

| 接口 | 功能 |
| --- | --- |
| `POST /api/chat/stream` | 流式 AI 问答 |
| `POST /api/search` | 搜索课程知识库 |
| `POST /api/documents?course_id=...` | 上传文档（管理员） |
| `GET /api/documents?course_id=...` | 查看课程文档 |
| `GET /api/sessions?course_id=...` | 查询历史对话 |
| `GET /health/ready` | 检查服务就绪状态 |

- [API 接口说明](docs/api.md)
- [部署与运维指南](docs/operations.md)
- [多用户部署指南](docs/cluster40.md)

**部署提醒：** 不要将 `.env` 提交到版本控制。如果需要从其他设备访问，应使用 HTTPS 反向代理；默认 Docker 配置仅监听本机回环地址。AI 回答可能存在错误，重要内容应结合引用的原始资料核对。
