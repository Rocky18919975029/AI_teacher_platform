from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Principal(BaseModel):
    key: SecretStr
    subject: str = Field(min_length=1, max_length=100)
    role: Literal['admin', 'student'] = 'student'
    courses: list[str] = Field(min_length=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    api_keys: list[Principal] = []
    data_dir: Path = Path('data')
    vllm_url: str = 'http://127.0.0.1:8000/v1/chat/completions'
    vllm_model: str = 'Qwen/Qwen2.5-7B-Instruct'
    vllm_api_key: SecretStr = SecretStr('EMPTY')
    llm_timeout: float = Field(default=90, ge=1, le=600)
    max_output_tokens: int = Field(default=1024, ge=1, le=4096)
    max_history_chars: int = Field(default=12000, ge=1000, le=60000)
    max_concurrent_chats: int = Field(default=4, ge=1, le=64)
    requests_per_minute: int = Field(default=60, ge=1)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=50 * 1024 * 1024)
    max_pending_jobs: int = Field(default=50, ge=1, le=1000)
    qdrant_url: str = 'http://127.0.0.1:6333'
    qdrant_api_key: str | None = None
    qdrant_collection: str = 'teacher_courses_v1'
    rag_embedding_model: str = 'BAAI/bge-small-en-v1.5'
    rag_embedding_revision: str | None = None
    rag_reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    rag_reranker_revision: str | None = None
    rag_rerank_enabled: bool = True
    rag_device: str = 'cpu'
    rag_chunk_tokens: int = Field(default=256, ge=32, le=1024)
    rag_chunk_overlap: int = Field(default=40, ge=0, le=256)
    rag_retrieve_k: int = Field(default=20, ge=1, le=100)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_context_tokens: int = Field(default=1800, ge=128, le=12000)
    rag_min_score: float = Field(default=0.25, ge=-1, le=1)
    rag_max_extracted_chars: int = Field(default=500000, ge=1000, le=5000000)
    rag_max_chunks: int = Field(default=2500, ge=1, le=10000)
    rag_max_pdf_pages: int = Field(default=300, ge=1, le=2000)
    rag_batch_size: int = Field(default=32, ge=1, le=128)
    rag_qdrant_timeout: int = Field(default=15, ge=1, le=120)
    tts_enabled: bool = False
    piper_binary: str = 'piper'
    piper_model: Path = Path('models/piper/en_US-libritts-high.onnx')
    piper_config: Path = Path('models/piper/en_US-libritts-high.onnx.json')
    piper_cuda: bool = True
    tts_timeout: float = Field(default=45, ge=1, le=120)
    audio_ttl_seconds: int = Field(default=3600, ge=60)

    @model_validator(mode='after')
    def secure_config(self):
        if self.rag_chunk_overlap >= self.rag_chunk_tokens:
            raise ValueError('RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_TOKENS')
        if self.rag_top_k > self.rag_retrieve_k:
            raise ValueError('RAG_TOP_K must not exceed RAG_RETRIEVE_K')
        if not self.api_keys:
            raise ValueError('Configure API_KEYS with scoped student/admin keys before startup')
        keys = [p.key.get_secret_value() for p in self.api_keys]
        if any(len(k) < 32 or 'CHANGE_ME' in k for k in keys) or len(set(keys)) != len(keys):
            raise ValueError('API keys must be unique and at least 32 characters')
        return self
