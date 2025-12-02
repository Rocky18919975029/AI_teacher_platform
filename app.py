from pathlib import Path
import subprocess
import uuid
from typing import Optional, Dict

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ==========================
# CONFIG
# ==========================

# vLLM OpenAI API
VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
VLLM_API_KEY = "EMPTY"   # must match --api-key

# File paths
BASE_DIR = Path(__file__).parent

# ---- Voice model registry ----
# For now, both voices can point to the same model.
# If you download more Piper voices, just add them here.
VOICE_MODELS: Dict[str, Dict[str, Path]] = {
    "teacher_female": {
        "model": BASE_DIR / "models" / "piper" / "en_US-libritts-high.onnx",
        "config": BASE_DIR / "models" / "piper" / "en_US-libritts-high.onnx.json",
    },
    "teacher_male": {
        "model": BASE_DIR / "models" / "piper" / "en_US-libritts-high.onnx",
        "config": BASE_DIR / "models" / "piper" / "en_US-libritts-high.onnx.json",
    },
}
DEFAULT_VOICE = "teacher_female"

AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

# ==========================
# FastAPI INIT
# ==========================

app = FastAPI(title="AI Teacher API (LLM + TTS)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # simplest: allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve /audio/*.wav as static files
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# ==========================
# Request + Response Schema
# ==========================

class AIRequest(BaseModel):
    prompt: str
    with_audio: bool = True
    voice: Optional[str] = DEFAULT_VOICE  # e.g. "teacher_female", "teacher_male"


class AIResponse(BaseModel):
    text: str
    audio_url: Optional[str] = None


# ==========================
# LLM CALLER
# ==========================

def call_llm(prompt: str) -> str:
    system_prompt = (
        "You are a patient, friendly AI teacher for teenagers. "
        "Explain concepts clearly with simple language and short paragraphs."
    )

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VLLM_API_KEY}",
    }

    try:
        resp = httpx.post(VLLM_URL, json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")

    try:
        text = resp.json()["choices"][0]["message"]["content"]
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid LLM response: {e}")


# ==========================
# TTS CALLER (multi-voice)
# ==========================

def synthesize_speech(text: str, voice: Optional[str] = None) -> str:
    voice_key = voice or DEFAULT_VOICE
    cfg = VOICE_MODELS.get(voice_key) or VOICE_MODELS[DEFAULT_VOICE]

    model_path = cfg["model"]
    config_path = cfg["config"]

    if not model_path.exists():
        raise HTTPException(500, f"Piper model missing: {model_path}")
    if not config_path.exists():
        raise HTTPException(500, f"Piper config missing: {config_path}")

    file_id = uuid.uuid4().hex
    wav_path = AUDIO_DIR / f"{file_id}.wav"

    cmd = [
        "piper",
        "-m", str(model_path),
        "-c", str(config_path),
        "-f", str(wav_path),
        "--cuda",
    ]

    try:
        subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise HTTPException(500, "Piper TTS failed")

    return f"/audio/{wav_path.name}"


# ==========================
# MAIN ENDPOINT
# ==========================

@app.post("/ai-teacher", response_model=AIResponse)
def ai_teacher(req: AIRequest):
    answer = call_llm(req.prompt)

    audio_url = None
    if req.with_audio:
        audio_url = synthesize_speech(answer, voice=req.voice)

    return AIResponse(
        text=answer,
        audio_url=audio_url,
    )


# ==========================
# HEALTH CHECK
# ==========================

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Teacher API running"}
