The following is the README.md for my project, read and understand it, later I will provide the codes: 📘 AI Teacher Platform
Local LLM (Qwen2.5) + Piper GPU TTS + FastAPI Backend + Interactive Web UI
🚀 Overview

The AI Teacher Platform is a fully local, privacy-preserving educational assistant that can:

Answer questions using a local LLM (Qwen2.5-7B-Instruct)

Speak the answers aloud using Piper TTS (GPU ONNX)

Provide a clean, interactive web chat UI

Automatically generate WAV audio responses

Maintain conversation history (configurable)

Support stop/continue audio playback with waveform visualization

Everything runs offline, on your own server, with no external API calls.

✨ Features
🤖 Local LLM (via vLLM)

Fast, efficient inference

Supports Qwen2.5-7B-Instruct (or any OpenAI-compatible model)

Chat-style structured responses

🔊 Piper TTS (GPU Accelerated)

Natural-sounding voice

ONNX Runtime GPU inference (--cuda)

Fast WAV generation

Easy to switch models

🌐 FastAPI Backend

/ai-teacher POST endpoint

JSON response:

{ "text": "...", "audio_url": "/audio/xxxx.wav" }


CORS enabled

Static audio file serving

🖥️ Web Frontend

Clean chat UI

Typewriter-style text streaming

Animated waveform using WaveSurfer.js

Stop/Continue audio control

Dark/Light mode toggle

Clear conversation button

Auto-play audio for new responses

🗂️ Project Structure
AI_teacher_platform/
│
├── app.py                     # FastAPI backend
│
├── models/
│   └── piper/
│       ├── en_US-libritts-high.onnx
│       └── en_US-libritts-high.onnx.json
│
├── audio/                     # Generated WAV files
│
└── frontend/
    └── index.html             # Interactive web UI

🔧 Requirements
Linux (Ubuntu recommended)
Python 3.10+
GPU with CUDA
Conda environment
📦 1. Environment Setup
Create & activate environment
conda create -n ai_teacher python=3.10 -y
conda activate ai_teacher

Install backend dependencies
pip install fastapi uvicorn httpx pydantic
pip install onnxruntime-gpu

Install vLLM
pip install vllm

Install Piper

Follow their installation guide or use:

pip install piper-tts

Verify Piper GPU
echo "hello" | piper -m models/piper/en_US-libritts-high.onnx \
                     -c models/piper/en_US-libritts-high.onnx.json \
                     --cuda -f test.wav


You should hear test.wav.

⚙️ 2. Start vLLM Server

Run in a separate terminal:

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key EMPTY \
  --tensor-parallel-size 1


This exposes:

POST http://127.0.0.1:8000/v1/chat/completions

▶️ 3. Start FastAPI Backend

From the project root:

uvicorn app:app --host 0.0.0.0 --port 9000


Test:

curl -X POST http://127.0.0.1:9000/ai-teacher \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello", "with_audio":true}'


Expected response:

{
  "text": "Hello! ...",
  "audio_url": "/audio/xxxx.wav"
}


WAV file is stored in:

AI_teacher_platform/audio/

🌐 4. Start the Web Frontend
cd frontend
python3 -m http.server 8080


Then open:

http://localhost:8080/index.html


If accessing from a Mac connecting to a Linux server, use SSH tunnel:

ssh -L 8080:localhost:8080 -L 9000:localhost:9000 user@server-ip

🧠 Conversation Flow

User sends a message

Backend forwards it to vLLM

vLLM generates a text answer

Piper synthesizes voice into a WAV file

WAV saved → URL returned

Frontend displays text (typewriter streaming)

Audio automatically plays via WaveSurfer.js

User can Stop / Continue the audio

🖼️ Frontend Features
✔ Chat bubbles (user & teacher)
✔ Typing indicator
✔ Streaming text animation
✔ Waveform visualization
✔ Stop / Continue playback
✔ Auto-play new audio
✔ Conversation memory
✔ Clear chat button
✔ Dark / Light theme toggle
✔ Multi-voice selector (extendable)
🔈 Switching TTS Voices

Add more Piper models into:

models/piper/


Update:

<select id="voice-select">
  <option value="teacher_female">Female Teacher</option>
  <option value="teacher_male">Male Teacher</option>
  <option value="british_female">British Female</option>
</select>


Modify backend to load different models.

🧩 API Specification
POST /ai-teacher
Request
{
  "prompt": "Explain gravity",
  "with_audio": true,
  "voice": "teacher_female"
}

Response
{
  "text": "Gravity is a force that pulls objects toward each other...",
  "audio_url": "/audio/71b5b9e4be02.wav"
}

🛠️ Customization Guide
Backend

Change system prompt

Change voice model

Add streaming responses

Add multi-language support

Frontend

Modify CSS styles

Replace chat bubbles

Change typewriter speed

Add avatars or animations

📌 Common Issues & Fixes
❌ Frontend can't connect to backend

👉 Fix: Use SSH tunnels

ssh -L 8080:localhost:8080 -L 9000:localhost:9000 user@server

❌ Piper "failed to load model"

👉 Fix: Model path wrong — ensure:

models/piper/model.onnx
models/piper/model.onnx.json

❌ No waveform / no audio

👉 Fix: Browser blocked autoplay
User must interact first (click anything).

🔮 Future Improvements

Streaming TTS

Real-time animated mic input

Multiple personas / teacher modes

Better memory & context windows

Mobile UI optimization

Dockerized deployment

HTTPS reverse proxy (Nginx)

📄 License

MIT License — free for both personal and commercial use.

🙌 Contributors

Zesheng Hong
