# SynthioLabs Voice Slide Deck — Backend Overview

## What We're Building
A FastAPI backend for a real-time voice-activated presentation system.
A user speaks into the browser mic; the AI narrates slides, answers questions,
navigates to relevant slides automatically, and can be interrupted mid-sentence.

## Tech Stack
| Component | Library | Purpose |
|-----------|---------|---------|
| Web framework | FastAPI + uvicorn | HTTP + WebSocket server |
| Orchestration | LangGraph | Agentic state machine |
| STT | Deepgram SDK (nova-2) | Speech → text, streaming |
| TTS | ElevenLabs SDK (eleven_turbo_v2) | Text → speech, streaming |
| LLM | OpenAI (gpt-4o-mini) | Intent detection + response generation |
| Config | pydantic-settings | Env var management |

## Repository Layout
```
backend/
├── app/
│   ├── main.py              # FastAPI app + /ws WebSocket endpoint
│   ├── config.py            # Settings (API keys, model names)
│   ├── api/
│   │   └── websocket.py     # WebSocket session handler
│   ├── agent/
│   │   ├── graph.py         # LangGraph state machine (compiled)
│   │   ├── nodes.py         # understand, navigate, respond nodes
│   │   ├── state.py         # AgentState TypedDict
│   │   └── prompts.py       # LLM system prompts
│   ├── services/
│   │   ├── stt.py           # Deepgram streaming client
│   │   ├── tts.py           # ElevenLabs streaming client
│   │   └── llm.py           # OpenAI wrapper
│   └── slides/
│       └── content.py       # 6 static slide definitions
├── tests/
├── docs/plans/              # ← You are here
├── requirements.txt
└── .env.example
```

## Build Order
| Plan | Feature | Depends On |
|------|---------|------------|
| [01-foundation](01-foundation.md) | FastAPI skeleton, WebSocket echo, slides REST | — |
| [02-stt-deepgram](02-stt-deepgram.md) | Deepgram STT streaming | 01 |
| [03-langgraph-agent](03-langgraph-agent.md) | LangGraph understand/navigate/respond | 02 |
| [04-tts-elevenlabs](04-tts-elevenlabs.md) | ElevenLabs TTS streaming | 03 |
| [05-interruption](05-interruption.md) | Interrupt + resume pipeline | 04 |

## Running Locally
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
uvicorn app.main:app --reload --port 8000
```

## Environment Variables (required)
```
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
```

## WebSocket Protocol (summary)
**Client → Server**
```json
{"type": "start"}
{"type": "audio_chunk", "data": "<base64 raw PCM 16kHz mono int16>"}
{"type": "interrupt"}
```
**Server → Client**
```json
{"type": "transcript",   "text": "...", "is_final": true}
{"type": "slide_change", "index": 2, "slide": {"title": "...", "bullets": [...]}}
{"type": "agent_text",   "text": "..."}
{"type": "tts_chunk",    "data": "<base64 MP3>"}
{"type": "tts_done"}
{"type": "error",        "message": "..."}
```
