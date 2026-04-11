# Backend Agent Plans — Overview

## Purpose
These plans are written for **autonomous Claude Code agents**.
Each plan is self-contained: an agent can open one file, read it completely,
and implement that phase without asking questions or needing additional context.

## How to Use
Open a plan in Claude Code and say:
> "Implement the plan in docs/agent-plans/NN-name.md"

The agent will read the plan, implement all code changes, and verify against
the acceptance criteria before marking complete.

## Plan Index

| Plan | Phase | What Gets Built | Blocking On |
|------|-------|-----------------|-------------|
| [01-foundation-verify.md](01-foundation-verify.md) | Foundation | Verify scaffold runs, fix any startup issues | Nothing |
| [02-stt-deepgram.md](02-stt-deepgram.md) | STT | Full Deepgram streaming integration | Plan 01 |
| [03-langgraph-nodes.md](03-langgraph-nodes.md) | Agent | LangGraph understand/navigate/respond nodes + WebSocket wiring | Plan 02 |
| [04-tts-elevenlabs.md](04-tts-elevenlabs.md) | TTS | ElevenLabs streaming audio back to client | Plan 03 |
| [05-interruption-pipeline.md](05-interruption-pipeline.md) | Interruption | asyncio task management + full E2E pipeline hardening | Plan 04 |

## Shared Context (read before any plan)

### Repository root
```
backend/
├── app/
│   ├── main.py          — FastAPI app entry point
│   ├── config.py        — pydantic-settings (OPENAI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY)
│   ├── api/
│   │   └── websocket.py — WebSocket session handler (main integration point)
│   ├── agent/
│   │   ├── graph.py     — LangGraph compiled graph (agent_graph)
│   │   ├── nodes.py     — LangGraph nodes (understand, navigate, respond)
│   │   ├── state.py     — AgentState TypedDict
│   │   └── prompts.py   — UNDERSTAND_SYSTEM, RESPOND_SYSTEM
│   ├── services/
│   │   ├── stt.py       — Deepgram client (stub → Plan 02)
│   │   ├── tts.py       — ElevenLabs client (stub → Plan 04)
│   │   └── llm.py       — OpenAI wrappers (complete)
│   └── slides/
│       └── content.py   — 6 static slides (complete, do not modify)
├── requirements.txt
└── .env.example
```

### Environment setup (run once before any plan)
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — fill OPENAI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY
```

### WebSocket protocol contract
```
Client → Server:
  {"type": "start"}
  {"type": "audio_chunk", "data": "<base64 PCM 16kHz mono int16>"}
  {"type": "interrupt"}
  {"type": "ping"}

Server → Client:
  {"type": "transcript",   "text": "...", "is_final": bool}
  {"type": "slide_change", "index": N,   "slide": {"title": "...", "bullets": [...]}}
  {"type": "agent_text",   "text": "..."}
  {"type": "tts_chunk",    "data": "<base64 MP3>"}
  {"type": "tts_done"}
  {"type": "error",        "message": "..."}
  {"type": "pong"}
```

### Key architectural decisions (already made)
- `run_agent` runs as an `asyncio.Task` — never awaited directly — so interrupt messages can arrive while TTS streams
- TTS synthesis happens OUTSIDE the LangGraph graph — `synthesize_stream` is called in `run_agent` after `agent_graph.ainvoke()` returns
- `interrupt_event: asyncio.Event` is shared between the receive loop and `run_agent`
- Deepgram reconnect is NOT implemented — session closes on Deepgram error
- User transcript is added to `messages` history before each `ainvoke`
