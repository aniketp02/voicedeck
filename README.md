# VoiceDeck — AI Voice-Activated Slide Presenter

A real-time voice-interactive presentation system. Ask questions, interrupt, navigate slides by speaking — the AI responds contextually and maintains conversation memory across the session.

> **Demo video:** _coming soon_  
> **Hosted demo:** _coming soon_

---

## What it does

- **Voice Q&A** — ask questions about any slide; the agent navigates to the relevant slide and responds with expert context
- **Barge-in interrupts** — speak while the AI is talking to redirect it instantly
- **Auto-present mode** — fully autonomous narration across all slides with natural slide transitions
- **Conversation memory** — the agent references and builds on what was said earlier in the session
- **Multi-deck** — ships with two presentation decks (AI in Clinical Trials, AI in Drug Discovery)

**Stack:** FastAPI + LangGraph (backend) · React 19 + TypeScript (frontend) · Deepgram STT · OpenAI LLM + TTS · ElevenLabs TTS (optional)

---

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+
- API keys: **OpenAI** (required), **Deepgram** (required), **ElevenLabs** (optional)

### 1. Clone and configure

This repository is the **backend**. Clone it next to the **voicedeck-web** frontend (same parent folder is convenient).

```bash
git clone https://github.com/aniketp02/voicedeck.git
git clone https://github.com/aniketp02/voicedeck-web.git
```

Or use SSH:

```bash
git clone git@github.com:aniketp02/voicedeck.git
git clone git@github.com:aniketp02/voicedeck-web.git
```

You should have:

```text
<parent>/
├── voicedeck/       # this repo — FastAPI backend
└── voicedeck-web/   # React + Vite SPA
```

### 2. Backend setup

```bash
cd voicedeck

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — fill in OPENAI_API_KEY, DEEPGRAM_API_KEY
```

Minimal `.env` for getting started:

```env
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
TTS_PROVIDER=openai          # or "deepgram" or "elevenlabs"
```

### 3. Frontend setup

```bash
cd ../voicedeck-web
npm install
```

### 4. Run

Open two terminals:

```bash
# Terminal 1 — backend (from voicedeck/)
source venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend (from voicedeck-web/)
npm run dev
```

Open **http://localhost:5173** in your browser. Allow microphone access when prompted.

---

## Usage

1. **Select a presentation** on the landing page
2. **Click the mic button** to start capturing — the orb turns green when you're heard
3. **Ask anything** — *"Tell me about patient recruitment"*, *"What's the biggest challenge here?"*
4. **Interrupt** mid-response by speaking — the agent pivots to your new question
5. **Navigate** with arrow keys or by asking — *"Go to slide 3"*, *"Take me back to the overview"*
6. **Auto Present** — click the play button in the header to let the AI narrate all slides automatically
7. **Wrap up** naturally — *"Great, I think that covers it"* triggers a graceful sign-off

---

## TTS providers

| Provider | Quality | Setup |
|---|---|---|
| `openai` | Natural, clear | `OPENAI_API_KEY` (same key, or set `OPENAI_TTS_API_KEY`) |
| `elevenlabs` | Best — most expressive | `ELEVENLABS_API_KEY` required |
| `deepgram` | Good, fastest TTFB | `DEEPGRAM_API_KEY` (same key as STT) |

Set `TTS_PROVIDER=` in `.env` to switch.

> **Restricted OpenAI keys:** TTS requires the `api.model.audio.request` scope. If your key is chat-only, set `OPENAI_TTS_API_KEY` to an unrestricted key.

---

## Configuration reference

Key settings in `.env` (see `.env.example` for full list):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4.1` | LLM for responses (full model = best quality) |
| `OPENAI_UNDERSTAND_MODEL` | `gpt-4.1-nano` | Intent parsing (fast, cheap) |
| `TTS_PROVIDER` | `elevenlabs` | `elevenlabs` / `openai` / `deepgram` |
| `DEEPGRAM_UTTERANCE_END_MS` | `2000` | Silence before STT fires (ms) |
| `OPENAI_TTS_VOICE` | `nova` | OpenAI voice — `nova`, `alloy`, `shimmer`, etc. |
| `ELEVENLABS_VOICE_ID` | `JBFqnCBsd6RMkjVDRZzb` | ElevenLabs voice ID |

---

## Running tests

```bash
cd voicedeck
source venv/bin/activate
pytest tests/ -v
```

77 tests, no API keys required (all external calls are mocked).

---

## Project structure

```
<root-folder>/
├── voicedeck/            # FastAPI + LangGraph
│   ├── app/
│   │   ├── agent/      # LangGraph state machine (nodes, prompts, narrate)
│   │   ├── api/        # WebSocket session handler
│   │   ├── services/   # STT (Deepgram), TTS (3 providers), LLM (OpenAI)
│   │   └── slides/     # Presentation content + registry
│   └── tests/
├── voicedeck-web/           # React 19 + TypeScript + Vite
    └── src/
        ├── hooks/      # useWebSocket, useAudioCapture, useAudioPlayer, useVoiceState
        └── components/ # SlideView, OrbPanel, Header, etc.
```

---

## Technical documentation

Full architecture, sequence diagrams, and API reference in [`voicedeck-docs/technical/`](voicedeck-docs/technical/):

- [`backend.md`](voicedeck-docs/technical/backend.md) — Agent graph, services, WebSocket handler
- [`frontend.md`](voicedeck-docs/technical/frontend.md) — Hook architecture, MSE pipeline, VAD
- [`protocol.md`](voicedeck-docs/technical/protocol.md) — WebSocket message protocol, audio format specs
- [`sequences.md`](voicedeck-docs/technical/sequences.md) — Mermaid sequence diagrams for all flows
