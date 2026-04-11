# Plan 01 — Foundation: FastAPI Skeleton + WebSocket Echo

## Goal
Get a running FastAPI server with:
- `GET /health` → `{"status": "ok"}`
- `GET /slides` → list of all 6 slides (title + bullets)
- `WS /ws` → accepts connection, sends initial slide, echoes ping/pong

**Success criterion:** `uvicorn app.main:app --reload` starts without errors;
WebSocket connects and responds to ping.

## Status
The scaffold files already exist. This plan describes what's already done
and what to verify/test.

## What's Already Implemented
All files are scaffolded. No TODOs for this plan.

| File | Status |
|------|--------|
| `app/main.py` | ✅ Complete — FastAPI app, CORS, /health, /slides, /ws |
| `app/config.py` | ✅ Complete — pydantic-settings, all env vars |
| `app/slides/content.py` | ✅ Complete — 6 slides on AI in Clinical Trials |
| `app/agent/state.py` | ✅ Complete — AgentState TypedDict |
| `app/agent/graph.py` | ✅ Complete — LangGraph graph structure (nodes are stubs) |
| `app/agent/prompts.py` | ✅ Complete — UNDERSTAND_SYSTEM, RESPOND_SYSTEM |
| `app/api/websocket.py` | ✅ Scaffold — placeholder echo loop, full pipeline in Plans 02-05 |

## Verification Steps

### 1. Install and start
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in at minimum OPENAI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY
uvicorn app.main:app --reload --port 8000
```

### 2. Test health endpoint
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

### 3. Test slides endpoint
```bash
curl http://localhost:8000/slides | python -m json.tool
# Expected: array of 6 objects with index, title, bullets
```

### 4. Test WebSocket connection
Use a WebSocket client (e.g. wscat or browser console):
```bash
npx wscat -c ws://localhost:8000/ws
# After connect, server sends: {"type":"slide_change","index":0,"slide":{...}}
# Send: {"type":"ping"}
# Receive: {"type":"pong"}
```

## Notes for Cursor
- Do NOT modify `app/slides/content.py` — the slide content is final.
- Do NOT modify `app/config.py` — settings are complete.
- The placeholder WebSocket loop in `app/api/websocket.py` will be replaced in Plans 02-05.
- `app/agent/nodes.py` has stub implementations — those are filled in Plan 03.
