# Agent Plan 01 — Foundation Verification

## Agent Instructions
You are an autonomous agent. Read this plan completely before taking any action.
Do not ask the user questions. Make decisions using the plan's guidance.
Mark each task complete as you finish it.

---

## Goal
Verify the scaffold starts cleanly, all imports resolve, and the three HTTP +
WebSocket endpoints respond correctly. Fix any issues found — do NOT leave
broken imports or missing dependencies.

## Contract — What Must Be True Before Starting
- `backend/` directory exists with `app/`, `requirements.txt`, `.env.example`
- Python venv created and activated: `source venv/bin/activate`
- `.env` file exists with at least placeholder values (copy from `.env.example`)

## Tasks

### Task 1: Verify all imports resolve

Run:
```bash
cd backend
source venv/bin/activate
python -c "from app.main import app; print('imports OK')"
```

**Expected:** `imports OK`

**If it fails with `ModuleNotFoundError`:**
- `pydantic_settings` not found → `pip install pydantic-settings`
- `langgraph` not found → `pip install langgraph`
- `deepgram` not found → `pip install deepgram-sdk`
- `elevenlabs` not found → `pip install elevenlabs`
- `openai` not found → `pip install openai`
- Any other → `pip install -r requirements.txt` and retry

**If it fails with `ValidationError` from pydantic-settings:**
The `.env` file is missing required keys. Add placeholder values:
```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-placeholder
DEEPGRAM_API_KEY=placeholder
ELEVENLABS_API_KEY=placeholder
EOF
```

### Task 2: Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

**Expected startup output (contains all of these):**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     SynthioLabs Voice Slide Deck backend starting up
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Leave the server running in the background. If it fails, fix the error before proceeding.

**Common startup failures:**
- `Port 8000 already in use` → kill the process: `lsof -ti:8000 | xargs kill -9`
- `AttributeError` on graph compile → check `app/agent/graph.py` imports match `app/agent/nodes.py` function names

### Task 3: Verify `/health` endpoint

```bash
curl -s http://localhost:8000/health
```

**Expected:** `{"status":"ok"}`

### Task 4: Verify `/slides` endpoint

```bash
curl -s http://localhost:8000/slides | python -m json.tool | head -20
```

**Expected:** JSON array starting with:
```json
[
    {
        "index": 0,
        "title": "The Broken Machine: Clinical Trials Today",
        "bullets": [...]
    },
    ...
]
```

Must have exactly 6 items. If not, check `app/slides/content.py` — do not modify slide content.

### Task 5: Verify WebSocket `/ws` endpoint

Install wscat if needed: `npm install -g wscat`

```bash
wscat -c ws://localhost:8000/ws
```

**After connection, server should immediately send:**
```json
{"type": "slide_change", "index": 0, "slide": {"title": "The Broken Machine: Clinical Trials Today", "bullets": [...]}}
```

**Send a ping:**
```
> {"type": "ping"}
```

**Expected response:**
```json
{"type": "pong"}
```

**Send an interrupt:**
```
> {"type": "interrupt"}
```

**Expected response:**
```json
{"type": "tts_done"}
```

### Task 6: Check log output format

With the server running and a WebSocket connection made, logs should appear like:
```
2024-xx-xx HH:MM:SS  INFO      app.api.websocket  WebSocket session started
```

If the format is wrong, check `logging.basicConfig` in `app/main.py`.

## Acceptance Criteria

All of the following must be true before this plan is complete:

- [ ] `python -c "from app.main import app"` exits 0
- [ ] `uvicorn app.main:app --reload --port 8000` starts without errors
- [ ] `GET /health` returns `{"status":"ok"}`
- [ ] `GET /slides` returns array of exactly 6 slides
- [ ] `WS /ws` sends `slide_change` on connect, responds to `ping` → `pong`
- [ ] No Python warnings or deprecation notices in startup logs

## Files That May Need Fixing

Only fix files that actually have errors. Do not make speculative changes.

| File | Likely issue |
|------|-------------|
| `app/config.py` | pydantic-settings field validators |
| `app/agent/graph.py` | Import path for nodes |
| `app/agent/nodes.py` | Missing imports |
| `requirements.txt` | Missing/wrong package versions |

## Do NOT Change
- `app/slides/content.py` — slide content is final
- `app/agent/state.py` — state schema is final
- `app/agent/prompts.py` — prompts are final
