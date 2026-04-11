# Agent Plan 05 — Interruption + Full Pipeline Hardening

## Agent Instructions
You are an autonomous agent. Read this plan completely before taking any action.
Do not ask the user questions. All decisions are made for you.
Implement every task in order. Verify each task before proceeding.
This is the final backend plan — by the end, the full E2E pipeline must work.

---

## Goal
Harden the full pipeline so that:
1. User can interrupt the AI mid-sentence and get a response to the new utterance
2. Rapid speech (multiple final transcripts in quick succession) is handled safely
3. Concurrent task management is bulletproof (no hanging tasks, no hung client)
4. The full E2E flow is tested and verified

## Contract — What Plan 04 Delivered
- `app/services/tts.py`: full ElevenLabs streaming, checks `interrupt_event`
- `app/api/websocket.py`: `run_agent` streams TTS chunks + always sends `tts_done`
- The full pipeline works end-to-end: speech → transcript → agent → TTS → audio
- Interruption is partially working (interrupt_event is set) but task cancellation
  is not yet bulletproof under rapid-speech or concurrent transcript scenarios

## Background: The Concurrency Problem

```
Timeline without this plan (broken):

t=0  User speaks "tell me about recruitment"
t=1  Final transcript arrives → run_agent Task A starts
t=2  Task A: LLM call in progress (waiting)
t=3  User speaks again "no wait, what about FDA"    ← second transcript
t=3  on_transcript cancels Task A (correct)
t=3  Task A: CancelledError in LLM await
t=3  Task A finally block: tries to send tts_done
t=3  Task B starts: run_agent for "what about FDA"
t=4  Both Task A finally AND Task B send tts_done  ← race condition → client confused

Timeline with this plan (correct):

t=3  on_transcript: interrupt_event.set() → cancel Task A → await cancellation
t=3  Task A: CancelledError → finally block sends tts_done → Task A fully done
t=3  interrupt_event.clear()
t=3  Task B starts fresh
t=4  Task B: full pipeline completes → tts_done
```

The fix: `on_transcript` must `await` the cancellation of Task A (not fire-and-forget),
and `interrupt_event` must be cleared before Task B starts.

---

## Task 1: Audit and fix `on_transcript` in `app/api/websocket.py`

Read the current `on_transcript` function. It should already be cancelling
`agent_task` when a new final transcript arrives. Verify the pattern is correct:

### The required pattern (replace if different)

```python
async def on_transcript(result: TranscriptResult) -> None:
    nonlocal agent_task

    # Forward all transcripts to client for live display
    await _send(websocket, {
        "type": "transcript",
        "text": result.text,
        "is_final": result.is_final,
    })

    # Only trigger agent on final, non-empty transcripts
    if not result.is_final or not result.text.strip():
        return

    logger.info(
        "Final transcript: %r (confidence=%.2f)", result.text, result.confidence
    )

    # Cancel any in-flight agent task and fully await its shutdown
    # This prevents double-send of tts_done from two concurrent tasks
    if agent_task and not agent_task.done():
        logger.info("Cancelling previous agent task for new utterance")
        interrupt_event.set()
        agent_task.cancel()
        try:
            await agent_task   # wait for the cancelled task to fully clean up
        except (asyncio.CancelledError, Exception):
            pass
        interrupt_event.clear()  # clear BEFORE starting new task

    # Start new agent task
    agent_task = asyncio.create_task(
        run_agent(websocket, state, result.text, interrupt_event)
    )
```

**Key requirements (verify each):**
1. `await agent_task` — must be awaited, not fire-and-forget
2. `interrupt_event.clear()` — must happen AFTER await, BEFORE new task
3. `except (asyncio.CancelledError, Exception): pass` — swallow both

### Verify Task 1
Read `websocket.py` and confirm `on_transcript` follows this pattern.
If it doesn't, update it to match exactly.

---

## Task 2: Audit and fix the `interrupt` message handler in the receive loop

Find the `elif msg_type == "interrupt":` block. It must follow this pattern:

```python
elif msg_type == "interrupt":
    logger.info("Interrupt signal received from client")
    interrupt_event.set()
    if agent_task and not agent_task.done():
        agent_task.cancel()
        try:
            await agent_task  # must await — same reason as on_transcript
        except (asyncio.CancelledError, Exception):
            pass
    # Only send tts_done if agent task was the one holding it
    # (run_agent's finally block sends it on cancel — but if no task was running,
    # we send it here to unblock the client)
    if not agent_task or agent_task.done():
        await _send(websocket, {"type": "tts_done"})
    interrupt_event.clear()
```

**Wait — this has a subtle bug.** If `run_agent`'s finally block sends `tts_done`
on cancel AND we send it here, the client gets two `tts_done`. Fix:

```python
elif msg_type == "interrupt":
    logger.info("Interrupt signal received from client")
    interrupt_event.set()
    task_was_running = bool(agent_task and not agent_task.done())
    if task_was_running:
        agent_task.cancel()
        try:
            await agent_task  # run_agent finally block sends tts_done here
        except (asyncio.CancelledError, Exception):
            pass
    else:
        # No task running — client may be stuck, send tts_done to unblock
        await _send(websocket, {"type": "tts_done"})
    interrupt_event.clear()
```

This ensures exactly one `tts_done` per interrupt.

---

## Task 3: Add connection-level error recovery

The WebSocket receive loop must handle `WebSocketDisconnect` and unexpected
errors gracefully. Verify the existing `try/except` in `handle_session` covers:

1. `WebSocketDisconnect` → log + break (already in finally)
2. `json.JSONDecodeError` on malformed message → log warning, continue loop
3. General `Exception` → send error message, close session

Update the receive loop to handle malformed JSON:

```python
try:
    async for raw in _receive_loop(websocket):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Malformed WebSocket message (not JSON): %r", raw[:100])
            continue

        msg_type = msg.get("type")

        if msg_type == "audio_chunk":
            # ... existing handler
        elif msg_type == "interrupt":
            # ... updated handler from Task 2
        elif msg_type == "ping":
            await _send(websocket, {"type": "pong"})
        elif msg_type == "start":
            pass
        else:
            logger.debug("Unknown message type: %r", msg_type)

except WebSocketDisconnect:
    logger.info("WebSocket session ended by client")
except Exception as e:
    logger.exception("Unexpected session error: %s", e)
    try:
        await _send(websocket, {"type": "error", "message": str(e)})
    except Exception:
        pass
```

---

## Task 4: Final `websocket.py` — complete file review

After Tasks 1-3, read the entire `app/api/websocket.py` and verify:

**Checklist:**
- [ ] `handle_session` initializes all AgentState fields
- [ ] `on_transcript` awaits task cancellation before starting new task
- [ ] `interrupt_event.clear()` happens after task cancellation in both paths
- [ ] `run_agent` has `tts_done_sent` flag in finally block
- [ ] `run_agent` re-raises `CancelledError` after finally block
- [ ] `synthesize_stream` is inside the `try` block of `run_agent` (not finally)
- [ ] `_send` silently swallows send errors (client may have disconnected)
- [ ] `stt_task` is cancelled in `handle_session`'s finally block with timeout
- [ ] Malformed JSON messages are caught and logged (don't crash the loop)

If any of these are wrong, fix them before proceeding.

---

## Task 5: Full E2E integration test

This is the definitive test. Run through the complete scenario.

### Setup
```bash
# Terminal 1: backend
cd backend && source venv/bin/activate
uvicorn app.main:app --port 8000

# Terminal 2: frontend (if available)
cd frontend && npm run dev
```

### Scenario A: Normal conversation
1. Open http://localhost:5173
2. Click "Start Presentation"
3. Click mic → grant permission
4. Say: **"Tell me about the problem with clinical trials"**

Expected sequence in backend logs:
```
INFO  Deepgram STT connection opened
DEBUG Deepgram transcript: is_final=False ...
INFO  Final transcript: 'Tell me about the problem with clinical trials'
INFO  understand_node: navigate=False ... (already on slide 0)
INFO  respond_node: slide=0 generated NNN chars
INFO  Sent agent_text: NNN chars
INFO  TTS complete: N chunks
INFO  Streamed N TTS chunks to client
```

Expected WebSocket messages to client:
- `transcript` (interim, multiple)
- `transcript` (final)
- `agent_text` (with real response text)
- `tts_chunk` × N
- `tts_done`

### Scenario B: Navigation
5. While on slide 0, say: **"Tell me about patient recruitment"**

Expected: `navigate_node: slide 0 → 1` in logs, `slide_change` to client.

### Scenario C: Interruption
6. Ask a question, wait for TTS to start playing
7. Start speaking mid-TTS

Expected backend logs:
```
INFO  Cancelling previous agent task for new utterance
INFO  TTS interrupted after N chunks
INFO  run_agent cancelled (interrupt or new transcript)
INFO  Final transcript: 'your new question'
INFO  understand_node: ...
```

Client must NOT receive double `tts_done`.

### Scenario D: Rapid speech
8. Quickly say two things in succession before the first TTS finishes

Expected: Only the second utterance produces a full agent response.
No errors in logs. No hung client state.

### Scenario E: Disconnect and reconnect
9. Close browser tab → reopen → start again

Expected: Clean session on reconnect. No zombie tasks from previous session.

---

## Acceptance Criteria

All of the following must be verified before this plan is complete:

- [ ] Scenario A: Normal conversation works end-to-end (speech → audio response)
- [ ] Scenario B: Navigation works (slide changes on relevant question)
- [ ] Scenario C: Interruption stops TTS, new utterance processed correctly
- [ ] Scenario D: Rapid speech handled without errors or double tts_done
- [ ] Scenario E: Reconnect works cleanly
- [ ] No `asyncio.CancelledError` in logs during normal operation
- [ ] No uncaught exceptions in logs during any scenario
- [ ] `tts_done` sent exactly once per agent invocation
- [ ] `interrupt_event` is always cleared after use

## Full `websocket.py` Reference

After all tasks are complete, `app/api/websocket.py` should have this structure:

```
Imports: asyncio, base64, json, logging, fastapi, langchain_core.messages,
         app.agent.graph, app.agent.state, app.services.stt,
         app.services.tts, app.slides.content

Functions (in order):
  1. handle_session(websocket)    — main session coroutine
     ├── on_transcript(result)    — nested async callback
  2. run_agent(ws, state, transcript, interrupt_event)
  3. _receive_loop(websocket)     — async generator
  4. _send(websocket, payload)    — fire-and-forget JSON send
```

Total file length: ~180-220 lines. If it's longer, look for duplication.

## Performance Targets (for reference)

| Metric | Target | How to measure |
|--------|--------|----------------|
| Time to first transcript | < 500ms from speaking | Deepgram log timestamp |
| Time to agent_text | < 1.5s from final transcript | Log timestamps |
| Time to first tts_chunk | < 2.5s from speaking | Log timestamps |
| TTS interruption latency | < 300ms | Interrupt log + tts_done timing |

If targets are not met, check network latency to Deepgram/OpenAI/ElevenLabs
servers. US East coast servers have lowest latency for all three services.
