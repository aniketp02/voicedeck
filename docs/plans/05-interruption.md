# Plan 05 — Interruption: Stop TTS When User Speaks

## Goal
When the user starts speaking while the AI is talking, the TTS stream
stops immediately and the system starts listening to the user.

**Success criterion:** Say anything while the AI is speaking → AI stops
mid-sentence within ~200ms, system starts transcribing the new input.

## Prerequisite
Plans 01–04 must be complete (full pipeline working end to end).

## How It Works

### Two-path interruption
The system supports interruption via two mechanisms:
1. **Explicit**: Client sends `{"type": "interrupt"}` when it detects voice activity
2. **Implicit**: STT detects speech while TTS is active (server-side)

For the MVP, we rely on **explicit** interruption from the client.
The frontend detects voice activity via the Web Audio API and sends the signal.

### Server-side mechanism
```
interrupt_event = asyncio.Event()    # shared between STT and TTS coroutines

# In synthesize_stream (tts.py):
  for each chunk:
    if interrupt_event.is_set(): stop and return

# In WebSocket handler:
  on {"type": "interrupt"}:
    interrupt_event.set()
    → synthesize_stream exits early
    → send tts_done to client
    interrupt_event.clear()
    → system is now listening again
```

### State machine diagram
```
IDLE ──────────────────── user speaks ──────────────────→ LISTENING
  ↑                                                           │
  │                                               transcript final
  │                                                           │
  └── tts_done ←────── SPEAKING ←── agent responds ─────────┘
                           │
                       user speaks
                           │
                    interrupt_event.set()
                           │
                    synthesize stops early
                           │
                    tts_done sent
                           │
                  interrupt_event.clear()
                           │
                         LISTENING (again)
```

## Files to Modify

### `app/api/websocket.py` — Make `run_agent` interruptible

The key change: `run_agent` must NOT block the WebSocket receive loop.
Run it as a background task so `interrupt` messages can still be received
while TTS is streaming.

Refactor `handle_session` to use a task for agent execution:

```python
agent_task: asyncio.Task | None = None

async def on_transcript(result: TranscriptResult) -> None:
    nonlocal agent_task
    await _send(websocket, {
        "type": "transcript",
        "text": result.text,
        "is_final": result.is_final,
    })
    if result.is_final and result.text.strip():
        # Cancel any running agent task (previous utterance)
        if agent_task and not agent_task.done():
            interrupt_event.set()
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            interrupt_event.clear()

        agent_task = asyncio.create_task(
            run_agent(websocket, state, result.text, interrupt_event)
        )

# In the receive loop, handle interrupt:
elif msg_type == "interrupt":
    interrupt_event.set()
    if agent_task and not agent_task.done():
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
    await _send(websocket, {"type": "tts_done"})
    interrupt_event.clear()
```

### `app/services/tts.py` — Already interrupt-aware ✅
The `synthesize_stream` generator already checks `interrupt_event.is_set()`
between chunks. No changes needed.

## Frontend Responsibilities (see frontend Plan 04)
The frontend must:
1. Detect voice activity using Web Audio API `AnalyserNode`
2. When user starts speaking AND TTS is playing, send `{"type": "interrupt"}`
3. Stop playing audio immediately (clear the audio queue)
4. Show a visual indicator that the AI was interrupted

Simple VAD threshold check:
```javascript
// In AudioWorklet or analyser loop:
if (rmsLevel > VAD_THRESHOLD && isTTSPlaying) {
  ws.send(JSON.stringify({ type: 'interrupt' }))
  stopAudioPlayback()
}
```

## Race Conditions to Handle
1. **Transcript arrives while agent is running**: Cancel current agent task, start new one
2. **interrupt arrives after TTS finishes**: interrupt_event.set() is a no-op after tts_done
3. **Multiple rapid interrupts**: Always clear interrupt_event after handling

## Verification
1. Start the full stack (backend + frontend)
2. Ask the AI a question → wait for TTS to start
3. Start speaking mid-sentence → AI should stop within 200ms
4. Your new speech should be transcribed and processed
5. Check logs for: `TTS interrupted after N chunks`

## Tuning
- If interruption is too sensitive: Add a debounce on the frontend VAD
- If interruption is too slow: Check that `interrupt_event` is truly shared (pass by reference, not copy)
- If agent_task cancel doesn't work: Ensure `run_agent` uses `asyncio.shield` only for DB writes, not for TTS
