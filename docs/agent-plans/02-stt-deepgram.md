# Agent Plan 02 — Deepgram STT Streaming

## Agent Instructions
You are an autonomous agent. Read this plan completely before taking any action.
Do not ask the user questions. All decisions are made for you.
Implement every task in order. Run verification after each task.

---

## Goal
Replace the stub in `app/services/stt.py` with a working Deepgram streaming
integration. Update `app/api/websocket.py` to pipe browser audio chunks through
Deepgram and fire a callback with each final transcript.

## Contract — What Plan 01 Delivered
- Server starts cleanly at `http://localhost:8000`
- `/ws` WebSocket connects and responds to ping/pong
- `DEEPGRAM_API_KEY` is set in `.env` (real key, not placeholder)

## Background: How Deepgram Streaming Works

```
Browser mic → base64 PCM → WebSocket msg {"type":"audio_chunk","data":"..."}
    ↓
websocket.py receives msg → decodes base64 → puts bytes in audio_queue
    ↓
transcribe_stream() (background task) reads from audio_queue → sends to Deepgram
    ↓
Deepgram fires on_message callback → on_transcript(TranscriptResult) called
    ↓
on_transcript:
  - sends {"type":"transcript",...} to client (all results, for live display)
  - if is_final=True AND text non-empty → triggers run_agent() [Plan 03 wires this]
```

Deepgram SDK v3 uses an event-driven async WebSocket connection.
Key classes: `DeepgramClient`, `LiveOptions`, `LiveTranscriptionEvents`.

---

## Task 1: Implement `app/services/stt.py`

Replace the entire file with:

```python
import asyncio
import logging
from dataclasses import dataclass

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    text: str
    is_final: bool
    confidence: float = 0.0


async def transcribe_stream(
    audio_queue: asyncio.Queue,
    on_transcript,  # async callable(TranscriptResult)
) -> None:
    """
    Read audio bytes from audio_queue, stream to Deepgram, call on_transcript
    for each result. Stops when audio_queue yields None (sentinel value).

    Audio format expected: linear16, 16000 Hz, mono (from browser AudioContext).
    """
    config = DeepgramClientOptions(options={"keepalive": "true"})
    client = DeepgramClient(settings.deepgram_api_key, config)

    live_options = LiveOptions(
        model=settings.deepgram_model,
        language=settings.deepgram_language,
        smart_format=True,
        interim_results=True,
        utterance_end_ms=1000,
        vad_events=True,
        encoding="linear16",
        sample_rate=16000,
        channels=1,
    )

    connection = client.listen.asynclive(live_options)

    async def _on_message(self, result, **kwargs):
        try:
            alternatives = result.channel.alternatives
            if not alternatives:
                return
            sentence = alternatives[0].transcript
            if not sentence:
                return
            confidence = alternatives[0].confidence
            is_final = result.is_final
            logger.debug(
                "Deepgram transcript: is_final=%s confidence=%.2f text=%r",
                is_final, confidence, sentence,
            )
            await on_transcript(TranscriptResult(
                text=sentence,
                is_final=is_final,
                confidence=float(confidence),
            ))
        except Exception as e:
            logger.error("Error in Deepgram on_message callback: %s", e)

    async def _on_error(self, error, **kwargs):
        logger.error("Deepgram error: %s", error)

    async def _on_close(self, close, **kwargs):
        logger.info("Deepgram connection closed: %s", close)

    connection.on(LiveTranscriptionEvents.Transcript, _on_message)
    connection.on(LiveTranscriptionEvents.Error, _on_error)
    connection.on(LiveTranscriptionEvents.Close, _on_close)

    started = await connection.start(live_options)
    if not started:
        raise RuntimeError("Failed to open Deepgram live transcription connection")
    logger.info("Deepgram STT connection opened (model=%s)", settings.deepgram_model)

    try:
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                logger.info("Deepgram STT received sentinel — closing connection")
                break
            await connection.send(chunk)
    except asyncio.CancelledError:
        logger.info("Deepgram STT task cancelled")
    finally:
        await connection.finish()
        logger.info("Deepgram STT connection finished")
```

### Verify Task 1
```bash
python -c "from app.services.stt import transcribe_stream, TranscriptResult; print('stt imports OK')"
```
Expected: `stt imports OK`

---

## Task 2: Implement `app/api/websocket.py`

Replace the entire file with the following. Read carefully — this is the full
session handler integrating STT. The `run_agent` call is a stub for now
(implemented in Plan 03).

```python
import asyncio
import base64
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from app.agent.state import AgentState
from app.services.stt import transcribe_stream, TranscriptResult
from app.slides.content import get_slide, SLIDES

logger = logging.getLogger(__name__)


async def handle_session(websocket: WebSocket) -> None:
    """
    Main WebSocket session lifecycle.

    Pipeline:
    1. Accept connection
    2. Send initial slide_change for slide 0
    3. Start Deepgram STT as background task
    4. Receive loop: route audio_chunk → audio_queue, interrupt → interrupt_event
    5. on_transcript callback: forward to client; on final → run_agent (Plan 03)
    6. Graceful shutdown on disconnect or error
    """
    await websocket.accept()
    logger.info("WebSocket session started")

    # Shared state for this session
    state: AgentState = {
        "current_slide": 0,
        "target_slide": None,
        "messages": [],
        "transcript": "",
        "response_text": "",
        "slide_changed": False,
        "interrupted": False,
        "should_navigate": False,
    }

    interrupt_event = asyncio.Event()
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=100)

    # Track the agent task so it can be cancelled on interrupt
    agent_task: asyncio.Task | None = None

    # Send initial slide
    initial_slide = get_slide(0)
    await _send(websocket, {
        "type": "slide_change",
        "index": 0,
        "slide": {"title": initial_slide.title, "bullets": initial_slide.bullets},
    })

    async def on_transcript(result: TranscriptResult) -> None:
        nonlocal agent_task

        # Always forward transcript to client for live display
        await _send(websocket, {
            "type": "transcript",
            "text": result.text,
            "is_final": result.is_final,
        })

        # Only trigger agent on final, non-empty transcripts
        if not result.is_final or not result.text.strip():
            return

        logger.info("Final transcript: %r (confidence=%.2f)", result.text, result.confidence)

        # Cancel any in-flight agent task before starting a new one
        if agent_task and not agent_task.done():
            logger.info("Cancelling previous agent task for new transcript")
            interrupt_event.set()
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            interrupt_event.clear()

        # Plan 03 will replace this stub with the real run_agent call
        agent_task = asyncio.create_task(
            _agent_stub(websocket, state, result.text, interrupt_event)
        )

    # Start STT background task
    stt_task = asyncio.create_task(
        transcribe_stream(audio_queue, on_transcript)
    )

    try:
        async for raw in _receive_loop(websocket):
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio_chunk":
                audio_bytes = base64.b64decode(msg["data"])
                try:
                    audio_queue.put_nowait(audio_bytes)
                except asyncio.QueueFull:
                    logger.warning("Audio queue full — dropping chunk")

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

            elif msg_type == "ping":
                await _send(websocket, {"type": "pong"})

            elif msg_type == "start":
                pass  # Already handled by initial slide send on connect

    except WebSocketDisconnect:
        logger.info("WebSocket session ended by client")
    except Exception as e:
        logger.exception("Unexpected session error: %s", e)
        try:
            await _send(websocket, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Shutdown STT
        await audio_queue.put(None)
        stt_task.cancel()
        try:
            await asyncio.wait_for(stt_task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        # Shutdown agent task
        if agent_task and not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass

        logger.info("WebSocket session cleaned up")


async def _receive_loop(websocket: WebSocket):
    """Yield raw text messages until disconnect."""
    while True:
        yield await websocket.receive_text()


async def _agent_stub(
    websocket: WebSocket,
    state: AgentState,
    transcript: str,
    interrupt_event: asyncio.Event,
) -> None:
    """
    Placeholder agent task — replaced in Plan 03.
    Echoes the transcript back as agent_text so the frontend can be tested.
    """
    logger.info("Agent stub called with transcript: %r", transcript)
    await _send(websocket, {
        "type": "agent_text",
        "text": f"[stub] You said: {transcript}",
    })
    await _send(websocket, {"type": "tts_done"})


async def _send(websocket: WebSocket, payload: dict) -> None:
    """Send JSON message to client. Silently drops if connection is closed."""
    try:
        await websocket.send_text(json.dumps(payload))
    except Exception:
        pass
```

### Verify Task 2
```bash
python -c "from app.api.websocket import handle_session; print('websocket imports OK')"
```
Expected: `websocket imports OK`

---

## Task 3: Integration test — STT pipeline

### Step 1: Start the server
```bash
uvicorn app.main:app --reload --port 8000
```

### Step 2: Generate a test audio chunk
Run this Python snippet to generate a base64 PCM chunk (1 second of 440Hz tone):
```python
import base64, struct, math, json

# 16kHz mono int16 — 1 second of 440Hz sine wave
samples = [int(32767 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(16000)]
pcm = struct.pack(f"<{len(samples)}h", *samples)
b64 = base64.b64encode(pcm).decode()
msg = json.dumps({"type": "audio_chunk", "data": b64})
print(msg[:100], "...")
print(f"\nFull message length: {len(msg)} chars")
```

### Step 3: Connect and send audio via wscat
```bash
wscat -c ws://localhost:8000/ws
```
After connecting, paste the audio_chunk JSON message.

### Step 4: Verify via real microphone (primary test)
Use the frontend (if available) or a browser WebSocket test page to stream
real microphone audio. Watch the backend logs for:
```
INFO  app.services.stt  Deepgram STT connection opened (model=nova-2)
DEBUG app.services.stt  Deepgram transcript: is_final=False ...
DEBUG app.services.stt  Deepgram transcript: is_final=True confidence=0.99 text="hello world"
INFO  app.api.websocket  Final transcript: 'hello world' (confidence=0.99)
```

WebSocket client should receive:
```json
{"type": "transcript", "text": "hello", "is_final": false}
{"type": "transcript", "text": "hello world", "is_final": true}
{"type": "agent_text", "text": "[stub] You said: hello world"}
{"type": "tts_done"}
```

---

## Acceptance Criteria

- [ ] `app/services/stt.py` has no `NotImplementedError` — full implementation
- [ ] `app/api/websocket.py` runs STT as background task
- [ ] Server starts without errors with real API key in `.env`
- [ ] Deepgram connection opens on first WebSocket connect (visible in logs)
- [ ] Sending real mic audio produces `transcript` messages with `is_final: true`
- [ ] Interrupt message stops any in-flight task and sends `tts_done`
- [ ] WebSocket disconnect triggers clean STT shutdown (no hanging tasks)
- [ ] No unhandled exceptions in logs during normal operation

## Error Handling Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `AuthenticationError` | Bad DEEPGRAM_API_KEY | Check `.env` key |
| `RuntimeError: Failed to open` | Network issue | Check internet + Deepgram status |
| `asyncio.QueueFull` | Audio arriving faster than Deepgram can accept | Reduce chunk rate (normal: log warning and drop) |
| `WebSocketDisconnect` during STT | Client closed browser | Normal — handled in finally block |
| `connection.start()` returns False | Invalid LiveOptions | Check encoding/sample_rate params |

## SDK Version Note

This plan targets `deepgram-sdk>=3.7.0`. The `client.listen.asynclive()` API
is the v3 pattern. If you see `AttributeError: 'Listen' object has no attribute 'asynclive'`,
upgrade: `pip install --upgrade deepgram-sdk`.
