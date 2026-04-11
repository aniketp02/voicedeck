# Plan 02 — STT: Deepgram Streaming Integration

## Goal
Implement real-time speech-to-text using Deepgram's streaming WebSocket API.
The browser sends raw PCM audio chunks; the backend streams them to Deepgram
and fires a callback with the final transcript.

**Success criterion:** Speak into the browser mic → see the transcribed text
appear in the server logs within ~1 second.

## Deepgram Setup
1. Sign up at https://deepgram.com (free $200 credit, no charges until exhausted)
2. Create an API key in the dashboard → copy to `.env` as `DEEPGRAM_API_KEY`
3. Use model `nova-2` (best accuracy/latency balance)

## Files to Modify

### `app/services/stt.py` — FULL IMPLEMENTATION REQUIRED

Replace the `transcribe_stream` stub with the real implementation:

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
    audio_queue: asyncio.Queue[bytes | None],
    on_transcript,   # async callable(TranscriptResult)
) -> None:
    """
    Read audio bytes from audio_queue, stream to Deepgram,
    call on_transcript for each result.
    Stops when audio_queue yields None (sentinel).
    """
    config = DeepgramClientOptions(options={"keepalive": "true"})
    client = DeepgramClient(settings.deepgram_api_key, config)

    connection = client.listen.asynclive(
        LiveOptions(
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
    )

    async def on_message(self, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if not sentence:
            return
        is_final = result.is_final
        confidence = result.channel.alternatives[0].confidence
        await on_transcript(TranscriptResult(
            text=sentence,
            is_final=is_final,
            confidence=confidence,
        ))

    async def on_error(self, error, **kwargs):
        logger.error("Deepgram error: %s", error)

    connection.on(LiveTranscriptionEvents.Transcript, on_message)
    connection.on(LiveTranscriptionEvents.Error, on_error)

    await connection.start()
    logger.info("Deepgram STT connection opened")

    try:
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            await connection.send(chunk)
    finally:
        await connection.finish()
        logger.info("Deepgram STT connection closed")
```

### `app/api/websocket.py` — REPLACE `_placeholder_loop` with real pipeline

Replace the `handle_session` function body (after initial slide send) with:

```python
async def handle_session(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("WebSocket session started")

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
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    # Send initial slide
    slide = get_slide(0)
    await _send(websocket, {
        "type": "slide_change",
        "index": 0,
        "slide": {"title": slide.title, "bullets": slide.bullets},
    })

    async def on_transcript(result: TranscriptResult) -> None:
        # Send interim transcripts to UI for live feedback
        await _send(websocket, {
            "type": "transcript",
            "text": result.text,
            "is_final": result.is_final,
        })
        if result.is_final and result.text.strip():
            # Run agent graph on final transcript (Plans 03-04 wire this up)
            logger.info("Final transcript: %r", result.text)
            # TODO Plan 03: await run_agent(websocket, state, result.text, interrupt_event)

    # Start STT in background
    stt_task = asyncio.create_task(
        transcribe_stream(audio_queue, on_transcript)
    )

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio_chunk":
                audio_bytes = base64.b64decode(msg["data"])
                await audio_queue.put(audio_bytes)

            elif msg_type == "interrupt":
                interrupt_event.set()
                await _send(websocket, {"type": "tts_done"})
                interrupt_event.clear()

            elif msg_type == "ping":
                await _send(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("Session error: %s", e)
        await _send(websocket, {"type": "error", "message": str(e)})
    finally:
        await audio_queue.put(None)  # signal STT to stop
        stt_task.cancel()
        try:
            await stt_task
        except asyncio.CancelledError:
            pass
```

Add these imports at the top of `websocket.py`:
```python
from app.services.stt import transcribe_stream, TranscriptResult
```

## Audio Format Contract
The browser must send **raw 16-bit PCM, 16kHz, mono** audio as base64.
The frontend plan (Plan 01) handles this with an AudioWorklet.
If testing manually, generate test audio:
```python
import base64, struct, math
samples = [int(32767 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(16000)]
pcm = struct.pack(f"<{len(samples)}h", *samples)
b64 = base64.b64encode(pcm).decode()
```

## Verification
1. Start backend: `uvicorn app.main:app --reload --port 8000`
2. Connect WebSocket client, send `{"type":"start"}`
3. Send base64 PCM audio chunks
4. Watch logs — should see `Final transcript: "your speech here"`
5. Check WebSocket receives `{"type":"transcript","text":"...","is_final":true}`

## Troubleshooting
- `AuthenticationError` → check `DEEPGRAM_API_KEY` in `.env`
- No transcripts → verify audio is `encoding=linear16, sample_rate=16000, channels=1`
- High latency → ensure `nova-2` model is set, check network to Deepgram
