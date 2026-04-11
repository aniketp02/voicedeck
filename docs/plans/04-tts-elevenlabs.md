# Plan 04 — TTS: ElevenLabs Streaming Audio

## Goal
Stream ElevenLabs TTS audio back to the client as MP3 chunks immediately
as they arrive (no buffering). The client receives `tts_chunk` messages
and plays them in sequence. A `tts_done` message signals end of speech.

**Success criterion:** Agent's text response plays as audio in the browser
within ~500ms of the agent finishing its text generation.

## ElevenLabs Setup
1. Sign up at https://elevenlabs.io (free tier: 10,000 chars/month)
2. Copy your API key → `.env` as `ELEVENLABS_API_KEY`
3. Default voice: `JBFqnCBsd6RMkjVDRZzb` (George — professional, clear)
   - Browse voices: https://elevenlabs.io/voice-library
   - Override in `.env` as `ELEVENLABS_VOICE_ID`

## Files to Modify

### `app/services/tts.py` — FULL IMPLEMENTATION REQUIRED

```python
import asyncio
import logging
from elevenlabs.client import AsyncElevenLabs
from app.config import settings

logger = logging.getLogger(__name__)
_client: AsyncElevenLabs | None = None


def get_client() -> AsyncElevenLabs:
    global _client
    if _client is None:
        _client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
    return _client


async def synthesize_stream(
    text: str,
    interrupt_event: asyncio.Event,
):
    """
    Yield MP3 audio chunks for text.
    Stops early if interrupt_event is set between chunks.
    """
    client = get_client()
    chunk_count = 0
    total_bytes = 0

    try:
        audio_stream = client.text_to_speech.convert_as_stream(
            voice_id=settings.elevenlabs_voice_id,
            text=text,
            model_id="eleven_turbo_v2",
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        )

        async for chunk in audio_stream:
            if interrupt_event.is_set():
                logger.info("TTS interrupted after %d chunks", chunk_count)
                return
            if isinstance(chunk, bytes) and chunk:
                chunk_count += 1
                total_bytes += len(chunk)
                yield chunk

    except Exception as e:
        logger.error("ElevenLabs TTS error: %s", e)
        raise
    finally:
        logger.info("TTS complete: %d chunks, %d bytes", chunk_count, total_bytes)
```

### `app/api/websocket.py` — Add TTS streaming to `run_agent`

Replace the `# TTS will be streamed in Plan 04` comment block in `run_agent` with:

```python
from app.services.tts import synthesize_stream

# Stream TTS audio to client
if response_text and not interrupt_event.is_set():
    async for chunk in synthesize_stream(response_text, interrupt_event):
        await _send_bytes(websocket, {
            "type": "tts_chunk",
            "data": base64.b64encode(chunk).decode(),
        })

await _send(websocket, {"type": "tts_done"})
```

Add the `_send_bytes` helper (same as `_send`, just a different name for clarity):
```python
async def _send_bytes(websocket: WebSocket, payload: dict) -> None:
    """Send a JSON message containing base64-encoded binary data."""
    await websocket.send_text(json.dumps(payload))
```

## Audio Playback Contract (Frontend)
The frontend (Plan 03 frontend) receives `tts_chunk` messages and uses the
Web Audio API to queue and play them. The key constraint:
- Chunks arrive in order
- Each chunk is a complete fragment of the MP3 stream (not a full file)
- The browser must concatenate chunks into a MediaSource or use AudioContext

## Verification
1. Complete Plans 02 and 03 first
2. Start backend, connect frontend
3. Speak a question → watch for:
   - `slide_change` (if navigation occurred)
   - `agent_text` with the response
   - Multiple `tts_chunk` messages (each ~4-16KB)
   - Final `tts_done`
4. Audio should play in browser (frontend Plan 04 must be done)

## Latency Optimization Tips
- `eleven_turbo_v2` is 2-3x faster than `eleven_multilingual_v2`
- Keep responses under 100 words for best latency
- Consider splitting long responses at sentence boundaries and streaming each sentence separately for lower time-to-first-audio

## Troubleshooting
- `401 Unauthorized` → check `ELEVENLABS_API_KEY`
- `422 Unprocessable Entity` → check `voice_id` is valid
- No audio in browser → check frontend Plan 04 (AudioContext implementation)
- Choppy audio → client is not buffering enough chunks before starting playback
