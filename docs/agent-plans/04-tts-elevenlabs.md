# Agent Plan 04 — ElevenLabs TTS Streaming

## Agent Instructions
You are an autonomous agent. Read this plan completely before taking any action.
Do not ask the user questions. All decisions are made for you.
Implement every task in order. Run verification commands after each task.

---

## Goal
Replace the TTS placeholder in `run_agent` with real ElevenLabs streaming.
`run_agent` calls `synthesize_stream()`, which yields MP3 byte chunks.
Each chunk is base64-encoded and sent to the client as a `tts_chunk` message.
A `tts_done` message is sent after the last chunk (or on interrupt/error).

## Contract — What Plan 03 Delivered
- `app/agent/nodes.py`: all three nodes implemented
- `app/api/websocket.py`: `run_agent` sends `agent_text` then `tts_done`
- `ELEVENLABS_API_KEY` is set in `.env` (real key)
- Speaking triggers full agent pipeline (slide navigation + text response)

## Background: ElevenLabs SDK Streaming

The ElevenLabs Python SDK v1.x provides `AsyncElevenLabs`.
The method `client.text_to_speech.convert_as_stream()` returns a generator.

**Critical SDK behaviour:**
In elevenlabs SDK >= 1.0, `convert_as_stream()` on `AsyncElevenLabs` returns
an `AsyncIterator[bytes]`. However, in some versions it may return a sync
`Iterator[bytes]`. The implementation below handles both cases.

The correct model for lowest latency is `eleven_turbo_v2` (not multilingual).

---

## Task 1: Implement `app/services/tts.py`

Replace the entire file with:

```python
import asyncio
import inspect
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
    Yield MP3 audio chunks for the given text.

    Uses ElevenLabs eleven_turbo_v2 model for minimum latency.
    Checks interrupt_event before yielding each chunk — stops early if set.

    Handles both sync and async iterators returned by the SDK.
    """
    if not text.strip():
        logger.debug("synthesize_stream: empty text, skipping")
        return

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

        # Handle both async and sync iterators (SDK version variance)
        if inspect.isasyncgen(audio_stream) or hasattr(audio_stream, "__aiter__"):
            async for chunk in audio_stream:
                if interrupt_event.is_set():
                    logger.info("TTS interrupted after %d chunks", chunk_count)
                    return
                if isinstance(chunk, bytes) and chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield chunk
        else:
            # Sync iterator — wrap in asyncio.to_thread to avoid blocking event loop
            chunks = await asyncio.to_thread(list, audio_stream)
            for chunk in chunks:
                if interrupt_event.is_set():
                    logger.info("TTS interrupted after %d chunks", chunk_count)
                    return
                if isinstance(chunk, bytes) and chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield chunk

    except asyncio.CancelledError:
        logger.info("TTS synthesis cancelled after %d chunks", chunk_count)
        raise
    except Exception as e:
        logger.error("ElevenLabs TTS error: %s", e)
        raise
    finally:
        if chunk_count > 0:
            logger.info("TTS complete: %d chunks, %d bytes", chunk_count, total_bytes)
```

### Verify Task 1
```bash
python -c "from app.services.tts import synthesize_stream, get_client; print('tts imports OK')"
```

Test that the API key is valid (this will make a real API call):
```bash
python - << 'EOF'
import asyncio
from app.services.tts import synthesize_stream

async def test():
    interrupt = asyncio.Event()
    chunks = []
    async for chunk in synthesize_stream("Hello, testing ElevenLabs.", interrupt):
        chunks.append(chunk)
    total = sum(len(c) for c in chunks)
    print(f"TTS test OK: {len(chunks)} chunks, {total} bytes")

asyncio.run(test())
EOF
```

Expected: `TTS test OK: N chunks, NNNNN bytes` (N > 0)

**If this fails:**
- `401 Unauthorized` → wrong `ELEVENLABS_API_KEY`
- `422 Unprocessable Entity` → wrong `ELEVENLABS_VOICE_ID` — reset to default: `JBFqnCBsd6RMkjVDRZzb`
- `AttributeError: has no attribute 'convert_as_stream'` → upgrade SDK: `pip install --upgrade elevenlabs`

---

## Task 2: Wire TTS into `run_agent` in `app/api/websocket.py`

### 2a. Add TTS import
Add to the top of `websocket.py`:
```python
import base64
from app.services.tts import synthesize_stream
```
(`base64` may already be imported — if so, skip it.)

### 2b. Replace the TTS placeholder in `run_agent`

Find this block in `run_agent` (the TTS placeholder from Plan 03):
```python
        # 6. TTS placeholder (Plan 04 replaces this with ElevenLabs streaming)
        await _send(websocket, {"type": "tts_done"})
        tts_done_sent = True
```

Replace with:
```python
        # 6. Stream TTS audio to client
        if response_text and not interrupt_event.is_set():
            chunk_count = 0
            async for chunk in synthesize_stream(response_text, interrupt_event):
                chunk_count += 1
                await _send(websocket, {
                    "type": "tts_chunk",
                    "data": base64.b64encode(chunk).decode(),
                })
            logger.info("Streamed %d TTS chunks to client", chunk_count)

        await _send(websocket, {"type": "tts_done"})
        tts_done_sent = True
```

### 2c. Verify the change
```bash
python -c "from app.api.websocket import handle_session, run_agent; print('websocket OK')"
```

---

## Task 3: Integration test

### Step 1: Start server
```bash
uvicorn app.main:app --reload --port 8000
```

### Step 2: Full voice test (with frontend or audio client)

Watch the server logs for the complete pipeline:
```
INFO  app.services.stt      Deepgram STT connection opened (model=nova-2)
INFO  app.api.websocket      Final transcript: 'tell me about AI in clinical trials'
INFO  app.agent.nodes        understand_node: navigate=False ...
INFO  app.agent.nodes        respond_node: slide=0 generated 224 chars
INFO  app.api.websocket      Sent agent_text: 224 chars
INFO  app.services.tts       TTS complete: 12 chunks, 48320 bytes
INFO  app.api.websocket      Streamed 12 TTS chunks to client
```

WebSocket client must receive (in order):
```json
{"type": "transcript", "text": "tell me about AI in clinical trials", "is_final": true}
{"type": "agent_text", "text": "AI is transforming clinical trials by..."}
{"type": "tts_chunk", "data": "//NExAAA..."}
{"type": "tts_chunk", "data": "//NExBwA..."}
... (multiple chunks)
{"type": "tts_done"}
```

### Step 3: Verify audio decoding
Decode and play one of the tts_chunk responses manually:
```python
import base64

# Paste a tts_chunk data value here
data = "//NExAA..."  # from WebSocket inspector
mp3_bytes = base64.b64decode(data)
with open("/tmp/test_chunk.mp3", "wb") as f:
    f.write(mp3_bytes)
print(f"Wrote {len(mp3_bytes)} bytes to /tmp/test_chunk.mp3")
# Play: mpv /tmp/test_chunk.mp3
```

### Step 4: Measure latency
Time from `Final transcript` log to first `tts_chunk` sent.
Target: < 2 seconds total (LLM call ~500ms + ElevenLabs ~500ms + network).

If latency is > 3 seconds:
- Check OpenAI model is `gpt-4o-mini` (not gpt-4)
- Check ElevenLabs model is `eleven_turbo_v2` (not multilingual)
- Consider response length — shorter responses = faster TTS start

---

## Acceptance Criteria

- [ ] `app/services/tts.py` — no `NotImplementedError`, handles both sync/async iterators
- [ ] TTS test script produces > 0 chunks without errors
- [ ] `run_agent` streams TTS chunks before sending `tts_done`
- [ ] Client receives multiple `tts_chunk` messages for a typical response
- [ ] `tts_done` is always the last message (even on error or interrupt)
- [ ] Interrupting during TTS: `interrupt_event.is_set()` → TTS stops early → `tts_done` sent
- [ ] Empty response text → no TTS chunks, just `tts_done`
- [ ] Audio in tts_chunk is valid MP3 (decodeable by browser Audio API)

## ElevenLabs Free Tier Limits

Free tier: 10,000 characters/month.
A typical response is ~200 characters.
This gives ~50 full responses before hitting the limit.
Monitor usage at https://elevenlabs.io/account

If you hit the limit mid-demo:
- Reduce response length in `RESPOND_SYSTEM` prompt ("Keep responses under 2 sentences")
- Or upgrade to Creator tier ($5/month for 30k chars)

## Chunking Strategy Note

`eleven_turbo_v2` starts streaming audio before generating the full text.
The first chunk usually arrives in ~300-500ms.
The client should start playing audio as soon as the first chunk arrives,
not wait for `tts_done`. This is handled in frontend Plan 03.

## Alternative: Sentence-by-Sentence TTS (advanced)

For lower time-to-first-audio, split the response at sentence boundaries
and call `synthesize_stream` on each sentence. First audio plays in ~200ms.
This is an enhancement beyond MVP scope — do not implement now.
