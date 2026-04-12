# Backend Agent Plan 12 — Dual TTS Provider (ElevenLabs + Deepgram Aura)

## Agent Instructions
Read this plan fully before acting. Do not ask questions. All decisions are made.
Plans 01–10 must be complete before starting.
No frontend changes — backend only.

---

## Goal
Add Deepgram's Aura TTS as a second provider behind a `TTS_PROVIDER` environment variable.
ElevenLabs remains the default (and the demo provider). Deepgram is used for development and
testing when ElevenLabs credits are exhausted.

Both providers expose the same `synthesize_stream(text, interrupt_event)` async generator
interface. `websocket.py` never changes — it imports `synthesize_stream` from one place and
the env flag determines which implementation runs.

**Result:**
```bash
# Development / testing (free Deepgram credits):
TTS_PROVIDER=deepgram uvicorn app.main:app --port 8000

# Demo (full ElevenLabs voice quality):
TTS_PROVIDER=elevenlabs uvicorn app.main:app --port 8000
# or just omit TTS_PROVIDER — elevenlabs is the default
```

---

## Deepgram Aura TTS primer

Deepgram TTS is a simple REST endpoint — no SDK streaming wrapper needed.

```
POST https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=mp3
Authorization: Token <DEEPGRAM_API_KEY>
Content-Type: application/json
{"text": "Hello world"}
```

The response body is the audio stream. With `httpx.AsyncClient.stream()`, we receive it
as chunks without waiting for the full file.

Available Aura voices (English):
- `aura-asteria-en` — female, warm, natural-sounding (recommended)
- `aura-luna-en` — female, conversational
- `aura-zeus-en` — male, authoritative
- `aura-orion-en` — male, neutral
- `aura-arcas-en` — male, casual

Output formats: `mp3`, `mp3_22050_32`, `mp3_44100_128` (same as ElevenLabs output)
Use `mp3` for simplicity — the client MSE player accepts any MP3.

---

## Files Changed

- `app/config.py` — add `tts_provider`, `deepgram_tts_voice`, `elevenlabs_api_key` becomes optional
- `app/services/tts_deepgram.py` — NEW: Deepgram Aura TTS implementation
- `app/services/tts.py` — rename to `tts_elevenlabs.py` and keep existing implementation unchanged
- `app/services/tts.py` — NEW: thin router that imports from the correct provider based on `settings.tts_provider`
- `tests/test_tts.py` — extend to cover Deepgram provider
- `.env.example` (if it exists) — add `TTS_PROVIDER` entry

---

## Task 1: Update `app/config.py`

```python
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # Deepgram (STT + optional TTS)
    deepgram_api_key: str
    deepgram_model: str = "nova-2"
    deepgram_language: str = "en-US"
    deepgram_tts_voice: str = "aura-asteria-en"

    # ElevenLabs (optional — only required when TTS_PROVIDER=elevenlabs)
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"  # George — clear, professional

    # TTS provider selection
    # "elevenlabs" — high quality, requires ELEVENLABS_API_KEY (demo mode)
    # "deepgram"   — good quality, uses DEEPGRAM_API_KEY (dev/test mode, free credits)
    tts_provider: Literal["elevenlabs", "deepgram"] = "elevenlabs"

    # App
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    log_level: str = "INFO"


settings = Settings()
```

Note: `elevenlabs_api_key` defaults to `""` so the app starts without it when
`TTS_PROVIDER=deepgram`. Validation is enforced at call time in `tts_elevenlabs.py`.

---

## Task 2: Rename `app/services/tts.py` → `app/services/tts_elevenlabs.py`

Copy the entire current `app/services/tts.py` to `app/services/tts_elevenlabs.py`.
No changes to the content — the existing ElevenLabs implementation stays exactly as-is.

After copying, add one guard at the top of `synthesize_stream` in `tts_elevenlabs.py` to
raise a clear error if the API key is missing:

In `tts_elevenlabs.py`, at the top of `synthesize_stream()`, add before the client call:

```python
async def synthesize_stream(
    text: str,
    interrupt_event: asyncio.Event,
):
    """...(existing docstring)..."""
    if not settings.elevenlabs_api_key:
        raise RuntimeError(
            "TTS_PROVIDER=elevenlabs but ELEVENLABS_API_KEY is not set. "
            "Set the key or switch to TTS_PROVIDER=deepgram."
        )
    if not text.strip():
        logger.debug("synthesize_stream: empty text, skipping")
        return
    # ... rest of existing implementation unchanged
```

---

## Task 3: Create `app/services/tts_deepgram.py`

New file. Implements `synthesize_stream` with the same signature as the ElevenLabs version —
an async generator that yields `bytes` chunks and respects `interrupt_event`.

```python
"""
Deepgram Aura TTS — streaming audio synthesis.

Uses Deepgram's /v1/speak REST endpoint with httpx streaming.
Yields MP3 audio bytes in chunks; respects interrupt_event between chunks.

Model: configured via settings.deepgram_tts_voice (default: aura-asteria-en)
Output: mp3 (native, no re-encoding needed)

Free tier: included in DEEPGRAM_API_KEY credits — use this provider when
ElevenLabs credits are exhausted or during development.
"""
import asyncio
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"

# Shared httpx client — created once and reused across requests.
# SSL defaults are fine for Deepgram (no OpenSSL strict-mode issues).
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60, follow_redirects=True)
    return _client


async def synthesize_stream(
    text: str,
    interrupt_event: asyncio.Event,
):
    """
    Yield MP3 audio chunks for the given text via Deepgram Aura TTS.

    Streams response body from Deepgram's /v1/speak endpoint.
    Checks interrupt_event before yielding each chunk.
    """
    if not text.strip():
        logger.debug("deepgram_tts: empty text, skipping")
        return

    client = _get_client()
    chunk_count = 0
    total_bytes = 0

    params = {
        "model": settings.deepgram_tts_voice,
        "encoding": "mp3",
    }
    headers = {
        "Authorization": f"Token {settings.deepgram_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    try:
        async with client.stream(
            "POST",
            _DEEPGRAM_TTS_URL,
            params=params,
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise RuntimeError(
                    f"Deepgram TTS returned HTTP {response.status_code}: {body[:200]!r}"
                )

            async for chunk in response.aiter_bytes(chunk_size=4096):
                if interrupt_event.is_set():
                    logger.info(
                        "deepgram_tts: interrupted after %d chunks", chunk_count
                    )
                    return
                if chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield chunk

    except asyncio.CancelledError:
        logger.info("deepgram_tts: cancelled after %d chunks", chunk_count)
        raise
    except Exception as e:
        logger.error("Deepgram TTS error: %s", e)
        raise
    finally:
        if chunk_count > 0:
            logger.info(
                "deepgram_tts: complete — %d chunks, %d bytes, voice=%s",
                chunk_count,
                total_bytes,
                settings.deepgram_tts_voice,
            )
```

---

## Task 4: Create new `app/services/tts.py` (the router)

The old `tts.py` is now `tts_elevenlabs.py`. Create a new `tts.py` that is the single
import point for the rest of the codebase. `websocket.py` already imports
`from app.services.tts import synthesize_stream` — this keeps that import working.

```python
"""
TTS provider router.

Selects the TTS backend based on settings.tts_provider.
All other modules import from here — provider switching is transparent.

  TTS_PROVIDER=elevenlabs (default) → app.services.tts_elevenlabs.synthesize_stream
  TTS_PROVIDER=deepgram             → app.services.tts_deepgram.synthesize_stream

Both providers implement the same interface:
  async def synthesize_stream(text: str, interrupt_event: asyncio.Event) -> AsyncGenerator[bytes, None]
"""
import logging

from app.config import settings

logger = logging.getLogger(__name__)

if settings.tts_provider == "deepgram":
    logger.info("TTS provider: Deepgram Aura (%s)", settings.deepgram_tts_voice)
    from app.services.tts_deepgram import synthesize_stream as synthesize_stream  # noqa: F401
else:
    logger.info("TTS provider: ElevenLabs (%s)", settings.elevenlabs_voice_id)
    from app.services.tts_elevenlabs import synthesize_stream as synthesize_stream  # noqa: F401
```

The `# noqa: F401` suppresses linter warnings about "imported but unused" — the import
exists purely to re-export the symbol.

**Important:** Nothing else in the codebase needs to change. `websocket.py` already imports
`from app.services.tts import synthesize_stream` and continues to work unchanged.

---

## Task 5: Update `tests/test_tts.py`

The existing 5 tests mock `app.services.tts._open_audio_stream` which was in the old
`tts.py`. After the rename, this path must be updated to `app.services.tts_elevenlabs`.

Also add tests for the Deepgram provider.

```python
"""
Unit tests for TTS services — both ElevenLabs and Deepgram providers.
"""
import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


async def _collect(async_gen):
    """Helper: collect all items from an async generator."""
    items = []
    async for item in async_gen:
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# ElevenLabs provider (tts_elevenlabs.py)
# ---------------------------------------------------------------------------

class TestElevenLabsSynthesizeStream:
    @pytest.mark.asyncio
    async def test_yields_chunks_from_async_iterator(self):
        from app.services.tts_elevenlabs import synthesize_stream

        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        async def fake_async_iter():
            for c in chunks:
                yield c

        mock_stream = fake_async_iter()
        interrupt = asyncio.Event()

        with patch("app.services.tts_elevenlabs.settings") as mock_settings, \
             patch("app.services.tts_elevenlabs._open_audio_stream", return_value=mock_stream):
            mock_settings.elevenlabs_api_key = "test-key"
            result = await _collect(synthesize_stream("Hello world", interrupt))

        assert result == chunks

    @pytest.mark.asyncio
    async def test_stops_on_interrupt_event(self):
        from app.services.tts_elevenlabs import synthesize_stream

        interrupt = asyncio.Event()

        async def fake_async_iter():
            yield b"chunk1"
            interrupt.set()
            yield b"chunk2"
            yield b"chunk3"

        mock_stream = fake_async_iter()

        with patch("app.services.tts_elevenlabs.settings") as mock_settings, \
             patch("app.services.tts_elevenlabs._open_audio_stream", return_value=mock_stream):
            mock_settings.elevenlabs_api_key = "test-key"
            result = await _collect(synthesize_stream("Hello world", interrupt))

        assert result == [b"chunk1"]

    @pytest.mark.asyncio
    async def test_skips_empty_text(self):
        from app.services.tts_elevenlabs import synthesize_stream

        interrupt = asyncio.Event()
        with patch("app.services.tts_elevenlabs._open_audio_stream") as mock_open, \
             patch("app.services.tts_elevenlabs.settings") as mock_settings:
            mock_settings.elevenlabs_api_key = "test-key"
            result = await _collect(synthesize_stream("   ", interrupt))

        mock_open.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_non_bytes_chunks(self):
        from app.services.tts_elevenlabs import synthesize_stream

        interrupt = asyncio.Event()

        async def fake_async_iter():
            yield b""
            yield "not bytes"
            yield b"real_chunk"

        mock_stream = fake_async_iter()

        with patch("app.services.tts_elevenlabs.settings") as mock_settings, \
             patch("app.services.tts_elevenlabs._open_audio_stream", return_value=mock_stream):
            mock_settings.elevenlabs_api_key = "test-key"
            result = await _collect(synthesize_stream("Hello", interrupt))

        assert result == [b"real_chunk"]

    @pytest.mark.asyncio
    async def test_handles_sync_iterator_via_to_thread(self):
        from app.services.tts_elevenlabs import synthesize_stream

        chunks = [b"a", b"b", b"c"]
        mock_stream = iter(chunks)
        interrupt = asyncio.Event()

        with patch("app.services.tts_elevenlabs.settings") as mock_settings, \
             patch("app.services.tts_elevenlabs._open_audio_stream", return_value=mock_stream):
            mock_settings.elevenlabs_api_key = "test-key"
            result = await _collect(synthesize_stream("Hello", interrupt))

        assert result == chunks

    @pytest.mark.asyncio
    async def test_raises_when_api_key_missing(self):
        from app.services.tts_elevenlabs import synthesize_stream

        interrupt = asyncio.Event()
        with patch("app.services.tts_elevenlabs.settings") as mock_settings:
            mock_settings.elevenlabs_api_key = ""
            with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
                await _collect(synthesize_stream("Hello", interrupt))


# ---------------------------------------------------------------------------
# Deepgram TTS provider (tts_deepgram.py)
# ---------------------------------------------------------------------------

class TestDeepgramSynthesizeStream:
    @pytest.mark.asyncio
    async def test_yields_chunks_from_http_stream(self):
        from app.services.tts_deepgram import synthesize_stream

        chunks = [b"audio1", b"audio2", b"audio3"]
        interrupt = asyncio.Event()

        # Mock httpx async response that yields chunks
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def fake_aiter_bytes(chunk_size=4096):
            for c in chunks:
                yield c

        mock_response.aiter_bytes = fake_aiter_bytes
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.tts_deepgram._get_client") as mock_client:
            mock_http = MagicMock()
            mock_http.stream.return_value = mock_response
            mock_client.return_value = mock_http

            result = await _collect(synthesize_stream("Hello world", interrupt))

        assert result == chunks

    @pytest.mark.asyncio
    async def test_skips_empty_text(self):
        from app.services.tts_deepgram import synthesize_stream

        interrupt = asyncio.Event()
        with patch("app.services.tts_deepgram._get_client") as mock_client:
            result = await _collect(synthesize_stream("   ", interrupt))

        mock_client.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_stops_on_interrupt(self):
        from app.services.tts_deepgram import synthesize_stream

        interrupt = asyncio.Event()

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def fake_aiter_bytes(chunk_size=4096):
            yield b"chunk1"
            interrupt.set()
            yield b"chunk2"  # should not be yielded

        mock_response.aiter_bytes = fake_aiter_bytes
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.tts_deepgram._get_client") as mock_client:
            mock_http = MagicMock()
            mock_http.stream.return_value = mock_response
            mock_client.return_value = mock_http

            result = await _collect(synthesize_stream("Hello world", interrupt))

        assert result == [b"chunk1"]

    @pytest.mark.asyncio
    async def test_raises_on_non_200_response(self):
        from app.services.tts_deepgram import synthesize_stream

        interrupt = asyncio.Event()

        mock_response = AsyncMock()
        mock_response.status_code = 401
        mock_response.aread = AsyncMock(return_value=b'{"error": "Unauthorized"}')
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.tts_deepgram._get_client") as mock_client:
            mock_http = MagicMock()
            mock_http.stream.return_value = mock_response
            mock_client.return_value = mock_http

            with pytest.raises(RuntimeError, match="HTTP 401"):
                await _collect(synthesize_stream("Hello world", interrupt))
```

---

## Task 6: Update `.env` (or `.env.example`)

Check if a `.env.example` or `.env` file exists at `backend/.env.example` or `backend/.env`.

If `.env.example` exists, add:
```bash
# TTS provider: "elevenlabs" (default, best quality) or "deepgram" (free, uses Deepgram credits)
TTS_PROVIDER=elevenlabs
DEEPGRAM_TTS_VOICE=aura-asteria-en
```

If only `.env` exists (no example file), add the same lines to `.env` under the Deepgram section.

**For immediate use during testing**, change the value in `.env` to:
```bash
TTS_PROVIDER=deepgram
```

---

## Task 7: Run tests

```bash
cd backend
source venv/bin/activate
pytest tests/test_tts.py -v
```

Expected: 10 tests pass (5 existing ElevenLabs + 5 Deepgram — 1 new ElevenLabs test for missing key).

Then run the full suite to ensure nothing else broke:
```bash
pytest tests/ -v
```

**Common failures:**
- Any test that imports from `app.services.tts` and patches `app.services.tts._open_audio_stream`
  needs updating to `app.services.tts_elevenlabs._open_audio_stream`
- The router `tts.py` runs its conditional import at module load time. If tests import `tts.py`
  and `TTS_PROVIDER` is not set in the test environment, it defaults to `elevenlabs`. This is
  correct behavior and requires no fixture.

---

## Task 8: Manual smoke test

```bash
# Test Deepgram TTS
TTS_PROVIDER=deepgram uvicorn app.main:app --port 8000 --log-level info
```

Start the frontend, connect, start mic, ask a question. Watch backend logs:
```
INFO  TTS provider: Deepgram Aura (aura-asteria-en)
INFO  deepgram_tts: complete — 12 chunks, 48302 bytes, voice=aura-asteria-en
```

Audio should play in the browser. Voice will sound different from ElevenLabs (slightly
more robotic) but the full agentic flow — transcript → LLM → TTS → slide navigation — works.

```bash
# Confirm ElevenLabs still works when you have credits
TTS_PROVIDER=elevenlabs uvicorn app.main:app --port 8000
```

---

## Acceptance Criteria

- [ ] `app/config.py` — `tts_provider` field (Literal `"elevenlabs"|"deepgram"`), defaults to `"elevenlabs"`; `elevenlabs_api_key` defaults to `""`; `deepgram_tts_voice` field
- [ ] `app/services/tts_elevenlabs.py` — identical to old `tts.py` plus missing-key guard
- [ ] `app/services/tts_deepgram.py` — httpx streaming, interrupt-respecting, error handling
- [ ] `app/services/tts.py` — router that re-exports `synthesize_stream` from correct provider
- [ ] `websocket.py` unchanged (still imports `from app.services.tts import synthesize_stream`)
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] `TTS_PROVIDER=deepgram` — audio plays in browser, agentic flow intact
- [ ] `TTS_PROVIDER=elevenlabs` — ElevenLabs still works when key is set
- [ ] `TTS_PROVIDER=elevenlabs` with missing key — raises clear `RuntimeError` on use (not on startup)

## File Checklist After This Plan

```
backend/
  app/
    config.py                   ← tts_provider, deepgram_tts_voice, elevenlabs_api_key optional
    services/
      tts.py                    ← NEW router (re-exports synthesize_stream from selected provider)
      tts_elevenlabs.py         ← renamed from old tts.py + missing-key guard
      tts_deepgram.py           ← NEW Deepgram Aura implementation
  tests/
    test_tts.py                 ← updated paths + Deepgram tests (10 tests total)
  .env / .env.example           ← TTS_PROVIDER + DEEPGRAM_TTS_VOICE entries
```
