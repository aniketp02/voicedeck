"""
OpenAI TTS — streaming audio synthesis.

Uses OpenAI's audio.speech endpoint with httpx streaming.
Yields MP3 audio bytes in chunks; respects interrupt_event between chunks.

Model:  configured via settings.openai_tts_model  (default: tts-1)
Voice:  configured via settings.openai_tts_voice  (default: nova)
Output: mp3 — native output format for tts-1

API key: uses OPENAI_TTS_API_KEY when set, otherwise falls back to OPENAI_API_KEY.
Restricted keys need the "api.model.audio.request" scope — if your main key is
restricted to chat completions only, set OPENAI_TTS_API_KEY to a key that has the
audio scope (or an unrestricted key).
"""
import asyncio
import logging
import ssl

import httpx
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 8192  # 8 KB — safe above typical MPEG frame size (~417 B at 128 kbps)

_tts_client: AsyncOpenAI | None = None


def _get_tts_client() -> AsyncOpenAI:
    global _tts_client
    if _tts_client is None:
        api_key = settings.openai_tts_api_key or settings.openai_api_key
        ctx = ssl.create_default_context()
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        _tts_client = AsyncOpenAI(
            api_key=api_key,
            http_client=httpx.AsyncClient(
                verify=ctx,
                timeout=120.0,
            ),
        )
        key_source = "OPENAI_TTS_API_KEY" if settings.openai_tts_api_key else "OPENAI_API_KEY"
        logger.info("OpenAI TTS client initialised (key source: %s)", key_source)
    return _tts_client


async def synthesize_stream(
    text: str,
    interrupt_event: asyncio.Event,
):
    """
    Yield MP3 audio chunks for the given text via OpenAI TTS.

    Streams the response body from the audio.speech endpoint.
    Checks interrupt_event before yielding each chunk.
    """
    if not text.strip():
        logger.debug("openai_tts: empty text, skipping")
        return

    client = _get_tts_client()
    chunk_count = 0
    total_bytes = 0

    try:
        async with client.audio.speech.with_streaming_response.create(
            model=settings.openai_tts_model,
            voice=settings.openai_tts_voice,  # type: ignore[arg-type]
            input=text,
            response_format="mp3",
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                if interrupt_event.is_set():
                    logger.info(
                        "openai_tts: interrupted after %d chunks", chunk_count
                    )
                    return
                if chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield chunk

    except asyncio.CancelledError:
        logger.info("openai_tts: cancelled after %d chunks", chunk_count)
        raise
    except Exception as e:
        logger.error("OpenAI TTS error: %s", e)
        raise
    finally:
        if chunk_count > 0:
            logger.info(
                "openai_tts: complete — %d chunks, %d bytes, voice=%s model=%s",
                chunk_count,
                total_bytes,
                settings.openai_tts_voice,
                settings.openai_tts_model,
            )
