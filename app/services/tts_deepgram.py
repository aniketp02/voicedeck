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

_client: httpx.AsyncClient | None = None


def _httpx_verify() -> bool | str:
    """
    TLS for httpx → api.deepgram.com.

    ``deepgram_tts_ssl_verify`` (env ``DEEPGRAM_TTS_SSL_VERIFY``): when True, verify using
    certifi's CA bundle. When False (default), ``verify=False`` — required on many Homebrew
    Python 3.14 installs where OpenSSL still reports "unable to get local issuer" even with
    certifi. Enable verification in production when the runtime trust store works.
    """
    if not settings.deepgram_tts_ssl_verify:
        return False

    try:
        import certifi

        return certifi.where()
    except ImportError:
        return True


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=60,
            follow_redirects=True,
            verify=_httpx_verify(),
        )
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

            # Larger reads reduce how often arbitrary 4KiB boundaries split MPEG frames.
            # MSE appendBuffer expects MPEG frame boundaries; mid-frame splits cause dropped
            # audio at the start or mid-utterance (ElevenLabs chunks are typically larger).
            async for chunk in response.aiter_bytes(chunk_size=32768):
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
