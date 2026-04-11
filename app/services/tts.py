"""
ElevenLabs TTS — streaming audio synthesis.

Yields MP3 audio bytes in chunks as they arrive from ElevenLabs.
Checks interrupt_event between chunks; stops early if set.
"""
import asyncio
import inspect
import logging
import ssl

import httpx
from elevenlabs.client import AsyncElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings

from app.config import settings

logger = logging.getLogger(__name__)

_client: AsyncElevenLabs | None = None


def _elevenlabs_ssl_context() -> ssl.SSLContext:
    """Match Deepgram STT: relax VERIFY_X509_STRICT for some TLS chains (Py 3.13+ / OpenSSL 3.4+)."""
    ctx = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


_DEFAULT_VOICE_SETTINGS = VoiceSettings(
    stability=0.5,
    similarity_boost=0.75,
    style=0.0,
    use_speaker_boost=True,
)


def get_client() -> AsyncElevenLabs:
    global _client
    if _client is None:
        _client = AsyncElevenLabs(
            api_key=settings.elevenlabs_api_key,
            httpx_client=httpx.AsyncClient(
                verify=_elevenlabs_ssl_context(),
                timeout=240,
                follow_redirects=True,
            ),
        )
    return _client


def _open_audio_stream(client: AsyncElevenLabs, text: str):
    """
    Open a streaming TTS request.

    elevenlabs>=1.x exposes `stream()`; older docs referred to `convert_as_stream()`.
    """
    tts = client.text_to_speech
    kwargs = dict(
        voice_id=settings.elevenlabs_voice_id,
        text=text,
        model_id="eleven_turbo_v2",
        output_format="mp3_44100_128",
        voice_settings=_DEFAULT_VOICE_SETTINGS,
    )
    if hasattr(tts, "stream"):
        return tts.stream(**kwargs)
    if hasattr(tts, "convert_as_stream"):
        return tts.convert_as_stream(**kwargs)
    raise AttributeError(
        "Async text_to_speech has neither stream() nor convert_as_stream(); upgrade elevenlabs"
    )


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
        audio_stream = _open_audio_stream(client, text)

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
