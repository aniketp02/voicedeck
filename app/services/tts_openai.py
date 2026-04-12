"""
OpenAI TTS — streaming audio synthesis.

Uses OpenAI's audio.speech endpoint with httpx streaming.
Yields MP3 audio bytes in chunks; respects interrupt_event between chunks.

Model:  configured via settings.openai_tts_model  (default: tts-1)
Voice:  configured via settings.openai_tts_voice  (default: nova)
Output: mp3 — native output format for tts-1

No extra credentials needed: reuses OPENAI_API_KEY already used for the LLM.
"""
import asyncio
import logging

from app.config import settings
from app.services.llm import get_client

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 8192  # 8 KB — safe above typical MPEG frame size (~417 B at 128 kbps)


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

    client = get_client()
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
