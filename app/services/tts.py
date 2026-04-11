"""
ElevenLabs TTS — streaming audio synthesis.

Usage pattern:
    async for chunk in synthesize_stream(text, interrupt_event):
        await websocket.send_bytes(chunk)

Yields MP3 audio bytes in chunks as they arrive from ElevenLabs.
Checks interrupt_event between chunks; stops early if set.
"""
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
    Stream TTS audio chunks for text. Yields bytes.
    Stops yielding if interrupt_event is set between chunks.

    TODO (Plan 04): Implement full ElevenLabs streaming integration.
    - Call client.text_to_speech.convert_as_stream()
    - Use voice_id from settings
    - model_id: "eleven_turbo_v2" (lowest latency)
    - output_format: "mp3_44100_128"
    - Yield each chunk, checking interrupt_event before each yield
    - Log chunk count and total bytes for debugging
    """
    raise NotImplementedError("ElevenLabs TTS not yet implemented — see docs/plans/04-tts-elevenlabs.md")
