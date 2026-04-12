"""
TTS provider router.

Selects the TTS backend based on settings.tts_provider.
All other modules import from here — provider switching is transparent.

  TTS_PROVIDER=elevenlabs (default) -> app.services.tts_elevenlabs.synthesize_stream
  TTS_PROVIDER=deepgram             -> app.services.tts_deepgram.synthesize_stream
  TTS_PROVIDER=openai               -> app.services.tts_openai.synthesize_stream

All providers implement the same interface:
  async def synthesize_stream(text: str, interrupt_event: asyncio.Event)
"""
import logging

from app.config import settings

logger = logging.getLogger(__name__)

if settings.tts_provider == "deepgram":
    logger.info("TTS provider: Deepgram Aura (%s)", settings.deepgram_tts_voice)
    from app.services.tts_deepgram import synthesize_stream as synthesize_stream  # noqa: F401
elif settings.tts_provider == "openai":
    logger.info(
        "TTS provider: OpenAI TTS (voice=%s model=%s)",
        settings.openai_tts_voice,
        settings.openai_tts_model,
    )
    from app.services.tts_openai import synthesize_stream as synthesize_stream  # noqa: F401
else:
    logger.info("TTS provider: ElevenLabs (%s)", settings.elevenlabs_voice_id)
    from app.services.tts_elevenlabs import synthesize_stream as synthesize_stream  # noqa: F401
