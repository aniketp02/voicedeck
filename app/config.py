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
    # Deepgram HTTPS (httpx): Homebrew Python 3.14 often fails CA verification ("unable to get
    # local issuer") even with certifi. Default False = verify=False (works locally). Set True in
    # production when the host trust store validates api.deepgram.com.
    deepgram_tts_ssl_verify: bool = False

    # ElevenLabs (optional — only required when TTS_PROVIDER=elevenlabs)
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"  # George — clear, professional

    # OpenAI TTS (optional — only required when TTS_PROVIDER=openai)
    openai_tts_voice: str = "nova"   # nova | alloy | echo | fable | onyx | shimmer
    openai_tts_model: str = "tts-1"  # tts-1 | tts-1-hd

    # TTS provider selection
    # "elevenlabs" — high quality, requires ELEVENLABS_API_KEY (demo mode)
    # "deepgram"   — good quality, uses DEEPGRAM_API_KEY (dev/test mode, free credits)
    # "openai"     — natural quality, uses OPENAI_API_KEY (no extra cost)
    tts_provider: Literal["elevenlabs", "deepgram", "openai"] = "elevenlabs"

    # App
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    log_level: str = "INFO"


settings = Settings()
