from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # Deepgram
    deepgram_api_key: str
    deepgram_model: str = "nova-2"
    deepgram_language: str = "en-US"

    # ElevenLabs
    elevenlabs_api_key: str
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"  # George — clear, professional

    # App
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    log_level: str = "INFO"


settings = Settings()
