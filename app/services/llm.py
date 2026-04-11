import json
import logging
import ssl

import httpx
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)
_client: AsyncOpenAI | None = None


def _openai_ssl_context() -> ssl.SSLContext:
    """
    Match Deepgram/ElevenLabs: some macOS/Python 3.13+ chains fail unless
    VERIFY_X509_STRICT is cleared; we still use the default CA store.
    """
    ctx = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            http_client=httpx.AsyncClient(
                verify=_openai_ssl_context(),
                timeout=120.0,
            ),
        )
    return _client


async def chat_completion(system: str, user: str) -> str:
    """Single-turn completion. Returns the assistant message text."""
    client = get_client()
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


async def chat_completion_json(system: str, user: str) -> dict:
    """Completion that expects and parses JSON response."""
    client = get_client()
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON: %s", raw)
        return {}
