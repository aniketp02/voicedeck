import json
import logging
import re
import ssl
from collections.abc import AsyncGenerator

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


async def chat_completion(system: str, user: str, model: str | None = None) -> str:
    """Single-turn completion. Returns the assistant message text."""
    client = get_client()
    response = await client.chat.completions.create(
        model=model or settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


async def chat_completion_stream(
    system: str, user: str, model: str | None = None
) -> AsyncGenerator[str, None]:
    """Yield raw token strings from the OpenAI streaming API."""
    client = get_client()
    response = await client.chat.completions.create(
        model=model or settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        stream=True,
    )
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MIN_FLUSH_CHARS = 20


async def sentence_stream(
    token_gen: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """
    Accumulate raw tokens into sentence-length chunks for streaming TTS.

    Splits on . ! ? followed by whitespace. Buffers until at least
    _MIN_FLUSH_CHARS characters have accumulated before splitting, to avoid
    flushing trivially short fragments. Yields any remainder at end of stream.
    """
    buf = ""
    async for token in token_gen:
        buf += token
        if len(buf) < _MIN_FLUSH_CHARS:
            continue
        parts = _SENTENCE_SPLIT.split(buf)
        # parts[-1] is the incomplete trailing fragment (no sentence-ending yet)
        for sentence in parts[:-1]:
            stripped = sentence.strip()
            if stripped:
                yield stripped
        buf = parts[-1]
    if buf.strip():
        yield buf.strip()


async def chat_completion_json(system: str, user: str, model: str | None = None) -> dict:
    """Completion that expects and parses JSON response."""
    client = get_client()
    response = await client.chat.completions.create(
        model=model or settings.openai_model,
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
