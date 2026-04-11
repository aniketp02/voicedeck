"""
Deepgram STT — streaming transcription over WebSocket.

The caller feeds raw PCM audio bytes (linear16, 16 kHz, mono) into
audio_queue. This module opens a Deepgram live connection, forwards
audio, and invokes on_transcript for every interim/final result.
Send None into audio_queue to signal end of stream.
"""
import asyncio
import logging
import ssl
from dataclasses import dataclass

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

from app.config import settings

logger = logging.getLogger(__name__)

# Deepgram closes the socket if it receives neither audio nor a keepalive within
# a short window (~10s). Send JSON KeepAlive periodically while waiting for audio.
_DEEPGRAM_KEEPALIVE_INTERVAL_SEC = 5.0

# Patched once: Python 3.13+ / OpenSSL 3.4+ may reject some TLS chains with
# "Basic Constraints of CA cert not marked critical" unless VERIFY_X509_STRICT
# is cleared. We still verify the server cert (default CA store).
_deepgram_ws_connect_patched = False


def _deepgram_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def _patch_deepgram_listen_websocket_connect() -> None:
    global _deepgram_ws_connect_patched
    if _deepgram_ws_connect_patched:
        return
    import deepgram.listen.v1.client as listen_v1_client

    _orig = listen_v1_client.websockets_client_connect

    def _patched(uri, *args, extra_headers=None, **kwargs):
        if str(uri).startswith("wss://") and kwargs.get("ssl") is None:
            kwargs["ssl"] = _deepgram_ssl_context()
        return _orig(uri, *args, extra_headers=extra_headers, **kwargs)

    listen_v1_client.websockets_client_connect = _patched  # type: ignore[assignment]
    _deepgram_ws_connect_patched = True


_patch_deepgram_listen_websocket_connect()


@dataclass
class TranscriptResult:
    text: str
    is_final: bool
    confidence: float = 0.0


async def transcribe_stream(
    audio_queue: asyncio.Queue,
    on_transcript,  # async callable(TranscriptResult)
) -> None:
    """
    Read audio bytes from audio_queue, stream to Deepgram, call on_transcript
    for each result. Stops when audio_queue yields None (sentinel value).
    """
    client = AsyncDeepgramClient(api_key=settings.deepgram_api_key)

    try:
        # Use "true"/"false" strings — Python True/False become "True"/"False" in the
        # query string and Deepgram returns HTTP 400 for invalid boolean params.
        async with client.listen.v1.connect(
            model=settings.deepgram_model,
            language=settings.deepgram_language,
            smart_format="true",
            interim_results="true",
            utterance_end_ms=1000,
            vad_events="true",
            encoding="linear16",
            sample_rate=16000,
            channels=1,
        ) as connection:

            async def _on_message(message):
                if not isinstance(message, ListenV1Results):
                    return
                try:
                    alternatives = message.channel.alternatives
                    if not alternatives:
                        return
                    sentence = alternatives[0].transcript
                    if not sentence:
                        return
                    confidence = alternatives[0].confidence
                    is_final = bool(message.is_final)
                    logger.debug(
                        "Deepgram transcript: is_final=%s confidence=%.2f text=%r",
                        is_final, confidence, sentence,
                    )
                    await on_transcript(TranscriptResult(
                        text=sentence,
                        is_final=is_final,
                        confidence=float(confidence),
                    ))
                except Exception as e:
                    logger.error("Error in Deepgram on_message callback: %s", e)

            async def _on_error(error):
                logger.error("Deepgram error: %s", error)

            async def _on_close(_):
                logger.info("Deepgram connection closed")

            connection.on(EventType.MESSAGE, _on_message)
            connection.on(EventType.ERROR, _on_error)
            connection.on(EventType.CLOSE, _on_close)

            listen_task = asyncio.create_task(connection.start_listening())
            logger.info("Deepgram STT connection opened (model=%s)", settings.deepgram_model)

            keepalive_stop = asyncio.Event()

            async def _keepalive_loop() -> None:
                try:
                    while True:
                        await asyncio.sleep(_DEEPGRAM_KEEPALIVE_INTERVAL_SEC)
                        if keepalive_stop.is_set():
                            break
                        try:
                            await connection.send_keep_alive()
                        except Exception as e:
                            logger.debug("Deepgram keepalive failed: %s", e)
                            break
                except asyncio.CancelledError:
                    raise

            keepalive_task = asyncio.create_task(_keepalive_loop())

            try:
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        logger.info("Deepgram STT received sentinel — closing connection")
                        break
                    await connection.send_media(chunk)
            except asyncio.CancelledError:
                logger.info("Deepgram STT task cancelled")
            finally:
                keepalive_stop.set()
                keepalive_task.cancel()
                try:
                    await keepalive_task
                except asyncio.CancelledError:
                    pass
                try:
                    await connection.send_close_stream()
                except Exception:
                    pass
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
                logger.info("Deepgram STT connection finished")
    except asyncio.CancelledError:
        logger.info("Deepgram STT task cancelled before connection established")
    except Exception as e:
        msg = str(e).strip() or repr(e)
        if not str(e).strip():
            logger.exception("Deepgram STT connection failed (%s)", type(e).__name__)
        else:
            logger.error("Deepgram STT connection failed: %s: %s", type(e).__name__, msg)
        raise  # surface to session handler so the WebSocket session closes cleanly
