"""
Deepgram STT — streaming transcription over WebSocket.

Usage pattern:
    async for transcript in transcribe_stream(audio_queue):
        if transcript.is_final:
            process(transcript.text)

The caller feeds raw PCM/WebM audio bytes into audio_queue.
This module opens a Deepgram streaming connection and yields
TranscriptResult objects as transcripts arrive.
"""
import asyncio
import logging
from dataclasses import dataclass
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    text: str
    is_final: bool
    confidence: float = 0.0


async def transcribe_stream(
    audio_queue: asyncio.Queue[bytes | None],
    on_transcript,  # async callable(TranscriptResult)
) -> None:
    """
    Consume audio bytes from audio_queue and call on_transcript for each result.
    Send None into audio_queue to signal end of stream.

    TODO (Plan 02): Implement full Deepgram streaming integration.
    - Create DeepgramClient with API key
    - Open LiveTranscriptionConnection with nova-2 model
    - Register handlers for Transcript, Error, Close events
    - Feed audio_queue bytes → connection.send()
    - Call on_transcript with TranscriptResult on each event
    - Handle graceful shutdown on None sentinel
    """
    raise NotImplementedError("Deepgram STT not yet implemented — see docs/plans/02-stt-deepgram.md")
