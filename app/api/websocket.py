"""
WebSocket session handler.

Single endpoint: ws://host/ws
One WebSocket connection = one presentation session.

Message protocol:
  Client → Server:
    {"type": "start"}
    {"type": "audio_chunk", "data": "<base64 PCM/WebM>"}
    {"type": "interrupt"}

  Server → Client:
    {"type": "transcript",   "text": "...", "is_final": bool}
    {"type": "slide_change", "index": N, "slide": {title, bullets}}
    {"type": "agent_text",   "text": "..."}
    {"type": "tts_chunk",    "data": "<base64 MP3>"}
    {"type": "tts_done"}
    {"type": "error",        "message": "..."}

TODO (Plan 02–05): Implement the full session loop.
See docs/plans/02-stt-deepgram.md through 05-interruption.md.
"""
import asyncio
import base64
import json
import logging
from fastapi import WebSocket, WebSocketDisconnect
from app.agent.graph import agent_graph
from app.agent.state import AgentState
from app.slides.content import SLIDES, get_slide

logger = logging.getLogger(__name__)


async def handle_session(websocket: WebSocket) -> None:
    """
    Main WebSocket session loop.

    TODO: Implement the full pipeline:
    1. Accept connection, send initial slide data
    2. Start Deepgram STT stream (audio_queue → transcribe_stream)
    3. On final transcript → run agent_graph → get response_text + slide action
    4. If slide changed → send slide_change message
    5. Send agent_text message
    6. Stream ElevenLabs TTS chunks → send tts_chunk messages → send tts_done
    7. Handle interrupt: set interrupt_event, cancel TTS, restart STT listen
    8. Handle WebSocketDisconnect gracefully
    """
    await websocket.accept()
    logger.info("WebSocket session started")

    # Initial state
    state: AgentState = {
        "current_slide": 0,
        "target_slide": None,
        "messages": [],
        "transcript": "",
        "response_text": "",
        "slide_changed": False,
        "interrupted": False,
        "should_navigate": False,
    }

    interrupt_event = asyncio.Event()
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    # Send initial slide
    slide = get_slide(0)
    await _send(websocket, {
        "type": "slide_change",
        "index": 0,
        "slide": {"title": slide.title, "bullets": slide.bullets},
    })

    try:
        # TODO: start STT background task and main receive loop
        await _placeholder_loop(websocket, state, interrupt_event, audio_queue)
    except WebSocketDisconnect:
        logger.info("WebSocket session ended")
    except Exception as e:
        logger.exception("Session error: %s", e)
        await _send(websocket, {"type": "error", "message": str(e)})


async def _placeholder_loop(websocket, state, interrupt_event, audio_queue):
    """
    Temporary echo loop — replaced by full pipeline in Plan 02.
    Accepts messages and echoes them back so the frontend can be tested.
    """
    while True:
        raw = await websocket.receive_text()
        msg = json.loads(raw)
        msg_type = msg.get("type")

        if msg_type == "interrupt":
            interrupt_event.set()
            await _send(websocket, {"type": "tts_done"})
            interrupt_event.clear()

        elif msg_type == "audio_chunk":
            # In the real implementation, forward to Deepgram
            pass

        elif msg_type == "ping":
            await _send(websocket, {"type": "pong"})


async def _send(websocket: WebSocket, payload: dict) -> None:
    await websocket.send_text(json.dumps(payload))
