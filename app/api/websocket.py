"""
WebSocket session handler.

Single endpoint: ws://host/ws
One WebSocket connection = one presentation session.

Message protocol:
  Client -> Server:
    {"type": "start"}
    {"type": "audio_chunk", "data": "<base64 PCM 16kHz mono int16>"}
    {"type": "interrupt"}
    {"type": "ping"}

  Server -> Client:
    {"type": "transcript",   "text": "...", "is_final": bool}
    {"type": "slide_change", "index": N, "slide": {title, bullets}}
    {"type": "agent_text",   "text": "..."}
    {"type": "tts_chunk",    "data": "<base64 MP3>"}
    {"type": "tts_done"}
    {"type": "error",        "message": "..."}
    {"type": "pong"}
"""
import asyncio
import base64
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from app.agent.state import AgentState
from app.services.stt import transcribe_stream, TranscriptResult
from app.slides.content import get_slide

logger = logging.getLogger(__name__)


async def handle_session(websocket: WebSocket) -> None:
    """
    Main WebSocket session lifecycle.

    Pipeline:
    1. Accept connection
    2. Send initial slide_change for slide 0
    3. Start Deepgram STT as background task
    4. Receive loop: route audio_chunk -> audio_queue, interrupt -> interrupt_event
    5. on_transcript callback: forward to client; on final -> run_agent (Plan 03)
    6. Graceful shutdown on disconnect or error
    """
    await websocket.accept()
    logger.info("WebSocket session started")

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
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=100)
    agent_task: asyncio.Task | None = None

    initial_slide = get_slide(0)
    await _send(websocket, {
        "type": "slide_change",
        "index": 0,
        "slide": {"title": initial_slide.title, "bullets": initial_slide.bullets},
    })

    async def on_transcript(result: TranscriptResult) -> None:
        nonlocal agent_task

        await _send(websocket, {
            "type": "transcript",
            "text": result.text,
            "is_final": result.is_final,
        })

        if not result.is_final or not result.text.strip():
            return

        logger.info("Final transcript: %r (confidence=%.2f)", result.text, result.confidence)

        if agent_task and not agent_task.done():
            logger.info("Cancelling previous agent task for new transcript")
            interrupt_event.set()
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            interrupt_event.clear()

        agent_task = asyncio.create_task(
            _agent_stub(websocket, state, result.text, interrupt_event)
        )

    stt_task = asyncio.create_task(
        transcribe_stream(audio_queue, on_transcript)
    )

    try:
        async for raw in _receive_loop(websocket):
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio_chunk":
                audio_bytes = base64.b64decode(msg["data"])
                try:
                    audio_queue.put_nowait(audio_bytes)
                except asyncio.QueueFull:
                    logger.warning("Audio queue full — dropping chunk")

            elif msg_type == "interrupt":
                interrupt_event.set()
                if agent_task and not agent_task.done():
                    agent_task.cancel()
                    try:
                        await agent_task
                    except asyncio.CancelledError:
                        pass
                await _send(websocket, {"type": "tts_done"})
                interrupt_event.clear()

            elif msg_type == "ping":
                await _send(websocket, {"type": "pong"})

            elif msg_type == "start":
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket session ended by client")
    except Exception as e:
        logger.exception("Unexpected session error: %s", e)
        try:
            await _send(websocket, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        await audio_queue.put(None)
        stt_task.cancel()
        try:
            await asyncio.wait_for(stt_task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        if agent_task and not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass

        logger.info("WebSocket session cleaned up")


async def _receive_loop(websocket: WebSocket):
    """Yield raw text messages until the client disconnects or the socket closes."""
    while True:
        try:
            yield await websocket.receive_text()
        except WebSocketDisconnect:
            raise
        except RuntimeError as e:
            # After close (client or server), Starlette may set application_state such that
            # receive_text() fails with this message — treat as disconnect, not an app bug.
            if "not connected" in str(e).lower():
                raise WebSocketDisconnect(code=1000) from e
            raise


async def _agent_stub(
    websocket: WebSocket,
    state: AgentState,
    transcript: str,
    interrupt_event: asyncio.Event,
) -> None:
    """
    Placeholder agent task — replaced in Plan 03.
    Echoes the transcript back as agent_text so the frontend can be tested.
    """
    logger.info("Agent stub called with transcript: %r", transcript)
    await _send(websocket, {
        "type": "agent_text",
        "text": f"[stub] You said: {transcript}",
    })
    await _send(websocket, {"type": "tts_done"})


async def _send(websocket: WebSocket, payload: dict) -> None:
    """Send JSON message to client. Silently drops if connection is closed."""
    try:
        await websocket.send_text(json.dumps(payload))
    except Exception:
        pass
