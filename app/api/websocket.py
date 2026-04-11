"""
WebSocket session handler.

Single endpoint: ws://host/ws
One WebSocket connection = one presentation session.

Message protocol:
  Client -> Server:
    {"type": "start", "presentation_id": "<optional id>"}
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
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import agent_graph
from app.agent.state import AgentState
from app.services.stt import transcribe_stream, TranscriptResult
from app.services.tts import synthesize_stream
from app.slides.presentations import DEFAULT_PRESENTATION_ID, get_presentation

logger = logging.getLogger(__name__)


async def run_agent(
    websocket: WebSocket,
    state: AgentState,
    transcript: str,
    interrupt_event: asyncio.Event,
) -> None:
    """
    Run the LangGraph agent pipeline for one user utterance.

    Merges graph output into session state, sends slide_change when the slide
    index changes, agent_text, streaming tts_chunk messages, then tts_done.
    """
    tts_done_sent = False
    try:
        state["messages"] = state["messages"] + [HumanMessage(content=transcript)]
        state["transcript"] = transcript
        state["slide_changed"] = False
        state["should_navigate"] = False
        state["target_slide"] = None

        slide_before = state["current_slide"]

        result = await agent_graph.ainvoke(state)

        for key in (
            "current_slide",
            "target_slide",
            "should_navigate",
            "response_text",
            "slide_changed",
            "messages",
        ):
            if key in result:
                state[key] = result[key]

        # respond_node clears slide_changed; detect navigation by index change
        if result.get("current_slide") != slide_before:
            presentation = get_presentation(state["presentation_id"])
            slide = presentation.slides[state["current_slide"]]
            await _send(
                websocket,
                {
                    "type": "slide_change",
                    "index": state["current_slide"],
                    "slide": {"title": slide.title, "bullets": slide.bullets},
                },
            )
            logger.info("Sent slide_change to client: index=%d", state["current_slide"])

        response_text = result.get("response_text", "")
        if response_text:
            state["messages"] = state["messages"] + [AIMessage(content=response_text)]
            await _send(websocket, {"type": "agent_text", "text": response_text})
            logger.info("Sent agent_text: %d chars", len(response_text))

        if response_text and not interrupt_event.is_set():
            chunk_count = 0
            async for chunk in synthesize_stream(response_text, interrupt_event):
                chunk_count += 1
                await _send(
                    websocket,
                    {
                        "type": "tts_chunk",
                        "data": base64.b64encode(chunk).decode(),
                    },
                )
            logger.info("Streamed %d TTS chunks to client", chunk_count)

        await _send(websocket, {"type": "tts_done"})
        tts_done_sent = True

    except asyncio.CancelledError:
        logger.info("run_agent cancelled (interrupt or new transcript)")
        raise
    except Exception as e:
        logger.exception("run_agent error: %s", e)
        await _send(websocket, {"type": "error", "message": f"Agent error: {e}"})
        raise
    finally:
        if not tts_done_sent:
            await _send(websocket, {"type": "tts_done"})


async def handle_session(websocket: WebSocket) -> None:
    """
    Main WebSocket session lifecycle.

    Pipeline:
    1. Accept connection
    2. Wait for {"type": "start", "presentation_id": "..."} from client
    3. Send initial slide_change for slide 0 of the chosen presentation
    4. Start Deepgram STT as background task
    5. Receive loop: route audio_chunk → audio_queue, interrupt → interrupt_event
    6. on_transcript callback: forward to client; on final → run_agent
    7. Graceful shutdown on disconnect or error
    """
    await websocket.accept()
    logger.info("WebSocket session started")

    presentation_id = DEFAULT_PRESENTATION_ID

    state: AgentState = {
        "current_slide": 0,
        "target_slide": None,
        "messages": [],
        "transcript": "",
        "response_text": "",
        "slide_changed": False,
        "interrupted": False,
        "should_navigate": False,
        "presentation_id": presentation_id,
    }

    interrupt_event = asyncio.Event()
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=100)
    agent_task: asyncio.Task | None = None

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
            logger.info("Cancelling previous agent task for new utterance")
            interrupt_event.set()
            agent_task.cancel()
            try:
                await agent_task
            except (asyncio.CancelledError, Exception):
                pass
            interrupt_event.clear()

        agent_task = asyncio.create_task(
            run_agent(websocket, state, result.text, interrupt_event)
        )

    stt_task = asyncio.create_task(
        transcribe_stream(audio_queue, on_transcript)
    )

    try:
        async for raw in _receive_loop(websocket):
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Malformed WebSocket message (not JSON): %r", raw[:100])
                continue

            msg_type = msg.get("type")

            if msg_type == "audio_chunk":
                audio_bytes = base64.b64decode(msg["data"])
                try:
                    audio_queue.put_nowait(audio_bytes)
                except asyncio.QueueFull:
                    logger.warning("Audio queue full — dropping chunk")

            elif msg_type == "interrupt":
                logger.info("Interrupt signal received from client")
                interrupt_event.set()
                task_was_running = bool(agent_task and not agent_task.done())
                if task_was_running:
                    agent_task.cancel()
                    try:
                        await agent_task
                    except (asyncio.CancelledError, Exception):
                        pass
                else:
                    await _send(websocket, {"type": "tts_done"})
                interrupt_event.clear()

            elif msg_type == "ping":
                await _send(websocket, {"type": "pong"})

            elif msg_type == "start":
                requested_id = msg.get("presentation_id", DEFAULT_PRESENTATION_ID)
                try:
                    presentation = get_presentation(requested_id)
                    presentation_id = requested_id
                except KeyError:
                    logger.warning(
                        "Unknown presentation_id %r — using default %r",
                        requested_id,
                        DEFAULT_PRESENTATION_ID,
                    )
                    presentation = get_presentation(DEFAULT_PRESENTATION_ID)

                state["presentation_id"] = presentation.meta.id
                initial_slide = presentation.slides[0]
                await _send(websocket, {
                    "type": "slide_change",
                    "index": 0,
                    "slide": {"title": initial_slide.title, "bullets": initial_slide.bullets},
                })
                logger.info(
                    "Session started: presentation=%r slides=%d",
                    presentation.meta.id,
                    presentation.meta.slide_count,
                )

            else:
                logger.debug("Unknown message type: %r", msg_type)

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


async def _send(websocket: WebSocket, payload: dict) -> None:
    """Send JSON message to client. Silently drops if connection is closed."""
    try:
        await websocket.send_text(json.dumps(payload))
    except Exception:
        pass
