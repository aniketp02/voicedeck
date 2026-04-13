"""
WebSocket session handler.

Single endpoint: ws://host/ws
One WebSocket connection = one presentation session.

Message protocol:
  Client -> Server:
    {"type": "start", "presentation_id": "<optional id>"}
    {"type": "audio_chunk", "data": "<base64 PCM 16kHz mono int16>"}
    {"type": "interrupt"}
    {"type": "navigate", "index": <int>}   # manual slide change (e.g. keyboard)
    {"type": "start_auto_narrate"}         # begin autonomous slide-by-slide narration
    {"type": "stop_auto_narrate"}          # stop auto-narration
    {"type": "tts_playback_done"}          # local audio playback ended (audio.ended event)
    {"type": "ping"}

  Server -> Client:
    {"type": "transcript",          "text": "...", "is_final": bool, "speech_final": bool}
    {"type": "slide_change",        "index": N, "slide": {title, bullets}}
    {"type": "agent_text",          "text": "..."}
    {"type": "tts_chunk",           "data": "<base64 MP3>"}
    {"type": "tts_done"}
    {"type": "auto_narrate_complete"}      # all slides narrated (or loop stopped)
    {"type": "error",               "message": "..."}
    {"type": "pong"}
"""
import asyncio
import base64
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import agent_graph, routing_graph
from app.agent.narrate import narrate_slide, narrate_slide_stream
from app.agent.nodes import build_respond_prompt
from app.agent.state import AgentState
from app.services.llm import chat_completion_stream, sentence_stream
from app.services.stt import transcribe_stream, TranscriptResult
from app.services.tts import synthesize_stream
from app.slides.presentations import DEFAULT_PRESENTATION_ID, get_presentation

logger = logging.getLogger(__name__)


def should_dispatch_agent_turn(result: TranscriptResult) -> bool:
    """
    True when this STT result should trigger the LangGraph agent.
    Uses speech_final (utterance end), not is_final (segment boundaries).
    """
    return bool(result.speech_final and result.text.strip())


async def run_agent(
    websocket: WebSocket,
    state: AgentState,
    transcript: str,
    interrupt_event: asyncio.Event,
) -> None:
    """
    Run the agent pipeline for one user utterance with streaming LLM → TTS.

    Phase 1 — routing: uses routing_graph (understand + navigate) to resolve
    intent and any slide navigation without blocking on response generation.

    Phase 2 — streaming: builds the respond prompt from state, then streams
    tokens from the LLM, splits them into sentences, and pipes each sentence
    to TTS immediately. This lets audio start after the first sentence (~500ms)
    rather than waiting for the full response (~1500ms).

    agent_text is sent incrementally (cumulative) after each sentence so the
    orb and conversation footer display text as soon as audio starts.
    """
    tts_done_sent = False
    try:
        state["messages"] = state["messages"] + [HumanMessage(content=transcript)]
        state["transcript"] = transcript
        state["slide_changed"] = False
        state["should_navigate"] = False
        state["target_slide"] = None

        slide_before = state["current_slide"]

        # Phase 1: routing (understand + optional navigate) — no respond node
        result = await routing_graph.ainvoke(state)

        for key in (
            "current_slide",
            "target_slide",
            "should_navigate",
            "end_session",
            "slide_changed",
            "interrupted",
            "messages",
        ):
            if key in result:
                state[key] = result[key]

        if result.get("current_slide", slide_before) != slide_before:
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

        # Phase 2: streaming response — sentence by sentence → TTS
        # Send incremental agent_text after each sentence so the orb and
        # footer display text as soon as the first audio chunk plays.
        system, user_msg = build_respond_prompt(state)
        response_parts: list[str] = []
        chunk_count = 0

        async for sentence in sentence_stream(chat_completion_stream(system, user_msg)):
            if interrupt_event.is_set():
                break
            response_parts.append(sentence)
            await _send(websocket, {"type": "agent_text", "text": " ".join(response_parts)})
            async for chunk in synthesize_stream(sentence, interrupt_event):
                chunk_count += 1
                await _send(
                    websocket,
                    {
                        "type": "tts_chunk",
                        "data": base64.b64encode(chunk).decode(),
                    },
                )

        full_response = " ".join(response_parts)
        if full_response:
            state["messages"] = state["messages"] + [AIMessage(content=full_response)]
            logger.info(
                "Streamed %d TTS chunks, sent agent_text: %d chars (%d sentences)",
                chunk_count,
                len(full_response),
                len(response_parts),
            )

        # Clear end_session after handling so the next turn starts clean
        if state.get("end_session"):
            state["end_session"] = False

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
        "end_session": False,
        "presentation_id": presentation_id,
    }

    interrupt_event = asyncio.Event()
    # Set by the client's "tts_playback_done" message — used by auto_narrate_loop
    # to know when the audio element has actually finished playing before advancing.
    playback_done_event = asyncio.Event()
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=100)
    agent_task: asyncio.Task | None = None
    auto_narrate_task: asyncio.Task | None = None

    async def on_transcript(result: TranscriptResult) -> None:
        nonlocal agent_task

        await _send(websocket, {
            "type": "transcript",
            "text": result.text,
            "is_final": result.is_final,
            "speech_final": result.speech_final,
        })

        if not should_dispatch_agent_turn(result):
            return

        logger.info(
            "Speech final: %r (confidence=%.2f)",
            result.text,
            result.confidence,
        )

        if agent_task and not agent_task.done():
            logger.info("Cancelling previous agent task for new utterance")
            state["interrupted"] = True
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

    async def auto_narrate_loop() -> None:
        """
        Autonomously narrate each slide from the current position to the end.

        For each slide: sends slide_change, streams narration TTS, then waits for
        the client's tts_playback_done before advancing.

        Optimisation — speculative pre-generation: while the current slide's TTS
        audio plays on the client, we kick off the LLM call for the *next* slide's
        narration as a background task. By the time the client signals playback_done
        and the brief inter-slide pause elapses, the text is ready — eliminating the
        ~800ms LLM wait from the perceived gap between slides.

        Inter-slide pause: 0.8 s (down from 1.5 s) — LLM latency no longer adds to it.
        """
        tts_done_sent = False
        prefetch_task: asyncio.Task | None = None

        try:
            presentation = get_presentation(state["presentation_id"])
            slides = presentation.slides
            prev_slide = None

            for idx in range(state["current_slide"], len(slides)):
                if interrupt_event.is_set():
                    break

                # Advance to next slide (skip on first iteration when already there)
                if idx != state["current_slide"] or prev_slide is not None:
                    state["current_slide"] = idx
                    slide = slides[idx]
                    await _send(websocket, {
                        "type": "slide_change",
                        "index": idx,
                        "slide": {"title": slide.title, "bullets": slide.bullets},
                    })
                    logger.info("auto_narrate: slide_change → %d", idx)

                if interrupt_event.is_set():
                    break

                slide = slides[state["current_slide"]]

                # Use pre-generated narration if the prefetch task completed.
                # Otherwise stream sentences directly — TTS starts after first sentence.
                prefetched_text: str | None = None
                if prefetch_task is not None and prefetch_task.done():
                    try:
                        prefetched_text = prefetch_task.result()
                        logger.info("auto_narrate: using pre-generated narration for slide %d", idx)
                    except Exception as e:
                        logger.warning("auto_narrate: prefetch failed (%s), will stream", e)
                    prefetch_task = None

                if interrupt_event.is_set():
                    break

                # Clear BEFORE any TTS so a fast tts_playback_done that arrives
                # in the window between tts_done and the wait() call is not lost.
                playback_done_event.clear()

                chunk_count = 0
                tts_done_sent = False
                next_idx = idx + 1
                prefetch_started = False
                narration_parts: list[str] = []

                if prefetched_text:
                    # Prefetch available: full text known upfront — send agent_text immediately
                    # so orb/footer display text as soon as audio starts.
                    narration_parts = [prefetched_text]
                    await _send(websocket, {"type": "agent_text", "text": prefetched_text})
                    async for chunk in synthesize_stream(prefetched_text, interrupt_event):
                        chunk_count += 1
                        await _send(websocket, {
                            "type": "tts_chunk",
                            "data": base64.b64encode(chunk).decode(),
                        })
                        if not prefetch_started and next_idx < len(slides):
                            prefetch_started = True
                            prefetch_task = asyncio.create_task(
                                narrate_slide(slides[next_idx], presentation, slide)
                            )
                else:
                    # No prefetch: stream sentences → TTS each sentence.
                    # Send cumulative agent_text before each sentence's TTS so the
                    # orb and footer update incrementally (same pattern as run_agent).
                    async for sentence in narrate_slide_stream(slide, presentation, prev_slide):
                        if interrupt_event.is_set():
                            break
                        narration_parts.append(sentence)
                        await _send(websocket, {"type": "agent_text", "text": " ".join(narration_parts)})
                        async for chunk in synthesize_stream(sentence, interrupt_event):
                            chunk_count += 1
                            await _send(websocket, {
                                "type": "tts_chunk",
                                "data": base64.b64encode(chunk).decode(),
                            })
                            if not prefetch_started and next_idx < len(slides):
                                prefetch_started = True
                                prefetch_task = asyncio.create_task(
                                    narrate_slide(slides[next_idx], presentation, slide)
                                )

                logger.info("auto_narrate: streamed %d TTS chunks for slide %d", chunk_count, idx)

                await _send(websocket, {"type": "tts_done"})
                tts_done_sent = True

                if interrupt_event.is_set():
                    break

                # Wait for client to signal playback complete.
                # Timeout = 45s: enough for a 2-3 sentence narration (~15-25s audio)
                # plus a ~20s buffer. Acts as a safety net if the client's audio.ended
                # event doesn't fire (MSE/autoplay edge case in auto-narrate mode).
                # On timeout, log a warning and advance — better than hanging forever.
                playback_task = asyncio.create_task(playback_done_event.wait())
                interrupt_task = asyncio.create_task(interrupt_event.wait())
                try:
                    done, pending = await asyncio.wait(
                        {playback_task, interrupt_task},
                        timeout=45.0,
                    )
                    if not done:
                        logger.warning(
                            "auto_narrate: tts_playback_done not received after 45s for slide %d "
                            "(client audio.ended may not have fired) — advancing anyway",
                            idx,
                        )
                    for t in pending:
                        t.cancel()
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass
                except asyncio.CancelledError:
                    playback_task.cancel()
                    interrupt_task.cancel()
                    raise

                if interrupt_event.is_set():
                    break

                # Brief natural pause before next slide (LLM latency already absorbed above)
                await asyncio.sleep(0.8)

                if interrupt_event.is_set():
                    break

                # If prefetch task is still running, wait for it now (short — usually done)
                if prefetch_task is not None and not prefetch_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(prefetch_task), timeout=3.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass

                prev_slide = slide

        except asyncio.CancelledError:
            logger.info("auto_narrate_loop: cancelled")
            if prefetch_task and not prefetch_task.done():
                prefetch_task.cancel()
            raise
        except Exception as e:
            logger.exception("auto_narrate_loop error: %s", e)
            await _send(websocket, {"type": "error", "message": f"Auto-narration error: {e}"})
        finally:
            if prefetch_task and not prefetch_task.done():
                prefetch_task.cancel()
            if not tts_done_sent:
                await _send(websocket, {"type": "tts_done"})
            await _send(websocket, {"type": "auto_narrate_complete"})
            logger.info("auto_narrate_loop: complete")

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
                state["interrupted"] = True
                interrupt_event.set()
                agent_running = bool(agent_task and not agent_task.done())
                narrate_running = bool(auto_narrate_task and not auto_narrate_task.done())
                if agent_running:
                    agent_task.cancel()
                    try:
                        await agent_task
                    except (asyncio.CancelledError, Exception):
                        pass
                if narrate_running:
                    auto_narrate_task.cancel()
                    try:
                        await auto_narrate_task
                    except (asyncio.CancelledError, Exception):
                        pass
                if not agent_running and not narrate_running:
                    await _send(websocket, {"type": "tts_done"})
                interrupt_event.clear()

            elif msg_type == "tts_playback_done":
                # Client's audio element fired the 'ended' event — playback is complete.
                # Used by auto_narrate_loop to know it's safe to advance to the next slide.
                playback_done_event.set()

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

            elif msg_type == "start_auto_narrate":
                logger.info("start_auto_narrate received")
                # Cancel any in-flight agent or narration task first
                interrupt_event.set()
                for task in (agent_task, auto_narrate_task):
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass
                interrupt_event.clear()
                state["interrupted"] = False
                auto_narrate_task = asyncio.create_task(auto_narrate_loop())

            elif msg_type == "stop_auto_narrate":
                logger.info("stop_auto_narrate received")
                interrupt_event.set()
                if auto_narrate_task and not auto_narrate_task.done():
                    auto_narrate_task.cancel()
                    try:
                        await auto_narrate_task
                    except (asyncio.CancelledError, Exception):
                        pass
                interrupt_event.clear()

            elif msg_type == "navigate":
                # Client-initiated manual slide navigation (keyboard arrow keys).
                target_index = msg.get("index")
                if not isinstance(target_index, int):
                    logger.warning("navigate message missing valid index: %r", msg)
                else:
                    presentation = get_presentation(state["presentation_id"])
                    if not (0 <= target_index < len(presentation.slides)):
                        logger.warning(
                            "navigate index %d out of range for %d-slide presentation",
                            target_index,
                            len(presentation.slides),
                        )
                    else:
                        if agent_task and not agent_task.done():
                            interrupt_event.set()
                            agent_task.cancel()
                            try:
                                await agent_task
                            except (asyncio.CancelledError, Exception):
                                pass
                            interrupt_event.clear()
                            await _send(websocket, {"type": "tts_done"})

                        state["current_slide"] = target_index
                        state["interrupted"] = False
                        slide = presentation.slides[target_index]
                        await _send(websocket, {
                            "type": "slide_change",
                            "index": target_index,
                            "slide": {"title": slide.title, "bullets": slide.bullets},
                        })
                        logger.info("Manual navigation to slide %d", target_index)

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

        if auto_narrate_task and not auto_narrate_task.done():
            auto_narrate_task.cancel()
            try:
                await auto_narrate_task
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
