"""
Unit tests for WebSocket session message routing.
Tests the protocol layer (message parsing, routing, state management)
without real STT/TTS/LLM calls.
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.slides.content import SLIDES


class MockWebSocket:
    """Minimal WebSocket mock that records sent messages."""

    def __init__(self, messages_to_receive=None):
        self._to_receive = list(messages_to_receive or [])
        self.sent = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, data: str):
        self.sent.append(json.loads(data))

    async def receive_text(self):
        if not self._to_receive:
            # Simulate disconnect after all messages consumed
            from fastapi import WebSocketDisconnect
            raise WebSocketDisconnect(code=1000)
        return json.dumps(self._to_receive.pop(0))

    def sent_types(self):
        return [m["type"] for m in self.sent]


class TestInitialSlideAfterStart:
    @pytest.mark.asyncio
    async def test_sends_slide_change_after_start_message(self):
        from app.api.websocket import handle_session

        ws = MockWebSocket(messages_to_receive=[{"type": "start"}])

        # Mock STT to do nothing (no audio chunks sent)
        async def fake_transcribe(audio_queue, on_transcript):
            await asyncio.sleep(0)  # yield control

        with patch("app.api.websocket.transcribe_stream", fake_transcribe):
            await handle_session(ws)

        assert ws.accepted is True
        assert ws.sent[0]["type"] == "slide_change"
        assert ws.sent[0]["index"] == 0
        assert ws.sent[0]["slide"]["title"] == SLIDES[0].title

    @pytest.mark.asyncio
    async def test_initial_slide_has_bullets(self):
        from app.api.websocket import handle_session

        ws = MockWebSocket(messages_to_receive=[{"type": "start"}])

        async def fake_transcribe(audio_queue, on_transcript):
            await asyncio.sleep(0)

        with patch("app.api.websocket.transcribe_stream", fake_transcribe):
            await handle_session(ws)

        slide_msg = ws.sent[0]
        assert len(slide_msg["slide"]["bullets"]) > 0

    @pytest.mark.asyncio
    async def test_start_with_presentation_id_uses_that_deck(self):
        from app.api.websocket import handle_session
        from app.slides.drug_discovery import DRUG_DISCOVERY_SLIDES

        ws = MockWebSocket(
            messages_to_receive=[{"type": "start", "presentation_id": "drug-discovery"}]
        )

        async def fake_transcribe(audio_queue, on_transcript):
            await asyncio.sleep(0)

        with patch("app.api.websocket.transcribe_stream", fake_transcribe):
            await handle_session(ws)

        assert ws.sent[0]["type"] == "slide_change"
        assert ws.sent[0]["slide"]["title"] == DRUG_DISCOVERY_SLIDES[0].title


class TestPingPong:
    @pytest.mark.asyncio
    async def test_ping_receives_pong(self):
        from app.api.websocket import handle_session

        ws = MockWebSocket(messages_to_receive=[{"type": "ping"}])

        async def fake_transcribe(audio_queue, on_transcript):
            await asyncio.sleep(0)

        with patch("app.api.websocket.transcribe_stream", fake_transcribe):
            await handle_session(ws)

        assert "pong" in ws.sent_types()


class TestInterruptWithNoAgentRunning:
    @pytest.mark.asyncio
    async def test_interrupt_sends_tts_done_when_no_task(self):
        from app.api.websocket import handle_session

        ws = MockWebSocket(messages_to_receive=[{"type": "interrupt"}])

        async def fake_transcribe(audio_queue, on_transcript):
            await asyncio.sleep(0)

        with patch("app.api.websocket.transcribe_stream", fake_transcribe):
            await handle_session(ws)

        assert "tts_done" in ws.sent_types()


class TestMalformedMessage:
    @pytest.mark.asyncio
    async def test_malformed_json_does_not_crash_session(self):
        from fastapi import WebSocketDisconnect

        ws = MockWebSocket()
        receive_calls = [0]

        async def bad_receive():
            call = receive_calls[0]
            receive_calls[0] += 1
            if call == 0:
                return "this is not json {"
            raise WebSocketDisconnect(code=1000)

        ws.receive_text = bad_receive

        async def fake_transcribe(audio_queue, on_transcript):
            await asyncio.sleep(0)

        from app.api.websocket import handle_session
        with patch("app.api.websocket.transcribe_stream", fake_transcribe):
            # Should not raise — malformed JSON is handled gracefully
            await handle_session(ws)


class TestRunAgent:
    @pytest.mark.asyncio
    async def test_run_agent_always_sends_tts_done(self):
        from app.api.websocket import run_agent
        from app.agent.state import AgentState

        ws = MockWebSocket()
        interrupt = asyncio.Event()
        state: AgentState = {
            "current_slide": 0, "target_slide": None, "messages": [],
            "transcript": "", "response_text": "", "slide_changed": False,
            "interrupted": False, "should_navigate": False,
            "presentation_id": "clinical-trials",
        }

        mock_result = {
            "current_slide": 0, "target_slide": None, "should_navigate": False,
            "response_text": "Test response text.", "slide_changed": False,
            "messages": [],
        }

        with patch("app.api.websocket.agent_graph") as mock_graph, \
             patch("app.api.websocket.synthesize_stream") as mock_tts:
            mock_graph.ainvoke = AsyncMock(return_value=mock_result)

            # TTS yields no chunks
            async def empty_tts(text, event):
                return
                yield  # make it an async generator

            mock_tts.return_value = empty_tts("", interrupt)

            await run_agent(ws, state, "hello", interrupt)

        assert "tts_done" in ws.sent_types()

    @pytest.mark.asyncio
    async def test_run_agent_sends_tts_done_even_on_graph_error(self):
        from app.api.websocket import run_agent
        from app.agent.state import AgentState

        ws = MockWebSocket()
        interrupt = asyncio.Event()
        state: AgentState = {
            "current_slide": 0, "target_slide": None, "messages": [],
            "transcript": "", "response_text": "", "slide_changed": False,
            "interrupted": False, "should_navigate": False,
            "presentation_id": "clinical-trials",
        }

        with patch("app.api.websocket.agent_graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

            with pytest.raises(RuntimeError):
                await run_agent(ws, state, "hello", interrupt)

        # tts_done must still have been sent
        assert "tts_done" in ws.sent_types()

    @pytest.mark.asyncio
    async def test_run_agent_sends_slide_change_on_navigation(self):
        from app.api.websocket import run_agent
        from app.agent.state import AgentState

        ws = MockWebSocket()
        interrupt = asyncio.Event()
        state: AgentState = {
            "current_slide": 0, "target_slide": None, "messages": [],
            "transcript": "recruitment", "response_text": "", "slide_changed": False,
            "interrupted": False, "should_navigate": False,
            "presentation_id": "clinical-trials",
        }

        # Graph returns navigation to slide 1
        mock_result = {
            "current_slide": 1,  # changed from 0
            "target_slide": 1,
            "should_navigate": True,
            "response_text": "Patient recruitment response",
            "slide_changed": False,  # respond_node resets this
            "messages": [],
        }

        with patch("app.api.websocket.agent_graph") as mock_graph, \
             patch("app.api.websocket.synthesize_stream") as mock_tts:
            mock_graph.ainvoke = AsyncMock(return_value=mock_result)

            async def empty_tts(text, event):
                return
                yield

            mock_tts.return_value = empty_tts("", interrupt)

            await run_agent(ws, state, "recruitment", interrupt)

        types = ws.sent_types()
        assert "slide_change" in types
        slide_msg = next(m for m in ws.sent if m["type"] == "slide_change")
        assert slide_msg["index"] == 1
        assert slide_msg["slide"]["title"] == SLIDES[1].title
