"""
Unit tests for LangGraph agent nodes.
All OpenAI calls are mocked — no API keys required.
"""
import pytest
from unittest.mock import AsyncMock, patch

from app.agent.nodes import understand_node, navigate_node, respond_node, should_navigate
from app.agent.state import AgentState


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "current_slide": 0,
        "target_slide": None,
        "messages": [],
        "transcript": "test transcript",
        "response_text": "",
        "slide_changed": False,
        "interrupted": False,
        "should_navigate": False,
        "presentation_id": "clinical-trials",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# understand_node
# ---------------------------------------------------------------------------

class TestUnderstandNode:
    @pytest.mark.asyncio
    async def test_navigate_to_valid_slide(self):
        state = _make_state(current_slide=0, transcript="tell me about patient recruitment")
        mock_response = {
            "should_navigate": True,
            "target_slide": 1,
            "intent_summary": "User wants to hear about patient recruitment",
        }
        with patch("app.agent.nodes.chat_completion_json", AsyncMock(return_value=mock_response)):
            result = await understand_node(state)

        assert result["should_navigate"] is True
        assert result["target_slide"] == 1

    @pytest.mark.asyncio
    async def test_no_navigation_for_current_slide_question(self):
        state = _make_state(current_slide=0, transcript="what is the main problem")
        mock_response = {
            "should_navigate": False,
            "target_slide": None,
            "intent_summary": "User asking about the current slide topic",
        }
        with patch("app.agent.nodes.chat_completion_json", AsyncMock(return_value=mock_response)):
            result = await understand_node(state)

        assert result["should_navigate"] is False
        assert result["target_slide"] is None

    @pytest.mark.asyncio
    async def test_ignores_out_of_range_target(self):
        state = _make_state(current_slide=0, transcript="go to slide 99")
        mock_response = {"should_navigate": True, "target_slide": 99, "intent_summary": ""}
        with patch("app.agent.nodes.chat_completion_json", AsyncMock(return_value=mock_response)):
            result = await understand_node(state)

        assert result["should_navigate"] is False
        assert result["target_slide"] is None

    @pytest.mark.asyncio
    async def test_ignores_navigation_to_same_slide(self):
        state = _make_state(current_slide=2, transcript="more about this")
        mock_response = {"should_navigate": True, "target_slide": 2, "intent_summary": ""}
        with patch("app.agent.nodes.chat_completion_json", AsyncMock(return_value=mock_response)):
            result = await understand_node(state)

        assert result["should_navigate"] is False
        assert result["target_slide"] is None

    @pytest.mark.asyncio
    async def test_ignores_navigate_true_with_null_target(self):
        state = _make_state(current_slide=0, transcript="something vague")
        mock_response = {"should_navigate": True, "target_slide": None, "intent_summary": ""}
        with patch("app.agent.nodes.chat_completion_json", AsyncMock(return_value=mock_response)):
            result = await understand_node(state)

        assert result["should_navigate"] is False

    @pytest.mark.asyncio
    async def test_handles_non_integer_target(self):
        state = _make_state(current_slide=0, transcript="test")
        mock_response = {"should_navigate": True, "target_slide": "two", "intent_summary": ""}
        with patch("app.agent.nodes.chat_completion_json", AsyncMock(return_value=mock_response)):
            result = await understand_node(state)

        assert result["should_navigate"] is False
        assert result["target_slide"] is None

    @pytest.mark.asyncio
    async def test_handles_empty_llm_response(self):
        state = _make_state(current_slide=0, transcript="test")
        with patch("app.agent.nodes.chat_completion_json", AsyncMock(return_value={})):
            result = await understand_node(state)

        assert result["should_navigate"] is False
        assert result["target_slide"] is None


# ---------------------------------------------------------------------------
# navigate_node
# ---------------------------------------------------------------------------

class TestNavigateNode:
    @pytest.mark.asyncio
    async def test_updates_current_slide(self):
        state = _make_state(current_slide=0, target_slide=3)
        result = await navigate_node(state)

        assert result["current_slide"] == 3
        assert result["slide_changed"] is True

    @pytest.mark.asyncio
    async def test_navigate_to_last_slide(self):
        state = _make_state(current_slide=4, target_slide=5)
        result = await navigate_node(state)

        assert result["current_slide"] == 5
        assert result["slide_changed"] is True


# ---------------------------------------------------------------------------
# respond_node
# ---------------------------------------------------------------------------

class TestRespondNode:
    @pytest.mark.asyncio
    async def test_generates_response_text(self):
        state = _make_state(current_slide=0, transcript="what is the problem")
        expected_response = "Clinical trials are fundamentally broken..."
        with patch("app.agent.nodes.chat_completion", AsyncMock(return_value=expected_response)):
            result = await respond_node(state)

        assert result["response_text"] == expected_response
        assert result["slide_changed"] is False  # always reset

    @pytest.mark.asyncio
    async def test_includes_navigation_context_when_slide_changed(self):
        state = _make_state(current_slide=1, transcript="tell me about recruitment", slide_changed=True)
        captured_user_msg = []

        async def capture_call(system, user_msg):
            captured_user_msg.append(user_msg)
            return "Patient recruitment response"

        with patch("app.agent.nodes.chat_completion", capture_call):
            await respond_node(state)

        assert len(captured_user_msg) == 1
        assert "navigated" in captured_user_msg[0].lower() or "slide 1" in captured_user_msg[0]

    @pytest.mark.asyncio
    async def test_resets_slide_changed_flag(self):
        state = _make_state(current_slide=0, slide_changed=True)
        with patch("app.agent.nodes.chat_completion", AsyncMock(return_value="response")):
            result = await respond_node(state)

        assert result["slide_changed"] is False


# ---------------------------------------------------------------------------
# should_navigate (conditional edge)
# ---------------------------------------------------------------------------

class TestShouldNavigate:
    def test_returns_navigate_when_both_flags_set(self):
        state = _make_state(should_navigate=True, target_slide=2)
        assert should_navigate(state) == "navigate"

    def test_returns_respond_when_no_navigation(self):
        state = _make_state(should_navigate=False, target_slide=None)
        assert should_navigate(state) == "respond"

    def test_returns_respond_when_navigate_true_but_no_target(self):
        state = _make_state(should_navigate=True, target_slide=None)
        assert should_navigate(state) == "respond"

    def test_returns_respond_when_target_set_but_navigate_false(self):
        # should not navigate if the flag wasn't set
        state = _make_state(should_navigate=False, target_slide=2)
        assert should_navigate(state) == "respond"
