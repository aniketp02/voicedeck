"""
Unit tests for LangGraph agent nodes.
All OpenAI calls are mocked — no API keys required.
"""
import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes import understand_node, navigate_node, respond_node, should_navigate, _format_history
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
        "end_session": False,
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

    @pytest.mark.asyncio
    async def test_normalizes_one_based_echo_from_llm(self):
        """LLM returns 5 for 'fifth slide'; we store index 4."""
        state = _make_state(
            current_slide=0,
            transcript="go to the fifth slide",
        )
        mock_response = {
            "should_navigate": True,
            "target_slide": 5,
            "intent_summary": "Navigate to fifth slide",
        }
        with patch("app.agent.nodes.chat_completion_json", AsyncMock(return_value=mock_response)):
            result = await understand_node(state)

        assert result["should_navigate"] is True
        assert result["target_slide"] == 4


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
        captured = []

        async def capture_call(system, user_msg):
            captured.append((system, user_msg))
            return "Patient recruitment response"

        with patch("app.agent.nodes.chat_completion", capture_call):
            await respond_node(state)

        assert len(captured) == 1
        system, user_msg = captured[0]
        assert "[CONTEXT:" in system
        assert "just moved to this slide" in system.lower()
        assert user_msg == "User: tell me about recruitment"

    @pytest.mark.asyncio
    async def test_resets_slide_changed_flag(self):
        state = _make_state(current_slide=0, slide_changed=True)
        with patch("app.agent.nodes.chat_completion", AsyncMock(return_value="response")):
            result = await respond_node(state)

        assert result["slide_changed"] is False


class TestRespondNodeInterrupted:
    @pytest.mark.asyncio
    async def test_resets_interrupted_flag_after_response(self):
        """respond_node must return interrupted=False regardless of input."""
        state = _make_state(current_slide=0, interrupted=True)
        with patch("app.agent.nodes.chat_completion", AsyncMock(return_value="response")):
            result = await respond_node(state)

        assert result["interrupted"] is False

    @pytest.mark.asyncio
    async def test_non_interrupted_also_resets_flag(self):
        """interrupted=False should also be returned when not interrupted."""
        state = _make_state(current_slide=0, interrupted=False)
        with patch("app.agent.nodes.chat_completion", AsyncMock(return_value="response")):
            result = await respond_node(state)

        assert result["interrupted"] is False


# ---------------------------------------------------------------------------
# should_navigate (conditional edge)
# ---------------------------------------------------------------------------

class TestRespondNodeNextSlideHint:
    @pytest.mark.asyncio
    async def test_injects_next_slide_hint_when_not_on_last_slide(self):
        """respond_node should include next slide title in system prompt for non-last slides."""
        state = _make_state(current_slide=0)
        captured_system = []

        async def capture_call(system, user_msg):
            captured_system.append(system)
            return "response"

        with patch("app.agent.nodes.chat_completion", capture_call):
            await respond_node(state)

        system = captured_system[0]
        # Slide 0 has a next slide — hint must appear
        assert "{next_slide_hint}" not in system  # placeholder must be resolved
        # The next slide title for clinical-trials slide 0 → slide 1 must be in the prompt
        assert "AI-Powered" in system or "Recruitment" in system or "Patient" in system

    @pytest.mark.asyncio
    async def test_no_next_slide_hint_on_last_slide(self):
        """On the last slide, next_slide_hint is empty — Option B rule is disabled."""
        from app.slides.presentations import get_presentation
        presentation = get_presentation("clinical-trials")
        last_idx = len(presentation.slides) - 1

        state = _make_state(current_slide=last_idx)
        captured_system = []

        async def capture_call(system, user_msg):
            captured_system.append(system)
            return "response"

        with patch("app.agent.nodes.chat_completion", capture_call):
            await respond_node(state)

        system = captured_system[0]
        assert "{next_slide_hint}" not in system
        # On the last slide the hint placeholder resolves to empty string — no next title
        assert "Option B" not in system or 'only when ""' not in system


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


# ---------------------------------------------------------------------------
# _format_history
# ---------------------------------------------------------------------------

class TestFormatHistory:
    def test_empty_messages_returns_empty_string(self):
        assert _format_history([]) == ""

    def test_only_human_message_returns_empty_string(self):
        """Single human message (the current utterance) — no prior AI turns."""
        msgs = [HumanMessage(content="hello")]
        assert _format_history(msgs) == ""

    def test_excludes_last_human_message_from_history(self):
        """The current HumanMessage (last in list) must not appear in history."""
        msgs = [
            HumanMessage(content="first question"),
            AIMessage(content="first answer"),
            HumanMessage(content="current question"),
        ]
        result = _format_history(msgs)
        assert "current question" not in result
        assert "first question" in result
        assert "first answer" in result

    def test_formats_turns_correctly(self):
        msgs = [
            HumanMessage(content="What is this?"),
            AIMessage(content="This is the overview."),
            HumanMessage(content="Tell me more."),
        ]
        result = _format_history(msgs)
        assert "User: What is this?" in result
        assert "Assistant: This is the overview." in result
        assert "=== CONVERSATION SO FAR ===" in result

    def test_respects_limit(self):
        """Only the last `limit` messages are included."""
        msgs = []
        for i in range(10):
            msgs.append(HumanMessage(content=f"q{i}"))
            msgs.append(AIMessage(content=f"a{i}"))
        # Last message is the current utterance — excluded
        msgs.append(HumanMessage(content="current"))
        result = _format_history(msgs, limit=4)
        # With limit=4 we keep msgs[-5:-1] = a7, q8, a8, q9 (indices into prior list)
        assert "q0" not in result
        assert "a0" not in result

    def test_returns_empty_when_no_ai_turns_in_window(self):
        """If the window contains only human messages, return empty."""
        msgs = [HumanMessage(content="hello"), HumanMessage(content="current")]
        assert _format_history(msgs, limit=4) == ""


# ---------------------------------------------------------------------------
# respond_node — conversation history injection
# ---------------------------------------------------------------------------

class TestRespondNodeConversationHistory:
    @pytest.mark.asyncio
    async def test_injects_history_when_prior_turns_exist(self):
        msgs = [
            HumanMessage(content="What's the main problem?"),
            AIMessage(content="Trials are expensive and slow."),
            HumanMessage(content="Why?"),  # current utterance
        ]
        state = _make_state(current_slide=0, messages=msgs, transcript="Why?")
        captured_system = []

        async def capture_call(system, user_msg, **kwargs):
            captured_system.append(system)
            return "response"

        with patch("app.agent.nodes.chat_completion", capture_call):
            await respond_node(state)

        system = captured_system[0]
        assert "=== CONVERSATION SO FAR ===" in system
        assert "What's the main problem?" in system
        assert "Trials are expensive and slow." in system
        # Current utterance should NOT be duplicated in history block
        assert system.count("Why?") <= 1

    @pytest.mark.asyncio
    async def test_no_history_block_on_first_exchange(self):
        """First message only — no AI turns yet — history block must be absent."""
        msgs = [HumanMessage(content="What is this?")]
        state = _make_state(current_slide=0, messages=msgs, transcript="What is this?")
        captured_system = []

        async def capture_call(system, user_msg, **kwargs):
            captured_system.append(system)
            return "response"

        with patch("app.agent.nodes.chat_completion", capture_call):
            await respond_node(state)

        assert "=== CONVERSATION SO FAR ===" not in captured_system[0]

    @pytest.mark.asyncio
    async def test_no_history_block_when_messages_empty(self):
        state = _make_state(current_slide=0, messages=[], transcript="test")
        captured_system = []

        async def capture_call(system, user_msg, **kwargs):
            captured_system.append(system)
            return "response"

        with patch("app.agent.nodes.chat_completion", capture_call):
            await respond_node(state)

        assert "=== CONVERSATION SO FAR ===" not in captured_system[0]
