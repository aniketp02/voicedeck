"""
Unit tests for narrate.py — slide narration LLM call.
All OpenAI calls are mocked — no API keys required.
"""
import pytest
from unittest.mock import AsyncMock, patch

from app.agent.narrate import narrate_slide
from app.slides.presentations import get_presentation


@pytest.fixture
def presentation():
    return get_presentation("clinical-trials")


class TestNarrateSlide:
    @pytest.mark.asyncio
    async def test_returns_narration_text(self, presentation):
        slide = presentation.slides[0]
        expected = "The clinical trial system is fundamentally broken."

        with patch("app.agent.narrate.chat_completion", AsyncMock(return_value=expected)):
            result = await narrate_slide(slide, presentation)

        assert result == expected

    @pytest.mark.asyncio
    async def test_no_transition_block_on_first_slide(self, presentation):
        slide = presentation.slides[0]
        captured = []

        async def capture(system, user_msg):
            captured.append(system)
            return "narration"

        with patch("app.agent.narrate.chat_completion", capture):
            await narrate_slide(slide, presentation, prev_slide=None)

        assert "[TRANSITION:" not in captured[0]

    @pytest.mark.asyncio
    async def test_transition_block_injected_with_prev_slide(self, presentation):
        prev_slide = presentation.slides[0]
        slide = presentation.slides[1]
        captured = []

        async def capture(system, user_msg):
            captured.append(system)
            return "narration"

        with patch("app.agent.narrate.chat_completion", capture):
            await narrate_slide(slide, presentation, prev_slide=prev_slide)

        assert "[TRANSITION:" in captured[0]
        assert prev_slide.title in captured[0]

    @pytest.mark.asyncio
    async def test_system_prompt_contains_slide_title(self, presentation):
        slide = presentation.slides[2]
        captured = []

        async def capture(system, user_msg):
            captured.append(system)
            return "narration"

        with patch("app.agent.narrate.chat_completion", capture):
            await narrate_slide(slide, presentation)

        assert slide.title in captured[0]

    @pytest.mark.asyncio
    async def test_user_message_is_narrate_this_slide(self, presentation):
        slide = presentation.slides[0]
        captured_user = []

        async def capture(system, user_msg):
            captured_user.append(user_msg)
            return "narration"

        with patch("app.agent.narrate.chat_completion", capture):
            await narrate_slide(slide, presentation)

        assert captured_user[0] == "Narrate this slide."
