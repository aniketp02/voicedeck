"""
Slide narration — direct LLM call for auto-narration mode.

Used by the auto_narrate_loop in websocket.py. Bypasses the full LangGraph
agent (understand → navigate → respond) because auto-narration doesn't need
intent parsing or slide navigation — just a high-quality narration of the
current slide, optionally bridging from the previous one.
"""
import logging

from app.agent.prompts import NARRATE_SYSTEM
from app.services.llm import chat_completion
from app.slides.content import Slide
from app.slides.presentations import Presentation

logger = logging.getLogger(__name__)


async def narrate_slide(
    slide: Slide,
    presentation: Presentation,
    prev_slide: Slide | None = None,
) -> str:
    """
    Generate spoken narration text for a slide.

    Args:
        slide:        The slide to narrate.
        presentation: The full presentation (for title context).
        prev_slide:   Previous slide, if any, for a natural transition bridge.

    Returns:
        Narration text string (2–5 sentences, voice-ready).
    """
    transition_block = ""
    if prev_slide is not None:
        transition_block = (
            f'[TRANSITION: You just finished "{prev_slide.title}". '
            f"Open with a natural bridge connecting that topic to this one — "
            f"make it feel like one flowing talk, not a hard cut.]\n\n"
        )

    system = NARRATE_SYSTEM.format(
        presentation_title=presentation.meta.title,
        slide_index=slide.index,
        slide_title=slide.title,
        slide_bullets="\n".join(f"- {b}" for b in slide.bullets),
        speaker_notes=slide.speaker_notes,
        transition_block=transition_block,
    )

    text = await chat_completion(system, "Narrate this slide.")
    logger.info(
        "narrate_slide: slide=%d prev=%s generated %d chars",
        slide.index,
        prev_slide.index if prev_slide else None,
        len(text),
    )
    return text
