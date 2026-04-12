"""
LangGraph nodes for the voice agent.

Node execution order:
  understand_node → (conditional) → navigate_node → respond_node
                  ↘ respond_node (if no navigation needed)
"""
import logging

from app.agent.slide_target import normalize_slide_target
from app.agent.state import AgentState
from app.agent.prompts import understand_system, RESPOND_SYSTEM
from app.services.llm import chat_completion, chat_completion_json
from app.slides.presentations import get_presentation

logger = logging.getLogger(__name__)


async def understand_node(state: AgentState) -> dict:
    """
    Use OpenAI to parse user intent from the transcript.
    Determines whether to navigate to a different slide and which one.

    Returns: should_navigate (bool), target_slide (int | None)
    """
    presentation = get_presentation(state["presentation_id"])
    slides = presentation.slides

    user_msg = (
        f"Current slide index: {state['current_slide']}\n"
        f"User said: {state['transcript']}"
    )

    system = understand_system(slides)
    result = await chat_completion_json(system, user_msg)

    should_nav = bool(result.get("should_navigate", False))
    target = result.get("target_slide")
    intent = result.get("intent_summary", "")

    # Validate target index is in bounds for this presentation
    if should_nav and target is not None:
        try:
            target = int(target)
            target = normalize_slide_target(
                state["transcript"], target, len(slides)
            )
            if not (0 <= target < len(slides)):
                logger.warning(
                    "LLM returned out-of-range slide index %d (presentation has %d slides) — ignoring",
                    target,
                    len(slides),
                )
                should_nav = False
                target = None
            elif target == state["current_slide"]:
                # No need to navigate to the same slide
                should_nav = False
                target = None
        except (TypeError, ValueError):
            logger.warning("LLM returned non-integer target_slide %r — ignoring", target)
            should_nav = False
            target = None
    elif should_nav and target is None:
        # LLM said navigate but gave no target — ignore
        should_nav = False

    logger.info(
        "understand_node: navigate=%s target=%s intent=%r",
        should_nav, target, intent,
    )

    return {
        "should_navigate": should_nav,
        "target_slide": target,
    }


async def navigate_node(state: AgentState) -> dict:
    """
    Update current_slide to target_slide and set slide_changed flag.
    Only reached when should_navigate=True.
    """
    prev = state["current_slide"]
    target = state["target_slide"]
    logger.info("navigate_node: slide %d → %d", prev, target)

    return {
        "current_slide": target,
        "slide_changed": True,
    }


async def respond_node(state: AgentState) -> dict:
    """
    Generate spoken response text for the current slide using OpenAI.
    Uses the slide's speaker_notes as knowledge base (not read verbatim).
    """
    presentation = get_presentation(state["presentation_id"])
    slide = presentation.slides[state["current_slide"]]

    context_lines = []
    if state.get("slide_changed"):
        context_lines.append(
            "[CONTEXT: You just moved to this slide in response to the user's question. "
            "Introduce the new topic naturally — don't announce the navigation.]"
        )
    if state.get("interrupted"):
        context_lines.append(
            "[CONTEXT: The user spoke while you were mid-response. "
            "Address their new question directly and naturally. "
            "A brief 'Right —' pivot is acceptable; referencing the interruption mechanically is not.]"
        )
    context_block = ("\n".join(context_lines) + "\n\n") if context_lines else ""

    # Inject next slide hint for proactive turn-ending (Option B in RESPOND_SYSTEM).
    # Only provided when a next slide exists; empty string disables the Option B rule.
    next_idx = slide.index + 1
    next_slide_hint = (
        f'"{presentation.slides[next_idx].title}" '
        if next_idx < len(presentation.slides)
        else ""
    )

    system = RESPOND_SYSTEM.format(
        presentation_title=presentation.meta.title,
        slide_index=slide.index,
        slide_title=slide.title,
        slide_bullets="\n".join(f"- {b}" for b in slide.bullets),
        speaker_notes=slide.speaker_notes,
        context_block=context_block,
        next_slide_hint=next_slide_hint,
    )

    user_msg = f"User: {state['transcript']}"

    response_text = await chat_completion(system, user_msg)
    logger.info(
        "respond_node: slide=%d interrupted=%s generated %d chars",
        slide.index,
        state.get("interrupted", False),
        len(response_text),
    )

    return {
        "response_text": response_text,
        "slide_changed": False,
        "interrupted": False,
    }


def should_navigate(state: AgentState) -> str:
    """Conditional edge: route to navigate_node or respond_node."""
    if state.get("should_navigate") and state.get("target_slide") is not None:
        return "navigate"
    return "respond"
