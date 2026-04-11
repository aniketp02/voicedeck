"""
LangGraph nodes for the voice agent.

Node execution order:
  understand_node → (conditional) → navigate_node → respond_node
                  ↘ respond_node (if no navigation needed)

TODO (Plan 03): Implement all nodes.
See docs/plans/03-langgraph-agent.md for full implementation details.
"""
import logging
from app.agent.state import AgentState
from app.agent.prompts import UNDERSTAND_SYSTEM, RESPOND_SYSTEM
from app.services.llm import chat_completion, chat_completion_json
from app.slides.content import SLIDES, get_slide

logger = logging.getLogger(__name__)


async def understand_node(state: AgentState) -> dict:
    """
    Uses OpenAI to parse user intent.
    Sets: should_navigate, target_slide.

    TODO: Call chat_completion_json with UNDERSTAND_SYSTEM prompt.
    Pass current_slide and transcript in user message.
    Return dict updating should_navigate and target_slide fields.
    """
    logger.info("understand_node: transcript=%r", state["transcript"])
    # TODO: implement
    return {"should_navigate": False, "target_slide": None}


async def navigate_node(state: AgentState) -> dict:
    """
    Updates current_slide to target_slide and sets slide_changed=True.

    TODO: Simply set current_slide = target_slide, slide_changed = True.
    Log the navigation. This node is only reached when should_navigate=True.
    """
    target = state["target_slide"]
    logger.info("navigate_node: %d → %d", state["current_slide"], target)
    # TODO: implement
    return {"current_slide": target, "slide_changed": True}


async def respond_node(state: AgentState) -> dict:
    """
    Generates spoken response text using the current slide as context.

    TODO: Build RESPOND_SYSTEM prompt with current slide data.
    Call chat_completion with the user transcript.
    Return dict with response_text set.
    """
    slide = get_slide(state["current_slide"])
    logger.info("respond_node: slide=%d %r", slide.index, slide.title)
    # TODO: implement
    return {"response_text": ""}


def should_navigate(state: AgentState) -> str:
    """Conditional edge: route to navigate_node or respond_node."""
    if state.get("should_navigate") and state.get("target_slide") is not None:
        return "navigate"
    return "respond"
