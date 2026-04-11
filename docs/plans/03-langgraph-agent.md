# Plan 03 — LangGraph Agent: Understand + Navigate + Respond

## Goal
Implement the three LangGraph nodes so that:
1. `understand_node` — LLM parses user intent, decides if a slide change is needed
2. `navigate_node` — updates current slide index, emits `slide_change` event
3. `respond_node` — LLM generates a spoken response about the current slide

Then wire the graph into the WebSocket handler so that a final transcript
triggers the full agent pipeline and sends `agent_text` back to the client.

**Success criterion:** Say "tell me about patient recruitment" → slide changes
to slide 1, agent speaks about it. Say "what's the main problem?" → stays on
current slide and answers the question.

## Prerequisite
Plan 02 must be complete (final transcripts arriving in WebSocket handler).

## Files to Modify

### `app/agent/nodes.py` — IMPLEMENT ALL THREE NODES

```python
import json
import logging
from app.agent.state import AgentState
from app.agent.prompts import UNDERSTAND_SYSTEM, RESPOND_SYSTEM
from app.services.llm import chat_completion, chat_completion_json
from app.slides.content import SLIDES, get_slide

logger = logging.getLogger(__name__)


async def understand_node(state: AgentState) -> dict:
    """
    Parse user intent. Decides whether to navigate and where.
    Returns: should_navigate (bool), target_slide (int | None)
    """
    user_msg = (
        f"Current slide index: {state['current_slide']}\n"
        f"User said: {state['transcript']}"
    )
    result = await chat_completion_json(UNDERSTAND_SYSTEM, user_msg)

    should_nav = bool(result.get("should_navigate", False))
    target = result.get("target_slide")

    # Validate target is in range
    if should_nav and target is not None:
        if not (0 <= int(target) < len(SLIDES)):
            logger.warning("LLM returned out-of-range slide %s, ignoring", target)
            should_nav = False
            target = None

    logger.info(
        "understand_node: navigate=%s target=%s intent=%r",
        should_nav, target, result.get("intent_summary", "")
    )
    return {
        "should_navigate": should_nav,
        "target_slide": int(target) if target is not None else None,
    }


async def navigate_node(state: AgentState) -> dict:
    """Update current_slide to target_slide."""
    target = state["target_slide"]
    logger.info("navigate_node: %d → %d", state["current_slide"], target)
    return {
        "current_slide": target,
        "slide_changed": True,
    }


async def respond_node(state: AgentState) -> dict:
    """Generate spoken response about the current slide."""
    slide = get_slide(state["current_slide"])

    system = RESPOND_SYSTEM.format(
        slide_index=slide.index,
        slide_title=slide.title,
        slide_bullets="\n".join(f"- {b}" for b in slide.bullets),
        speaker_notes=slide.speaker_notes,
    )

    # Build user message: transcript + context about whether we just navigated
    nav_context = ""
    if state.get("slide_changed"):
        nav_context = f"[We just navigated to this slide in response to the user's question.] "

    user_msg = f"{nav_context}User: {state['transcript']}"

    response_text = await chat_completion(system, user_msg)
    logger.info("respond_node: generated %d chars", len(response_text))

    return {
        "response_text": response_text,
        "slide_changed": False,  # reset for next turn
    }


def should_navigate(state: AgentState) -> str:
    """Conditional edge function."""
    if state.get("should_navigate") and state.get("target_slide") is not None:
        return "navigate"
    return "respond"
```

### `app/api/websocket.py` — Wire agent graph into transcript handler

Add a `run_agent` helper function and call it from `on_transcript`:

```python
from app.agent.graph import agent_graph
from app.slides.content import get_slide

async def run_agent(
    websocket: WebSocket,
    state: AgentState,
    transcript: str,
    interrupt_event: asyncio.Event,
) -> None:
    """Run the agent graph on a transcript, update state, stream results."""
    state["transcript"] = transcript
    state["slide_changed"] = False

    # Run LangGraph
    result = await agent_graph.ainvoke(state)

    # Update shared state
    state.update(result)

    # If slide changed, notify client
    if result.get("slide_changed") or result.get("current_slide") != state.get("current_slide"):
        slide = get_slide(state["current_slide"])
        await _send(websocket, {
            "type": "slide_change",
            "index": state["current_slide"],
            "slide": {"title": slide.title, "bullets": slide.bullets},
        })

    # Send agent text
    response_text = result.get("response_text", "")
    if response_text:
        await _send(websocket, {"type": "agent_text", "text": response_text})

    # TTS will be streamed in Plan 04 — for now just log
    logger.info("Agent response: %r", response_text[:100])
```

In `on_transcript`, replace the `# TODO Plan 03` comment with:
```python
if result.is_final and result.text.strip():
    await run_agent(websocket, state, result.text, interrupt_event)
```

## State Flow Diagram
```
User speaks → Deepgram → on_transcript(final) → run_agent()
                                                      ↓
                                              agent_graph.ainvoke(state)
                                                      ↓
                                              understand_node (LLM)
                                                      ↓
                                    [should_navigate?]
                                      ↓ yes        ↓ no
                                 navigate_node   respond_node
                                      ↓              ↓
                                 respond_node ←──────┘
                                      ↓
                              state.response_text set
                                      ↓
                              send slide_change (if nav)
                              send agent_text
                              (TTS in Plan 04)
```

## Verification
1. Complete Plans 01 and 02 first
2. Start backend
3. Connect WebSocket, speak: "Tell me about patient recruitment"
4. Expect server log: `navigate_node: 0 → 1`
5. Expect WebSocket message: `{"type":"slide_change","index":1,...}`
6. Expect WebSocket message: `{"type":"agent_text","text":"..."}`
7. Speak: "What's the main challenge?" → should NOT navigate, just respond

## Common Issues
- `json.JSONDecodeError` from understand_node → LLM not returning JSON.
  Check UNDERSTAND_SYSTEM prompt has `response_format: json_object` set in llm.py ✅
- Slide keeps navigating unnecessarily → tighten UNDERSTAND_SYSTEM prompt keywords
- `KeyError` in state → ensure all AgentState fields are initialized in `handle_session`
