# Agent Plan 03 — LangGraph Agent Nodes + WebSocket Wiring

## Agent Instructions
You are an autonomous agent. Read this plan completely before taking any action.
Do not ask the user questions. All decisions are made for you.
Implement every task in order. Run verification commands after each task.

---

## Goal
Implement the three LangGraph nodes (`understand_node`, `navigate_node`,
`respond_node`) and wire the compiled graph into the WebSocket session handler.
Replace the `_agent_stub` in `websocket.py` with a real `run_agent` function
that invokes the graph and sends `slide_change` + `agent_text` messages.

TTS synthesis is NOT part of this plan — `run_agent` will log the response
text and send `tts_done`. ElevenLabs is wired in Plan 04.

## Contract — What Plan 02 Delivered
- `app/services/stt.py`: full Deepgram streaming implementation
- `app/api/websocket.py`: STT background task + `_agent_stub` placeholder
- Final transcripts arrive → `_agent_stub` echoes them back
- `OPENAI_API_KEY` is set in `.env` (real key)

## Background: How the LangGraph Graph Works

```
agent_graph.ainvoke(state) is called with the full AgentState dict.

Graph topology:
  START → understand_node → [conditional: should_navigate?]
                                 ↓ yes          ↓ no
                            navigate_node    respond_node
                                 ↓
                            respond_node
                                 ↓
                               END

LangGraph returns a NEW state dict (it does not mutate the input).
The caller (run_agent) merges the returned fields back into the session state.

The `messages` field uses the add_messages reducer:
  - Accumulates conversation history across turns
  - We prepend the user's transcript as a HumanMessage before each ainvoke
  - The graph can read message history for context
```

## Background: Prompts (read-only — do not modify prompts.py)

`UNDERSTAND_SYSTEM` (pre-formatted at import time):
- Includes the full slide index with keywords
- Expects JSON response: `{should_navigate, target_slide, intent_summary}`
- Is called with: `f"Current slide index: {N}\nUser said: {transcript}"`

`RESPOND_SYSTEM` (formatted at runtime in respond_node):
- Template vars: `{slide_index}`, `{slide_title}`, `{slide_bullets}`, `{speaker_notes}`
- Called with: `f"[navigation context if any] User: {transcript}"`

---

## Task 1: Implement `app/agent/nodes.py`

Replace the entire file with:

```python
import logging
from langchain_core.messages import HumanMessage, AIMessage

from app.agent.state import AgentState
from app.agent.prompts import UNDERSTAND_SYSTEM, RESPOND_SYSTEM
from app.services.llm import chat_completion, chat_completion_json
from app.slides.content import SLIDES, get_slide

logger = logging.getLogger(__name__)


async def understand_node(state: AgentState) -> dict:
    """
    Use OpenAI to parse user intent from the transcript.
    Determines whether to navigate to a different slide and which one.

    Returns: should_navigate (bool), target_slide (int | None)
    """
    user_msg = (
        f"Current slide index: {state['current_slide']}\n"
        f"User said: {state['transcript']}"
    )

    result = await chat_completion_json(UNDERSTAND_SYSTEM, user_msg)

    should_nav = bool(result.get("should_navigate", False))
    target = result.get("target_slide")
    intent = result.get("intent_summary", "")

    # Validate target index is in bounds
    if should_nav and target is not None:
        try:
            target = int(target)
            if not (0 <= target < len(SLIDES)):
                logger.warning(
                    "LLM returned out-of-range slide index %d — ignoring navigation",
                    target,
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
    slide = get_slide(state["current_slide"])

    system = RESPOND_SYSTEM.format(
        slide_index=slide.index,
        slide_title=slide.title,
        slide_bullets="\n".join(f"- {b}" for b in slide.bullets),
        speaker_notes=slide.speaker_notes,
    )

    # Add navigation context if we just moved to this slide
    nav_context = ""
    if state.get("slide_changed"):
        nav_context = (
            f"[We just navigated to slide {slide.index} '{slide.title}' "
            f"in response to the user's question.] "
        )

    user_msg = f"{nav_context}User: {state['transcript']}"

    response_text = await chat_completion(system, user_msg)
    logger.info(
        "respond_node: slide=%d generated %d chars",
        slide.index,
        len(response_text),
    )

    return {
        "response_text": response_text,
        "slide_changed": False,  # reset for next turn
    }


def should_navigate(state: AgentState) -> str:
    """Conditional edge: route to navigate_node or respond_node."""
    if state.get("should_navigate") and state.get("target_slide") is not None:
        return "navigate"
    return "respond"
```

### Verify Task 1
```bash
python -c "from app.agent.nodes import understand_node, navigate_node, respond_node, should_navigate; print('nodes OK')"
```

---

## Task 2: Verify `app/agent/graph.py` (no changes needed — just confirm)

Read `app/agent/graph.py`. It should look like:

```python
from langgraph.graph import StateGraph, START, END
from app.agent.state import AgentState
from app.agent.nodes import understand_node, navigate_node, respond_node, should_navigate

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("understand", understand_node)
    builder.add_node("navigate", navigate_node)
    builder.add_node("respond", respond_node)
    builder.add_edge(START, "understand")
    builder.add_conditional_edges(
        "understand",
        should_navigate,
        {"navigate": "navigate", "respond": "respond"},
    )
    builder.add_edge("navigate", "respond")
    builder.add_edge("respond", END)
    return builder.compile()

agent_graph = build_graph()
```

Verify it compiles:
```bash
python -c "from app.agent.graph import agent_graph; print('graph compiled OK')"
```

If the graph fails to compile due to LangGraph version differences, check:
- LangGraph >= 0.2.50 is required
- `StateGraph` takes the TypedDict class, not an instance
- `add_conditional_edges` signature: `(source_node, condition_fn, mapping_dict)`

---

## Task 3: Implement `run_agent` in `app/api/websocket.py`

This is the most critical task. Add a `run_agent` function and replace the
`_agent_stub` call with it.

### 3a. Add imports to the top of `websocket.py`

Add these imports (after the existing imports):
```python
from langchain_core.messages import HumanMessage
from app.agent.graph import agent_graph
from app.slides.content import get_slide
```

### 3b. Add the `run_agent` function (add BEFORE `handle_session`)

```python
async def run_agent(
    websocket: WebSocket,
    state: AgentState,
    transcript: str,
    interrupt_event: asyncio.Event,
) -> None:
    """
    Run the LangGraph agent pipeline for one user utterance.

    Steps:
    1. Add user transcript to message history
    2. Set transcript on state
    3. ainvoke the agent graph
    4. Merge returned fields back into session state
    5. If slide changed: send slide_change message
    6. Send agent_text message
    7. TTS streaming: Plan 04 (for now: just send tts_done)

    Always sends tts_done in finally block — client must receive this
    to exit the "waiting" state regardless of errors or cancellation.
    """
    tts_done_sent = False
    try:
        # 1. Append user transcript to message history
        state["messages"] = state["messages"] + [HumanMessage(content=transcript)]
        state["transcript"] = transcript
        state["slide_changed"] = False
        state["should_navigate"] = False
        state["target_slide"] = None

        # 2. Run the graph (async)
        result = await agent_graph.ainvoke(state)

        # 3. Merge result back into session state
        # Only update fields that LangGraph nodes actually set
        for key in ("current_slide", "target_slide", "should_navigate",
                    "response_text", "slide_changed", "messages"):
            if key in result:
                state[key] = result[key]

        # 4. Send slide_change if navigation occurred
        if result.get("slide_changed"):
            slide = get_slide(state["current_slide"])
            await _send(websocket, {
                "type": "slide_change",
                "index": state["current_slide"],
                "slide": {"title": slide.title, "bullets": slide.bullets},
            })
            logger.info("Sent slide_change to client: index=%d", state["current_slide"])

        # 5. Send agent text
        response_text = result.get("response_text", "")
        if response_text:
            # Append AI response to message history
            state["messages"] = state["messages"] + [AIMessage(content=response_text)]
            await _send(websocket, {"type": "agent_text", "text": response_text})
            logger.info("Sent agent_text: %d chars", len(response_text))

        # 6. TTS placeholder (Plan 04 replaces this with ElevenLabs streaming)
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
        # Guarantee tts_done is always sent so client doesn't hang
        if not tts_done_sent:
            await _send(websocket, {"type": "tts_done"})
```

### 3c. Replace the `_agent_stub` call in `on_transcript`

Find this line in `on_transcript`:
```python
agent_task = asyncio.create_task(
    _agent_stub(websocket, state, result.text, interrupt_event)
)
```

Replace with:
```python
agent_task = asyncio.create_task(
    run_agent(websocket, state, result.text, interrupt_event)
)
```

### 3d. Remove the `_agent_stub` function

Delete the entire `_agent_stub` function from `websocket.py` — it's no longer needed.

### Verify Task 3
```bash
python -c "from app.api.websocket import handle_session, run_agent; print('websocket OK')"
```

---

## Task 4: Integration test

### Step 1: Start server
```bash
uvicorn app.main:app --reload --port 8000
```

### Step 2: Connect and speak (or simulate)

If frontend is available, use it. Otherwise use wscat + manually trigger
a transcript by sending an audio chunk that Deepgram can transcribe.

Watch logs for the full pipeline:
```
INFO  app.api.websocket     Final transcript: 'tell me about patient recruitment'
INFO  app.agent.nodes       understand_node: navigate=True target=1 intent='...'
INFO  app.agent.nodes       navigate_node: slide 0 → 1
INFO  app.agent.nodes       respond_node: slide=1 generated 187 chars
INFO  app.api.websocket     Sent slide_change to client: index=1
INFO  app.api.websocket     Sent agent_text: 187 chars
```

WebSocket client must receive (in order):
```json
{"type": "transcript", "text": "...", "is_final": true}
{"type": "slide_change", "index": 1, "slide": {"title": "AI-Powered Patient Recruitment", "bullets": [...]}}
{"type": "agent_text", "text": "Patient recruitment is the single biggest bottleneck..."}
{"type": "tts_done"}
```

### Step 3: Test no-navigation case
Say something like "what's the main problem?" while on slide 0.
Expected: NO `slide_change` message, just `agent_text` + `tts_done`.

### Step 4: Verify message history accumulates
After 2 turns, add a debug log temporarily:
```python
logger.debug("Message history: %d messages", len(state["messages"]))
```
Should increment by 2 per turn (1 HumanMessage + 1 AIMessage).

---

## Acceptance Criteria

- [ ] `app/agent/nodes.py` — all three nodes fully implemented, no stubs
- [ ] `app/agent/graph.py` — compiles without errors
- [ ] `run_agent` function exists in `websocket.py`
- [ ] `_agent_stub` function is removed from `websocket.py`
- [ ] Speaking "tell me about patient recruitment" → logs show `navigate=True target=1`
- [ ] Client receives `slide_change` with correct slide data
- [ ] Client receives `agent_text` with a real OpenAI-generated response (not a stub)
- [ ] Client always receives `tts_done` (even on error or cancellation)
- [ ] Speaking a question about the current slide → NO `slide_change` sent
- [ ] Message history grows by 2 per turn in `state["messages"]`

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `json.JSONDecodeError` in understand_node | OpenAI not returning JSON | Check `response_format: json_object` in `llm.py` `chat_completion_json` |
| Always navigating, even for current-slide questions | UNDERSTAND_SYSTEM keywords too loose | Increase specificity in slide keywords in `content.py` |
| `KeyError: 'current_slide'` | State not initialized | Check all AgentState fields initialized in `handle_session` |
| `AttributeError` on `result["messages"]` | LangGraph not returning messages field | Use `.get()` pattern in state merge loop |
| OpenAI rate limit | Too many requests | gpt-4o-mini has high rate limits — unlikely; check API key tier |

## Important: LangGraph State Merge

`agent_graph.ainvoke(state)` returns a dict with the fields that changed.
The `messages` field is special — LangGraph returns the FULL accumulated list
(after applying the `add_messages` reducer), not just the new messages.
So `state["messages"] = result["messages"]` is correct (replaces, not appends again).
