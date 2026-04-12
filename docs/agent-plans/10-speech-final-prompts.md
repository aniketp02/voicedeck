# Backend Agent Plan 10 — Speech Final Fix + Natural Prompts + Manual Navigation

## Agent Instructions
Read this plan fully before acting. Do not ask questions. All decisions are made.
Plans 01–08 must be complete before starting.

**Parallel execution:** This plan runs in parallel with frontend Plan 11.
Both plans are self-contained — no shared files.

---

## Background: Three Bugs Being Fixed

### Bug 1 — Premature agent trigger (the "thinking flash")
**Root cause:** `on_transcript` in `websocket.py` calls `run_agent` on every `is_final=True`
transcript from Deepgram. Deepgram fires `is_final=True` multiple times per utterance — once
per finalized speech segment (mid-sentence boundary). A natural breath mid-sentence fires
`is_final=True` for the first half, which incorrectly triggers the agent.

**Correct trigger:** `speech_final=True`. Deepgram only fires this once per utterance, after
`utterance_end_ms` ms of silence at the end. This is the true utterance completion signal.

**Defense-in-depth:** Also increase `utterance_end_ms` from 1000ms to 1200ms to give users
more headroom for natural pauses before Deepgram decides the utterance is done.

### Bug 2 — `interrupted` state never set
**Root cause:** `AgentState.interrupted` exists in `state.py` but is never set to `True`
anywhere in `websocket.py`. The agent and `respond_node` can't distinguish an interrupted
turn from a fresh one, so it always responds as if no interruption occurred.

**Fix:** Set `state["interrupted"] = True` in both interrupt paths:
1. When `on_transcript` cancels a running `agent_task` because a new utterance arrived
2. When the client sends an explicit `{"type": "interrupt"}` message
Reset it in `respond_node` by returning `"interrupted": False`.

### Bug 3 — Robot acknowledgments
**Root cause:** `RESPOND_SYSTEM` has no rules about:
- How to handle interruption context (the agent doesn't even know it was interrupted)
- How to transition into a new slide naturally (nav_context is a system-tag in user_msg)
- What NOT to say (the LLM defaults to helper-assistant patterns: "Of course!", "Certainly!")

**Fix:** Rewrite `RESPOND_SYSTEM` with explicit anti-patterns, natural opening rules,
occasional-acknowledgment guidance, and a `{context_block}` system prompt section
(instead of inline nav prefix in user_msg).

---

## Files Changed

- `app/services/stt.py` — `TranscriptResult.speech_final`, `utterance_end_ms=1200`
- `app/api/websocket.py` — gate on `speech_final`, set `interrupted`, forward `speech_final`,
  add `navigate` message handler
- `app/agent/prompts.py` — rewrite `RESPOND_SYSTEM`, add `context_block` parameter
- `app/agent/nodes.py` — build `context_block`, pass `presentation_title`, reset `interrupted`
- `app/agent/state.py` — no structural change needed (field already exists)
- `tests/test_agent_nodes.py` — add `interrupted` to relevant test states
- `tests/test_websocket_protocol.py` — verify `speech_final` gating

---

## Task 1: Update `app/services/stt.py`

Two targeted changes:

**Change A:** Add `speech_final: bool = False` to `TranscriptResult`.

**Change B:** Read `message.speech_final` in `_on_message` and set it on the result.

**Change C:** Increase `utterance_end_ms` from 1000 to 1200.

Replace the `TranscriptResult` dataclass (lines 59–63 in current file):

```python
@dataclass
class TranscriptResult:
    text: str
    is_final: bool
    speech_final: bool = False
    confidence: float = 0.0
```

Replace the `_on_message` inner function body (inside `transcribe_stream`, the try block):

```python
            async def _on_message(message):
                if not isinstance(message, ListenV1Results):
                    return
                try:
                    alternatives = message.channel.alternatives
                    if not alternatives:
                        return
                    sentence = alternatives[0].transcript
                    if not sentence:
                        return
                    confidence = alternatives[0].confidence
                    is_final = bool(message.is_final)
                    # speech_final fires once per utterance (after utterance_end_ms of silence).
                    # is_final fires at internal segment boundaries (mid-sentence on breath pauses).
                    speech_final = bool(getattr(message, "speech_final", False))
                    logger.debug(
                        "Deepgram transcript: is_final=%s speech_final=%s confidence=%.2f text=%r",
                        is_final, speech_final, confidence, sentence,
                    )
                    await on_transcript(TranscriptResult(
                        text=sentence,
                        is_final=is_final,
                        speech_final=speech_final,
                        confidence=float(confidence),
                    ))
                except Exception as e:
                    logger.error("Error in Deepgram on_message callback: %s", e)
```

Replace `utterance_end_ms=1000` with `utterance_end_ms=1200` in the `client.listen.v1.connect(...)` call.

---

## Task 2: Update `app/api/websocket.py`

Four targeted changes to this file.

### Change A: Gate `run_agent` on `speech_final` (not `is_final`)

In `on_transcript`, replace the early-return condition and log line.

Current lines:
```python
        if not result.is_final or not result.text.strip():
            return

        logger.info("Final transcript: %r (confidence=%.2f)", result.text, result.confidence)
```

Replace with:
```python
        if not result.speech_final or not result.text.strip():
            return

        logger.info(
            "Speech final: %r (confidence=%.2f)",
            result.text,
            result.confidence,
        )
```

The `await _send(websocket, {"type": "transcript", ...})` call stays as-is above this — we
still forward all interim and is_final transcripts to the client for live display.

### Change B: Forward `speech_final` to the client in the transcript message

Update the `_send` call for transcripts inside `on_transcript`:

```python
        await _send(websocket, {
            "type": "transcript",
            "text": result.text,
            "is_final": result.is_final,
            "speech_final": result.speech_final,  # NEW
        })
```

### Change C: Set `state["interrupted"] = True` on both interrupt paths

**Path 1 — New utterance arriving while agent is running** (inside `on_transcript`):

Current code:
```python
        if agent_task and not agent_task.done():
            logger.info("Cancelling previous agent task for new utterance")
            interrupt_event.set()
            agent_task.cancel()
            try:
                await agent_task
            except (asyncio.CancelledError, Exception):
                pass
            interrupt_event.clear()
```

Replace with:
```python
        if agent_task and not agent_task.done():
            logger.info("Cancelling previous agent task for new utterance")
            state["interrupted"] = True   # NEW: tell respond_node it was mid-turn
            interrupt_event.set()
            agent_task.cancel()
            try:
                await agent_task
            except (asyncio.CancelledError, Exception):
                pass
            interrupt_event.clear()
```

**Path 2 — Explicit interrupt message from client** (in the receive loop):

Current code:
```python
            elif msg_type == "interrupt":
                logger.info("Interrupt signal received from client")
                interrupt_event.set()
                task_was_running = bool(agent_task and not agent_task.done())
```

Replace with:
```python
            elif msg_type == "interrupt":
                logger.info("Interrupt signal received from client")
                state["interrupted"] = True   # NEW
                interrupt_event.set()
                task_was_running = bool(agent_task and not agent_task.done())
```

### Change D: Add `navigate` message handler

Add this elif branch in the receive loop, after the `elif msg_type == "start":` block:

```python
            elif msg_type == "navigate":
                # Client-initiated manual slide navigation (keyboard arrow keys).
                # Cancels any running agent, updates state, sends slide_change back.
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
                        state["interrupted"] = False   # manual nav is not an interrupt
                        slide = presentation.slides[target_index]
                        await _send(websocket, {
                            "type": "slide_change",
                            "index": target_index,
                            "slide": {"title": slide.title, "bullets": slide.bullets},
                        })
                        logger.info("Manual navigation to slide %d", target_index)
```

---

## Task 3: Rewrite `app/agent/prompts.py`

Full replacement. Key changes:
- `RESPOND_SYSTEM` now takes `{context_block}` and `{presentation_title}` parameters
- Natural voice rules with specific anti-patterns listed
- Occasional brief acknowledgments allowed (not mandated)
- Navigation and interruption context handled via system prompt (not user_msg prefix)

```python
from app.slides.content import Slide, slides_summary


def understand_system(slides: list[Slide]) -> str:
    """
    System prompt for understand_node. Generated at call time with the correct
    presentation's slide list so each session gets the right navigation context.
    """
    return f"""\
You are an AI presentation assistant.

The slide deck has the following slides:
{slides_summary(slides)}

Each line starts with [n] where n is the **0-based index** you must use for target_slide
(the first slide is [0], the second is [1], etc.).

Your job is to analyze the user's question or comment and decide:
1. Should we navigate to a different slide? (yes/no)
2. If yes, which slide index (0-based, matching the [n] in the list)?
3. What is the user's core intent in 1 sentence?

Respond ONLY with valid JSON in this exact format:
{{
  "should_navigate": true or false,
  "target_slide": <integer> or null,
  "intent_summary": "one sentence summary"
}}

Rules:
- If the user asks about a topic covered in a different slide, set should_navigate=true and target_slide to that slide's 0-based index from the list above.
- If the user says "slide N" or "Nth slide" counting from 1 (first slide = 1), convert to index N minus 1 (e.g. third slide → 2, fifth slide → 4).
- If the question is about the current slide or a general question, set should_navigate=false and target_slide=null.
- Never navigate away from a slide just because the user asks a clarifying question about it.
- Current slide index is provided in the user message (also 0-based).
"""


RESPOND_SYSTEM = """\
You are a live conference presenter on {presentation_title}. You're speaking to a small expert audience — it feels like a high-stakes technical conversation, not a lecture.

Current slide ({slide_index}): "{slide_title}"
Slide content:
{slide_bullets}

Background knowledge (your source material — use it, never read it verbatim):
{speaker_notes}

{context_block}\
=== VOICE RESPONSE RULES — follow all of these ===

FORBIDDEN OPENINGS — never start a response with:
"Of course", "Certainly", "Sure!", "Absolutely", "Great question", "Excellent question",
"That's a great point", "I'd be happy to", "Let me explain", "So basically", "As I mentioned"

NATURAL OPENINGS — vary how you begin:
- Most of the time: open directly with the substance.
  Examples: "The key bottleneck here is...", "What makes this interesting is...", "So on that point —"
- Occasionally (roughly 1 in 4 responses): a brief natural acknowledgment, then the answer.
  Acceptable pivots: "Right —", "Good point —", "Exactly —", "Fair question —"
  These must feel earned, not formulaic. If it doesn't fit naturally, skip it.

SLIDE TRANSITIONS — when you've just navigated to a new slide, introduce the topic naturally:
WRONG: "Great, we've moved to slide 3 which covers protocol deviation detection."
RIGHT: "Protocol monitoring is where most trials quietly lose control of their data..."

INTERRUPTIONS — if the context shows the user spoke while you were mid-response:
WRONG: "I was interrupted, but to address your new question..."
RIGHT: At most say "Right —" then answer the new question. Or just answer directly.
The user's new question is what matters. Don't reference the interruption mechanically.

LENGTH — 2 to 4 spoken sentences. This is a voice conversation, not an essay.
No bullet lists. No numbered points. Pure flowing spoken language.

STYLE — speak like an expert who knows this cold. Direct, specific, confident.
Never say: "bullet point", "slide", "as per", "in terms of", "it's worth noting", "I should mention"

END each response on a statement or insight, not a question. The user drives the conversation.
"""
```

---

## Task 4: Update `app/agent/nodes.py`

Three targeted changes:

### Change A: Build `context_block` in `respond_node` (replaces nav_context approach)

### Change B: Pass `presentation_title` to RESPOND_SYSTEM format call

### Change C: Return `"interrupted": False` from `respond_node` to reset state after handling

```python
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
                should_nav = False
                target = None
        except (TypeError, ValueError):
            logger.warning("LLM returned non-integer target_slide %r — ignoring", target)
            should_nav = False
            target = None
    elif should_nav and target is None:
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

    # Build context block for system prompt — shapes how the model handles
    # navigation transitions and interruptions without polluting user_msg.
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

    system = RESPOND_SYSTEM.format(
        presentation_title=presentation.meta.title,
        slide_index=slide.index,
        slide_title=slide.title,
        slide_bullets="\n".join(f"- {b}" for b in slide.bullets),
        speaker_notes=slide.speaker_notes,
        context_block=context_block,
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
        "slide_changed": False,   # reset for next turn
        "interrupted": False,     # reset after handling
    }


def should_navigate(state: AgentState) -> str:
    """Conditional edge: route to navigate_node or respond_node."""
    if state.get("should_navigate") and state.get("target_slide") is not None:
        return "navigate"
    return "respond"
```

### Also: merge `interrupted` back into session state in `run_agent`

In `app/api/websocket.py`, the `run_agent` function merges graph output keys back into
`state` via:

```python
        for key in (
            "current_slide",
            "target_slide",
            "should_navigate",
            "response_text",
            "slide_changed",
            "messages",
        ):
```

Add `"interrupted"` to this tuple so `respond_node`'s reset propagates back:

```python
        for key in (
            "current_slide",
            "target_slide",
            "should_navigate",
            "response_text",
            "slide_changed",
            "interrupted",   # NEW — respond_node resets this to False
            "messages",
        ):
```

---

## Task 5: Update `tests/test_agent_nodes.py`

The tests use `_make_state()`. Add `interrupted: False` default (it already exists in
`AgentState` but we want explicit default in the helper). Also add a test for the
`interrupted=True` context path in `respond_node`.

**Update `_make_state()`** — add `interrupted=False` to the base dict:

```python
def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "current_slide": 0,
        "target_slide": None,
        "messages": [],
        "transcript": "test transcript",
        "response_text": "",
        "slide_changed": False,
        "interrupted": False,         # ensure explicit default
        "should_navigate": False,
        "presentation_id": "clinical-trials",
    }
    base.update(overrides)
    return base
```

**Add one new test class** for the interrupted context:

```python
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
```

---

## Task 6: Run tests

```bash
cd backend
source venv/bin/activate
pytest tests/ -v
```

All existing tests must pass. The new test class adds 2 tests.

**Common failure:** If `test_websocket_protocol.py` tests that mock `transcribe_stream` and
check that `run_agent` is called — they may need updating because `on_transcript` now
gates on `speech_final`. Check if any test passes `TranscriptResult(is_final=True)` without
`speech_final=True`. If so, add `speech_final=True` to those mock objects.

---

## Task 7: Manual verification

```bash
source venv/bin/activate
uvicorn app.main:app --port 8000 --log-level debug
```

### Test A: No premature thinking on mid-sentence pause
1. Start the frontend, connect, start mic
2. Speak a sentence with a deliberate pause mid-way: "Tell me about... [2 second pause] ...AlphaFold"
3. **Expected:** Deepgram logs show `speech_final=False` on "Tell me about", then
   `speech_final=True` on the complete transcript. Agent runs ONCE with the full sentence.
4. **Old behavior:** Agent would have run on "Tell me about" then been cancelled by "AlphaFold"

### Test B: Natural prompt tone — no "Of course!"
1. Ask any question
2. Check the backend log for `agent_text:` content and/or listen to the response
3. **Expected:** Response opens directly with substance, or occasionally with "Right—" / "Good point—"
4. **Not expected:** "Of course!", "Certainly!", "That's a great question!", "Let me explain"

### Test C: Interrupted acknowledgment is brief
1. Ask a question, wait for TTS to start playing
2. Interrupt with a different question
3. **Expected:** Response to new question, possibly starting with "Right—", doesn't mechanically
   reference being interrupted ("as I was saying...", "I was mid-response but...")

### Test D: Navigate message works (keyboard nav will test this via Plan 11)
```bash
# Via wscat or browser DevTools WS console:
# After connecting and getting slide 0:
{"type": "navigate", "index": 2}
# Expected: {"type": "slide_change", "index": 2, "slide": {...}}
```

---

## Acceptance Criteria

- [ ] `TranscriptResult` has `speech_final: bool = False`
- [ ] `utterance_end_ms=1200` in Deepgram connect params
- [ ] `on_transcript` gates `run_agent` on `result.speech_final` (not `result.is_final`)
- [ ] Transcript WebSocket message includes `speech_final` field
- [ ] `state["interrupted"] = True` set in both interrupt paths in `websocket.py`
- [ ] `"interrupted"` in the `run_agent` merge key tuple
- [ ] `navigate` message handler in `websocket.py` — updates slide, sends `slide_change`
- [ ] `RESPOND_SYSTEM` — has `{context_block}`, `{presentation_title}`, no forbidden openings in rules
- [ ] `respond_node` — builds `context_block`, returns `"interrupted": False`
- [ ] All tests pass (38 tests: 36 existing + 2 new)

## File Checklist After This Plan

```
backend/
  app/
    services/
      stt.py              ← speech_final field + utterance_end_ms=1200
    api/
      websocket.py        ← speech_final gate + interrupted state + speech_final forwarded + navigate handler
    agent/
      prompts.py          ← RESPOND_SYSTEM rewritten with context_block + presentation_title
      nodes.py            ← respond_node builds context_block, resets interrupted
  tests/
    test_agent_nodes.py   ← interrupted default in _make_state, 2 new tests
```
