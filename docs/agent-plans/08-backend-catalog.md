# Backend Agent Plan 08 — Multi-Presentation Support

## Agent Instructions
Read this plan fully before acting. Do not ask questions. All decisions are made.
Plans 01–05 (backend) must be complete before starting.

**Parallel execution:** This plan runs in parallel with frontend Plans 06 and 07.
Frontend Plan 09 depends on this plan completing first.

---

## Goal
Add a second presentation ("AI in Drug Discovery") and make the backend route sessions to the
correct presentation based on the `presentation_id` sent in the WebSocket `start` message.

The primary blocker is `app/agent/prompts.py` — `UNDERSTAND_SYSTEM` is a module-level
formatted string baked with clinical trials content at import time. Every session uses it
regardless of which presentation is active. This plan converts it to a runtime function.

**Files changed:**
- `app/slides/drug_discovery.py` — NEW: 6 Drug Discovery slides with full content
- `app/slides/presentations.py` — NEW: registry, `PresentationMeta`, lookup functions
- `app/slides/content.py` — `slides_summary()` accepts optional `slides` param
- `app/agent/state.py` — add `presentation_id: str` field
- `app/agent/prompts.py` — `UNDERSTAND_SYSTEM` → `understand_system(slides)` function
- `app/agent/nodes.py` — use `understand_system()` + presentation-aware slide lookup
- `app/api/websocket.py` — read `presentation_id` from `start` message, seed state
- `app/main.py` — add `GET /presentations` endpoint
- `tests/test_agent_nodes.py` — add `presentation_id` to `_make_state()`

---

## Task 1: Create `app/slides/drug_discovery.py`

New file. Contains 6 slides on AI in Drug Discovery with full speaker notes and keywords.

```python
from app.slides.content import Slide

DRUG_DISCOVERY_SLIDES: list[Slide] = [
    Slide(
        index=0,
        title="The Drug Discovery Crisis",
        bullets=[
            "Developing a new drug costs $2.6B on average and takes 12–15 years",
            "Less than 10% of candidates that enter clinical trials reach patients",
            "Target identification and lead optimization account for 60% of failures",
            "The industry loses ~$50B annually to late-stage clinical failures",
        ],
        speaker_notes=(
            "Let's start with the scope of the problem. Drug discovery is one of the most expensive "
            "and failure-prone endeavors in science. The numbers are staggering — over two billion "
            "dollars and more than a decade for a single drug. And the failure rate is brutal: nine "
            "out of ten drugs that make it to clinical trials never reach patients. Most of that "
            "failure traces back to decisions made early in the pipeline — wrong target, wrong "
            "molecule, undetected toxicity. AI attacks all three of those root causes."
        ),
        keywords=[
            "problem", "crisis", "traditional", "current", "overview", "introduction",
            "challenges", "broken", "cost", "failure", "start", "beginning",
        ],
    ),
    Slide(
        index=1,
        title="AI-Powered Target Identification",
        bullets=[
            "Deep learning models identify disease-associated proteins from genomic datasets",
            "Graph neural networks map protein-protein interactions to surface novel targets",
            "Multimodal AI integrates CRISPR screens, proteomics, and literature at scale",
            "Reduces target identification phase from 3–5 years to weeks",
        ],
        speaker_notes=(
            "The first step in drug discovery is finding the right target — a protein or biological "
            "pathway whose disruption would treat the disease. Traditionally this took years of "
            "painstaking bench science. AI flips this. By training deep learning models on genomic "
            "and proteomic data from thousands of patients, we can identify which proteins are "
            "causally linked to disease rather than just correlated with it. Graph neural networks "
            "are particularly powerful here — they can map the full protein-protein interaction "
            "network and identify chokepoints that human biologists would never find manually. "
            "Companies like Recursion and Exscientia are already using this approach to generate "
            "target hypotheses in weeks instead of years."
        ),
        keywords=[
            "target", "target identification", "proteins", "genomics", "CRISPR",
            "deep learning", "graph neural networks", "proteomics", "disease",
            "biological target", "pathway",
        ],
    ),
    Slide(
        index=2,
        title="AlphaFold and the Structural Revolution",
        bullets=[
            "AlphaFold2 predicted structures for 200M+ proteins — essentially all known proteins",
            "Enables structure-based drug design without expensive X-ray crystallography",
            "Opens previously 'undruggable' targets to small-molecule intervention",
            "BioNTech, Novo Nordisk, and GSK actively use AlphaFold structures in pipelines",
        ],
        speaker_notes=(
            "AlphaFold2 was one of the most significant scientific breakthroughs of the decade. "
            "For fifty years, predicting a protein's 3D structure from its amino acid sequence — "
            "the protein folding problem — was considered one of biology's grand challenges. "
            "DeepMind solved it. The AlphaFold database now contains predicted structures for over "
            "200 million proteins. What does this unlock for drug discovery? Structure-based drug "
            "design. Instead of first crystallizing a protein (expensive, slow, sometimes impossible), "
            "you start with the predicted structure and design molecules that fit its binding site. "
            "This is especially powerful for 'undruggable' targets — proteins that were previously "
            "inaccessible because we couldn't determine their structure. The addressable target "
            "space just expanded dramatically."
        ),
        keywords=[
            "AlphaFold", "protein structure", "structural biology", "drug design",
            "crystallography", "undruggable", "structure-based", "Deepmind",
            "protein folding", "3D structure",
        ],
    ),
    Slide(
        index=3,
        title="ADMET Prediction and Lead Optimization",
        bullets=[
            "AI predicts Absorption, Distribution, Metabolism, Excretion, Toxicity in silico",
            "Multi-property optimization: efficacy + safety + synthesizability simultaneously",
            "Cuts lead optimization from 18–24 months to weeks",
            "Eliminates wet-lab failures by predicting problem compounds before synthesis",
        ],
        speaker_notes=(
            "Finding a target is only the first step. You then need to design a molecule that hits "
            "that target effectively AND is safe AND can be manufactured AND will actually survive "
            "the journey through the body to reach its target. This is the ADMET problem — "
            "Absorption, Distribution, Metabolism, Excretion, Toxicity. Traditionally, lead "
            "optimization meant synthesizing hundreds of compounds, testing each one, failing most, "
            "and iterating. AI models trained on historical ADMET data can now predict these "
            "properties in silico — before a single molecule is synthesized. The game-changer is "
            "multi-property optimization: you can simultaneously optimize for binding affinity, "
            "low toxicity, good bioavailability, and synthetic accessibility. What used to take "
            "18–24 months of iterative chemistry now takes weeks of computation."
        ),
        keywords=[
            "ADMET", "lead optimization", "toxicity", "absorption", "metabolism",
            "in silico", "safety", "efficacy", "prediction", "bioavailability",
            "pharmacokinetics", "PK", "DMPK",
        ],
    ),
    Slide(
        index=4,
        title="Generative Molecular Design",
        bullets=[
            "Diffusion models and VAEs generate novel molecules with target binding properties",
            "Reinforcement learning optimizes for binding affinity and synthetic accessibility",
            "First AI-designed clinical candidates: Insilico INS018_055, Exscientia EXS21546",
            "Generative chemistry compresses the design-make-test cycle from months to days",
        ],
        speaker_notes=(
            "The most exciting frontier is generative molecular design — using AI not just to "
            "evaluate molecules but to create them from scratch. Diffusion models (the same "
            "architecture behind image generators like DALL-E) can generate novel molecular "
            "structures that have never existed before, optimized for specific binding properties. "
            "Reinforcement learning agents explore chemical space, learning which modifications "
            "improve the target score. The proof is real: Insilico Medicine used AI to design "
            "INS018_055, a drug for idiopathic pulmonary fibrosis, which entered Phase II trials "
            "in 2023 after being designed in just 18 months. Exscientia has multiple AI-designed "
            "candidates in clinical trials. The design-make-test cycle that previously took "
            "months now takes days — the bottleneck has shifted from ideation to synthesis."
        ),
        keywords=[
            "generative", "molecular design", "diffusion", "VAE", "drug design",
            "synthesis", "molecule generation", "Insilico", "Exscientia",
            "binding affinity", "chemical space", "AI designed",
        ],
    ),
    Slide(
        index=5,
        title="Toward Autonomous Drug Discovery",
        bullets=[
            "Closed-loop platforms: AI designs, robotic labs synthesize, AI re-evaluates",
            "BioFoundries at Ginkgo and UCSF run 10,000+ experiments weekly, all AI-directed",
            "Estimated 10× reduction in preclinical R&D costs by 2030",
            "First wave of fully AI-discovered drugs entering pivotal trials by 2026",
        ],
        speaker_notes=(
            "Where is this heading? The 2030 vision is a fully autonomous drug discovery engine. "
            "Closed-loop platforms already exist — AI designs a molecule, a robotic lab synthesizes "
            "it and runs the assay, the result feeds back into the AI model, which generates the "
            "next design. Ginkgo Bioworks and UCSF's BioFoundries run tens of thousands of "
            "experiments per week, all AI-directed. No human biologist could keep up with that "
            "throughput. The economics are transformational: estimates suggest 10× reduction in "
            "preclinical R&D costs by 2030. We're already seeing the first wave of fully "
            "AI-discovered drugs entering pivotal trials. The question for pharma incumbents is "
            "not whether to adopt AI — it's how fast they can transform their pipelines before "
            "AI-native startups eat their lunch."
        ),
        keywords=[
            "autonomous", "future", "2030", "closed-loop", "robotic", "biofoundry",
            "roadmap", "next steps", "vision", "conclusion", "automation",
            "Ginkgo", "UCSF", "platform",
        ],
    ),
]
```

---

## Task 2: Create `app/slides/presentations.py`

New file. Registry of all available presentations. Exposes typed metadata for the API endpoint
and lookup functions for the agent.

```python
"""
Presentation registry.

Add new presentations by:
1. Creating app/slides/<topic>.py with a SLIDES list (see drug_discovery.py for the pattern)
2. Adding an entry to PRESENTATIONS below
"""
from dataclasses import dataclass

from app.slides.content import SLIDES as CLINICAL_TRIALS_SLIDES, Slide
from app.slides.drug_discovery import DRUG_DISCOVERY_SLIDES


@dataclass
class PresentationMeta:
    id: str
    title: str
    description: str
    slide_count: int


@dataclass
class Presentation:
    meta: PresentationMeta
    slides: list[Slide]


PRESENTATIONS: dict[str, Presentation] = {
    "clinical-trials": Presentation(
        meta=PresentationMeta(
            id="clinical-trials",
            title="AI in Clinical Trials",
            description="How AI is transforming patient recruitment, protocol monitoring, and trial design",
            slide_count=len(CLINICAL_TRIALS_SLIDES),
        ),
        slides=CLINICAL_TRIALS_SLIDES,
    ),
    "drug-discovery": Presentation(
        meta=PresentationMeta(
            id="drug-discovery",
            title="AI in Drug Discovery",
            description="From target identification and AlphaFold to generative molecular design",
            slide_count=len(DRUG_DISCOVERY_SLIDES),
        ),
        slides=DRUG_DISCOVERY_SLIDES,
    ),
}

DEFAULT_PRESENTATION_ID = "clinical-trials"


def get_presentation(presentation_id: str) -> Presentation:
    """Return the Presentation for the given ID. Raises KeyError if not found."""
    if presentation_id not in PRESENTATIONS:
        raise KeyError(
            f"Unknown presentation {presentation_id!r}. "
            f"Available: {list(PRESENTATIONS.keys())}"
        )
    return PRESENTATIONS[presentation_id]


def list_presentations() -> list[PresentationMeta]:
    """Return metadata for all registered presentations (for the catalog API)."""
    return [p.meta for p in PRESENTATIONS.values()]
```

---

## Task 3: Update `app/slides/content.py`

Make `slides_summary()` accept an optional `slides` parameter so nodes can call it with
any presentation's slides. The no-arg form keeps backward compatibility.

**Targeted edit:** Replace the `slides_summary` function only (lines 125–136 in current file):

```python
def slides_summary(slides: list[Slide] | None = None) -> str:
    """Compact representation for LLM system prompts."""
    target = slides if slides is not None else SLIDES
    lines = []
    for s in target:
        lines.append(f"[{s.index}] {s.title} — keywords: {', '.join(s.keywords)}")
    return "\n".join(lines)
```

---

## Task 4: Update `app/agent/state.py`

Add `presentation_id: str` to `AgentState`. This is how nodes know which presentation is active.

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Slide navigation
    current_slide: int                       # 0-based index of active slide
    target_slide: Optional[int]              # set by understand_node, consumed by navigate_node

    # Conversation
    messages: Annotated[list, add_messages]  # full chat history (LangGraph managed)
    transcript: str                          # latest user utterance

    # Agent output
    response_text: str                       # text the agent will speak
    slide_changed: bool                      # whether a slide change just occurred

    # Control flow
    interrupted: bool                        # set True when client sends interrupt
    should_navigate: bool                    # set by understand_node

    # Presentation routing — NEW
    presentation_id: str                     # which presentation this session uses
```

---

## Task 5: Update `app/agent/prompts.py`

Convert `UNDERSTAND_SYSTEM` from a module-level formatted string to a runtime function.
`RESPOND_SYSTEM` stays as a template string (it's already runtime-formatted in nodes.py).

The old `UNDERSTAND_SYSTEM = "...".format(slides_summary=slides_summary())` baked the clinical
trials summary at import time. The new function calls `slides_summary(slides)` at call time.

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

Your job is to analyze the user's question or comment and decide:
1. Should we navigate to a different slide? (yes/no)
2. If yes, which slide index?
3. What is the user's core intent in 1 sentence?

Respond ONLY with valid JSON in this exact format:
{{
  "should_navigate": true or false,
  "target_slide": <integer> or null,
  "intent_summary": "one sentence summary"
}}

Rules:
- If the user asks about a topic covered in a different slide, set should_navigate=true and target_slide to that slide's index.
- If the question is about the current slide or a general question, set should_navigate=false and target_slide=null.
- Never navigate away from a slide just because the user asks a clarifying question about it.
- Current slide index is provided in the user message.
"""


RESPOND_SYSTEM = """\
You are an engaging AI presenter delivering a talk to an expert audience.

You are currently presenting slide {slide_index}: "{slide_title}"

Slide bullets:
{slide_bullets}

Speaker notes (your knowledge base — do NOT read these verbatim):
{speaker_notes}

Rules:
- Speak naturally, like a knowledgeable presenter — not like you're reading bullets.
- Keep responses concise: 2-4 sentences max unless the user asks for more detail.
- If the user asked a specific question, answer it directly first, then connect back to the slide.
- If we just navigated to this slide, introduce it briefly before answering.
- Never say "bullet point" or "as per the slide". Just speak.
- End with a natural pause point, not a cliffhanger question (the user will speak next).
"""
```

---

## Task 6: Update `app/agent/nodes.py`

Three changes:
1. `understand_node`: call `understand_system(slides)` with the presentation's slides
2. `understand_node`: bounds check against `len(slides)` not hardcoded `len(SLIDES)`
3. `respond_node`: look up slide from the correct presentation, not always clinical trials

```python
"""
LangGraph nodes for the voice agent.

Node execution order:
  understand_node → (conditional) → navigate_node → respond_node
                  ↘ respond_node (if no navigation needed)
"""
import logging

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

---

## Task 7: Update `app/api/websocket.py`

Read `presentation_id` from the `start` WebSocket message.
Validate it — fall back to `"clinical-trials"` if unknown.
Seed `AgentState` with `presentation_id`.
Use the correct presentation's slides for the initial `slide_change` message.

**Targeted changes to `handle_session()`:**

1. Add import at the top of the file:
```python
from app.slides.presentations import get_presentation, DEFAULT_PRESENTATION_ID
```

2. Inside `handle_session()`, move the `initial_slide` send to AFTER the receive loop starts
   processing the `start` message. Change the initial state to include `presentation_id`.
   
   Replace the current state initialization and initial slide send block with:

```python
    # presentation_id is determined when the client sends {"type": "start", "presentation_id": "..."}
    # We initialize state with a placeholder; it's replaced when start is received.
    presentation_id = DEFAULT_PRESENTATION_ID

    state: AgentState = {
        "current_slide": 0,
        "target_slide": None,
        "messages": [],
        "transcript": "",
        "response_text": "",
        "slide_changed": False,
        "interrupted": False,
        "should_navigate": False,
        "presentation_id": presentation_id,
    }
```

3. Replace the current `start` message handler in the receive loop:

```python
            elif msg_type == "start":
                # Client may specify which presentation to use.
                # Default to clinical-trials if absent or unrecognized.
                requested_id = msg.get("presentation_id", DEFAULT_PRESENTATION_ID)
                try:
                    presentation = get_presentation(requested_id)
                    presentation_id = requested_id
                except KeyError:
                    logger.warning(
                        "Unknown presentation_id %r — using default %r",
                        requested_id,
                        DEFAULT_PRESENTATION_ID,
                    )
                    presentation = get_presentation(DEFAULT_PRESENTATION_ID)

                state["presentation_id"] = presentation.meta.id
                initial_slide = presentation.slides[0]
                await _send(websocket, {
                    "type": "slide_change",
                    "index": 0,
                    "slide": {"title": initial_slide.title, "bullets": initial_slide.bullets},
                })
                logger.info(
                    "Session started: presentation=%r slides=%d",
                    presentation.meta.id,
                    presentation.meta.slide_count,
                )
```

4. Remove the hard-coded `initial_slide` send that currently happens before the receive loop
   (lines ~151–157 in the current file). The initial slide is now sent in response to the
   `start` message, so the client always controls when the session begins.

**The complete updated `handle_session` signature and first section** (everything before the
receive loop):

```python
async def handle_session(websocket: WebSocket) -> None:
    """
    Main WebSocket session lifecycle.

    Pipeline:
    1. Accept connection
    2. Wait for {"type": "start", "presentation_id": "..."} from client
    3. Send initial slide_change for slide 0 of the chosen presentation
    4. Start Deepgram STT as background task
    5. Receive loop: route audio_chunk → audio_queue, interrupt → interrupt_event
    6. on_transcript callback: forward to client; on final → run_agent
    7. Graceful shutdown on disconnect or error
    """
    await websocket.accept()
    logger.info("WebSocket session started")

    presentation_id = DEFAULT_PRESENTATION_ID

    state: AgentState = {
        "current_slide": 0,
        "target_slide": None,
        "messages": [],
        "transcript": "",
        "response_text": "",
        "slide_changed": False,
        "interrupted": False,
        "should_navigate": False,
        "presentation_id": presentation_id,
    }

    interrupt_event = asyncio.Event()
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=100)
    agent_task: asyncio.Task | None = None
    # (do NOT send initial slide_change here — wait for start message)
```

Also add the import at the top of the file with the other imports:
```python
from app.slides.presentations import get_presentation, DEFAULT_PRESENTATION_ID
```

And remove the old import:
```python
from app.slides.content import get_slide   # ← delete this line (no longer used in websocket.py)
```

Note: `get_slide` from `content.py` is still used in `nodes.py`, just not in `websocket.py`.
The presentation-aware lookup happens in nodes now.

---

## Task 8: Update `app/main.py`

Add `GET /presentations` endpoint that returns the catalog for the frontend selector.

```python
@app.get("/presentations")
async def list_presentations_endpoint():
    """Return all available presentations for the catalog UI."""
    from app.slides.presentations import list_presentations
    presentations = list_presentations()
    return [
        {
            "id": p.id,
            "title": p.title,
            "description": p.description,
            "slide_count": p.slide_count,
        }
        for p in presentations
    ]
```

Add this after the existing `@app.get("/slides")` route.

---

## Task 9: Update `tests/test_agent_nodes.py`

The `_make_state()` helper creates `AgentState` dicts for tests. After adding `presentation_id`
to `AgentState`, any test that creates a state without this field will fail with a TypedDict
validation error or mypy error.

**Update `_make_state()`** to include `presentation_id`:

```python
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
        "presentation_id": "clinical-trials",   # NEW
    }
    base.update(overrides)
    return base
```

No other test changes needed — the nodes use `get_presentation(state["presentation_id"])` which
resolves to clinical-trials slides, matching the existing test assertions.

---

## Task 10: Run tests

```bash
cd backend
source venv/bin/activate
pytest tests/ -v
```

All 36 existing tests should pass. If any test fails:
- `test_agent_nodes.py` failures: likely missing `presentation_id` in `_make_state()` — apply Task 9
- `test_websocket_protocol.py` failures: the `start` message tests may need `presentation_id` added.
  Check if any test sends `{"type": "start"}` and expects an immediate `slide_change` response —
  after this plan, `slide_change` is sent in response to the `start` message (same behavior,
  just ordered differently). Adjust test timing if needed.

---

## Task 11: Manual verification

```bash
# Start backend
source venv/bin/activate
uvicorn app.main:app --port 8000 --log-level info

# In another terminal — verify catalog endpoint
curl http://localhost:8000/presentations
```

Expected response:
```json
[
  {
    "id": "clinical-trials",
    "title": "AI in Clinical Trials",
    "description": "How AI is transforming patient recruitment, protocol monitoring, and trial design",
    "slide_count": 6
  },
  {
    "id": "drug-discovery",
    "title": "AI in Drug Discovery",
    "description": "From target identification and AlphaFold to generative molecular design",
    "slide_count": 6
  }
]
```

Then test WebSocket routing manually (optional — the frontend Plan 09 will test this end-to-end).

---

## Acceptance Criteria

- [ ] `app/slides/drug_discovery.py` — 6 slides with titles, bullets, speaker_notes, keywords
- [ ] `app/slides/presentations.py` — registry with both presentations, `get_presentation()`, `list_presentations()`
- [ ] `app/slides/content.py` — `slides_summary(slides=None)` works with and without arg
- [ ] `app/agent/state.py` — `AgentState` has `presentation_id: str`
- [ ] `app/agent/prompts.py` — `understand_system(slides)` function, not module-level string
- [ ] `app/agent/nodes.py` — uses `understand_system()`, slides from presentation registry, bounds check uses `len(slides)`
- [ ] `app/api/websocket.py` — `presentation_id` read from `start` message, initial slide from correct presentation
- [ ] `app/main.py` — `GET /presentations` returns catalog JSON
- [ ] All 36 existing tests pass
- [ ] `curl /presentations` returns 2 presentations

## File Checklist After This Plan

```
backend/
  app/
    slides/
      content.py            ← slides_summary() accepts optional slides param
      drug_discovery.py     ← NEW: 6 Drug Discovery slides
      presentations.py      ← NEW: registry + lookup functions
    agent/
      state.py              ← presentation_id field added
      prompts.py            ← understand_system() function (not module-level string)
      nodes.py              ← presentation-aware lookup in understand + respond
    api/
      websocket.py          ← reads presentation_id from start message
    main.py                 ← GET /presentations endpoint
  tests/
    test_agent_nodes.py     ← _make_state() includes presentation_id
```
