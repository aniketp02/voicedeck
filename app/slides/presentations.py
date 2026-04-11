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
