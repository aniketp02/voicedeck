from dataclasses import dataclass, field


@dataclass
class Slide:
    index: int
    title: str
    bullets: list[str]
    speaker_notes: str
    keywords: list[str]  # used by LLM to route navigation


SLIDES: list[Slide] = [
    Slide(
        index=0,
        title="The Broken Machine: Clinical Trials Today",
        bullets=[
            "Average trial takes 10–15 years and costs $2.6 billion",
            "90% of drugs that enter trials never reach patients",
            "Only 3–5% of eligible patients actually enroll",
            "Largely paper-based, siloed, and manually intensive",
        ],
        speaker_notes=(
            "Let's set the scene. The clinical trial industry is fundamentally broken. "
            "It takes over a decade and more than two billion dollars to bring a single drug to market — "
            "and nine out of ten drugs that enter trials still fail. The system is slow, expensive, "
            "and leaves most patients behind. This is the problem AI has the opportunity to solve."
        ),
        keywords=["problem", "broken", "today", "current", "traditional", "challenges", "overview", "introduction", "start"],
    ),
    Slide(
        index=1,
        title="AI-Powered Patient Recruitment",
        bullets=[
            "NLP scans EHR records to match patients to eligibility criteria",
            "Reduces recruitment timeline by up to 80%",
            "Surfaces underrepresented populations for diversity goals",
            "Real-time eligibility screening at point of care",
        ],
        speaker_notes=(
            "Patient recruitment is the single biggest bottleneck in clinical research — "
            "it accounts for over 30% of trial delays. AI changes this fundamentally. "
            "By applying natural language processing to electronic health records, we can scan "
            "millions of patient records in hours instead of months, identifying candidates who "
            "meet complex eligibility criteria that would take human coordinators weeks to verify."
        ),
        keywords=["recruitment", "patients", "enrollment", "EHR", "eligibility", "NLP", "matching", "diversity"],
    ),
    Slide(
        index=2,
        title="Real-Time Protocol Deviation Detection",
        bullets=[
            "AI monitors site data continuously for protocol violations",
            "Flags deviations before they become audit findings",
            "Reduces trial invalidation risk by catching errors early",
            "Integrates with EDC systems for zero-friction monitoring",
        ],
        speaker_notes=(
            "Protocol deviations are expensive. A single undetected deviation can invalidate an "
            "entire trial arm and cost tens of millions in remediation. Traditional monitoring is "
            "retrospective — auditors find issues after they've already caused damage. "
            "AI flips this to real-time: continuously watching every data point entered across "
            "every site and flagging anomalies the moment they occur."
        ),
        keywords=["protocol", "deviation", "monitoring", "compliance", "audit", "detection", "EDC", "data quality"],
    ),
    Slide(
        index=3,
        title="Real-World Evidence & Synthetic Control Arms",
        bullets=[
            "RWE from claims, registries, and wearables supplements trial data",
            "Synthetic control arms reduce or eliminate placebo groups",
            "FDA has accepted synthetic controls in rare disease approvals",
            "Cuts trial size and cost while maintaining statistical validity",
        ],
        speaker_notes=(
            "One of the most exciting developments is the use of real-world evidence and synthetic "
            "control arms. Instead of randomizing half your patients to a placebo — which is expensive "
            "and ethically fraught in rare diseases — you can construct a synthetic control arm from "
            "historical data. The FDA has already accepted this approach in several rare disease "
            "approvals. This is AI enabling better science, not just faster science."
        ),
        keywords=["real-world evidence", "RWE", "synthetic control", "placebo", "FDA", "rare disease", "wearables", "registries"],
    ),
    Slide(
        index=4,
        title="Regulatory AI: FDA's Evolving Framework",
        bullets=[
            "FDA's 2021 AI/ML Action Plan outlines adaptive approval pathways",
            "Predetermined Change Control Plans (PCCPs) enable model updates post-approval",
            "Good Machine Learning Practice (GMLP) is the emerging standard",
            "EU AI Act creates parallel compliance requirements for global trials",
        ],
        speaker_notes=(
            "Regulation is often seen as the enemy of innovation, but the FDA has been surprisingly "
            "forward-thinking on AI. The 2021 AI/ML Action Plan and the concept of Predetermined "
            "Change Control Plans mean that AI systems can be updated after approval without full "
            "resubmission — a massive unlock for adaptive AI in clinical settings. "
            "Understanding this regulatory landscape is non-negotiable for anyone building in this space."
        ),
        keywords=["regulation", "FDA", "regulatory", "compliance", "PCCP", "GMLP", "EU AI Act", "approval", "framework"],
    ),
    Slide(
        index=5,
        title="The Road Ahead: Autonomous Trial Management",
        bullets=[
            "Fully adaptive trials that self-modify based on interim results",
            "AI principal investigators managing multi-site coordination",
            "Continuous consent and real-time patient engagement via voice AI",
            "From 10 years to 18 months: the 2030 vision",
        ],
        speaker_notes=(
            "Where does this all lead? The 2030 vision is a clinical trial that runs itself. "
            "Adaptive protocols that modify dosing and eligibility in real time based on interim data. "
            "AI coordinators that manage hundreds of sites simultaneously. "
            "And voice AI — exactly what we're building today — as the interface between patients "
            "and the trial itself: consent, reminders, symptom reporting, all through natural conversation. "
            "The opportunity is enormous. The tools exist today. The question is who builds it first."
        ),
        keywords=["future", "vision", "autonomous", "2030", "voice AI", "adaptive", "next steps", "roadmap", "conclusion"],
    ),
]


def get_slide(index: int) -> Slide:
    if 0 <= index < len(SLIDES):
        return SLIDES[index]
    raise IndexError(f"Slide index {index} out of range (0-{len(SLIDES)-1})")


def slides_summary() -> str:
    """Compact representation for LLM system prompts."""
    lines = []
    for s in SLIDES:
        lines.append(f"[{s.index}] {s.title} — keywords: {', '.join(s.keywords)}")
    return "\n".join(lines)
