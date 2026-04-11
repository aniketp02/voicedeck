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
