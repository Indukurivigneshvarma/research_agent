import os
from typing import Dict, List
from openai import OpenAI

# --------------------------------------------------
# LLM CLIENT SETUP (OpenRouter)
# --------------------------------------------------
# OpenRouter is used as a unified gateway to access large
# instruction-tuned models for long-form structured writing.
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Default large model for report synthesis
# Can be overridden via environment variable.
MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.1-70b-instruct"
)


def write_report(
    title: str,
    headings: List[str],
    summaries: Dict[str, str],
    references: List[str],
) -> str:
    """
    ACADEMIC REPORT GENERATOR
    =========================

    Purpose
    -------
    This function generates the **final academic research report text**
    using the validated summaries as the ONLY source of factual content.

    This is the core synthesis stage where the system converts:
        • structured headings
        • resolved summaries
        • citation mappings
    into a coherent, publication-style research report.

    Why this step is critical
    -------------------------
    The pipeline separates:
        1. Retrieval
        2. Validation & scoring
        3. Conflict resolution
        4. Structure generation
        5. **Final synthesis (THIS STEP)**

    This ensures the final report:
        ✓ Is grounded in sources
        ✓ Uses consistent structure
        ✓ Enforces strict citation discipline
        ✓ Avoids hallucinated facts

    Inputs
    ------
    title : str
        Final academic report title generated earlier.

    headings : List[str]
        Ordered list of report section headings.
        These are mandatory and cannot be altered by the LLM.

    summaries : Dict[str, str]
        Mapping:
            {
              "S1": "summary text",
              "S2": "summary text"
            }
        These are the ONLY allowed factual sources.

    references : List[str]
        Pre-built reference entries (deterministic).
        The LLM must copy them exactly without modification.

    Output
    ------
    str
        Fully formatted research report containing:
        - Title block
        - All required sections
        - Sentence-level citations
        - Final references section

    Key Guarantees Enforced by Prompt
    ---------------------------------
    • Sentence-level citations required everywhere
    • No uncited claims
    • No new facts allowed
    • Structured section format
    • References section is deterministic
    """

    # --------------------------------------------------
    # Prepare structured inputs for LLM
    # --------------------------------------------------

    # Summaries block: labeled source text for citation linking
    summary_block = "\n".join(
        f"{sid}: {text}"
        for sid, text in summaries.items()
    )

    # Headings wrapped in markers so the model must follow structure
    heading_block = "\n".join(
        f"@@{h}@@" for h in headings
    )

    # Reference entries that must be copied verbatim
    refs_block = "\n".join(references)

    # --------------------------------------------------
    # Prompt enforces formatting, grounding, and citation discipline
    # --------------------------------------------------
    prompt = f"""
You are writing an academic research synthesis.

YOU MUST FOLLOW THE FORMAT EXACTLY.

================ FORMAT RULES ================
- Use ONLY the headings provided.
- Each heading MUST appear exactly once.
- Headings MUST be wrapped like this: @@Heading Name@@
- The title MUST be wrapped like this:
  @@TITLE@@
  <title>
  @@TITLE@@
- Plain text only.
- No markdown.
- No bullet points.

================ CITATION RULES ================
- EVERY declarative sentence in ALL sections
  (including Executive Summary and Conclusion)
  MUST end with one or more citation markers.
- Citations must be placed immediately after the sentence.
- Use format: [S1] or [S1][S3]
- NO paragraph-level citations.
- NO uncited sentences anywhere except References.

================ CONTENT RULES ================
- You MAY paraphrase and synthesize.
- You MAY combine information from multiple summaries.
- You MUST NOT introduce facts not present in the summaries.
- If evidence is uncertain, state uncertainty explicitly.

- Executive Summary:
  Write 5–6 well-developed sentences.

- EACH topical section:
  Write 2–3 coherent paragraphs.
  Each paragraph must contain 4–5 sentences.
  Do NOT merge all content into a single paragraph.

- Conclusion:
  Write 5–6 sentences that synthesize arguments,
  implications, and limitations.

- Short or underdeveloped sections are NOT acceptable.

================ REFERENCES RULE ================
- When you reach @@References@@
- DO NOT write new text.
- OUTPUT EXACTLY the reference list provided.
- Do NOT modify references.

================ INPUTS ================

TITLE:
{title}

HEADINGS:
{heading_block}

SUMMARIES (ONLY SOURCE OF FACTS):
{summary_block}

REFERENCES (VERBATIM — DO NOT CHANGE):
{refs_block}

================ OUTPUT ================
Return the COMPLETE report using the exact format described.
Do NOT include anything else.
""".strip()

    # --------------------------------------------------
    # LLM Call (long-form generation)
    # --------------------------------------------------
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # low temperature → controlled synthesis
        max_tokens=2500,  # large context for full report
    )

    return r.choices[0].message.content.strip()
