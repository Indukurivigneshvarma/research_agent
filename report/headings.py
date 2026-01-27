import os
import json
from typing import List, Dict
from groq import Groq

# --------------------------------------------------
# LLM CLIENT SETUP
# --------------------------------------------------
# Groq is used here for fast, structured generation of
# the report title and section headings.
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Large model chosen because:
# • Task requires structural reasoning
# • Needs semantic alignment with research question
MODEL = "llama-3.3-70b-versatile"


def generate_title_and_headings(
    user_query: str,
    summaries: List[str],
    max_topics: int = 3,
) -> Dict[str, List[str]]:
    """
    TITLE & STRUCTURE GENERATOR
    ===========================

    Purpose
    -------
    This function generates the **structural skeleton** of the academic
    research report. It does NOT generate content — only:

    • One formal academic title
    • A fixed section structure

    This structure controls how the later report-writing LLM organizes
    the synthesis.

    Why this step exists
    --------------------
    Separating *structure generation* from *content generation* ensures:
    • The report stays aligned with the research question
    • No scope drift happens
    • Headings are academically formatted and consistent

    Inputs
    ------
    user_query : str
        Original research question from the user.
        This is the PRIMARY anchor for title creation.

    summaries : List[str]
        Final resolved summaries collected from sources.
        These provide topical signals but cannot expand scope.

    max_topics : int
        Number of topical body sections (default = 3).

    Output
    ------
    Dict:
    {
        "title": "Academic report title",
        "headings": [
            "Executive Summary",
            "Topic Heading 1",
            "Topic Heading 2",
            "Topic Heading 3",
            "Conclusion",
            "References"
        ]
    }

    Notes
    -----
    • Headings are used later by report.writer.py
    • JSON-only output is strictly enforced
    • Any malformed LLM output raises an error
    """

    # Convert summaries into bullet-style evidence block for LLM
    summary_block = "\n".join(f"- {s}" for s in summaries)

    # --------------------------------------------------
    # Prompt carefully engineered to prevent scope drift
    # --------------------------------------------------
    prompt = f"""
You are generating the structural outline of an academic research report.

================ CORE PRINCIPLE ================
- The research question is the PRIMARY semantic anchor.
- The title MUST be derived directly from the research question.
- The summaries are SECONDARY and may only help refine phrasing,
  emphasis, or academic tone.
- You MUST NOT introduce new scope, perspectives, or abstractions
  that are not implied by the research question.

================ TASK ================
1. Generate ONE formal academic report title.
2. Generate section headings.

================ TITLE RULES (ABSOLUTE) ================
- One sentence only
- Formal academic tone
- Declarative (not a question)
- Semantically close to the research question
- More polished and academic than the query,
  but NOT broader in meaning
- Do NOT copy the research question verbatim
- Do NOT introduce new themes or frames
- Do NOT include phrases like "A Study of" or "This Report"

================ HEADING RULES (ABSOLUTE) ================
- Include these sections EXACTLY once:
  Executive Summary
  Conclusion
  References
- Generate EXACTLY {max_topics} topical section headings
- Topic headings must:
  - Be derived from the summaries
  - Support the research question directly
  - Be academic and non-overlapping
  - Use Title Case
- No numbering
- No markdown

================ OUTPUT FORMAT (JSON ONLY) ================
{{
  "title": "<title>",
  "headings": [
    "Executive Summary",
    "<Topical Heading 1>",
    "<Topical Heading 2>",
    "...",
    "Conclusion",
    "References"
  ]
}}

================ RESEARCH QUESTION ================
{user_query}

================ SUMMARIES (SUPPORTING EVIDENCE) ================
{summary_block}
""".strip()

    # --------------------------------------------------
    # Call Groq LLM
    # --------------------------------------------------
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # low temperature → structured output
        max_tokens=300,
    )

    raw = r.choices[0].message.content.strip()

    # --------------------------------------------------
    # Strict JSON cleaning (LLMs sometimes add code fences)
    # --------------------------------------------------
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    data = json.loads(raw)

    # Safety validation
    if "title" not in data or "headings" not in data:
        raise ValueError("Invalid headings output")

    return data
