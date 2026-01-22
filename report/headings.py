import os
import json
from typing import List, Dict
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


def generate_title_and_headings(
    user_query: str,
    summaries: List[str],
    max_topics: int = 3,
) -> Dict[str, List[str]]:
    """
    Generates:
    - ONE academic report title
    - A fixed heading structure:
        Executive Summary
        2–3 topic headings
        Conclusion
        References
    """

    summary_block = "\n".join(f"- {s}" for s in summaries)

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

    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )

    raw = r.choices[0].message.content.strip()

    # ---- strict JSON cleaning ----
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    data = json.loads(raw)

    if "title" not in data or "headings" not in data:
        raise ValueError("Invalid headings output")

    return data
