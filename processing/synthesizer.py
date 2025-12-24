# processing/synthesizer.py

from llm.ollama_client import call_ollama


def synthesize(query: str, article_summaries: list[str], research_mode: str) -> str:
    """
    Two-pass synthesis with:
    - Strict heading constraints
    - Controlled analytical expansion
    """

    evidence = "\n\n".join(article_summaries)

    # ======================================================
    # PASS 1 — Structured academic draft with strict headings
    # ======================================================
    base_prompt = f"""
You are writing a FORMAL ACADEMIC-STYLE RESEARCH PAPER.

RESEARCH QUESTION:
{query}

RESEARCH NOTES:
{evidence}

STRUCTURE REQUIREMENTS:
- Use 4–6 section headings (excluding Conclusion)
- EACH heading must be SHORT and CONCISE:
  • Maximum 5 words
  • Noun-phrase style (no full sentences)
  • Examples: "Job Market Shifts", "Skill Transformation"
- First section must be titled exactly: "Introduction"
  • High-level overview only
  • Minimal statistics
- Final section must be titled exactly: "Conclusion"

LANGUAGE & STYLE:
- Impersonal, formal research tone
- No first-person language
- Do NOT refer to the document itself
  (avoid “this report”, “this paper aims”, etc.)

CONTENT RULES:
- Use Markdown headings (## Heading)
- Paragraph-based writing only
- No references inside the text
- Do NOT invent statistics or sources

BEGIN DRAFT:
"""

    base_report = call_ollama(
        base_prompt,
        temperature=0.25,
        tier="large"
    )

    # ======================================================
    # PASS 2 — Mandatory analytical expansion
    # ======================================================
    expansion_prompt = f"""
You are expanding an academic research paper to journal length.

TASK:
- Expand EACH BODY section below (exclude Introduction and Conclusion)
- Do NOT add new sections
- Do NOT introduce new facts, numbers, or sources
- For EACH BODY section:
  • Write AT LEAST 3 long paragraphs
  • Reuse and restate existing quantitative evidence
  • Explicitly mention percentages, projections, or estimates already present
  • Analyze implications, trade-offs, and long-term effects
- Keep Introduction and Conclusion largely conceptual
- Preserve all headings exactly as given
- Maintain formal academic language
- Avoid repetition across sections
- Target length: AT LEAST 3 FULL A4 PAGES

PAPER TO EXPAND:
{base_report}

BEGIN EXPANDED PAPER:
"""

    expanded_report = call_ollama(
        expansion_prompt,
        temperature=0.2,
        tier="large"
    )

    return expanded_report
