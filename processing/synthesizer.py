# processing/synthesizer.py

from llm.ollama_client import call_ollama


def synthesize(query: str, article_summaries: list[str], research_mode: str) -> str:
    """
    Final analytical synthesis.
    Produces a long, structured, analytical research report.
    """

    evidence = "\n\n".join(article_summaries)

    prompt = f"""
You are writing a FULL-LENGTH ANALYTICAL RESEARCH REPORT.

RESEARCH QUESTION:
{query}

RESEARCH MODE:
{research_mode}

CONSOLIDATED RESEARCH NOTES:
{evidence}

ANALYTICAL REQUIREMENTS:
- Decide your OWN section headings (4–6 sections)
- Each section must contain MULTIPLE PARAGRAPHS
- Each section should include at least ONE analytical insight, such as:
  • quantitative impact (percentages, productivity change, job impact)
  • attribution to organizations (e.g., Gartner, McKinsey, Microsoft, academic studies)
- Prefer explicit attribution like “According to X (year)…”
- Preserve numbers, percentages, years, and organization names when present
- Analyze implications, trade-offs, and long-term effects
- Do NOT repeat the same idea across sections
- Do NOT invent statistics or sources
- Do NOT include references inside the content
- Length MUST be sufficient to fill AT LEAST 3 FULL A4 PAGES

FORMAT:
- Use Markdown headings (## Section Title)
- Paragraph-based writing only (no bullet points)

BEGIN REPORT:
"""

    return call_ollama(prompt, temperature=0.25, tier="large")
