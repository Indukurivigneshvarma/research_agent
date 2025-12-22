# processing/iteration_summarizer.py

from llm.ollama_client import call_ollama

MIN_CONFIDENCE = 0.4   # 🔥 LOWERED ON PURPOSE


def summarize_iteration(evidence_items: list[dict]) -> str:
    """
    Generate rich iteration research notes.
    """

    usable = [
        item["text"]
        for item in evidence_items
        if item.get("confidence", 0.0) >= MIN_CONFIDENCE
    ]

    if len(usable) < 2:
        return ""

    joined = "\n\n".join(usable[:10])

    prompt = f"""
You are writing INTERNAL RESEARCH NOTES.

TASK:
- Expand all provided evidence into detailed explanations
- Preserve all useful insights, even minority viewpoints
- Explain impacts, mechanisms, examples, and implications
- **When quantitative data, percentages, years, or organizations appear,
  preserve them explicitly (e.g., “According to Gartner (2023), ~30%…”).**
- Avoid generic introductions
- Write in paragraphs (NO bullet points)
- Minimum length: 700 words

EVIDENCE:
{joined}
"""

    return call_ollama(prompt, temperature=0.25)
