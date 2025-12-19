from llm.ollama_client import call_ollama


def synthesize(query: str, article_summaries: list) -> str:
    combined_summaries = "\n\n".join(article_summaries)

    prompt = f"""
You are writing a MINI RESEARCH REPORT.

RESEARCH QUESTION:
{query}

SUMMARIZED EVIDENCE FROM MULTIPLE SOURCES:
{combined_summaries}

STRICT INSTRUCTIONS:
- Use ONLY the summarized evidence
- Do NOT invent facts or statistics
- Be analytical, not repetitive

OUTPUT EXACTLY THESE SECTIONS:

## Introduction
## Key Findings
## Analysis
## Limitations
## Conclusion
"""

    return call_ollama(prompt, temperature=0.2)
