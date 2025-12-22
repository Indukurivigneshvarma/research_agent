# agent/outline_generator.py
from llm.ollama_client import call_ollama


def generate_outline(query: str, knowledge: list[dict]) -> list[str]:
    """
    Generate a dynamic report outline based on
    early research evidence (not hardcoded sections).
    """

    # Extract textual content safely
    evidence_preview = "\n".join(
        k["content"] for k in knowledge[:5]
    )

    prompt = f"""
You are a research planning assistant.

RESEARCH QUESTION:
{query}

PARTIAL EVIDENCE FROM SOURCES:
{evidence_preview}

TASK:
Propose a logical outline for a research report.
- Choose section headings appropriate for the topic
- Avoid generic templates if unnecessary
- Return 4 to 7 clear section titles
- DO NOT include explanations, only headings

FORMAT:
One heading per line
"""

    response = call_ollama(prompt, temperature=0.2)

    outline = [
        line.strip("- ").strip()
        for line in response.split("\n")
        if line.strip()
    ]

    return outline
