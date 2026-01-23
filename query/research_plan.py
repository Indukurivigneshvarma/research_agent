# query/research_plan.py

import os
import json
import cohere
from typing import Dict, List

co = cohere.Client(os.getenv("COHERE_API_KEY"))

MODEL = "command-a-03-2025"


def generate_research_plan(user_query: str) -> Dict[str, List[str]]:
    """
    Generates a conceptual research plan (goal + dimensions).
    """

    prompt = f"""
You are a research planning assistant.

TASK:
Decompose the research question into high-level conceptual dimensions
that must be covered to answer it thoroughly.

RULES:
- Dimensions are NOT search queries
- Dimensions represent themes, angles, or aspects
- Be domain-agnostic
- Avoid redundancy or overlap
- Produce 4–6 dimensions
- Do NOT explain anything

OUTPUT FORMAT (JSON ONLY):
{{
  "goal": "<rephrased research goal>",
  "dimensions": [
    "Dimension 1",
    "Dimension 2"
  ]
}}

RESEARCH QUESTION:
{user_query}
""".strip()

    # ---- Cohere Chat API (SDK-compatible) ----
    r = co.chat(
        model=MODEL,
        message=prompt,
        temperature=0.3,
        max_tokens=400,
    )

    raw = r.text.strip()

    # ---- strict JSON cleaning ----
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    data = json.loads(raw)

    if "goal" not in data or "dimensions" not in data:
        raise ValueError("Invalid research plan output")

    return data
