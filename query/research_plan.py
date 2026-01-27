"""
research_plan.py
────────────────────────────────────────────────────────────
Purpose:
    This module generates the *conceptual research plan* for a user’s query.

    It is the FIRST intelligence step in the pipeline. Before searching,
    the system must understand:
        • What is the real research goal?
        • What major dimensions must be covered?

    Instead of jumping directly to web search, we use an LLM to
    decompose the research question into structured research axes.

Why this matters:
    Without this step, the system would perform shallow search and
    miss key perspectives. The research plan acts as the
    "coverage blueprint" for the entire pipeline.
"""

import os
import json
import cohere
from typing import Dict, List


# --------------------------------------------------
# LLM Client Setup (Cohere)
# --------------------------------------------------
# We use Cohere's command model because it performs
# structured reasoning and decomposition tasks well.

co = cohere.Client(os.getenv("COHERE_API_KEY"))

# High-reasoning planning model
MODEL = "command-a-03-2025"


# --------------------------------------------------
# Research Plan Generator
# --------------------------------------------------
def generate_research_plan(user_query: str) -> Dict[str, List[str]]:
    """
    Uses an LLM to transform a raw research question into a
    structured research plan.

    Input:
        user_query (str)
            The original question provided by the user.

    Output:
        Dict with keys:
            "goal"        → Rephrased core research objective
            "dimensions"  → List of 4–6 conceptual research axes

    Example Output:
        {
          "goal": "Assess the impact of generative AI on software engineering",
          "dimensions": [
              "Job Role Transformation",
              "Skillset Evolution",
              "Productivity Implications",
              "Ethical Considerations"
          ]
        }

    Notes:
        • Dimensions are NOT search queries
        • They represent thematic coverage requirements
        • Later modules use these to guide query generation
    """

    # --------------------------------------------------
    # Prompt instructs the LLM to perform structured decomposition
    # --------------------------------------------------
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

    # --------------------------------------------------
    # LLM Call — Planning Stage
    # --------------------------------------------------
    r = co.chat(
        model=MODEL,
        message=prompt,
        temperature=0.3,   # slight creativity, but mostly structured
        max_tokens=400,
    )

    raw = r.text.strip()

    # --------------------------------------------------
    # JSON Cleaning (LLMs often return fenced blocks)
    # --------------------------------------------------
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    data = json.loads(raw)

    # --------------------------------------------------
    # Safety Check
    # Ensures LLM followed format before pipeline proceeds
    # --------------------------------------------------
    if "goal" not in data or "dimensions" not in data:
        raise ValueError("Invalid research plan output")

    return data
