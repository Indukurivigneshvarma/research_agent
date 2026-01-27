"""
coverage_refiner.py
────────────────────────────────────────────────────────────
Purpose:
    This module generates NEW search queries after the first
    discovery round.

    It analyzes:
        • The intended research plan
        • The summaries already collected

    And decides what knowledge is STILL missing.

Pipeline Role:
    Iteration 2+ ONLY
    (Iteration 1 uses subqueries.py instead)

This is what makes the system **iterative and intelligent**
instead of a one-shot search engine.
"""

import os
import cohere
from typing import Dict, List


# --------------------------------------------------
# LLM Client Setup (Cohere)
# --------------------------------------------------
# Using a strong instruction model here because this task requires:
# • Gap analysis
# • Coverage reasoning
# • Strategic query refinement

co = cohere.Client(os.getenv("COHERE_API_KEY"))

MODEL = "command-a-03-2025"


# --------------------------------------------------
# Coverage Refinement Query Generator
# --------------------------------------------------
def refine_queries(
    research_plan: Dict[str, List[str]],
    summaries: Dict[str, str],
    n_queries: int = 2,
) -> List[str]:
    """
    Generates new search queries to improve coverage.

    Inputs:
        research_plan (dict)
            Output of generate_research_plan():
            {
              "goal": "...",
              "dimensions": [...]
            }

        summaries (dict)
            Current knowledge state:
            { "S1": "...", "S2": "...", ... }

        n_queries (int)
            Number of new queries to generate.

    Output:
        List[str] → Exactly n_queries search-ready queries.

    What this step does:
        • Detects missing angles
        • Detects weak evidence areas
        • Avoids redundancy
        • Deepens research coverage
    """

    # --------------------------------------------------
    # Build context blocks for the LLM
    # --------------------------------------------------

    # Existing knowledge
    summary_block = "\n".join(
        f"{sid}: {text}"
        for sid, text in summaries.items()
    )

    # Intended coverage axes
    plan_block = "\n".join(
        f"- {d}" for d in research_plan.get("dimensions", [])
    )

    # --------------------------------------------------
    # Prompt instructs LLM to act as a research strategist
    # performing gap analysis rather than random expansion.
    # --------------------------------------------------
    prompt = f"""
You are refining a research process.

TASK:
Given the research plan and summaries collected so far,
generate EXACTLY {n_queries} NEW search sub-queries that would
most improve coverage, depth, or clarity.

RULES:
- Queries must be suitable for academic web search
- Queries must target missing, weak, or underdeveloped dimensions
- Do NOT repeat existing summaries
- Do NOT explain
- One query per line
- No numbering
- Broad but precise

RESEARCH GOAL:
{research_plan.get("goal")}

RESEARCH DIMENSIONS:
{plan_block}

CURRENT SUMMARIES:
{summary_block}

OUTPUT:
Exactly {n_queries} lines, each a search query.
""".strip()

    # --------------------------------------------------
    # LLM Call — Coverage Gap Reasoning Stage
    # --------------------------------------------------
    r = co.chat(
        model=MODEL,
        message=prompt,
        temperature=0.4,
        max_tokens=200,
    )

    text = r.text

    # --------------------------------------------------
    # Parse LLM output into clean query list
    # --------------------------------------------------
    lines = [
        l.strip()
        for l in text.split("\n")
        if l.strip()
    ]

    # Hard safety guarantee
    if len(lines) < n_queries:
        raise ValueError("Coverage refiner returned too few queries")

    return lines[:n_queries]
