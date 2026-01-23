# query/coverage_refiner.py

import os
import cohere
from typing import Dict, List

co = cohere.Client(os.getenv("COHERE_API_KEY"))

MODEL = "command-a-03-2025"


def refine_queries(
    research_plan: Dict[str, List[str]],
    summaries: Dict[str, str],
    n_queries: int = 2,
) -> List[str]:
    """
    Generates exactly `n_queries` new sub-queries to deepen coverage.

    Inputs:
    - research_plan: output of generate_research_plan
    - summaries: { "S1": "...", "S2": "..." }

    Output:
    - List[str] of new search queries
    """

    summary_block = "\n".join(
        f"{sid}: {text}"
        for sid, text in summaries.items()
    )

    plan_block = "\n".join(
        f"- {d}" for d in research_plan.get("dimensions", [])
    )

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

    # ---- Cohere Chat API (correct) ----
    r = co.chat(
     model=MODEL,
     message=prompt,
     temperature=0.4,
     max_tokens=200,
   )

    text = r.text


    lines = [
        l.strip()
        for l in text.split("\n")
        if l.strip()
    ]

    # ---- Hard guarantee ----
    if len(lines) < n_queries:
        raise ValueError("Coverage refiner returned too few queries")

    return lines[:n_queries]
