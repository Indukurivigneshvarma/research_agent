"""
subqueries.py
────────────────────────────────────────────────────────────
Purpose:
    This module generates the FIRST set of search queries used
    in the research pipeline.

    It converts the abstract research plan into concrete
    search engine queries.

Pipeline Role:
    Research Plan  →  Initial Sub-Queries  →  Retrieval

Key Idea:
    We DO NOT search using the raw user question.
    Instead, we generate structured, coverage-aware sub-queries
    so the search step is deliberate rather than naive.

This step happens ONLY during iteration 1.
Later iterations use coverage_refiner.py instead.
"""

import os
from typing import List
from groq import Groq


# --------------------------------------------------
# LLM Client Setup (Groq)
# --------------------------------------------------
# Using a fast 8B model because this task is:
# • Structured
# • Low hallucination risk
# • Not deep reasoning

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"


# --------------------------------------------------
# Initial Sub-Query Generator
# --------------------------------------------------
def generate_initial_subqueries(
    user_query: str,
    research_goal: str,
    dimensions: List[str],
    n_queries: int = 2,
) -> List[str]:
    """
    Generates the first set of search queries for the pipeline.

    Inputs:
        user_query (str)
            The original research question.

        research_goal (str)
            Rephrased research objective produced by research_plan.py.

        dimensions (List[str])
            Conceptual coverage axes that must be addressed.

        n_queries (int)
            Number of search queries to generate (typically 2).

    Output:
        List[str] → Exactly n_queries search-ready queries.

    Important Constraints:
        • Queries must jointly cover multiple dimensions
        • We DO NOT want one query per dimension
        • Queries must be broad but precise
        • No explanations — raw search strings only
    """

    # Convert dimension list into readable block for LLM context
    dim_block = "\n".join(f"- {d}" for d in dimensions)

    # --------------------------------------------------
    # Prompt instructs the LLM to perform *coverage-aware*
    # query design rather than naive keyword generation.
    # --------------------------------------------------
    prompt = f"""
You are generating INITIAL SEARCH QUERIES for academic research.

TASK:
Generate EXACTLY {n_queries} search queries that together provide
broad initial coverage of the research goal.

RULES (MANDATORY):
- Queries MUST be grounded in the research goal and user query
- Queries MUST collectively cover multiple dimensions
- Do NOT generate one query per dimension
- Do NOT introduce new scope, locations, or populations
- Queries must be suitable for academic or web search
- One query per line
- No numbering
- No explanations

RESEARCH GOAL:
{research_goal}

USER QUERY:
{user_query}

RESEARCH DIMENSIONS:
{dim_block}

OUTPUT:
Exactly {n_queries} lines, each a search query.
""".strip()

    # --------------------------------------------------
    # LLM Call — Query Design Stage
    # --------------------------------------------------
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )

    text = response.choices[0].message.content.strip()

    # --------------------------------------------------
    # Parse LLM output into clean query list
    # --------------------------------------------------
    queries = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    # Safety guard — ensures pipeline stability
    if len(queries) < n_queries:
        raise ValueError(
            f"Expected {n_queries} initial sub-queries, got {len(queries)}"
        )

    return queries[:n_queries]
