# query/subqueries.py

import os
from typing import List
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"


def generate_initial_subqueries(
    user_query: str,
    research_goal: str,
    dimensions: List[str],
    n_queries: int = 2,
) -> List[str]:
    """
    Generates EXACTLY `n_queries` initial search sub-queries
    grounded in the research plan dimensions.

    Used ONLY for iteration 1.
    """

    dim_block = "\n".join(f"- {d}" for d in dimensions)

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

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )

    text = response.choices[0].message.content.strip()

    queries = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    if len(queries) < n_queries:
        raise ValueError(
            f"Expected {n_queries} initial sub-queries, got {len(queries)}"
        )

    return queries[:n_queries]
