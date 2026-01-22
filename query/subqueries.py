import os
from groq import Groq
from typing import List
from config import MODES

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "llama-3.1-8b-instant"


def generate_subqueries(
    user_query: str,
    mode: str,
) -> List[str]:
    """
    Generates grounded sub-queries based strictly on the user query.

    quick → 2
    standard → 4
    """

    n = MODES[mode]["subqueries"]

    prompt = f"""
You are decomposing a research question into focused sub-questions.

================ CORE PRINCIPLE ================
- The user query defines the FULL scope.
- Each sub-query must stay strictly within the
  semantic scope of the user query.
- Sub-queries must NOT generalize, broaden, or
  abstract beyond the user query.

================ TASK ================
Generate EXACTLY {n} sub-queries.

================ SUB-QUERY RULES (ABSOLUTE) ================
- Each sub-query must be a focused reformulation,
  angle, or analytical lens on the SAME problem.
- Do NOT introduce broader regions, populations,
  or domains than those implied in the user query.
- Do NOT generalize to related but wider contexts.
- Do NOT add comparative or cross-regional framing
  unless explicitly present in the user query.
- Each sub-query must still independently reflect
  the original research intent.

================ OUTPUT RULES ================
- One sub-query per line
- No numbering
- No explanations
- Academic phrasing
- Directly grounded in the user query

================ USER QUERY (SCOPE ANCHOR) ================
{user_query}
""".strip()

    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
    )

    lines = [
        l.strip()
        for l in r.choices[0].message.content.split("\n")
        if l.strip()
    ]

    return lines[:n]
