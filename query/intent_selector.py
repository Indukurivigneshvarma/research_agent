import os
import json
from typing import Dict, List
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


# --------------------------------------------------
# JSON CLEANER (REQUIRED)
# --------------------------------------------------
def _clean_llm_json(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    # Remove ```json fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    # Remove leading 'json'
    if text.lower().startswith("json"):
        text = text[4:].strip()

    # Trim before first '{'
    first = text.find("{")
    if first != -1:
        text = text[first:]

    return text.strip()


def select_best_intents(
    subqueries: List[str],
    candidates_by_subquery: Dict[str, Dict[str, str]],
) -> Dict[str, str | None]:
    """
    subqueries:
        ["subquery text 1", "subquery text 2", ...]

    candidates_by_subquery:
        {
          "Q1": { "VS_01": "query text", "VS_02": "query text" },
          "Q2": { "VS_11": "query text", "VS_12": "query text" }
        }

    Returns:
        {
          "Q1": "VS_02",
          "Q2": null
        }
    """

    blocks = []

    for i, sq in enumerate(subqueries, 1):
        qkey = f"Q{i}"
        cands = candidates_by_subquery.get(qkey, {})

        cand_block = (
            "\n".join(f"{cid}: {ctext}" for cid, ctext in cands.items())
            if cands else
            "NONE"
        )

        blocks.append(
            f"""
SUB-QUERY {qkey}:
{sq}

CANDIDATE QUESTIONS:
{cand_block}
""".strip()
        )

    blocks_text = "\n\n".join(blocks)

    prompt = f"""
You are comparing research questions.

TASK:
For each sub-query, decide whether any candidate question
is essentially asking the SAME question.

IMPORTANT:
- Treat BOTH the sub-query and candidate as plain questions.
- Do NOT assume usefulness or relevance.
- Do NOT generalize or abstract.
- Do NOT match based on vague overlap.

DECISION RULE:
Select a candidate ONLY if a careful human reader would say:
“Yes — these two questions are basically asking the same thing.”

If they are not clearly the same, return null.

RULES:
- Select at most ONE ID per sub-query
- You MAY return null
- Do NOT reuse the same ID more than once
- Use ONLY the provided IDs
- Do NOT explain
- Return JSON ONLY

{blocks_text}

OUTPUT FORMAT:
{{
  "Q1": "VS_02",
  "Q2": null
}}
""".strip()

    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
    )

    raw = r.choices[0].message.content or ""
    cleaned = _clean_llm_json(raw)

    try:
        parsed = json.loads(cleaned)
    except Exception:
        # Fail-safe: select NONE for all sub-queries
        return {
            f"Q{i+1}": None
            for i in range(len(subqueries))
        }

    # Final safety: ensure all keys exist
    for i in range(len(subqueries)):
        parsed.setdefault(f"Q{i+1}", None)

    return parsed
