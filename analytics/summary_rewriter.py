import os
import json
from typing import Dict
from groq import Groq

# --------------------------------------------------
# LLM setup
# --------------------------------------------------
# This module uses a fast Groq-hosted LLM to rewrite summaries
# after conflict resolution. The goal is to REMOVE only specific
# contradictory claims while preserving all other information.

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


def rewrite_summaries(
    rewrite_plan: Dict[str, Dict[str, object]]
) -> Dict[str, str]:
    """
    SUMMARY REWRITING MODULE
    =========================

    Purpose
    -------
    After conflict detection + resolution, certain claims are marked
    for removal from specific summaries. This function asks an LLM to
    carefully rewrite those summaries without the flagged claims.

    The system does NOT delete text directly because:
    - Claims may be embedded in sentences
    - Context may need light rephrasing
    - Removing text blindly can break coherence

    So we use a controlled LLM rewrite.

    rewrite_plan format:
    {
        "S4": {
            "summary": "<original summary text>",
            "remove_claims": [
                "claim text 1",
                "claim text 2"
            ]
        },
        ...
    }

    Returns:
    {
        "S4": "<rewritten summary>",
        ...
    }
    """

    # If no summaries need rewriting, return empty result
    if not rewrite_plan:
        return {}

    # --------------------------------------------------
    # Build structured blocks for each summary
    # --------------------------------------------------
    # Each block includes:
    # - Summary ID
    # - Original summary text
    # - List of claims that must be removed

    blocks = []

    for sid, data in rewrite_plan.items():
        claims = "\n".join(f"- {c}" for c in data.get("remove_claims", []))

        blocks.append(
            f"""SUMMARY ID: {sid}

ORIGINAL SUMMARY:
{data["summary"]}

CLAIMS TO REMOVE:
{claims}"""
        )

    joined_blocks = "\n\n".join(blocks)

    # --------------------------------------------------
    # LLM Prompt Design
    # --------------------------------------------------
    # Carefully constrained to:
    # ✔ Remove only specified claims
    # ✔ Preserve all other information
    # ✔ Avoid adding or inventing facts
    # ✔ Avoid summarization or commentary
    # ✔ Maintain original tone and detail level

    prompt = f"""
You are editing research summaries.

TASK:
For EACH summary, rewrite it so that the listed claims are NO LONGER PRESENT.

IMPORTANT RULES:
- You MAY rewrite or rephrase sentences if needed to remove the ideas
- Keep all other content as close as possible to the original
- Preserve tone, scope, and level of detail
- Do NOT add new facts
- Do NOT summarize
- Do NOT explain your changes
- Do NOT invent information

OUTPUT FORMAT (JSON ONLY):
{{
  "rewritten": {{
    "S1": "rewritten summary text",
    "S2": "rewritten summary text"
  }}
}}

SUMMARIES:
{joined_blocks}
""".strip()

    # --------------------------------------------------
    # Run LLM rewrite
    # --------------------------------------------------

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,   # Deterministic editing
        max_tokens=2000,   # Enough space for multiple rewrites
    )

    content = response.choices[0].message.content.strip()

    # --------------------------------------------------
    # Parse strict JSON output
    # --------------------------------------------------
    # The model MUST return:
    # { "rewritten": { "S1": "...", "S2": "..." } }

    try:
        data = json.loads(content)
    except Exception as e:
        raise ValueError(
            f"Summary rewriter returned invalid JSON:\n{content}"
        ) from e

    rewritten = data.get("rewritten")
    if not isinstance(rewritten, dict):
        raise ValueError("Missing or invalid 'rewritten' field")

    return rewritten
