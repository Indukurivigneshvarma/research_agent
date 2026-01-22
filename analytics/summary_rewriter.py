import os
import json
from typing import Dict
from groq import Groq

# --------------------------------------------------
# LLM setup
# --------------------------------------------------

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


def rewrite_summaries(
    rewrite_plan: Dict[str, Dict[str, object]]
) -> Dict[str, str]:
    """
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

    if not rewrite_plan:
        return {}

    # --------------------------------------------------
    # Build input blocks safely (NO f-string backslashes)
    # --------------------------------------------------

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
    # Prompt (carefully designed — no hard deletion)
    # --------------------------------------------------

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
    # Run LLM
    # --------------------------------------------------

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000,
    )

    content = response.choices[0].message.content.strip()

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
