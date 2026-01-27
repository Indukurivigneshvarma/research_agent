# analytics/conflict_detector.py

"""
FACTUAL CONFLICT DETECTOR
=========================

Purpose
-------
Detects HARD factual contradictions between research summaries.

This module ensures:
• Logical consistency across sources
• Prevention of mutually exclusive claims entering the final report
• Evidence reliability through conflict resolution

This module DOES NOT:
✗ Decide which source is correct
✗ Rewrite summaries
✗ Judge credibility

It ONLY identifies contradictions for downstream resolution.

Model Used
----------
Gemini Flash — fast structured reasoning over paired claims.
"""

import os
import json
from typing import List, Dict
import google.generativeai as genai


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "gemini-flash-latest"


# --------------------------------------------------
# Conflict detection
# --------------------------------------------------

def detect_conflicts(
    summaries: List[Dict[str, str]],
) -> Dict:
    """
    Detects HARD factual contradictions between summaries.

    Parameters
    ----------
    summaries : List[Dict]
        Format:
        [
          {"id": "S1", "summary": "..."},
          {"id": "S2", "summary": "..."}
        ]

    Returns
    -------
    Dict

    Example:
    {
      "conflicts": [
        {
          "ids": ["S1", "S2"],
          "claim_a": "...",
          "claim_b": "..."
        }
      ]
    }

    Conflict Definition (STRICT)
    ----------------------------
    A contradiction exists ONLY if:
    • Both claims refer to the same phenomenon/variable
    • They cannot logically both be true in the same world
    """

    # Not enough summaries for comparison
    if len(summaries) < 2:
        return {"conflicts": []}

    # Combine summaries into one block for LLM analysis
    block = "\n\n".join(
        f"{s['id']}:\n{s['summary']}"
        for s in summaries
    )

    prompt = f"""
You are detecting HARD FACTUAL CONTRADICTIONS between research summaries.

DEFINITION (STRICT):
A factual contradiction exists ONLY if both claims refer to the same
phenomenon or variable AND they cannot logically be true at the same time
in the same real world.

CRITICAL TEST (MANDATORY):
Before marking a conflict, explicitly apply this test internally:
"If both claims were true simultaneously, would this create a logical
impossibility?"

If the answer is NO → DO NOT mark a conflict.

TASK:
Compare EACH summary against EVERY OTHER summary.
Identify ONLY hard contradictions that pass the above test.

DO NOT mark conflicts for:
- Different examples, lists, or enumerations
- Partial overlap of factors, causes, effects, or benefits
- Differences in emphasis, framing, categorization, or prioritization
- Additive or complementary claims
- Missing information in one summary
- Different scopes or levels of detail
- Descriptive vs analytical differences

MARK a conflict ONLY for:
- Direct numeric oppositions
- Mutually exclusive states or requirements
- Opposite trends or outcomes
- Explicit denial of the same factual claim

RULES:
- Output at most ONE conflict per summary pair
- Extract the exact conflicting claims verbatim
- Use ONLY information explicitly stated
- Do NOT infer, generalize, or invent facts
- Output JSON ONLY

OUTPUT FORMAT:
{{
  "conflicts": [
    {{
      "ids": ["S1", "S2"],
      "claim_a": "explicit factual claim from S1",
      "claim_b": "explicit factual claim from S2"
    }}
  ]
}}

If no hard contradictions exist:
{{ "conflicts": [] }}

SUMMARIES:
{block}
""".strip()

    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(prompt)

    raw = response.text.strip()

    # --------------------------------------------------
    # Clean markdown fences if LLM added them
    # --------------------------------------------------

    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    # --------------------------------------------------
    # Parse JSON safely
    # --------------------------------------------------

    try:
        return json.loads(raw)
    except Exception as e:
        raise ValueError(
            f"Conflict detector returned invalid JSON:\n{raw}"
        ) from e
