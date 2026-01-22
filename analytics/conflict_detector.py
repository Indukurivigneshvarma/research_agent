import os
import json
from typing import List, Dict
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL = "gemini-flash-latest"


def detect_conflicts(
    summaries: List[Dict[str, str]],
) -> Dict:
    """
    Detects pairwise factual contradictions between summaries.
    LLM only EXTRACTS conflicts, does NOT resolve them.
    """

    if len(summaries) < 2:
        return {"conflicts": []}

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

    # Remove markdown fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    try:
        return json.loads(raw)
    except Exception as e:
        raise ValueError(
            f"Conflict detector returned invalid JSON:\n{raw}"
        ) from e
