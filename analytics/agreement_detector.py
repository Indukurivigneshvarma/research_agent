import os
import json
from typing import List, Dict
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL = "gemini-flash-latest"

ALLOWED_LABELS = {
    "strongly_supports",
    "partially_supports",
    "independent",
}


# ------------------------------
# JSON CLEANER (REQUIRED)
# ------------------------------
def _clean_llm_json(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    # Remove ```json / ``` fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    # Remove leading "json" label if present
    if text.lower().startswith("json"):
        text = text[4:].strip()

    # Final safety: trim until first '{'
    first_brace = text.find("{")
    if first_brace != -1:
        text = text[first_brace:]

    return text.strip()



def detect_agreements(
    summaries: List[Dict[str, str]],
) -> Dict[str, Dict[str, str]]:

    if len(summaries) < 2:
        return {}

    block = "\n\n".join(
        f"{s['id']}:\n{s['summary']}"
        for s in summaries
    )

    prompt = f"""
Analyze cross-source agreement.

Allowed labels ONLY:
- strongly_supports
- partially_supports
- independent

Rules:
- Compare EACH summary against EVERY OTHER summary
- Directional (A→B may differ from B→A)
- NO explanations
- Return JSON ONLY

SUMMARIES:
{block}

OUTPUT FORMAT:
{{
  "S1": {{ "S2": "strongly_supports" }}
}}
""".strip()

    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(prompt)

    cleaned = _clean_llm_json(response.text)

    try:
        data = json.loads(cleaned)
    except Exception as e:
        raise ValueError(
            f"Agreement LLM returned invalid JSON:\n{cleaned}"
        ) from e

    # ------------------------------
    # Validation
    # ------------------------------
    ids = {s["id"] for s in summaries}

    for src, relations in data.items():
        if src not in ids:
            raise ValueError(f"Unknown source ID: {src}")

        if not isinstance(relations, dict):
            raise ValueError(f"Relations for {src} must be a dict")

        for tgt, label in relations.items():
            if tgt not in ids:
                raise ValueError(f"Unknown target ID: {tgt}")
            if src == tgt:
                raise ValueError("Self-relations are not allowed")
            if label not in ALLOWED_LABELS:
                raise ValueError(f"Invalid agreement label: {label}")

    return data
