# analytics/agreement_detector.py

"""
CROSS-SOURCE AGREEMENT DETECTOR
===============================

Purpose:
--------
Uses an LLM to detect how strongly research summaries
support one another.

This enables:
• Evidence triangulation
• Reliability boosting via consensus
• Agreement-based scoring
• Conflict-aware synthesis

This module DOES NOT:
✗ Judge credibility
✗ Detect contradictions
✗ Rewrite content

It ONLY detects SUPPORT relationships.

Model Used:
-----------
Gemini Flash (fast reasoning for structured analysis)
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
# Allowed agreement labels
# --------------------------------------------------

ALLOWED_LABELS = {
    "strongly_supports",
    "partially_supports",
    "independent",
}


# --------------------------------------------------
# JSON cleaner (LLM output safety)
# --------------------------------------------------

def _clean_llm_json(text: str) -> str:
    """
    Cleans LLM output to ensure it can be parsed as JSON.

    Handles:
    • Markdown fences (```json)
    • Leading 'json' labels
    • Extra commentary before JSON

    This is required because LLMs often add formatting.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove ```json fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    # Remove leading 'json' word
    if text.lower().startswith("json"):
        text = text[4:].strip()

    # Keep only content starting from first brace
    first_brace = text.find("{")
    if first_brace != -1:
        text = text[first_brace:]

    return text.strip()


# --------------------------------------------------
# Agreement detection
# --------------------------------------------------

def detect_agreements(
    summaries: List[Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    """
    Determines support relationships between summaries.

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
    Dict[str, Dict[str, str]]

    Example:
    {
      "S1": {"S2": "strongly_supports"},
      "S2": {"S1": "partially_supports"}
    }

    Properties
    ----------
    • Directional (A→B can differ from B→A)
    • Only allowed labels are returned
    • Fully validated before returning
    """

    # Not enough sources for agreement analysis
    if len(summaries) < 2:
        return {}

    # Combine summaries into a single block for LLM analysis
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

    # --------------------------------------------------
    # Validation (critical for system integrity)
    # --------------------------------------------------

    ids = {s["id"] for s in summaries}

    for src, relations in data.items():

        # Source ID must be valid
        if src not in ids:
            raise ValueError(f"Unknown source ID: {src}")

        if not isinstance(relations, dict):
            raise ValueError(f"Relations for {src} must be a dict")

        for tgt, label in relations.items():

            # Target ID must be valid
            if tgt not in ids:
                raise ValueError(f"Unknown target ID: {tgt}")

            # No self-relations allowed
            if src == tgt:
                raise ValueError("Self-relations are not allowed")

            # Only allowed labels
            if label not in ALLOWED_LABELS:
                raise ValueError(f"Invalid agreement label: {label}")

    return data
