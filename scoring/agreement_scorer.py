"""
agreement_scorer.py
===================

Purpose
-------
Calculates how strongly each summary is supported by other summaries.

This is the **cross-source validation score** in the system.

While credibility scoring evaluates *who the source is*, agreement scoring
evaluates *how many other sources support it*.

This score is later combined with credibility to form:

    total_score = credibility_score + agreement_score

This directly affects:
- Which sources win during conflict resolution
- Which summaries are treated as stronger evidence
"""

from typing import Dict


# ------------------------------------------------------------------
# Agreement strength weights
# ------------------------------------------------------------------
# These labels come from the LLM agreement detector.
# Higher weight = stronger cross-source validation.
AGREEMENT_WEIGHTS = {
    "strongly_supports": 5,     # Clear, strong agreement
    "partially_supports": 3,    # Overlapping but not identical
    "independent": 1,           # Neutral or unrelated but not conflicting
}


# ------------------------------------------------------------------
# Main scoring function
# ------------------------------------------------------------------
def compute_agreement_scores(
    agreement_map: Dict[str, Dict[str, str]]
) -> Dict[str, int]:
    """
    Computes agreement scores based on *incoming support*.

    Input Format (agreement_map):
        {
            "S1": { "S2": "strongly_supports" },
            "S2": { "S1": "partially_supports" }
        }

    Interpretation:
        "S1": { "S2": "strongly_supports" }
        means:
            Summary S1 says S2 strongly supports it
        → So S2 receives +5

    Key Idea:
        A summary's agreement score is determined by how many other
        summaries support it, and how strongly.

    Output:
        {
            "S1": 3,
            "S2": 5
        }
    """

    scores: Dict[str, int] = {}

    # --------------------------------------------------
    # Initialize score entries for all IDs
    # --------------------------------------------------
    # Ensures every summary gets at least a 0 score.
    for src, relations in agreement_map.items():
        scores.setdefault(src, 0)
        for tgt in relations.keys():
            scores.setdefault(tgt, 0)

    # --------------------------------------------------
    # Accumulate incoming agreement weights
    # --------------------------------------------------
    for _, relations in agreement_map.items():
        for tgt, label in relations.items():
            scores[tgt] += AGREEMENT_WEIGHTS.get(label, 0)

    return scores
