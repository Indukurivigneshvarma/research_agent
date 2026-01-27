from typing import Dict, List


def resolve_conflicts(
    conflicts: Dict,
    scores: Dict[str, int],
) -> Dict[str, List[str]]:
    """
    CONFLICT RESOLUTION MODULE
    ==========================

    Purpose
    -------
    Determines which factual claims must be removed when two summaries
    contain mutually contradictory information.

    This module is a **decision layer** after:
    1. Conflict detection (LLM identifies contradictions)
    2. Credibility + agreement scoring

    It uses summary strength scores to decide which side of a contradiction
    should be removed.

    Inputs
    ------
    conflicts : Dict
        Output of conflict detector. Example:
        {
          "conflicts": [
            {
              "ids": ["S1", "S2"],
              "claim_a": "...",
              "claim_b": "..."
            }
          ]
        }

    scores : Dict[str, int]
        Total strength score per summary.
        (credibility score + agreement score)

    Resolution Logic
    ----------------
    • Higher score summary = more reliable
    • Lower score summary loses its conflicting claim
    • Equal scores → no action (system avoids arbitrary decisions)

    Returns
    -------
    Dict[str, List[str]]

    Format:
    {
      "S4": ["claim text 1", "claim text 2"],
      "S2": ["claim text 3"]
    }

    Meaning:
    These specific claims must be removed from those summaries.
    Downstream module (summary_rewriter) performs the actual rewriting.
    """

    removals: Dict[str, List[str]] = {}

    # Iterate over each detected conflict pair
    for conflict in conflicts.get("conflicts", []):

        # Extract summary IDs and conflicting claims
        ids = conflict["ids"]
        claim_a = conflict["claim_a"]
        claim_b = conflict["claim_b"]

        # Safety: must be exactly 2 summaries in a conflict
        if len(ids) != 2:
            continue

        s1, s2 = ids

        # Fetch reliability scores (default = 0 if missing)
        score1 = scores.get(s1, 0)
        score2 = scores.get(s2, 0)

        # --------------------------------------------------
        # Tie → system avoids arbitrary deletion
        # --------------------------------------------------
        if score1 == score2:
            continue

        # --------------------------------------------------
        # Lower-scoring summary loses the conflicting claim
        # --------------------------------------------------
        if score1 > score2:
            loser, claim = s2, claim_b
        else:
            loser, claim = s1, claim_a

        # Record which claim must be removed
        removals.setdefault(loser, []).append(claim)

    return removals
