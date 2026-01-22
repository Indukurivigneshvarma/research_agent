from typing import Dict, List


def resolve_conflicts(
    conflicts: Dict,
    scores: Dict[str, int],
) -> Dict[str, List[str]]:
    """
    Determines which claims must be removed from which summaries.

    Returns:
    {
      "S4": ["claim text 1", "claim text 2"],
      "S2": ["claim text 3"]
    }
    """

    removals: Dict[str, List[str]] = {}

    for conflict in conflicts.get("conflicts", []):
        ids = conflict["ids"]
        claim_a = conflict["claim_a"]
        claim_b = conflict["claim_b"]

        if len(ids) != 2:
            continue

        s1, s2 = ids
        score1 = scores.get(s1, 0)
        score2 = scores.get(s2, 0)

        # Tie → no resolution
        if score1 == score2:
            continue

        if score1 > score2:
            loser, claim = s2, claim_b
        else:
            loser, claim = s1, claim_a

        removals.setdefault(loser, []).append(claim)

    return removals
