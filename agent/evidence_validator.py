# agent/evidence_validator.py
from difflib import SequenceMatcher
from typing import List, Tuple


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def validate_evidence(
    summaries: List[str],
    similarity_threshold: float = 0.65,
    min_support: int = 2
) -> Tuple[List[str], List[str]]:
    """
    Perform redundancy-based validation within an iteration.

    Returns:
    - validated_summaries: supported by multiple sources
    - disputed_summaries: weak or conflicting evidence
    """

    clusters = []

    for summary in summaries:
        placed = False
        for cluster in clusters:
            if _similar(summary, cluster[0]) >= similarity_threshold:
                cluster.append(summary)
                placed = True
                break
        if not placed:
            clusters.append([summary])

    validated = []
    disputed = []

    for cluster in clusters:
        if len(cluster) >= min_support:
            # retain one representative summary
            validated.append(cluster[0])
        else:
            disputed.extend(cluster)

    return validated, disputed
