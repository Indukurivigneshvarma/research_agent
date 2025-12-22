# agent/evidence_scorer.py

from difflib import SequenceMatcher
from typing import List, Dict


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def score_evidence(
    summaries: List[str],
    similarity_threshold: float = 0.55
) -> List[Dict]:
    """
    Soft cross-source agreement scoring.
    """

    scored = []

    for i, summary in enumerate(summaries):
        support = 0
        for j, other in enumerate(summaries):
            if i != j and _similar(summary, other) >= similarity_threshold:
                support += 1

        confidence = min(1.0, 0.55 + 0.1 * support)

        scored.append({
            "text": summary,
            "confidence": round(confidence, 2),
            "support": support + 1
        })

    return scored
