# agent/sufficiency_checker.py

from difflib import SequenceMatcher
from typing import Tuple, Dict


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def check_sufficiency(
    prev_summary: str,
    curr_summary: str,
    min_similarity: float = 0.85,
    min_new_words: int = 120
) -> Tuple[bool, Dict]:
    """
    Decide whether research has reached sufficiency using:
    1. Knowledge stability (summary similarity)
    2. Diminishing information gain (new words)
    """

    if not prev_summary or not curr_summary:
        return False, {
            "reason": "Insufficient history",
            "similarity": 0.0,
            "new_words": 999
        }

    similarity = _similar(prev_summary, curr_summary)

    prev_words = set(prev_summary.split())
    curr_words = set(curr_summary.split())
    new_words = len(curr_words - prev_words)

    sufficient = (
        similarity >= min_similarity
        and new_words <= min_new_words
    )

    return sufficient, {
        "similarity": round(similarity, 2),
        "new_words": new_words,
        "reason": (
            "Information stabilized"
            if sufficient
            else "New information still emerging"
        )
    }
