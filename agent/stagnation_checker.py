# agent/stagnation_checker.py
from difflib import SequenceMatcher
from typing import List


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def is_stagnating(
    recent_queries: List[str],
    similarity_threshold: float = 0.85,
    window: int = 3
) -> bool:
    """
    Detect planner stagnation when recent search queries
    are highly similar to each other.

    Args:
        recent_queries: list of executed search queries
        similarity_threshold: similarity score to consider queries redundant
        window: number of recent queries to compare

    Returns:
        True if stagnation detected, else False
    """

    if len(recent_queries) < window:
        return False

    last_queries = recent_queries[-window:]

    similarities = []
    for i in range(len(last_queries) - 1):
        sim = _similar(last_queries[i], last_queries[i + 1])
        similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities)

    return avg_similarity >= similarity_threshold
