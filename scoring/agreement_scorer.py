from typing import Dict

# --------------------------------------------------
# Agreement weights
# --------------------------------------------------

AGREEMENT_WEIGHTS = {
    "strongly_supports": 5,
    "partially_supports": 3,
    "independent": 1,
}


def compute_agreement_scores(
    agreement_map: Dict[str, Dict[str, str]]
) -> Dict[str, int]:
    """
    Computes agreement scores from incoming support.
    """

    scores: Dict[str, int] = {}

    # Initialize all IDs (sources and targets)
    for src, relations in agreement_map.items():
        scores.setdefault(src, 0)
        for tgt in relations.keys():
            scores.setdefault(tgt, 0)

    # Accumulate incoming agreement
    for _, relations in agreement_map.items():
        for tgt, label in relations.items():
            scores[tgt] += AGREEMENT_WEIGHTS.get(label, 0)

    return scores
