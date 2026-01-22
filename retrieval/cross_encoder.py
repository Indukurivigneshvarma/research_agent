# retrieval/cross_encoder.py

from typing import List, Dict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Cross-encoder for precise intent matching.

    Input:
      - query: str
      - candidates: List[Dict] with at least:
          { "vs_id": str, "query_text": str }

    Output:
      - same dicts, sorted by relevance (desc)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:

        if not candidates:
            return []

        # Build cross-encoder pairs
        pairs = [
            (query, c["query_text"])
            for c in candidates
        ]

        scores = self.model.predict(pairs)

        # Attach scores
        for c, score in zip(candidates, scores):
            c["score"] = float(score)

        # Sort by score desc
        ranked = sorted(
            candidates,
            key=lambda x: x["score"],
            reverse=True,
        )

        return ranked[:top_k]
