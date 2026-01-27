"""
cross_encoder.py
────────────────────────────────────────────────────────────
Purpose:
    Re-ranks vector search results using a cross-encoder model
    for high-precision semantic matching.

Why This Exists:
    Vector search (FAISS) is FAST but approximate.
    It compares embeddings independently.

    Cross-encoder is SLOWER but MUCH MORE PRECISE.
    It looks at BOTH query and candidate text TOGETHER
    and predicts true semantic relevance.

Pipeline Role:
    Step 1 → Vector search finds broad candidates
    Step 2 → Cross-encoder selects the BEST true matches

This reduces:
    • False positives from vector similarity
    • Intent mismatches
    • Wrong summary reuse
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Uses a cross-encoder model for fine-grained relevance scoring.

    Unlike bi-encoder embeddings, this model processes
    (query, candidate) pairs jointly, leading to better
    semantic alignment decisions.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Loads the cross-encoder model once at initialization.

        Model:
            ms-marco-MiniLM cross-encoder
            - Lightweight
            - Fast enough for reranking
            - Trained for query-document relevance
        """
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Re-ranks vector search candidates.

        Inputs:
            query:
                The current sub-query (user intent)

            candidates:
                List of records containing at least:
                {
                    "vs_id": ID from vector search,
                    "query_text": stored query intent text
                }

            top_k:
                Number of top-scoring candidates to keep

        Returns:
            Same candidate dictionaries with an added "score" field,
            sorted by descending relevance.
        """

        if not candidates:
            return []

        # --------------------------------------------------
        # Build (query, candidate) pairs for cross-encoder
        # --------------------------------------------------
        pairs = [
            (query, c["query_text"])
            for c in candidates
        ]

        # Model predicts semantic relevance scores
        scores = self.model.predict(pairs)

        # --------------------------------------------------
        # Attach scores back to candidate records
        # --------------------------------------------------
        for c, score in zip(candidates, scores):
            c["score"] = float(score)

        # Sort candidates by descending score
        ranked = sorted(
            candidates,
            key=lambda x: x["score"],
            reverse=True,
        )

        return ranked[:top_k]
