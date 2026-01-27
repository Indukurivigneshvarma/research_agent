"""
vector_search.py
────────────────────────────────────────────────────────────
Purpose:
    Performs semantic search over previously stored research
    summaries using vector similarity.

Pipeline Role:
    This module enables KNOWLEDGE REUSE.

    Instead of searching the web again, the system first checks:
    "Have we already answered a similar question before?"

    If similar past research exists, we reuse it and avoid:
      • Extra API calls
      • Duplicate sources
      • Redundant summaries
      • Increased cost

Key Concept:
    We do NOT store raw documents in the vector DB.
    We store:
        - Query text
        - Its summary
        - Metadata

    So we are matching "research intents", not full articles.
"""

from typing import List, Dict


class VectorSearcher:
    """
    Thin abstraction over the VectorStoreClient.

    This layer keeps retrieval logic separate from the
    underlying FAISS storage implementation.
    """

    def __init__(self, vector_client):
        """
        vector_client:
            Instance of VectorStoreClient (FAISS wrapper).
        """
        self.client = vector_client

    def search(
        self,
        query_embedding,
        top_k: int,
    ) -> List[Dict]:
        """
        Performs vector similarity search.

        Input:
            query_embedding:
                Embedding of the new sub-query (semantic intent).

            top_k:
                Number of nearest stored records to retrieve.

        Returns:
            List of candidate summary records that might match
            the user's current research question.

            Each record contains:
                - query_text       (original stored intent)
                - summary          (its summary)
                - embedding        (stored vector)
                - url              (source URL)
                - domain
                - author
                - venue_type
                - date_published
                - date_retrieved
        """

        # Query FAISS index via VectorStoreClient
        results = self.client.search(
            embedding=query_embedding,
            top_k=top_k,
        )

        # Normalize output structure
        hits = []
        for r in results:
            hits.append({
                "query_text": r["query_text"],
                "summary": r["summary"],
                "embedding": r["embedding"],
                "url": r.get("url"),
                "domain": r.get("domain"),
                "author": r.get("author"),
                "venue_type": r.get("venue_type"),
                "date_published": r.get("date_published"),
                "date_retrieved": r.get("date_retrieved"),
            })

        return hits
