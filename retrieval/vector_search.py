# retrieval/vector_search.py

from typing import List, Dict


class VectorSearcher:
    def __init__(self, vector_client):
        self.client = vector_client

    def search(
        self,
        query_embedding,
        top_k: int,
    ) -> List[Dict]:
        """
        Vector search over STORED QUERY TEXTS.
        """

        results = self.client.search(
            embedding=query_embedding,
            top_k=top_k,
        )

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
