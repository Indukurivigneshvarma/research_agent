# vector_store/client.py

import os
import json
from typing import List, Dict
import numpy as np
import faiss


class VectorStoreClient:
    """
    Persistent FAISS vector store for SUMMARY-LEVEL records.
    Embedding = query_text embedding
    """

    def __init__(
        self,
        persist_dir: str = "vector_data",
        embedding_dim: int = 384,
    ):
        self.persist_dir = persist_dir
        self.embedding_dim = embedding_dim

        os.makedirs(self.persist_dir, exist_ok=True)

        self.index_path = os.path.join(persist_dir, "index.faiss")
        self.meta_path = os.path.join(persist_dir, "metadata.json")

        self._load_or_create()

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    def _load_or_create(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata: List[Dict] = json.load(f)
        else:
            self.metadata = []

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                "FAISS index and metadata out of sync."
            )

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        embedding: List[float],
        top_k: int,
    ) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        query = np.array([embedding], dtype="float32")
        _, idxs = self.index.search(query, top_k)

        results = []
        for i in idxs[0]:
            if i == -1:
                continue
            results.append(self.metadata[i])

        return results

    # --------------------------------------------------
    # Upsert (append-only)
    # --------------------------------------------------

    def upsert(self, records: List[Dict]):
        """
        records MUST contain:
        - embedding (query_text embedding)
        """

        if not records:
            return

        vectors = np.array(
            [r["embedding"] for r in records],
            dtype="float32",
        )

        self.index.add(vectors)
        self.metadata.extend(records)

        self._persist()

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------
    # Reset protection
    # --------------------------------------------------

    def reset(self):
        raise RuntimeError("Vector DB reset disabled.")
