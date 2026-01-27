# vector_store/client.py

"""
VECTOR STORE CLIENT
====================

This module implements a persistent vector database using FAISS.

Purpose:
--------
Stores and retrieves previously processed research summaries so the
system can reuse past knowledge instead of re-searching the web.

Design Philosophy:
------------------
• Each stored vector represents the *intent of a query* (query_text embedding),
  NOT the document content embedding.
• This allows the system to reuse summaries when a *new question*
  is semantically similar to a previous one.

Persistence:
------------
Two files are maintained:
1. index.faiss   → dense vector index (fast similarity search)
2. metadata.json → structured metadata for each stored summary

Together they form the system's long-term research memory.
"""

import os
import json
from typing import List, Dict
import numpy as np
import faiss


class VectorStoreClient:
    """
    Persistent FAISS vector store for SUMMARY-LEVEL records.

    Each entry corresponds to one stored research summary and includes:
    - query_text (the question that led to the summary)
    - embedding (vector of the query_text)
    - summary content
    - metadata (url, author, domain, dates, etc.)

    This store allows:
    ✔ Semantic retrieval of past research
    ✔ Avoiding duplicate web searches
    ✔ Intent reuse across different research sessions
    """

    def __init__(
        self,
        persist_dir: str = "vector_data",
        embedding_dim: int = 384,
    ):
        """
        Initializes the vector store.

        Parameters
        ----------
        persist_dir : str
            Folder where FAISS index and metadata are stored.
        embedding_dim : int
            Dimension of the embedding model used (must match embedder).
        """

        self.persist_dir = persist_dir
        self.embedding_dim = embedding_dim

        # Ensure storage folder exists
        os.makedirs(self.persist_dir, exist_ok=True)

        # Paths to persistent files
        self.index_path = os.path.join(persist_dir, "index.faiss")
        self.meta_path = os.path.join(persist_dir, "metadata.json")

        # Load existing memory or create a new one
        self._load_or_create()

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    def _load_or_create(self):
        """
        Loads the FAISS index and metadata from disk if available.
        Otherwise creates an empty index.

        A strict consistency check ensures:
        #vectors in FAISS index == #metadata records
        """

        # Load vector index or create a new one
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            # Inner Product index (used like cosine similarity if embeddings are normalized)
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Load metadata or initialize empty list
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata: List[Dict] = json.load(f)
        else:
            self.metadata = []

        # Safety check: index and metadata must match
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
        """
        Performs semantic search over stored query embeddings.

        Parameters
        ----------
        embedding : List[float]
            Embedding of the current user query.
        top_k : int
            Number of similar past summaries to retrieve.

        Returns
        -------
        List[Dict]
            Matching metadata records (not raw vectors).
        """

        # If database is empty, return nothing safely
        if self.index.ntotal == 0:
            return []

        # Convert to FAISS-compatible format
        query = np.array([embedding], dtype="float32")

        # Perform nearest neighbor search
        _, idxs = self.index.search(query, top_k)

        # Map indices back to metadata
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
        Adds new summary records to the vector store.

        Important:
        ----------
        This is APPEND-ONLY. Stored research is never overwritten.

        Each record MUST contain:
        - embedding (vector of query_text)

        This ensures reproducibility and a growing research memory.
        """

        if not records:
            return

        # Extract embeddings
        vectors = np.array(
            [r["embedding"] for r in records],
            dtype="float32",
        )

        # Add to FAISS index
        self.index.add(vectors)

        # Add metadata
        self.metadata.extend(records)

        # Persist to disk
        self._persist()

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def _persist(self):
        """
        Writes both the FAISS index and metadata to disk.
        This guarantees memory survives application restarts.
        """

        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
