"""
credibility_loader.py
=====================

Purpose
-------
Provides a deterministic, external credibility knowledge base.

This module loads trusted:
1. Authors
2. Source domains (journals, repositories, etc.)

These are used by `summary_scorer.py` to assign credibility scores
to each summary based on WHO wrote it and WHERE it was published.

Key Design Principle
--------------------
This data is loaded **once at startup**, not during runtime loops.
That makes credibility scoring:
- Fast
- Consistent
- Non-LLM based (deterministic)

This prevents credibility from being influenced by model output.
"""

import pandas as pd


class CredibilityStore:
    """
    Holds two external credibility datasets:

    1. authors → Set of trusted authors
    2. sources → Mapping of domain → venue_type

    Example:
        authors = {"john doe", "jane smith"}

        sources = {
            "nature.com": "journal",
            "arxiv.org": "repository"
        }
    """

    def __init__(
        self,
        authors_path: str = "datasets/authors.xlsx",
        sources_path: str = "datasets/sources.xlsx",
    ):
        # Set of normalized credible author names
        self.authors = set()

        # Domain → venue type mapping
        self.sources = {}

        # Load both datasets at initialization
        self._load_authors(authors_path)
        self._load_sources(sources_path)

    # --------------------------------------------------
    # Load credible authors
    # --------------------------------------------------
    def _load_authors(self, path: str):
        """
        Reads authors.xlsx and builds a normalized author set.

        Expected column:
            - author

        All names are lowercased and stripped to ensure
        matching is case-insensitive.
        """
        df = pd.read_excel(path)

        self.authors = {
            str(author).strip().lower()
            for author in df["author"].dropna().tolist()
        }

    # --------------------------------------------------
    # Load credible source domains
    # --------------------------------------------------
    def _load_sources(self, path: str):
        """
        Reads sources.xlsx and builds a domain → venue_type map.

        Expected columns:
            - domain
            - venue_type

        Example:
            "nature.com" → "journal"
            "arxiv.org" → "repository"
        """
        df = pd.read_excel(path)

        for _, row in df.iterrows():
            domain = str(row["domain"]).strip().lower()
            venue_type = str(row["venue_type"]).strip().lower()

            if domain:
                self.sources[domain] = venue_type


# ------------------------------------------------------------------
# Global instance (singleton-style)
# ------------------------------------------------------------------
# This ensures datasets are loaded once and reused everywhere.
CRED_STORE = CredibilityStore()
