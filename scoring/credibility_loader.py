# scoring/credibility_loader.py

import pandas as pd


class CredibilityStore:
    """
    Loads external credibility datasets ONCE at startup.

    - authors.xlsx  → known credible authors
    - sources.xlsx  → domain → venue_type mapping

    Used by summary_scorer.py for deterministic scoring.
    """

    def __init__(
        self,
        authors_path: str = "datasets/authors.xlsx",
        sources_path: str = "datasets/sources.xlsx",
    ):
        self.authors = set()
        self.sources = {}

        self._load_authors(authors_path)
        self._load_sources(sources_path)

    # --------------------------------------------------
    # Load credible authors
    # --------------------------------------------------

    def _load_authors(self, path: str):
        """
        Expected columns in authors.xlsx:
        - author
        """
        df = pd.read_excel(path)

        self.authors = {
            str(author).strip().lower()
            for author in df["author"].dropna().tolist()
        }

    # --------------------------------------------------
    # Load credible sources / venues
    # --------------------------------------------------

    def _load_sources(self, path: str):
        """
        Expected columns in sources.xlsx:
        - domain
        - venue_type
        """
        df = pd.read_excel(path)

        for _, row in df.iterrows():
            domain = str(row["domain"]).strip().lower()
            venue_type = str(row["venue_type"]).strip().lower()

            if domain:
                self.sources[domain] = venue_type
                
CRED_STORE = CredibilityStore()