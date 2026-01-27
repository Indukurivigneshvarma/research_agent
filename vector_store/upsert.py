# vector_store/upsert.py

"""
VECTOR STORE UPSERT LOGIC
==========================

Purpose:
--------
This module controls how new research summaries are inserted into the
persistent vector database.

Key Responsibility:
-------------------
Prevent duplicate storage of the same web source.

Why This Matters:
-----------------
Without this layer, the system could store the same article multiple times
(across different queries or sessions), which would:

• Pollute the research memory
• Bias vector retrieval
• Increase storage unnecessarily
• Break scoring fairness

This file ensures:
✔ Each URL is stored at most once
✔ The vector DB remains clean and deduplicated
✔ Research memory grows without redundancy
"""

from typing import List, Dict, Tuple
from vector_store.client import VectorStoreClient


# --------------------------------------------------
# Fingerprinting Strategy
# --------------------------------------------------

def _fingerprint(r: Dict) -> Tuple:
    """
    Generates a unique identity signature for a summary record.

    Current strategy:
    ------------------
    Uses the URL as the unique identifier.

    Why URL?
    --------
    Each summary is derived from a specific web source.
    If the same URL appears again, we treat it as already stored.

    Returns
    -------
    Tuple
        Tuple is used so the system can be extended later
        (e.g., include domain + date if needed).
    """
    return (
        r.get("url"),
    )


# --------------------------------------------------
# Deduplicated Upsert
# --------------------------------------------------

def upsert_summaries(
    client: VectorStoreClient,
    records: List[Dict],
):
    """
    Inserts new summary records into the vector store,
    but only if they are not already present.

    Parameters
    ----------
    client : VectorStoreClient
        The persistent vector DB instance.
    records : List[Dict]
        New summary records generated during ingestion.

    Behavior
    --------
    1. Builds a set of existing fingerprints from the vector DB.
    2. Filters out any new records whose URL already exists.
    3. Only truly new sources are inserted.

    This guarantees:
    ----------------
    • No duplicate web sources in memory
    • Clean long-term research store
    • Stable vector retrieval behavior
    """

    # Build a set of fingerprints for all stored records
    existing = {
        _fingerprint(r)
        for r in client.metadata
    }

    new_records = []

    for r in records:
        fp = _fingerprint(r)

        # Skip if URL already stored
        if fp in existing:
            continue

        new_records.append(r)
        existing.add(fp)

    # Only upsert if there is something new
    if new_records:
        client.upsert(new_records)
