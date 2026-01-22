# vector_store/upsert.py

from typing import List, Dict, Tuple
from vector_store.client import VectorStoreClient


def _fingerprint(r: Dict) -> Tuple:
    return (
        r.get("url"),
    )


def upsert_summaries(
    client: VectorStoreClient,
    records: List[Dict],
):
    """
    Persist ONLY summary-level records.
    """

    existing = {
        _fingerprint(r)
        for r in client.metadata
    }

    new_records = []

    for r in records:
        fp = _fingerprint(r)
        if fp in existing:
            continue
        new_records.append(r)
        existing.add(fp)

    if new_records:
        client.upsert(new_records)
