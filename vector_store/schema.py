from typing import TypedDict, Optional, List


class SummaryRecord(TypedDict):
    # intent
    query_text: str
    embedding: List[float]

    # content
    summary: str

    # metadata
    url: Optional[str]
    domain: Optional[str]
    author: Optional[str]
    venue_type: Optional[str]

    date_published: Optional[str]   # YYYY-MM-DD
    date_retrieved: str             # YYYY-MM-DD
