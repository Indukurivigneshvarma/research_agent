# retrieval/tavily_client.py

import os
from tavily import TavilyClient
from urllib.parse import urlparse


_tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


def tavily_search(
    query: str,
    max_results: int = 3,
):
    """
    Returns a list of Tavily result objects.
    Each result contains at least:
      - url
      - score   (relevance score)
    """
    response = _tavily.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=False,
        include_answer=False,
    )
    return response.get("results", [])


def tavily_extract(url: str):
    """
    Extracts raw text for a single URL.
    """
    response = _tavily.extract(
        urls=[url],
        include_raw_content=True,
    )

    data = response.get("results", [])
    if not data:
        return None

    item = data[0]
    domain = urlparse(url).netloc.replace("www.", "")

    return {
        "url": url,
        "domain": domain,
        "raw_text": item.get("raw_content") or item.get("content"),
        "published_date": item.get("published_date"),
    }
