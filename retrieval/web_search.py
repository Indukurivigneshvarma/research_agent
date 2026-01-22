# retrieval/web_search.py

from retrieval.tavily_client import tavily_search


def search_web(
    query: str,
    max_results: int = 3,
):
    return tavily_search(
        query=query,
        max_results=max_results,
    )
