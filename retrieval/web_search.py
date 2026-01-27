"""
web_search.py
────────────────────────────────────────────────────────────
Purpose:
    Provides a clean abstraction for web search.

Why This File Exists:
    Your pipeline should NOT depend directly on Tavily
    (or any specific search provider). This wrapper makes
    the search layer replaceable.

    Today → Tavily
    Tomorrow → Google, Bing, SerpAPI, custom crawler

Pipeline Role:
    When vector reuse is not possible, the system performs
    fresh web search to gather new evidence sources.
"""

from retrieval.tavily_client import tavily_search


def search_web(
    query: str,
    max_results: int = 3,
):
    """
    Performs web search for a given research query.

    Inputs:
        query:
            Academic-style search query generated during discovery.

        max_results:
            Number of URLs to retrieve from the search provider.

    Returns:
        List of search result dictionaries. Each result typically contains:
            {
                "url": source URL,
                "score": provider relevance score
            }

    Notes:
        • This function intentionally contains NO provider logic.
        • It simply delegates to the configured search backend.
    """
    return tavily_search(
        query=query,
        max_results=max_results,
    )
