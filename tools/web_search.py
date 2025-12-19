import requests
from config.settings import TAVILY_API_KEY, MAX_SEARCH_RESULTS

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def web_search(query: str):
    if not TAVILY_API_KEY:
        raise RuntimeError("Tavily API key not found")

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": MAX_SEARCH_RESULTS,
        "include_raw_content": False
    }

    response = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data.get("results", [])
