# tools/source_evaluator.py

from urllib.parse import urlparse

TRUSTED_DOMAINS = [
    "harvard.edu",
    "mit.edu",
    "stanford.edu",
    "forbes.com",
    "medium.com",
    "britannica.com",
    "espn.com",
    "cricinfo.com"
]


def evaluate_sources(pages, research_mode: str):
    """
    Soft credibility scoring.
    NEVER filters sources.
    """

    evaluated = []

    for page in pages:
        score = 0.4
        url = page["url"].lower()
        domain = urlparse(url).netloc

        if any(d in domain for d in TRUSTED_DOMAINS):
            score += 0.15

        if domain.endswith(".edu") or domain.endswith(".gov"):
            score += 0.2

        if url.startswith("https"):
            score += 0.05

        length = len(page["content"])
        if length >= 800:
            score += 0.1
        elif length < 300:
            score -= 0.1

        page["credibility_score"] = round(min(score, 1.0), 2)
        evaluated.append(page)

    return evaluated
