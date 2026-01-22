# scoring/summary_scorer.py

from scoring.credibility_loader import CredibilityStore
from urllib.parse import urlparse


CRED_STORE = CredibilityStore()

VENUE_SCORES = {
    "journal": 3,
    "repository": 2,
    "book series": 1,
}


def _extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    netloc = urlparse(url).netloc.lower()
    return netloc.replace("www.", "")


def compute_summary_score(summary_record: dict) -> int:
    """
    Computes base credibility score for a summary.
    """

    score = 0

    # Author credibility
    author = summary_record.get("author")
    if author and author.strip().lower() in CRED_STORE.authors:
        score += 5

    # Domain + venue
    domain = (
        summary_record.get("domain")
        or _extract_domain(summary_record.get("url"))
    )

    if domain and domain in CRED_STORE.sources:
        score += 5
        venue = CRED_STORE.sources[domain]
        score += VENUE_SCORES.get(venue, 0)

    return score
