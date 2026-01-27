"""
summary_scorer.py
=================

Purpose
-------
Computes a **base credibility score** for each collected summary before
cross-source agreement analysis.

This score reflects how trustworthy a source is based on:
1. Author credibility (known expert authors)
2. Source/domain credibility (known journals, repositories, etc.)
3. Venue type weight (journal > repository > book series)

This score later contributes to:
    total_score = credibility_score + agreement_score

So this file directly influences which sources dominate when conflicts occur.
"""

from scoring.credibility_loader import CredibilityStore
from urllib.parse import urlparse


# ------------------------------------------------------------------
# Load credibility datasets ONCE at startup
# ------------------------------------------------------------------
# Contains:
#   - Set of trusted authors
#   - Mapping of domain → venue_type
CRED_STORE = CredibilityStore()


# ------------------------------------------------------------------
# Extra scoring weight by venue type
# ------------------------------------------------------------------
# After a domain is known to be credible, venue type adds nuance.
# Example:
#   Journal article is more authoritative than a repository preprint.
VENUE_SCORES = {
    "journal": 3,
    "repository": 2,
    "book series": 1,
}


# ------------------------------------------------------------------
# Helper: extract clean domain from URL
# ------------------------------------------------------------------
def _extract_domain(url: str | None) -> str | None:
    """
    Extracts domain name from a URL.

    Example:
        https://www.nature.com/article → nature.com
    """
    if not url:
        return None
    netloc = urlparse(url).netloc.lower()
    return netloc.replace("www.", "")


# ------------------------------------------------------------------
# Main scoring function
# ------------------------------------------------------------------
def compute_summary_score(summary_record: dict) -> int:
    """
    Computes the base credibility score for a single summary.

    Input:
        summary_record (dict) — must contain metadata such as:
            - author
            - url or domain

    Scoring Logic:
        +5  if author is in trusted author list
        +5  if domain is in trusted source list
        +0–3 additional points based on venue type

    Output:
        Integer credibility score
    """

    score = 0

    # --------------------------------------------------
    # 1. Author credibility
    # --------------------------------------------------
    author = summary_record.get("author")
    if author and author.strip().lower() in CRED_STORE.authors:
        score += 5

    # --------------------------------------------------
    # 2. Domain credibility
    # --------------------------------------------------
    domain = (
        summary_record.get("domain")
        or _extract_domain(summary_record.get("url"))
    )

    if domain and domain in CRED_STORE.sources:
        score += 5

        # Add venue-type bonus
        venue = CRED_STORE.sources[domain]
        score += VENUE_SCORES.get(venue, 0)

    return score
