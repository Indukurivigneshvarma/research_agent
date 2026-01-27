"""
metadata_extractor.py
────────────────────────────────────────────────────────────
Purpose:
    Extracts structured metadata (author + publication date)
    from web articles.

Why This File Exists:
    The Tavily extractor gives us raw text, but credibility
    scoring also depends on:

        • Author reputation
        • Publication source quality

    This module provides author + date information to support:
        → credibility scoring
        → reference formatting
        → research trace logging

Pipeline Role:
    Called during the WEB INGESTION stage in run.py,
    immediately after raw page text is fetched.
"""

from newspaper import Article
from typing import Dict, Optional


def extract_metadata(url: str) -> Dict[str, Optional[str]]:
    """
    Extracts metadata from a webpage using newspaper3k.

    Inputs:
        url:
            The webpage URL selected for ingestion.

    Returns:
        {
            "author": str | None,
            "date_published": str (YYYY-MM-DD) | None
        }

    Behavior:
        • Attempts to download and parse the article.
        • If parsing fails, returns None values safely.
        • Multiple authors are joined into a single string.

    Notes:
        • Author is used in credibility scoring.
        • Publication date is normalized later in utils/dates.py.
        • Failure here NEVER breaks the pipeline.
    """

    try:
        article = Article(url)
        article.download()
        article.parse()
    except Exception:
        # Metadata extraction failure is non-fatal
        return {
            "author": None,
            "date_published": None,
        }

    # ---------------------------
    # Author extraction
    # ---------------------------
    author = None
    if article.authors:
        # Combine multiple authors into a single string
        author = ", ".join(article.authors)

    # ---------------------------
    # Publication date extraction
    # ---------------------------
    date_published = None
    if article.publish_date:
        date_published = article.publish_date.date().isoformat()

    return {
        "author": author,
        "date_published": date_published,
    }
