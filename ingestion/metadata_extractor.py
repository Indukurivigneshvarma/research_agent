# ingestion/metadata_extractor.py

from newspaper import Article
from typing import Dict, Optional


def extract_metadata(url: str) -> Dict[str, Optional[str]]:
    """
    Extracts author and publish date using newspaper3k.
    """

    try:
        article = Article(url)
        article.download()
        article.parse()
    except Exception:
        return {
            "author": None,
            "date_published": None,
        }

    author = None
    if article.authors:
        author = ", ".join(article.authors)

    date_published = None
    if article.publish_date:
        date_published = article.publish_date.date().isoformat()

    return {
        "author": author,
        "date_published": date_published,
    }
