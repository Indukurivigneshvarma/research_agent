# utils/dates.py

from datetime import datetime
import re


def normalize_date(date_str: str | None) -> str | None:
    """
    Normalizes date strings to YYYY-MM-DD if possible.
    """
    if not date_str:
        return None

    try:
        return datetime.fromisoformat(date_str).date().isoformat()
    except Exception:
        pass

    # fallback: try to extract YYYY-MM-DD pattern
    match = re.search(r"\d{4}-\d{2}-\d{2}", date_str)
    if match:
        return match.group(0)

    return None


def today_iso() -> str:
    return datetime.utcnow().date().isoformat()
