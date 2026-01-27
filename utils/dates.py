# utils/dates.py

"""
DATE UTILITIES
==============

Purpose:
--------
Standardizes date handling across the entire research pipeline.

Why This Is Important:
----------------------
Web sources return publication dates in many messy formats:
• ISO strings
• Text blobs
• Mixed formats
• Sometimes missing entirely

The research system needs:
✔ Consistent date formatting
✔ Safe parsing (never crash)
✔ Machine-comparable dates

This module ensures all dates become:

    YYYY-MM-DD

Which allows:
• Stable storage in vector DB
• Reliable sorting and filtering
• Clean metadata in reports
"""

from datetime import datetime
import re


# --------------------------------------------------
# Normalize arbitrary date strings
# --------------------------------------------------

def normalize_date(date_str: str | None) -> str | None:
    """
    Attempts to convert a date string into ISO format (YYYY-MM-DD).

    Parameters
    ----------
    date_str : str | None
        Raw date string extracted from metadata.

    Returns
    -------
    str | None
        Clean ISO date if extraction succeeds,
        otherwise None.

    Strategy
    --------
    1. Try strict ISO parsing first.
    2. If that fails, search the string for a date pattern.
    3. If nothing valid is found, return None.

    This makes the system:
    • Fault-tolerant
    • Format-agnostic
    • Safe from ingestion crashes
    """

    if not date_str:
        return None

    # Attempt ISO parsing (best case)
    try:
        return datetime.fromisoformat(date_str).date().isoformat()
    except Exception:
        pass

    # Fallback: extract a YYYY-MM-DD pattern from text
    match = re.search(r"\d{4}-\d{2}-\d{2}", date_str)
    if match:
        return match.group(0)

    return None


# --------------------------------------------------
# Today's date (UTC)
# --------------------------------------------------

def today_iso() -> str:
    """
    Returns today's date in ISO format (UTC).

    Used for:
    ---------
    • date_retrieved metadata
    • Tracking when sources entered the system

    Using UTC ensures:
    • Timezone-independent logging
    • Consistent audit trails
    """

    return datetime.utcnow().date().isoformat()
