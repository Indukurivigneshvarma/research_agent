from typing import List, Dict


def build_references(
    summaries: List[Dict],
) -> List[str]:
    """
    REFERENCES BUILDER
    ==================

    Purpose
    -------
    This function constructs the **References section** of the final
    academic report. Each summary used in the synthesis becomes one
    citation entry.

    These references are:
    • Deterministic (no LLM involvement)
    • Directly tied to collected summaries
    • Used verbatim in the report writer step

    Input
    -----
    summaries : List[Dict]
        A list of processed summary records produced during the
        research pipeline. Each record may contain:
        - id
        - author
        - domain
        - url
        (Some fields may be missing.)

    Output
    ------
    List[str]
        A list of formatted reference strings.

    Format
    ------
    [S1] Author. domain. url

    Example:
    [S3] John Smith. nature.com. https://www.nature.com/article...

    Notes
    -----
    • Missing metadata is replaced with safe defaults
      ("Unknown Author", "Unknown Source").
    • These references correspond to citation markers [S1], [S2], etc.,
      used inside the report text.
    """

    refs = []

    # Each summary becomes one reference entry
    for s in summaries:
        sid = s["id"]

        # Use fallback values if metadata is missing
        author = s.get("author") or "Unknown Author"
        domain = s.get("domain") or "Unknown Source"
        url = s.get("url") or ""

        # Build reference string
        refs.append(f"[{sid}] {author}. {domain}. {url}")

    return refs
