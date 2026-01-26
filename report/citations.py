from typing import List, Dict


def build_references(
    summaries: List[Dict],
) -> List[str]:
    """
    Builds reference entries for the References section.

    Format:
      [S1] Author. domain. url
    """
    refs = []

    for s in summaries:
        sid = s["id"]
        author = s.get("author") or "Unknown Author"
        domain = s.get("domain") or "Unknown Source"
        url = s.get("url") or ""

        refs.append(f"[{sid}] {author}. {domain}. {url}")

    return refs
