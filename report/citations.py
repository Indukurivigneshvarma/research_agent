from typing import List, Dict


def build_cited_summaries(
    summaries: List[Dict],
) -> Dict[str, str]:
    """
    Maps summary IDs to citation markers.

    Output:
      {
        "S1": "[S1]",
        "S2": "[S2]"
      }
    """
    return {
        s["id"]: f"[{s['id']}]"
        for s in summaries
    }


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

        refs.append(
            f"[{sid}] {author}. {domain}. {url}"
        )

    return refs
