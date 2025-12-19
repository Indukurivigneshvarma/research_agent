def evaluate_sources(pages):
    evaluated = []

    for page in pages:
        url = page["url"]

        if any(domain in url for domain in [
            "mckinsey.com",
            "gartner.com",
            "forbes.com",
            "medium.com",
            "harvard.edu",
            "mit.edu",
            "stackoverflow.com"
        ]):
            credibility = "high"
        else:
            credibility = "medium"

        page["credibility"] = credibility
        evaluated.append(page)

    return evaluated
