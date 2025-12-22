# agent/research_mode.py

def detect_research_mode(query: str) -> str:
    q = query.lower()

    analytical_keywords = [
        "impact", "effect", "influence", "role", "future",
        "architecture", "system", "technology", "engineering"
    ]

    historical_keywords = [
        "history", "ancient", "civilization", "evolution",
        "culture", "heritage", "empire", "era"
    ]

    biographical_keywords = [
        "career", "life", "biography", "achievements",
        "records", "early life"
    ]

    comparative_keywords = [
        "compare", "vs", "difference", "better",
        "pros", "cons", "advantages", "disadvantages"
    ]

    if any(k in q for k in analytical_keywords):
        return "analytical"
    if any(k in q for k in historical_keywords):
        return "historical"
    if any(k in q for k in biographical_keywords):
        return "biographical"
    if any(k in q for k in comparative_keywords):
        return "comparative"

    # Safe default (slightly strict)
    return "analytical"
