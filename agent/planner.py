# agent/planner.py

import re
from llm.ollama_client import call_ollama


def _sanitize_query(text: str) -> str:
    """
    Defensive cleaning so search queries never contain
    explanations, punctuation noise, or long sentences.
    """
    text = text.strip()
    text = re.sub(r"\(.*?\)", "", text)                 # remove parentheses
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)          # remove punctuation
    words = text.split()
    return " ".join(words[:12])                         # cap length


def generate_sub_questions(query: str) -> list[str]:
    """
    Generate SHORT, CLEAN research sub-questions.
    No explanations. No examples.
    """

    prompt = f"""
Break the following research topic into 4 to 6 short research sub-questions.

RULES:
- Each sub-question must be ONE clear sentence
- No explanations
- No examples
- No numbering
- No markdown

Topic:
{query}
"""

    response = call_ollama(
        prompt,
        temperature=0.2,
        tier="small"
    )

    questions = []
    for line in response.split("\n"):
        line = line.strip()
        if line:
            questions.append(line)

    return questions[:6]


def plan_search(query: str, sub_questions: list[str], iteration: int) -> list[str]:
    """
    Convert sub-questions into CLEAN, SEARCHABLE queries.
    """

    # Use one sub-question per iteration
    if iteration < len(sub_questions):
        sq = sub_questions[iteration]

        prompt = f"""
Convert the following research question into a concise web search query.

RULES:
- Return ONLY the search query
- No explanations
- No punctuation
- Max 10 words

Question:
{sq}
"""

        raw_query = call_ollama(
            prompt,
            temperature=0.0,
            tier="small"
        )

        search_query = _sanitize_query(raw_query)

        if search_query:
            return [search_query]

    # Fallback if sub-questions exhausted
    return [_sanitize_query(query)]
