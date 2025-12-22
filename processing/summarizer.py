# processing/summarizer.py

import re


def summarize_article(article_text: str, **kwargs) -> str:
    """
    NON-LLM article compression.
    Extracts the most information-dense sentences.
    """

    sentences = re.split(r'(?<=[.!?])\s+', article_text)

    # keep longer, informative sentences
    selected = [
        s.strip()
        for s in sentences
        if len(s.split()) >= 15
    ]

    # cap to avoid explosion
    compressed = selected[:8]

    return " ".join(compressed)
