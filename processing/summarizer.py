import time
from llm.ollama_client import call_ollama


def summarize_article(article_text: str, source_url: str = None, **kwargs) -> str:
    """
    Summarize a single article.

    Accepts extra keyword arguments (like source_url)
    to remain compatible with controller calls.
    """

    prompt = f"""
Summarize the following content clearly and concisely.
Focus only on factual, useful research information.
Do not add opinions or assumptions.

Content:
{article_text}
"""

    summary = call_ollama(prompt, temperature=0.2)

    # 🛑 IMPORTANT: throttle Groq calls to avoid burst errors
    time.sleep(2)

    return summary


def summarize_iteration(texts: list[str]) -> str:
    """
    Summarize evidence collected in one research iteration.
    """

    joined = "\n\n".join(texts[:2])  # 🔒 limit to first 2 articles

    prompt = f"""
Summarize the following research evidence into key findings.
Use clear bullet points.
Do not add opinions.

Evidence:
{joined}
"""

    summary = call_ollama(prompt, temperature=0.2)

    # 🛑 throttle again
    time.sleep(2)

    return summary
