# utils/text.py

import re


def clean_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
