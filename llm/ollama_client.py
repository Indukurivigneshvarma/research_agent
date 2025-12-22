# llm/ollama_client.py

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Groq configuration
# ----------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL_NAME = "llama-3.1-8b-instant"

# ----------------------------
# Token budgets (IMPORTANT)
# ----------------------------
# Keep these conservative to stay under 6000 TPM
TOKEN_BUDGETS = {
    "small": 250,     # planning, rewriting
    "medium": 500,    # article summaries
    "large": 1800     # iteration + synthesis
}

# ----------------------------
# Rate-limit handling
# ----------------------------

BASE_SLEEP_SECONDS = 4        # normal pacing
RETRY_SLEEP_SECONDS = 12      # when 429 occurs
MAX_RETRIES = 5               # safety


def call_ollama(
    prompt: str,
    temperature: float = 0.2,
    tier: str = "medium"
) -> str:
    """
    Safe Groq client with:
    - Tiered token budgets
    - Automatic retry on TPM (429)
    - Built-in pacing
    """

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not found in environment variables")

    max_tokens = TOKEN_BUDGETS.get(tier, TOKEN_BUDGETS["medium"])

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a precise research assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    retries = 0

    while retries < MAX_RETRIES:
        response = requests.post(
            GROQ_API_URL,
            json=payload,
            headers=headers,
            timeout=60
        )

        # ----------------------------
        # SUCCESS
        # ----------------------------
        if response.status_code == 200:
            data = response.json()

            # normal pacing to avoid burst TPM
            time.sleep(BASE_SLEEP_SECONDS)

            return data["choices"][0]["message"]["content"]

        # ----------------------------
        # RATE LIMIT (TPM)
        # ----------------------------
        if response.status_code == 429:
            retries += 1
            print(
                f"[Groq TPM] Rate limit hit. "
                f"Retrying in {RETRY_SLEEP_SECONDS}s "
                f"(attempt {retries}/{MAX_RETRIES})"
            )
            time.sleep(RETRY_SLEEP_SECONDS)
            continue

        # ----------------------------
        # OTHER ERRORS
        # ----------------------------
        print("Groq API error:", response.status_code)
        print(response.text)
        raise RuntimeError("Groq API request failed")

    raise RuntimeError("Groq API failed after maximum retries")
