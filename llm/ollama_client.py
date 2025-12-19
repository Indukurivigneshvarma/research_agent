import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ✅ Supported + stable on Groq
MODEL_NAME = "llama-3.1-8b-instant"

print("USING GROQ MODEL:", MODEL_NAME)


def call_ollama(prompt: str, temperature: float = 0.2) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not found")

    # 🔒 HARD CAP PROMPT SIZE (Groq-safe)
    MAX_PROMPT_CHARS = 5000
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS]

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a precise research assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 800
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        GROQ_API_URL,
        json=payload,
        headers=headers,
        timeout=60
    )

    if response.status_code != 200:
        print("Groq error status:", response.status_code)
        print("Groq error body:", response.text)
        raise RuntimeError("Groq API call failed")

    return response.json()["choices"][0]["message"]["content"]
