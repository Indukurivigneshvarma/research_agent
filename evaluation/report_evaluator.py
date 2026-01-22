import os
import json
from typing import Dict, List
from openai import OpenAI


# --------------------------------------------------
# LLM setup
# --------------------------------------------------

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = os.getenv(
    "OPENROUTER_EVAL_MODEL",
    "meta-llama/llama-3.1-70b-instruct"
)


# --------------------------------------------------
# JSON CLEANER (MANDATORY)
# --------------------------------------------------

def _clean_llm_json(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    # Remove ```json fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    # Remove leading 'json'
    if text.lower().startswith("json"):
        text = text[4:].strip()

    # Trim before first '{'
    first = text.find("{")
    if first != -1:
        text = text[first:]

    return text.strip()


# --------------------------------------------------
# Evaluator
# --------------------------------------------------

def evaluate_report(
    user_query: str,
    report_text: str,
    summaries: Dict[str, str],
    headings: List[str],
    references: List[str],
) -> Dict:
    """
    Performs a post-generation evaluation of the research report.

    Evaluates:
    - Factual grounding in summaries
    - Coverage of the user query
    - Structural quality
    - Citation discipline
    """

    summary_block = "\n".join(
        f"{sid}: {text}"
        for sid, text in summaries.items()
    )

    heading_block = "\n".join(headings)
    refs_block = "\n".join(references)

    prompt = f"""
You are a strict academic research evaluator.

YOUR TASK:
Evaluate the quality of the generated research report.

You must assess ONLY what is provided.
Do NOT invent missing information.
Do NOT rewrite or fix the report.

================ EVALUATION CRITERIA ================

1. Accuracy & Grounding
- Are claims supported by the summaries?
- Any hallucinated or unsupported facts?

2. Coverage & Completeness
- Does the report fully address the research question?
- Are any major dimensions missing?

3. Citation Quality
- Sentence-level citations in topical sections
- References match citation markers

4. Structure & Clarity
- Logical flow
- Adequate paragraph depth
- Balanced sections

================ OUTPUT RULES ================
- Return JSON ONLY
- No explanations
- No markdown
- No commentary

================ OUTPUT FORMAT ================
{{
  "overall_score": <float 0-10>,
  "accuracy": {{ "score": <0-10>, "notes": "<text>" }},
  "completeness": {{ "score": <0-10>, "notes": "<text>" }},
  "citation_quality": {{ "score": <0-10>, "notes": "<text>" }},
  "structure": {{ "score": <0-10>, "notes": "<text>" }},
  "limitations": [ "<limitation 1>", "<limitation 2>" ],
  "confidence_level": "low | medium | high"
}}

================ INPUTS ================

RESEARCH QUESTION:
{user_query}

HEADINGS:
{heading_block}

SUMMARIES (GROUND TRUTH):
{summary_block}

REFERENCES:
{refs_block}

GENERATED REPORT:
{report_text}

================ OUTPUT ================
Return ONLY the JSON object.
""".strip()

    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=800,
    )

    raw = r.choices[0].message.content or ""
    cleaned = _clean_llm_json(raw)

    try:
        return json.loads(cleaned)
    except Exception:
        # Fail-safe: NEVER crash pipeline
        return {
            "status": "evaluation_failed",
            "reason": "Evaluator returned invalid JSON",
            "raw_output": raw.strip()[:1000],
        }

