import os
import json
from typing import Dict, List
from openai import OpenAI


# --------------------------------------------------
# LLM setup
# --------------------------------------------------
# This evaluator uses an LLM (via OpenRouter) to score the
# *quality* of the generated research report.
# It is NOT used to generate research content — only to judge it.

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
# LLMs often wrap JSON in ```json fences or add text before it.
# This function strips formatting noise and returns clean JSON text.

def _clean_llm_json(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    # Remove leading "json" label
    if text.lower().startswith("json"):
        text = text[4:].strip()

    # Trim everything before first JSON object
    first = text.find("{")
    if first != -1:
        text = text[first:]

    return text.strip()


# --------------------------------------------------
# Evaluator
# --------------------------------------------------

def evaluate_report(
    user_query: str,
    research_plan: Dict[str, List[str]],
    report_text: str,
    summaries: Dict[str, str],
    headings: List[str],
    references: List[str],
) -> Dict:
    """
    REPORT QUALITY EVALUATOR
    ========================

    Purpose
    -------
    Performs a *post-generation academic quality evaluation* of the
    research report using an LLM acting as a strict evaluator.

    This is a meta-analysis step — it does NOT modify the report.
    It only scores and diagnoses quality.

    The evaluation is PLAN-AWARE and EVIDENCE-GROUNDED.

    Evaluation Dimensions
    ---------------------
    1. Accuracy & Grounding
       - Are claims supported by summaries?
       - Any hallucinations or overstatements?

    2. Coverage & Completeness
       - Are all planned research dimensions addressed?
       - Is coverage balanced and aligned with the research goal?

    3. Citation Quality
       - Sentence-level citation discipline
       - Marker-to-reference correctness

    4. Structure & Clarity
       - Logical flow
       - Section balance
       - Paragraph depth

    Inputs
    ------
    user_query : str
        Original research question.

    research_plan : Dict
        Output of research planner:
        {
            "goal": "...",
            "dimensions": [...]
        }

    report_text : str
        Fully generated academic report.

    summaries : Dict[str, str]
        Ground truth factual base used during generation.

    headings : List[str]
        Final report headings.

    references : List[str]
        Final formatted references list.

    Output
    ------
    Dict
        Structured evaluation report with scores, notes, and limitations.
    """

    # Convert summaries into evaluator-readable block
    summary_block = "\n".join(
        f"{sid}: {text}"
        for sid, text in summaries.items()
    )

    heading_block = "\n".join(headings)
    refs_block = "\n".join(references)

    # Planned dimensions list
    plan_block = "\n".join(
        f"- {d}"
        for d in research_plan.get("dimensions", [])
    )

    # --------------------------------------------------
    # Evaluation Prompt
    # --------------------------------------------------
    # The prompt enforces strict grounding:
    #   Research Plan = Intended Scope
    #   Summaries     = Allowed Facts
    #   Report        = Object Being Judged

    prompt = f"""
You are a strict academic research evaluator.

YOUR TASK:
Evaluate the quality of the generated research report relative to the
USER'S RESEARCH QUESTION and the INTENDED RESEARCH PLAN.

You must assess ONLY what is provided.
Do NOT invent missing information.
Do NOT rewrite or fix the report.

================ EVALUATION PRINCIPLES ================

- The research plan defines the INTENDED SCOPE.
- The summaries define the ONLY allowed factual ground truth.
- The report must be evaluated based on how well it uses the summaries
  to fulfill the research plan.

================ EVALUATION CRITERIA ================

1. Accuracy & Grounding
- Are all claims in the report supported by the provided summaries?
- Are there any hallucinated, unsupported, or overstated claims?

2. Coverage & Completeness
- Does the report address ALL planned research dimensions?
- Are any dimensions missing, weakly covered, or unevenly developed?
- Does the synthesis align with the stated research goal?

3. Citation Quality
- Are declarative sentences properly cited?
- Do citation markers correspond correctly to the references?

4. Structure & Clarity
- Logical flow and coherence
- Adequate paragraph depth
- Balanced treatment of sections

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

RESEARCH PLAN (INTENDED SCOPE):
Goal:
{research_plan.get("goal")}

Planned Dimensions:
{plan_block}

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

    # Run evaluator model
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=900,
    )

    raw = r.choices[0].message.content or ""
    cleaned = _clean_llm_json(raw)

    # Attempt to parse evaluation JSON
    try:
        return json.loads(cleaned)
    except Exception:
        # Fail-safe: return diagnostic error instead of crashing pipeline
        return {
            "status": "evaluation_failed",
            "reason": "Evaluator returned invalid JSON",
            "raw_output": raw.strip()[:1000],
        }
