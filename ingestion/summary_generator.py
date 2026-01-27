"""
summary_generator.py
────────────────────────────────────────────────────────────
Purpose:
    Converts raw web article text into a dense, evidence-rich
    research summary suitable for synthesis into an academic report.

Why This File Exists:
    Raw web pages are too long, noisy, and inconsistent to use directly.
    This module performs controlled LLM-based compression while preserving:

        • Facts
        • Data points
        • Quantitative evidence
        • Named entities (tools, orgs, studies)
        • Trends and projections

Pipeline Role:
    Called during WEB INGESTION in run.py after:
        1) URL is selected
        2) Raw text is extracted
        3) Length limits are enforced

    Output feeds:
        → vector storage
        → agreement detection
        → conflict analysis
        → report generation
"""

import os
from groq import Groq
from openai import OpenAI
from config import MAX_SUMMARY_TOKENS

# ------------------------------
# Models Used for Summarization
# ------------------------------
# Smaller models are used here for efficiency, since the task
# is compression rather than deep reasoning.

GROQ_MODEL = "llama-3.1-8b-instant"              # Fast, cost-efficient
OR_MODEL   = "meta-llama/llama-3.1-8b-instruct"  # Similar scale via OpenRouter


# ------------------------------
# Provider Clients
# ------------------------------
# Two providers are supported so the pipeline can alternate
# between them for robustness and cost control.

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

or_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


# ------------------------------
# Main Summary Function
# ------------------------------

def generate_summary(
    raw_text: str,
    provider: str = "groq",  # "groq" | "openrouter"
) -> str:
    """
    Generates a structured, high-density research summary.

    Inputs:
        raw_text:
            Article text already truncated and filtered upstream.
        provider:
            Which LLM backend to use.

    Output:
        A single paragraph summary used as the canonical
        representation of that source in the research pipeline.

    Design Principles:
        • High recall over heavy compression
        • Extractive-style abstraction
        • Strict factual grounding
        • No stylistic fluff
    """

    provider = provider.lower().strip()

    # ------------------------------------------------------
    # Prompt Design
    # ------------------------------------------------------
    # Forces the model into an "evidence extraction" mindset,
    # not a general summarization style.
    # Prevents:
    #   - introductions
    #   - commentary
    #   - stylistic padding
    # Ensures:
    #   - data-rich output
    #   - report-ready format
    # ------------------------------------------------------

    prompt = f"""
Write ONLY the summary text.

This summary will be inserted directly into a research report.
Do NOT include introductions, explanations, labels, or meta commentary.

GOAL:
Produce a comprehensive, evidence-dense summary by extracting
all distinct claims, data points, roles, tasks, tools, trends,
and projections mentioned in the text.

REQUIREMENTS:
- Output EXACTLY ONE paragraph
- Aim for a thorough and complete summary of approximately
  1500–2000 characters, prioritizing recall over compression
- Do NOT overly compress related ideas into a single sentence
- Neutral, analytical tone
- Include all numbers, percentages, ranges, timelines,
  named studies, organizations, tools, and projections IF PRESENT
- Explicitly state when quantitative evidence is missing
- Do NOT say phrases like "this report", "this article",
  or "here is a summary"
- No bullet points, headings, lists, or line breaks
- No speculation
- Use ONLY the provided text

TEXT:
{raw_text}
""".strip()

    # ------------------------------
    # OpenRouter path
    # ------------------------------
    if provider == "openrouter":
        r = or_client.chat.completions.create(
            model=OR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,              # Low randomness for stability
            max_tokens=MAX_SUMMARY_TOKENS # Hard length cap
        )
        return r.choices[0].message.content.strip()

    # ------------------------------
    # Default: Groq path
    # ------------------------------
    r = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=MAX_SUMMARY_TOKENS,
    )

    return r.choices[0].message.content.strip()
