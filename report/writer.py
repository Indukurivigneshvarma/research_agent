import os
from typing import Dict, List
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.1-70b-instruct"
)


def write_report(
    title: str,
    headings: List[str],
    summaries: Dict[str, str],
    references: List[str],
) -> str:
    """
    Writes a full academic research report with:

    - Sentence-level citations in ALL narrative sections
    - Deterministic References section
    """

    summary_block = "\n".join(
        f"{sid}: {text}"
        for sid, text in summaries.items()
    )

    heading_block = "\n".join(
        f"@@{h}@@" for h in headings
    )

    refs_block = "\n".join(references)

    prompt = f"""
You are writing an academic research synthesis.

YOU MUST FOLLOW THE FORMAT EXACTLY.

================ FORMAT RULES ================
- Use ONLY the headings provided.
- Each heading MUST appear exactly once.
- Headings MUST be wrapped like this: @@Heading Name@@
- The title MUST be wrapped like this:
  @@TITLE@@
  <title>
  @@TITLE@@
- Plain text only.
- No markdown.
- No bullet points.

================ CITATION RULES ================
- EVERY declarative sentence in ALL sections
  (including Executive Summary and Conclusion)
  MUST end with one or more citation markers.
- Citations must be placed immediately after the sentence.
- Use format: [S1] or [S1][S3]
- NO paragraph-level citations.
- NO uncited sentences anywhere except References.

================ CONTENT RULES ================
- You MAY paraphrase and synthesize.
- You MAY combine information from multiple summaries.
- You MUST NOT introduce facts not present in the summaries.
- If evidence is uncertain, state uncertainty explicitly.

- Executive Summary:
  Write 5–6 well-developed sentences.

- EACH topical section:
  Write 2–3 coherent paragraphs.
  Each paragraph must contain 4–5 sentences.
  Do NOT merge all content into a single paragraph.

- Conclusion:
  Write 5–6 sentences that synthesize arguments,
  implications, and limitations.

- Short or underdeveloped sections are NOT acceptable.

================ REFERENCES RULE ================
- When you reach @@References@@
- DO NOT write new text.
- OUTPUT EXACTLY the reference list provided.
- Do NOT modify references.

================ INPUTS ================

TITLE:
{title}

HEADINGS:
{heading_block}

SUMMARIES (ONLY SOURCE OF FACTS):
{summary_block}

REFERENCES (VERBATIM — DO NOT CHANGE):
{refs_block}

================ OUTPUT ================
Return the COMPLETE report using the exact format described.
Do NOT include anything else.
""".strip()

    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2500,
    )

    return r.choices[0].message.content.strip()
