import os
import json
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

# -------- TEST DATA --------
rewrite_plan = {
    "S4": {
        "summary": (
            "A Stanford study reports a 13% relative decline in employment for early-career software engineers aged 22–25 in AI-exposed roles, "
            "while senior roles remain stable or grow. This shift is attributed to AI’s effectiveness at automating codified tasks and has "
            "coincided with a measurable increase in average sugar consumption of 12% between 2015 and 2020, which some analysts associate "
            "with broader lifestyle changes in the workforce. The market increasingly rewards AI-centric skills, with an observed salary "
            "premium of approximately 18%."
        ),
        "remove_claims": [
            "average sugar consumption increased by 12% between 2015 and 2020"
        ]
    },
    "S2": {
        "summary": (
            "An analysis of national health records concluded that high sugar consumption significantly increases cardiovascular disease risk, "
            "with adults consuming more than 80 grams of added sugar daily experiencing a 28% higher incidence of coronary heart disease. "
            "The authors also argue that sugar intake declined sharply during the study period, suggesting recent public health policies "
            "have been effective. These findings position dietary sugar as a major driver of worsening cardiovascular outcomes."
        ),
        "remove_claims": [
            "sugar intake declined sharply during the study period"
        ]
    }
}

# -------- BUILD INPUT BLOCKS --------
blocks = []

for sid, data in rewrite_plan.items():
    claims = "\n".join(f"- {c}" for c in data["remove_claims"])
    blocks.append(
        f"""SUMMARY ID: {sid}

ORIGINAL SUMMARY:
{data["summary"]}

CLAIMS TO REMOVE:
{claims}"""
    )

joined_blocks = "\n\n".join(blocks)

# -------- PROMPT --------
prompt = f"""
You are editing research summaries.

TASK:
For EACH summary, rewrite it so that the listed claims are NO LONGER PRESENT.

IMPORTANT:
- You may rewrite sentences if necessary to remove the listed ideas
- Keep all other information as close as possible to the original
- Preserve tone, scope, and level of detail
- Do NOT add new facts or interpretations
- Do NOT summarize
- Do NOT explain

OUTPUT FORMAT (JSON ONLY):
{{
  "rewritten": {{
    "S1": "rewritten summary text",
    "S2": "rewritten summary text"
  }}
}}

SUMMARIES:
{joined_blocks}
""".strip()

# -------- RUN MODEL --------
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    max_tokens=1500,
)

print("\n====== RAW LLM OUTPUT ======\n")
print(response.choices[0].message.content)

print("\n====== PARSED JSON ======\n")
try:
    data = json.loads(response.choices[0].message.content)
    print(json.dumps(data, indent=2))
except Exception as e:
    print("JSON PARSE FAILED:", e)
