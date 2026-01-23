import os
import json
import cohere

# ------------------------------
# Configure Cohere
# ------------------------------
co = cohere.Client(os.getenv("COHERE_API_KEY"))


MODEL = "command-a-03-2025"

# ------------------------------
# JSON CLEANER (CRITICAL FIX)
# ------------------------------
def clean_llm_json(text: str) -> str:
    """
    Removes ```json fences and stray formatting from LLM output
    so it can be safely parsed by json.loads().
    """
    if not text:
        return ""

    text = text.strip()

    # Remove ```json or ``` fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]

    return text.strip()


# ---- Test summaries ----
summaries = [
    {
        "id": "S1",
        "summary": "The software industry is at an inflection point, with AI coding evolving from autocomplete to autonomous task execution. Companies are increasingly prioritizing profitability over growth, experienced hires over fresh graduates, and smaller teams equipped with better tools. A new generation of developers is entering the workforce with a different mindset, valuing career stability, questioning hustle culture, and relying heavily on AI assistance. Multiple scenarios could shape the future of software engineering through 2026: junior developer hiring could collapse as AI automates entry-level tasks, or rebound as software demand expands across all industries. A Harvard study reports a 9–10% decline in junior developer employment within six quarters of generative AI adoption, while senior employment remains stable. Big Tech firms have hired 50% fewer fresh graduates over the past three years, yet an alternative projection suggests a 15% growth in software jobs from 2024 to 2034 if AI unlocks broader demand. The long-term risk is a leadership gap within 5–10 years if fewer juniors progress into senior roles. Adaptation strategies emphasize AI proficiency, cross-domain versatility, and human oversight skills, with surveys showing that 84% of developers already use AI assistance regularly."
    },
    {
        "id": "S2",
        "summary": "PwC provides a broad portfolio of professional services spanning audit and assurance, artificial intelligence, consulting, cybersecurity, risk and regulatory compliance, digital transformation, and workforce solutions. The firm serves a wide range of industries including financial services, healthcare, energy, manufacturing, retail, telecommunications, technology, media, and public sector organizations. PwC also publishes research and industry insights such as the Global Digital Trust Insights Survey and produces supporting resources including podcasts, webcasts, case studies, and analytical tools. While the content highlights PwC’s scale, service breadth, and industry reach, it does not present quantitative evidence on outcomes, performance improvements, or adoption metrics related specifically to AI-driven software engineering productivity or labor-market impacts."
    },
    {
        "id": "S3",
        "summary": "Generative Artificial Intelligence (GenAI) is built on large-scale foundation models such as Large Language Models and Generative Adversarial Networks, enabling the generation of human-like text, images, and software code. Over the past decade, GenAI has evolved from rule-based systems into advanced neural architectures capable of supporting software engineers across the development lifecycle. Current applications include requirements analysis, code generation, debugging, test case creation, and documentation. Multiple studies report measurable productivity gains: GenAI accelerates manual work, jump-starts code drafts, speeds up updates, and improves developers’ ability to solve novel problems. Evidence includes a Cornell University case study showing programmers using GitHub Copilot completed tasks 55.8% faster, and broader industry research suggesting a 20–45% potential impact on global software engineering spending. For greenfield projects, GenAI-supported workflows have demonstrated 20–40% improvements in code quality. While productivity benefits in design and planning stages are less conclusive, GenAI is expected to democratize software development, accelerate digital transformation across traditional sectors, and shift human effort toward higher-value engineering tasks."
    },
    {
        "id": "S4",
        "summary": "The rapid adoption of generative AI since late 2022 has produced a structural shift in the software engineering labor market, particularly affecting early-career roles. A Stanford study reports a 13% relative decline in employment for engineers aged 22–25 in AI-exposed occupations, while senior roles remain stable or continue to grow. This divergence is attributed to AI’s effectiveness at automating tasks based on codified knowledge—such as boilerplate coding and routine implementation—while struggling with tacit knowledge developed through experience. As a result, traditional entry-level pathways into software engineering are narrowing, with new graduates accounting for only 7% of new hires at major technology firms, down 25% from 2023 levels. The market increasingly rewards AI-centric skills, with an observed salary premium of approximately 18% for engineers proficient in AI-assisted development. The emerging baseline competency for software engineers is the ability to orchestrate, validate, and debug AI-generated outputs, signaling a shift toward “human-on-the-loop” roles as agentic AI systems become more prevalent."
    },
]

# Build input block
block = "\n\n".join(
    f"{s['id']}:\n{s['summary']}"
    for s in summaries
)

prompt = f"""
Analyze cross-source agreement between the following summaries.

Allowed labels ONLY:
- strongly_supports
- partially_supports
- independent

Rules:
- Compare EACH summary against EVERY OTHER summary
- Directional (A→B may differ from B→A)
- NO explanations
- Return JSON ONLY

SUMMARIES:
{block}

OUTPUT FORMAT:
{{
  "S1": {{ "S2": "strongly_supports", "S3": "partially_supports", "S4": "independent" }},
  "S2": {{ "S1": "partially_supports", "S3": "independent", "S4": "partially_supports" }}
}}
""".strip()


# ------------------------------
# Call Cohere
# ------------------------------
response = co.chat(
    model=MODEL,
    message=prompt,
    temperature=0,
)

raw_text = response.text

print("\n====== RAW LLM OUTPUT ======\n")
print(repr(raw_text))

cleaned = clean_llm_json(raw_text)

print("\n====== CLEANED JSON TEXT ======\n")
print(cleaned)

print("\n====== PARSED JSON ======\n")
try:
    data = json.loads(cleaned)
    print(json.dumps(data, indent=2))
except Exception as e:
    print("JSON PARSE FAILED:", e)
