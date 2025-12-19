from agent.planner import plan_search
from agent.sufficiency_checker import is_sufficient
from agent.research_trace import ResearchTrace

from tools.web_search import web_search
from tools.content_fetcher import fetch_content
from tools.source_evaluator import evaluate_sources

from processing.cleaner import clean_text
from processing.summarizer import summarize_article
from processing.synthesizer import synthesize

from report.citation_manager import CitationManager
from report.confidence_report import build_confidence_report
from output.pdf_generator import generate_pdf


def run_agent(query: str):
    trace = ResearchTrace()
    citations = CitationManager()

    knowledge = []          # stores article-level summaries
    iteration = 0
    MAX_ITERATIONS = 2      # safe, realistic, defendable

    while iteration < MAX_ITERATIONS:
        iteration += 1

        # 1️⃣ Plan search
        search_query = plan_search(query, iteration)
        trace.add_iteration(iteration, search_query)

        # 2️⃣ Web search
        results = web_search(search_query)
        citations.collect(results)

        # 3️⃣ Fetch article content
        pages = fetch_content(results)

        if not pages:
            trace.mark_complete("No usable content retrieved")
            break

        # 4️⃣ Evaluate credibility
        scored_pages = evaluate_sources(pages)

        # 5️⃣ Summarize EACH article independently
        for page in scored_pages:
            cleaned_text = clean_text(page["content"])

            if len(cleaned_text) < 300:
                continue  # skip thin or low-value pages

            article_summary = summarize_article(
                article_text=cleaned_text,
                source_url=page["url"]
            )

            knowledge.append(article_summary)

        # 6️⃣ Autonomous sufficiency decision
        if is_sufficient(query, knowledge):
            trace.mark_complete(
                "Information sufficient based on coverage and saturation"
            )
            break

    # 7️⃣ Final synthesis from summaries only
    report_text = synthesize(query, knowledge)

    # 8️⃣ Confidence & limitations
    confidence = build_confidence_report(knowledge)

    # 9️⃣ Generate PDF
    pdf_path = generate_pdf(
        report_text,
        citations.format_references(),
        confidence
    )

    return pdf_path, trace.export()
