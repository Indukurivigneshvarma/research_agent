# agent/controller.py

from agent.planner import generate_sub_questions, plan_search
from agent.research_state import ResearchState
from agent.research_trace import ResearchTrace
from agent.evidence_scorer import score_evidence
from agent.stagnation_checker import is_stagnating
from agent.sufficiency_checker import check_sufficiency
from agent.research_mode import detect_research_mode

from tools.web_search import web_search
from tools.content_fetcher import fetch_content
from tools.source_evaluator import evaluate_sources

from processing.cleaner import clean_text
from processing.summarizer import summarize_article
from processing.iteration_summarizer import summarize_iteration
from processing.synthesizer import synthesize

from report.citation_manager import CitationManager
from output.pdf_generator import generate_pdf


MAX_ARTICLES_PER_ITERATION = 5   # 🔥 KEY FIX


def run_agent(query: str):
    trace = ResearchTrace()
    citations = CitationManager()
    state = ResearchState(query)

    research_mode = detect_research_mode(query)
    trace.steps.append(f"Research mode: {research_mode}")

    executed_queries = []
    previous_iteration_summary = ""

    sub_questions = generate_sub_questions(query)
    state.add_sub_questions(sub_questions)

    MAX_ITERATIONS = 5
    MIN_TOTAL_WORDS = 1500

    while state.iteration < MAX_ITERATIONS:
        state.iteration += 1
        trace.steps.append(f"\n=== Iteration {state.iteration} ===")

        article_summaries = []

        search_queries = plan_search(
            query=query,
            sub_questions=sub_questions,
            iteration=state.iteration - 1
        )

        for sq in search_queries:
            trace.add_iteration(state.iteration, sq)
            executed_queries.append(sq)

            if is_stagnating(executed_queries):
                trace.mark_complete("Search stagnation detected")
                break

            results = web_search(sq)
            citations.collect(results)

            pages = fetch_content(results)
            pages = evaluate_sources(pages, research_mode)

            for page in pages[:MAX_ARTICLES_PER_ITERATION]:
                cleaned = clean_text(page["content"])
                credibility = page.get("credibility_score", 0.5)

                if len(cleaned) < 300:
                    continue

                if credibility < 0.3:
                    continue

                summary = summarize_article(
                    article_text=cleaned,
                    credibility=credibility,
                    research_mode=research_mode
                )

                if summary:
                    article_summaries.append(summary)

        if not article_summaries:
            continue

        evidence_objects = score_evidence(article_summaries)
        iteration_summary = summarize_iteration(evidence_objects)

        if not iteration_summary:
            continue

        state.iteration_summaries.append(iteration_summary)

        sufficient, meta = check_sufficiency(
            previous_iteration_summary,
            iteration_summary
        )

        total_words = sum(
            len(s.split()) for s in state.iteration_summaries
        )

        trace.steps.append(
            f"Sufficiency → similarity={meta.get('similarity')}, "
            f"new_words={meta.get('new_words')}, "
            f"total_words={total_words}"
        )

        if sufficient and total_words >= MIN_TOTAL_WORDS:
            trace.mark_complete("Information sufficiency reached")
            break

        previous_iteration_summary = iteration_summary

    report_text = synthesize(
        query=query,
        article_summaries=state.iteration_summaries,
        research_mode=research_mode
    )

    pdf_path = generate_pdf(
        report_text,
        citations.format_references()
    )

    return pdf_path, trace.export()
