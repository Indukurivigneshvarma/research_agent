from typing import List, Dict
import time
import json

from config import (
    MIN_RAW_CHARS,
    MAX_RAW_CHARS,
    MODES,
    VECTOR_TOP_K,
    CROSS_TOP_K,
)

# --------------------------------------------------
# Query planning
# --------------------------------------------------
from query.subqueries import generate_initial_subqueries
from query.intent_selector import select_best_intents
from query.research_plan import generate_research_plan
from query.coverage_refiner import refine_queries

# --------------------------------------------------
# Retrieval
# --------------------------------------------------
from retrieval.web_search import search_web
from retrieval.tavily_client import tavily_extract
from retrieval.vector_search import VectorSearcher
from retrieval.cross_encoder import CrossEncoderReranker

# --------------------------------------------------
# Ingestion
# --------------------------------------------------
from ingestion.metadata_extractor import extract_metadata
from ingestion.summary_generator import generate_summary

# --------------------------------------------------
# Scoring & analytics
# --------------------------------------------------
from scoring.summary_scorer import compute_summary_score
from scoring.agreement_scorer import compute_agreement_scores
from analytics.agreement_detector import detect_agreements
from analytics.conflict_detector import detect_conflicts
from analytics.conflict_resolver import resolve_conflicts
from analytics.summary_rewriter import rewrite_summaries

# --------------------------------------------------
# Vector store
# --------------------------------------------------
from vector_store.upsert import upsert_summaries

# --------------------------------------------------
# Report generation
# --------------------------------------------------
from report.citations import build_references
from report.headings import generate_title_and_headings
from report.writer import write_report
from report.pdf_generator import generate_pdf

# --------------------------------------------------
# Utils / trace
# --------------------------------------------------
from utils.dates import normalize_date, today_iso
from trace.research_trace import ResearchTrace

# --------------------------------------------------
# Embeddings
# --------------------------------------------------
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Evaluator
# --------------------------------------------------
from evaluation.report_evaluator import evaluate_report


def run_pipeline(
    user_query: str,
    mode: str,
    vector_client,
    trace: ResearchTrace,
):
    summaries: List[Dict] = []
    seen_urls = set()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vector_searcher = VectorSearcher(vector_client)
    cross_encoder = CrossEncoderReranker()

    mode_cfg = MODES.get(mode)
    if not mode_cfg:
        raise ValueError(f"Unknown mode: {mode}")

    max_iterations = mode_cfg["iterations"]
    queries_per_iteration = mode_cfg["queries_per_iteration"]

    # ==================================================
    # 0. Research plan (ONCE)
    # ==================================================

    research_plan = generate_research_plan(user_query)

    trace.log_research_plan(
        goal=research_plan["goal"],
        dimensions=research_plan["dimensions"],
    )

    sid_counter = 1
    current_queries: List[str] = []

    # ==================================================
    # 1. Iterative discovery (collect summaries ONLY)
    # ==================================================

    for iteration in range(1, max_iterations + 1):

        # ---------- Sub-query source ----------
        if iteration == 1:
            current_queries = generate_initial_subqueries(
                user_query=user_query,
                research_goal=research_plan["goal"],
                dimensions=research_plan["dimensions"],
                n_queries=queries_per_iteration,
            )
        else:
            summary_map = {s["id"]: s["summary"] for s in summaries}
            current_queries = refine_queries(
                research_plan=research_plan,
                summaries=summary_map,
                n_queries=queries_per_iteration,
            )

            trace.log_coverage_refinement(
                iteration=iteration,
                new_queries=current_queries,
            )

        trace.log_iteration_start(iteration, current_queries)

        subq_map = {
            f"Q{idx+1}": q
            for idx, q in enumerate(current_queries)
        }

        retrieved_candidates: Dict[str, Dict[str, Dict]] = {}

        # ==================================================
        # Vector search + cross encoder
        # ==================================================

        for qkey, qtext in subq_map.items():
            query_embedding = embedder.encode(qtext).tolist()

            hits = vector_searcher.search(
                query_embedding=query_embedding,
                top_k=VECTOR_TOP_K,
            )

            # ---- URL de-duplication (vector level) ----
            filtered_hits = []
            for h in hits:
                url = h.get("url")
                if url in seen_urls:
                    trace.log_url_skip(
                        url=url,
                        reason="already used earlier (vector search)",
                    )
                    continue
                filtered_hits.append(h)

            hits = filtered_hits

            if not hits:
                continue

            vs_map = {
                f"VS_{i:02d}": h
                for i, h in enumerate(hits, 1)
            }

            trace.log_vector_search(
                qkey=qkey,
                query=qtext,
                retrieved={
                    vs_id: data["query_text"]
                    for vs_id, data in vs_map.items()
                },
            )

            reranked = cross_encoder.rerank(
                query=qtext,
                candidates=[
                    {"vs_id": vs_id, **data}
                    for vs_id, data in vs_map.items()
                ],
                top_k=CROSS_TOP_K,
            )

            reranked_map = {
                c["vs_id"]: c
                for c in reranked
            }

            trace.log_cross_encoder(
                qkey=qkey,
                reranked={
                    vs_id: data["query_text"]
                    for vs_id, data in reranked_map.items()
                },
            )

            retrieved_candidates[qkey] = reranked_map

        # ==================================================
        # Intent selection
        # ==================================================

        candidates_for_llm = {
            qkey: {
                vs_id: data["query_text"]
                for vs_id, data in cands.items()
            }
            for qkey, cands in retrieved_candidates.items()
            if cands
        }

        selected_intents = select_best_intents(
            subqueries=list(subq_map.values()),
            candidates_by_subquery=candidates_for_llm,
        )

        trace.log_intent_selection(selected_intents)

        # ==================================================
        # Reuse OR web ingestion
        # ==================================================

        for qkey, qtext in subq_map.items():
            vs_id = selected_intents.get(qkey)

            # ---------- VECTOR REUSE ----------
            if vs_id and qkey in retrieved_candidates:
                record = retrieved_candidates[qkey].get(vs_id)
                url = record.get("url") if record else None

                if record and url not in seen_urls:
                    r = record.copy()
                    r["id"] = f"S{sid_counter}"
                    sid_counter += 1

                    r["credibility_score"] = compute_summary_score(r)
                    summaries.append(r)
                    seen_urls.add(url)

                    trace.log_intent_reuse(r["id"], url)
                    continue

                if url in seen_urls:
                    trace.log_url_skip(
                        url=url,
                        reason="already used earlier (intent reuse)",
                    )

            # ---------- WEB INGESTION ----------
            trace.log_web_ingestion_start(qkey, qtext)

            results = search_web(qtext, max_results=3)
            accepted = False

            for idx, r in enumerate(results, 1):
                url = r.get("url")

                if not url:
                    trace.log_rejection("missing URL")
                    continue

                if url in seen_urls:
                    trace.log_url_skip(
                        url=url,
                        reason="already used earlier (web search)",
                    )
                    continue

                extracted = tavily_extract(url)
                raw_text = extracted.get("raw_text") if extracted else None

                trace.log_web_candidate(
                    idx,
                    url,
                    r.get("score"),
                    len(raw_text) if raw_text else None,
                )

                if not raw_text or len(raw_text) < MIN_RAW_CHARS:
                    trace.log_rejection("insufficient content")
                    continue

                if len(raw_text) > MAX_RAW_CHARS:
                    raw_text = raw_text[:MAX_RAW_CHARS]

                meta = extract_metadata(url)

                provider = "groq" if mode == "quick" else "openrouter"
                summary_text = generate_summary(raw_text, provider)

                record = {
                    "id": f"S{sid_counter}",
                    "query_text": qtext,
                    "summary": summary_text,
                    "url": url,
                    "domain": extracted.get("domain"),
                    "author": meta.get("author"),
                    "venue_type": None,
                    "date_published": normalize_date(meta.get("date_published")),
                    "date_retrieved": today_iso(),
                }

                sid_counter += 1

                record["credibility_score"] = compute_summary_score(record)
                record["embedding"] = embedder.encode(qtext).tolist()

                summaries.append(record)
                seen_urls.add(url)
                upsert_summaries(vector_client, [record])

                trace.log_metadata(
                    record["author"],
                    record["date_published"],
                    record["domain"],
                    record["date_retrieved"],
                )

                trace.log_summary_generation(provider)
                trace.log_storage(record["id"], url)

                accepted = True
                break

            if not accepted:
                trace.log("No valid web source accepted.")

        trace.log_iteration_end(iteration)

    # ==================================================
    # 2. Agreement + conflict + report (ONCE)
    # ==================================================

    if len(summaries) < 2:
        trace.log_pipeline_complete()
        return summaries, trace.render(), None, None, None

    agreement_map = detect_agreements(
        [{"id": s["id"], "summary": s["summary"]} for s in summaries]
    )

    trace.log_agreement_map(agreement_map)

    agreement_scores = compute_agreement_scores(agreement_map)

    for s in summaries:
        s["agreement_score"] = agreement_scores.get(s["id"], 0)
        s["total_score"] = (
            s.get("credibility_score", 0) + s["agreement_score"]
        )

    trace.log_total_scores(summaries)

    conflicts = detect_conflicts(
        [{"id": s["id"], "summary": s["summary"]} for s in summaries]
    )

    trace.log_conflicts(conflicts)

    removals = resolve_conflicts(
        conflicts,
        {s["id"]: s["total_score"] for s in summaries},
    )

    trace.log_conflict_resolutions(removals)

    if removals:
        rewrite_plan = {
            sid: {
                "summary": next(
                    s for s in summaries if s["id"] == sid
                )["summary"],
                "remove_claims": claims,
            }
            for sid, claims in removals.items()
        }

        rewritten = rewrite_summaries(rewrite_plan)

        for s in summaries:
            if s["id"] in rewritten:
                trace.log_summary_rewrite(
                    s["id"],
                    s["summary"],
                    rewritten[s["id"]],
                )
                s["summary"] = rewritten[s["id"]]

    # ==================================================
    # 3. Report generation
    # ==================================================

    references = build_references(summaries)

    title_headings = generate_title_and_headings(
        user_query=user_query,
        summaries=[s["summary"] for s in summaries],
    )

    report_text = write_report(
        title=title_headings["title"],
        headings=title_headings["headings"],
        summaries={s["id"]: s["summary"] for s in summaries},
        references=references,
    )

    pdf_path = generate_pdf(
        report_text=report_text,
        output_path=f"report_{int(time.time())}.pdf",
    )

    trace.log_report_generation(pdf_path)

    # ==================================================
    # 4. Evaluation
    # ==================================================

    evaluation = evaluate_report(
        user_query=user_query,
        research_plan=research_plan, 
        report_text=report_text,
        summaries={s["id"]: s["summary"] for s in summaries},
        headings=title_headings["headings"],
        references=references,
    )

    trace.section("SELF-EVALUATION")
    trace.log(json.dumps(evaluation, indent=2))

    trace.log_pipeline_complete()

    return summaries, trace.render(), report_text, pdf_path, evaluation
