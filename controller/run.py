from typing import List, Dict
import time
import json

from config import MIN_RAW_CHARS, MAX_RAW_CHARS

# --------------------------------------------------
# Query planning
# --------------------------------------------------
from query.subqueries import generate_subqueries
from query.intent_selector import select_best_intents

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
from report.citations import build_cited_summaries, build_references
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


VECTOR_TOP_K = 10
CROSS_TOP_K = 5


def run_pipeline(
    user_query: str,
    mode: str,
    vector_client,
    trace: ResearchTrace,
):
    summaries: List[Dict] = []

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vector_searcher = VectorSearcher(vector_client)
    cross_encoder = CrossEncoderReranker()

    # ==================================================
    # 1. Sub-query generation
    # ==================================================

    subqueries = generate_subqueries(user_query, mode)
    trace.log_subqueries(subqueries)

    subq_map = {f"Q{i+1}": q for i, q in enumerate(subqueries)}

    # ==================================================
    # 2. Vector search + cross encoder
    # ==================================================

    retrieved_candidates: Dict[str, Dict[str, Dict]] = {}

    for qkey, qtext in subq_map.items():
        query_embedding = embedder.encode(qtext).tolist()

        hits = vector_searcher.search(
            query_embedding=query_embedding,
            top_k=VECTOR_TOP_K,
        )

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
    # 3. Intent selection
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
        subqueries=subqueries,
        candidates_by_subquery=candidates_for_llm,
    )

    trace.log_intent_selection(selected_intents)

    # ==================================================
    # 4. Reuse OR web ingestion
    # ==================================================

    sid_counter = 1

    for qkey, qtext in subq_map.items():
        vs_id = selected_intents.get(qkey)

        # ---------- REUSE ----------
        if vs_id and qkey in retrieved_candidates:
            record = retrieved_candidates[qkey].get(vs_id)
            if record:
                r = record.copy()
                r["id"] = f"S{sid_counter}"
                sid_counter += 1

                r["credibility_score"] = compute_summary_score(r)
                summaries.append(r)
                continue

        # ---------- WEB INGESTION ----------
        trace.log_web_ingestion_start(qkey, qtext)

        results = search_web(qtext, max_results=3)
        accepted = False

        for idx, r in enumerate(results, 1):
            url = r.get("url")
            extracted = tavily_extract(url)
            raw_text = extracted.get("raw_text") if extracted else None

            trace.log_web_candidate(
                idx,
                url,
                r.get("score"),
                len(raw_text) if raw_text else None,
            )

            if not raw_text:
                trace.log_rejection("extraction failed")
                continue

            if len(raw_text) < MIN_RAW_CHARS:
                trace.log_rejection("too short")
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
            upsert_summaries(vector_client, [record])

            trace.log_metadata(
                record["author"],
                record["date_published"],
                record["domain"],
                record["date_retrieved"],
            )

            trace.log_summary_generation(provider)

            accepted = True
            break

        if not accepted:
            trace.log("No valid web source accepted.")

    # ==================================================
    # 5. Agreement + conflict pipeline
    # ==================================================

    if len(summaries) < 2:
        trace.log_pipeline_complete()
        return summaries, trace.render(), None, None

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
    # 6. REPORT GENERATION
    # ==================================================

    trace.section("REPORT GENERATION")

    # Build summary map for writer: { "S1": "summary text", ... }
    summary_map = {
        s["id"]: s["summary"]
        for s in summaries
    }
    # Deterministic references
    references = build_references(summaries)

    # Generate title + headings
    title_headings = generate_title_and_headings(
        user_query=user_query,
        summaries=list(summary_map.values()),
    )

    # Write academic report
    report_text = write_report(
        title=title_headings["title"],
        headings=title_headings["headings"],
        summaries=summary_map,          # ✅ CORRECT
        references=references,
    )
    # Generate PDF (auto-versioned for Windows safety)
    pdf_path = generate_pdf(
        report_text=report_text,
    output_path=f"report_{int(time.time())}.pdf",
   )

    trace.log_report_generation(pdf_path)

    trace.log_pipeline_complete()


    evaluation = evaluate_report(
        user_query=user_query,
        report_text=report_text,
        summaries=summary_map,
        headings=title_headings["headings"],
        references=references,
    )
    
    trace.log_pipeline_complete()

    return summaries, trace.render(), report_text, pdf_path, evaluation