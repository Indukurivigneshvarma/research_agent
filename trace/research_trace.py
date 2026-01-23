from typing import List, Dict, Optional


class ResearchTrace:
    def __init__(self):
        self.lines: List[str] = []

    # ==================================================
    # Core helpers
    # ==================================================

    def section(self, title: str):
        self.lines.append("")
        self.lines.append("=" * 70)
        self.lines.append(title)
        self.lines.append("=" * 70)

    def log(self, text: str = ""):
        self.lines.append(str(text))

    def render(self) -> str:
        return "\n".join(self.lines)

    # ==================================================
    # Research plan
    # ==================================================

    def log_research_plan(self, goal: str, dimensions: List[str]):
        self.section("RESEARCH PLAN")
        self.log("Goal:")
        self.log(goal)
        self.log("")
        self.log("Dimensions:")
        for d in dimensions:
            self.log(f"- {d}")

    # ==================================================
    # Iterations
    # ==================================================

    def log_iteration_start(self, iteration: int, subqueries: List[str]):
        self.section(f"DISCOVERY ITERATION {iteration}")
        self.log("Sub-queries:")
        for q in subqueries:
            self.log(f"- {q}")

    def log_iteration_end(self, iteration: int):
        self.log("")
        self.log(f"End of discovery iteration {iteration}")

    # ==================================================
    # Coverage refinement
    # ==================================================

    def log_coverage_refinement(
        self,
        iteration: int,
        new_queries: List[str],
    ):
        self.section(f"COVERAGE REFINEMENT — ITERATION {iteration}")

        if not new_queries:
            self.log("No new sub-queries generated.")
            return

        self.log("New sub-queries:")
        for q in new_queries:
            self.log(f"- {q}")

    # ==================================================
    # Vector search + reranking
    # ==================================================

    def log_vector_search(
        self,
        qkey: str,
        query: str,
        retrieved: Dict[str, str],
    ):
        self.section(f"VECTOR SEARCH — {qkey}")
        self.log(f"Query: {query}")

        if not retrieved:
            self.log("No vector candidates found.")
            return

        for vs_id, text in retrieved.items():
            self.log(f"{vs_id}: {text}")

    def log_cross_encoder(
        self,
        qkey: str,
        reranked: Dict[str, str],
    ):
        self.section(f"CROSS ENCODER RERANK — {qkey}")

        if not reranked:
            self.log("No candidates after reranking.")
            return

        for vs_id, text in reranked.items():
            self.log(f"{vs_id}: {text}")

    # ==================================================
    # Intent selection
    # ==================================================

    def log_intent_selection(self, selections: Dict[str, Optional[str]]):
        self.section("LLM INTENT SELECTION")

        for qkey, vs_id in selections.items():
            if vs_id:
                self.log(f"{qkey}: selected intent {vs_id}")
            else:
                self.log(f"{qkey}: no reusable intent")

    # ==================================================
    # Reuse / URL filtering
    # ==================================================

    def log_intent_reuse(self, sid: str, url: str):
        self.log(f"Reused summary {sid} from vector DB (URL: {url})")

    def log_url_skip(self, url: str, reason: str):
        self.log(f"Skipped URL: {url} → {reason}")

    # ==================================================
    # Web ingestion
    # ==================================================

    def log_web_ingestion_start(self, qkey: str, query: str):
        self.section(f"WEB INGESTION — {qkey}")
        self.log(f"Query: {query}")

    def log_web_candidate(
        self,
        idx: int,
        url: str,
        score: Optional[float],
        raw_len: Optional[int],
    ):
        self.log(
            f"[Candidate {idx}] URL={url} | score={score} | raw_len={raw_len}"
        )

    def log_rejection(self, reason: str):
        self.log(f"Rejected → {reason}")

    # ==================================================
    # Metadata + summary
    # ==================================================

    def log_metadata(
        self,
        author: Optional[str],
        published: Optional[str],
        domain: Optional[str],
        retrieved: str,
    ):
        self.section("METADATA")
        self.log(f"Author: {author}")
        self.log(f"Published: {published}")
        self.log(f"Domain: {domain}")
        self.log(f"Retrieved: {retrieved}")

    def log_summary_generation(self, provider: str):
        self.section("SUMMARY GENERATION")
        self.log(f"Provider: {provider.upper()}")

    def log_storage(self, sid: str, url: str):
        self.log(f"Stored summary {sid} in vector DB (URL: {url})")

    # ==================================================
    # Agreement + scoring
    # ==================================================

    def log_agreement_map(self, agreement_map: Dict):
        self.section("AGREEMENT MAP")

        if not agreement_map:
            self.log("No agreements detected.")
            return

        for src, relations in agreement_map.items():
            for tgt, label in relations.items():
                self.log(f"{src} → {tgt}: {label}")

    def log_total_scores(self, summaries: List[Dict]):
        self.section("TOTAL SCORES")
        for s in summaries:
            self.log(
                f"{s['id']}: total={s['total_score']} "
                f"(credibility={s.get('credibility_score', 0)}, "
                f"agreement={s.get('agreement_score', 0)})"
            )

    # ==================================================
    # Conflicts
    # ==================================================

    def log_conflicts(self, conflicts: Dict):
        self.section("CONFLICTS DETECTED")

        items = conflicts.get("conflicts", [])
        if not items:
            self.log("No conflicts detected.")
            return

        for i, c in enumerate(items, 1):
            self.log(f"Conflict {i}: {', '.join(c['ids'])}")
            self.log(f"  Claim A: {c['claim_a']}")
            self.log(f"  Claim B: {c['claim_b']}")

    def log_conflict_resolutions(self, removals: Dict[str, List[str]]):
        self.section("CONFLICT RESOLUTION")

        if not removals:
            self.log("No summaries rewritten.")
            return

        for sid, claims in removals.items():
            self.log(f"{sid}: removed {len(claims)} claim(s)")

    # ==================================================
    # Rewriting
    # ==================================================

    def log_summary_rewrite(self, sid: str, old: str, new: str):
        self.section(f"SUMMARY REWRITE — {sid}")
        self.log("OLD:")
        self.log(old)
        self.log("")
        self.log("NEW:")
        self.log(new)

    # ==================================================
    # Report
    # ==================================================

    def log_report_generation(self, pdf_path: str):
        self.section("REPORT GENERATION")
        self.log(f"PDF generated: {pdf_path}")

    # ==================================================
    # Completion
    # ==================================================

    def log_pipeline_complete(self):
        self.section("PIPELINE COMPLETE")
