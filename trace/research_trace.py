from typing import List, Dict, Optional


class ResearchTrace:
    def __init__(self):
        self.lines: List[str] = []

    # ==================================================
    # Core helpers (Gradio-friendly)
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
    # Sub-queries
    # ==================================================

    def log_subqueries(self, subqueries: List[str]):
        self.section("GENERATED SUB-QUERIES")
        for i, q in enumerate(subqueries, 1):
            self.log(f"Q{i}: {q}")

    # ==================================================
    # Vector search + cross encoder
    # ==================================================

    def log_vector_search(
        self,
        qkey: str,
        query: str,
        retrieved: Dict[str, str],
    ):
        self.section(f"VECTOR SEARCH — {qkey}")
        self.log(f"Sub-query: {query}")

        if not retrieved:
            self.log("No vector candidates found.")
            return

        self.log("Top-10 retrieved query texts:")
        for vs_id, text in retrieved.items():
            self.log(f"{vs_id}: {text}")

    def log_cross_encoder(
        self,
        qkey: str,
        reranked: Dict[str, str],
    ):
        self.section(f"CROSS ENCODER — {qkey}")

        if not reranked:
            self.log("No candidates after cross-encoder rerank.")
            return

        self.log("Top-5 after cross-encoder rerank:")
        for vs_id, text in reranked.items():
            self.log(f"{vs_id}: {text}")

    # ==================================================
    # Intent selection
    # ==================================================

    def log_intent_selection(self, selections: Dict[str, Optional[str]]):
        self.section("LLM INTENT SELECTION")
        for qkey, vs_id in selections.items():
            if vs_id:
                self.log(f"{qkey}: selected {vs_id}")
            else:
                self.log(f"{qkey}: no reusable intent selected")

    # ==================================================
    # Web ingestion / reuse
    # ==================================================

    def log_web_ingestion_start(self, qkey: str, query: str):
        self.section(f"WEB INGESTION — {qkey}")
        self.log(f"Sub-query: {query}")

    def log_web_candidate(
        self,
        idx: int,
        url: str,
        score: Optional[float],
        raw_len: Optional[int],
    ):
        self.log(
            f"[Candidate {idx}] URL: {url} | "
            f"score={score} | raw_len={raw_len}"
        )

    def log_rejection(self, reason: str):
        self.log(f"Rejected → {reason}")

    # ==================================================
    # Metadata + summary generation
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
        self.log(f"Published date: {published}")
        self.log(f"Domain: {domain}")
        self.log(f"Retrieved date: {retrieved}")

    def log_summary_generation(self, provider: str):
        self.section("SUMMARY GENERATION")
        self.log(f"LLM provider: {provider.upper()}")

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
        self.section("TOTAL SCORE COMPUTATION")
        for s in summaries:
            self.log(
                f"{s['id']}: total={s['total_score']} "
                f"(base={s.get('credibility_score', 0)} "
                f"+ agreement={s.get('agreement_score', 0)})"
            )

    # ==================================================
    # Conflict detection + resolution
    # ==================================================

    def log_conflicts(self, conflicts: Dict):
        self.section("CONFLICTS DETECTED")

        items = conflicts.get("conflicts", [])
        if not items:
            self.log("No factual conflicts detected.")
            return

        for i, c in enumerate(items, 1):
            ids = ", ".join(c["ids"])
            self.log(f"Conflict {i}: {ids}")
            self.log(f"  Claim A: {c['claim_a']}")
            self.log(f"  Claim B: {c['claim_b']}")

    def log_conflict_resolutions(self, removals: Dict[str, List[str]]):
        self.section("CONFLICT RESOLUTIONS")

        if not removals:
            self.log("No summaries required rewriting.")
            return

        for sid, claims in removals.items():
            self.log(f"{sid}: removed {len(claims)} conflicting claim(s)")

    # ==================================================
    # Summary rewriting
    # ==================================================

    def log_summary_rewrite(self, sid: str, old: str, new: str):
        self.section(f"SUMMARY REWRITE — {sid}")
        self.log("OLD SUMMARY:")
        self.log(old)
        self.log("")
        self.log("NEW SUMMARY:")
        self.log(new)

    # ==================================================
    # Report generation
    # ==================================================

    def log_report_generation(self, pdf_path: str):
        self.section("REPORT GENERATION")
        self.log("Academic report generated successfully.")
        self.log(f"PDF file: {pdf_path}")

    # ==================================================
    # Completion
    # ==================================================

    def log_pipeline_complete(self):
        self.section("PIPELINE COMPLETE")
