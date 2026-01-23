import json
import gradio as gr

from vector_store.client import VectorStoreClient
from controller.run import run_pipeline
from trace.research_trace import ResearchTrace


# --------------------------------------------------
# Vector store (persistent)
# --------------------------------------------------

VECTOR_CLIENT = VectorStoreClient(
    persist_dir="vector_data",
    embedding_dim=384,
)


# --------------------------------------------------
# Helper: extract research plan from trace
# --------------------------------------------------

def extract_research_plan(trace_text: str) -> str:
    if not trace_text:
        return ""

    lines = trace_text.splitlines()
    plan_lines = []
    in_plan = False
    content_started = False

    for line in lines:
        if line.strip() == "RESEARCH PLAN":
            in_plan = True
            continue

        if in_plan:
            # Skip separator lines before content starts
            if line.strip().startswith("=") and not content_started:
                continue

            # Stop when next section separator appears AFTER content
            if line.strip().startswith("=") and content_started:
                break

            if line.strip():
                content_started = True

            plan_lines.append(line)

    return "\n".join(plan_lines).strip()



# --------------------------------------------------
# Gradio handler
# --------------------------------------------------

def run_app(query: str, mode: str):
    trace = ResearchTrace()

    (
        summaries,
        trace_text,
        report_text,
        pdf_path,
        evaluation,
    ) = run_pipeline(
        user_query=query,
        mode=mode,
        vector_client=VECTOR_CLIENT,
        trace=trace,
    )

    # -----------------------------
    # Summaries output
    # -----------------------------

    if summaries:
        summaries_text = "\n\n".join(
            f"{s['id']} (score={s.get('total_score', 'NA')}):\n{s['summary']}"
            for s in summaries
        )
    else:
        summaries_text = ""

    # -----------------------------
    # Research plan output
    # -----------------------------

    plan_text = extract_research_plan(trace_text)

    # -----------------------------
    # Evaluation output (safe JSON)
    # -----------------------------

    evaluation_text = ""
    if evaluation:
        try:
            evaluation_text = json.dumps(evaluation, indent=2)
        except Exception:
            evaluation_text = str(evaluation)

    return (
        summaries_text,
        plan_text,
        trace_text,
        report_text or "",
        pdf_path,
        evaluation_text,
    )


# --------------------------------------------------
# Gradio UI
# --------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown(
        """
## 🧠 Research Agent — Iterative Academic Discovery Pipeline

**Modes**
- **Quick**: Initial discovery (2 queries, 1 iteration)
- **Standard**: +1 coverage refinement (2 iterations total)
- **Deep**: +2 coverage refinements (3 iterations total)
"""
    )

    with gr.Row():
        query = gr.Textbox(
            label="Research Question",
            placeholder="Enter a formal research question...",
            lines=2,
        )

    mode = gr.Radio(
        choices=["quick", "standard", "deep"],
        value="standard",
        label="Research Mode",
    )

    btn = gr.Button("Run Research")

    with gr.Tabs():
        with gr.Tab("🧠 Research Plan"):
            plan_out = gr.Textbox(
                lines=18,
                label="Research Goal & Dimensions",
            )

        with gr.Tab("📌 Collected Summaries"):
            summaries_out = gr.Textbox(
                lines=25,
                label="Final Summaries (Post-Scoring & Resolution)",
            )

        with gr.Tab("🧾 Research Trace (Full Audit Trail)"):
            trace_out = gr.Textbox(
                lines=40,
                label="End-to-End Research Trace",
            )

        with gr.Tab("📝 Final Report"):
            report_out = gr.Textbox(
                lines=35,
                label="Academic Report Text",
            )

        with gr.Tab("📄 Download PDF"):
            pdf_out = gr.File(
                label="Download Generated PDF Report",
            )

        with gr.Tab("🔍 Self-Evaluation"):
            evaluation_out = gr.Textbox(
                lines=25,
                label="Post-Generation Quality Assessment",
            )

    btn.click(
        run_app,
        inputs=[query, mode],
        outputs=[
            summaries_out,
            plan_out,
            trace_out,
            report_out,
            pdf_out,
            evaluation_out,
        ],
    )

demo.launch()
