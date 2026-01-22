import json
import gradio as gr

from vector_store.client import VectorStoreClient
from controller.run import run_pipeline
from trace.research_trace import ResearchTrace


# --------------------------------------------------
# Vector store (persistent, reuse-ready)
# --------------------------------------------------

VECTOR_CLIENT = VectorStoreClient(
    persist_dir="vector_data",
    embedding_dim=384,
)


# --------------------------------------------------
# Gradio handler
# --------------------------------------------------

def run_app(query: str, mode: str):
    trace = ResearchTrace()

    # run_pipeline returns:
    # summaries: List[Dict]
    # trace_text: str
    # report_text: str | None
    # pdf_path: str | None
    # evaluation: Dict | None
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

    # --------------------------------------------------
    # Summaries output
    # --------------------------------------------------

    if summaries:
        summaries_text = "\n\n".join(
            f"{s['id']} (total_score={s.get('total_score', 'NA')}):\n{s['summary']}"
            for s in summaries
        )
    else:
        summaries_text = ""

    # --------------------------------------------------
    # Report text output
    # --------------------------------------------------

    report_text = report_text or ""

    # --------------------------------------------------
    # Evaluation output
    # --------------------------------------------------

    evaluation_text = (
        json.dumps(evaluation, indent=2)
        if evaluation
        else ""
    )

    return (
        summaries_text,
        trace_text,
        report_text,
        pdf_path,
        evaluation_text,
    )


# --------------------------------------------------
# Gradio UI
# --------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Research Agent — End-to-End Academic Pipeline")

    with gr.Row():
        query = gr.Textbox(
            label="Research Question",
            placeholder="Enter a formal research question...",
            lines=2,
        )

    mode = gr.Radio(
        ["quick", "standard"],
        value="standard",
        label="Mode",
    )

    btn = gr.Button("Run Research")

    with gr.Tabs():
        with gr.Tab("📌 Generated Summaries"):
            summaries_out = gr.Textbox(
                lines=25,
                label="Final Summaries (Post-Scoring & Conflict Resolution)",
            )

        with gr.Tab("🧾 Research Trace"):
            trace_out = gr.Textbox(
                lines=40,
                label="Full Pipeline Trace",
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

        with gr.Tab("🔍 Self-Evaluation & Critique"):
            evaluation_out = gr.Textbox(
                lines=25,
                label="Post-Generation Quality Assessment",
            )

    btn.click(
        run_app,
        inputs=[query, mode],
        outputs=[
            summaries_out,
            trace_out,
            report_out,
            pdf_out,
            evaluation_out,
        ],
    )

demo.launch()
