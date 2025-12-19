import gradio as gr
from agent.controller import run_agent

def launch_ui():
    with gr.Blocks() as demo:
        query = gr.Textbox(label="Research Question")
        output = gr.File(label="Generated PDF")
        trace = gr.Textbox(label="Research Trace")

        def run(q):
            pdf, logs = run_agent(q)
            return pdf, logs

        btn = gr.Button("Run Research Agent")
        btn.click(run, inputs=query, outputs=[output, trace])

    demo.launch()
