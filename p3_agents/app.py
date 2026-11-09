"""Gradio demo: shows each agent's output. In a notebook: app.build().launch(share=True)."""
import json

from p3_agents import graph


def build(hub):
    import gradio as gr

    def run(image, request, critic, seed):
        s = graph.run(hub, request, image, seed=int(seed), enable_critic=bool(critic))
        by = {t["agent"]: t for t in s.trace}
        j = lambda n: json.dumps({"input": by[n]["input"], "output": by[n]["output"], "seconds": by[n]["seconds"]}, indent=2, default=str) if n in by else "(not run)"
        status = ("REFUSED: " if s.refused else "") + (s.message or "") + (s.answer or "")
        return (s.output_path, status.strip() or "done", j("Planner"), j("Vision"), j("PromptEngineer"), j("ImageGeneration"),
                j("Critique"), f"route={s.route} revisions={s.revisions} total={s.total_seconds}s")

    with gr.Blocks(title="DATA266 Lab 2 - multi-agent image system") as demo:
        gr.Markdown("### Planner -> Vision -> Prompt Engineering -> Image Generation -> Critique (max 2 revisions)")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="filepath", label="Input image (optional)")
                req = gr.Textbox(label="Request", lines=2)
                crit = gr.Checkbox(True, label="Enable critique/revision loop")
                seed = gr.Number(0, label="Seed", precision=0)
                btn = gr.Button("Run", variant="primary")
            with gr.Column():
                out = gr.Image(label="Output image")
                msg = gr.Textbox(label="Answer / message", lines=3)
                summ = gr.Textbox(label="Summary")
        boxes = [gr.Code(label=n, language="json") for n in ("1 Planner", "2 Vision", "3 Prompt Engineering", "4 Image Generation", "5 Critique")]
        btn.click(run, [img, req, crit, seed], [out, msg, *boxes, summ])
    return demo
