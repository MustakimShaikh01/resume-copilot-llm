"""
app.py
------
Gradio web demo for ResumeCopilot-LLM.

Features:
  - Paste resume text OR upload a PDF
  - Choose analysis mode (Analyse / Skill Gaps / Interview Qs / Score)
  - Optional RAG toggle (uses FAISS job-description index)
  - Runs on TinyLlama base or fine-tuned local/Hub model

Launch:
  python app.py
  python app.py --mode local
  python app.py --mode hub --hub-repo mustakim/resume-copilot-llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from inference.inference import load_pipeline, build_prompt, generate, extract_pdf_text
from rag.retriever import Retriever

# ── Load model once ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode",     choices=["local", "hub", "base"], default="base")
parser.add_argument("--hub-repo", default="mustakim/resume-copilot-llm")
parser.add_argument("--max-tokens", type=int, default=512)
args, _ = parser.parse_known_args()

print(f"Loading pipeline (mode={args.mode}) ...")
PIPE = load_pipeline(args.mode, args.hub_repo)

retriever = Retriever(auto_load=True)   # loads FAISS index if it exists

TASK_MAP = {
    "Analyse & Improve":       "Analyse this resume and suggest specific, actionable improvements.",
    "Identify Skill Gaps":     "Identify the key skill gaps in this resume for the target role.",
    "Generate Interview Qs":   "Generate 5 targeted interview questions based on this resume.",
    "Score Resume (out of 10)":"Score this resume out of 10. Give a detailed justification.",
}


# ── Core function ─────────────────────────────────────────────────────────────

def analyse_resume(
    resume_text: str,
    pdf_file,
    task: str,
    custom_instruction: str,
    use_rag: bool,
    target_role: str,
    max_tokens: int,
) -> str:
    # ── Get resume text
    if pdf_file is not None:
        try:
            resume_text = extract_pdf_text(pdf_file.name)
        except Exception as e:
            return f"Error reading PDF: {e}"

    resume_text = resume_text.strip()
    if not resume_text:
        return "Please paste your resume text or upload a PDF."

    # ── Build instruction
    instruction = custom_instruction.strip() if custom_instruction.strip() else TASK_MAP.get(task, task)
    if target_role.strip():
        instruction += f" Target role: {target_role.strip()}."

    # ── Optional RAG context
    context = ""
    if use_rag and retriever.store.index.ntotal > 0:
        context = retriever.get_context(resume_text, top_k=3)

    # ── Build prompt & generate
    prompt = build_prompt(instruction, resume_text, context)
    try:
        return generate(PIPE, prompt, max_new_tokens=int(max_tokens))
    except Exception as e:
        return f"Generation error: {e}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
#header { text-align: center; margin-bottom: 10px; }
#output-box { font-size: 15px; line-height: 1.7; }
"""

with gr.Blocks(
    title="ResumeCopilot LLM",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
    css=CSS,
) as demo:

    gr.Markdown("# ResumeCopilot LLM", elem_id="header")
    gr.Markdown(
        "AI-powered resume analyser fine-tuned with QLoRA on TinyLlama. "
        "Paste your resume or upload a PDF to get started.",
        elem_id="header",
    )

    with gr.Row():
        with gr.Column(scale=3):
            resume_box = gr.Textbox(
                label="Resume Text",
                placeholder="Paste your resume here...",
                lines=18,
                max_lines=30,
            )
            pdf_upload = gr.File(
                label="Or upload a Resume PDF",
                file_types=[".pdf"],
            )

        with gr.Column(scale=2):
            task_radio = gr.Radio(
                choices=list(TASK_MAP.keys()),
                value="Analyse & Improve",
                label="Analysis Mode",
            )
            custom_instr = gr.Textbox(
                label="Custom Instruction (overrides mode)",
                placeholder="e.g. Rewrite my summary section for a senior ML role",
                lines=2,
            )
            target_role = gr.Textbox(
                label="Target Role (optional)",
                placeholder="e.g. Senior Data Scientist at a fintech startup",
            )
            use_rag = gr.Checkbox(
                label="Use RAG (job description context)",
                value=False,
            )
            max_tokens = gr.Slider(
                minimum=128, maximum=1024, value=args.max_tokens, step=64,
                label="Max output tokens",
            )
            run_btn = gr.Button("Analyse Resume", variant="primary")

    output_box = gr.Textbox(
        label="ResumeCopilot Analysis",
        lines=20,
        interactive=False,
        elem_id="output-box",
    )

    gr.Examples(
        examples=[
            [
                "John Doe | Python Developer | 3 years experience with Django, REST APIs, and PostgreSQL. "
                "Worked at Startup XYZ building backend services. B.Sc. Computer Science 2020.",
                None, "Identify Skill Gaps", "", "Senior Backend Engineer", False, 512
            ],
            [
                "Jane Smith | Marketing Specialist | Google Ads, SEO, content marketing. "
                "2 years at agency. MBA graduate.",
                None, "Score Resume (out of 10)", "", "Product Marketing Manager", False, 512
            ],
        ],
        inputs=[resume_box, pdf_upload, task_radio, custom_instr, target_role, use_rag, max_tokens],
        label="Example Resumes",
    )

    run_btn.click(
        fn=analyse_resume,
        inputs=[resume_box, pdf_upload, task_radio, custom_instr, use_rag, target_role, max_tokens],
        outputs=output_box,
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )

