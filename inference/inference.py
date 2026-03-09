"""
inference.py
------------
Load the fine-tuned ResumeCopilot LoRA model and run inference.

Modes:
  1. Local adapter  (training/output/) -- loaded after training
  2. Hub model      (mustakim/resume-copilot-llm) -- loaded from HF Hub
  3. Base-only      (TinyLlama) -- useful for comparison / before training

Usage:
  python inference/inference.py --mode local
  python inference/inference.py --mode hub --hub-repo mustakim/resume-copilot-llm
  python inference/inference.py --mode base
  python inference/inference.py --pdf my_resume.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

ROOT        = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "training" / "output"
BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYS_OPEN  = "<|" + "system" + "|>"
USR_OPEN  = "<|" + "user" + "|>"
ASST_OPEN = "<|" + "assistant" + "|>"
EOS       = "</s>"

SYSTEM_PROMPT = (
    "You are ResumeCopilot, an expert AI career coach. "
    "Analyse resumes, identify skill gaps, suggest improvements, "
    "and generate targeted interview questions."
)


# ── Model loading ─────────────────────────────────────────────────────────────

def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(mode: str, hub_repo: str) -> pipeline:
    device = detect_device()
    print(f"Device: {device.upper()}")

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    if mode == "hub":
        print(f"Loading model from Hub: {hub_repo}")
        tokenizer = AutoTokenizer.from_pretrained(hub_repo)
        model     = AutoModelForCausalLM.from_pretrained(hub_repo, dtype=dtype)

    elif mode == "local":
        if not ADAPTER_DIR.exists():
            print(f"Adapter directory not found: {ADAPTER_DIR}")
            print("Run training/train_qlora.py first, or use --mode base")
            sys.exit(1)
        print(f"Loading base model + local LoRA adapters from {ADAPTER_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=dtype,
            device_map={"": device},
        )
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        model = model.merge_and_unload()

    else:  # base
        print(f"Loading base model only: {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model     = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=dtype, device_map={"": device}
        )

    tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map={"": device},
        dtype=dtype,
    )
    return pipe


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(instruction: str, resume_text: str = "", context: str = "") -> str:
    user_parts = []
    if context:
        user_parts.append(context)
        user_parts.append("")
    user_parts.append(instruction)
    if resume_text.strip():
        user_parts.append("")
        user_parts.append("Resume:")
        user_parts.append(resume_text.strip())
    user_content = "\n".join(user_parts)

    return (
        f"{SYS_OPEN}\n{SYSTEM_PROMPT}\n{EOS}\n"
        f"{USR_OPEN}\n{user_content}\n{EOS}\n"
        f"{ASST_OPEN}\n"
    )


# ── Inference ─────────────────────────────────────────────────────────────────

def generate(pipe, prompt: str, max_new_tokens: int = 512) -> str:
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        return_full_text=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    return out[0]["generated_text"].strip()


# ── PDF resume parsing ────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("pypdf not installed. Run: pip install pypdf")
        sys.exit(1)

    reader = PdfReader(pdf_path)
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    return text.strip()


# ── Demo prompts ──────────────────────────────────────────────────────────────

DEMO_RESUME = (
    "John Doe | johndoe@email.com | linkedin.com/in/johndoe\n"
    "Summary: 4 years of experience as a Data Analyst. Proficient in Python, "
    "SQL, and Tableau. Familiar with machine learning basics.\n"
    "Experience:\n"
    "  Data Analyst @ Acme Corp (2021-2025): Built ETL pipelines, created "
    "dashboards, ran ad-hoc SQL analyses.\n"
    "Education: B.Sc. Computer Science, 2020.\n"
    "Skills: Python, SQL, Tableau, Excel, Git."
)

DEMO_QUESTIONS = [
    ("Analyse this resume and suggest specific improvements.", DEMO_RESUME),
    ("List 5 targeted interview questions based on this resume.", DEMO_RESUME),
    ("What skills is this candidate missing for a senior Data Scientist role?", DEMO_RESUME),
    ("Score this resume out of 10 and explain why.", DEMO_RESUME),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ResumeCopilot Inference")
    parser.add_argument("--mode",     choices=["local", "hub", "base"], default="base")
    parser.add_argument("--hub-repo", default="mustakim/resume-copilot-llm")
    parser.add_argument("--pdf",      default=None, help="Path to a resume PDF")
    parser.add_argument("--question", default=None, help="Custom question to ask")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    pipe = load_pipeline(args.mode, args.hub_repo)

    if args.pdf:
        print(f"\nExtracting text from {args.pdf} ...")
        resume_text = extract_pdf_text(args.pdf)
        instruction = args.question or "Analyse this resume and suggest improvements."
        prompt  = build_prompt(instruction, resume_text)
        print("\n" + "=" * 60)
        print(generate(pipe, prompt, args.max_tokens))
        return

    # Demo mode: run all built-in questions
    print("\n" + "=" * 60)
    print("ResumeCopilot Demo — running 4 sample questions")
    print("=" * 60)
    for instruction, resume in DEMO_QUESTIONS:
        prompt = build_prompt(instruction, resume)
        print(f"\nQ: {instruction}")
        print("-" * 50)
        print(generate(pipe, prompt, args.max_tokens))
        print()


if __name__ == "__main__":
    main()
