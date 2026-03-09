"""
train_qlora.py
--------------
QLoRA fine-tuning of TinyLlama-1.1B-Chat-v1.0 on the ResumeCopilot dataset.
Optimised for Apple Silicon (MPS) but gracefully falls back to CPU.

Usage:
  python training/train_qlora.py
  python training/train_qlora.py --epochs 1 --samples 500   # quick smoke-test

What it does:
  1. Loads TinyLlama in 4-bit (bitsandbytes) or fp16 (MPS fallback)
  2. Applies LoRA adapters with PEFT
  3. Trains with SFTTrainer (TRL)
  4. Saves adapters to training/output/
  5. Optionally merges + pushes full model to Hugging Face Hub
"""

import argparse
import os
import platform
from pathlib import Path

import torch
from datasets import load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data" / "processed" / "hf_dataset"
OUTPUT    = ROOT / "training" / "output"

# ── Model ────────────────────────────────────────────────────────────────────
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── LoRA config ──────────────────────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


def detect_device() -> str:
    if torch.backends.mps.is_available():
        print("Device: Apple MPS (Metal)")
        return "mps"
    if torch.cuda.is_available():
        print("Device: CUDA GPU")
        return "cuda"
    print("Device: CPU (slow, but works)")
    return "cpu"


def load_model_and_tokenizer(use_4bit: bool, device: str):
    print(f"Loading base model: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_4bit and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # MPS / CPU: load in float16 (bitsandbytes 4-bit not supported on MPS)
        dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
        )

    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return model, tokenizer


def load_dataset(samples: int | None):
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} not found.\n"
            "Run: python pipelines/build_dataset.py  first."
        )
    ds = load_from_disk(str(DATA_DIR))
    train_ds = ds["train"]
    if samples:
        train_ds = train_ds.select(range(min(samples, len(train_ds))))
    print(f"Training samples: {len(train_ds)}")
    return train_ds


def build_sft_config(epochs: int, device: str) -> SFTConfig:
    use_fp16 = device == "cuda"
    use_bf16 = False  # MPS does not support bf16 reliably yet

    return SFTConfig(
        output_dir=str(OUTPUT),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,           # replaces deprecated warmup_ratio
        logging_steps=10,
        save_strategy="epoch",
        fp16=use_fp16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        report_to="none",
        dataloader_pin_memory=False,  # required for MPS
        # SFTConfig-specific
        dataset_text_field="text",
        max_length=512,            # trl 0.29+ uses max_length not max_seq_length
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,  default=3)
    parser.add_argument("--samples", type=int,  default=None,
                        help="Limit training samples (e.g. 500 for a quick test)")
    parser.add_argument("--push",    action="store_true",
                        help="Push trained adapters to Hugging Face Hub")
    parser.add_argument("--hub-repo", default="mustakim/resume-copilot-llm")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    device  = detect_device()
    use_4bit = device == "cuda"

    model, tokenizer = load_model_and_tokenizer(use_4bit, device)
    # Attach tokenizer to model config so SFTTrainer picks it up automatically
    model.config.tokenizer_name_or_path = BASE_MODEL
    train_ds  = load_dataset(args.samples)
    sft_cfg   = build_sft_config(args.epochs, device)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,   # new API: replaces 'tokenizer' kwarg
        train_dataset=train_ds,
        args=sft_cfg,
    )

    print("\nStarting training ...")
    trainer.train()

    print(f"\nSaving LoRA adapters -> {OUTPUT}")
    trainer.model.save_pretrained(str(OUTPUT))
    tokenizer.save_pretrained(str(OUTPUT))

    if args.push:
        print(f"Pushing to Hub: {args.hub_repo} ...")
        trainer.model.push_to_hub(args.hub_repo)
        tokenizer.push_to_hub(args.hub_repo)
        print(f"Done! https://huggingface.co/{args.hub_repo}")
    else:
        print("\nTo push to Hub re-run with: --push --hub-repo <your/repo>")


if __name__ == "__main__":
    main()
