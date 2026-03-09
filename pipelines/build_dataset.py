"""
build_dataset.py
----------------
Converts data/processed/instructions.json -> Hugging Face Dataset,
splits into train/test, and (optionally) pushes to the Hub.

Usage:
  # Local only
  python pipelines/build_dataset.py

  # Push to Hugging Face Hub
  python pipelines/build_dataset.py --push --hub-repo mustakim/resume-instruction-dataset
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
JSON_PATH = PROC_DIR / "instructions.json"

# Build special tokens at runtime to avoid editor truncation
SYS_OPEN  = "<|" + "system" + "|>"
USR_OPEN  = "<|" + "user" + "|>"
ASST_OPEN = "<|" + "assistant" + "|>"
EOS       = "</s>"
SYSTEM_PROMPT = "You are ResumeCopilot, an expert AI career coach."


def load_json(path: Path) -> list:
    print(f"Loading {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   Total examples: {len(data)}")
    return data


def format_for_training(example: dict) -> dict:
    """
    Combine instruction + input into a single text field using the
    TinyLlama ChatML template that SFTTrainer will train on.
    """
    instruction = example["instruction"]
    input_text  = example.get("input", "").strip()
    output_text = example["output"].strip()

    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction

    # Assemble ChatML-formatted training text
    text = (
        f"{SYS_OPEN}\n{SYSTEM_PROMPT}\n{EOS}\n"
        f"{USR_OPEN}\n{user_content}\n{EOS}\n"
        f"{ASST_OPEN}\n{output_text}\n{EOS}"
    )
    return {"text": text, **example}


def stats(dataset_dict: DatasetDict):
    print("\nDataset summary:")
    for split, ds in dataset_dict.items():
        print(f"  {split:8s}: {len(ds):,} examples")
    sample = dataset_dict["train"][0]
    print("\nSample text field (first 400 chars):")
    print(sample["text"][:400])


def main():
    parser = argparse.ArgumentParser(description="Build Hugging Face Dataset")
    parser.add_argument("--push", action="store_true",
                        help="Push dataset to Hugging Face Hub")
    parser.add_argument("--hub-repo", default="mustakim/resume-instruction-dataset",
                        help="Hub repo id, e.g. username/dataset-name")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Fraction of data to use as test split (default 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load raw instruction data
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"{JSON_PATH} not found.\n"
            "Run: python pipelines/preprocess_data.py first."
        )
    data = load_json(JSON_PATH)

    # Build HF Dataset
    dataset = Dataset.from_list(data)

    # Format each example into ChatML text
    dataset = dataset.map(format_for_training, desc="Formatting examples")

    # Train / test split
    splits = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    dataset_dict = DatasetDict({"train": splits["train"], "test": splits["test"]})

    stats(dataset_dict)

    # Save locally
    local_path = PROC_DIR / "hf_dataset"
    dataset_dict.save_to_disk(str(local_path))
    print(f"\nSaved locally -> {local_path}")

    # Optionally push to Hub
    if args.push:
        print(f"\nPushing to Hub: {args.hub_repo} ...")
        dataset_dict.push_to_hub(args.hub_repo)
        print(f"Pushed! View at https://huggingface.co/datasets/{args.hub_repo}")
    else:
        print("\nTo push to Hub, re-run with: --push --hub-repo <your/repo>")


if __name__ == "__main__":
    main()