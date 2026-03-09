"""
kaggle_download.py
------------------
Downloads the two Kaggle datasets needed for ResumeCopilot-LLM.

Datasets:
  1. gauravduttakiit/resume-dataset
  2. PromptCloudHQ/skills-extracted-from-job-descriptions

Prerequisites:
  - `kaggle` package installed
  - ~/.kaggle/kaggle.json with your API credentials
    OR set KAGGLE_USERNAME + KAGGLE_KEY env vars

Usage:
  python pipelines/kaggle_download.py
"""

import os
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS = [
    {
        "slug": "gauravduttakiit/resume-dataset",
        "desc": "Resume Dataset (UpdatedResumeDataSet.csv)",
    },
    {
        "slug": "PromptCloudHQ/skills-extracted-from-job-descriptions",
        "desc": "Job Descriptions Skill Dataset",
    },
]


def check_kaggle_credentials():
    """Ensure Kaggle credentials are available."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    if not kaggle_json.exists() and not has_env:
        print("❌  Kaggle credentials not found!")
        print("   Option A: Place your kaggle.json at ~/.kaggle/kaggle.json")
        print("   Option B: Set KAGGLE_USERNAME and KAGGLE_KEY env vars")
        print("\n   Get your token at: https://www.kaggle.com/settings → API")
        sys.exit(1)
    print("✅  Kaggle credentials found.")


def download_dataset(slug: str, desc: str):
    """Download and unzip a Kaggle dataset into data/raw/."""
    print(f"\n📥  Downloading: {desc}")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", slug,
        "-p", str(RAW_DIR),
        "--unzip",
        "--force",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌  Failed to download {slug}")
        print(result.stderr)
    else:
        print(f"✅  Saved to {RAW_DIR}")


def list_raw_files():
    print("\n📂  Files in data/raw:")
    for f in sorted(RAW_DIR.glob("*")):
        size_mb = f.stat().st_size / 1_048_576
        print(f"   {f.name:40s}  {size_mb:.2f} MB")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    check_kaggle_credentials()
    for ds in DATASETS:
        download_dataset(ds["slug"], ds["desc"])
    list_raw_files()
    print("\n✅  Download complete!")


if __name__ == "__main__":
    main()
