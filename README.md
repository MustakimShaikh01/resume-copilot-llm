# ResumeCopilot-LLM

QLoRA fine-tuned TinyLlama that analyses resumes, spots skill gaps, suggests improvements,
and generates interview questions. Runs on Mac M2 (Apple MPS).

## Architecture

Kaggle Dataset -> preprocess_data.py -> build_dataset.py -> train_qlora.py -> HF Hub -> inference.py -> app.py (Gradio)

## File Structure

  data/raw/                   Kaggle CSV files
  data/processed/             instructions.json + hf_dataset/
  pipelines/kaggle_download.py
  pipelines/preprocess_data.py
  pipelines/build_dataset.py
  training/train_qlora.py
  training/output/            LoRA adapters
  rag/embeddings.py
  rag/vector_store.py
  rag/retriever.py
  inference/inference.py
  app.py                      Gradio demo

## Quick Start

### 1. Setup environment
  python -m venv llm
  source llm/bin/activate
  pip install -r requirements.txt

### 2. Download Kaggle data
  Place kaggle.json in ~/.kaggle/ (Kaggle -> Settings -> API)
  python pipelines/kaggle_download.py

### 3. Preprocess & build dataset
  python pipelines/preprocess_data.py
  python pipelines/build_dataset.py
  # Optional Hub push:
  python pipelines/build_dataset.py --push --hub-repo mustakim/resume-instruction-dataset

### 4. Fine-tune with QLoRA
  # Full run (~1-2 hrs on M2)
  python training/train_qlora.py
  # Quick smoke-test (500 samples, 1 epoch)
  python training/train_qlora.py --epochs 1 --samples 500
  # Train + push to Hub
  python training/train_qlora.py --push --hub-repo mustakim/resume-copilot-llm

### 5. Run inference
  python inference/inference.py --mode base       # base TinyLlama
  python inference/inference.py --mode local      # local LoRA adapters
  python inference/inference.py --mode hub --hub-repo mustakim/resume-copilot-llm
  python inference/inference.py --mode local --pdf my_resume.pdf

### 6. Launch Gradio demo
  python app.py
  python app.py --mode local
  Open: http://localhost:7860

## Training Config (Mac M2)

  Base model    : TinyLlama/TinyLlama-1.1B-Chat-v1.0
  LoRA rank     : 16
  LoRA alpha    : 32
  Epochs        : 3
  Batch size    : 2 (effective 8 with gradient_accumulation=4)
  Learning rate : 2e-4
  Max seq len   : 512
  Precision     : float16 (MPS) / 4-bit NF4 (CUDA)
  ETA 2k samples: ~1 hour on M2

## Dataset Format

  {
    "instruction": "Analyse this resume and suggest improvements.",
    "input": "Python developer with Flask and ML experience...",
    "output": "1. Add quantifiable achievements\n2. Include ATS keywords..."
  }

  Formatted into TinyLlama ChatML (system / user / assistant blocks) by build_dataset.py.

## Datasets Used

  Resume dataset  : https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
  Job skills      : https://www.kaggle.com/datasets/PromptCloudHQ/skills-extracted-from-job-descriptions

## Skills Demonstrated

  - QLoRA / PEFT fine-tuning
  - Dataset engineering (instruction tuning format)
  - RAG with FAISS vector search
  - PDF resume parsing
  - Hugging Face Transformers & Hub
  - Apple MPS (M2) optimisation
  - Gradio demo UI
# resume-copilot-llm
