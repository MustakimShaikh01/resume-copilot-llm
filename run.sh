#!/usr/bin/env bash
# =============================================================================
# run.sh — ResumeCopilot-LLM  |  One-command setup & launch
# Usage:
#   chmod +x run.sh
#   ./run.sh            # full pipeline (setup -> preprocess -> train -> demo)
#   ./run.sh --demo     # demo only (skip training, use base model)
#   ./run.sh --train    # setup + preprocess + train (no demo)
#   ./run.sh --help
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}${BOLD}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}${BOLD}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}${BOLD}[ERROR]${RESET} $*"; exit 1; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}"
echo "  ██████╗ ███████╗███████╗██╗   ██╗███╗   ███╗███████╗"
echo "  ██╔══██╗██╔════╝██╔════╝██║   ██║████╗ ████║██╔════╝"
echo "  ██████╔╝█████╗  ███████╗██║   ██║██╔████╔██║█████╗  "
echo "  ██╔══██╗██╔══╝  ╚════██║██║   ██║██║╚██╔╝██║██╔══╝  "
echo "  ██║  ██║███████╗███████║╚██████╔╝██║ ╚═╝ ██║███████╗"
echo "  ╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝"
echo ""
echo "   C O P I L O T  -  L L M     (QLoRA · TinyLlama · FAISS · RAG)"
echo -e "${RESET}"

# ── Flags ─────────────────────────────────────────────────────────────────────
DEMO_ONLY=false
TRAIN_ONLY=false
SKIP_KAGGLE=false
EPOCHS=1
SAMPLES=500
MODE=base

for arg in "$@"; do
  case $arg in
    --demo)       DEMO_ONLY=true ;;
    --train)      TRAIN_ONLY=true ;;
    --skip-kaggle) SKIP_KAGGLE=true ;;
    --epochs=*)   EPOCHS="${arg#*=}" ;;
    --samples=*)  SAMPLES="${arg#*=}" ;;
    --full)       EPOCHS=3; SAMPLES=3000 ;;
    --help|-h)
      echo "Usage: ./run.sh [options]"
      echo ""
      echo "  (no flags)       Full pipeline: setup+preprocess+train+demo"
      echo "  --demo           Skip training, launch Gradio demo with base model"
      echo "  --train          Setup+preprocess+train only, no demo"
      echo "  --full           Train 3 epochs on 3000 samples (2-3 hrs on M2)"
      echo "  --epochs=N       Number of training epochs (default: 1)"
      echo "  --samples=N      Training samples limit (default: 500)"
      echo "  --skip-kaggle    Skip Kaggle download (use existing raw data or synthetic)"
      echo "  --help           Show this help"
      exit 0
      ;;
  esac
done

# ── Detect OS & Python ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info "Platform: $(uname -sm)"

if command -v python3 &>/dev/null; then
  PYTHON=$(command -v python3)
elif command -v python &>/dev/null; then
  PYTHON=$(command -v python)
else
  error "Python 3 not found. Install from https://www.python.org or via pyenv."
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
info "Python: $PY_VERSION  ($PYTHON)"

# ── Virtual environment ───────────────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/llm_env"

if [ ! -d "$VENV_DIR" ]; then
  info "Creating virtual environment at $VENV_DIR ..."
  $PYTHON -m venv "$VENV_DIR"
  success "Virtual environment created."
else
  info "Virtual environment found: $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

# ── Dependencies ──────────────────────────────────────────────────────────────
info "Installing / updating dependencies ..."
$PIP install --quiet --upgrade pip
$PIP install --quiet -r requirements.txt
success "Dependencies ready."

# ── Demo-only shortcut ────────────────────────────────────────────────────────
if [ "$DEMO_ONLY" = true ]; then
  info "Launching Gradio demo (base model — no fine-tuning) ..."
  warn "Run ./run.sh first (without --demo) to fine-tune before the demo."
  if [ -d "$SCRIPT_DIR/training/output" ]; then
    info "Fine-tuned adapters found — using local mode."
    $PYTHON app.py --mode local
  else
    $PYTHON app.py --mode base
  fi
  exit 0
fi

# ── Kaggle data download ──────────────────────────────────────────────────────
if [ "$SKIP_KAGGLE" = false ]; then
  if [ -f "$HOME/.kaggle/kaggle.json" ] || { [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; }; then
    info "Downloading Kaggle datasets ..."
    $PYTHON pipelines/kaggle_download.py
    success "Datasets downloaded."
  else
    warn "Kaggle credentials not found — using synthetic training data."
    warn "To use real data: place kaggle.json at ~/.kaggle/kaggle.json"
    warn "Get it from: https://www.kaggle.com/settings → API"
    SKIP_KAGGLE=true
  fi
fi

# ── Preprocessing ─────────────────────────────────────────────────────────────
info "Preprocessing data (instruction tuning format) ..."
$PYTHON pipelines/preprocess_data.py
success "Preprocessing complete."

info "Building Hugging Face dataset ..."
$PYTHON pipelines/build_dataset.py
success "Dataset ready."

# ── Training ──────────────────────────────────────────────────────────────────
info "Starting QLoRA fine-tuning  (epochs=$EPOCHS  samples=$SAMPLES) ..."
info "This runs on Apple MPS (M2/M3) or CUDA. CPU is supported but slow."
echo ""
$PYTHON training/train_qlora.py --epochs "$EPOCHS" --samples "$SAMPLES"
success "Training complete! Adapters saved to training/output/"

# ── Train-only shortcut ───────────────────────────────────────────────────────
if [ "$TRAIN_ONLY" = true ]; then
  echo ""
  success "All done. To launch the demo run:"
  echo "       source $VENV_DIR/bin/activate"
  echo "       python app.py --mode local"
  exit 0
fi

# Set MODE based on available adapters
if [ -d "$SCRIPT_DIR/training/output" ]; then
  MODE=local
fi

# ── Launch Gradio demo ────────────────────────────────────────────────────────
echo ""
success "Launching ResumeCopilot demo at http://localhost:7860"
info "Press Ctrl+C to stop the server."
echo ""
$PYTHON app.py --mode "$MODE"
