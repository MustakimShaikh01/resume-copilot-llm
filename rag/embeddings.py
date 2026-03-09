"""
embeddings.py
-------------
Generates sentence embeddings for resume/job-description text using
sentence-transformers. Produces vectors ready for FAISS indexing.

Model: all-MiniLM-L6-v2  (fast, 384-dim, runs well on CPU/MPS)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load singleton embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of strings.

    Args:
        texts: List of plain text strings.
        batch_size: Number of sentences per encoding batch.
        show_progress: Whether to display a progress bar.

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine sim via inner product after norm
    )
    return embeddings.astype(np.float32)


def embed_single(text: str) -> np.ndarray:
    """Embed a single string. Returns shape (1, 384)."""
    return embed_texts([text], show_progress=False)


if __name__ == "__main__":
    sample = ["Python developer with 5 years experience in ML and NLP."]
    vec = embed_single(sample[0])
    print(f"Embedding shape : {vec.shape}")
    print(f"Embedding norm  : {np.linalg.norm(vec):.4f}  (should be ~1.0 after normalisation)")
