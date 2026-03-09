"""
vector_store.py
---------------
FAISS-backed vector store for job descriptions and resumes.

Supports:
- add_texts()   : embed + index a list of documents
- search()      : top-k semantic search by query string
- save() / load(): persist index to disk

Index file: rag/store/faiss.index  +  rag/store/metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np

from rag.embeddings import embed_texts, embed_single

STORE_DIR = Path(__file__).resolve().parent / "store"
INDEX_PATH = STORE_DIR / "faiss.index"
META_PATH  = STORE_DIR / "metadata.json"
DIM = 384   # all-MiniLM-L6-v2 output dimension


class VectorStore:
    """
    Thin wrapper around a FAISS IndexFlatIP (inner-product / cosine) index.
    All embeddings are L2-normalised before insertion, so inner product == cosine similarity.
    """

    def __init__(self):
        self.index    = faiss.IndexFlatIP(DIM)
        self.metadata: List[dict] = []   # parallel list: one dict per vector

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """
        Embed texts and add to the index.

        Args:
            texts: Plain text strings to index.
            metadatas: Optional list of dicts (same length as texts).
                       Stored alongside each vector for retrieval.
        """
        if not texts:
            return

        embeddings = embed_texts(texts)
        self.index.add(embeddings)

        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            self.metadata.append({"text": text, **meta})

        print(f"Indexed {len(texts)} docs  |  Total: {self.index.ntotal}")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, dict]]:
        """
        Semantic search.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of (score, metadata_dict) sorted by descending similarity.
        """
        if self.index.ntotal == 0:
            return []

        q_vec = embed_single(query)
        scores, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((float(score), self.metadata[idx]))
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, index_path: Path = INDEX_PATH, meta_path: Path = META_PATH):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved index ({self.index.ntotal} vectors) -> {index_path}")

    def load(self, index_path: Path = INDEX_PATH, meta_path: Path = META_PATH):
        if not index_path.exists():
            raise FileNotFoundError(f"No index found at {index_path}. Run indexing first.")
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"Loaded index: {self.index.ntotal} vectors")


# ── CLI convenience ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    store = VectorStore()
    sample_docs = [
        "Looking for a Data Scientist with Python, SQL, and machine learning experience.",
        "Senior Software Engineer role requiring microservices, Docker, and Kubernetes.",
        "Marketing Manager with SEO, Google Ads, and content strategy skills needed.",
    ]
    store.add_texts(sample_docs, metadatas=[{"source": "sample"} for _ in sample_docs])
    store.save()

    results = store.search("machine learning engineer", top_k=2)
    print("\nSearch results:")
    for score, meta in results:
        print(f"  [{score:.3f}]  {meta['text'][:80]}")
