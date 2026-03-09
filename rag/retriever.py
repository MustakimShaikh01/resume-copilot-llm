"""
retriever.py
------------
High-level RAG retriever: wraps VectorStore + LangChain to build
context-augmented prompts for the ResumeCopilot LLM.

Usage (standalone):
  python rag/retriever.py

Usage (from other modules):
  from rag.retriever import Retriever
  retriever = Retriever()
  context   = retriever.get_context("Python ML engineer resume")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from rag.vector_store import VectorStore, INDEX_PATH, META_PATH

# Optional: index job descriptions from a CSV on first run
JOB_CSV = Path(__file__).resolve().parent.parent / "data" / "raw"


class Retriever:
    """
    Retrieves relevant job descriptions given a resume snippet,
    then formats a context block to prepend to the LLM prompt.
    """

    def __init__(self, auto_load: bool = True):
        self.store = VectorStore()
        if auto_load and INDEX_PATH.exists():
            self.store.load()
        elif not INDEX_PATH.exists():
            print("No FAISS index found. Call build_index() or index your job descriptions first.")

    # ── Index building ────────────────────────────────────────────────────────

    def build_index(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Index a batch of job descriptions."""
        self.store.add_texts(texts, metadatas=metadatas)
        self.store.save()

    def build_index_from_csv(self, csv_path: Path, text_col: str = "job_description"):
        """Build index from a CSV file with a text column."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        if text_col not in df.columns:
            available = list(df.columns)
            raise ValueError(f"Column '{text_col}' not found. Available: {available}")
        texts = df[text_col].dropna().tolist()
        metas = df.drop(columns=[text_col]).to_dict(orient="records")
        self.build_index(texts, metas[:len(texts)])
        print(f"Indexed {len(texts)} job descriptions from {csv_path.name}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_context(self, query: str, top_k: int = 3, max_chars: int = 1200) -> str:
        """
        Search for relevant job descriptions and return a formatted context string.

        Args:
            query:     Resume text or user question.
            top_k:     Number of job descriptions to retrieve.
            max_chars: Maximum total characters to include in context block.

        Returns:
            Formatted context string ready to inject into an LLM prompt.
        """
        results = self.store.search(query, top_k=top_k)
        if not results:
            return ""

        parts = ["[RELEVANT JOB MARKET CONTEXT]"]
        total_chars = 0
        for i, (score, meta) in enumerate(results, 1):
            snippet = meta.get("text", "")[:400]
            total_chars += len(snippet)
            if total_chars > max_chars:
                break
            parts.append(f"Job {i} (relevance {score:.2f}):\n{snippet}")

        return "\n\n".join(parts)

    def build_rag_prompt(
        self,
        user_query: str,
        resume_text: str,
        top_k: int = 3,
    ) -> str:
        """
        Build a full RAG-augmented prompt to send to the LLM.

        Args:
            user_query:  e.g. "What skills am I missing for a data science role?"
            resume_text: Raw resume text from the user.
            top_k:       Number of job descriptions to retrieve.

        Returns:
            Full prompt string (context + instruction + resume).
        """
        context = self.get_context(resume_text, top_k=top_k)

        prompt_parts = []
        if context:
            prompt_parts.append(context)
            prompt_parts.append("")

        prompt_parts.append(f"Task: {user_query}")
        prompt_parts.append("")
        prompt_parts.append("Resume:")
        prompt_parts.append(resume_text.strip())

        return "\n".join(prompt_parts)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = Retriever(auto_load=False)

    # Index sample job descriptions
    sample_jobs = [
        "Data Scientist: Python, Pandas, Scikit-learn, SQL, A/B testing, Tableau.",
        "ML Engineer: PyTorch, MLflow, Docker, Kubernetes, model deployment expertise.",
        "Backend Engineer: Go, gRPC, PostgreSQL, Kafka, microservices architecture.",
        "Product Manager: roadmap planning, cross-functional leadership, OKRs, Jira.",
    ]
    retriever.build_index(
        sample_jobs,
        metadatas=[{"source": "sample", "role": s.split(":")[0]} for s in sample_jobs],
    )

    sample_resume = (
        "5 years as a Data Analyst. Proficient in Python, SQL, and Excel. "
        "Built dashboards with Tableau. Some exposure to Scikit-learn."
    )
    prompt = retriever.build_rag_prompt(
        user_query="What technical skills am I missing for a senior Data Scientist role?",
        resume_text=sample_resume,
    )
    print("\n=== RAG Prompt Preview ===")
    print(prompt)
