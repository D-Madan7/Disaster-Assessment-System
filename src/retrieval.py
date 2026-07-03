from __future__ import annotations
from typing import Any

import joblib
import numpy as np

from .config import (
    ACTION_KEYWORDS,
    BAD_PATTERNS,
    DOC_INDICES_PATH,
    EMBEDDING_MODEL,
    ENABLE_RAG_RETRIEVAL,
    IMPORTANT_TERMS,
)
from .utils import extract_incident_info, infer_doc_folder

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def load_doc_indices(doc_indices_path=DOC_INDICES_PATH) -> dict[str, Any]:
    """Load prebuilt retrieval indices from disk."""
    return joblib.load(doc_indices_path)


def retrieval_score(chunk: str, query: str) -> int:
    """Compute the notebook's heuristic retrieval score."""
    text = chunk.lower()
    query = query.lower()
    score = 0

    for word in ACTION_KEYWORDS:
        if word in text:
            score += 2

    for word in IMPORTANT_TERMS:
        if word in text:
            score += 3

    for word in query.split():
        if len(word) > 4 and word in text:
            score += 1

    for bad in BAD_PATTERNS:
        if bad in text:
            score -= 5

    word_count = len(text.split())

    if word_count > 350:
        score -= 2
    elif word_count < 25:
        score -= 2

    return score


def build_scene_query(row) -> str:
    """Build the scene query text exactly as in the notebook."""
    affected = row["pred_affected_ratio"]

    if affected < 0.25:
        severity = "low impact"
    elif affected < 0.60:
        severity = "moderate impact"
    else:
        severity = "high impact"

    disaster_type, event_name = extract_incident_info(row["image_id"])

    return f"""
{event_name}

{disaster_type} disaster

{severity}

Post-disaster building assessment

{int(row['pred_destroyed'])} destroyed buildings

{int(row['pred_damaged'])} damaged buildings

{int(row['pred_no_damage'])} undamaged buildings

Building safety inspection

Structural damage assessment

Emergency response

Recovery operations

Unsafe buildings

Critical infrastructure

Debris removal

Utility restoration
"""


class RetrievalEngine:
    """Loads the embedding model and prebuilt document indices for retrieval."""

    def __init__(
        self,
        enable_rag_retrieval: bool = ENABLE_RAG_RETRIEVAL,
        embedding_model_name: str = EMBEDDING_MODEL,
        doc_indices_path=DOC_INDICES_PATH,
    ) -> None:
        self.enable_rag_retrieval = enable_rag_retrieval
        self.embedding_model_name = embedding_model_name
        self.doc_indices_path = doc_indices_path
        self.embedding_model = None
        self.doc_indices: dict[str, Any] = {}

        if self.enable_rag_retrieval and SentenceTransformer is not None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.doc_indices = load_doc_indices(self.doc_indices_path)


def retrieve_doc_chunks(
    query_text: str,
    image_id: str,
    retrieval_engine: RetrievalEngine,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Retrieve case-specific document chunks exactly as in the notebook."""
    if (
        not retrieval_engine.enable_rag_retrieval
        or retrieval_engine.embedding_model is None
        or not retrieval_engine.doc_indices
    ):
        return []

    disaster_folder = infer_doc_folder(image_id)
    search_categories = [disaster_folder, "Building Assessment"]
    query_embedding = retrieval_engine.embedding_model.encode(
        [query_text],
        show_progress_bar=False,
    ).astype("float32")

    candidates: list[dict[str, Any]] = []

    for category in search_categories:
        if category not in retrieval_engine.doc_indices:
            continue

        df = retrieval_engine.doc_indices[category]["df"]
        index = retrieval_engine.doc_indices[category]["index"]

        k = min(10, len(df))
        distances, indices = index.search(query_embedding, k)

        for distance, idx in zip(distances[0], indices[0]):
            row = df.iloc[int(idx)]
            chunk = row["chunk_text"]

            candidates.append(
                {
                    "distance": float(distance),
                    "semantic_score": retrieval_score(chunk, query_text),
                    "category": row["category"],
                    "file_name": row["file_name"],
                    "chunk_text": chunk,
                }
            )

    candidates = sorted(candidates, key=lambda x: (-x["semantic_score"], x["distance"]))
    seen = set()
    results = []

    for item in candidates:
        text = item["chunk_text"].strip()

        if text in seen:
            continue

        seen.add(text)

        results.append(
            {
                "rank": len(results) + 1,
                "category": item["category"],
                "file_name": item["file_name"],
                "chunk_text": item["chunk_text"],
            }
        )

        if len(results) == top_k:
            break

    return results
