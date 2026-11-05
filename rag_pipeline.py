# rag_pipeline.py — RAG pipeline: embeddings, ChromaDB vector store, and retrieval

from __future__ import annotations

import os
import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import chromadb

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    TOP_K_RETRIEVAL,
)


# ---------------------------------------------------------------------------
# Embedding backend — tries sentence-transformers (GPU/CPU), falls back to
# a lightweight TF-IDF-style hash embedding that needs NO torch/native DLLs.
# ---------------------------------------------------------------------------

_EMBED_MODEL = None
_EMBED_BACKEND: str = "none"   # "sentence_transformers" | "chromadb_default" | "hash"


def _try_load_sentence_transformers():
    """Attempt to import and load sentence-transformers. Returns model or None."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception:
        return None


def _get_embed_model():
    """
    Lazily resolve the best available embedding backend.

    Priority:
      1. sentence-transformers (all-MiniLM-L6-v2) — best quality
      2. chromadb built-in default embedding function — no torch needed
      3. Hash-based fallback — pure Python, zero dependencies
    """
    global _EMBED_MODEL, _EMBED_BACKEND
    if _EMBED_BACKEND != "none":
        return _EMBED_MODEL

    model = _try_load_sentence_transformers()
    if model is not None:
        _EMBED_MODEL = model
        _EMBED_BACKEND = "sentence_transformers"
        return _EMBED_MODEL

    # sentence-transformers unavailable (e.g. torch DLL error on Windows)
    # Fall back to chromadb's built-in embedding function (uses onnxruntime)
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction  # noqa: PLC0415
        _EMBED_MODEL = DefaultEmbeddingFunction()
        _EMBED_BACKEND = "chromadb_default"
        return _EMBED_MODEL
    except Exception:
        pass

    # Last resort: deterministic hash-based embeddings (no external deps)
    _EMBED_MODEL = None
    _EMBED_BACKEND = "hash"
    return None


# -- Hash-based fallback embedder (pure Python, 256-dim) --------------------

def _hash_embed(text: str, dims: int = 256) -> List[float]:
    """
    Deterministic pseudo-embedding via SHA-256 based word hashing.
    Quality is lower than neural embeddings but always works.
    """
    vec = [0.0] * dims
    words = text.lower().split()
    for word in words:
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        idx = h % dims
        vec[idx] += 1.0
    # L2-normalise
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return float embeddings for a list of text strings."""
    model = _get_embed_model()

    if _EMBED_BACKEND == "sentence_transformers":
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    if _EMBED_BACKEND == "chromadb_default":
        # DefaultEmbeddingFunction accepts a list and returns a list of lists
        return model(texts)

    # hash fallback
    return [_hash_embed(t) for t in texts]


def get_embedding_backend() -> str:
    """Return the name of the currently active embedding backend."""
    _get_embed_model()   # ensure initialised
    return _EMBED_BACKEND


# ---------------------------------------------------------------------------
# ChromaDB client (singleton)
# ---------------------------------------------------------------------------

_CHROMA_CLIENT: Optional[chromadb.PersistentClient] = None


def _get_chroma_client() -> chromadb.PersistentClient:
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
        )
    return _CHROMA_CLIENT


def _get_collection() -> chromadb.Collection:
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def add_documents_to_index(
    chunks: List[str],
    metadata_list: List[Dict[str, Any]],
    source_name: str,
) -> int:
    """
    Embed and upsert document chunks into the ChromaDB collection.

    Returns the number of chunks added.
    """
    if not chunks:
        return 0

    collection = _get_collection()

    # Build stable IDs: source + chunk index avoids duplicates on re-upload
    ids = [f"{source_name}::chunk_{m['chunk_index']}" for m in metadata_list]

    # Check which IDs already exist so we only add new ones
    existing = set(collection.get(ids=ids)["ids"])
    new_mask = [i for i, doc_id in enumerate(ids) if doc_id not in existing]

    if not new_mask:
        return 0  # all chunks already indexed

    new_chunks = [chunks[i] for i in new_mask]
    new_ids = [ids[i] for i in new_mask]
    new_meta = [metadata_list[i] for i in new_mask]

    embeddings = embed_texts(new_chunks)

    collection.add(
        documents=new_chunks,
        embeddings=embeddings,
        metadatas=new_meta,
        ids=new_ids,
    )
    return len(new_mask)


def list_indexed_sources() -> List[str]:
    """Return a deduplicated list of source document names in the index."""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    sources = {m.get("source", "unknown") for m in result["metadatas"]}
    return sorted(sources)


def get_index_stats() -> Dict[str, Any]:
    """Return basic statistics about the current vector index."""
    collection = _get_collection()
    count = collection.count()
    sources = list_indexed_sources()
    return {
        "total_chunks": count,
        "document_count": len(sources),
        "sources": sources,
    }


def clear_index() -> None:
    """Delete all documents from the vector store (destructive)."""
    client = _get_chroma_client()
    client.delete_collection(COLLECTION_NAME)
    # Re-create so subsequent operations work normally
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most semantically relevant document chunks for a query.

    Parameters
    ----------
    query:
        Natural-language query string.
    top_k:
        Number of results to return.
    source_filter:
        If provided, restrict results to chunks from this document source.

    Returns
    -------
    List of dicts, each with keys: ``text``, ``score``, ``source``,
    ``chunk_index``.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]

    where_filter = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance → similarity score (1 − distance)
        score = round(1.0 - dist, 4)
        chunks.append(
            {
                "text": doc,
                "score": score,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
            }
        )

    # Sort by descending relevance score
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def build_context_string(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a single context block for the LLM prompt.
    """
    if not chunks:
        return "No relevant document context found."

    parts: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Context {i} | Source: {chunk['source']} | Relevance: {chunk['score']:.2f}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)
