"""
rag_module.py — NOVA Product Knowledge RAG
Importable module used by Task 5 (Multi-Agent Platform)

Usage:
    from rag_module import retrieve_and_answer, build_rag_pipeline
"""

import os
import json
import time
import numpy as np
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_PATH     = "./chroma_db"
DB_PATH         = "./nova_mock_db.json"

# ── Global state (loaded once) ────────────────────────────────
_embedding_model = None
_reranker        = None
_collection      = None
_bm25_index      = None
_all_docs        = None
_client          = None


def _load_components(groq_api_key: str):
    """Load all RAG components into memory (called once)."""
    global _embedding_model, _reranker, _collection
    global _bm25_index, _all_docs, _client

    if _collection is not None:
        return  # Already loaded

    _client          = Groq(api_key=groq_api_key)
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    _reranker        = CrossEncoder(RERANKER_MODEL)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection   = chroma_client.get_collection("nova_products")

    with open(DB_PATH, "r") as f:
        db = json.load(f)

    _all_docs = []
    for p in db["products"]:
        _all_docs.append({"doc_id": p["product_id"],
                          "text": f"Product: {p['name']}\n{p['description']}",
                          "metadata": {"product_id": p["product_id"],
                                       "name": p["name"],
                                       "category": p["category"]}})
    for f in db["faqs"]:
        _all_docs.append({"doc_id": f["id"],
                          "text": f"FAQ: {f['question']}\nAnswer: {f['answer']}",
                          "metadata": {"product_id": f["id"],
                                       "name": f["question"],
                                       "category": f["category"]}})

    tokenized  = [d["text"].lower().split() for d in _all_docs]
    _bm25_index = BM25Okapi(tokenized)


def retrieve_and_answer(query: str, groq_api_key: str = None,
                        top_k: int = 3) -> dict:
    """
    Main RAG entry point for Task 5.
    Performs hybrid retrieval and generates a grounded answer.
    """
    if groq_api_key:
        _load_components(groq_api_key)

    start           = time.time()
    query_embedding = _embedding_model.encode(
        query, normalize_embeddings=True).tolist()

    dense_results = _collection.query(
        query_embeddings=[query_embedding], n_results=10,
        include=["documents", "metadatas", "distances"])

    dense_docs = [{"text": d, "metadata": m,
                   "score": 1 - dist, "source": "dense"}
                  for d, m, dist in zip(
                      dense_results["documents"][0],
                      dense_results["metadatas"][0],
                      dense_results["distances"][0])]

    bm25_scores  = _bm25_index.get_scores(query.lower().split())
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:10]
    max_score    = max(bm25_scores) if max(bm25_scores) > 0 else 1

    sparse_docs = [{"text": _all_docs[i]["text"],
                    "metadata": _all_docs[i]["metadata"],
                    "score": bm25_scores[i] / max_score,
                    "source": "sparse"}
                   for i in top_bm25_idx if bm25_scores[i] > 0]

    seen, merged = set(), []
    for doc in dense_docs + sparse_docs:
        pid = doc["metadata"]["product_id"]
        if pid not in seen:
            seen.add(pid)
            merged.append(doc)

    if not merged:
        return {"query": query, "answer": "No relevant products found.",
                "sources": [], "context": "", "latency_ms": 0}

    pairs         = [[query, d["text"]] for d in merged]
    rerank_scores = _reranker.predict(pairs)

    for doc, score in zip(merged, rerank_scores):
        doc["rerank_score"] = float(score)

    top_docs = sorted(merged, key=lambda x: x["rerank_score"],
                      reverse=True)[:top_k]
    context  = "\n\n".join(
        f"[Source {i+1}]\n{d['text']}"
        for i, d in enumerate(top_docs))

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",
             "content": "Answer using ONLY the product info provided. Be concise."},
            {"role": "user",
             "content": f"Question: {query}\n\nProduct Info:\n{context}"}
        ],
        max_tokens=300, temperature=0.2)

    return {
        "query"     : query,
        "answer"    : response.choices[0].message.content,
        "sources"   : [{"product_id": d["metadata"]["product_id"],
                        "name": d["metadata"]["name"],
                        "rerank_score": round(d["rerank_score"], 3)}
                       for d in top_docs],
        "context"   : context,
        "latency_ms": round((time.time() - start) * 1000, 2)
    }
