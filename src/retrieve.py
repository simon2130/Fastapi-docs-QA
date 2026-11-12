"""BM25 baseline and dense (transformer) retriever over data/chunks.jsonl

Quick test:
    python src/retrieve.py "How do I declare query parameters?"
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

CHUNKS = Path("data/chunks.jsonl")
EMB_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # recommended for bge models


def load_chunks(path=CHUNKS):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _tok(text):
    return re.findall(r"\w+", text.lower())


def _doc_text(c):
    """Text that is indexed: 'Page title > Section heading' followed by the chunk text."""
    title, heading = c.get("title", ""), c.get("heading", "")
    head = f"{title} > {heading}" if title and heading and title != heading else (heading or title)
    return f"{head}\n{c['text']}"


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tok(_doc_text(c)) for c in chunks])

    def search(self, query, k=5):
        scores = self.bm25.get_scores(_tok(query))
        top = np.argsort(scores)[::-1][:k]
        return [dict(self.chunks[i], score=float(scores[i])) for i in top]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks, model_name=EMB_MODEL):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        texts = [_doc_text(c) for c in chunks]
        # The cache file name depends on the exact indexed text + model, so changing the chunks
        # (or the model) can never load stale embeddings.
        key = hashlib.md5((model_name + "\n".join(texts)).encode()).hexdigest()[:10]
        cache = Path(f"data/emb_{key}.npy")
        if cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                         show_progress_bar=True)
            cache.parent.mkdir(exist_ok=True)
            np.save(cache, self.emb)

    def search(self, query, k=5):
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = self.emb @ q   # cosine similarity (vectors are normalized)
        top = np.argsort(scores)[::-1][:k]
        return [dict(self.chunks[i], score=float(scores[i])) for i in top]


class HybridRetriever:
    """Reciprocal Rank Fusion of BM25 and dense rankings (both must index the same chunks)."""
    name = "hybrid"

    def __init__(self, bm25, dense, rrf_k=60, pool=30):
        self.bm25, self.dense, self.rrf_k, self.pool = bm25, dense, rrf_k, pool

    def search(self, query, k=5):
        fused = {}
        for retriever in (self.bm25, self.dense):
            for rank, h in enumerate(retriever.search(query, k=self.pool), 1):
                entry = fused.setdefault(h["id"], [h, 0.0])
                entry[1] += 1.0 / (self.rrf_k + rank)
        top = sorted(fused.values(), key=lambda e: -e[1])[:k]
        return [dict(h, score=score) for h, score in top]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "How do I declare query parameters?"
    chunks = load_chunks()
    bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks)
    for retriever in (bm25, dense, HybridRetriever(bm25, dense)):
        print(f"\n=== {retriever.name} ===")
        for r in retriever.search(query, k=3):
            print(f"[{r['score']:.3f}] {r['source']} :: {r['heading']}")
