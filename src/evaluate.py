"""Evaluation helpers.

Retrieval eval (run from repo root):
    python src/evaluate.py
Compares BM25, the dense (transformer) retriever and a hybrid of both on data/questions.json using
Hit@k and MRR at page level: a retrieved chunk counts as correct if its source
file is one of the question's gold_sources.
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import load_chunks, BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402

QUESTIONS = Path("data/questions.json")
RESULTS = Path("results")
KS = (1, 3, 5)
MAX_K = 10


def load_questions(path=QUESTIONS):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def first_rank(hits, gold):
    """1-based rank of the first retrieved chunk whose source is in gold, else None."""
    for i, h in enumerate(hits, 1):
        if h["source"] in gold:
            return i
    return None


def retrieval_eval(retriever, questions, ks=KS, max_k=MAX_K):
    rows, ranks = [], []
    for q in questions:
        if not q["answerable"]:
            continue
        hits = retriever.search(q["question"], k=max_k)
        r = first_rank(hits, set(q["gold_sources"]))
        ranks.append(r)
        rows.append({"id": q["id"], "question": q["question"], "gold": ";".join(q["gold_sources"]),
                     "rank": r if r else "", "top1_source": hits[0]["source"]})
    n = len(ranks)
    metrics = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in ks}
    metrics["mrr"] = sum(1 / r for r in ranks if r) / n
    metrics["n_questions"] = n
    return metrics, rows


# ---------- answer-quality helpers (used in the notebook) ----------
def _toks(s):
    return re.findall(r"\w+", s.lower())


def token_f1(pred, ref):
    p, r = Counter(_toks(pred)), Counter(_toks(ref))
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)


REFUSAL_PATTERNS = ("don't know", "do not know", "not in the context", "not mentioned",
                    "no information", "cannot answer", "can't answer", "not provided", "not found in")


def is_refusal(text):
    t = text.lower()
    return any(p in t for p in REFUSAL_PATTERNS)


def save_csv(path, rows):
    RESULTS.mkdir(exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    chunks = load_chunks()
    questions = load_questions()
    RESULTS.mkdir(exist_ok=True)
    summary = {}
    bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks)
    for retriever in (bm25, dense, HybridRetriever(bm25, dense)):
        metrics, rows = retrieval_eval(retriever, questions)
        summary[retriever.name] = metrics
        save_csv(RESULTS / f"retrieval_{retriever.name}.csv", rows)
    (RESULTS / "retrieval_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"{'retriever':10s}" + "".join(f"{'hit@'+str(k):>8s}" for k in KS) + f"{'MRR':>8s}")
    for name, m in summary.items():
        print(f"{name:10s}" + "".join(f"{m[f'hit@{k}']:8.3f}" for k in KS) + f"{m['mrr']:8.3f}")
    print(f"(n = {next(iter(summary.values()))['n_questions']} answerable questions)")


if __name__ == "__main__":
    main()
