"""Demo UI.  Run from the repo root (after `python src/ingest.py`):

    python app.py

The dense (transformer) retriever is the default because it was the best in our evaluation.
"""
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag import RAG                                                    # noqa: E402
from retrieve import load_chunks, BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402

chunks = load_chunks()
bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks)
RETRIEVERS = {"dense": dense, "bm25": bm25, "hybrid": HybridRetriever(bm25, dense)}
rag = RAG(dense)


def ui_answer(question, retriever_name, k):
    ans, hits = rag.answer(question, k=int(k), retriever=RETRIEVERS[retriever_name])
    sources = "\n\n".join(
        f"**[{i}] {h['source']}** - {h['heading']}\n\n> " + h["text"][:300].replace("\n", " ") + "..."
        for i, h in enumerate(hits, 1))
    return ans, sources


def build_ui():
    kwargs = dict(
        fn=ui_answer,
        inputs=[gr.Textbox(label="Question about FastAPI", lines=2),
                gr.Radio(["dense", "bm25", "hybrid"], value="dense", label="Retriever"),
                gr.Slider(1, 6, value=4, step=1, label="Top-k chunks")],
        outputs=[gr.Textbox(label="Answer"), gr.Markdown(label="Retrieved sources")],
        title="FastAPI Documentation QA (Transformer RAG)",
        examples=[["How do I raise a 404 error?", "dense", 4],
                  ["How do I allow a frontend on another domain to call my API?", "dense", 4]],
    )
    try:                                   # Gradio 5+
        return gr.Interface(flagging_mode="never", **kwargs)
    except TypeError:                      # older Gradio
        return gr.Interface(allow_flagging="never", **kwargs)


if __name__ == "__main__":
    build_ui().launch()
