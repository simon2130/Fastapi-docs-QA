# Transformer-based Question Answering over Software Documentation

Group project for *Selected Topics in Software Engineering*.
We built a system that answers developers' questions about a software library by retrieving the relevant
parts of its documentation and generating a grounded answer with a local transformer model.
The documentation we use is the official [FastAPI](https://github.com/fastapi/fastapi) docs.

> Demo video Link:  

## 1. Problem

Developers constantly search long documentation for answers to questions like *"How do I return a 404 error?"*.
Keyword search misses answers phrased differently from the docs, and a language model asked without the
docs may invent code. Our research question:

**Does a transformer-based retriever (retrieval-augmented generation) answer documentation questions better than a keyword baseline, and which design choices matter?**

## 2. How the system works

```
 question
    |
    v
 [ Retriever ] --- finds the 4 most relevant documentation chunks
    |                 (BM25 keyword baseline | dense transformer embeddings | hybrid)
    v
 [ Generator ] --- a local LLM (Qwen2.5-3B-Instruct) answers using ONLY those chunks and cites them
    |
    v
 answer + the source chunks it used
```

1. **Ingest** (`src/ingest.py`): the Markdown docs are split by headings into chunks of at most 1,200 characters. Every chunk keeps its page, section heading and page title. Non-documentation pages (release notes, people, newsletter, ...) are excluded.
2. **Retrieve** (`src/retrieve.py`):
   - *BM25*: classic keyword ranking (baseline).
   - *Dense*: chunks and questions are embedded with the transformer encoder `BAAI/bge-small-en-v1.5` and ranked by cosine similarity.
   - *Hybrid*: Reciprocal Rank Fusion of the two rankings.
3. **Generate** (`src/rag.py`): the top chunks are given to `Qwen2.5-3B-Instruct` with instructions to answer only from the context, not to write code that is not in the context, to say "I don't know" when the answer is missing, and to cite sources.
4. **Evaluate** (`src/evaluate.py`, `data/questions.json`): see section 5.
5. **Demo** (`app.py`): a Gradio web app showing the answer and the retrieved sources.

## 3. Repository structure

```
app.py                  demo UI
requirements.txt
src/ingest.py           chunking of the documentation
src/retrieve.py         BM25, dense and hybrid retrievers
src/rag.py              retrieval + generation pipeline
src/evaluate.py         metrics and retrieval evaluation
data/questions.json     47 evaluation questions with reference answers and source pages
doc_qa_notebook.ipynb   Colab notebook that runs everything (evaluation, ablations, demo)
results/                metrics, tables, charts and answers produced by the notebook
results/answers_scored.xlsx   every answer of the three systems with correctness/faithfulness scores and notes
```

## 4. How to run

**Option A - Google Colab (recommended, free GPU).** Upload `doc_qa_notebook.ipynb`, set the runtime to T4 GPU and run the cells in order (approve the Google Drive popup on the first cell). Results are written to `MyDrive/doc-qa/results`.

**Option B - local machine.**
```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/fastapi/fastapi data/fastapi   # the documentation
python src/ingest.py        # builds data/chunks.jsonl
python src/evaluate.py      # retrieval evaluation -> results/
python app.py               # demo UI at http://127.0.0.1:7860
```
A GPU is strongly recommended. On a CPU-only machine set `QA_MODEL=Qwen/Qwen2.5-1.5B-Instruct` for a smaller, faster model.

## 5. Evaluation

**Question set.** 47 questions written from real documentation pages (`data/questions.json`): 44 answerable questions, each with a reference answer and the page that contains the answer, and 3 off-topic questions that the system should refuse.

**Retrieval metrics.** A retrieved chunk counts as correct if it comes from the gold page. *Hit@k* is the share of questions with a correct chunk in the top k; *MRR* is the mean reciprocal rank of the first correct chunk.

**Answer metrics.** Token F1 against the reference answer (rough), the share of off-topic questions correctly refused, and a manual check of every answer for *correctness* and *faithfulness* (every claim supported by the retrieved sources).

## 6. Results

### 6.1 Retrieval: transformer vs. keyword baseline (final index, 1,216 chunks)

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| BM25 (baseline) | 0.659 | 0.795 | 0.864 | 0.746 |
| **Dense (transformer)** | **0.750** | **0.955** | **0.977** | **0.847** |
| Hybrid (BM25 + dense) | 0.750 | 0.864 | 0.932 | 0.821 |

The transformer retriever finds the right page in the top 3 for 42 of 44 questions, against 35 for BM25.
Combining it with BM25 did not help: the weaker keyword ranking dilutes the dense ranking.
With only 44 questions, one question is worth about 2.3 percentage points, so differences of one to three questions (for example dense vs. hybrid) are suggestive, not conclusive.

### 6.2 What improved retrieval

| Variant | BM25 MRR | Dense MRR | Hybrid MRR |
|---|---|---|---|
| Baseline: all pages, no page title | 0.682 | 0.847 | 0.750 |
| + exclude non-documentation pages | 0.762 | 0.859 | 0.807 |
| + page title in each chunk | 0.698 | 0.835 | 0.799 |
| + both (final system) | 0.746 | 0.847 | 0.821 |

- **Removing non-documentation pages mattered most for keyword search.** `release-notes.md` alone made up 821 of 2,089 chunks (39%) and is full of terms like "404" and "response_model", so it outranked the real documentation. BM25 MRR rose from 0.682 to 0.762. The dense retriever was barely affected (0.847 to 0.859).
- **Adding the page title to chunks** gave mixed, small changes (it raised dense Hit@3/Hit@5 but lowered Hit@1), so we treat its effect as inconclusive. We kept it in the final system because the generator reads the top 4 chunks, so Hit@3 and Hit@5 matter more than Hit@1.

### 6.3 Chunk size (final system)

| Max chunk size (chars) | Chunks | BM25 MRR | Dense MRR | Hybrid MRR |
|---|---|---|---|---|
| 600 | 1,703 | 0.788 | 0.845 | 0.787 |
| **1200** | 1,216 | 0.746 | 0.847 | 0.821 |
| 2000 | 1,096 | 0.749 | 0.831 | 0.800 |

The dense retriever is stable across chunk sizes (MRR 0.831-0.847); only very large chunks were slightly worse. We kept 1,200 characters, which gives the generator enough context per chunk without a larger index.

### 6.4 Answer quality (Qwen2.5-3B-Instruct, top-4 chunks, final index)

| Retriever | Gold page in top 4 | Correct answers | Faithful answers | Token F1 | Off-topic questions refused |
|---|---|---|---|---|---|
| BM25 | 0.818 | 34 / 44 (0.77) | 33 / 44 (0.75) | 0.246 | 3 / 3 |
| Dense (transformer) | **0.977** | 35 / 44 (0.80) | 32 / 44 (0.73) | 0.243 | 3 / 3 |
| Hybrid | 0.909 | **38 / 44 (0.86)** | **35 / 44 (0.80)** | 0.267 | 3 / 3 |

*Correct* = the key point of the reference answer is present with no contradiction (a documented alternative method counts). *Faithful* = no invented or documentation-contradicting claims; code must be valid and consistent with the docs.
These judgments were made by an LLM judge (Claude), not by the team, using the reference answers and the documentation pages; every answer, score and a note for each failure is in `results/answers_scored.xlsx` (sheet `answers`; totals in sheet `summary`).

What the table shows:
- **The retrieval advantage of the transformer retriever did not carry through to final answers.** Dense found the right page far more often (0.977 vs 0.818), yet the three systems give a similar number of correct answers (34 / 35 / 38 out of 44). With 44 questions, a gap of 1-4 questions is within noise, so we cannot rank the three systems on answer quality.
- **The generator is the bottleneck.** Several questions are answered wrongly by all three systems even though the right page was retrieved (for example the middleware-order and async-test questions, section 6.5). Many questions are also simple enough for keyword search to find the answer.
- **Token F1 is not informative** here (about 0.25 for all systems): the model writes long answers and the references are short, so it mostly measures length. The automatic refusal detector also flagged one correct answer as a refusal (question 1, "not provided in the request"); the true false-refusal rate is 0.
- **Off-topic questions:** all three systems refused all 3 (too few questions to say more).

### 6.5 Error analysis

1. **The context contains no code.** The documentation pages include their code examples with markers such as `{* ../../docs_src/middleware/tutorial001.py hl[10:13] *}` instead of the code itself, so the retrieved chunks never contain the code. Every code example in an answer therefore comes from the model's memory, and several are wrong: `await background_tasks.add_task(...)` (not awaitable), `from fastapi import Annotated` (wrong module), `response: Response = Depends()`, or `if item_id != 5` instead of the docs' `item_id not in items`. Three answers even printed the raw marker (dense Q29, BM25 Q33, hybrid Q1). Inlining the example files into the chunks is the most promising improvement.
2. **Vocabulary gap (Q15, "let a frontend on a different domain call my API").** The docs call this CORS and the question never uses the word. BM25 retrieved the OAuth2 security page and answered with OAuth2 authentication: a confident wrong answer. Dense retrieved `tutorial/cors.md`, but the model still described generic CORS headers and never mentioned `CORSMiddleware`.
3. **Retrieval miss (Q36, "share common query parameters between endpoints").** The gold page (`dependencies/index.md`) did not appear in the dense top 10; all systems answered with Pydantic query-parameter models, a real but different feature.
4. **Right page, wrong reading (Q30, middleware order).** All three systems retrieved `tutorial/middleware.md` and all three got the order wrong (they said the innermost or first-added middleware runs first; the docs say the last added is outermost and runs first). A 3B model misreads a short, subtle sentence.
5. **Right page, wrong scenario (Q26, "do test functions need to be async?").** Retrieval put the advanced async-tests page first, so all three systems answered "Yes" (true only when using `AsyncClient`); the tutorial answer is "No".
6. **Earlier index version.** Before we removed non-documentation pages, the 404 question retrieved release-note and frontend chunks and the model invented an example. Filtering `release-notes.md` (39% of all chunks) fixed this retrieval (section 6.2).

## 7. Limitations

- Retrieval is scored at page level: a retrieved chunk from the right page counts as correct even if it is the wrong section. Page-level Hit@k therefore overstates section-level accuracy, which is why we also check answers manually.
- 47 questions written by us from the same documentation are a small test set; results show clear trends, not statistically precise differences.
- A 3B-parameter local model often misreads subtle passages and rarely follows the citation instruction reliably (section 6.5).
- Answer correctness and faithfulness were judged by an LLM (Claude), not by humans, and faithfulness was checked against the documentation pages (the answer file stores page names, not chunk text); a team spot-check is still pending.
- Future work: the code examples are not in the retrieved text (error analysis 1). Replacing the `{* ... *}` markers by the real example files before chunking, and giving the generator the code while indexing only the prose, is the most promising next step.
- One library's documentation (FastAPI); results may differ for other documentation styles.

## 8. Team members

| Name | ID |
|---|---|
| Biniyam Girma | ATE/7146/14 |
| Estifanos Yitayew | ATE/9512/14 |
| Simon Mesfin | ATE/7211/14 |
| Yared Kiros  | ATE/7702/14 |
| Yeabsira Addis | ATE/8574/14 |
| Yosef Ashebir | ATE/4638/14 |
