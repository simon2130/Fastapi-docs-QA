"""Split the FastAPI Markdown docs into heading-based chunks -> data/chunks.jsonl

Setup (run once from the repo root):
    git clone --depth 1 https://github.com/fastapi/fastapi data/fastapi
Then:
    python src/ingest.py
"""
import json
import re
from pathlib import Path

DOCS_DIR = Path("data/fastapi/docs/en/docs")
OUT = Path("data/chunks.jsonl")
MAX_CHARS = 1200   # split sections longer than this at paragraph boundaries
MIN_CHARS = 80     # drop tiny fragments

# Pages that are not documentation of the framework itself. release-notes.md alone is ~40% of all
# chunks and is full of keywords (404, response_model, ...) that pull it above the real docs.
EXCLUDE_FILES = {"release-notes.md", "fastapi-people.md", "external-links.md", "newsletter.md",
                 "management.md", "management-tasks.md", "contributing.md", "help-fastapi.md",
                 "translations.md", "translation-banner.md", "_llm-test.md", "agent-banner.md"}
EXCLUDE_DIRS = {"about", "resources", "css", "js", "img"}


def clean_heading(h):
    """'Query Parameters { #query-parameters }' -> 'Query Parameters'"""
    return re.sub(r"\s*\{\s*#[^}]*\}\s*$", "", h).strip()


def is_excluded(rel):
    return rel in EXCLUDE_FILES or rel.split("/")[0] in EXCLUDE_DIRS


def split_by_headings(text):
    """Return [(heading, body)] splitting on #, ## and ### (ignores headings inside code fences)."""
    sections, heading, buf, in_code = [], "", [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,3}\s", line):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = clean_heading(line.lstrip("# ").strip()), []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def split_long(text, max_chars):
    """Greedy paragraph packing so no chunk is (much) longer than max_chars."""
    chunks, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) > max_chars:
            chunks.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def strip_front_matter(text):
    return re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)


def build_chunks(max_chars=MAX_CHARS, docs_dir=DOCS_DIR, exclude=True, with_title=True):
    """Return the list of chunk dicts.

    exclude:    drop non-documentation pages (see EXCLUDE_FILES / EXCLUDE_DIRS)
    with_title: store the page title (first H1) on every chunk so retrieval knows which page it is from
    """
    if not docs_dir.exists():
        raise SystemExit(f"{docs_dir} not found. Clone the repo first (see top of file).")
    chunks = []
    for path in sorted(docs_dir.rglob("*.md")):
        rel = path.relative_to(docs_dir).as_posix()
        if exclude and is_excluded(rel):
            continue
        text = strip_front_matter(path.read_text(encoding="utf-8"))
        m = re.search(r"^#\s+(.+)$", text, flags=re.M)
        title = clean_heading(m.group(1)) if m else ""
        for heading, body in split_by_headings(text):
            for piece in split_long(body, max_chars):
                if len(piece) < MIN_CHARS:
                    continue
                rec = {"id": len(chunks), "source": rel, "heading": heading, "text": piece}
                if with_title:
                    rec["title"] = title
                chunks.append(rec)
    return chunks


def main():
    chunks = build_chunks()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for rec in chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks to {OUT}")


if __name__ == "__main__":
    main()
