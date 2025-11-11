# ===============================================
# tests/test_search.py
# -----------------------------------------------
# Self-contained smoke test for the search layer.
# Creates a tiny SQLite FTS5 DB + FAISS index in
# tests/_tmp_search/, runs Retriever, prints hits.
# ===============================================

import os
import sys
import shutil
import sqlite3
from pathlib import Path
import numpy as np

# Make project root importable (so `src` is on sys.path)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.search import Retriever  # uses your real retriever

TMP_DIR = Path("tests/_tmp_search")
DB_PATH = TMP_DIR / "portfolio.db"
FAISS_PATH = TMP_DIR / "faiss.index"
IDS_PATH = TMP_DIR / "ids.npy"


def setup_tmp_env():
    """Create a tiny FTS5 DB + FAISS index with a sidecar ids.npy mapping."""
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # --- SQLite FTS5 documents table ---
    conn = sqlite3.connect(DB_PATH.as_posix())
    # Virtual FTS5 table; porter stemmer is fine for the test
    conn.execute("CREATE VIRTUAL TABLE documents USING fts5(id, text, source, tokenize='porter');")

    docs = [
        ("1", "Matteo-bot uses FastAPI and Ollama for local inference.", "portfolio.md"),
        ("2", "CI/CD uses GitHub Actions and Watchtower on EC2 to auto-update containers.", "cicd.md"),
        ("3", "Retrieval is hybrid: FAISS for vectors and SQLite FTS5 for lexical.", "architecture.md"),
        ("4", "Watchtower monitors GHCR images and pulls latest tags on schedule.", "ops.md"),
        ("5", "FAISS index stores embeddings from the BGE model.", "retrieval.md"),
    ]
    conn.executemany("INSERT INTO documents(id, text, source) VALUES (?, ?, ?);", docs)
    conn.commit()
    conn.close()

    # --- FAISS index + ids.npy mapping ---
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("FAISS is not installed. Install faiss-cpu in your venv.") from e

    dim = 64  # tiny test dimension
    index = faiss.IndexFlatIP(dim)

    rng = np.random.default_rng(42)
    vecs = rng.random((len(docs), dim), dtype=np.float32)
    faiss.normalize_L2(vecs)  # makes IP behave like cosine if you normalize queries too

    index.add(vecs)
    faiss.write_index(index, FAISS_PATH.as_posix())

    # Save FAISS row → document.id mapping
    doc_ids = np.array([d[0] for d in docs])
    np.save(IDS_PATH.as_posix(), doc_ids)


def run_search(query: str, top_k: int = 5, alpha: float = 0.5):
    """Instantiate Retriever on the temp artifacts, print merged results."""
    r = Retriever(
        db_path=DB_PATH.as_posix(),
        faiss_path=FAISS_PATH.as_posix(),
        top_k=top_k,
    )
    chunks = r.retrieve(query=query, alpha=alpha)
    print(f"\nQuery: {query}\nTop {top_k} (alpha={alpha})")
    for i, c in enumerate(chunks, start=1):
        preview = c.text[:140].replace("\n", " ")
        print(f"{i:>2}. id={c.id:<8} score={c.score:.3f} src={c.source:<14} :: {preview}")
    r.close()


if __name__ == "__main__":
    setup_tmp_env()
    try:
        run_search("watchtower")
        run_search("FAISS")
        run_search("FastAPI Ollama")
    finally:
        # Comment out to inspect files:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
