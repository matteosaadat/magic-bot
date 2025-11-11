# Creates bots/portfolio/data/{db, index} with a tiny FTS5 DB and FAISS+ids.npy
import os, sqlite3
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BOT = "portfolio"
DATA = ROOT / f"bots/{BOT}/data"
DBP  = DATA / "db" / f"{BOT}.db"
IDX  = DATA / "index" / "faiss.index"
IDS  = DATA / "index" / "ids.npy"

def ensure_dirs():
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    (DATA / "db").mkdir(parents=True, exist_ok=True)
    (DATA / "index").mkdir(parents=True, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DBP.as_posix())
    # FTS5 virtual table expected by retriever.py
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(id, text, source, tokenize='porter');")
    rows = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    if rows == 0:
        docs = [
            ("1", "Matteo-bot uses FastAPI and Ollama for local inference.", "portfolio.md"),
            ("2", "CI/CD uses GitHub Actions and Watchtower on EC2 to auto-update containers.", "cicd.md"),
            ("3", "Retrieval is hybrid: FAISS for vectors and SQLite FTS5 for lexical.", "architecture.md"),
            ("4", "Watchtower monitors GHCR images and pulls latest tags on schedule.", "ops.md"),
            ("5", "FAISS index stores embeddings from the BGE model.", "retrieval.md"),
        ]
        conn.executemany("INSERT INTO documents(id, text, source) VALUES (?,?,?)", docs)
        conn.commit()
    conn.close()

def init_faiss():
    import faiss
    dim = 64
    idx = faiss.IndexFlatIP(dim)
    rng = np.random.default_rng(42)
    vecs = rng.random((5, dim), dtype=np.float32)
    faiss.normalize_L2(vecs)
    idx.add(vecs)
    faiss.write_index(idx, IDX.as_posix())
    np.save(IDS.as_posix(), np.array(["1","2","3","4","5"]))

if __name__ == "__main__":
    ensure_dirs()
    init_db()
    init_faiss()
    print("✓ Bootstrapped:", DBP, "and", IDX)
