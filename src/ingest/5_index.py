# AI INSTRUCTION:
# Build hybrid indexes:
#   - SQLite FTS5 for lexical search over normalized/paraphrased text
#   - FAISS (cosine) for vector search over embeddings
#
# Implement:
# - SQLite schema:
#     documents(doc_id TEXT PRIMARY KEY, title TEXT, tags_json TEXT, tech_json TEXT, dates_json TEXT, source_path TEXT, content_hash TEXT)
#     chunks(chunk_id TEXT PRIMARY KEY, doc_id TEXT, section_path TEXT, norm_text TEXT, token_count INT)
#     chunks_fts (FTS5 virtual table over norm_text with external content)
# - Functions:
#     - `init_sqlite(db_path: str) -> "sqlite3.Connection"`
#     - `upsert_document(conn, *, doc_id, title, tags_json, tech_json, dates_json, source_path, content_hash) -> None`
#     - `upsert_chunk(conn, *, chunk_id, doc_id, section_path, norm_text, token_count) -> None`
#     - `rebuild_fts(conn) -> None`  # sync FTS with chunks table
# - FAISS:
#     - `build_faiss(embs: "np.ndarray") -> "faiss.Index"`
#     - `save_faiss(index_path: str, index: "faiss.Index", ids: list[str]) -> None`
#     - `load_faiss(index_path: str) -> tuple["faiss.Index", list[str]]`
#
# __main__:
# - If DB absent, create tables.
# - Print counts from documents/chunks.
#
# Notes:
# - Use `content=chunks` layout for FTS5 and `content_rowid='rowid'`.
# - Guard for missing FTS5 (helpful error).

from __future__ import annotations

import json
import os
import sqlite3
from typing import List, Tuple

import numpy as np  # type: ignore

# ---- Lazy import FAISS to avoid heavy import cost if not used ----
def _import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise RuntimeError(
            "FAISS is required for vector indexing. Install `faiss-cpu` (or `faiss-gpu`) "
            "e.g. `pip install faiss-cpu`."
        ) from e


# ============================================================================
# SQLite (FTS5) INDEX
# ============================================================================

_SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT,
    tags_json    TEXT,
    tech_json    TEXT,
    dates_json   TEXT,
    source_path  TEXT,
    content_hash TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    section_path TEXT,
    norm_text    TEXT,
    token_count  INT,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

-- External-content FTS5: stores only the inverted index, reads text from chunks.norm_text via rowid
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    norm_text,
    content='chunks',
    content_rowid='rowid'
);
"""


def _ensure_fts5_available(conn: sqlite3.Connection) -> None:
    """
    Create a tiny temp FTS5 table to verify FTS5 support. Raise a helpful error if missing.
    """
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS __fts5_probe")
    except sqlite3.OperationalError as e:
        # Common when Python/SQLite builds without FTS5
        raise RuntimeError(
            "SQLite FTS5 is not available in your Python sqlite3 build. "
            "You need a SQLite with FTS5 enabled. On many systems, installing `pysqlite3-binary` "
            "or upgrading Python can help."
        ) from e


def init_sqlite(db_path: str) -> sqlite3.Connection:
    """
    Initialize (or connect to) the SQLite DB and create schema if missing.
    Returns an open connection.
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    _ensure_fts5_available(conn)

    # Create schema
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def upsert_document(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    title: str | None,
    tags_json: str | None,
    tech_json: str | None,
    dates_json: str | None,
    source_path: str | None,
    content_hash: str | None,
) -> None:
    """
    Insert or update a document row by doc_id.
    """
    conn.execute(
        """
        INSERT INTO documents (doc_id, title, tags_json, tech_json, dates_json, source_path, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            title=excluded.title,
            tags_json=excluded.tags_json,
            tech_json=excluded.tech_json,
            dates_json=excluded.dates_json,
            source_path=excluded.source_path,
            content_hash=excluded.content_hash
        """,
        (doc_id, title, tags_json, tech_json, dates_json, source_path, content_hash),
    )
    conn.commit()


def upsert_chunk(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    doc_id: str,
    section_path: str | None,
    norm_text: str,
    token_count: int | None,
) -> None:
    """
    Insert or update a chunk row by chunk_id.
    """
    conn.execute(
        """
        INSERT INTO chunks (chunk_id, doc_id, section_path, norm_text, token_count)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            doc_id=excluded.doc_id,
            section_path=excluded.section_path,
            norm_text=excluded.norm_text,
            token_count=excluded.token_count
        """,
        (chunk_id, doc_id, section_path, norm_text, token_count),
    )
    conn.commit()


def rebuild_fts(conn: sqlite3.Connection) -> None:
    """
    Rebuild FTS external-content index from the chunks table.
    Uses the special 'rebuild' command supported by FTS5.
    """
    # Ensure the virtual table exists (init_sqlite should have created it)
    conn.execute("DELETE FROM chunks_fts")
    # This special command instructs FTS5 to repopulate from content='chunks'
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild');")
    conn.commit()


# ============================================================================
# FAISS (COSINE) INDEX
# ============================================================================

def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("Expected 2D array for embeddings")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def build_faiss(embs: np.ndarray):
    """
    Build a cosine-similarity FAISS index (Inner Product over L2-normalized vectors).
    Returns a faiss.Index with all vectors added.
    """
    faiss = _import_faiss()
    if embs.size == 0:
        # Create an empty index with unknown dim? We need dim to instantiate.
        # Fall back to 0-d guard; caller should handle empty case.
        raise ValueError("Cannot build FAISS index from empty embeddings")
    if embs.ndim != 2:
        raise ValueError("embs must be 2D (N, D)")

    embs = embs.astype(np.float32, copy=False)
    embs = _l2_normalize_rows(embs)

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine via inner product on unit vectors
    index.add(embs)               # vectors stored in the order given
    return index


def save_faiss(index_path: str, index, ids: List[str]) -> None:
    """
    Persist FAISS index and aligned IDs.
    Writes two files:
      - {index_path}            : FAISS index binary via faiss.write_index
      - {index_path}.ids.json   : JSON list of string IDs in the same order as vectors in the index
    """
    faiss = _import_faiss()
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)

    # Save the index
    faiss.write_index(index, index_path)

    # Save the aligned IDs as JSON
    ids_path = f"{index_path}.ids.json"
    with open(ids_path, "w", encoding="utf-8") as fh:
        json.dump(ids, fh, ensure_ascii=False)


def load_faiss(index_path: str) -> Tuple[object, List[str]]:
    """
    Load FAISS index and aligned IDs saved by save_faiss().
    Returns (index, ids).
    """
    faiss = _import_faiss()
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    ids_path = f"{index_path}.ids.json"
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"FAISS IDs sidecar not found: {ids_path}")

    index = faiss.read_index(index_path)
    with open(ids_path, "r", encoding="utf-8") as fh:
        ids: List[str] = json.load(fh)

    # Basic sanity: vector count should match ids length
    if hasattr(index, "ntotal") and index.ntotal != len(ids):
        raise RuntimeError(f"FAISS index size ({index.ntotal}) does not match ids ({len(ids)})")

    return index, ids


# ============================================================================
# __main__: ensure DB exists and print basic counts
# ============================================================================

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?;", (name,))
    return cur.fetchone() is not None


def _count(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.execute(f"SELECT COUNT(*) FROM {table};")
    row = cur.fetchone()
    return int(row[0]) if row else 0


def _print_counts(conn: sqlite3.Connection) -> None:
    docs = _count(conn, "documents") if _table_exists(conn, "documents") else 0
    chs = _count(conn, "chunks") if _table_exists(conn, "chunks") else 0
    print(f"documents: {docs}")
    print(f"chunks:    {chs}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid index bootstrap: SQLite (FTS5) + FAISS helpers")
    parser.add_argument("--db", required=True, help="Path to SQLite database file (will be created if missing)")
    parser.add_argument("--rebuild-fts", action="store_true", help="Rebuild FTS5 index from chunks table")
    args = parser.parse_args()

    conn = init_sqlite(args.db)
    if args.rebuild_fts:
        rebuild_fts(conn)
        print("FTS5 rebuilt from chunks.")

    _print_counts(conn)
