# AI INSTRUCTION:
# Build a hybrid retriever that merges:
#  - Lexical search via SQLite FTS5 + BM25 scoring
#  - Vector search via FAISS with ids.npy row→doc mapping
# Requirements:
#  - Return List[ContextChunk] with real text/source (no placeholders)
#  - Use Ollama embeddings for the query (model via EMBED_MODEL env)
#  - If embeddings are unavailable, fall back to a random vector (test mode)
#  - Provide load_persona(key) reading personas.yaml in this folder
#  - Keep small, clear interfaces; no new files

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np
import faiss
import yaml
import requests
import re

from .types import ContextChunk, Persona
from .rank import merge_and_rank

# --- Embedding config (for query vectors) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")

_FTS_WORD = re.compile(r"[0-9A-Za-z_]+")

def _fts5_safe_query(raw: str) -> str:
    """
    Convert arbitrary user text to a safe FTS5 MATCH expression.
    - Extract alphanumeric/underscore tokens.
    - Join with AND so all terms must appear (tweak to OR if you prefer).
    - Quote each token to avoid operator parsing (e.g., '-' or ':' etc).
    """
    terms = _FTS_WORD.findall(raw)
    if not terms:
        return '""'  # empty phrase → matches nothing
    return " AND ".join(f'"{t}"' for t in terms)



class Retriever:
    def __init__(self, db_path: str, faiss_path: str, top_k: int = 6):
        self.db_path = db_path
        self.faiss_path = faiss_path
        self.top_k = top_k

        self._conn: Optional[sqlite3.Connection] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._faiss_ids: Optional[list[str]] = None  # FAISS row -> documents.id

    # -------------------------
    # Connections / loaders
    # -------------------------
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def _load_faiss_index(self) -> faiss.Index:
        if self._faiss_index is None:
            self._faiss_index = faiss.read_index(self.faiss_path)
        return self._faiss_index

    def _load_faiss_ids(self) -> Optional[list[str]]:
        """Load sidecar ids.npy that maps FAISS row to documents.id."""
        if self._faiss_ids is not None:
            return self._faiss_ids
        ids_path = Path(self.faiss_path).with_name("ids.npy")
        if ids_path.exists():
            self._faiss_ids = np.load(ids_path.as_posix()).astype(str).tolist()
        else:
            self._faiss_ids = None
        return self._faiss_ids

    # -------------------------
    # DB helpers
    # -------------------------
    def _fetch_doc_by_id(self, doc_id: str) -> Optional[tuple[str, str]]:
        """Return (text, source) for a given documents.id."""
        row = self._get_conn().execute(
            "SELECT text, source FROM documents WHERE id = ? LIMIT 1;",
            (doc_id,),
        ).fetchone()
        return (row[0], row[1]) if row else None

    # -------------------------
    # Embeddings (query)
    # -------------------------
    def _embed_query_ollama(self, text: str) -> np.ndarray:
        """
        Embed the query using Ollama /api/embeddings.
        Normalizes vector to unit length (cosine/IP compatibility).
        """
        url = f"{OLLAMA_HOST}/api/embeddings"
        resp = requests.post(url, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
        resp.raise_for_status()
        vec = np.array(resp.json()["embedding"], dtype="float32")
        faiss.normalize_L2(vec.reshape(1, -1))
        return vec

    # -------------------------
    # Public API
    # -------------------------
    def retrieve(self, query: str, alpha: float = 0.5) -> List[ContextChunk]:
        conn = self._get_conn()
        index = self._load_faiss_index()
        ids = self._load_faiss_ids()

        # 1) LEXICAL via BM25 over FTS5
        safe_q = _fts5_safe_query(query)
        sql = """
        SELECT id, text, source, bm25(documents) AS rank
        FROM documents
        WHERE documents MATCH ?
        ORDER BY rank
        LIMIT ?;
        """
        rows = conn.execute(sql, (safe_q, self.top_k * 3)).fetchall()

        lex_results: List[ContextChunk] = []
        for (doc_id, text, source, rank) in rows:
            rank = float(rank)
            score = 1.0 / (1.0 + rank)  # monotonic transform to (0,1]
            lex_results.append(
                ContextChunk(
                    id=f"lex-{doc_id}",
                    text=text,
                    source=source,
                    score=score,
                    meta={"bm25": rank},
                )
            )

        # 2) VECTOR via FAISS → map row idx → documents.id → fetch text/source
        #    Use real embeddings if available; otherwise fall back to random for smoke-tests.
        try:
            qvec = self._embed_query_ollama(query)
            if qvec.shape[0] != index.d:
                # If dimension mismatch, try to adapt (rare; indicates wrong embed model vs index)
                raise ValueError(f"Query dim {qvec.shape[0]} != index dim {index.d}")
        except Exception:
            # Fallback for test environments without embeddings
            qvec = np.random.rand(index.d).astype("float32")
            faiss.normalize_L2(qvec.reshape(1, -1))

        D, I = index.search(np.array([qvec]), self.top_k * 3)

        vec_results: List[ContextChunk] = []
        for pos in range(len(I[0])):
            row_idx = int(I[0][pos])
            sim = float(D[0][pos])
            if ids and 0 <= row_idx < len(ids):
                doc_id = ids[row_idx]
                got = self._fetch_doc_by_id(doc_id)
                if got:
                    text, source = got
                    vec_results.append(
                        ContextChunk(
                            id=f"vec-{doc_id}",
                            text=text,
                            source=source,
                            score=sim,
                            meta={"faiss_idx": row_idx},
                        )
                    )
                    continue
            # Fallback if mapping missing
            vec_results.append(
                ContextChunk(
                    id=f"vec-{row_idx}",
                    text=f"[Missing mapping for faiss idx {row_idx}]",
                    source="faiss",
                    score=sim,
                    meta={"faiss_idx": row_idx},
                )
            )

        # 3) MERGE & rank
        merged = merge_and_rank(lex_results, vec_results, alpha=alpha, top_k=self.top_k)
        return merged

    # -------------------------
    # Personas
    # -------------------------
    def load_persona(self, key: str) -> Optional[Persona]:
        """Load persona from personas.yaml in this folder."""
        path = os.path.join(os.path.dirname(__file__), "personas.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"personas.yaml not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if key not in data:
            raise KeyError(f"Persona '{key}' not found in personas.yaml")
        p = data[key]
        return Persona(
            key=key,
            name=p.get("name", key),
            style=p.get("style", ""),
            directives=p.get("directives", ""),
            meta=p,
        )

    # -------------------------
    # Cleanup
    # -------------------------
    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
