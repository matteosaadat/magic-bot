# ============================================================
# AI INSTRUCTION (READ ME):
# - This project uses NUMBERED modules (1_normalize.py, 4_embed.py, 5_index.py).
# - NEVER do: `from 1_normalize import X` (invalid in Python).
# - ALWAYS import dynamically via importlib and the "ingest." path.
#   Example:
#       import importlib
#       embed_fn = importlib.import_module("ingest.4_embed").embed_texts
# ============================================================

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Make sure project root is importable (so "ingest.*" works when run from repo root)
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# -------------------------
# Small utilities
# -------------------------

def _import_attr(mod_a: str, mod_b: str, attr: str):
    # Try ingest.<mod> first, then bare module name.
    for m in (f"ingest.{mod_a}", mod_a, f"ingest.{mod_b}", mod_b):
        try:
            return getattr(importlib.import_module(m), attr)
        except Exception:
            continue
    return None

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
    except Exception:
        pass
    return conn

def _preview(text: str, limit: int = 240) -> str:
    t = " ".join(text.strip().split())
    return t if len(t) <= limit else t[:limit - 1] + "…"

def _normalize_scores(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 1.0 for k, _ in pairs}
    return {k: (s - lo) / (hi - lo) for k, s in pairs}

# -------------------------
# FTS search (robust bm25 fallback)
# -------------------------

def fts_search(conn: sqlite3.Connection, query: str, k: int = 20) -> List[Tuple[int, float]]:
    """
    Returns list[(rowid, score)] from FTS.
    - If bm25() exists: use it (lower is better; we invert so higher is better).
    - Else: compute Python-side term frequency on matched rows so lex isn't always 0.
    """
    # Does chunks_fts exist?
    try:
        conn.execute("SELECT 1 FROM chunks_fts LIMIT 1")
    except sqlite3.OperationalError:
        # crude LIKE fallback over base table
        cur = conn.execute(
            "SELECT rowid, norm_text FROM chunks WHERE norm_text LIKE ? LIMIT ?",
            (f"%{query}%", k),
        )
        rows = cur.fetchall()
        q_terms = [t for t in re.findall(r"\w+", query.lower()) if t]
        scored = []
        for r in rows:
            text = r["norm_text"] or ""
            hits = sum(text.lower().count(t) for t in q_terms)
            scored.append((int(r["rowid"]), float(hits)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # Probe bm25()
    bm25_ok = True
    try:
        conn.execute("SELECT bm25(chunks_fts) FROM chunks_fts LIMIT 1")
    except Exception:
        bm25_ok = False

    if bm25_ok:
        # Use bm25 and invert so higher=better
        sql = """
            SELECT rowid, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
        """
        rows = conn.execute(sql, (query, k)).fetchall()
        return [(int(r["rowid"]), -float(r["score"])) for r in rows]

    # No bm25(): match with FTS, then compute TF in Python
    sql = "SELECT rowid, norm_text FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?"
    rows = conn.execute(sql, (query, k)).fetchall()
    q_terms = [t for t in re.findall(r"\w+", query.lower()) if t]
    scored = []
    for r in rows:
        text = r["norm_text"] or ""
        hits = sum(text.lower().count(t) for t in q_terms)
        scored.append((int(r["rowid"]), float(hits)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# -------------------------
# Vector search (FAISS or fallbacks)
# -------------------------

def _load_faiss_index(path: Path):
    try:
        import faiss  # type: ignore
        index = faiss.read_index(str(path))
        return ("faiss", index)
    except Exception:
        pass
    # Fallbacks
    base = path.with_suffix("")  # strip .index
    npz = base.with_suffix(".npz")
    jsn = base.with_suffix(".json")
    if npz.exists():
        import numpy as np  # type: ignore
        data = np.load(npz, allow_pickle=True)
        embs = data["embeddings"].astype("float32")
        ids = [str(x) for x in data["ids"].tolist()]
        return ("npz", (embs, ids))
    if jsn.exists():
        with open(jsn, "r", encoding="utf-8") as f:
            obj = json.load(f)
        import numpy as np  # type: ignore
        embs = np.array(obj["embeddings"], dtype="float32")
        ids = [str(x) for x in obj["ids"]]
        return ("json", (embs, ids))
    return (None, None)

def _embed_query(q: str):
    fn = _import_attr("4_embed", "4_embed", "embed_texts")
    if callable(fn):
        try:
            vec = fn([q])  # type: ignore[misc]
            return vec[0]
        except Exception:
            pass
    # hashing fallback (dim=384)
    import hashlib
    vec = [0.0] * 384
    for tok in q.lower().split():
        h = int(hashlib.blake2b(tok.encode(), digest_size=3).hexdigest(), 16) % 384
        vec[h] += 1.0
    return vec

def vec_search(conn: sqlite3.Connection, faiss_path: Path, query: str, k: int = 50) -> List[Tuple[int, float]]:
    """
    Returns list[(rowid, score)] by nearest-neighbor over embeddings.
    If FAISS is present, uses it; otherwise uses local .npz/.json fallback arrays.
    """
    mode, idx = _load_faiss_index(faiss_path)
    if not mode:
        return []

    # Map chunk_id -> rowid for joining
    rows = conn.execute("SELECT rowid, chunk_id FROM chunks").fetchall()
    id2rowid = {str(r["chunk_id"]): int(r["rowid"]) for r in rows}

    import numpy as np  # type: ignore
    qv = _embed_query(query)
    qv = np.array(qv, dtype="float32")
    if mode == "faiss":
        import faiss  # type: ignore
        faiss.normalize_L2(qv.reshape(1, -1))
        D, I = idx.search(qv.reshape(1, -1), k)
        out: List[Tuple[int, float]] = []
        for score, idx_row in zip(D[0].tolist(), I[0].tolist()):
            if idx_row < 0:
                continue
            # We need ids file to map index->chunk_id; try sibling .ids next to .index
            ids_file = faiss_path.with_suffix("").with_suffix(".ids")
            chunk_id = None
            if ids_file.exists():
                with open(ids_file, "r", encoding="utf-8") as f:
                    ids = [line.strip() for line in f]
                if idx_row < len(ids):
                    chunk_id = ids[idx_row]
            if chunk_id and chunk_id in id2rowid:
                out.append((id2rowid[chunk_id], float(score)))
        return out

    # npz/json fallback: brute-force cosine
    embs, ids = idx
    # normalize
    norms = (embs ** 2).sum(axis=1) ** 0.5
    qn = (qv ** 2).sum() ** 0.5 or 1.0
    sims = (embs @ qv) / (norms * qn + 1e-9)
    order = np.argsort(-sims)[:k].tolist()
    out: List[Tuple[int, float]] = []
    for j in order:
        cid = ids[j]
        if cid in id2rowid:
            out.append((id2rowid[cid], float(sims[j])))
    return out

# -------------------------
# Fetch + merge results
# -------------------------

def _fetch_chunks(conn: sqlite3.Connection, rowids: List[int]) -> Dict[int, sqlite3.Row]:
    if not rowids:
        return {}
    qmarks = ",".join("?" for _ in rowids)
    sql = f"SELECT rowid, doc_id, section_path, norm_text FROM chunks WHERE rowid IN ({qmarks})"
    rows = conn.execute(sql, rowids).fetchall()
    return {int(r["rowid"]): r for r in rows}

def _fetch_docs(conn: sqlite3.Connection, doc_ids: List[str]) -> Dict[str, sqlite3.Row]:
    if not doc_ids:
        return {}
    qmarks = ",".join("?" for _ in doc_ids)
    sql = f"SELECT doc_id, title, source_path FROM documents WHERE doc_id IN ({qmarks})"
    rows = conn.execute(sql, doc_ids).fetchall()
    return {str(r["doc_id"]): r for r in rows}

def hybrid_search(conn: sqlite3.Connection, faiss_path: Path, query: str,
                  k_lex: int = 20, k_vec: int = 50, alpha: float = 0.5, top_final: int = 10):
    # 1) lexical
    lex_pairs = fts_search(conn, query, k=k_lex)  # [(rowid, score)]
    lex_norm = _normalize_scores([(str(rid), sc) for rid, sc in lex_pairs])

    # 2) vector
    vec_pairs = vec_search(conn, faiss_path, query, k=k_vec)  # [(rowid, score)]
    vec_norm = _normalize_scores([(str(rid), sc) for rid, sc in vec_pairs])

    # 3) merge
    keys = set(lex_norm.keys()) | set(vec_norm.keys())
    merged: List[Tuple[int, float, float, float]] = []
    for k in keys:
        l = lex_norm.get(k, 0.0)
        v = vec_norm.get(k, 0.0)
        s = alpha * l + (1.0 - alpha) * v
        merged.append((int(k), s, l, v))
    merged.sort(key=lambda x: x[1], reverse=True)
    if top_final:
        merged = merged[:top_final]

    # 4) fetch metadata
    chunks = _fetch_chunks(conn, [rid for rid, *_ in merged])
    doc_ids = list({str(chunks[rid]["doc_id"]) for rid, *_ in merged if rid in chunks})
    docs = _fetch_docs(conn, doc_ids)

    # 5) build printable results
    results = []
    for rid, score, lsc, vsc in merged:
        ch = chunks.get(rid)
        if not ch:
            continue
        doc = docs.get(str(ch["doc_id"]))
        title = doc["title"] if doc else "(untitled)"
        section = ch["section_path"] or ""
        text = ch["norm_text"]
        results.append({
            "rowid": rid,
            "title": title,
            "section": section,
            "score": round(score, 3),
            "lex": round(lsc, 3),
            "vec": round(vsc, 3),
            "preview": _preview(text, 260),
        })
    return results

# -------------------------
# CLI
# -------------------------

def main():
    p = argparse.ArgumentParser(description="Hybrid search over SQLite FTS5 and FAISS.")
    p.add_argument("--db", required=True, help="SQLite DB path")
    p.add_argument("--faiss", required=True, help="FAISS index path (or .npz/.json fallback)")
    p.add_argument("--query", required=True, help="Query text")
    p.add_argument("--k-lex", type=int, default=20, help="FTS candidates")
    p.add_argument("--k-vec", type=int, default=50, help="Vector candidates")
    p.add_argument("--alpha", type=float, default=0.5, help="Weight for lexical (1-alpha for vector)")
    p.add_argument("--top-final", type=int, default=10, help="Final results to display")
    p.add_argument("--show", type=int, default=260, help="Preview characters to show")
    args = p.parse_args()

    conn = _connect(args.db)
    results = hybrid_search(conn, Path(args.faiss), args.query,
                            k_lex=args.k_lex, k_vec=args.k_vec,
                            alpha=args.alpha, top_final=args.top_final)

    if not results:
        print("No results.")
        return

    print(f"\nQuery: {args.query}")
    print(f"Results (alpha={args.alpha}, top={args.top_final}):\n")
    for i, r in enumerate(results, 1):
        sec = f" — {r['section']}" if r['section'] else ""
        print(f"{i}. {r['title']}{sec}")
        print(f"   score={r['score']} (lex={r['lex']}, vec={r['vec']})  rowid={r['rowid']}")
        print(f"   {r['preview']}\n")

if __name__ == "__main__":
    main()
