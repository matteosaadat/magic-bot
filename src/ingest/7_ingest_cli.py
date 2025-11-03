# ingest_cli.py
# ============================================================
# AI INSTRUCTION:
# Create an end-to-end CLI that:
#   1) Discovers Markdown files under --root (default: ../knowledge)
#   2) Parses YAML front-matter + body; chunk by headings with size guardrails (~400–800 tokens)
#   3) Normalizes text (import from 1_normalize.py)
#   4) Paraphrases in batch with local LLM (2_paraphrase.py), with caching & safety checks
#   5) Dedup: simhash pre-filter, then semantic dedup (3_dedup.py)
#   6) Embeds kept chunks (4_embed.py)
#   7) Upserts documents/chunks into SQLite and rebuilds FTS (5_index.py)
#   8) Builds/saves FAISS for chunk embeddings (5_index.py)
#
# CLI args:
#   --root ../knowledge
#   --db data/db/portfolio.db
#   --faiss data/index/faiss.index
#   --paraphrase / --no-paraphrase (default: --paraphrase)
#
# Output:
#   - Prints stats (docs, total_chunks, dropped_simhash, dropped_semantic, kept, elapsed)
#   - Writes JSONL run log to data/logs/ingest-YYYYMMDD.jsonl
#
# Implementation notes:
#   - Use `python-frontmatter` or a simple front-matter parser.
#   - Keep imports local in functions to avoid heavy load at import-time.
#   - Generate `doc_id` and `chunk_id` as stable hashes of source path + section offsets.
#   - Include a `main()` with `if __name__ == "__main__":`
# ============================================================

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import importlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Ensure we can import "ingest.*" even when running from project root or elsewhere
_THIS_FILE = Path(__file__).resolve()
_INGEST_DIR = _THIS_FILE.parent
_PROJECT_ROOT = _INGEST_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# -------------------------
# Lightweight utilities
# -------------------------

def sha1_hex(data: str) -> str:
    return hashlib.sha1(data.encode("utf-8")).hexdigest()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def estimate_tokens(text: str) -> int:
    """
    Dependency-light token estimate.
    Roughly: tokens ≈ max(1, round(chars/4)) bounded by word count.
    """
    chars = len(text)
    words = max(1, len(text.split()))
    approx = max(1, int(round(chars / 4)))
    return max(1, min(approx, words))

def _sql_exec_retry(conn: sqlite3.Connection, sql: str, params=None, retries: int = 8, delay: float = 0.25):
    """Execute SQL with brief retries on 'database is locked'."""
    for attempt in range(retries):
        try:
            return conn.execute(sql, {} if params is None else params)
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "database is locked" in msg or "database is busy" in msg:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                continue
            raise

# -------------------------
# Front matter parsing
# -------------------------

_FRONT_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)

def parse_front_matter(text: str) -> Tuple[Dict, str]:
    """
    Parse YAML front-matter if present. Falls back gracefully without PyYAML.
    Returns: (metadata_dict, body_text)
    """
    m = _FRONT_RE.match(text)
    if not m:
        return {}, text

    yaml_block, body = m.group(1), m.group(2)
    meta: Dict = {}
    try:
        import yaml  # optional
        meta = yaml.safe_load(yaml_block) or {}
        if not isinstance(meta, dict):
            meta = {}
    except Exception:
        # very simple key: value parser (best-effort)
        meta = {}
        for line in yaml_block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta, body

# -------------------------
# Markdown chunking
# -------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

def iter_heading_blocks(md: str) -> Iterator[Tuple[List[str], str]]:
    """
    Yields (heading_path, section_text) where heading_path is a list of headings
    leading to this block (e.g., ["Intro"], ["API","Usage"], ...).
    If no headings exist, yield one block with empty path.
    """
    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        yield ([], md.strip())
        return

    # Build ranges between headings
    headings: List[Tuple[int, str]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)

        # update heading path stack
        if not headings:
            headings = [(level, title)]
        else:
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, title))

        block = md[start:end].strip()
        yield ([t for _, t in headings], block)

def chunk_block_with_guardrails(
    heading_path: List[str],
    block: str,
    min_tokens: int = 400,
    max_tokens: int = 800,
) -> List[Tuple[List[str], str]]:
    """
    Split a block into sub-chunks targeting ~400–800 tokens.
    Strategy: split on paragraph boundaries, greedily pack until near max.
    """
    paras = [p.strip() for p in re.split(r"\n{2,}", block) if p.strip()]
    out: List[Tuple[List[str], str]] = []
    cur: List[str] = []
    cur_tok = 0

    def flush():
        nonlocal cur, cur_tok
        if cur:
            out.append((heading_path, "\n\n".join(cur).strip()))
            cur = []
            cur_tok = 0

    for p in paras:
        t = estimate_tokens(p)
        if cur_tok + t > max_tokens and cur_tok >= min_tokens:
            flush()
        cur.append(p)
        cur_tok += t

    flush()
    if not out and block.strip():
        out = [(heading_path, block.strip())]
    return out

def chunk_markdown(md: str) -> List[Tuple[List[str], str]]:
    chunks: List[Tuple[List[str], str]] = []
    for path, block in iter_heading_blocks(md):
        chunks.extend(chunk_block_with_guardrails(path, block))
    return chunks

# -------------------------
# Dynamic helpers for numbered modules
# -------------------------

def _import_attr(mod_a: str, mod_b: str, attr: str):
    """
    Try importing attr from module `mod_a`, then `mod_b`.
    This allows both 'ingest.1_normalize' and '1_normalize' depending on PYTHONPATH.
    """
    for mod in (mod_a, mod_b):
        try:
            return getattr(importlib.import_module(mod), attr)
        except Exception:
            continue
    return None

# -------------------------
# Normalization (1_normalize.py)
# -------------------------

def normalize_text(text: str) -> str:
    fn = _import_attr("ingest.1_normalize", "1_normalize", "normalize_text")
    if callable(fn):
        try:
            return fn(text)  # type: ignore[misc]
        except Exception:
            pass
    # lightweight fallback
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# -------------------------
# Paraphrase (2_paraphrase.py) + cache
# -------------------------

def _load_paraphrase_cache(path: Path) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    cache[obj["key"]] = obj["val"]
                except Exception:
                    continue
    return cache

def _append_paraphrase_cache(path: Path, items: Dict[str, str]) -> None:
    if not items:
        return
    with path.open("a", encoding="utf-8") as f:
        for k, v in items.items():
            f.write(json.dumps({"key": k, "val": v}, ensure_ascii=False) + "\n")

def paraphrase_batch(
    texts: List[str],
    cache_path: Path,
    enable: bool = True,
    max_shrink: float = 0.5,
    max_expand: float = 1.8,
) -> List[str]:
    """
    Calls 2_paraphrase.py batch paraphraser if available; otherwise returns original texts.
    Caches by sha1 of the normalized input text.
    Safety checks: prevent extreme shrink/expand.
    """
    cache = _load_paraphrase_cache(cache_path)
    out: List[str] = []
    to_store: Dict[str, str] = {}

    para_fn = _import_attr("ingest.2_paraphrase", "2_paraphrase", "paraphrase_texts")
    if not para_fn:
        # try single-text API
        one_fn = _import_attr("ingest.2_paraphrase", "2_paraphrase", "paraphrase_text")
        if one_fn:
            para_fn = lambda arr: [one_fn(a) for a in arr]  # type: ignore[func-returns-value]

    if not enable or not callable(para_fn):
        return texts

    keys = [sha1_hex(t) for t in texts]
    pending_idx = [i for i, k in enumerate(keys) if k not in cache]
    pending = [texts[i] for i in pending_idx]

    if pending:
        try:
            paras = para_fn(pending)  # type: ignore[misc]
        except Exception:
            paras = pending

        safe_paras = []
        for src, hypo in zip(pending, paras):
            if not isinstance(hypo, str) or not hypo.strip():
                safe_paras.append(src); continue
            r = len(hypo) / max(1, len(src))
            if r < max_shrink or r > max_expand:
                safe_paras.append(src)
            else:
                safe_paras.append(hypo)

        for i, s in enumerate(pending_idx):
            to_store[keys[s]] = safe_paras[i]

    for i, k in enumerate(keys):
        if k in cache:
            out.append(cache[k])
        elif k in to_store:
            out.append(to_store[k])
        else:
            out.append(texts[i])

    if to_store:
        ensure_dir(cache_path.parent)
        _append_paraphrase_cache(cache_path, to_store)

    return out

# -------------------------
# Dedup (3_dedup.py)
# -------------------------

def simhash_64(text: str) -> int:
    """
    Best-effort simhash: use 3_dedup.py if provided, else simple fallback.
    """
    fn = _import_attr("ingest.3_dedup", "3_dedup", "simhash_64")
    if callable(fn):
        try:
            return int(fn(text))  # type: ignore[misc]
        except Exception:
            pass
    # very simple fallback: hash of tokens
    tokens = re.findall(r"\w+", text.lower())
    h = 0
    for tok in tokens:
        h ^= int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
    return h & ((1 << 64) - 1)

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def semantic_dedup(
    texts: List[str],
    embeddings: Optional[List[List[float]]] = None,
    sim_threshold: float = 0.92,
) -> List[bool]:
    """
    Returns keep_mask for semantic dedup.
    Uses 3_dedup.py if available; otherwise cosine-sim iterative filter.
    """
    fn = _import_attr("ingest.3_dedup", "3_dedup", "semantic_dedup")
    if callable(fn):
        try:
            return list(fn(texts, embeddings=embeddings, sim_threshold=sim_threshold))  # type: ignore[misc]
        except Exception:
            pass

    # Fallback cosine sim using hashing bag embeddings if none provided
    import math
    def dot(a, b): return sum(x * y for x, y in zip(a, b))
    def norm(a): return math.sqrt(sum(x * x for x in a))

    if embeddings is None:
        embeddings = []
        for t in texts:
            vec = [0.0] * 256
            for tok in re.findall(r"\w+", t.lower()):
                h = int(hashlib.blake2b(tok.encode(), digest_size=2).hexdigest(), 16) % 256
                vec[h] += 1.0
            embeddings.append(vec)

    keep = [True] * len(texts)
    kept_vecs: List[List[float]] = []
    for i, v in enumerate(embeddings):
        if not kept_vecs:
            kept_vecs.append(v); continue
        n_v = norm(v) or 1.0
        too_similar = False
        for u in kept_vecs:
            sim = dot(u, v) / ((norm(u) or 1.0) * n_v)
            if sim >= sim_threshold:
                too_similar = True
                break
        if too_similar:
            keep[i] = False
        else:
            kept_vecs.append(v)
    return keep

# -------------------------
# Embeddings (4_embed.py)
# -------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Use 4_embed.py if available; else deterministic hashing vectors.
    """
    fn = _import_attr("ingest.4_embed", "4_embed", "embed_texts")
    if callable(fn):
        try:
            return list(fn(texts))  # type: ignore[misc]
        except Exception:
            pass

    # Fallback: 384-d hashing vectors
    vecs: List[List[float]] = []
    for t in texts:
        v = [0.0] * 384
        for tok in re.findall(r"\w+", t.lower()):
            h = int(hashlib.blake2b(tok.encode(), digest_size=3).hexdigest(), 16)
            v[h % 384] += 1.0
        vecs.append(v)
    return vecs

# -------------------------
# Indexing & FAISS (5_index.py)
# -------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    fn = _import_attr("ingest.5_index", "5_index", "init_sqlite")
    if callable(fn):
        try:
            return fn(db_path)  # type: ignore[misc]
        except Exception:
            pass

    # Longer timeout so SQLite waits for locks to clear
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)  # autocommit mode
    try:
        # Reduce lock contention and allow concurrent readers
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")  # 30s
    except Exception:
        pass

    _sql_exec_retry(conn, """
        CREATE TABLE IF NOT EXISTS documents(
          doc_id TEXT PRIMARY KEY, title TEXT, tags_json TEXT,
          tech_json TEXT, dates_json TEXT, source_path TEXT, content_hash TEXT
        )
    """)
    _sql_exec_retry(conn, """
        CREATE TABLE IF NOT EXISTS chunks(
          chunk_id TEXT PRIMARY KEY, doc_id TEXT, section_path TEXT,
          norm_text TEXT, token_count INT
        )
    """)
    try:
        _sql_exec_retry(conn, "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(norm_text, content='chunks', content_rowid='rowid')")
    except Exception:
        # FTS5 not available or already created; ignore
        pass
    return conn


def upsert_document(conn: sqlite3.Connection, **kwargs) -> None:
    fn = _import_attr("ingest.5_index", "5_index", "upsert_document")
    if callable(fn):
        try:
            fn(conn, **kwargs)  # type: ignore[misc]
            return
        except Exception:
            pass
    _sql_exec_retry(conn, """
        INSERT INTO documents(doc_id, title, tags_json, tech_json, dates_json, source_path, content_hash)
        VALUES(:doc_id, :title, :tags_json, :tech_json, :dates_json, :source_path, :content_hash)
        ON CONFLICT(doc_id) DO UPDATE SET
          title=excluded.title, tags_json=excluded.tags_json, tech_json=excluded.tech_json,
          dates_json=excluded.dates_json, source_path=excluded.source_path, content_hash=excluded.content_hash
    """, kwargs)

def upsert_chunk(conn: sqlite3.Connection, **kwargs) -> None:
    fn = _import_attr("ingest.5_index", "5_index", "upsert_chunk")
    if callable(fn):
        try:
            fn(conn, **kwargs)  # type: ignore[misc]
            return
        except Exception:
            pass
    _sql_exec_retry(conn, """
        INSERT INTO chunks(chunk_id, doc_id, section_path, norm_text, token_count)
        VALUES(:chunk_id, :doc_id, :section_path, :norm_text, :token_count)
        ON CONFLICT(chunk_id) DO UPDATE SET
          doc_id=excluded.doc_id, section_path=excluded.section_path,
          norm_text=excluded.norm_text, token_count=excluded.token_count
    """, kwargs)

def rebuild_fts(conn: sqlite3.Connection) -> None:
    fn = _import_attr("ingest.5_index", "5_index", "rebuild_fts")
    if callable(fn):
        try:
            fn(conn)  # type: ignore[misc]
            return
        except Exception:
            pass
    try:
        _sql_exec_retry(conn, "DELETE FROM chunks_fts")
        _sql_exec_retry(conn, "INSERT INTO chunks_fts(rowid, norm_text) SELECT rowid, norm_text FROM chunks")
    except Exception:
        pass

def build_and_save_faiss(embeddings: List[List[float]], ids: List[str], path: str) -> None:
    """
    Try to use faiss if available; else save a numpy .npz with embeddings + ids.
    """
    try:
        import numpy as np  # type: ignore
        import faiss  # type: ignore

        ensure_dir(Path(path).parent)
        arr = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(arr)
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        faiss.write_index(index, path)
        idspath = os.path.splitext(path)[0] + ".ids"
        with open(idspath, "w", encoding="utf-8") as f:
            for cid in ids:
                f.write(cid + "\n")
        return
    except Exception:
        pass

    # Fallbacks (npz, then json)
    try:
        import numpy as np  # type: ignore
        ensure_dir(Path(path).parent)
        base = os.path.splitext(path)[0]
        np.savez(base + ".npz", embeddings=np.array(embeddings, dtype="float32"), ids=np.array(ids, dtype=object))
    except Exception:
        base = os.path.splitext(path)[0]
        ensure_dir(Path(path).parent)
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({"ids": ids, "embeddings": embeddings}, f)

# -------------------------
# Discovery & processing
# -------------------------

def discover_markdown(root: Path) -> List[Path]:
    exts = {".md", ".mdx", ".markdown"}
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])

def section_path_str(headings: List[str]) -> str:
    return " > ".join(headings) if headings else ""

def build_doc_id(source_path: Path, meta: Dict) -> str:
    base = str(source_path.resolve())
    title = str(meta.get("title", "") or "")
    key = f"{base}::{title}"
    return sha1_hex(key)

def build_chunk_id(doc_id: str, section_path: str, local_idx: int, text: str) -> str:
    key = f"{doc_id}::{section_path}::{local_idx}::{sha1_hex(text)[:10]}"
    return sha1_hex(key)

# -------------------------
# Main pipeline
# -------------------------

def run_pipeline(
    root: Path,
    db_path: Path,
    faiss_path: Path,
    do_paraphrase: bool = True,
) -> Dict[str, int | float]:
    t0 = time.time()
    files = discover_markdown(root)

    # Prepare DB
    ensure_dir(db_path.parent)
    conn = init_db(str(db_path))
    conn.row_factory = sqlite3.Row

    # Stats
    docs_count = 0
    total_chunks = 0
    dropped_simhash = 0
    dropped_semantic = 0
    kept = 0

    # For FAISS
    all_ids: List[str] = []
    all_embs: List[List[float]] = []

    # Cache location
    cache_paraphrase = Path("data/cache/paraphrase.jsonl")

    for fp in files:
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        meta, body = parse_front_matter(raw)
        doc_id = build_doc_id(fp, meta)
        title = meta.get("title") or fp.stem
        tags_json = json.dumps(meta.get("tags", []), ensure_ascii=False)
        tech_json = json.dumps(meta.get("tech", []), ensure_ascii=False)
        content_hash = sha1_hex(body)

        def _json_safe(o):
            import datetime
            if isinstance(o, (datetime.date, datetime.datetime)):
                return o.isoformat()
            return o

        dates = meta.get("dates", {})
        # Convert any datetime/date objects to ISO strings
        if isinstance(dates, dict):
            dates = {k: _json_safe(v) for k, v in dates.items()}
        dates_json = json.dumps(dates, ensure_ascii=False)

        upsert_document(
            conn,
            doc_id=doc_id,
            title=title,
            tags_json=tags_json,
            tech_json=tech_json,
            dates_json=dates_json,
            source_path=str(fp),
            content_hash=content_hash,
        )
        docs_count += 1

        # Chunk & normalize
        chunked = chunk_markdown(body)
        norm_chunks: List[Tuple[str, str]] = []
        for headings, txt in chunked:
            section_path = section_path_str(headings)
            norm_txt = normalize_text(txt)
            if not norm_txt:
                continue
            norm_chunks.append((section_path, norm_txt))

        if not norm_chunks:
            continue

        # Paraphrase (batch)
        texts_for_para = [t for _, t in norm_chunks]
        paras = paraphrase_batch(
            texts_for_para,
            cache_path=cache_paraphrase,
            enable=do_paraphrase,
        )

        # Simhash prefilter
        seen: List[int] = []
        keep_mask_sim: List[bool] = []
        for para in paras:
            h = simhash_64(para)
            dup = any(hamming64(h, prev) <= 3 for prev in seen)
            if dup:
                keep_mask_sim.append(False); dropped_simhash += 1
            else:
                keep_mask_sim.append(True); seen.append(h)

        kept_sections_sim = [(norm_chunks[i][0], paras[i]) for i, k in enumerate(keep_mask_sim) if k]
        if not kept_sections_sim:
            continue

        # Embed (for semantic dedup and FAISS)
        texts_kept_sim = [t for _, t in kept_sections_sim]
        embs = embed_texts(texts_kept_sim)

        # Semantic dedup
        keep_mask_sem = semantic_dedup(texts_kept_sim, embeddings=embs, sim_threshold=0.92)
        kept_sections = [(kept_sections_sim[i][0], kept_sections_sim[i][1]) for i, k in enumerate(keep_mask_sem) if k]
        kept_embs = [embs[i] for i, k in enumerate(keep_mask_sem) if k]
        dropped_semantic += sum(1 for k in keep_mask_sem if not k)

        # Upsert chunks & collect for FAISS
        local_idx = 0
        for (section_path, text), vec in zip(kept_sections, kept_embs):
            total_chunks += 1
            token_count = estimate_tokens(text)
            chunk_id = build_chunk_id(doc_id, section_path, local_idx, text)
            local_idx += 1

            upsert_chunk(
                conn,
                chunk_id=chunk_id,
                doc_id=doc_id,
                section_path=section_path,
                norm_text=text,
                token_count=token_count,
            )

            all_ids.append(chunk_id)
            all_embs.append(vec)
            kept += 1

        conn.commit()

    # Rebuild FTS
    rebuild_fts(conn)
    conn.commit()

    # Build FAISS ...
    if all_ids and all_embs:
        ensure_dir(Path(faiss_path).parent)
        build_and_save_faiss(all_embs, all_ids, str(faiss_path))

    elapsed = round(time.time() - t0, 3)
    stats = {
        "docs": docs_count,
        "total_chunks": total_chunks,
        "dropped_simhash": dropped_simhash,
        "dropped_semantic": dropped_semantic,
        "kept": kept,
        "elapsed_seconds": elapsed,
    }

    try:
        conn.close()
    except Exception:
        pass

    return stats


# -------------------------
# Logging
# -------------------------

def write_run_log(stats: Dict[str, int | float]) -> Path:
    log_dir = Path("data/logs")
    ensure_dir(log_dir)
    stamp = dt.datetime.now().strftime("%Y%m%d")
    path = log_dir / f"ingest-{stamp}.jsonl"
    entry = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "stats": stats,
        "cwd": str(Path.cwd()),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return path

# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="End-to-end ingestion CLI (MD -> chunks -> index).")
    parser.add_argument("--root", default="knowledge", help="Root folder containing Markdown.")
    parser.add_argument("--db", default="data/db/portfolio.db", help="SQLite database path.")
    parser.add_argument("--faiss", default="data/index/faiss.index", help="FAISS index output path.")
    parser.add_argument("--paraphrase", dest="paraphrase", action="store_true", help="Enable paraphrasing.")
    parser.add_argument("--no-paraphrase", dest="paraphrase", action="store_false", help="Disable paraphrasing.")
    parser.set_defaults(paraphrase=True)

    args = parser.parse_args()
    root = Path(args.root)
    db_path = Path(args.db)
    faiss_path = Path(args.faiss)

    ensure_dir(db_path.parent)
    ensure_dir(faiss_path.parent)

    stats = run_pipeline(root=root, db_path=db_path, faiss_path=faiss_path, do_paraphrase=args.paraphrase)
    log_path = write_run_log(stats)

    print(
        "Ingest complete | "
        f"docs={stats['docs']} "
        f"total_chunks={stats['total_chunks']} "
        f"dropped_simhash={stats['dropped_simhash']} "
        f"dropped_semantic={stats['dropped_semantic']} "
        f"kept={stats['kept']} "
        f"elapsed={stats['elapsed_seconds']}s"
    )
    print(f"Run log: {log_path}")

if __name__ == "__main__":
    main()
