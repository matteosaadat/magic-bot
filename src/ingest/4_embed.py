# AI INSTRUCTION:
# CLI to embed deduplicated chunks into SQLite/FAISS using a local SBERT model.
# Self-contained version (no lib/ folder). Uses sentence-transformers locally.
# ------------------------------------------------------------------------------

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Silence tokenizer parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ==============================================================
# === Local embedding utility (no external API calls) ==========
# ==============================================================

_MODEL = None
_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")  # cpu | cuda


def _get_model(local_only: bool = False):
    """Lazy-load and cache a single global SentenceTransformer model."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # heavy import delayed

        kwargs = {"device": _DEVICE}
        if local_only:
            kwargs["local_files_only"] = True
        _MODEL = SentenceTransformer(_MODEL_NAME, **kwargs)
    return _MODEL


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("Expected a 2D array for normalization")
    norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def embed_texts(
    texts: List[str],
    batch_size: int = 64,
    normalize: bool = True,
    local_only: bool = False,
) -> np.ndarray:
    """
    Embed a list of strings into a (N, D) float32 numpy array using a local model.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = _get_model(local_only=local_only)

    embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        vecs = model.encode(
            chunk,
            batch_size=min(batch_size, len(chunk)),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        embs.append(vecs)

    out = np.vstack(embs).astype(np.float32, copy=False)
    if normalize:
        out = _l2_normalize(out)
    return out


def save_embeddings(path: str, embs: np.ndarray, ids: List[str]) -> None:
    if embs.ndim != 2:
        raise ValueError("embs must be 2D (N, D)")
    if len(ids) != embs.shape[0]:
        raise ValueError("len(ids) must match number of rows in embs")
    ids_arr = np.array(ids, dtype=object)
    np.savez_compressed(path, embs=embs.astype(np.float32, copy=False), ids=ids_arr)


# ==============================================================
# === CLI + file IO helpers ====================================
# ==============================================================

def _read_all_lines(input_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Read *.txt files under input_dir (one chunk per line).
    Returns texts + unique IDs (file:line).
    """
    texts, ids = [], []
    for p in sorted(input_dir.rglob("*.txt")):
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    texts.append(line)
                    ids.append(f"{p.name}:{i}")
        except Exception as e:
            print(f">> WARN: failed to read {p}: {e}", file=sys.stderr)
    return texts, ids


def _heartbeat(i: int, total: int, t0: float, every: int = 1000) -> None:
    if i % every == 0 and i > 0:
        dt = time.time() - t0
        rate = i / max(dt, 1e-6)
        print(f">> Embedded {i}/{total} chunks  ({rate:.1f} ch/s)", flush=True)


def _save_to_faiss_and_db(embs: np.ndarray, ids: List[str], db_path: Path, faiss_path: Path) -> None:
    """
    Persist embeddings:
      1) Save a .npz snapshot for debugging
      2) Save a FAISS IndexFlatIP (cosine-ready because we normalized vectors)
      3) Save a sidecar ids file next to the index for lookups
    """
    # 1) Snapshot (.npz)
    out_npz = faiss_path.with_suffix(".npz")
    save_embeddings(str(out_npz), embs, ids)
    print(f">> Saved embeddings snapshot: {out_npz}", flush=True)

    # 2) FAISS index (Inner Product == Cosine since embs are L2-normalized)
    try:
        import faiss  # type: ignore
    except Exception as e:
        print(f">> WARN: FAISS not available ({e}). Skipping index write.", file=sys.stderr)
        return

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss.write_index(index, str(faiss_path))
    print(f">> Saved FAISS index: {faiss_path}  (ntotal={index.ntotal})", flush=True)

    # 3) Sidecar ids
    ids_path = faiss_path.with_suffix(".ids.txt")
    with ids_path.open("w", encoding="utf-8") as f:
        for _id in ids:
            f.write(_id + "\n")
    print(f">> Saved ID map: {ids_path}", flush=True)



# ==============================================================
# === Main =====================================================
# ==============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Dir with deduped chunks (txt lines)")
    ap.add_argument("--db", required=True, help="SQLite DB path")
    ap.add_argument("--faiss", required=True, help="FAISS index path")
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("EMBED_BATCH", "64")))
    ap.add_argument("--local-only", action="store_true", help="Require model to exist locally (no download)")
    ap.add_argument("--verbose", action="store_true", help="Show detailed progress/config")
    args = ap.parse_args()

    input_dir = Path(args.input)
    db_path = Path(args.db)
    faiss_path = Path(args.faiss)

    if args.verbose:
        print("== Embed config ==")
        print(f"input     : {input_dir}")
        print(f"db        : {db_path}")
        print(f"faiss     : {faiss_path}")
        print(f"batch-size: {args.batch_size}")
        print(f"local-only: {args.local_only}")
        print(f"EMBED_MODEL_NAME: {_MODEL_NAME}")
        print(f"HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE', '(default)')}")
        sys.stdout.flush()

    texts, ids = _read_all_lines(input_dir)
    if not texts:
        print(">> Nothing to embed. Exiting.", flush=True)
        return

    print(f">> Embedding {len(texts)} chunks...", flush=True)
    t0 = time.time()

    embs = embed_texts(texts, batch_size=args.batch_size, normalize=True, local_only=args.local_only)

    print(f">> Done. {len(texts)} chunks in {time.time() - t0:.2f}s. Shape={embs.shape}", flush=True)

    _save_to_faiss_and_db(embs, ids, db_path, faiss_path)
    print(">> Embed step complete.", flush=True)


if __name__ == "__main__":
    main()
