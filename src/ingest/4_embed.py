# AI INSTRUCTION:
# Implement local embedding utilities using sentence-transformers (no external APIs).
#
# Requirements:
# - Default model: "BAAI/bge-small-en-v1.5" (384-dim). Lazy-load a single global model instance.
# - `embed_texts(texts: list[str], batch_size=64, normalize=True) -> "np.ndarray"`  # returns (N, D)
# - Persistence helpers:
#     - `save_embeddings(path: str, embs: "np.ndarray", ids: list[str]) -> None`
#     - `load_embeddings(path: str) -> tuple["np.ndarray", list[str]]`
# - Add a `__main__` that reads lines from stdin, embeds them, and prints shape + first vector slice.
#
# Implementation details:
# - Use sentence_transformers.SentenceTransformer
# - If `normalize=True`, L2-normalize vectors.
# - Avoid importing heavy libs at module import; load in the first call.

from __future__ import annotations

from typing import List, Tuple
import io
import sys
import os
import numpy as np

# ---- Lazy model globals (avoid heavy imports at module import) ----
_MODEL = None
_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")


def _get_model():
    """
    Lazy-load and cache a single global SentenceTransformer model instance.
    Avoid importing heavy libraries until first use.
    """
    global _MODEL
    if _MODEL is None:
        # Import here to keep module import light
        from sentence_transformers import SentenceTransformer  # type: ignore

        # It's a small model (384-d), loads quickly; no API calls required.
        _MODEL = SentenceTransformer(_MODEL_NAME)  # uses CPU/GPU automatically if available
    return _MODEL


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize rows of a 2D array.
    """
    if x.ndim != 2:
        raise ValueError("Expected a 2D array for normalization")
    norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def embed_texts(
    texts: List[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """
    Embed a list of strings into a (N, D) float32 numpy array using a local model.

    Args:
        texts: List of input strings.
        batch_size: Encode batch size.
        normalize: If True, L2-normalize each vector.

    Returns:
        np.ndarray of shape (N, D), dtype float32.
    """
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")
    if len(texts) == 0:
        return np.empty((0, 0), dtype=np.float32)

    model = _get_model()

    # sentence-transformers handles batching internally, but we keep an explicit loop
    # to maintain control and ensure consistent dtype/concatenation.
    embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        # convert_to_numpy returns float32 by default for sbert
        vecs = model.encode(
            chunk,
            batch_size=len(chunk),  # we already chunked; let it run as one mini-batch
            convert_to_numpy=True,
            normalize_embeddings=False,  # we'll handle normalization ourselves
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        embs.append(vecs)

    out = np.vstack(embs).astype(np.float32, copy=False)

    if normalize:
        out = _l2_normalize(out)

    return out


def save_embeddings(path: str, embs: np.ndarray, ids: List[str]) -> None:
    """
    Persist embeddings and their IDs to disk as a single .npz file.

    Args:
        path: Destination file path ('.npz' recommended). Parent dirs must exist.
        embs: (N, D) float32 array.
        ids:  List of N string identifiers, aligned with rows of embs.
    """
    if embs.ndim != 2:
        raise ValueError("embs must be 2D (N, D)")
    if len(ids) != embs.shape[0]:
        raise ValueError("len(ids) must match number of rows in embs")
    # Store IDs as a numpy array of dtype=object to preserve arbitrary strings
    ids_arr = np.array(ids, dtype=object)
    np.savez_compressed(path, embs=embs.astype(np.float32, copy=False), ids=ids_arr)


def load_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings and IDs previously saved with save_embeddings().

    Returns:
        (embs, ids)
        embs: (N, D) float32 array
        ids:  list[str] of length N
    """
    with np.load(path, allow_pickle=True) as data:
        embs = data["embs"].astype(np.float32, copy=False)
        ids_arr = data["ids"]
        # Ensure we return a plain list of str
        ids_list = [str(x) for x in ids_arr.tolist()]
    return embs, ids_list


def _read_stdin_lines() -> List[str]:
    """
    Read non-empty, stripped lines from stdin. Keeps empty lines if they are significant content?
    Here we drop purely empty lines to avoid accidental blanks.
    """
    # Use sys.stdin.buffer to avoid encoding surprises; then decode as utf-8.
    raw = sys.stdin.buffer.read()
    if not raw:
        return []
    text = raw.decode("utf-8", errors="replace")
    # Splitlines preserves logical lines without trailing newline
    lines = text.splitlines()
    # Keep lines as-is except strip trailing carriage returns/spaces
    cleaned = [ln.strip() for ln in lines if ln.strip() != ""]
    return cleaned


def _print_demo_output(embs: np.ndarray) -> None:
    """
    Print shape and the first vector slice (first 8 values) as a small sanity check.
    """
    if embs.size == 0:
        print("shape=(0, 0), first=[]")
        return
    first_slice = embs[0, :8]
    # round for readability
    rounded = np.round(first_slice.astype(float), 4).tolist()
    print(f"shape={tuple(embs.shape)}, first[:8]={rounded}")


if __name__ == "__main__":
    # Usage:
    #   echo -e "hello\nworld" | python 4_embed.py
    #   cat lines.txt | python 4_embed.py
    lines = _read_stdin_lines()
    if not lines:
        # If no stdin, demonstrate with a tiny default batch
        lines = ["Hello world", "Sentence embeddings are handy."]
    embs = embed_texts(lines, batch_size=64, normalize=True)
    _print_demo_output(embs)
