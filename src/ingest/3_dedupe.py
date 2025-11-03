# AI INSTRUCTION:
# Implement dedup utilities combining SimHash (pre-filter) and embedding-based semantic dedup.
#
# Functions:
#   - `simhash_64(text: str) -> int`  # 64-bit simhash using 5-gram shingles
#   - `hamming_distance64(a: int, b: int) -> int`
#   - `is_near_dup_simhash(a: int, b: int, max_hamming: int = 3) -> bool`
#   - `filter_near_dups_simhash(chunks: list[str], max_hamming=3) -> list[int]`  # returns keep indices
#   - `semantic_dedup(embs: "np.ndarray", same_doc_threshold=0.92, cross_doc_threshold=0.96, doc_ids: list[str] | None = None) -> list[int]`
#
# Notes:
#   - SimHash is a fast same-doc pre-clean. Use 5-gram word shingles and a simple hashing scheme.
#   - `filter_near_dups_simhash` should keep the first occurrence and drop subsequent near-dups.
#   - `semantic_dedup` should cluster by cosine similarity; when two items exceed the threshold:
#       * if `doc_ids` is provided, use `same_doc_threshold` for pairs within same doc, otherwise `cross_doc_threshold`.
#       * keep the first occurrence; drop the rest.
#   - Avoid heavy deps; rely on stdlib and numpy only.

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Tuple

import numpy as np


# =========================
# Tokenization & Shingling
# =========================

_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def _tokenize_words(text: str) -> List[str]:
    """
    Lightweight, stable tokenizer for SimHash:
    - lowercases
    - keeps alphanumerics/underscores as tokens
    - ignores punctuation/spaces
    """
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _shingles(tokens: List[str], size: int = 5) -> Iterable[Tuple[str, ...]]:
    """Yield word shingles (tuples) of given size."""
    if size <= 0:
        return
    n = len(tokens)
    if n < size:
        if n == 0:
            return
        # If fewer than 'size' tokens, treat the whole thing as one shingle.
        yield tuple(tokens)
        return
    for i in range(n - size + 1):
        yield tuple(tokens[i : i + size])


def _hash64(s: bytes) -> int:
    """
    Stable 64-bit positive integer from bytes.
    Using MD5 (128-bit) then folding to 64-bit via xor of high/low halves.
    """
    d = hashlib.md5(s).digest()  # 16 bytes
    hi = int.from_bytes(d[:8], "big", signed=False)
    lo = int.from_bytes(d[8:], "big", signed=False)
    return (hi ^ lo) & ((1 << 64) - 1)


# ===============
# 64-bit SimHash
# ===============

def simhash_64(text: str) -> int:
    """
    Compute a 64-bit SimHash over 5-gram word shingles.
    - Build a 64-dim integer accumulator vector.
    - For each shingle hash, add +1 where bit==1 and -1 where bit==0.
    - Positive positions -> 1, else 0.
    Returns an unsigned 64-bit integer (as Python int).
    """
    tokens = _tokenize_words(text)
    acc = [0] * 64
    has_any = False

    for sh in _shingles(tokens, size=5):
        has_any = True
        h = _hash64((" ".join(sh)).encode("utf-8"))
        for bit in range(64):
            if (h >> bit) & 1:
                acc[bit] += 1
            else:
                acc[bit] -= 1

    # Handle degenerate case: no shingles/tokens -> hash of empty string
    if not has_any:
        h = _hash64(b"")
        return h

    # Convert accumulator signs to bits
    out = 0
    for bit in range(64):
        if acc[bit] > 0:
            out |= (1 << bit)
    return out


# =======================
# Hamming-related helpers
# =======================

def hamming_distance64(a: int, b: int) -> int:
    """Return the Hamming distance between two 64-bit integers."""
    return (a ^ b).bit_count()


def is_near_dup_simhash(a: int, b: int, max_hamming: int = 3) -> bool:
    """
    True if the Hamming distance between a and b is <= max_hamming.
    Typical near-dup thresholds for 64-bit SimHash: 2–5 depending on corpus.
    """
    return hamming_distance64(a, b) <= max_hamming


# =========================================
# Greedy near-dup filter based on SimHash
# =========================================

def filter_near_dups_simhash(chunks: List[str], max_hamming: int = 3) -> List[int]:
    """
    Return indices of chunks to KEEP (first occurrence wins).
    O(N^2) worst-case, which is fine for typical ingest batch sizes.
    If you need to scale, add simple LSH banding, but this is adequate for ~thousands.

    Strategy:
      - Iterate in original order; compute simhash for current.
      - Compare to kept simhashes; if any within threshold, mark as dup (drop).
      - Else keep.
    """
    keep_indices: List[int] = []
    kept_hashes: List[int] = []

    for i, text in enumerate(chunks):
        h = simhash_64(text)
        duplicate = False
        for kh in kept_hashes:
            if is_near_dup_simhash(h, kh, max_hamming=max_hamming):
                duplicate = True
                break
        if not duplicate:
            keep_indices.append(i)
            kept_hashes.append(h)

    return keep_indices


# ==========================================
# Embedding-based semantic dedup (cosine)
# ==========================================

def _normalize_embeddings(embs: np.ndarray) -> np.ndarray:
    """L2-normalize rows; safe for zero vectors."""
    embs = np.asarray(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embs / norms


def semantic_dedup(
    embs: np.ndarray,
    same_doc_threshold: float = 0.92,
    cross_doc_threshold: float = 0.96,
    doc_ids: List[str] | None = None,
) -> List[int]:
    """
    Greedy semantic dedup using cosine similarity.
    - Assumes each row of `embs` corresponds to the same index in `doc_ids` (if provided).
    - Keeps the first occurrence, drops later items that exceed the similarity threshold
      vs ANY kept item. Threshold depends on whether the pair is from the same document.

    Args:
      embs: (N, D) float array (need not be normalized; we'll L2-normalize).
      same_doc_threshold: cosine similarity above which two items from the same doc are considered duplicates.
      cross_doc_threshold: cosine similarity above which two items from different docs are considered duplicates.
      doc_ids: optional list of length N; if None, uses `cross_doc_threshold` for all pairs.

    Returns:
      keep_indices: list of indices to keep, in original order.
    """
    if embs.ndim != 2:
        raise ValueError("embs must be a 2D array of shape (N, D)")

    N = embs.shape[0]
    if N == 0:
        return []

    # Normalize vectors for cosine similarity as dot product
    nen = _normalize_embeddings(embs)

    keep: List[int] = []
    kept_matrix: List[np.ndarray] = []  # store normalized embeddings of kept
    kept_docs: List[str] = [] if doc_ids is not None else None  # type: ignore

    for i in range(N):
        v = nen[i]
        is_dup = False

        if keep:
            # Stack kept vectors into (K, D) for a single matmul
            K = len(kept_matrix)
            kept_mat = np.vstack(kept_matrix)  # (K, D)
            sims = kept_mat @ v  # (K,)
            # Determine thresholds vs each kept item
            if doc_ids is None:
                th = cross_doc_threshold
                if np.any(sims >= th):
                    is_dup = True
            else:
                # per-pair threshold depending on same vs cross doc
                di = doc_ids[i]
                # vectorized comparison per kept item
                # same doc mask
                same_mask = np.array([kd == di for kd in kept_docs], dtype=bool)  # type: ignore
                cross_mask = ~same_mask
                # If any same-doc sim >= same_doc_threshold, dup
                if np.any(sims[same_mask] >= same_doc_threshold):
                    is_dup = True
                # If any cross-doc sim >= cross_doc_threshold, dup
                if not is_dup and np.any(sims[cross_mask] >= cross_doc_threshold):
                    is_dup = True

        if not is_dup:
            keep.append(i)
            kept_matrix.append(v)
            if kept_docs is not None:
                kept_docs.append(doc_ids[i])  # type: ignore

    return keep


# ==========================
# __main__ Smoke Test
# ==========================

def _smoke_simhash():
    a = "This is a small example about CI/CD with GitHub Actions and Docker on AWS."
    b = "This is a tiny example about CI CD using GitHub Actions & Docker in AWS."
    c = "Completely different topic about gardening and soil biology."

    ha = simhash_64(a)
    hb = simhash_64(b)
    hc = simhash_64(c)

    print("[SimHash]")
    print("a hash:", f"{ha:016x}")
    print("b hash:", f"{hb:016x}")
    print("c hash:", f"{hc:016x}")
    print("H(a,b):", hamming_distance64(ha, hb))
    print("H(a,c):", hamming_distance64(ha, hc))
    print("keep (a,b,c) with max_hamming=3:", filter_near_dups_simhash([a, b, c], max_hamming=3))

def _smoke_semantic():
    # Two close vectors + one distant
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 8)).astype(np.float32)
    near1 = base + rng.normal(scale=0.01, size=(1, 8)).astype(np.float32)
    near2 = base + rng.normal(scale=0.015, size=(1, 8)).astype(np.float32)
    far = rng.normal(size=(1, 8)).astype(np.float32)

    embs = np.vstack([near1, near2, far])
    # Doc ids: first two same doc, third different
    docs = ["docA", "docA", "docB"]

    keep = semantic_dedup(embs, same_doc_threshold=0.92, cross_doc_threshold=0.96, doc_ids=docs)
    print("[Semantic]")
    print("keep indices:", keep)

if __name__ == "__main__":
    print("Running 3_dedup.py smoke tests...\n")
    _smoke_simhash()
    print()
    _smoke_semantic()
