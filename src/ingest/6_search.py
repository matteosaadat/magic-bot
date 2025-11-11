# AI INSTRUCTION:
# Minimal semantic search over FAISS index created by 4_embed.py.
# Loads faiss.index + .ids.txt, embeds a query with the same SBERT model,
# does top-k search, and prints (score, id, text snippet).

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# --- keep the model name consistent with 4_embed.py ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")  # "cpu" or "cuda"

def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME, device=_DEVICE)

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms

def _embed_query(text: str) -> np.ndarray:
    model = _get_model()
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32, copy=False)
    vec = _l2_normalize(vec)
    return vec  # shape (1, D)

def _load_ids(ids_path: Path) -> List[str]:
    ids: List[str] = []
    with ids_path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.append(line.rstrip("\n"))
    return ids

def _build_text_lookup(dedup_dir: Path) -> Dict[str, str]:
    """
    Build a mapping from "filename:lineno" -> line text by reading all *.txt in dedup_dir.
    """
    out: Dict[str, str] = {}
    for p in sorted(dedup_dir.rglob("*.txt")):
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, start=1):
                    key = f"{p.name}:{i}"
                    out[key] = line.strip()
        except Exception:
            pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss", required=True, help="Path to faiss.index file")
    ap.add_argument("--ids", help="Path to ids sidecar (.ids.txt). Defaults to faiss.index → .ids.txt")
    ap.add_argument("--input", required=True, help="Deduped chunks directory (same as used by 4_embed.py) for text lookup")
    ap.add_argument("--query", required=True, help="Search query")
    ap.add_argument("--k", type=int, default=10, help="Top-K results")
    args = ap.parse_args()

    faiss_path = Path(args.faiss)
    ids_path = Path(args.ids) if args.ids else faiss_path.with_suffix(".ids.txt")
    dedup_dir = Path(args.input)

    # Load FAISS
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(f"FAISS not available: {e}")

    index = faiss.read_index(str(faiss_path))
    ids = _load_ids(ids_path)
    if index.ntotal != len(ids):
        raise RuntimeError(f"Index size ({index.ntotal}) != ids count ({len(ids)})")

    # Build quick text lookup
    text_map = _build_text_lookup(dedup_dir)

    # Embed query (cosine == inner product thanks to L2-normalization)
    q = _embed_query(args.query)
    scores, idxs = index.search(q, args.k)  # shapes: (1,k), (1,k)
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    print(f'Query: {args.query}\nResults (top={args.k}):\n')
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        if i < 0:
            continue
        hit_id = ids[i]
        snippet = text_map.get(hit_id, "(text not found)")
        # scores are inner products in [-1,1] for normalized vectors
        print(f"{rank}. {hit_id}  score={s:.4f}")
        print(f"   {snippet}\n")

if __name__ == "__main__":
    main()
