# AI INSTRUCTION:
# Define merge and ranking helpers for combining lexical and vector results.
# Keep it simple and stateless; return top_k ContextChunks.

from __future__ import annotations
from typing import List
from .types import ContextChunk

def merge_and_rank(
    lex_results: List[ContextChunk],
    vec_results: List[ContextChunk],
    alpha: float = 0.5,
    top_k: int = 6,
) -> List[ContextChunk]:
    # Normalize scores
    for lst in (lex_results, vec_results):
        if lst:
            scores = [r.score for r in lst]
            lo, hi = min(scores), max(scores)
            rng = hi - lo if hi != lo else 1.0
            for r in lst:
                r.score = (r.score - lo) / rng

    combined = {}
    for r in lex_results:
        combined[r.id] = r

    # Merge vector results; average with lexical if duplicate
    for v in vec_results:
        if v.id in combined:
            combined[v.id].score = (alpha * combined[v.id].score) + ((1 - alpha) * v.score)
        else:
            combined[v.id] = v

    # Sort and clip
    ranked = sorted(combined.values(), key=lambda x: x.score, reverse=True)
    return ranked[:top_k]
