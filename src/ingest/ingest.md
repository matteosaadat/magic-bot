# Matteo-bot — Ingest Project Orchestrator
# =============================================================================
# Purpose
#   This file provides project-level instructions for Cursor to bootstrap and
#   maintain the self-hosted AI ingest pipeline. The ingest pipeline:
#     - Reads Markdown stories with YAML front-matter
#     - Normalizes text to a consistent tone
#     - Deduplicates near-identical chunks
#     - Generates embeddings locally
#     - Builds hybrid search indexes (SQLite FTS5 + FAISS)
#   Each _TASK creates or updates a file that contains its own "AI INSTRUCTION"
#   header. After a file is created, open it, select the header, and run AI to
#   complete/refresh the implementation.
#
# Conventions
#   - Files are ordered numerically (1_, 2_, 3_...) to show pipeline order.
#   - Keep implementations deterministic & idempotent.
#   - No OpenAI usage. Use local models (llama.cpp / vLLM) + sentence-transformers.
#
# Usage in Cursor
#   - Select a _TASK block and press ⌘K (Mac) / Ctrl+K (Win/Linux).
#   - Cursor will create/overwrite the target file with the AI INSTRUCTION.
#   - Then open that file, select its AI INSTRUCTION, run AI to implement it.
# =============================================================================


# -----------------------------------------------------------------------------
# PIPELINE OVERVIEW
# -----------------------------------------------------------------------------
# Files to generate (in this order):
#   1_normalize.py        -> safe Markdown/plain-text normalization utilities
#   2_paraphrase.py       -> batch paraphrasing via local LLM (llama.cpp), with safety checks
#   3_dedup.py            -> SimHash pre-filter + embedding-based semantic dedup
#   4_embed.py            -> local embedding (bge-small-en-v1.5) + persistence
#   5_index.py            -> SQLite FTS5 + FAISS index builders
#   6_search.py           -> hybrid retrieval: FTS5 ∪ FAISS (+ optional rerank)
#   7_ingest_cli.py       -> CLI: parse→chunk→normalize→paraphrase→dedup→embed→index
#   __init__.py           -> exports for easier imports

# ============================================================
# FOLDER AI INSTRUCTION:
# This project intentionally uses NUMBERED modules like:
#   1_normalize.py, 2_paraphrase.py, 3_dedup.py, 4_embed.py, 5_index.py
#
# IMPORTANT:
# - Python CANNOT import these with "from 1_normalize import X".
# - NEVER use direct/static imports for numbered files.
# - ALWAYS use dynamic imports via importlib and the "ingest." package path.
#
# ✅ DO THIS (dynamic import):
#     import importlib
#     _norm = importlib.import_module("ingest.1_normalize").normalize_text
#
# ❌ DO NOT DO THIS (invalid):
#     from 1_normalize import normalize_text
#
# Structure assumptions:
# - This file lives in the "ingest/" package.
# - Keep filenames and numbering as-is; do not rename them.
# - If you need an attr from a numbered file, import it dynamically.
# ============================================================

# =============================================================================
# TASKS — CREATE FILES WITH AI INSTRUCTIONS
# After creating a file, open it, select the AI INSTRUCTION block, and run AI.
# =============================================================================

_TASK: create file "1_normalize.py" with the following exact text:

# AI INSTRUCTION:
# Create a Python module in this file that defines a safe text normalization utility for Markdown or plain text.
# It should:
# - Strip control characters (except newlines/tabs)
# - Collapse redundant whitespace
# - Normalize Unicode (NFKC) and unescape HTML entities
# - Protect and restore fenced code blocks (``` or ~~~)
# - Collapse 3+ newlines into max 2
# Return a `normalize_text(raw: str) -> str` function.
# Include helper functions `_protect_code_blocks` and `_restore_code_blocks`.
# Use `re`, `unicodedata`, and `html.unescape`.
# Add clear comments explaining each section.
# Include a minimal `if __name__ == "__main__":` smoke test reading from stdin.


# -----------------------------------------------------------------------------

_TASK: create file "2_paraphrase.py" with the following exact text:

# AI INSTRUCTION:
# Implement a local paraphrasing service that rewrites text into a consistent "Matteo tone" (confident, concise, concrete),
# using a locally hosted LLM (no OpenAI). Prefer llama.cpp (GGUF) with Llama-3.1-8B-Instruct Q4_K_M.
#
# Requirements:
# - Provide `paraphrase_batch(chunks: list[str], style_prompt: str, max_new_tokens: int = 256) -> list[str]`.
# - Provide a simple adapter for llama.cpp:
#     - First try HTTP server mode at http://localhost:8080/completion (or configurable VIA ENV VARS).
#     - Fallback to CLI invocation if server isn't available (document CLI args you expect).
# - Use a fixed system/style prompt to enforce "no hallucinations, keep entities/dates/numbers".
# - Add safety checks per item:
#     - Reject if numeric values changed (except formatting).
#     - Reject if new named entities appear vs original.
#     - Enforce ROUGE-L or token-overlap >= 0.75; on fail, return original text.
# - Cache results to `data/cache/paraphrase.sqlite` keyed by content hash + style params.
# - Log basic stats: total, cached, rewritten, rejected.
#
# Expose:
#   - `PARAPHRASE_STYLE_PROMPT` (default includes tone rules)
#   - `paraphrase_one(text: str, style_prompt: str | None = None) -> str`
#   - `paraphrase_batch(...)`
#
# Add a minimal CLI: `python 2_paraphrase.py < input.txt` prints rewritten text.


# -----------------------------------------------------------------------------

_TASK: create file "3_dedup.py" with the following exact text:

# AI INSTRUCTION:
# Implement dedup utilities combining SimHash (pre-filter) and embedding-based semantic dedup.
#
# Functions:
#   - `simhash_64(text: str) -> int`  # 64-bit simhash using 5-gram shingles
#   - `is_near_dup_simhash(a: int, b: int, max_hamming: int = 3) -> bool`
#   - `filter_near_dups_simhash(chunks: list[str], max_hamming=3) -> list[int]`  # returns keep indices
#   - `semantic_dedup(embs: np.ndarray, same_doc_threshold=0.92, cross_doc_threshold=0.96, doc_ids: list[str] | None = None) -> list[int]`
#
# Notes:
#   - SimHash is fast same-doc pre-clean.
#   - Semantic dedup runs on normalized or paraphrased text embeddings.
#   - Implement small helpers to compute Hamming distance and to keep 1 representative per dup cluster.


# -----------------------------------------------------------------------------

_TASK: create file "4_embed.py" with the following exact text:

# AI INSTRUCTION:
# Implement local embedding utilities using sentence-transformers (no external APIs).
#
# Requirements:
# - Default model: "BAAI/bge-small-en-v1.5" (384-dim), lazy-loaded singleton.
# - `embed_texts(texts: list[str], batch_size=64, normalize=True) -> np.ndarray`  # returns (N, D)
# - Persist embeddings to disk when asked:
#     - `save_embeddings(path: str, embs: np.ndarray, ids: list[str])`
#     - `load_embeddings(path: str) -> tuple[np.ndarray, list[str]]`
# - Add a quick self-test in `__main__` to embed a couple lines from stdin.


# -----------------------------------------------------------------------------

_TASK: create file "5_index.py" with the following exact text:

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
# - `init_sqlite(db_path: str) -> sqlite3.Connection`
# - `upsert_document(...)`, `upsert_chunk(...)`, `rebuild_fts(conn)`
# - FAISS:
#     - `build_faiss(embs: np.ndarray) -> faiss.Index`
#     - `save_faiss(index_path: str, index: faiss.Index, ids: list[str])`
#     - `load_faiss(index_path: str) -> tuple[faiss.Index, list[str]]`
#
# Add a `__main__` that creates tables if missing and prints basic counts.


# -----------------------------------------------------------------------------

_TASK: create file "6_search.py" with the following exact text:

# AI INSTRUCTION:
# Implement hybrid retrieval combining SQLite FTS5 and FAISS.
#
# Functions:
#   - `fts_search(conn, query: str, k: int = 50) -> list[tuple[chunk_id, score]]`  # use bm25() order
#   - `faiss_search(index, query_emb: np.ndarray, ids: list[str], k: int = 50) -> list[tuple[chunk_id, score]]`
#   - `hybrid_search(conn, index, ids: list[str], query: str, query_emb: np.ndarray, k_lex=50, k_vec=50, top_final=8) -> list[str]`
#     * union + score normalization + diversity by doc
# - Optional: add a hook for cross-encoder reranking (stub only).
#
# Provide a `__main__` that runs a demo hybrid search from CLI args.


# -----------------------------------------------------------------------------

_TASK: create file "7_ingest_cli.py" with the following exact text:

# AI INSTRUCTION:
# Create an end-to-end CLI that:
#   1) Discovers Markdown files under ../knowledge/
#   2) Parses front-matter + body; chunks by headings with size guardrails
#   3) Normalizes text (import from 1_normalize.py)
#   4) Paraphrases in batch with local LLM (2_paraphrase.py), with caching & safety checks
#   5) Runs dedup: simhash pre-filter, then semantic dedup (3_dedup.py)
#   6) Embeds kept chunks (4_embed.py)
#   7) Upserts documents/chunks into SQLite and rebuilds FTS (5_index.py)
#   8) Builds/saves FAISS for chunk embeddings (5_index.py)
#
# Requirements:
#   - CLI args: --db data/db/portfolio.db --faiss data/index/faiss.index --root ../knowledge --paraphrase/--no-paraphrase
#   - Print ingest stats summary (docs, chunks, dropped_simhash, dropped_semantic, kept, elapsed)
#   - Write JSONL run log to data/logs/ingest-YYYYMMDD.jsonl
#
# Provide `main()` guarded by `if __name__ == "__main__":`


# -----------------------------------------------------------------------------

_TASK: create file "__init__.py" with the following exact text:

# AI INSTRUCTION:
# Add convenient re-exports for public functions used by the rest of the project.
# Do not import heavy models at import-time; keep imports light.
# Export: normalize_text, paraphrase_one, paraphrase_batch, simhash_64, semantic_dedup,
#         embed_texts, init_sqlite, build_faiss, load_faiss, save_faiss, hybrid_search.


# =============================================================================
# TASKS — EXECUTE (FOLLOW THE AI INSTRUCTIONS IN EACH FILE)
# After files are created, open each one, select the "AI INSTRUCTION" block, run AI.
# =============================================================================

_TASK: open "1_normalize.py", select the AI INSTRUCTION, and implement it.

_TASK: open "2_paraphrase.py", select the AI INSTRUCTION, and implement it.

_TASK: open "3_dedup.py", select the AI INSTRUCTION, and implement it.

_TASK: open "4_embed.py", select the AI INSTRUCTION, and implement it.

_TASK: open "5_index.py", select the AI INSTRUCTION, and implement it.

_TASK: open "6_search.py", select the AI INSTRUCTION, and implement it.

_TASK: open "7_ingest_cli.py", select the AI INSTRUCTION, and implement it.


# =============================================================================
# OPTIONAL TASKS — QUICK SMOKE
# =============================================================================

_TASK: Create a file "SMOKE.md" describing how to run a local smoke test:
# AI INSTRUCTION:
# Document a smoke test:
#   - Start llama.cpp server (or explain CLI fallback).
#   - Run: `python ingest/7_ingest_cli.py --root knowledge --db data/db/portfolio.db --faiss data/index/faiss.index --paraphrase`
#   - Then try a hybrid search: `python -m ingest.6_search "ci/cd pipeline with github actions"`
#   - Include expected outputs and troubleshooting tips (missing FTS5, missing FAISS, missing models).
