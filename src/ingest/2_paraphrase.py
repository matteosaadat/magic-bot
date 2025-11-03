"""
Paraphrasing service using a locally hosted LLM (llama.cpp).

Exposes:
- PARAPHRASE_STYLE_PROMPT
- paraphrase_one(text: str, style_prompt: str | None = None) -> str
- paraphrase_batch(chunks: list[str], style_prompt: str, max_new_tokens: int = 256) -> list[str]

Behavior:
- Prefer llama.cpp HTTP server at LLAMA_SERVER_URL (default: http://localhost:8080) /completion
- Fallback to CLI via LLAMA_CPP_CLI (path to llama.cpp binary) with expected args documented below
- Safety checks per item: numbers preserved, no new entities, ROUGE-L/token-overlap >= 0.75; else return original
- Cache results to data/cache/paraphrase.sqlite keyed by content hash + style params
- Log basic stats

Safe patterns only; no eval/exec.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


# ----------------------------------------------------------------------------
# Configuration & Defaults
# ----------------------------------------------------------------------------

LLAMA_SERVER_URL_DEFAULT = "http://localhost:8080"
LLAMA_SERVER_COMPLETION_PATH = "/completion"
LLAMA_ENV_SERVER_URL = "LLAMA_SERVER_URL"
LLAMA_ENV_CLI_PATH = "LLAMA_CPP_CLI"  # path to llama.cpp CLI binary (e.g., ./main or llama-cli)
LLAMA_ENV_MODEL_PATH = "LLAMA_CPP_MODEL"  # path to GGUF model file for CLI fallback
LLAMA_ENV_NUM_THREADS = "LLAMA_NUM_THREADS"  # threads for CLI fallback

CACHE_DIR = Path("data/cache")
CACHE_DB_PATH = CACHE_DIR / "paraphrase.sqlite"


# ----------------------------------------------------------------------------
# Style Prompt (exported)
# ----------------------------------------------------------------------------

PARAPHRASE_STYLE_PROMPT: str = (
    "You are a careful rewriting assistant. Rewrite the user text in Matteo's tone: "
    "confident, concise, concrete. Do not hallucinate. Preserve all entities, dates, "
    "numbers, and specific terms. Keep the same meaning and key details. Avoid adding "
    "new claims or removing important facts. Use clear, direct sentences with strong verbs."
)


# ----------------------------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------------------------

logger = logging.getLogger("paraphrase")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ----------------------------------------------------------------------------
# Caching Utilities (SQLite)
# ----------------------------------------------------------------------------

def _ensure_cache_schema() -> None:
    """Create the cache directory and SQLite table if they don't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paraphrase_cache (
                cache_key TEXT PRIMARY KEY,
                input_hash TEXT NOT NULL,
                style_hash TEXT NOT NULL,
                params_hash TEXT NOT NULL,
                output_text TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _sha256(text: str) -> str:
    """Compute SHA-256 hex digest for text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compute_cache_key(text: str, style_prompt: str, max_new_tokens: int) -> str:
    """Derive a stable cache key from inputs and a version salt for invalidation."""
    version_salt = "v1:matteo-paraphrase"
    payload = json.dumps(
        {
            "v": version_salt,
            "t": _sha256(text),
            "s": _sha256(style_prompt),
            "m": max_new_tokens,
        },
        sort_keys=True,
    )
    return _sha256(payload)


def _cache_get(cache_key: str) -> Optional[str]:
    _ensure_cache_schema()
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        cur = conn.execute(
            "SELECT output_text FROM paraphrase_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def _cache_put(cache_key: str, text: str, style_prompt: str, max_new_tokens: int, output: str) -> None:
    _ensure_cache_schema()
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO paraphrase_cache (cache_key, input_hash, style_hash, params_hash, output_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                _sha256(text),
                _sha256(style_prompt),
                _sha256(str(max_new_tokens)),
                output,
            ),
        )
        conn.commit()


# ----------------------------------------------------------------------------
# Safety Checks: numbers, entities, and ROUGE-L (token overlap)
# ----------------------------------------------------------------------------

_number_pattern = re.compile(r"(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?%?")


def _extract_numbers(text: str) -> List[str]:
    """Extract numeric tokens (allow commas, decimals, optional %)."""
    nums = _number_pattern.findall(text)
    normalized = []
    for n in nums:
        n2 = n.replace(",", "").rstrip("%")
        normalized.append(n2)
    return normalized


def _numbers_preserved(src: str, dst: str) -> bool:
    """Ensure numeric multiset equality between source and destination."""
    from collections import Counter

    return Counter(_extract_numbers(src)) == Counter(_extract_numbers(dst))


def _extract_named_entities_simple(text: str) -> List[str]:
    """
    Approximate named entity extraction without heavy NLP dependencies:
    - Capture sequences of capitalized words (e.g., "New York", "OpenAI", "John")
    - Ignore capitalized words at the start of a sentence when alone if common
    This is heuristic and intentionally conservative.
    """
    # Split into tokens and identify capitalized words
    # Treat sequences of Capitalized tokens as entities
    tokens = re.findall(r"\b[\w'\-]+\b", text)
    entities: List[str] = []
    current: List[str] = []
    for tok in tokens:
        if tok[:1].isupper() and tok[1:].islower() or (len(tok) > 1 and tok.isupper()):
            current.append(tok)
        else:
            if current:
                entities.append(" ".join(current))
                current = []
    if current:
        entities.append(" ".join(current))
    # Deduplicate while preserving case
    seen = set()
    result = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            result.append(e)
    return result


def _no_new_entities(src: str, dst: str) -> bool:
    src_set = set(_extract_named_entities_simple(src))
    dst_set = set(_extract_named_entities_simple(dst))
    # Allow subset or equal; forbid new entities not in source
    return dst_set.issubset(src_set)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute LCS length for sequences using dynamic programming."""
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            temp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def _rouge_l_score(src: str, dst: str) -> float:
    """
    Compute ROUGE-L recall-oriented F-like score simplified to overlap ratio:
    score = LCS(src_tokens, dst_tokens) / max(len(src_tokens), len(dst_tokens))
    """
    a = _tokenize(src)
    b = _tokenize(dst)
    if not a or not b:
        return 0.0
    lcs = _lcs_length(a, b)
    denom = max(len(a), len(b))
    return lcs / denom if denom else 0.0


def _passes_safety(src: str, dst: str, threshold: float = 0.75) -> bool:
    if not _numbers_preserved(src, dst):
        return False
    if not _no_new_entities(src, dst):
        return False
    if _rouge_l_score(src, dst) < threshold:
        return False
    return True


# ----------------------------------------------------------------------------
# llama.cpp Adapters
# ----------------------------------------------------------------------------

def _compose_prompt(style_prompt: str, user_text: str) -> str:
    """Compose a strict prompt instructing faithful rewriting in the given style."""
    system = (
        f"{style_prompt}\n"
        "Rules: strictly preserve facts, entities, dates, and numbers. "
        "Do not add or invent new entities. Keep meaning identical. "
        "Rewrite clearly and concisely.\n"
    )
    return (
        f"[SYSTEM]\n{system}\n"
        f"[USER]\nRewrite the following text faithfully, returning only the rewritten text.\n\n"
        f"{user_text}\n"
        f"[ASSISTANT]\n"
    )


def _http_completion(server_url: str, prompt: str, max_new_tokens: int) -> Optional[str]:
    """
    Call llama.cpp server /completion endpoint.
    Expected minimal payload compatible with common llama.cpp server:
    {
      "prompt": str,
      "n_predict": int,
      "temperature": float,
      "stop": ["</s>"]
    }
    Returns the 'content'/'content' equivalent text depending on server variant.
    """
    url = server_url.rstrip("/") + LLAMA_SERVER_COMPLETION_PATH
    payload = {
        "prompt": prompt,
        "n_predict": int(max_new_tokens),
        "temperature": 0.2,
        "stop": ["</s>"],
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(raw)
        # Common response shapes: {"content": "..."} or {"choices":[{"text":"..."}]}
        if isinstance(obj, dict):
            if "content" in obj and isinstance(obj["content"], str):
                return obj["content"].strip()
            if "choices" in obj and obj["choices"]:
                choice = obj["choices"][0]
                text = choice.get("text") or choice.get("content")
                if isinstance(text, str):
                    return text.strip()
        return None
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def _cli_completion(cli_path: str, model_path: str, prompt: str, max_new_tokens: int) -> Optional[str]:
    """
    Call llama.cpp CLI as a fallback.

    Expected CLI (documented):
    - Binary: LLAMA_CPP_CLI (e.g., ./main or llama-cli)
    - Required args:
        --model <GGUF path>
        --prompt <prompt>
        -n <max_new_tokens>
    - Optional env:
        LLAMA_NUM_THREADS (int)

    We run with low temperature via default binary settings; callers can adjust binary defaults.
    """
    threads = os.getenv(LLAMA_ENV_NUM_THREADS)
    cmd = [cli_path, "--model", model_path, "--prompt", prompt, "-n", str(int(max_new_tokens))]
    if threads and threads.isdigit():
        cmd.extend(["-t", threads])
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=120,
        )
        out = (completed.stdout or "").strip()
        return out if out else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def paraphrase_batch(
    chunks: List[str],
    style_prompt: str,
    max_new_tokens: int = 256,
) -> List[str]:
    """
    Paraphrase a batch of text chunks using llama.cpp server or CLI fallback.

    Caches results. Applies safety checks per item; if a rewrite fails checks,
    returns the original text for that item.
    """
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list of strings")
    if any(not isinstance(x, str) for x in chunks):
        raise TypeError("each chunk must be a string")
    if not isinstance(style_prompt, str) or not style_prompt.strip():
        raise ValueError("style_prompt must be a non-empty string")

    server_url = os.getenv(LLAMA_ENV_SERVER_URL, LLAMA_SERVER_URL_DEFAULT)
    cli_path = os.getenv(LLAMA_ENV_CLI_PATH)
    model_path = os.getenv(LLAMA_ENV_MODEL_PATH)

    total = len(chunks)
    cached = 0
    rewritten = 0
    rejected = 0

    outputs: List[str] = []
    for text in chunks:
        cache_key = _compute_cache_key(text, style_prompt, max_new_tokens)
        cached_out = _cache_get(cache_key)
        if cached_out is not None:
            outputs.append(cached_out)
            cached += 1
            continue

        prompt = _compose_prompt(style_prompt, text)
        result: Optional[str] = None

        # Try HTTP server first
        result = _http_completion(server_url, prompt, max_new_tokens)

        # Fallback to CLI if needed
        if result is None and cli_path and model_path:
            result = _cli_completion(cli_path, model_path, prompt, max_new_tokens)

        # If still None, keep original
        if result is None:
            outputs.append(text)
            rejected += 1
            continue

        # Safety checks
        if _passes_safety(text, result, threshold=0.75):
            outputs.append(result)
            _cache_put(cache_key, text, style_prompt, max_new_tokens, result)
            rewritten += 1
        else:
            outputs.append(text)
            _cache_put(cache_key, text, style_prompt, max_new_tokens, text)
            rejected += 1

    logger.info(
        "paraphrase stats - total=%d cached=%d rewritten=%d rejected=%d",
        total,
        cached,
        rewritten,
        rejected,
    )
    return outputs


def paraphrase_one(text: str, style_prompt: str | None = None) -> str:
    """Paraphrase a single text chunk with optional style prompt override."""
    sp = style_prompt if (style_prompt and style_prompt.strip()) else PARAPHRASE_STYLE_PROMPT
    return paraphrase_batch([text], sp, max_new_tokens=256)[0]


# ----------------------------------------------------------------------------
# Minimal CLI
# ----------------------------------------------------------------------------

def _read_stdin_text() -> str:
    return sys.stdin.read()


def _main_cli(argv: List[str]) -> int:
    """
    Usage:
      python 2_paraphrase.py < input.txt

    Env configuration:
      - LLAMA_SERVER_URL: HTTP server base URL (default http://localhost:8080)
      - LLAMA_CPP_CLI: path to llama.cpp CLI binary for fallback
      - LLAMA_CPP_MODEL: path to GGUF model for CLI fallback
      - LLAMA_NUM_THREADS: threads for CLI fallback
      - PARAPHRASE_STYLE_PROMPT: override default style prompt
    """
    style = os.getenv("PARAPHRASE_STYLE_PROMPT", PARAPHRASE_STYLE_PROMPT)
    raw = _read_stdin_text()
    if not raw.strip():
        return 0
    out = paraphrase_one(raw, style_prompt=style)
    sys.stdout.write(out)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main_cli(sys.argv[1:]))


