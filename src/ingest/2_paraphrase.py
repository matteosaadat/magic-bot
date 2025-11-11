#!/usr/bin/env python3
# =============================================================
# 2_paraphrase.py
# -------------------------------------------------------------
# Modes:
#   1) Directory mode:
#        --input/--in  <dir-of-.txt-chunks>
#        --output/--out <dir>
#      Walks input dir, paraphrases each .txt, writes mirrored files to output.
#
#   2) Stdin/Stdout mode (legacy):
#        cat file.txt | python 2_paraphrase.py > out.txt
#
# Paraphraser:
#   - Tries llama.cpp HTTP server (LLAMA_SERVER_URL, default http://localhost:8080)
#   - Falls back to llama.cpp CLI if LLAMA_CPP_CLI + LLAMA_CPP_MODEL are set
#   - If neither is available, pass-through (write original text) so it never hangs
#
# Caching:
#   - SQLite cache at <repo>/src/data/cache/paraphrase.sqlite
# =============================================================

from __future__ import annotations
import argparse
import hashlib
import html
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

# ---------- Paths
SRC_ROOT = Path(__file__).resolve().parents[1]            # .../src
DATA_ROOT = SRC_ROOT / "data"
CACHE_DIR = DATA_ROOT / "cache"
CACHE_DB_PATH = CACHE_DIR / "paraphrase.sqlite"

# ---------- Llama config
LLAMA_SERVER_URL_DEFAULT = "http://localhost:8080"
LLAMA_SERVER_COMPLETION_PATH = "/completion"
LLAMA_ENV_SERVER_URL = "LLAMA_SERVER_URL"
LLAMA_ENV_CLI_PATH = "LLAMA_CPP_CLI"
LLAMA_ENV_MODEL_PATH = "LLAMA_CPP_MODEL"
LLAMA_ENV_NUM_THREADS = "LLAMA_NUM_THREADS"

PARAPHRASE_STYLE_PROMPT: str = (
    "You are a careful rewriting assistant. Rewrite the user text in Matteo's tone: "
    "confident, concise, concrete. Do not hallucinate. Preserve all entities, dates, "
    "numbers, and specific terms. Keep the same meaning and key details. Avoid adding "
    "new claims or removing important facts. Use clear, direct sentences with strong verbs."
)

# ---------- Logging
logger = logging.getLogger("paraphrase")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ---------- Cache
def _ensure_cache_schema() -> None:
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

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _cache_key(text: str, style: str, max_new_tokens: int) -> str:
    return _sha256(json.dumps(
        {"v":"v1", "t":_sha256(text), "s":_sha256(style), "m":max_new_tokens}, sort_keys=True
    ))

def _cache_get(key: str) -> Optional[str]:
    _ensure_cache_schema()
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        row = conn.execute("SELECT output_text FROM paraphrase_cache WHERE cache_key = ?", (key,)).fetchone()
        return row[0] if row else None

def _cache_put(key: str, text: str) -> None:
    _ensure_cache_schema()
    # store hashed inputs as metadata (not strictly needed for now)
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO paraphrase_cache (cache_key, input_hash, style_hash, params_hash, output_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, "", "", "", text),
        )
        conn.commit()

# ---------- Safety checks
_number_pattern = re.compile(r"(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?%?")

def _extract_numbers(t: str) -> List[str]:
    nums = _number_pattern.findall(t)
    return [n.replace(",", "").rstrip("%") for n in nums]

def _numbers_preserved(a: str, b: str) -> bool:
    from collections import Counter
    return Counter(_extract_numbers(a)) == Counter(_extract_numbers(b))

def _tokenize(t: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", t.lower(), flags=re.UNICODE)

def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    dp = [0]*(m+1)
    for i in range(1, n+1):
        prev = 0
        ai = a[i-1]
        for j in range(1, m+1):
            tmp = dp[j]
            dp[j] = prev+1 if ai == b[j-1] else max(dp[j], dp[j-1])
            prev = tmp
    return dp[m]

def _rouge_l(a: str, b: str) -> float:
    A, B = _tokenize(a), _tokenize(b)
    if not A or not B: return 0.0
    return _lcs_len(A,B)/max(len(A),len(B))

def _passes_safety(src: str, dst: str, thr: float = 0.75) -> bool:
    return _numbers_preserved(src, dst) and _rouge_l(src, dst) >= thr

# ---------- llama adapters
def _compose_prompt(style: str, text: str) -> str:
    system = (
        f"{style}\n"
        "Rules: strictly preserve facts, entities, dates, and numbers. "
        "Do not add new entities. Keep meaning identical. Rewrite clearly and concisely.\n"
    )
    return f"[SYSTEM]\n{system}\n[USER]\nRewrite the following text faithfully, returning only the rewritten text.\n\n{text}\n[ASSISTANT]\n"

def _http_completion(url_base: str, prompt: str, max_new_tokens: int, timeout: int = 20) -> Optional[str]:
    url = url_base.rstrip("/") + LLAMA_SERVER_COMPLETION_PATH
    payload = {"prompt": prompt, "n_predict": int(max_new_tokens), "temperature": 0.2, "stop": ["</s>"]}
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(raw)
        if isinstance(obj, dict):
            if isinstance(obj.get("content"), str):
                return obj["content"].strip()
            ch = obj.get("choices") or []
            if ch and isinstance(ch[0], dict):
                txt = ch[0].get("text") or ch[0].get("content")
                if isinstance(txt, str):
                    return txt.strip()
        return None
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

def _cli_completion(cli: str, model: str, prompt: str, max_new_tokens: int) -> Optional[str]:
    threads = os.getenv(LLAMA_ENV_NUM_THREADS)
    cmd = [cli, "--model", model, "--prompt", prompt, "-n", str(int(max_new_tokens))]
    if threads and threads.isdigit():
        cmd += ["-t", threads]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
        out = (r.stdout or "").strip()
        return out or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

def _paraphrase_one(text: str, style: str, max_new_tokens: int, server_url: Optional[str], cli: Optional[str], model: Optional[str]) -> str:
    # cache
    key = _sha256(json.dumps({"v":"v1","t":_sha256(text),"s":_sha256(style),"m":max_new_tokens}))
    cached = _cache_get(key)
    if cached is not None:
        return cached

    prompt = _compose_prompt(style, text)
    out = None
    if server_url:
        out = _http_completion(server_url, prompt, max_new_tokens, timeout=20)
    if out is None and cli and model:
        out = _cli_completion(cli, model, prompt, max_new_tokens)

    # fallback: pass-through
    if out is None or not _passes_safety(text, out):
        out = text

    _cache_put(key, out)
    return out

# ---------- simple normalizer for tiny cleanups before paraphrase (optional)
_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]")
def _light_clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = html.unescape(s)
    s = _CTRL.sub("", s)
    return s.strip()

# ---------- directory mode helpers
def _discover_txts(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".txt" else []
    return [p for p in root.rglob("*.txt") if p.is_file()]

def _write_mirrored(out_root: Path, in_root: Path, src_file: Path, content: str) -> Path:
    rel = src_file.relative_to(in_root)
    out_file = out_root / rel
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(content, encoding="utf-8")
    return out_file

# ---------- CLI
def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paraphrase chunks (directory mode or stdin mode).")
    p.add_argument("--input", "--in", dest="inp", help="Input .txt file or directory (directory mode)")
    p.add_argument("--output", "--out", dest="out", help="Output directory for paraphrased files (directory mode)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--style", default=os.getenv("PARAPHRASE_STYLE_PROMPT", PARAPHRASE_STYLE_PROMPT))
    return p.parse_args(argv)

def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    server_url = os.getenv(LLAMA_ENV_SERVER_URL, LLAMA_SERVER_URL_DEFAULT)
    cli = os.getenv(LLAMA_ENV_CLI_PATH)
    model = os.getenv(LLAMA_ENV_MODEL_PATH)

    # Directory mode if both in/out provided
    if args.inp and args.out:
        in_path = Path(args.inp).resolve()
        out_root = Path(args.out).resolve()
        files = _discover_txts(in_path)
        if not files:
            print("!! paraphrase: no .txt files found under", in_path)
            out_root.mkdir(parents=True, exist_ok=True)
            return 0
        written = 0
        for f in files:
            raw = f.read_text(encoding="utf-8", errors="replace")
            raw = _light_clean(raw)
            out = _paraphrase_one(raw, args.style, args.max_new_tokens, server_url, cli, model)
            of = _write_mirrored(out_root, in_path if in_path.is_dir() else f.parent, f, out)
            if written < 6:
                print(f">> {f}  →  {of}  ({len(out)} chars)")
            written += 1
        print(f"== paraphrase stats == {{'inputs': {len(files)}, 'written': {written}}}")
        return 0

    # Stdin mode
    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    out = _paraphrase_one(_light_clean(raw), args.style, args.max_new_tokens, server_url, cli, model)
    sys.stdout.write(out)
    sys.stdout.flush()
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
