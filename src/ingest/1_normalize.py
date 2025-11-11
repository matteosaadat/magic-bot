#!/usr/bin/env python3
# =============================================================
# 1_normalize.py
# -------------------------------------------------------------
# AI INSTRUCTION:
# Create a Python module that defines a safe text normalization
# utility for Markdown or plain text. It should:
# - Strip control characters (except newlines/tabs)
# - Collapse redundant whitespace
# - Normalize Unicode (NFKC) and unescape HTML entities
# - Protect and restore fenced code blocks (``` or ~~~)
# - Collapse 3+ newlines into max 2
# Chunk output to --chunk-size with --chunk-overlap characters
# Write .txt files under --out mirroring input subfolders
# Accept BOTH (--input/--output) and (--in/--out).
# Print a summary at the end.
# =============================================================

from __future__ import annotations
import argparse
import html
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

# -------- regexes
_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]")  # keep \n and \t
# fenced code: ```...``` or ~~~...~~~
_CODE_FENCE = re.compile(r"(^```.*?^```)|(^~~~.*?^~~~)", re.DOTALL | re.MULTILINE)

def _protect_code_blocks(text: str) -> Tuple[str, List[str]]:
    """Replace fenced code blocks with tokens we can restore later."""
    stash: List[str] = []
    def _repl(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"[[[CODE_{len(stash)-1}]]]"
    return _CODE_FENCE.sub(_repl, text), stash

def _restore_code_blocks(text: str, stash: List[str]) -> str:
    def _repl(m: re.Match) -> str:
        idx = int(m.group(1))
        return stash[idx] if 0 <= idx < len(stash) else m.group(0)
    return re.sub(r"\[\[\[CODE_(\d+)\]\]\]", _repl, text)

def normalize_text(s: str) -> str:
    # Unicode normalize
    s = unicodedata.normalize("NFKC", s)
    # Unescape HTML entities (&amp;, &lt;, ...)
    s = html.unescape(s)
    # Protect code fences
    s, stash = _protect_code_blocks(s)
    # Strip control chars (but keep newlines/tabs)
    s = _CTRL.sub("", s)
    # Collapse intra-line whitespace
    s = _WS.sub(" ", s)
    # Trim trailing spaces per line
    s = re.sub(r"[ \t]+\n", "\n", s)
    # Collapse 3+ newlines
    s = _MULTI_NL.sub("\n\n", s)
    # Restore code fences
    s = _restore_code_blocks(s, stash)
    return s.strip()

def chunk_text(s: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        return [s] if s else []
    n = len(s)
    if n == 0:
        return []
    chunks: List[str] = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(s[i:j])
        i += step
    return chunks

def discover_inputs(input_path: Path) -> List[Path]:
    """Return a list of .txt files. If a file was passed, return it if .txt."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".txt" else []
    if input_path.is_dir():
        return [p for p in input_path.rglob("*.txt") if p.is_file()]
    return []

def write_mirrored(out_root: Path, in_root: Path, src_file: Path, chunk_idx: int, content: str) -> Path:
    rel = src_file.relative_to(in_root)  # preserve subfolders
    # For chunks, append .partNN before .txt
    stem = rel.stem
    suffix = rel.suffix  # ".txt"
    rel_parent = rel.parent
    if chunk_idx is None:
        out_file = out_root / rel_parent / f"{stem}{suffix}"
    else:
        out_file = out_root / rel_parent / f"{stem}.part{chunk_idx:03d}{suffix}"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(content, encoding="utf-8")
    return out_file

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize and chunk collected text files.")
    # Accept both styles:
    p.add_argument("--input", "--in", dest="inp", required=True, help="Input directory of .txt (from collect) or a single .txt")
    p.add_argument("--output", "--out", dest="out", required=True, help="Output directory for normalized chunks")
    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--chunk-overlap", type=int, default=120)
    return p.parse_args(argv)

def main(argv: List[str]) -> int:
    args = parse_args(argv)
    in_path = Path(args.inp).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"== normalize ==")
    print(f"in : {in_path}")
    print(f"out: {out_root}")
    print(f"chunk_size={args['chunk_size'] if isinstance(args, dict) else args.chunk_size} overlap={args['chunk_overlap'] if isinstance(args, dict) else args.chunk_overlap}")

    files = discover_inputs(in_path)
    if not files:
        print("!! No .txt files found to normalize. (Did collect run?)")
        return 0

    written = 0
    for f in files:
        try:
            raw = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"!! Failed to read {f}: {e}")
            continue
        norm = normalize_text(raw)
        chunks = chunk_text(norm, args.chunk_size, args.chunk_overlap)

        # If text is small, still write at least one chunk
        if not chunks:
            chunks = [norm]

        if len(chunks) == 1:
            of = write_mirrored(out_root, in_path if in_path.is_dir() else f.parent, f, None, chunks[0])
            print(f">> {f}  →  {of}  ({len(chunks[0])} chars)")
            written += 1
        else:
            for ci, ch in enumerate(chunks):
                of = write_mirrored(out_root, in_path if in_path.is_dir() else f.parent, f, ci, ch)
                if ci == 0:
                    print(f">> {f}  →  {of.parent}/{f.stem}.part###.txt  (chunked x{len(chunks)})")
            written += len(chunks)

    print(f"== normalize stats == {{'inputs': {len(files)}, 'written': {written}}}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
