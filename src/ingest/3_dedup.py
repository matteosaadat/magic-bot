#!/usr/bin/env python3
# =============================================================
# 3_dedup.py
# -------------------------------------------------------------
# Deduplicate text chunk files (near-duplicate removal).
#
# Features:
# - Recursively scans an input directory for .txt files
# - Builds shingled Jaccard signatures (word 5-grams by default)
# - Treats certain aliases as identical *for fingerprinting only*,
#   e.g., "Persian" == "Farsi", so repeated mentions collapse.
# - Keeps the first occurrence, skips subsequent near-duplicates.
# - Accepts BOTH (--input/--output) and (--in/--out).
#
# CLI:
#   --input/--in  <dir or file>
#   --output/--out <dir>
#   --shingle-size 5
#   --threshold 0.90   # Jaccard similarity at or above => consider duplicate
#
# Output:
#   Writes only non-duplicate files to --out, mirroring subfolders.
#   Logs "SKIP duplicate of: <path>" for skipped ones.
# =============================================================

from __future__ import annotations
import argparse
import html
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# ---------- alias map used for *fingerprinting only* (does not rewrite files)
ALIASES: Dict[str, str] = {
    # language canonicalization
    r"\bpersian\b": "farsi",
    r"\bthe italian\b": "italian",
    r"\bitaliano\b": "italian",
    r"\bfārsi\b": "farsi",
    # soft “native language” hint; helps collapse multiple ways of saying same thing
    r"\bnative language\b": "farsi",  # you can remove this if too aggressive
}

_WORD = re.compile(r"\b[\w'-]+\b", re.UNICODE)

def _light_normalize_for_fp(s: str) -> str:
    """Light normalization for fingerprinting only (not written back)."""
    s = unicodedata.normalize("NFKC", s)
    s = html.unescape(s)
    s = s.lower()
    # cheap whitespace / control cleanup
    s = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    # apply alias replacements
    for pat, repl in ALIASES.items():
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return s.strip()

def _words(text: str) -> List[str]:
    return _WORD.findall(text)

def _shingles(tokens: List[str], k: int) -> Set[Tuple[str, ...]]:
    if k <= 1:
        return { (t,) for t in tokens }
    out: Set[Tuple[str, ...]] = set()
    for i in range(0, max(0, len(tokens) - k + 1)):
        out.add(tuple(tokens[i : i + k]))
    return out

def _jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

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

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Near-duplicate removal for text chunks.")
    p.add_argument("--input", "--in", dest="inp", required=True, help="Input .txt file or directory")
    p.add_argument("--output", "--out", dest="out", required=True, help="Output directory for non-duplicates")
    p.add_argument("--shingle-size", type=int, default=5, help="Word shingle size (n-gram length)")
    p.add_argument("--threshold", type=float, default=0.90, help="Jaccard similarity threshold to mark as duplicate")
    return p.parse_args(argv)

def main(argv: List[str]) -> int:
    args = parse_args(argv)
    in_path = Path(args.inp).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    files = _discover_txts(in_path)
    if not files:
        print("!! dedup: no .txt files found under", in_path)
        return 0

    print(f"== dedup ==")
    print(f"in : {in_path}")
    print(f"out: {out_root}")
    print(f"shingle={args.shingle_size} threshold={args.threshold}")

    kept = 0
    skipped = 0

    # We store (source_path, signature) for the first kept copy of each “concept”.
    signatures: List[Tuple[Path, Set[Tuple[str, ...]]]] = []

    for f in files:
        raw = f.read_text(encoding="utf-8", errors="replace")
        fp_text = _light_normalize_for_fp(raw)
        toks = _words(fp_text)
        sig = _shingles(toks, args.shingle_size)

        # Compare with previously kept signatures
        is_dup = False
        for kept_path, kept_sig in signatures:
            sim = _jaccard(sig, kept_sig)
            if sim >= args.threshold:
                print(f"SKIP duplicate of: {kept_path}  <- {f}  (sim={sim:.2f})")
                is_dup = True
                break

        if is_dup:
            skipped += 1
            continue

        # Keep first occurrence (write ORIGINAL raw text, not canonicalized)
        of = _write_mirrored(out_root, in_path if in_path.is_dir() else f.parent, f, raw)
        if kept < 6:
            print(f">> KEEP {f}  →  {of}  ({len(raw)} chars)")
        signatures.append((f, sig))
        kept += 1

    print(f"== dedup stats == {{'inputs': {len(files)}, 'kept': {kept}, 'skipped': {skipped}}}")
    return 0

if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
