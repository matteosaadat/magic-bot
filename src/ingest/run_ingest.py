#!/usr/bin/env python3
# ================================================================
# run_ingest.py
# ----------------------------------------------------------------
# Orchestrates: 0_collect_extract → 1_normalize → 2_paraphrase → 3_dedup → 4_embed
# Uses ingest.config.yaml if present.
#
# Flags (activate-only):
#   --collect-only
#   --normalize-only
#   --paraphrase-only
#   --dedup-only
#   --embed-only
#
# General:
#   --src   <one or more input dirs>  (required unless running only stages that don't need raw sources)
#   --db    <sqlite path>             (needed for embed stage)
#   --faiss <faiss index path>        (needed for embed stage)
#   --ocr-images                      (pass to collect stage)
#
# Behavior:
#   - If any *-only flag is set, **only that stage** runs.
#   - If none set, full pipeline runs (collect→normalize→paraphrase→dedup→embed).
#   - Temporary working dirs live under: src/data/tmp/{extracted,normalized,paraphrased,deduped}
# ================================================================

from __future__ import annotations
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import pathlib

os.environ.setdefault("PYTHONPATH", str(pathlib.Path(__file__).resolve().parent))

def sh(cmd: List[str]) -> None:
    """Run a subprocess and stream its output; raise on nonzero."""
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def _load_cfg() -> dict:
    """Load ingest.config.yaml if present (optional)."""
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    cfg_path = Path(__file__).with_name("ingest.config.yaml")
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text()) or {}

def _nonempty_dir(p: Path) -> bool:
    return p.exists() and any(p.iterdir())

def _require_dir(p: Path, msg: str) -> None:
    if not _nonempty_dir(p):
        print(f"ERROR: {msg}: {p}", flush=True)
        sys.exit(2)

def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Orchestrator (activate-only flags)")
    ap.add_argument("--src", nargs="+", help="One or more input folders (raw sources)")
    ap.add_argument("--db", help="SQLite DB output path (required for embed)")
    ap.add_argument("--faiss", help="FAISS index output path (required for embed)")
    ap.add_argument("--ocr-images", action="store_true", help="Enable OCR in collect stage")

    # Mutually exclusive: run exactly one stage (otherwise full pipeline)
    only = ap.add_mutually_exclusive_group()
    only.add_argument("--collect-only", action="store_true")
    only.add_argument("--normalize-only", action="store_true")
    only.add_argument("--paraphrase-only", action="store_true")
    only.add_argument("--dedup-only", action="store_true")
    only.add_argument("--embed-only", action="store_true")

    args = ap.parse_args()

    # Layout
    # repo_root/src/ingest/run_ingest.py → repo_root/src
    src_root = Path(__file__).resolve().parents[1]
    ingest_dir = src_root / "ingest"
    data_root = src_root / "data"
    tmp = data_root / "tmp"
    extracted = tmp / "extracted"
    normalized = tmp / "normalized"
    paraphrased = tmp / "paraphrased"
    deduped = tmp / "deduped"

    # Ensure working dirs exist
    for p in (tmp, extracted, normalized, paraphrased, deduped):
        p.mkdir(parents=True, exist_ok=True)

    # Config (optional)
    cfg = _load_cfg()
    chunk_size = str(cfg.get("ingest", {}).get("chunk_size", 900))
    chunk_overlap = str(cfg.get("ingest", {}).get("chunk_overlap", 120))

    # Echo run context
    print("== Ingest run info ==")
    print("script  :", Path(__file__).resolve())
    print("argv    :", sys.argv)
    print("python  :", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("cwd     :", os.getcwd())
    print("src     :", args.src)
    print("db      :", args.db)
    print("faiss   :", args.faiss)
    print("ocr     :", args.ocr_images)
    print(
        "modes   :",
        f"collect-only={args.collect_only},",
        f"normalize-only={args.normalize_only},",
        f"paraphrase-only={args.paraphrase_only},",
        f"dedup-only={args.dedup_only},",
        f"embed-only={args.embed_only}",
        flush=True,
    )

    # Helper: ensure src provided when needed
    def _require_src():
        if not args.src:
            print("ERROR: --src is required for collect/full pipeline.", flush=True)
            sys.exit(2)

    # ========== Single-stage modes ==========
    if args.collect_only:
        _require_src()
        sh([
            sys.executable, str(ingest_dir / "0_collect_extract.py"),
            "--inputs", *[str(s) for s in args.src],
            "--out", str(extracted),
            *(["--ocr-images"] if args.ocr_images else []),
        ])
        print("== collect-only complete ==", flush=True)
        return

    if args.normalize_only:
        _require_dir(extracted, "Nothing to normalize; expected collected/extracted data")
        sh([
            sys.executable, str(ingest_dir / "1_normalize.py"),
            "--input", str(extracted),
            "--output", str(normalized),
            "--chunk-size", chunk_size,
            "--chunk-overlap", chunk_overlap,
        ])
        print("== normalize-only complete ==", flush=True)
        return

    if args.paraphrase_only:
        in_dir = normalized if _nonempty_dir(normalized) else extracted
        _require_dir(in_dir, "Nothing to paraphrase; expected normalized or extracted data")
        sh([
            sys.executable, str(ingest_dir / "2_paraphrase.py"),
            "--input", str(in_dir),
            "--output", str(paraphrased),
        ])
        print("== paraphrase-only complete ==", flush=True)
        return

    if args.dedup_only:
        # prefer paraphrased → normalized → extracted
        in_dir = (
            paraphrased if _nonempty_dir(paraphrased)
            else normalized if _nonempty_dir(normalized)
            else extracted
        )
        _require_dir(in_dir, "Nothing to dedup; expected output from a prior stage")
        sh([
            sys.executable, str(ingest_dir / "3_dedup.py"),
            "--input", str(in_dir),
            "--output", str(deduped),
        ])
        print("== dedup-only complete ==", flush=True)
        return

    if args.embed_only:
        if not args.db or not args.faiss:
            print("ERROR: --db and --faiss are required for embed-only.", flush=True)
            sys.exit(2)
        # prefer deduped → paraphrased → normalized → extracted
        in_dir = (
            deduped if _nonempty_dir(deduped)
            else paraphrased if _nonempty_dir(paraphrased)
            else normalized if _nonempty_dir(normalized)
            else extracted
        )
        _require_dir(in_dir, "Nothing to embed; expected output from a prior stage")
        sh([
            sys.executable, str(ingest_dir / "4_embed.py"),
            "--input", str(in_dir),
            "--db", args.db,
            "--faiss", args.faiss,
        ])
        print("== embed-only complete ==", flush=True)
        return

    # ========== Full pipeline ==========
    # If we got here, no *-only flags were provided → run all stages.
    _require_src()

    # 0) Collect
    sh([
        sys.executable, str(ingest_dir / "0_collect_extract.py"),
        "--inputs", *[str(s) for s in args.src],
        "--out", str(extracted),
        *(["--ocr-images"] if args.ocr_images else []),
    ])

    # 1) Normalize
    _require_dir(extracted, "Nothing to normalize; expected collected/extracted data")
    sh([
        sys.executable, str(ingest_dir / "1_normalize.py"),
        "--input", str(extracted),
        "--output", str(normalized),
        "--chunk-size", chunk_size,
        "--chunk-overlap", chunk_overlap,
    ])

    # 2) Paraphrase
    in_dir = normalized if _nonempty_dir(normalized) else extracted
    _require_dir(in_dir, "Nothing to paraphrase; expected normalized or extracted data")
    sh([
        sys.executable, str(ingest_dir / "2_paraphrase.py"),
        "--input", str(in_dir),
        "--output", str(paraphrased),
    ])

    # 3) Dedup
    in_dir = paraphrased if _nonempty_dir(paraphrased) else in_dir
    _require_dir(in_dir, "Nothing to dedup; expected paraphrased/normalized/extracted data")
    sh([
        sys.executable, str(ingest_dir / "3_dedup.py"),
        "--input", str(in_dir),
        "--output", str(deduped),
    ])

    # 4) Embed
    if not args.db or not args.faiss:
        print("ERROR: --db and --faiss are required for embed stage.", flush=True)
        sys.exit(2)
    in_dir = deduped if _nonempty_dir(deduped) else in_dir
    _require_dir(in_dir, "Nothing to embed; expected output from previous stage(s)")
    sh([
        sys.executable, str(ingest_dir / "4_embed.py"),
        "--input", str(in_dir),
        "--db", args.db,
        "--faiss", args.faiss,
    ])

    print("== Ingest complete ==", flush=True)

if __name__ == "__main__":
    main()
