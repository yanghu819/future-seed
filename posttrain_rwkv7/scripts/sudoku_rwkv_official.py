#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cache_defaults import apply_cache_env, ensure_cache_dirs, future_seed_cache_root

PINNED_COMMIT = "207f693603e4f041d4d683a70a809055b06c507e"
OFFICIAL_REPO = "Jellyfish042/Sudoku-RWKV"
OFFICIAL_FILES = (
    "README.md",
    "demo.py",
    "formatter.py",
    "generate_sudoku_data.py",
    "minimum_inference.py",
    "model.py",
    "rwkv_model.py",
    "sudoku_data.jsonl",
    "sudoku_vocab.txt",
    "utils.py",
    "assets/loss.png",
    "assets/menu.png",
    "assets/perfect_solution_rate.png",
    "assets/token_usage.png",
    "sudoku_rwkv_20241120.pth",
)
CHECKPOINT_FILE = "sudoku_rwkv_20241120.pth"


def snapshot_root(root: str | Path | None = None) -> Path:
    base = Path(root).expanduser() if root else future_seed_cache_root()
    return base / "sudoku-rwkv" / PINNED_COMMIT


def raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{OFFICIAL_REPO}/{PINNED_COMMIT}/{path}"


def metadata_path(dst: Path) -> Path:
    return dst / "snapshot_meta.json"


def iter_files(include_checkpoint: bool = True) -> Iterable[str]:
    for rel in OFFICIAL_FILES:
        if not include_checkpoint and rel == CHECKPOINT_FILE:
            continue
        yield rel


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dst: Path, *, verbose: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[download] {url} -> {dst}")
    with urllib.request.urlopen(url, timeout=60) as resp, dst.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def ensure_snapshot(
    *,
    root: str | Path | None = None,
    include_checkpoint: bool = True,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    apply_cache_env()
    ensure_cache_dirs()
    dst = snapshot_root(root)
    dst.mkdir(parents=True, exist_ok=True)
    for rel in iter_files(include_checkpoint=include_checkpoint):
        path = dst / rel
        if force or (not path.exists()) or path.stat().st_size == 0:
            _download_file(raw_url(rel), path, verbose=verbose)
    meta = {
        "repo": OFFICIAL_REPO,
        "commit": PINNED_COMMIT,
        "files": list(iter_files(include_checkpoint=include_checkpoint)),
    }
    metadata_path(dst).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dst


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fetch the pinned official Sudoku-RWKV snapshot")
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--skip-checkpoint", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--print-dir", action="store_true")
    ap.add_argument("--print-sha256", action="store_true")
    ap.add_argument("--self_test", action="store_true")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    dst = ensure_snapshot(
        root=args.root,
        include_checkpoint=not args.skip_checkpoint,
        force=args.force,
        verbose=not args.quiet,
    )
    if args.self_test:
        expected = {rel for rel in iter_files(include_checkpoint=not args.skip_checkpoint)}
        actual = {str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file()}
        missing = sorted(expected - actual)
        if missing:
            raise SystemExit(f"missing files: {missing}")
        print("sudoku_rwkv_snapshot_self_test_ok")
    if args.print_dir:
        print(dst)
    if args.print_sha256:
        ckpt = dst / CHECKPOINT_FILE
        if not ckpt.exists():
            raise SystemExit(f"checkpoint missing: {ckpt}")
        print(sha256_file(ckpt))


if __name__ == "__main__":
    main()
