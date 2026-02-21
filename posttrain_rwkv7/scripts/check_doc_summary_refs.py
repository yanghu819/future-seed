#!/usr/bin/env python3
"""
Check that summary files referenced by docs exist in `results/`.
Also reports unreferenced summary files and manifest coverage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SUMMARY_REF_RE = re.compile(r"((?:results|runs)/_summary_[A-Za-z0-9_\-\.]+\.txt)")


def extract_refs(path: Path) -> set[str]:
    refs: set[str] = set()
    text = path.read_text(encoding="utf-8")
    for m in SUMMARY_REF_RE.finditer(text):
        refs.add(m.group(1))
    return refs


def normalize_ref(ref: str) -> str:
    if ref.startswith("runs/"):
        return "results/" + ref.split("/", 1)[1]
    return ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    docs = [
        root / "README.md",
        root / "paper/FS_POSTTRAIN_PROGRESS_2026-02-19.md",
        root / "paper/DETAILED_EXPERIMENT_LOG.md",
    ]
    missing_docs = [p for p in docs if not p.exists()]
    if missing_docs:
        raise FileNotFoundError(f"missing docs: {missing_docs}")

    referenced_raw: set[str] = set()
    for doc in docs:
        referenced_raw |= extract_refs(doc)
    referenced = {normalize_ref(x) for x in referenced_raw}
    legacy_runs_refs = sorted(x for x in referenced_raw if x.startswith("runs/"))

    existing = {
        str(p.as_posix())
        for p in sorted((root / "results").glob("_summary_*.txt"))
    }
    existing_rel = {
        f"results/{Path(p).name}" for p in existing
    }

    missing = sorted(ref for ref in referenced if ref not in existing_rel)
    unreferenced = sorted(x for x in existing_rel if x not in referenced)

    manifest_path = root / "paper/exp_manifest.json"
    manifest_missing = []
    manifest_unmatched = []
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_entries = {
            exp.get("summary")
            for exp in data.get("experiments", [])
            if isinstance(exp.get("summary"), str)
        }
        manifest_missing = sorted(x for x in manifest_entries if x not in existing_rel)
        manifest_unmatched = sorted(x for x in existing_rel if x not in manifest_entries)

    print("doc_refs:", len(referenced))
    print("legacy_runs_refs:", len(legacy_runs_refs))
    for x in legacy_runs_refs:
        print("  LEGACY_RUNS_REF:", x)
    print("result_files:", len(existing_rel))
    print("missing_referenced:", len(missing))
    for x in missing:
        print("  MISSING:", x)
    print("unreferenced_results:", len(unreferenced))
    for x in unreferenced:
        print("  UNREFERENCED:", x)

    if manifest_path.exists():
        print("manifest_missing:", len(manifest_missing))
        for x in manifest_missing:
            print("  MANIFEST_MISSING_FILE:", x)
        print("manifest_unmatched:", len(manifest_unmatched))
        for x in manifest_unmatched:
            print("  FILE_NOT_IN_MANIFEST:", x)

    if args.strict and (missing or manifest_missing):
        sys.exit(1)


if __name__ == "__main__":
    main()
