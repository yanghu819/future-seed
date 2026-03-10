#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
MAIN_TEX = ROOT / "main.tex"
AUDIT_MD = ROOT / "REFERENCE_AUDIT.md"

ALLOWED_DOMAINS = {
    "proceedings.neurips.cc",
    "aclanthology.org",
    "jmlr.org",
    "openaccess.thecvf.com",
    "doi.org",
    "arxiv.org",
}


def extract_cite_keys(tex: str) -> list[str]:
    keys: list[str] = []
    for raw in re.findall(r"\\citep\{([^}]+)\}", tex):
        keys.extend(part.strip() for part in raw.split(",") if part.strip())
    return sorted(set(keys))


def extract_summary_urls(md: str) -> dict[str, str]:
    urls: dict[str, str] = {}
    table_pat = re.compile(r"^\| `([^`]+)` \| .* \| (https?://[^| ]+) \|$", re.MULTILINE)
    for key, url in table_pat.findall(md):
        urls[key] = url
    return urls


def extract_detailed_heads(md: str) -> list[str]:
    return re.findall(r"^### `([^`]+)`$", md, re.MULTILINE)


def validate_domain(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in ALLOWED_DOMAINS


def main() -> None:
    tex = MAIN_TEX.read_text(encoding="utf-8")
    md = AUDIT_MD.read_text(encoding="utf-8")

    cite_keys = extract_cite_keys(tex)
    summary_urls = extract_summary_urls(md)
    detail_heads = extract_detailed_heads(md)

    errors: list[str] = []

    missing_summary = [key for key in cite_keys if key not in summary_urls]
    missing_detail = [key for key in cite_keys if key not in detail_heads]
    extra_summary = [key for key in summary_urls if key not in cite_keys]
    extra_detail = [key for key in detail_heads if key not in cite_keys]

    if missing_summary:
        errors.append(f"missing summary rows: {', '.join(missing_summary)}")
    if missing_detail:
        errors.append(f"missing detail sections: {', '.join(missing_detail)}")
    if extra_summary:
        errors.append(f"extra summary rows not cited in main.tex: {', '.join(extra_summary)}")
    if extra_detail:
        errors.append(f"extra detail sections not cited in main.tex: {', '.join(extra_detail)}")

    for key, url in sorted(summary_urls.items()):
        if not validate_domain(url):
            errors.append(f"{key}: unapproved source domain for {url}")

        section_pat = re.compile(
            rf"### `{re.escape(key)}`\n(?P<body>.*?)(?=\n### `|\Z)",
            re.DOTALL,
        )
        match = section_pat.search(md)
        if not match:
            continue
        body = match.group("body")
        required_markers = [
            "Why cited here:",
            "What claim it supports in our paper:",
            "What it does **not** support here:",
            "Primary source:",
        ]
        for marker in required_markers:
            if marker not in body:
                errors.append(f"{key}: missing marker '{marker}' in detailed audit section")

    if errors:
        print("reference_audit_failed")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("reference_audit_ok")
    print(f"- cited keys: {len(cite_keys)}")
    print(f"- audited keys: {len(detail_heads)}")
    print(f"- allowed domains: {', '.join(sorted(ALLOWED_DOMAINS))}")


if __name__ == "__main__":
    main()
