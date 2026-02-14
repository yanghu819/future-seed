#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build uint16 token .bin files (with a 256-int32 header) from HuggingFace datasets.

Tokenizer: UTF-8 bytes (0..255). Use VOCAB_SIZE=256 in rwkv_diff_future_seed.py.

Example:
  python tools/build_hf_bins.py \
    --dataset wikitext --config wikitext-2-raw-v1 \
    --train_split train --val_split validation \
    --fields text \
    --out_dir data/wikitext2_bytes

  python tools/build_hf_bins.py \
    --dataset mbpp \
    --train_split train --val_split test \
    --fields code \
    --out_dir data/mbpp_bytes
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List

import numpy as np


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: datasets. Install with:\n"
            "  pip install -U datasets\n"
            f"Original error: {e}"
        )
    return load_dataset


def iter_text(ds, fields: List[str]) -> Iterable[str]:
    for ex in ds:
        parts = []
        for f in fields:
            if f not in ex:
                raise KeyError(f"field not found: {f} (available: {list(ex.keys())})")
            v = ex[f]
            if v is None:
                continue
            parts.append(str(v))
        if not parts:
            continue
        yield "\n".join(parts).strip() + "\n"


def encode_utf8_bytes(texts: Iterable[str], max_bytes: int | None) -> np.ndarray:
    buf = bytearray()
    for t in texts:
        b = t.encode("utf-8", errors="ignore")
        if max_bytes is not None and len(buf) + len(b) > max_bytes:
            remain = max_bytes - len(buf)
            if remain > 0:
                buf.extend(b[:remain])
            break
        buf.extend(b)
    arr = np.frombuffer(bytes(buf), dtype=np.uint8).astype(np.uint16)
    return arr


def write_bin(path: str, tokens_u16: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = np.zeros(256, dtype=np.int32)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_u16.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset name, e.g. wikitext / mbpp")
    ap.add_argument("--config", default="", help="HF dataset config, e.g. wikitext-2-raw-v1")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="validation")
    ap.add_argument("--fields", default="text", help="Comma-separated fields to concatenate, e.g. text,code")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_bytes", type=int, default=0, help="If >0, cap total bytes per split.")
    args = ap.parse_args()

    load_dataset = _require_datasets()

    name = args.dataset
    cfg = args.config if args.config else None
    fields = [s.strip() for s in args.fields.split(",") if s.strip()]
    max_bytes = None if args.max_bytes <= 0 else int(args.max_bytes)

    ds_train = load_dataset(name, cfg, split=args.train_split)
    ds_val = load_dataset(name, cfg, split=args.val_split)

    train_tokens = encode_utf8_bytes(iter_text(ds_train, fields), max_bytes=max_bytes)
    val_tokens = encode_utf8_bytes(iter_text(ds_val, fields), max_bytes=max_bytes)

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    write_bin(train_path, train_tokens)
    write_bin(val_path, val_tokens)

    print("Wrote:")
    print(f"  {train_path}  bytes={train_tokens.size}")
    print(f"  {val_path}    bytes={val_tokens.size}")


if __name__ == "__main__":
    main()

