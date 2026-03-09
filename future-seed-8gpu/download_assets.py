#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import snapshot_download
from modelscope.msdatasets import MsDataset

MODEL_ID_DEFAULT = "RWKV/RWKV7-Goose-World3-2.9B-HF"

DATASET_SPECS = {
    "mbpp": {
        "candidates": [
            {"id": "AI-ModelScope/mbpp", "subset": None, "train": "train", "eval": "test"},
            {"id": "modelscope/mbpp", "subset": None, "train": "train", "eval": "test"},
            {"id": "mbpp", "subset": None, "train": "train", "eval": "test"},
        ],
    },
    "arc": {
        "candidates": [
            {"id": "AI-ModelScope/ai2_arc", "subset": "ARC-Challenge", "train": "train", "eval": "test"},
            {"id": "AI-ModelScope/arc", "subset": "ARC-Challenge", "train": "train", "eval": "test"},
            {"id": "ai2_arc", "subset": "ARC-Challenge", "train": "train", "eval": "test"},
            {"id": "arc", "subset": "ARC-Challenge", "train": "train", "eval": "test"},
        ],
    },
    "race": {
        "candidates": [
            {"id": "AI-ModelScope/race", "subset": "all", "train": "train", "eval": "test"},
            {"id": "modelscope/race", "subset": "all", "train": "train", "eval": "test"},
            {"id": "race", "subset": "all", "train": "train", "eval": "test"},
        ],
    },
    "humaneval": {
        "candidates": [
            {"id": "AI-ModelScope/humaneval", "subset": None, "train": "test", "eval": "test"},
            {"id": "modelscope/humaneval", "subset": None, "train": "test", "eval": "test"},
            {"id": "humaneval", "subset": None, "train": "test", "eval": "test"},
        ],
    },
}


def materialize_records(ds: Any, limit: int | None = None) -> list[dict[str, Any]]:
    if hasattr(ds, "to_hf_dataset"):
        ds = ds.to_hf_dataset()
    records = []
    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        records.append(dict(row))
    return records


def try_load_dataset(name: str, cache_dir: str, limit_train: int, limit_eval: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    last_err = None
    for cand in DATASET_SPECS[name]["candidates"]:
        try:
            kwargs = {"split": cand["train"], "cache_dir": cache_dir}
            if cand["subset"] is not None:
                kwargs["subset_name"] = cand["subset"]
            train_ds = MsDataset.load(cand["id"], **kwargs)

            kwargs = {"split": cand["eval"], "cache_dir": cache_dir}
            if cand["subset"] is not None:
                kwargs["subset_name"] = cand["subset"]
            eval_ds = MsDataset.load(cand["id"], **kwargs)

            train_records = materialize_records(train_ds, limit_train)
            eval_records = materialize_records(eval_ds, limit_eval)
            if train_records and eval_records:
                meta = {"dataset_id": cand["id"], "subset": cand["subset"], "train_split": cand["train"], "eval_split": cand["eval"]}
                return train_records, eval_records, meta
        except Exception as exc:
            last_err = repr(exc)
    raise RuntimeError(f"failed_to_load_{name}_from_modelscope: {last_err}")


def normalize_mbpp(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in records:
        prompt = r.get("prompt") or r.get("text") or r.get("description")
        code = r.get("code") or r.get("canonical_solution") or r.get("solution")
        if prompt and code:
            out.append({"prompt": str(prompt), "code": str(code)})
    return out


def normalize_arc(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in records:
        q = r.get("question") or r.get("query") or r.get("text")
        answer = r.get("answerKey") or r.get("answer")
        choices = r.get("choices") or r.get("options")
        opts = []
        if isinstance(choices, dict):
            labels = choices.get("label") or choices.get("labels") or []
            texts = choices.get("text") or choices.get("texts") or []
            opts = [{"label": str(l), "text": str(t)} for l, t in zip(labels, texts)]
        elif isinstance(choices, list):
            for idx, c in enumerate(choices):
                if isinstance(c, dict):
                    label = c.get("label") or chr(ord('A') + idx)
                    text = c.get("text") or c.get("content") or ""
                else:
                    label = chr(ord('A') + idx)
                    text = str(c)
                opts.append({"label": str(label), "text": str(text)})
        if q and answer and len(opts) >= 2:
            out.append({"question": str(q), "choices": opts, "answer": str(answer).strip()})
    return out


def normalize_race(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in records:
        article = r.get("article") or r.get("context") or r.get("passage")
        question = r.get("question")
        options = r.get("options") or r.get("choices")
        answer = r.get("answer")
        opts = []
        if isinstance(options, list):
            for idx, c in enumerate(options):
                if isinstance(c, dict):
                    text = c.get("text") or c.get("content") or ""
                else:
                    text = str(c)
                opts.append({"label": chr(ord('A') + idx), "text": text})
        if article and question and answer and len(opts) >= 2:
            out.append({"article": str(article), "question": str(question), "choices": opts, "answer": str(answer).strip()})
    return out


def normalize_humaneval(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in records:
        prompt = r.get("prompt") or r.get("text")
        canonical = r.get("canonical_solution") or r.get("solution")
        entry = r.get("entry_point")
        if prompt and canonical:
            out.append({"prompt": str(prompt), "code": str(canonical), "entry_point": str(entry or "")})
    return out


NORMALIZERS = {
    "mbpp": normalize_mbpp,
    "arc": normalize_arc,
    "race": normalize_race,
    "humaneval": normalize_humaneval,
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("data"))
    ap.add_argument("--cache_dir", type=Path, default=Path("cache/modelscope"))
    ap.add_argument("--model_dir", type=Path, default=Path("models"))
    ap.add_argument("--model_id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--datasets", type=str, default="mbpp,arc,race,humaneval")
    ap.add_argument("--train_limit", type=int, default=8000)
    ap.add_argument("--eval_limit", type=int, default=1024)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(args.model_dir / args.model_id.replace('/', '__')),
        local_dir_use_symlinks=False,
    )
    print(json.dumps({"model_id": args.model_id, "model_path": model_path}))

    for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        train_rows, eval_rows, meta = try_load_dataset(name, str(args.cache_dir), args.train_limit, args.eval_limit)
        norm = NORMALIZERS[name]
        train_rows = norm(train_rows)
        eval_rows = norm(eval_rows)
        if not train_rows or not eval_rows:
            raise RuntimeError(f"normalized dataset empty for {name}")
        random.Random(1234).shuffle(train_rows)
        write_jsonl(args.out_dir / name / "train.jsonl", train_rows)
        write_jsonl(args.out_dir / name / "eval.jsonl", eval_rows)
        write_jsonl(args.out_dir / name / "meta.jsonl", [{"task": name, **meta, "train_rows": len(train_rows), "eval_rows": len(eval_rows)}])
        print(json.dumps({"task": name, **meta, "train_rows": len(train_rows), "eval_rows": len(eval_rows)}))
