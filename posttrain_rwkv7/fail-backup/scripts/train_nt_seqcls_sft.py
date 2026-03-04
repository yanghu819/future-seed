#!/usr/bin/env python3

"""Nucleotide Transformer downstream sequence classification probe for Future-Seed.

This script adapts the NT downstream benchmark
(`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`) to the existing
RWKV Future-Seed quick->med pipeline.

Task format:
  - Input prompt contains one DNA sequence + task instruction.
  - Model predicts a single class symbol token (A/B/C/...) as the label.

We compare:
  - no_fs: normal causal prefill
  - prompt_fs: Future-Seed enabled for prompt/prefill

Metric:
  - val_cls_acc (also mirrored into val_tok_acc for compatibility with existing
    round orchestrators).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset

# Allow running from either repo root or scripts/ path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from rwkv_tokenizer import RWKVWorldTokenizer
from rwkv7_g1d import RWKV7G1DLM


CLASS_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DNA_CLEAN_RE = re.compile(r"[^ACGTN]")


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def pad_left(seqs: List[List[int]], pad_id: int, multiple: int = 16) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    max_len = round_up(max_len, multiple)
    out = []
    for s in seqs:
        out.append([pad_id] * (max_len - len(s)) + s)
    return torch.tensor(out, dtype=torch.long)


def _clean_seq(raw: str) -> str:
    s = str(raw).upper().replace("\n", "").replace("\r", "").replace(" ", "")
    return DNA_CLEAN_RE.sub("N", s)


def _class_symbol(label: int) -> str:
    if label < 0 or label >= len(CLASS_SYMBOLS):
        raise ValueError(f"label out of range: {label}")
    return CLASS_SYMBOLS[label]


def _load_dataset_dict(ds: str, retries: int = 6) -> dict:
    last_err = None
    dl_cfg = DownloadConfig(max_retries=10)
    for i in range(retries):
        try:
            return load_dataset(ds, download_config=dl_cfg)
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait_s = min(60, 8 * (i + 1))
            print(f"[warn] load_dataset failed ({i + 1}/{retries}): {repr(e)}; retry in {wait_s}s", flush=True)
            time.sleep(wait_s)
    raise RuntimeError(f"Failed to load dataset after {retries} retries: ds={ds}") from last_err


def _collect_rows(
    *,
    ds_obj,
    task_name: str,
    min_seq_len: int,
    max_seq_len: int,
) -> Tuple[List[Tuple[str, int, str]], List[int]]:
    rows: List[Tuple[str, int, str]] = []
    labels = set()
    for ex in ds_obj:
        if str(ex.get("task", "")) != task_name:
            continue

        seq = _clean_seq(ex.get("sequence", ""))
        if len(seq) < min_seq_len or len(seq) > max_seq_len:
            continue

        try:
            y = int(ex.get("label", -1))
        except Exception:
            continue
        if y < 0:
            continue

        name = str(ex.get("name", ""))
        rows.append((seq, y, name))
        labels.add(y)

    if not rows:
        raise RuntimeError(f"No usable rows for task={task_name}.")

    uniq = sorted(labels)
    if not uniq:
        raise RuntimeError(f"No labels found for task={task_name}.")
    if max(uniq) >= len(CLASS_SYMBOLS):
        raise RuntimeError(f"Too many classes for symbol mapping: max_label={max(uniq)}.")
    return rows, uniq


def _build_prompt(
    *,
    seq: str,
    sample_name: str,
    task_name: str,
    label_symbols: Dict[int, str],
    q_first: bool,
    include_name: bool,
) -> str:
    opts = "\n".join([f"{label_symbols[k]} = class {k}" for k in sorted(label_symbols)])
    name_block = f"Sequence ID: {sample_name}\n" if (include_name and sample_name) else ""

    if q_first:
        return (
            f"Task: classify this DNA sequence for {task_name}.\n"
            f"Label space:\n{opts}\n\n"
            f"{name_block}Sequence:\n{seq}\n\n"
            "Answer:"
        )
    return (
        f"{name_block}Sequence:\n{seq}\n\n"
        f"Task: classify this DNA sequence for {task_name}.\n"
        f"Label space:\n{opts}\n"
        "Answer:"
    )


def build_examples(
    *,
    ds_obj,
    task_name: str,
    tok: RWKVWorldTokenizer,
    n: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    min_seq_len: int,
    max_seq_len: int,
    seed: int,
    q_first: bool,
    include_name: bool,
) -> Tuple[List[Tuple[List[int], int]], Dict[int, str]]:
    rows, uniq_labels = _collect_rows(
        ds_obj=ds_obj,
        task_name=task_name,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
    )
    label_symbols = {y: _class_symbol(y) for y in uniq_labels}

    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)

    out: List[Tuple[List[int], int]] = []
    for i in idxs:
        seq, y, sample_name = rows[i]
        prompt = _build_prompt(
            seq=seq,
            sample_name=sample_name,
            task_name=task_name,
            label_symbols=label_symbols,
            q_first=q_first,
            include_name=include_name,
        )

        p_ids = tok.encode(prompt)
        if len(p_ids) > max_prompt_tokens:
            p_ids = p_ids[-max_prompt_tokens:]
        if len(p_ids) < min_prompt_tokens:
            continue

        label_ids = tok.encode(label_symbols[y])
        if not label_ids:
            continue
        out.append((p_ids, int(label_ids[0])))
        if len(out) >= n:
            break

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} examples (wanted {n}). Try lowering constraints.")
    return out, label_symbols


@torch.no_grad()
def cls_acc_from_logits(logits: torch.Tensor, tgt: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == tgt).float().mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["no_fs", "prompt_fs"], default="no_fs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_data_seed", type=int, default=0)
    ap.add_argument("--val_data_seed", type=int, default=1234)

    ap.add_argument("--ds", type=str, default="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    ap.add_argument("--task_name", type=str, default="splice_sites_all")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="test")
    ap.add_argument("--q_first", action="store_true", help="Instruction before sequence (causal-friendlier control).")
    ap.add_argument("--include_name", action="store_true", help="Include sample coordinate/id field in prompt.")

    ap.add_argument("--fs_variant", choices=["scalar", "head"], default="scalar")
    ap.add_argument("--alpha_head_lr", type=float, default=None)
    ap.add_argument("--alpha_head_init", type=float, default=None)

    ap.add_argument("--n_train", type=int, default=2400)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--max_prompt_tokens", type=int, default=1536)
    ap.add_argument("--min_prompt_tokens", type=int, default=384)
    ap.add_argument("--min_seq_len", type=int, default=200)
    ap.add_argument("--max_seq_len", type=int, default=2000)

    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--time_budget_sec", type=int, default=240)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=30)
    ap.add_argument("--val_batches", type=int, default=8)

    ap.add_argument("--model_lr", type=float, default=3e-5)
    ap.add_argument("--alpha_lr", type=float, default=0.0)
    ap.add_argument("--alpha_init", type=float, default=-2.0)
    ap.add_argument("--seed_scale", type=float, default=1.0)
    ap.add_argument("--fs_layer_start", type=int, default=8)
    ap.add_argument("--fs_alpha_schedule", choices=["none", "linear", "cosine"], default="none")
    ap.add_argument("--fs_alpha_min", type=float, default=1.0)
    ap.add_argument("--fs_alpha_max", type=float, default=1.0)
    ap.add_argument("--fs_norm", action="store_true")
    ap.add_argument("--fs_clip", type=float, default=1.0)
    ap.add_argument("--fs_detach", action="store_true")

    ap.add_argument("--weights", type=str, default="assets/weights/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
    ap.add_argument("--vocab", type=str, default="assets/tokenizer/rwkv_vocab_v20230424.txt")
    ap.add_argument("--cuda_src", type=str, default="cuda/rwkv_cuda_wind")
    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--run_dir", type=str, default="runs")
    args = ap.parse_args()

    if args.alpha_head_lr is None:
        args.alpha_head_lr = float(args.alpha_lr)
    if args.alpha_head_init is None:
        args.alpha_head_init = float(args.alpha_init)

    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_transformers")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = RWKVWorldTokenizer(args.vocab)
    pad_id = tok.eot_id

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_meta = {
        "ds": args.ds,
        "task_name": args.task_name,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "q_first": bool(args.q_first),
        "include_name": bool(args.include_name),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "min_prompt_tokens": int(args.min_prompt_tokens),
        "min_seq_len": int(args.min_seq_len),
        "max_seq_len": int(args.max_seq_len),
        "train_data_seed": int(args.train_data_seed),
        "val_data_seed": int(args.val_data_seed),
        "vocab": str(args.vocab),
    }
    cache_key = hashlib.md5(json.dumps(cache_meta, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / f"nt_seqcls_tok_{cache_key}.pt"

    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu")
        train_ex = data["train_ex"]
        val_ex = data["val_ex"]
        label_symbols = data["label_symbols"]
        print(f"Loaded cache: {cache_path}")
    else:
        print("Loading data...")
        dsd = _load_dataset_dict(args.ds)
        if args.train_split not in dsd:
            raise RuntimeError(f"train split not found: {args.train_split}")
        if args.val_split not in dsd:
            raise RuntimeError(f"val split not found: {args.val_split}")
        train_ex, train_symbols = build_examples(
            ds_obj=dsd[args.train_split],
            task_name=args.task_name,
            tok=tok,
            n=int(args.n_train),
            max_prompt_tokens=int(args.max_prompt_tokens),
            min_prompt_tokens=int(args.min_prompt_tokens),
            min_seq_len=int(args.min_seq_len),
            max_seq_len=int(args.max_seq_len),
            seed=int(args.train_data_seed),
            q_first=bool(args.q_first),
            include_name=bool(args.include_name),
        )
        val_ex, val_symbols = build_examples(
            ds_obj=dsd[args.val_split],
            task_name=args.task_name,
            tok=tok,
            n=int(args.n_val),
            max_prompt_tokens=int(args.max_prompt_tokens),
            min_prompt_tokens=int(args.min_prompt_tokens),
            min_seq_len=int(args.min_seq_len),
            max_seq_len=int(args.max_seq_len),
            seed=int(args.val_data_seed),
            q_first=bool(args.q_first),
            include_name=bool(args.include_name),
        )
        label_symbols = dict(train_symbols)
        label_symbols.update(val_symbols)
        torch.save({"train_ex": train_ex, "val_ex": val_ex, "label_symbols": label_symbols, "meta": cache_meta}, cache_path)
        print(f"Saved cache: {cache_path}")

    run_root = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S") / "nt_seqcls_sft" / args.mode
    run_root.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg["label_symbols"] = label_symbols
    (run_root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    train_rng = random.Random(int(args.seed))
    val_rng = random.Random(int(args.val_data_seed))

    model = RWKV7G1DLM.from_pth(args.weights, cuda_src_dir=args.cuda_src, device=device)
    model.train()

    alpha = torch.nn.Parameter(torch.full((model.cfg.num_layers,), float(args.alpha_init), device=device))
    alpha_head = None
    if args.fs_variant == "head":
        alpha_head = torch.nn.Parameter(
            torch.full((model.cfg.num_layers, model.cfg.num_heads), float(args.alpha_head_init), device=device)
        )

    param_groups = [
        {"params": model.parameters(), "lr": float(args.model_lr), "weight_decay": 0.01},
        {"params": [alpha], "lr": float(args.alpha_lr), "weight_decay": 0.0},
    ]
    if alpha_head is not None:
        param_groups.append({"params": [alpha_head], "lr": float(args.alpha_head_lr), "weight_decay": 0.0})
    opt = torch.optim.AdamW(param_groups)

    metrics_path = run_root / "metrics.jsonl"

    def sample_batch(
        examples: List[Tuple[List[int], int]],
        *,
        rng: random.Random,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ps, ys = [], []
        for _ in range(int(args.bsz)):
            p_ids, y = rng.choice(examples)
            ps.append(p_ids)
            ys.append(y)
        p = pad_left(ps, pad_id=pad_id, multiple=16).to(device)
        y = torch.tensor(ys, device=device, dtype=torch.long)
        return p, y

    t0 = time.time()
    step = 0
    while True:
        if time.time() - t0 > float(args.time_budget_sec):
            break
        if args.max_steps and step >= int(args.max_steps):
            break

        opt.zero_grad(set_to_none=True)
        prompt_ids, y = sample_batch(train_ex, rng=train_rng)
        use_fs = args.mode == "prompt_fs"
        hidden, _ = model(
            prompt_ids,
            future_seed=use_fs,
            fs_alpha=(alpha if use_fs else None),
            fs_alpha_head=(alpha_head if use_fs else None),
            seed_scale=float(args.seed_scale),
            fs_layer_start=int(args.fs_layer_start),
            fs_alpha_schedule=str(args.fs_alpha_schedule),
            fs_alpha_min=float(args.fs_alpha_min),
            fs_alpha_max=float(args.fs_alpha_max),
            fs_norm=bool(args.fs_norm),
            fs_clip=float(args.fs_clip),
            fs_detach=bool(args.fs_detach),
            return_states=False,
        )
        logits = model.project(hidden[:, -1, :])
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        if (step % int(args.eval_every)) == 0:
            model.eval()
            with torch.no_grad():
                val_losses: List[float] = []
                val_accs: List[float] = []
                for _ in range(int(args.val_batches)):
                    vp, vy = sample_batch(val_ex, rng=val_rng)
                    vhidden, _ = model(
                        vp,
                        future_seed=(args.mode == "prompt_fs"),
                        fs_alpha=(alpha if args.mode == "prompt_fs" else None),
                        fs_alpha_head=(alpha_head if args.mode == "prompt_fs" else None),
                        seed_scale=float(args.seed_scale),
                        fs_layer_start=int(args.fs_layer_start),
                        fs_alpha_schedule=str(args.fs_alpha_schedule),
                        fs_alpha_min=float(args.fs_alpha_min),
                        fs_alpha_max=float(args.fs_alpha_max),
                        fs_norm=bool(args.fs_norm),
                        fs_clip=float(args.fs_clip),
                        fs_detach=bool(args.fs_detach),
                        return_states=False,
                    )
                    vlogits = model.project(vhidden[:, -1, :])
                    vloss = F.cross_entropy(vlogits, vy)
                    vacc = cls_acc_from_logits(vlogits, vy)
                    val_losses.append(float(vloss))
                    val_accs.append(float(vacc))

                v_acc = sum(val_accs) / len(val_accs)
                rec = {
                    "t": round(time.time() - t0, 2),
                    "step": step,
                    "train_loss": float(loss),
                    "val_loss": sum(val_losses) / len(val_losses),
                    "val_tok_acc": v_acc,
                    "val_cls_acc": v_acc,
                    "alpha_mean": float(torch.sigmoid(alpha[1:]).mean()),
                    "fs_alpha_schedule": str(args.fs_alpha_schedule),
                    "fs_alpha_min": float(args.fs_alpha_min),
                    "fs_alpha_max": float(args.fs_alpha_max),
                    "alpha_head_mean": (float(torch.sigmoid(alpha_head[1:]).mean()) if alpha_head is not None else None),
                }
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            model.train()

        step += 1

    print(str(run_root))


if __name__ == "__main__":
    main()
