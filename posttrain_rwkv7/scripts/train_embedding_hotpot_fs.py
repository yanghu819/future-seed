#!/usr/bin/env python3
"""Contrastive embedding probe on HotpotQA with optional Future-Seed.

This script trains a lightweight retrieval head (and optionally FS alpha gates)
on top of RWKV7 frozen backbone, then reports retrieval metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

_HERE = Path(__file__).resolve()
_PARENT = _HERE.parent.parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from rwkv7_g1d import RWKV7G1DLM
from rwkv_tokenizer import RWKVWorldTokenizer


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def pad_right(seqs: List[List[int]], pad_id: int, multiple: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(s) for s in seqs), default=0)
    if max_len == 0:
        return torch.empty((len(seqs), 0), dtype=torch.long), torch.zeros((len(seqs),), dtype=torch.long)
    max_len_pad = round_up(max_len, multiple)
    out = []
    lens = []
    for s in seqs:
        out.append(s + [pad_id] * (max_len_pad - len(s)))
        lens.append(len(s))
    return torch.tensor(out, dtype=torch.long), torch.tensor(lens, dtype=torch.long)


def _clean(s: str) -> str:
    return str(s).replace("\n", " ").replace("\r", " ").strip()


def _build_context(ex: dict) -> str:
    parts: List[str] = []
    ctx = ex.get("context", None)
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sents = ctx.get("sentences", [])
        for title, sent_list in zip(titles, sents):
            t = _clean(title)
            if t:
                parts.append(t + ":")
            if isinstance(sent_list, list):
                chunk = " ".join(_clean(x) for x in sent_list if _clean(x))
                if chunk:
                    parts.append(chunk)
            else:
                chunk = _clean(str(sent_list))
                if chunk:
                    parts.append(chunk)
    elif isinstance(ctx, list):
        for item in ctx:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            t = _clean(item[0])
            if t:
                parts.append(t + ":")
            sent_list = item[1]
            if isinstance(sent_list, list):
                chunk = " ".join(_clean(x) for x in sent_list if _clean(x))
                if chunk:
                    parts.append(chunk)
            else:
                chunk = _clean(str(sent_list))
                if chunk:
                    parts.append(chunk)
    return "\n\n".join([p for p in parts if p])


def build_pairs(
    *,
    ds: str,
    cfg: str,
    split: str,
    tok: RWKVWorldTokenizer,
    n: int,
    max_q_tokens: int,
    max_doc_tokens: int,
    min_doc_tokens: int,
    seed: int,
) -> List[Tuple[List[int], List[int]]]:
    ds_obj = load_dataset(ds, cfg, split=split)
    rng = random.Random(seed)
    idxs = list(range(len(ds_obj)))
    rng.shuffle(idxs)

    out: List[Tuple[List[int], List[int]]] = []
    for i in idxs:
        ex = ds_obj[int(i)]
        q = _clean(ex.get("question", ""))
        if not q:
            continue
        ctx = _build_context(ex)
        if not ctx:
            continue

        q_text = f"Question: {q}"
        d_text = f"Context:\n{ctx}"

        q_ids = tok.encode(q_text)
        d_ids = tok.encode(d_text)
        if len(q_ids) > max_q_tokens:
            q_ids = q_ids[:max_q_tokens]
        if len(d_ids) > max_doc_tokens:
            d_ids = d_ids[:max_doc_tokens]
        if len(d_ids) < min_doc_tokens:
            continue
        if len(q_ids) < 4:
            continue

        out.append((q_ids, d_ids))
        if len(out) >= n:
            break

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} pairs, want {n}. Lower --min_doc_tokens or n.")
    return out


class EmbHead(nn.Module):
    def __init__(self, hidden: int, emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden, emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


@torch.no_grad()
def retrieval_metrics(q: torch.Tensor, d: torch.Tensor) -> Tuple[float, float, float]:
    # q,d: [N,D], already normalized
    sim = q @ d.T
    n = sim.size(0)
    ranks = []
    for i in range(n):
        row = sim[i]
        order = torch.argsort(row, descending=True)
        rank = int((order == i).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)
    r1 = sum(1 for r in ranks if r == 1) / n
    r5 = sum(1 for r in ranks if r <= 5) / n
    mrr10 = sum((1.0 / r) if r <= 10 else 0.0 for r in ranks) / n
    return float(r1), float(r5), float(mrr10)


def batch_iter(data: Sequence[Tuple[List[int], List[int]]], bsz: int, rng: random.Random):
    idxs = list(range(len(data)))
    rng.shuffle(idxs)
    for i in range(0, len(idxs), bsz):
        chunk = idxs[i : i + bsz]
        yield [data[j] for j in chunk]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "fs"], default="baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_data_seed", type=int, default=0)
    ap.add_argument("--val_data_seed", type=int, default=1234)

    ap.add_argument("--ds", type=str, default="hotpot_qa")
    ap.add_argument("--ds_cfg", type=str, default="distractor")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="validation")

    ap.add_argument("--n_train", type=int, default=1024)
    ap.add_argument("--n_val", type=int, default=256)
    ap.add_argument("--max_q_tokens", type=int, default=128)
    ap.add_argument("--max_doc_tokens", type=int, default=1024)
    ap.add_argument("--min_doc_tokens", type=int, default=256)

    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=240)
    ap.add_argument("--time_budget_sec", type=int, default=1200)
    ap.add_argument("--eval_every", type=int, default=20)

    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--temp", type=float, default=0.07)

    ap.add_argument("--alpha_lr", type=float, default=5e-4)
    ap.add_argument("--alpha_init", type=float, default=-2.0)
    ap.add_argument("--fs_layer_start", type=int, default=8)
    ap.add_argument("--seed_scale", type=float, default=1.0)
    ap.add_argument("--fs_norm", action="store_true")
    ap.add_argument("--fs_detach", action="store_true")
    ap.add_argument("--fs_clip", type=float, default=1.0)

    ap.add_argument("--weights", type=str, default="assets/weights/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
    ap.add_argument("--vocab", type=str, default="assets/tokenizer/rwkv_vocab_v20230424.txt")
    ap.add_argument("--cuda_src", type=str, default="cuda/rwkv_cuda_wind")
    ap.add_argument("--run_dir", type=str, default="runs")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = RWKVWorldTokenizer(args.vocab)

    cache_sig = hashlib.sha1(
        json.dumps(
            {
                "ds": args.ds,
                "cfg": args.ds_cfg,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "n_train": args.n_train,
                "n_val": args.n_val,
                "max_q_tokens": args.max_q_tokens,
                "max_doc_tokens": args.max_doc_tokens,
                "min_doc_tokens": args.min_doc_tokens,
                "train_data_seed": args.train_data_seed,
                "val_data_seed": args.val_data_seed,
                "vocab": args.vocab,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]

    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"embed_pairs_{cache_sig}.pt"
    if cache_path.exists():
        obj = torch.load(cache_path)
        train_pairs = obj["train"]
        val_pairs = obj["val"]
    else:
        train_pairs = build_pairs(
            ds=args.ds,
            cfg=args.ds_cfg,
            split=args.train_split,
            tok=tok,
            n=args.n_train,
            max_q_tokens=args.max_q_tokens,
            max_doc_tokens=args.max_doc_tokens,
            min_doc_tokens=args.min_doc_tokens,
            seed=args.train_data_seed,
        )
        val_pairs = build_pairs(
            ds=args.ds,
            cfg=args.ds_cfg,
            split=args.val_split,
            tok=tok,
            n=args.n_val,
            max_q_tokens=args.max_q_tokens,
            max_doc_tokens=args.max_doc_tokens,
            min_doc_tokens=args.min_doc_tokens,
            seed=args.val_data_seed,
        )
        torch.save({"train": train_pairs, "val": val_pairs}, cache_path)

    model = RWKV7G1DLM.from_pth(args.weights, cuda_src_dir=args.cuda_src, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    head = EmbHead(model.cfg.hidden_size, args.emb_dim).to(device)

    fs_alpha = None
    if args.mode == "fs":
        fs_alpha = nn.Parameter(torch.full((model.cfg.num_layers,), float(args.alpha_init), device=device, dtype=torch.float32))

    params = [{"params": list(head.parameters()), "lr": args.lr, "weight_decay": args.weight_decay}]
    if fs_alpha is not None:
        params.append({"params": [fs_alpha], "lr": args.alpha_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(params)

    run_base = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S") / "embedding_hotpot" / args.mode
    run_base.mkdir(parents=True, exist_ok=True)
    (run_base / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    metrics_f = (run_base / "metrics.jsonl").open("w", encoding="utf-8")

    def encode(ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if args.mode == "fs":
            kwargs.update(
                dict(
                    future_seed=True,
                    fs_alpha=fs_alpha,
                    seed_scale=float(args.seed_scale),
                    fs_layer_start=int(args.fs_layer_start),
                    fs_norm=bool(args.fs_norm),
                    fs_detach=bool(args.fs_detach),
                    fs_clip=float(args.fs_clip),
                )
            )
        h, _ = model(ids, **kwargs)
        # causal model + right padding: hidden at real positions is unaffected by pad suffix.
        t = h.size(1)
        mask = (torch.arange(t, device=h.device)[None, :] < lengths[:, None]).float()
        pooled = (h.float() * mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp(min=1).float().unsqueeze(-1)
        return head(pooled)

    @torch.no_grad()
    def eval_retrieval() -> Tuple[float, float, float]:
        model.eval()
        head.eval()
        q_all = []
        d_all = []
        for chunk in batch_iter(val_pairs, args.bsz, random.Random(1234)):
            q_ids = [x[0] for x in chunk]
            d_ids = [x[1] for x in chunk]
            q_pad, q_len = pad_right(q_ids, pad_id=tok.eot_id)
            d_pad, d_len = pad_right(d_ids, pad_id=tok.eot_id)
            q_pad = q_pad.to(device)
            d_pad = d_pad.to(device)
            q_len = q_len.to(device)
            d_len = d_len.to(device)
            q_emb = encode(q_pad, q_len)
            d_emb = encode(d_pad, d_len)
            q_all.append(q_emb)
            d_all.append(d_emb)
        q = torch.cat(q_all, dim=0)
        d = torch.cat(d_all, dim=0)
        return retrieval_metrics(q, d)

    best = {"step": -1, "val_r1": -1.0, "val_r5": -1.0, "val_mrr10": -1.0}

    start = time.time()
    step = 0
    train_rng = random.Random(args.seed + 123)
    while True:
        for chunk in batch_iter(train_pairs, args.bsz, train_rng):
            step += 1
            model.eval()
            head.train()

            q_ids = [x[0] for x in chunk]
            d_ids = [x[1] for x in chunk]
            q_pad, q_len = pad_right(q_ids, pad_id=tok.eot_id)
            d_pad, d_len = pad_right(d_ids, pad_id=tok.eot_id)
            q_pad = q_pad.to(device)
            d_pad = d_pad.to(device)
            q_len = q_len.to(device)
            d_len = d_len.to(device)

            q_emb = encode(q_pad, q_len)
            d_emb = encode(d_pad, d_len)

            logits = (q_emb @ d_emb.T) / float(args.temp)
            labels = torch.arange(logits.size(0), device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            now = time.time()
            rec = {
                "step": step,
                "train_loss": float(loss.item()),
                "elapsed_sec": float(now - start),
            }

            if step % int(args.eval_every) == 0 or step == 1:
                r1, r5, mrr10 = eval_retrieval()
                rec.update({"val_r1": r1, "val_r5": r5, "val_mrr10": mrr10})
                if r1 > best["val_r1"]:
                    best = {"step": step, "val_r1": r1, "val_r5": r5, "val_mrr10": mrr10}

            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()

            if step >= int(args.max_steps):
                break
            if now - start >= float(args.time_budget_sec):
                break
        if step >= int(args.max_steps) or (time.time() - start >= float(args.time_budget_sec)):
            break

    metrics_f.close()

    out = {
        "best": best,
        "last_step": step,
        "elapsed_sec": float(time.time() - start),
    }
    (run_base / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(run_base.as_posix())


if __name__ == "__main__":
    main()
