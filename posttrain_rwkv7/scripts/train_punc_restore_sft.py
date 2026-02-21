#!/usr/bin/env python3

"""Punctuation/case restoration probe (ASR post-processing proxy) for Future-Seed.

Task:
  - Input prompt contains a noisy transcript (lowercased, punctuation removed).
  - Model predicts the clean sentence.

This is a real-text repair setting from WikiText lines, useful as a fast proxy for
ASR post-processing where right-side context helps punctuation decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset

from rwkv_tokenizer import RWKVWorldTokenizer
from rwkv7_g1d import RWKV7G1DLM


PUNC_CHARS = set(".,;:!?\"()[]{}")


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def pad_left(seqs: List[List[int]], pad_id: int, multiple: int = 16) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    max_len = round_up(max_len, multiple)
    out = []
    for s in seqs:
        out.append([pad_id] * (max_len - len(s)) + s)
    return torch.tensor(out, dtype=torch.long)


def pad_right(seqs: List[List[int]], pad_id: int, multiple: int = 16) -> Tuple[torch.Tensor, int]:
    max_len = max((len(s) for s in seqs), default=0)
    if max_len == 0:
        return torch.empty((len(seqs), 0), dtype=torch.long), 0
    max_len_pad = round_up(max_len, multiple)
    out = []
    for s in seqs:
        out.append(s + [pad_id] * (max_len_pad - len(s)))
    return torch.tensor(out, dtype=torch.long), max_len


def _clean_line(s: str) -> str:
    s = str(s).replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_candidate(s: str, min_chars: int, max_chars: int) -> bool:
    if len(s) < min_chars or len(s) > max_chars:
        return False
    if not any(ch in PUNC_CHARS for ch in s):
        return False
    alpha = sum(ch.isalpha() for ch in s)
    return alpha >= max(24, len(s) // 3)


def _corrupt_text(s: str) -> str:
    out = []
    for ch in s:
        if ch in PUNC_CHARS:
            out.append(" ")
        else:
            out.append(ch.lower())
    t = "".join(out)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _build_prompt(
    *,
    noisy: str,
    notes: List[str],
    tok: RWKVWorldTokenizer,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    fill_notes_to_max: bool,
    rng: random.Random,
) -> List[int] | None:
    prefix = "Background notes:\n"
    suffix = f"\n\nNoisy transcript:\n{noisy}\n\nTask: Restore punctuation and casing.\nAnswer:"

    prefix_ids = tok.encode(prefix)
    suffix_ids = tok.encode(suffix)

    if len(prefix_ids) + len(suffix_ids) > max_prompt_tokens:
        return None

    if fill_notes_to_max:
        budget = max_prompt_tokens - len(prefix_ids) - len(suffix_ids)
        note_ids: List[int] = []
        tries = 0
        while len(note_ids) < budget and tries < 256:
            tries += 1
            n = rng.choice(notes)
            ids = tok.encode(n + "\n")
            if not ids:
                continue
            remain = budget - len(note_ids)
            if remain <= 0:
                break
            note_ids.extend(ids[:remain])
        p_ids = prefix_ids + note_ids + suffix_ids
    else:
        n = rng.choice(notes) if notes else ""
        p_ids = prefix_ids + tok.encode(n) + suffix_ids
        if len(p_ids) > max_prompt_tokens:
            p_ids = p_ids[-max_prompt_tokens:]

    if len(p_ids) < min_prompt_tokens:
        return None
    if len(p_ids) > max_prompt_tokens:
        p_ids = p_ids[-max_prompt_tokens:]
    return p_ids


def build_examples(
    *,
    ds: str,
    ds_cfg: str,
    split: str,
    tok: RWKVWorldTokenizer,
    n: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    max_answer_tokens: int,
    min_chars: int,
    max_chars: int,
    fill_notes_to_max: bool,
    note_pool_size: int,
    seed: int,
) -> List[Tuple[List[int], List[int]]]:
    ds_obj = load_dataset(ds, ds_cfg, split=split)
    rng = random.Random(seed)

    lines: List[str] = []
    for ex in ds_obj:
        t = _clean_line(ex.get("text", ""))
        if not t:
            continue
        if _is_candidate(t, min_chars=min_chars, max_chars=max_chars):
            lines.append(t)
    if not lines:
        raise RuntimeError("No candidate lines found for punctuation restoration.")

    rng.shuffle(lines)
    notes = lines[: min(len(lines), max(64, note_pool_size))]

    out: List[Tuple[List[int], List[int]]] = []
    ptr = 0
    while len(out) < n and ptr < len(lines):
        clean = lines[ptr]
        ptr += 1
        noisy = _corrupt_text(clean)
        if noisy == clean:
            continue

        p_ids = _build_prompt(
            noisy=noisy,
            notes=notes,
            tok=tok,
            max_prompt_tokens=max_prompt_tokens,
            min_prompt_tokens=min_prompt_tokens,
            fill_notes_to_max=fill_notes_to_max,
            rng=rng,
        )
        if p_ids is None:
            continue

        a_ids = tok.encode(" " + clean)
        if len(a_ids) > max_answer_tokens:
            a_ids = a_ids[:max_answer_tokens]
        if not a_ids:
            continue
        out.append((p_ids, a_ids))

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} examples (wanted {n}). Try reducing min constraints.")
    return out


@torch.no_grad()
def token_acc_from_preds(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    ok = ((pred == tgt) & mask).float().sum().item()
    denom = mask.float().sum().clamp(min=1).item()
    return float(ok / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["no_fs", "prompt_fs"], default="no_fs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_data_seed", type=int, default=0)
    ap.add_argument("--val_data_seed", type=int, default=1234)

    ap.add_argument("--ds", type=str, default="wikitext")
    ap.add_argument("--ds_cfg", type=str, default="wikitext-2-raw-v1")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="validation")

    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--n_val", type=int, default=240)
    ap.add_argument("--min_chars", type=int, default=64)
    ap.add_argument("--max_chars", type=int, default=220)
    ap.add_argument("--fill_notes_to_max", action="store_true")
    ap.add_argument("--note_pool_size", type=int, default=1024)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--min_prompt_tokens", type=int, default=768)
    ap.add_argument("--max_answer_tokens", type=int, default=180)

    ap.add_argument("--fs_variant", choices=["scalar", "head"], default="scalar")
    ap.add_argument("--alpha_head_lr", type=float, default=None)
    ap.add_argument("--alpha_head_init", type=float, default=None)

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
        "ds_cfg": args.ds_cfg,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "min_chars": int(args.min_chars),
        "max_chars": int(args.max_chars),
        "fill_notes_to_max": bool(args.fill_notes_to_max),
        "note_pool_size": int(args.note_pool_size),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "min_prompt_tokens": int(args.min_prompt_tokens),
        "max_answer_tokens": int(args.max_answer_tokens),
        "train_data_seed": int(args.train_data_seed),
        "val_data_seed": int(args.val_data_seed),
        "vocab": str(args.vocab),
    }
    cache_key = hashlib.md5(json.dumps(cache_meta, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / f"punc_restore_tok_{cache_key}.pt"

    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu")
        train_ex = data["train_ex"]
        val_ex = data["val_ex"]
        print(f"Loaded cache: {cache_path}")
    else:
        print("Loading data...")
        train_ex = build_examples(
            ds=args.ds,
            ds_cfg=args.ds_cfg,
            split=args.train_split,
            tok=tok,
            n=int(args.n_train),
            max_prompt_tokens=int(args.max_prompt_tokens),
            min_prompt_tokens=int(args.min_prompt_tokens),
            max_answer_tokens=int(args.max_answer_tokens),
            min_chars=int(args.min_chars),
            max_chars=int(args.max_chars),
            fill_notes_to_max=bool(args.fill_notes_to_max),
            note_pool_size=int(args.note_pool_size),
            seed=int(args.train_data_seed),
        )
        val_ex = build_examples(
            ds=args.ds,
            ds_cfg=args.ds_cfg,
            split=args.val_split,
            tok=tok,
            n=int(args.n_val),
            max_prompt_tokens=int(args.max_prompt_tokens),
            min_prompt_tokens=int(args.min_prompt_tokens),
            max_answer_tokens=int(args.max_answer_tokens),
            min_chars=int(args.min_chars),
            max_chars=int(args.max_chars),
            fill_notes_to_max=bool(args.fill_notes_to_max),
            note_pool_size=int(args.note_pool_size),
            seed=int(args.val_data_seed),
        )
        torch.save({"train_ex": train_ex, "val_ex": val_ex, "meta": cache_meta}, cache_path)
        print(f"Saved cache: {cache_path}")

    run_root = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S") / "punc_restore_sft" / args.mode
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

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

    def sample_batch(examples: List[Tuple[List[int], List[int]]], *, rng: random.Random) -> Tuple[torch.Tensor, List[List[int]]]:
        ps, ans = [], []
        for _ in range(int(args.bsz)):
            p_ids, a_ids = rng.choice(examples)
            ps.append(p_ids)
            ans.append(a_ids)
        p = pad_left(ps, pad_id=pad_id, multiple=16).to(device)
        return p, ans

    t0 = time.time()
    step = 0
    while True:
        if time.time() - t0 > float(args.time_budget_sec):
            break
        if args.max_steps and step >= int(args.max_steps):
            break

        opt.zero_grad(set_to_none=True)
        prompt_ids, ans_list = sample_batch(train_ex, rng=train_rng)
        use_fs = args.mode == "prompt_fs"
        prompt_hidden, prompt_states = model(
            prompt_ids,
            future_seed=use_fs,
            fs_alpha=alpha,
            fs_alpha_head=alpha_head,
            seed_scale=float(args.seed_scale),
            fs_layer_start=int(args.fs_layer_start),
            fs_alpha_schedule=str(args.fs_alpha_schedule),
            fs_alpha_min=float(args.fs_alpha_min),
            fs_alpha_max=float(args.fs_alpha_max),
            fs_norm=bool(args.fs_norm),
            fs_clip=float(args.fs_clip),
            fs_detach=bool(args.fs_detach),
            return_states=True,
        )
        assert prompt_states is not None

        a0 = torch.tensor([a[0] for a in ans_list], device=device, dtype=torch.long)
        logits0 = model.project(prompt_hidden[:, -1, :])
        loss0 = F.cross_entropy(logits0, a0)

        ans_in = [a[:-1] for a in ans_list]
        ans_tgt = [a[1:] for a in ans_list]
        ans_in_pad, ans_in_len = pad_right(ans_in, pad_id=pad_id, multiple=16)
        ans_in_pad = ans_in_pad.to(device)

        loss_rest = torch.tensor(0.0, device=device)
        if ans_in_len > 0:
            ans_hidden, _ = model(
                ans_in_pad,
                seed_states=prompt_states,
                future_seed=False,
                fs_alpha=None,
                fs_alpha_head=None,
                seed_scale=1.0,
                return_states=False,
            )
            ans_hidden = ans_hidden[:, :ans_in_len, :].contiguous()
            logits = model.project(ans_hidden)
            tgt = torch.full((int(args.bsz), ans_in_len), -100, device=device, dtype=torch.long)
            for i, t in enumerate(ans_tgt):
                if not t:
                    continue
                tt = torch.tensor(t, device=device, dtype=torch.long)
                tgt[i, : tt.numel()] = tt
            loss_rest = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-100)

        loss = loss0 + loss_rest
        loss.backward()
        opt.step()

        if (step % int(args.eval_every)) == 0:
            model.eval()
            with torch.no_grad():
                val_losses: List[float] = []
                val_accs: List[float] = []
                for _ in range(int(args.val_batches)):
                    vp, vans_list = sample_batch(val_ex, rng=val_rng)
                    v_hidden, v_states = model(
                        vp,
                        future_seed=(args.mode == "prompt_fs"),
                        fs_alpha=alpha,
                        fs_alpha_head=alpha_head,
                        seed_scale=float(args.seed_scale),
                        fs_layer_start=int(args.fs_layer_start),
                        fs_alpha_schedule=str(args.fs_alpha_schedule),
                        fs_alpha_min=float(args.fs_alpha_min),
                        fs_alpha_max=float(args.fs_alpha_max),
                        fs_norm=bool(args.fs_norm),
                        fs_clip=float(args.fs_clip),
                        fs_detach=bool(args.fs_detach),
                        return_states=True,
                    )
                    assert v_states is not None

                    max_len = max(len(a) for a in vans_list)
                    vtgt = torch.full((int(args.bsz), max_len), -100, device=device, dtype=torch.long)
                    vmask = torch.zeros((int(args.bsz), max_len), device=device, dtype=torch.bool)
                    for i, a in enumerate(vans_list):
                        aa = torch.tensor(a, device=device, dtype=torch.long)
                        vtgt[i, : aa.numel()] = aa
                        vmask[i, : aa.numel()] = True

                    va0 = vtgt[:, 0].contiguous()
                    vlogits0 = model.project(v_hidden[:, -1, :])
                    vloss0 = F.cross_entropy(vlogits0, va0)
                    vpred = torch.empty((int(args.bsz), max_len), device=device, dtype=torch.long)
                    vpred[:, 0] = vlogits0.argmax(dim=-1)

                    vloss_rest = torch.tensor(0.0, device=device)
                    if max_len > 1:
                        v_ans_in = [a[:-1] for a in vans_list]
                        v_ans_tgt = [a[1:] for a in vans_list]
                        v_in_pad, v_in_len = pad_right(v_ans_in, pad_id=pad_id, multiple=16)
                        v_in_pad = v_in_pad.to(device)
                        v_ans_hidden, _ = model(
                            v_in_pad,
                            seed_states=v_states,
                            future_seed=False,
                            fs_alpha=None,
                            fs_alpha_head=None,
                            seed_scale=1.0,
                            return_states=False,
                        )
                        v_ans_hidden = v_ans_hidden[:, :v_in_len, :].contiguous()
                        v_logits = model.project(v_ans_hidden)
                        v_rest_tgt = torch.full((int(args.bsz), v_in_len), -100, device=device, dtype=torch.long)
                        for i, t in enumerate(v_ans_tgt):
                            if not t:
                                continue
                            tt = torch.tensor(t, device=device, dtype=torch.long)
                            v_rest_tgt[i, : tt.numel()] = tt
                        vloss_rest = F.cross_entropy(
                            v_logits.view(-1, v_logits.size(-1)), v_rest_tgt.view(-1), ignore_index=-100
                        )
                        vpred[:, 1 : 1 + v_in_len] = v_logits.argmax(dim=-1)

                    vloss = vloss0 + vloss_rest
                    vacc = token_acc_from_preds(vpred, vtgt, vmask)
                    val_losses.append(float(vloss))
                    val_accs.append(float(vacc))

                rec = {
                    "t": round(time.time() - t0, 2),
                    "step": step,
                    "train_loss": float(loss),
                    "val_loss": sum(val_losses) / len(val_losses),
                    "val_tok_acc": sum(val_accs) / len(val_accs),
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

