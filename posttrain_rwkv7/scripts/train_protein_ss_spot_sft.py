#!/usr/bin/env python3

"""Protein secondary-structure spot-label SFT probe for Future-Seed.

Real task (AI for Science):
  Predict DSSP-like secondary-structure symbols at queried residue indices.

We compare:
  - no_fs: normal causal prompt/prefill
  - prompt_fs: Future-Seed enabled for prompt/prefill only; decoding stays causal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from datasets import load_dataset

from rwkv_tokenizer import RWKVWorldTokenizer
from rwkv7_g1d import RWKV7G1DLM


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


def _clean(s: Any) -> str:
    return str(s).replace("\r", " ").replace("\n", " ").strip()


def _safe_seq(s: str) -> str:
    keep = []
    for ch in s.upper():
        if "A" <= ch <= "Z":
            keep.append(ch)
    return "".join(keep)


def _safe_ss(s: str) -> str:
    # Keep compact 1-char labels; dataset uses symbols like H/E/T/S/G/B/~/...
    keep = []
    for ch in str(s).strip():
        if ch not in {" ", "\n", "\r", "\t"}:
            keep.append(ch)
    return "".join(keep)


def _bucket_key(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 10


def _load_all_rows(ds: str, split: str) -> list[tuple[str, str, str]]:
    obj = load_dataset(ds, split=split)
    out: list[tuple[str, str, str]] = []
    for ex in obj:
        pid = _clean(ex.get("PDB_ID", ""))
        seq = _safe_seq(_clean(ex.get("Sequence", "")))
        ss = _safe_ss(_clean(ex.get("Secondary_structure", "")))
        if not seq or not ss:
            continue
        if len(seq) != len(ss):
            continue
        out.append((pid, seq, ss))
    return out


def _crop_pair(seq: str, ss: str, max_seq_len: int, rng: random.Random) -> tuple[str, str]:
    if len(seq) <= max_seq_len:
        return seq, ss
    start = rng.randrange(0, len(seq) - max_seq_len + 1)
    end = start + max_seq_len
    return seq[start:end], ss[start:end]


def _build_note_pool(
    rows: list[tuple[str, str, str]],
    *,
    tok: RWKVWorldTokenizer,
    rng: random.Random,
    pool_size: int,
    max_note_seq_len: int,
) -> list[list[int]]:
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    out: list[list[int]] = []
    for i in idxs:
        pid, seq, _ss = rows[int(i)]
        s = seq[:max_note_seq_len]
        txt = f"Protein note {pid}: {s}"
        ids = tok.encode(txt)
        if ids:
            out.append(ids)
        if len(out) >= pool_size:
            break
    return out


def _pick_positions(
    L: int,
    num_queries: int,
    query_region: str,
    rng: random.Random,
) -> list[int]:
    k = max(1, min(num_queries, L))
    if query_region == "prefix":
        return list(range(k))
    if query_region == "suffix":
        return list(range(L - k, L))
    return sorted(rng.sample(list(range(L)), k=k))


def _build_prompt_ids(
    *,
    tok: RWKVWorldTokenizer,
    seq: str,
    q_idx_1based: list[int],
    max_prompt_tokens: int,
    q_first: bool,
    fill_notes_to_max: bool,
    note_pool_ids: list[list[int]],
    rng: random.Random,
) -> list[int]:
    q_txt = ",".join(str(x) for x in q_idx_1based)
    inst = (
        "Task: Protein secondary-structure spot labeling.\n"
        "Output one symbol per queried residue index (same order).\n"
    )

    if not fill_notes_to_max:
        if q_first:
            p = f"{inst}\nNotes:\n\nSequence:\n{seq}\nQueries(1-based): {q_txt}\nAnswer:\n"
        else:
            p = f"{inst}\nSequence:\n{seq}\nQueries(1-based): {q_txt}\nNotes:\n\nAnswer:\n"
        p_ids = tok.encode(p)
        if len(p_ids) > max_prompt_tokens:
            p_ids = p_ids[-max_prompt_tokens:]
        return p_ids

    if q_first:
        prefix_ids = tok.encode(f"{inst}\nNotes:\n")
        suffix_ids = tok.encode(f"\n\nSequence:\n{seq}\nQueries(1-based): {q_txt}\nAnswer:\n")
    else:
        prefix_ids = tok.encode(f"{inst}\nSequence:\n{seq}\nQueries(1-based): {q_txt}\nNotes:\n")
        suffix_ids = tok.encode("\n\nAnswer:\n")

    budget_notes = max_prompt_tokens - len(prefix_ids) - len(suffix_ids)
    if budget_notes <= 0:
        p = f"{inst}\nSequence:\n{seq}\nQueries(1-based): {q_txt}\nAnswer:\n"
        p_ids = tok.encode(p)
        return p_ids[-max_prompt_tokens:]

    sep_ids = tok.encode("\n\n")
    notes: list[int] = []
    remaining = budget_notes
    while remaining > 0:
        if not note_pool_ids:
            notes.extend([tok.eot_id] * remaining)
            break
        d_ids = note_pool_ids[rng.randrange(len(note_pool_ids))]
        piece = sep_ids + d_ids
        if len(piece) <= remaining:
            notes.extend(piece)
            remaining -= len(piece)
        else:
            notes.extend(piece[:remaining])
            remaining = 0

    p_ids = prefix_ids + notes + suffix_ids
    if len(p_ids) > max_prompt_tokens:
        p_ids = p_ids[-max_prompt_tokens:]
    elif len(p_ids) < max_prompt_tokens:
        p_ids = [tok.eot_id] * (max_prompt_tokens - len(p_ids)) + p_ids
    return p_ids


def _make_cache_key(payload: dict[str, Any]) -> str:
    h = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return h[:16]


def _build_examples(
    *,
    rows: list[tuple[str, str, str]],
    is_val: bool,
    tok: RWKVWorldTokenizer,
    n: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    max_answer_tokens: int,
    max_seq_len: int,
    min_seq_len: int,
    num_queries: int,
    query_region: str,
    seed: int,
    q_first: bool,
    fill_notes_to_max: bool,
    note_pool_size: int,
    max_note_seq_len: int,
) -> list[tuple[list[int], list[int]]]:
    rng = random.Random(seed)

    pool: list[tuple[str, str, str]] = []
    for pid, seq, ss in rows:
        b = _bucket_key(f"{pid}|{len(seq)}|{seq[:16]}")
        in_val = b >= 8
        if in_val != is_val:
            continue
        if len(seq) < min_seq_len:
            continue
        pool.append((pid, seq, ss))

    if not pool:
        raise RuntimeError("No protein rows after split/filter.")

    note_pool_ids = _build_note_pool(
        pool, tok=tok, rng=rng, pool_size=note_pool_size, max_note_seq_len=max_note_seq_len
    ) if fill_notes_to_max else []

    idxs = list(range(len(pool)))
    rng.shuffle(idxs)

    out: list[tuple[list[int], list[int]]] = []
    for i in idxs:
        _pid, seq0, ss0 = pool[int(i)]
        seq, ss = _crop_pair(seq0, ss0, max_seq_len=max_seq_len, rng=rng)
        if len(seq) < min_seq_len:
            continue
        pos = _pick_positions(len(seq), num_queries=num_queries, query_region=query_region, rng=rng)
        q_idx = [p + 1 for p in pos]
        ans = "".join(ss[p] for p in pos)

        p_ids = _build_prompt_ids(
            tok=tok,
            seq=seq,
            q_idx_1based=q_idx,
            max_prompt_tokens=max_prompt_tokens,
            q_first=q_first,
            fill_notes_to_max=fill_notes_to_max,
            note_pool_ids=note_pool_ids,
            rng=rng,
        )
        if len(p_ids) < min_prompt_tokens:
            continue

        a_ids = tok.encode(ans)
        if len(a_ids) > max_answer_tokens:
            a_ids = a_ids[:max_answer_tokens]
        if not a_ids:
            continue
        out.append((p_ids, a_ids))
        if len(out) >= n:
            break

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} examples (wanted {n}).")
    return out


@torch.no_grad()
def token_acc_from_preds(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    ok = ((pred == tgt) & mask).float().sum().item()
    denom = mask.float().sum().clamp(min=1).item()
    return float(ok / denom)


@torch.no_grad()
def seq_acc_from_preds(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    # exact match per sample on masked region
    eq = (pred == tgt) | (~mask)
    per = eq.all(dim=1).float().mean().item()
    return float(per)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["no_fs", "prompt_fs"], default="no_fs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_data_seed", type=int, default=0)
    ap.add_argument("--val_data_seed", type=int, default=1234)

    ap.add_argument("--ds", type=str, default="lamm-mit/protein_secondary_structure_from_PDB")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--n_train", type=int, default=1600)
    ap.add_argument("--n_val", type=int, default=320)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--min_seq_len", type=int, default=96)
    ap.add_argument("--num_queries", type=int, default=48)
    ap.add_argument("--query_region", choices=["random", "prefix", "suffix"], default="random")
    ap.add_argument("--q_first", action="store_true")
    ap.add_argument("--fill_notes_to_max", action="store_true")
    ap.add_argument("--note_pool_size", type=int, default=2048)
    ap.add_argument("--max_note_seq_len", type=int, default=256)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--min_prompt_tokens", type=int, default=1024)
    ap.add_argument("--max_answer_tokens", type=int, default=128)

    ap.add_argument("--fs_variant", choices=["scalar", "head"], default="scalar")
    ap.add_argument("--alpha_head_lr", type=float, default=None)
    ap.add_argument("--alpha_head_init", type=float, default=None)

    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--time_budget_sec", type=int, default=300)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--val_batches", type=int, default=24)

    ap.add_argument("--model_lr", type=float, default=3e-5)
    ap.add_argument("--alpha_lr", type=float, default=0.0)
    ap.add_argument("--alpha_init", type=float, default=-2.0)
    ap.add_argument("--seed_scale", type=float, default=1.0)
    ap.add_argument("--fs_layer_start", type=int, default=6)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = RWKVWorldTokenizer(args.vocab)
    pad_id = tok.eot_id

    run_root = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S") / "protein_ss_spot_sft" / args.mode
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    os.makedirs(args.cache_dir, exist_ok=True)
    key = _make_cache_key(
        {
            "ds": args.ds,
            "split": args.split,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "max_seq_len": args.max_seq_len,
            "min_seq_len": args.min_seq_len,
            "num_queries": args.num_queries,
            "query_region": args.query_region,
            "q_first": bool(args.q_first),
            "fill_notes_to_max": bool(args.fill_notes_to_max),
            "note_pool_size": args.note_pool_size,
            "max_note_seq_len": args.max_note_seq_len,
            "max_prompt_tokens": args.max_prompt_tokens,
            "min_prompt_tokens": args.min_prompt_tokens,
            "max_answer_tokens": args.max_answer_tokens,
            "train_data_seed": args.train_data_seed,
            "val_data_seed": args.val_data_seed,
        }
    )
    cache_path = Path(args.cache_dir) / f"protein_ss_spot_tok_{key}.pt"

    if cache_path.exists():
        data = torch.load(cache_path)
        train_ex = data["train_ex"]
        val_ex = data["val_ex"]
        print(f"Loaded cache: {cache_path}")
    else:
        print("Loading protein dataset...")
        rows = _load_all_rows(args.ds, args.split)
        train_ex = _build_examples(
            rows=rows,
            is_val=False,
            tok=tok,
            n=int(args.n_train),
            max_prompt_tokens=int(args.max_prompt_tokens),
            min_prompt_tokens=int(args.min_prompt_tokens),
            max_answer_tokens=int(args.max_answer_tokens),
            max_seq_len=int(args.max_seq_len),
            min_seq_len=int(args.min_seq_len),
            num_queries=int(args.num_queries),
            query_region=str(args.query_region),
            seed=int(args.train_data_seed),
            q_first=bool(args.q_first),
            fill_notes_to_max=bool(args.fill_notes_to_max),
            note_pool_size=int(args.note_pool_size),
            max_note_seq_len=int(args.max_note_seq_len),
        )
        val_ex = _build_examples(
            rows=rows,
            is_val=True,
            tok=tok,
            n=int(args.n_val),
            max_prompt_tokens=int(args.max_prompt_tokens),
            min_prompt_tokens=int(args.min_prompt_tokens),
            max_answer_tokens=int(args.max_answer_tokens),
            max_seq_len=int(args.max_seq_len),
            min_seq_len=int(args.min_seq_len),
            num_queries=int(args.num_queries),
            query_region=str(args.query_region),
            seed=int(args.val_data_seed),
            q_first=bool(args.q_first),
            fill_notes_to_max=bool(args.fill_notes_to_max),
            note_pool_size=int(args.note_pool_size),
            max_note_seq_len=int(args.max_note_seq_len),
        )
        torch.save({"train_ex": train_ex, "val_ex": val_ex}, cache_path)
        print(f"Saved cache: {cache_path}")

    train_rng = random.Random(int(args.seed))
    val_rng = random.Random(int(args.val_data_seed) + 999)

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

    def sample_batch(ex: list[tuple[list[int], list[int]]], rng: random.Random) -> tuple[torch.Tensor, list[list[int]]]:
        picks = [ex[rng.randrange(len(ex))] for _ in range(int(args.bsz))]
        ps = [p for p, _ in picks]
        ans = [a for _, a in picks]
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
        prompt_ids, ans_list = sample_batch(train_ex, train_rng)
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
                if t:
                    tt = torch.tensor(t, device=device, dtype=torch.long)
                    tgt[i, : tt.numel()] = tt
            loss_rest = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-100)

        loss = loss0 + loss_rest
        loss.backward()
        opt.step()

        if (step % int(args.eval_every)) == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                val_accs = []
                val_exacts = []
                for _ in range(int(args.val_batches)):
                    vp, vans_list = sample_batch(val_ex, val_rng)
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
                            if t:
                                tt = torch.tensor(t, device=device, dtype=torch.long)
                                v_rest_tgt[i, : tt.numel()] = tt
                        vloss_rest = F.cross_entropy(
                            v_logits.view(-1, v_logits.size(-1)),
                            v_rest_tgt.view(-1),
                            ignore_index=-100,
                        )
                        vpred[:, 1 : 1 + v_in_len] = v_logits.argmax(dim=-1)

                    vloss = vloss0 + vloss_rest
                    vacc = token_acc_from_preds(vpred, vtgt, vmask)
                    vexact = seq_acc_from_preds(vpred, vtgt, vmask)
                    val_losses.append(float(vloss))
                    val_accs.append(float(vacc))
                    val_exacts.append(float(vexact))

                rec = {
                    "t": round(time.time() - t0, 2),
                    "step": step,
                    "train_loss": float(loss),
                    "val_loss": sum(val_losses) / len(val_losses),
                    "val_tok_acc": sum(val_accs) / len(val_accs),
                    "val_seq_acc": sum(val_exacts) / len(val_exacts),
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
