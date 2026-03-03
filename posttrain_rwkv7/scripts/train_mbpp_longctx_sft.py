#!/usr/bin/env python3

"""MBPP SFT probe for Future-Seed post-training (real code task).

We compare:
  - no_fs: normal causal prompt/prefill
  - prompt_fs: Future-Seed enabled for prompt/prefill only; decoding remains causal

Task format:
  Generate solution code from MBPP problem + tests.

Context-order control:
  - q_first=False (default): problem/tests appear early, then long distractor notes, then answer trigger.
  - q_first=True: distractor notes first, problem/tests close to answer trigger.
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
    return str(s).replace("\r", " ").strip()


def _pick_first(*vals: Any) -> str:
    for v in vals:
        if v is None:
            continue
        s = _clean(v)
        if s:
            return s
    return ""


def _join_tests(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return _clean(v)
    if isinstance(v, Iterable):
        parts = []
        for x in v:
            sx = _clean(x)
            if sx:
                parts.append(sx)
        return "\n".join(parts)
    return _clean(v)


def _load_split(ds: str, cfg: str, split: str):
    if cfg:
        return load_dataset(ds, cfg, split=split)
    return load_dataset(ds, split=split)


def _load_any_split(ds: str, cfg: str, candidates: List[str]):
    last_e: Optional[Exception] = None
    for s in candidates:
        try:
            return _load_split(ds, cfg, s), s
        except Exception as e:
            last_e = e
            continue
    raise RuntimeError(f"Cannot load any split from {candidates}. last_error={last_e!r}")


def _build_distractor_pool(
    ds_obj,
    *,
    tok: RWKVWorldTokenizer,
    rng: random.Random,
    pool_size: int,
) -> List[List[int]]:
    idxs = list(range(len(ds_obj)))
    rng.shuffle(idxs)
    out: List[List[int]] = []
    for i in idxs:
        ex = ds_obj[int(i)]
        text = _pick_first(ex.get("text"), ex.get("prompt"), ex.get("description"))
        tests = _join_tests(ex.get("test_list", ex.get("tests")))
        if not text:
            continue
        blob = f"Problem: {text}\nTests:\n{tests}" if tests else f"Problem: {text}"
        ids = tok.encode(blob)
        if ids:
            out.append(ids)
        if len(out) >= pool_size:
            break
    return out


def _build_prompt_ids(
    *,
    tok: RWKVWorldTokenizer,
    problem: str,
    tests: str,
    max_prompt_tokens: int,
    q_first: bool,
    fill_notes_to_max: bool,
    note_pool_ids: List[List[int]],
    rng: random.Random,
) -> List[int]:
    tests_block = f"\n\nTests:\n{tests}" if tests else ""

    if not fill_notes_to_max:
        if q_first:
            prompt = f"Notes:\n\nProblem:\n{problem}{tests_block}\n\nPython solution:\n"
        else:
            prompt = f"Problem:\n{problem}{tests_block}\n\nNotes:\n\nPython solution:\n"
        p_ids = tok.encode(prompt)
        if len(p_ids) > max_prompt_tokens:
            p_ids = p_ids[-max_prompt_tokens:]
        return p_ids

    if q_first:
        prefix_ids = tok.encode("Notes:\n")
        suffix_ids = tok.encode(f"\n\nProblem:\n{problem}{tests_block}\n\nPython solution:\n")
    else:
        prefix_ids = tok.encode(f"Problem:\n{problem}{tests_block}\n\nNotes:\n")
        suffix_ids = tok.encode("\n\nPython solution:\n")

    budget_notes = max_prompt_tokens - len(prefix_ids) - len(suffix_ids)
    if budget_notes <= 0:
        prompt = f"Problem:\n{problem}{tests_block}\n\nPython solution:\n"
        p_ids = tok.encode(prompt)
        return p_ids[-max_prompt_tokens:]

    sep_ids = tok.encode("\n\n")
    notes: List[int] = []
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
    seed: int,
    q_first: bool = False,
    fill_notes_to_max: bool = True,
    note_pool_size: int = 2048,
) -> List[Tuple[List[int], List[int]]]:
    ds_obj = _load_split(ds, ds_cfg, split)
    rng = random.Random(seed)

    note_pool_ids = _build_distractor_pool(ds_obj, tok=tok, rng=rng, pool_size=note_pool_size) if fill_notes_to_max else []

    idxs = list(range(len(ds_obj)))
    rng.shuffle(idxs)

    out: List[Tuple[List[int], List[int]]] = []
    for i in idxs:
        ex = ds_obj[int(i)]
        problem = _pick_first(ex.get("text"), ex.get("prompt"), ex.get("description"))
        code = _pick_first(ex.get("code"), ex.get("canonical_solution"), ex.get("completion"))
        tests = _join_tests(ex.get("test_list", ex.get("tests")))

        if not problem or not code:
            continue

        p_ids = _build_prompt_ids(
            tok=tok,
            problem=problem,
            tests=tests,
            max_prompt_tokens=max_prompt_tokens,
            q_first=q_first,
            fill_notes_to_max=fill_notes_to_max,
            note_pool_ids=note_pool_ids,
            rng=rng,
        )
        if len(p_ids) < min_prompt_tokens:
            continue

        a_ids = tok.encode("\n" + code)
        if len(a_ids) > max_answer_tokens:
            a_ids = a_ids[:max_answer_tokens]
        if not a_ids:
            continue

        out.append((p_ids, a_ids))
        if len(out) >= n:
            break

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} examples (wanted {n}). Try lowering --min_prompt_tokens or --n_*.")
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

    ap.add_argument("--ds", type=str, default="mbpp")
    ap.add_argument("--ds_cfg", type=str, default="")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="test")
    ap.add_argument("--q_first", action="store_true", help="Place problem/tests near answer (control).")
    ap.add_argument("--fill_notes_to_max", action="store_true")
    ap.add_argument("--note_pool_size", type=int, default=2048)

    ap.add_argument("--fs_variant", choices=["scalar", "head"], default="scalar")
    ap.add_argument("--alpha_head_lr", type=float, default=None)
    ap.add_argument("--alpha_head_init", type=float, default=None)

    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--max_prompt_tokens", type=int, default=4096)
    ap.add_argument("--min_prompt_tokens", type=int, default=1536)
    ap.add_argument("--max_answer_tokens", type=int, default=160)

    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--time_budget_sec", type=int, default=480)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--val_batches", type=int, default=16)

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
    ap.add_argument("--run_dir", type=str, default="runs")
    ap.add_argument("--cache_dir", type=str, default="cache")
    args = ap.parse_args()

    if args.alpha_head_lr is None:
        args.alpha_head_lr = float(args.alpha_lr)
    if args.alpha_head_init is None:
        args.alpha_head_init = float(args.alpha_init)

    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_transformers")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = RWKVWorldTokenizer(args.vocab)
    pad_id = tok.eot_id

    # Resolve val split fallback if needed.
    try:
        _ = _load_split(args.ds, args.ds_cfg, args.val_split)
        val_split = args.val_split
    except Exception:
        _, val_split = _load_any_split(args.ds, args.ds_cfg, ["validation", "test", "train"])

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_meta = {
        "ds": args.ds,
        "ds_cfg": args.ds_cfg,
        "train_split": args.train_split,
        "val_split": val_split,
        "q_first": bool(args.q_first),
        "fill_notes_to_max": bool(args.fill_notes_to_max),
        "note_pool_size": int(args.note_pool_size),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "min_prompt_tokens": int(args.min_prompt_tokens),
        "max_answer_tokens": int(args.max_answer_tokens),
        "train_data_seed": int(args.train_data_seed),
        "val_data_seed": int(args.val_data_seed),
        "vocab": str(args.vocab),
    }
    cache_key = hashlib.md5(json.dumps(cache_meta, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / f"mbpp_longctx_tok_{cache_key}.pt"

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
            seed=int(args.train_data_seed),
            q_first=bool(args.q_first),
            fill_notes_to_max=bool(args.fill_notes_to_max),
            note_pool_size=int(args.note_pool_size),
        )
        val_ex = build_examples(
            ds=args.ds,
            ds_cfg=args.ds_cfg,
            split=val_split,
            tok=tok,
            n=int(args.n_val),
            max_prompt_tokens=int(args.max_prompt_tokens),
            min_prompt_tokens=int(args.min_prompt_tokens),
            max_answer_tokens=int(args.max_answer_tokens),
            seed=int(args.val_data_seed),
            q_first=bool(args.q_first),
            fill_notes_to_max=bool(args.fill_notes_to_max),
            note_pool_size=int(args.note_pool_size),
        )
        torch.save({"train_ex": train_ex, "val_ex": val_ex, "meta": cache_meta}, cache_path)
        print(f"Saved cache: {cache_path}")

    run_root = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S") / "mbpp_longctx_sft" / args.mode
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
        ps = []
        ans = []
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
                            if t:
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

