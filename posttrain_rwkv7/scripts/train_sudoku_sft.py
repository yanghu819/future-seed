#!/usr/bin/env python3

"""Synthetic Sudoku SFT probe for Future-Seed post-training.

We compare:
  - no_fs: normal causal prompt/prefill
  - prompt_fs: Future-Seed enabled for prompt/prefill only; decoding remains causal

Task:
  Given a Sudoku puzzle string with blanks as '0', output the full solved board string.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from rwkv_tokenizer import RWKVWorldTokenizer
from rwkv7_g1d import RWKV7G1DLM


_ALPH = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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


def _shuffle(rng: random.Random, xs: List[int]) -> List[int]:
    ys = xs[:]
    rng.shuffle(ys)
    return ys


def _solved_board(base: int, rng: random.Random) -> List[int]:
    side = base * base
    if side > len(_ALPH):
        raise ValueError("side too large")

    def pattern(r: int, c: int) -> int:
        return (base * (r % base) + r // base + c) % side

    rbase = list(range(base))
    rows = [g * base + r for g in _shuffle(rng, rbase) for r in _shuffle(rng, rbase)]
    cols = [g * base + c for g in _shuffle(rng, rbase) for c in _shuffle(rng, rbase)]
    nums = _shuffle(rng, list(range(1, side + 1)))

    board = [nums[pattern(r, c)] for r in rows for c in cols]
    return board


def _render_board(flat: List[int], side: int) -> str:
    return "".join(_ALPH[v - 1] if v > 0 else "0" for v in flat[: side * side])


def sample_sudoku(
    rng: random.Random,
    *,
    base: int,
    mask_count: int,
    mask_region: str,
) -> Tuple[str, str]:
    side = base * base
    n = side * side

    solved = _solved_board(base, rng)
    puzzle = solved[:]

    k = max(1, min(mask_count, n - 1))
    if mask_region == "prefix":
        idxs = list(range(k))
    elif mask_region == "suffix":
        idxs = list(range(n - k, n))
    elif mask_region == "random":
        idxs = rng.sample(list(range(n)), k=k)
    else:
        raise ValueError(mask_region)

    for i in idxs:
        puzzle[i] = 0

    puzzle_s = _render_board(puzzle, side)
    solved_s = _render_board(solved, side)

    prompt = f"SUDOKU{side}|P={puzzle_s}|A="
    answer = solved_s
    return prompt, answer


@torch.no_grad()
def token_acc_from_preds(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    ok = ((pred == tgt) & mask).float().sum().item()
    denom = mask.float().sum().clamp(min=1).item()
    return float(ok / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["no_fs", "prompt_fs"], default="no_fs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=1234)

    ap.add_argument("--base", type=int, default=3, help="2 for 4x4, 3 for 9x9.")
    ap.add_argument("--mask_count", type=int, default=40)
    ap.add_argument("--mask_region", choices=["prefix", "suffix", "random"], default="prefix")

    ap.add_argument("--fs_variant", choices=["scalar", "head"], default="scalar")
    ap.add_argument("--alpha_head_lr", type=float, default=None)
    ap.add_argument("--alpha_head_init", type=float, default=None)

    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--time_budget_sec", type=int, default=240)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=50)
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
    args = ap.parse_args()

    if args.alpha_head_lr is None:
        args.alpha_head_lr = float(args.alpha_lr)
    if args.alpha_head_init is None:
        args.alpha_head_init = float(args.alpha_init)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = RWKVWorldTokenizer(args.vocab)
    pad_id = tok.eot_id

    run_root = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S") / "sudoku_sft" / args.mode
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_rng = random.Random(int(args.seed))
    val_rng = random.Random(int(args.val_seed))

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

    def sample_batch(rng: random.Random) -> Tuple[torch.Tensor, List[List[int]]]:
        ps = []
        ans = []
        for _ in range(int(args.bsz)):
            prompt, answer = sample_sudoku(
                rng,
                base=int(args.base),
                mask_count=int(args.mask_count),
                mask_region=str(args.mask_region),
            )
            p_ids = tok.encode(prompt)
            a_ids = tok.encode(answer)
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
        prompt_ids, ans_list = sample_batch(train_rng)
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
                    vp, vans_list = sample_batch(val_rng)
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

