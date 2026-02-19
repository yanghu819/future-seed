#!/usr/bin/env python3

"""HotpotQA SFT probe for Future-Seed post-training (real long-context QA).

We want a *real* scenario where the question comes after a long context.
HotpotQA provides multi-paragraph Wikipedia context + short answer.

We compare:
  - no_fs: normal causal prompt/prefill
  - prompt_fs: Future-Seed enabled for prompt/prefill only; decoding remains causal

Metrics are teacher-forced (loss + token acc) on the answer span.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

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


def _clean(s: str) -> str:
    return str(s).replace("\n", " ").replace("\r", " ").strip()


def _build_context(ex: dict) -> str:
    # hotpot_qa: ex["context"] is usually a dict:
    #   {"title": [...], "sentences": [[...], ...]}
    # Older variants may store a list of [title, [sentences...]].
    parts = []
    ctx = ex.get("context", None)
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sents = ctx.get("sentences", [])
        for title, sent_list in zip(titles, sents):
            title = _clean(title)
            if title:
                parts.append(title + ":")
            if isinstance(sent_list, list):
                parts.append(" ".join(_clean(x) for x in sent_list if _clean(x)))
            else:
                parts.append(_clean(str(sent_list)))
    elif isinstance(ctx, list):
        for item in ctx:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            title, sent_list = item[0], item[1]
            title = _clean(title)
            if title:
                parts.append(title + ":")
            if isinstance(sent_list, list):
                parts.append(" ".join(_clean(x) for x in sent_list if _clean(x)))
            else:
                parts.append(_clean(str(sent_list)))
    ctx = "\n\n".join([p for p in parts if p])
    return ctx


def build_examples(
    *,
    ds: str,
    cfg: str,
    split: str,
    tok: RWKVWorldTokenizer,
    n: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    max_answer_tokens: int,
    seed: int,
    q_first: bool = False,
) -> List[Tuple[List[int], List[int]]]:
    ds_obj = load_dataset(ds, cfg, split=split)
    rng = random.Random(seed)

    idxs = list(range(len(ds_obj)))
    rng.shuffle(idxs)

    out: List[Tuple[List[int], List[int]]] = []
    for i in idxs:
        ex = ds_obj[int(i)]
        q = _clean(ex.get("question", ""))
        a = _clean(ex.get("answer", ""))
        if not q or not a:
            continue
        ctx = _build_context(ex)
        if not ctx:
            continue

        if q_first:
            prompt = f"Question:\n{q}\n\nContext:\n{ctx}\n\nAnswer:"
        else:
            prompt = f"Context:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:"

        p_ids = tok.encode(prompt)
        if len(p_ids) > max_prompt_tokens:
            # Keep suffix (question + Answer:) intact by trimming from the left.
            p_ids = p_ids[-max_prompt_tokens:]
        if len(p_ids) < min_prompt_tokens:
            continue

        a_ids = tok.encode(" " + a)
        if len(a_ids) > max_answer_tokens:
            a_ids = a_ids[:max_answer_tokens]
        if not a_ids:
            continue

        out.append((p_ids, a_ids))
        if len(out) >= n:
            break

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} examples (wanted {n}). Try lowering --min_prompt_tokens.")
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

    ap.add_argument("--ds", type=str, default="hotpot_qa")
    ap.add_argument("--ds_cfg", type=str, default="distractor")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="validation")
    ap.add_argument("--q_first", action="store_true", help="Question before context (causal-friendly control).")

    ap.add_argument("--fs_variant", choices=["scalar", "head"], default="scalar")
    ap.add_argument("--alpha_head_lr", type=float, default=None)
    ap.add_argument("--alpha_head_init", type=float, default=None)

    ap.add_argument("--n_train", type=int, default=2048)
    ap.add_argument("--n_val", type=int, default=512)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--min_prompt_tokens", type=int, default=1024)
    ap.add_argument("--max_answer_tokens", type=int, default=24)

    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--time_budget_sec", type=int, default=240)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--val_batches", type=int, default=16)

    ap.add_argument("--model_lr", type=float, default=3e-5)
    ap.add_argument("--alpha_lr", type=float, default=0.0)
    ap.add_argument("--alpha_init", type=float, default=-2.0)
    ap.add_argument("--fs_in_gate", action="store_true", help="Enable input-adaptive FS gate (per sample).")
    ap.add_argument("--fs_w_lr", type=float, default=None, help="LR for fs_in_w (default: alpha_lr).")
    ap.add_argument("--fs_b_lr", type=float, default=None, help="LR for fs_in_b (default: alpha_lr).")
    ap.add_argument("--train_gate_only", action="store_true", help="Freeze model weights; train gate params only.")
    ap.add_argument("--seed_scale", type=float, default=1.0)
    ap.add_argument("--fs_layer_start", type=int, default=1)
    ap.add_argument("--fs_norm", action="store_true")
    ap.add_argument("--fs_clip", type=float, default=0.0)
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
    if args.fs_w_lr is None:
        args.fs_w_lr = float(args.alpha_lr)
    if args.fs_b_lr is None:
        args.fs_b_lr = float(args.alpha_lr)

    # Keep HF caches off the system disk.
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_transformers")
    # AutoDL/China networks often can't reach huggingface.co reliably.
    # Users can override externally; this default keeps experiments unblocked.
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
        "q_first": bool(args.q_first),
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
    cache_path = cache_dir / f"hotpot_longctx_tok_{cache_key}.pt"

    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu")
        train_ex = data["train_ex"]
        val_ex = data["val_ex"]
        print(f"Loaded cache: {cache_path}")
    else:
        print("Loading data...")
        train_ex = build_examples(
            ds=args.ds,
            cfg=args.ds_cfg,
            split=args.train_split,
            tok=tok,
            n=int(args.n_train),
            max_prompt_tokens=int(args.max_prompt_tokens),
            min_prompt_tokens=int(args.min_prompt_tokens),
            max_answer_tokens=int(args.max_answer_tokens),
            seed=int(args.train_data_seed),
            q_first=bool(args.q_first),
        )
        val_ex = build_examples(
            ds=args.ds,
            cfg=args.ds_cfg,
            split=args.val_split,
            tok=tok,
            n=int(args.n_val),
            max_prompt_tokens=int(args.max_prompt_tokens),
            min_prompt_tokens=int(args.min_prompt_tokens),
            max_answer_tokens=int(args.max_answer_tokens),
            seed=int(args.val_data_seed),
            q_first=bool(args.q_first),
        )
        torch.save({"train_ex": train_ex, "val_ex": val_ex, "meta": cache_meta}, cache_path)
        print(f"Saved cache: {cache_path}")

    run_root = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S") / "hotpot_longctx_sft" / args.mode
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

    fs_in_w = None
    fs_in_b = None
    if args.fs_in_gate:
        fs_in_w = torch.nn.Parameter(torch.zeros((model.cfg.num_layers, model.cfg.hidden_size), device=device))
        fs_in_b = torch.nn.Parameter(torch.zeros((model.cfg.num_layers,), device=device))

    if args.train_gate_only:
        for p in model.parameters():
            p.requires_grad_(False)
        param_groups = [
            {"params": [alpha], "lr": float(args.alpha_lr), "weight_decay": 0.0},
        ]
    else:
        param_groups = [
            {"params": model.parameters(), "lr": float(args.model_lr), "weight_decay": 0.01},
            {"params": [alpha], "lr": float(args.alpha_lr), "weight_decay": 0.0},
        ]
    if alpha_head is not None:
        param_groups.append({"params": [alpha_head], "lr": float(args.alpha_head_lr), "weight_decay": 0.0})
    if fs_in_w is not None:
        param_groups.append({"params": [fs_in_w], "lr": float(args.fs_w_lr), "weight_decay": 0.0})
    if fs_in_b is not None:
        param_groups.append({"params": [fs_in_b], "lr": float(args.fs_b_lr), "weight_decay": 0.0})

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
            fs_in_w=fs_in_w,
            fs_in_b=fs_in_b,
            seed_scale=float(args.seed_scale),
            fs_layer_start=int(args.fs_layer_start),
            fs_norm=bool(args.fs_norm),
            fs_clip=float(args.fs_clip),
            fs_detach=bool(args.fs_detach),
            return_states=True,
        )
        assert prompt_states is not None

        # First answer token
        a0 = torch.tensor([a[0] for a in ans_list], device=device, dtype=torch.long)
        logits0 = model.project(prompt_hidden[:, -1, :])
        loss0 = F.cross_entropy(logits0, a0)

        # Remaining answer tokens
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
                val_losses = []
                val_accs = []
                for _ in range(int(args.val_batches)):
                    vp, vans_list = sample_batch(val_ex, rng=val_rng)
                    v_hidden, v_states = model(
                        vp,
                        future_seed=(args.mode == "prompt_fs"),
                        fs_alpha=alpha,
                        fs_alpha_head=alpha_head,
                        fs_in_w=fs_in_w,
                        fs_in_b=fs_in_b,
                        seed_scale=float(args.seed_scale),
                        fs_layer_start=int(args.fs_layer_start),
                        fs_norm=bool(args.fs_norm),
                        fs_clip=float(args.fs_clip),
                        fs_detach=bool(args.fs_detach),
                        return_states=True,
                    )
                    assert v_states is not None

                    # Build targets / mask
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

                        # Build tgt for rest
                        v_rest_tgt = torch.full((int(args.bsz), v_in_len), -100, device=device, dtype=torch.long)
                        v_rest_mask = torch.zeros((int(args.bsz), v_in_len), device=device, dtype=torch.bool)
                        for i, t in enumerate(v_ans_tgt):
                            if not t:
                                continue
                            tt = torch.tensor(t, device=device, dtype=torch.long)
                            v_rest_tgt[i, : tt.numel()] = tt
                            v_rest_mask[i, : tt.numel()] = True

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
                    "alpha_head_mean": (float(torch.sigmoid(alpha_head[1:]).mean()) if alpha_head is not None else None),
                }
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            model.train()

        step += 1

    print(str(run_root))


if __name__ == "__main__":
    main()
