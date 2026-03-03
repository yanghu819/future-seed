#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
import random
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from rwkv_tokenizer import RWKVWorldTokenizer
from rwkv7_g1d import RWKV7G1DLM
from tasks import (
    MaskedSample,
    sample_copy_a2q,
    sample_kvsort,
    sample_middlematch,
    sample_nameindex,
    sample_retrieval,
    sample_retrieval_a2q,
)


def build_byte_encoder(tok: RWKVWorldTokenizer) -> dict[int, int]:
    byte2id = {}
    for tid, b in tok.idx2token.items():
        if len(b) == 1:
            bb = b[0]
            if bb in byte2id:
                continue
            byte2id[bb] = tid
    # sanity for ascii
    for ch in b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=;:,|#QASORT_":
        if ch not in byte2id:
            raise RuntimeError(f"missing single-byte token for {ch!r}")
    return byte2id


def encode_ascii(byte2id: dict[int, int], s: str) -> List[int]:
    b = s.encode("utf-8")
    return [byte2id[x] for x in b]


def pad_to(ids: List[int], length: int, pad_id: int) -> List[int]:
    if len(ids) > length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def pad_tensor_to_multiple(x: torch.Tensor, *, multiple: int, pad_id: int) -> torch.Tensor:
    """Right-pad [B,T] to T % multiple == 0. Safe for causal kernels."""
    T = int(x.shape[1])
    need = (-T) % int(multiple)
    if need == 0:
        return x
    pad = x.new_full((x.shape[0], need), int(pad_id))
    return torch.cat([x, pad], dim=1)


def align_prompt(prompt: str, *, pad_ch: str = "#", multiple: int = 16) -> str:
    # Pad the prompt to `multiple` while keeping the suffix intact (e.g. ";Q=..;A=" stays adjacent).
    # This is required for the prompt-only pass in prompt_fs (CUDA kernel expects T % 16 == 0).
    if not prompt.endswith("A="):
        raise RuntimeError("prompt must end with 'A='")
    need = (-len(prompt)) % multiple
    if need == 0:
        return prompt
    return (pad_ch * need) + prompt


def with_fs_markers(prompt: str) -> str:
    """Add explicit region markers without changing semantics.

    This is a lightweight way to make "FS-on" prompts visually distinct.
    Note: markers do NOT change the Future-Seed mechanism itself.
    """
    if not prompt.endswith("A="):
        raise RuntimeError("prompt must end with 'A='")
    core = prompt[:-2]
    return f"|FS_BEGIN|{core}|FS_END|A="


def make_batch_qa(
    *,
    rng: random.Random,
    task: str,
    mode: str,
    fs_markers: bool,
    byte2id: dict[int, int],
    bsz: int,
    retr_k: int,
    retr_vlen: int,
    retr_query_last: bool,
    retr_q_first: bool,
    kv_n: int,
    kv_vlen: int,
    ni_n: int,
    ni_name_width: int,
    ni_idx_width: int,
    ni_q_first: bool,
    mm_n: int,
    mm_name_width: int,
    mm_q_first: bool,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return (prompt_ids, ans_ids, prompt_len, ans_len) for standard Q->A."""
    ps: List[List[int]] = []
    ans: List[List[int]] = []

    prompt_len = None
    ans_len = None

    for _ in range(bsz):
        if task == "retrieval":
            s = sample_retrieval(
                rng,
                k=retr_k,
                vlen=retr_vlen,
                query_last=retr_query_last,
                q_first=retr_q_first,
            )
        elif task == "nameindex":
            s = sample_nameindex(
                rng,
                n=ni_n,
                name_width=ni_name_width,
                idx_width=ni_idx_width,
                q_first=ni_q_first,
            )
        elif task == "middlematch":
            s = sample_middlematch(
                rng,
                n=mm_n,
                name_width=mm_name_width,
                q_first=mm_q_first,
            )
        elif task == "kvsort":
            s = sample_kvsort(rng, n=kv_n, vlen=kv_vlen)
        else:
            raise ValueError(task)

        prompt = s.prompt
        answer = s.answer

        if mode == "jrt2":
            # Repeat context (everything before trailing "A="), then append single "A=".
            if not prompt.endswith("A="):
                raise RuntimeError("prompt must end with A=")
            ctx = prompt[:-2]
            prompt = ctx + ctx + "A="

        if fs_markers and mode == "prompt_fs":
            prompt = with_fs_markers(prompt)

        prompt = align_prompt(prompt)  # enforce prompt_len % 16 == 0

        p_ids = encode_ascii(byte2id, prompt)
        a_ids = encode_ascii(byte2id, answer)

        ps.append(p_ids)
        ans.append(a_ids)

        if prompt_len is None:
            prompt_len = len(p_ids)
        if ans_len is None:
            ans_len = len(a_ids)
        if len(p_ids) != prompt_len:
            raise RuntimeError("prompt length drift; use fixed-format prompts")
        if len(a_ids) != ans_len:
            raise RuntimeError("answer length drift; use fixed-format answers")

    return (
        torch.tensor(ps, dtype=torch.long),
        torch.tensor(ans, dtype=torch.long),
        int(prompt_len),
        int(ans_len),
    )


def make_batch(
    *,
    rng: random.Random,
    task: str,
    mode: str,
    fs_markers: bool,
    byte2id: dict[int, int],
    pad_id: int,
    seq_len: int,
    bsz: int,
    retr_k: int,
    retr_vlen: int,
    retr_query_last: bool,
    retr_q_first: bool,
    kv_n: int,
    kv_vlen: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Return (full_ids, labels, prompt_ids, loss_start, loss_len).

    - full_ids: [B,T]
    - labels : [B,T] with -100 ignore except answer region
    - prompt_ids: [B,prompt_len]
    loss_start and loss_len are in tokens (bytes).
    """

    xs = []
    ys = []
    ps = []

    loss_start = None
    loss_len = None

    for _ in range(bsz):
        if task in ("retrieval_a2q", "copy_a2q"):
            if mode == "jrt2":
                raise RuntimeError("jrt2 not supported for retrieval_a2q")
            if task == "retrieval_a2q":
                s = sample_retrieval_a2q(rng, k=retr_k, vlen=retr_vlen)
            else:
                s = sample_copy_a2q(rng)

            full = s.text
            if len(full) > seq_len:
                raise RuntimeError(f"sequence too long: {len(full)} > {seq_len}")
            full = full + ("#" * (seq_len - len(full)))
            full_ids = encode_ascii(byte2id, full)
            assert len(full_ids) == seq_len

            labels = [-100] * seq_len
            tgt_ids = encode_ascii(byte2id, s.target)
            for j, tid in enumerate(tgt_ids):
                labels[s.mask_start + j] = tid

            xs.append(full_ids)
            ys.append(labels)

            # For prompt-fs pass A, we need the whole sequence (includes suffix info).
            ps.append(full_ids)

            if loss_start is None:
                loss_start = int(s.mask_start)
            if loss_len is None:
                loss_len = int(s.mask_len)
            if int(s.mask_start) != loss_start:
                raise RuntimeError("mask start drift; use fixed-format samples")
            if int(s.mask_len) != loss_len:
                raise RuntimeError("mask len drift; use fixed-format samples")
        else:
            if task == "retrieval":
                s = sample_retrieval(
                    rng,
                    k=retr_k,
                    vlen=retr_vlen,
                    query_last=retr_query_last,
                    q_first=retr_q_first,
                )
            elif task == "kvsort":
                s = sample_kvsort(rng, n=kv_n, vlen=kv_vlen)
            else:
                raise ValueError(task)

            prompt = s.prompt
            if fs_markers and mode == "prompt_fs":
                prompt = with_fs_markers(prompt)
            prompt = align_prompt(prompt)
            answer = s.answer

            if mode == "jrt2":
                full = prompt + prompt + answer
                p_len = len(prompt) * 2
            else:
                full = prompt + answer
                p_len = len(prompt)

            # pad with constant # to fixed seq_len
            if len(full) > seq_len:
                raise RuntimeError(f"sequence too long: {len(full)} > {seq_len}")
            full = full + ("#" * (seq_len - len(full)))

            full_ids = encode_ascii(byte2id, full)
            assert len(full_ids) == seq_len

            labels = [-100] * seq_len
            a_ids = encode_ascii(byte2id, answer)
            for j, tid in enumerate(a_ids):
                labels[p_len + j] = tid

            xs.append(full_ids)
            ys.append(labels)

            # prompt ids for prompt-fs pass A (single prompt, not jrt2)
            prompt_ids = encode_ascii(byte2id, prompt)
            ps.append(prompt_ids)

            if loss_start is None:
                loss_start = int(p_len)
            if loss_len is None:
                loss_len = int(len(a_ids))
            if int(p_len) != loss_start:
                raise RuntimeError("prompt length drift; use fixed-format prompts")
            if int(len(a_ids)) != loss_len:
                raise RuntimeError("answer length drift; use fixed-format answers")

    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)

    p = torch.tensor(ps, dtype=torch.long)
    return x, y, p, int(loss_start), int(loss_len)


@torch.no_grad()
def eval_exact_match(logits_ans: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits_ans.argmax(dim=-1)
    ok = (pred == targets).all(dim=1).float().mean().item()
    return float(ok)


@torch.no_grad()
def eval_token_acc(logits_ans: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits_ans.argmax(dim=-1)
    ok = (pred == targets).float().mean().item()
    return float(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        choices=["retrieval", "nameindex", "middlematch", "retrieval_a2q", "copy_a2q", "kvsort"],
        default="retrieval",
    )
    ap.add_argument("--mode", choices=["no_fs", "prompt_fs", "jrt2"], default="no_fs")
    ap.add_argument("--time_budget_sec", type=int, default=1800)
    ap.add_argument("--max_steps", type=int, default=0, help="Optional hard stop by steps (0 = ignore).")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--val_batches", type=int, default=4, help="Number of validation batches per eval.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--model_lr", type=float, default=1e-4)
    ap.add_argument("--alpha_lr", type=float, default=5e-2)
    ap.add_argument("--alpha_init", type=float, default=0.0)
    ap.add_argument("--fs_variant", choices=["scalar", "head"], default="scalar")
    ap.add_argument("--alpha_head_lr", type=float, default=5e-2)
    ap.add_argument("--alpha_head_init", type=float, default=0.0)
    ap.add_argument(
        "--train_fs_only",
        action="store_true",
        help="Freeze base model weights; only train FS gate params (alpha/alpha_head).",
    )
    ap.add_argument("--fs_markers", action="store_true", help="Add |FS_BEGIN| / |FS_END| markers to FS-on prompts.")
    ap.add_argument("--seed_scale", type=float, default=1.0)

    # fixed-format task params (chosen to make prompt_len % 16 == 0)
    ap.add_argument("--retr_k", type=int, default=10)
    ap.add_argument("--retr_vlen", type=int, default=6)
    ap.add_argument("--retr_query_last", action="store_true", help="Place queried key-value pair last in doc (easy sanity-check).")
    ap.add_argument("--retr_q_first", action="store_true", help="Put question before doc (causal-friendly sanity-check).")
    ap.add_argument("--kv_n", type=int, default=7)
    ap.add_argument("--kv_vlen", type=int, default=6)
    ap.add_argument("--ni_n", type=int, default=64)
    ap.add_argument("--ni_name_width", type=int, default=2)
    ap.add_argument("--ni_idx_width", type=int, default=2)
    ap.add_argument("--ni_q_first", action="store_true", help="NameIndex: put question before list (control).")
    ap.add_argument("--mm_n", type=int, default=96)
    ap.add_argument("--mm_name_width", type=int, default=2)
    ap.add_argument("--mm_q_first", action="store_true", help="MiddleMatch: put question before list (control).")

    ap.add_argument("--weights", type=str, default="assets/weights/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
    ap.add_argument("--vocab", type=str, default="assets/tokenizer/rwkv_vocab_v20230424.txt")
    ap.add_argument("--cuda_src", type=str, default="cuda/rwkv_cuda_wind")
    ap.add_argument("--run_dir", type=str, default="runs")
    args = ap.parse_args()

    assert args.seq_len % 16 == 0

    rng = random.Random(args.seed)

    run_root = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S") / args.task / args.mode
    run_root.mkdir(parents=True, exist_ok=True)

    with open(run_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    tok = RWKVWorldTokenizer(args.vocab)
    byte2id = build_byte_encoder(tok)
    pad_id = byte2id[ord("#")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RWKV7G1DLM.from_pth(args.weights, cuda_src_dir=args.cuda_src, device=device)
    model.train()
    if args.train_fs_only:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

    # FS-only params: alpha per layer (+ optional per-head alpha_head)
    alpha = torch.nn.Parameter(torch.full((model.cfg.num_layers,), float(args.alpha_init), device=device))
    alpha_head: Optional[torch.nn.Parameter] = None
    if args.fs_variant == "head":
        alpha_head = torch.nn.Parameter(
            torch.full((model.cfg.num_layers, model.cfg.num_heads), float(args.alpha_head_init), device=device)
        )

    opt_groups = []
    if not args.train_fs_only:
        opt_groups.append({"params": model.parameters(), "lr": args.model_lr, "weight_decay": 0.01})
    opt_groups.append({"params": [alpha], "lr": args.alpha_lr, "weight_decay": 0.0})
    if alpha_head is not None:
        opt_groups.append({"params": [alpha_head], "lr": args.alpha_head_lr, "weight_decay": 0.0})

    opt = torch.optim.AdamW(opt_groups)

    metrics_path = run_root / "metrics.jsonl"

    t0 = time.time()
    step = 0

    # small fixed val set
    val_rng = random.Random(1234)

    while True:
        now = time.time()
        if now - t0 > args.time_budget_sec:
            break
        if args.max_steps and step >= args.max_steps:
            break

        opt.zero_grad(set_to_none=True)
        if args.task in ("retrieval", "nameindex", "middlematch", "kvsort"):
            # Standard Q->A post-training: future-seed can be enabled on the prompt/prefill only.
            prompt_ids, ans_ids, prompt_len, ans_len = make_batch_qa(
                rng=rng,
                task=args.task,
                mode=args.mode,
                fs_markers=args.fs_markers,
                byte2id=byte2id,
                bsz=args.bsz,
                retr_k=args.retr_k,
                retr_vlen=args.retr_vlen,
                retr_query_last=args.retr_query_last,
                retr_q_first=args.retr_q_first,
                kv_n=args.kv_n,
                kv_vlen=args.kv_vlen,
                ni_n=args.ni_n,
                ni_name_width=args.ni_name_width,
                ni_idx_width=args.ni_idx_width,
                ni_q_first=args.ni_q_first,
                mm_n=args.mm_n,
                mm_name_width=args.mm_name_width,
                mm_q_first=args.mm_q_first,
            )
            prompt_ids = prompt_ids.to(device)
            ans_ids = ans_ids.to(device)

            use_fs = (args.mode == "prompt_fs")
            prompt_hidden, prompt_states = model(
                prompt_ids,
                future_seed=use_fs,
                fs_alpha=alpha,
                fs_alpha_head=alpha_head,
                seed_scale=args.seed_scale,
                return_states=True,
            )
            assert prompt_states is not None

            # token0: predicted from last prompt position
            logits0 = model.project(prompt_hidden[:, -1, :])  # [B,V]
            tgt0 = ans_ids[:, 0]
            loss0 = F.cross_entropy(logits0, tgt0)

            if ans_len > 1:
                ans_in = ans_ids[:, :-1]  # [B,ans_len-1]
                ans_in_pad = pad_tensor_to_multiple(ans_in, multiple=16, pad_id=pad_id)
                ans_hidden, _ = model(
                    ans_in_pad,
                    seed_states=prompt_states,
                    future_seed=False,
                    fs_alpha=None,
                    fs_alpha_head=None,
                    seed_scale=1.0,
                    return_states=False,
                )
                ans_hidden = ans_hidden[:, : ans_len - 1, :].contiguous()
                logits_rest = model.project(ans_hidden)  # [B,ans_len-1,V]
                tgt_rest = ans_ids[:, 1:].contiguous()
                loss_rest = F.cross_entropy(
                    logits_rest.view(-1, logits_rest.size(-1)),
                    tgt_rest.view(-1),
                )
                loss = loss0 + loss_rest
            else:
                loss = loss0
        else:
            # Masked-prefix probes (need future info within same sequence).
            x, y, p_ids, loss_start, loss_len = make_batch(
                rng=rng,
                task=args.task,
                mode=args.mode,
                fs_markers=args.fs_markers,
                byte2id=byte2id,
                pad_id=pad_id,
                seq_len=args.seq_len,
                bsz=args.bsz,
                retr_k=args.retr_k,
                retr_vlen=args.retr_vlen,
                retr_query_last=args.retr_query_last,
                retr_q_first=args.retr_q_first,
                kv_n=args.kv_n,
                kv_vlen=args.kv_vlen,
            )
            x = x.to(device)
            y = y.to(device)
            p_ids = p_ids.to(device)

            if args.mode == "prompt_fs":
                hidden, _ = model(
                    x,
                    future_seed=True,
                    fs_alpha=alpha,
                    fs_alpha_head=alpha_head,
                    seed_scale=args.seed_scale,
                    return_states=False,
                )
            else:
                hidden, _ = model(
                    x,
                    seed_states=None,
                    future_seed=False,
                    fs_alpha=None,
                    seed_scale=1.0,
                    return_states=False,
                )

            h = hidden[:, loss_start - 1 : loss_start - 1 + loss_len, :].contiguous()
            logits = model.project(h)
            targets = y[:, loss_start : loss_start + loss_len].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        loss.backward()
        opt.step()

        if (step % args.eval_every) == 0:
            with torch.no_grad():
                # eval on a few batches (keep small for speed)
                val_losses = []
                val_ems = []
                val_tok_accs = []
                for _ in range(args.val_batches):
                    if args.task in ("retrieval", "nameindex", "middlematch", "kvsort"):
                        vp, va, vplen, vans = make_batch_qa(
                            rng=val_rng,
                            task=args.task,
                            mode=args.mode,
                            fs_markers=args.fs_markers,
                            byte2id=byte2id,
                            bsz=args.bsz,
                            retr_k=args.retr_k,
                            retr_vlen=args.retr_vlen,
                            retr_query_last=args.retr_query_last,
                            retr_q_first=args.retr_q_first,
                            kv_n=args.kv_n,
                            kv_vlen=args.kv_vlen,
                            ni_n=args.ni_n,
                            ni_name_width=args.ni_name_width,
                            ni_idx_width=args.ni_idx_width,
                            ni_q_first=args.ni_q_first,
                            mm_n=args.mm_n,
                            mm_name_width=args.mm_name_width,
                            mm_q_first=args.mm_q_first,
                        )
                        vp = vp.to(device)
                        va = va.to(device)

                        use_fs = (args.mode == "prompt_fs")
                        v_prompt_hidden, v_prompt_states = model(
                            vp,
                            future_seed=use_fs,
                            fs_alpha=alpha,
                            fs_alpha_head=alpha_head,
                            seed_scale=args.seed_scale,
                            return_states=True,
                        )
                        assert v_prompt_states is not None

                        v_logits0 = model.project(v_prompt_hidden[:, -1, :])
                        v_tgt0 = va[:, 0]
                        v_loss0 = F.cross_entropy(v_logits0, v_tgt0)

                        if vans > 1:
                            v_in = va[:, :-1]
                            v_in_pad = pad_tensor_to_multiple(v_in, multiple=16, pad_id=pad_id)
                            v_ans_hidden, _ = model(
                                v_in_pad,
                                seed_states=v_prompt_states,
                                future_seed=False,
                                fs_alpha=None,
                                fs_alpha_head=None,
                                seed_scale=1.0,
                                return_states=False,
                            )
                            v_ans_hidden = v_ans_hidden[:, : vans - 1, :].contiguous()
                            v_logits_rest = model.project(v_ans_hidden)
                            v_tgt_rest = va[:, 1:].contiguous()
                            v_loss_rest = F.cross_entropy(
                                v_logits_rest.view(-1, v_logits_rest.size(-1)),
                                v_tgt_rest.view(-1),
                            )
                            v_loss = v_loss0 + v_loss_rest

                            v_pred0 = v_logits0.argmax(dim=-1, keepdim=True)
                            v_pred_rest = v_logits_rest.argmax(dim=-1)
                            v_pred = torch.cat([v_pred0, v_pred_rest], dim=1)
                        else:
                            v_loss = v_loss0
                            v_pred = v_logits0.argmax(dim=-1, keepdim=True)

                        val_losses.append(float(v_loss))
                        val_ems.append(float((v_pred == va).all(dim=1).float().mean().item()))
                        val_tok_accs.append(float((v_pred == va).float().mean().item()))
                    else:
                        vx, vy, vp, v_loss_start, v_loss_len = make_batch(
                            rng=val_rng,
                            task=args.task,
                            mode=args.mode,
                            fs_markers=args.fs_markers,
                            byte2id=byte2id,
                            pad_id=pad_id,
                            seq_len=args.seq_len,
                            bsz=args.bsz,
                            retr_k=args.retr_k,
                            retr_vlen=args.retr_vlen,
                            retr_query_last=args.retr_query_last,
                            retr_q_first=args.retr_q_first,
                            kv_n=args.kv_n,
                            kv_vlen=args.kv_vlen,
                        )
                        vx = vx.to(device)
                        vy = vy.to(device)
                        vp = vp.to(device)

                        if args.mode == "prompt_fs":
                            v_hidden, _ = model(
                                vx,
                                future_seed=True,
                                fs_alpha=alpha,
                                fs_alpha_head=alpha_head,
                                seed_scale=args.seed_scale,
                                return_states=False,
                            )
                        else:
                            v_hidden, _ = model(
                                vx,
                                seed_states=None,
                                future_seed=False,
                                fs_alpha=None,
                                seed_scale=1.0,
                                return_states=False,
                            )

                        vh = v_hidden[:, v_loss_start - 1 : v_loss_start - 1 + v_loss_len, :].contiguous()
                        v_logits = model.project(vh)
                        v_targets = vy[:, v_loss_start : v_loss_start + v_loss_len].contiguous()
                        v_loss = F.cross_entropy(
                            v_logits.view(-1, v_logits.size(-1)),
                            v_targets.view(-1),
                            ignore_index=-100,
                        )
                        val_losses.append(float(v_loss))
                        val_ems.append(eval_exact_match(v_logits, v_targets))
                        val_tok_accs.append(eval_token_acc(v_logits, v_targets))

                rec = {
                    "t": round(time.time() - t0, 2),
                    "step": step,
                    "train_loss": float(loss),
                    "val_loss": sum(val_losses) / len(val_losses),
                    "val_em": sum(val_ems) / len(val_ems),
                    "val_tok_acc": sum(val_tok_accs) / len(val_tok_accs),
                    "alpha_mean": float(torch.sigmoid(alpha[1:]).mean()),
                }
                if alpha_head is not None:
                    rec["alpha_head_mean"] = float(torch.sigmoid(alpha_head[1:]).mean())

                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

        step += 1

    print(str(run_root))


if __name__ == "__main__":
    main()
