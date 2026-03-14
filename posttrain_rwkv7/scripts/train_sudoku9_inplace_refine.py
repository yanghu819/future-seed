#!/usr/bin/env python3
"""9x9 Sudoku solved-board in-place repair benchmark for Future-Seed."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, REPO_ROOT):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from cache_defaults import apply_cache_env, ensure_cache_dirs
from sudoku9_inplace import (
    MASK_CHAR,
    SudokuInplaceExample,
    generate_example,
    grouped_by_mask,
    is_clue_consistent,
    is_valid_solution,
    load_manifest,
    masked_token_accuracy,
    string_to_board,
    validate_example,
)

PROMPT_PREFIX = "SUDOKU9|B="
BOARD_LEN = 81
ALLOWED_CHARS = set("SUDOKU9|B=0123456789")


@dataclass(frozen=True)
class FSConfig:
    mode: str
    fs_variant: str
    alpha_lr: float
    alpha_init: float
    alpha_head_lr: float
    alpha_head_init: float
    seed_scale: float
    fs_layer_start: int
    fs_alpha_schedule: str
    fs_alpha_min: float
    fs_alpha_max: float
    fs_norm: bool
    fs_clip: float
    fs_detach: bool


def parse_int_list(text: str) -> list[int]:
    vals = [int(x) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one integer")
    return vals


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def record_rank_key(row: dict[str, object]) -> tuple[float, float]:
    return (
        float(row.get("val_focus_board_exact_mean", 0.0)),
        float(row.get("val_focus_masked_acc_mean", 0.0)),
    )


class TrainPool:
    def __init__(self, *, masks: Sequence[int], cache_per_mask: int, seed: int):
        self.masks = [int(m) for m in masks]
        self.cache_per_mask = int(cache_per_mask)
        self.rng = random.Random(int(seed))
        self.bank: dict[int, list[SudokuInplaceExample]] = {m: [] for m in self.masks}
        self.next_index: dict[int, int] = {m: 0 for m in self.masks}

    def _top_up(self, mask_count: int) -> None:
        bank = self.bank[mask_count]
        while len(bank) < self.cache_per_mask:
            item_seed = self.rng.randrange(1 << 61)
            item_idx = self.next_index[mask_count]
            bank.append(generate_example(split="train", mask_target=mask_count, seed=item_seed, index=item_idx))
            self.next_index[mask_count] = item_idx + 1

    def sample(self, batch_size: int) -> list[SudokuInplaceExample]:
        out: list[SudokuInplaceExample] = []
        for _ in range(int(batch_size)):
            mask_count = self.rng.choice(self.masks)
            self._top_up(mask_count)
            bank = self.bank[mask_count]
            pos = self.rng.randrange(len(bank))
            out.append(bank.pop(pos))
        return out


class SudokuCharCodec:
    def __init__(self, char_to_id: dict[str, int]):
        self.char_to_id = dict(char_to_id)
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

    @classmethod
    def from_tokenizer(cls, tok) -> "SudokuCharCodec":
        char_to_id: dict[str, int] = {}
        for ch in sorted(ALLOWED_CHARS):
            ids = tok.encode(ch)
            if len(ids) != 1:
                raise ValueError(f"character {ch!r} does not map to a single token: {ids}")
            char_to_id[ch] = int(ids[0])
        return cls(char_to_id)

    def encode_text(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text]

    def decode_ids(self, ids: Sequence[int]) -> str:
        try:
            return "".join(self.id_to_char[int(i)] for i in ids)
        except KeyError as exc:
            raise ValueError(f"unknown Sudoku token id: {exc}") from exc


def board_prompt(example: SudokuInplaceExample) -> tuple[str, str, list[bool], list[bool]]:
    validate_example(example)
    prompt = f"{PROMPT_PREFIX}{example.masked_board}"
    solution = example.solution
    clue_mask = [ch != MASK_CHAR for ch in example.masked_board]
    masked_positions = [not x for x in clue_mask]
    return prompt, solution, clue_mask, masked_positions


def self_test() -> None:
    ex = generate_example(split="self_test", mask_target=32, seed=1234, index=0)
    prompt, solution, clue_mask, masked_positions = board_prompt(ex)
    assert prompt.startswith(PROMPT_PREFIX)
    assert len(solution) == BOARD_LEN
    assert len(clue_mask) == BOARD_LEN
    assert len(masked_positions) == BOARD_LEN
    assert sum(masked_positions) == ex.mask_count
    print("sudoku9_inplace_train_self_test_ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--mode", choices=["no_fs", "prompt_fs"], default="no_fs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=1234)
    ap.add_argument("--train_masks", type=str, default="32,36")
    ap.add_argument("--focus_masks", type=str, default="32")
    ap.add_argument("--eval_masks", type=str, default="28,32,36,40")
    ap.add_argument("--train_cache_per_mask", type=int, default=32)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--time_budget_sec", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--eval_examples_per_mask", type=int, default=64)
    ap.add_argument("--final_eval_examples_per_mask", type=int, default=0)
    ap.add_argument("--refine_steps_train", type=int, default=1)
    ap.add_argument("--refine_steps_eval", type=int, default=1)
    ap.add_argument("--model_lr", type=float, default=3e-5)
    ap.add_argument("--alpha_lr", type=float, default=0.0)
    ap.add_argument("--alpha_init", type=float, default=-2.0)
    ap.add_argument("--fs_variant", choices=["scalar", "head"], default="scalar")
    ap.add_argument("--alpha_head_lr", type=float, default=None)
    ap.add_argument("--alpha_head_init", type=float, default=None)
    ap.add_argument("--seed_scale", type=float, default=1.0)
    ap.add_argument("--fs_layer_start", type=int, default=6)
    ap.add_argument("--fs_alpha_schedule", choices=["none", "linear", "cosine"], default="none")
    ap.add_argument("--fs_alpha_min", type=float, default=1.0)
    ap.add_argument("--fs_alpha_max", type=float, default=1.0)
    ap.add_argument("--fs_norm", action="store_true")
    ap.add_argument("--fs_clip", type=float, default=1.0)
    ap.add_argument("--fs_detach", action="store_true")
    ap.add_argument("--consistency_lambda", type=float, default=0.0)
    ap.add_argument("--val_manifest", type=str, default="")
    ap.add_argument("--test_manifest", type=str, default="")
    ap.add_argument("--weights", type=str, default="assets/weights/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
    ap.add_argument("--vocab", type=str, default="assets/tokenizer/rwkv_vocab_v20230424.txt")
    ap.add_argument("--cuda_src", type=str, default="cuda/rwkv_cuda_wind")
    ap.add_argument("--run_dir", type=str, default="runs")
    ap.add_argument("--tag", type=str, default="baseline")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.alpha_head_lr is None:
        args.alpha_head_lr = float(args.alpha_lr)
    if args.alpha_head_init is None:
        args.alpha_head_init = float(args.alpha_init)

    train_masks = parse_int_list(args.train_masks)
    eval_masks = parse_int_list(args.eval_masks)
    focus_masks = parse_int_list(args.focus_masks)

    apply_cache_env()
    ensure_cache_dirs()

    if not args.val_manifest:
        raise ValueError("--val_manifest is required")

    import torch
    import torch.nn.functional as F
    from rwkv_tokenizer import RWKVWorldTokenizer
    from rwkv7_g1d import RWKV7G1DLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = RWKVWorldTokenizer(args.vocab)
    codec = SudokuCharCodec.from_tokenizer(tok)
    pad_id = int(tok.eot_id)
    digit_ids = torch.tensor([codec.char_to_id[str(i)] for i in range(1, 10)], dtype=torch.long, device=device)
    board_start = len(PROMPT_PREFIX)
    seq_len = board_start + BOARD_LEN
    seq_len_pad = round_up(seq_len, 16)

    val_examples = load_manifest(Path(args.val_manifest))
    val_by_mask = grouped_by_mask(val_examples)
    test_by_mask = grouped_by_mask(load_manifest(Path(args.test_manifest))) if args.test_manifest else {}

    run_root = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S") / "sudoku9_inplace_refine" / str(args.tag)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

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

    fs_cfg = FSConfig(
        mode=str(args.mode),
        fs_variant=str(args.fs_variant),
        alpha_lr=float(args.alpha_lr),
        alpha_init=float(args.alpha_init),
        alpha_head_lr=float(args.alpha_head_lr),
        alpha_head_init=float(args.alpha_head_init),
        seed_scale=float(args.seed_scale),
        fs_layer_start=int(args.fs_layer_start),
        fs_alpha_schedule=str(args.fs_alpha_schedule),
        fs_alpha_min=float(args.fs_alpha_min),
        fs_alpha_max=float(args.fs_alpha_max),
        fs_norm=bool(args.fs_norm),
        fs_clip=float(args.fs_clip),
        fs_detach=bool(args.fs_detach),
    )
    train_pool = TrainPool(masks=train_masks, cache_per_mask=int(args.train_cache_per_mask), seed=int(args.seed))

    def encode_examples(examples: Sequence[SudokuInplaceExample]):
        full_rows = []
        solution_rows = []
        masked_rows = []
        clue_masks = []
        masked_boards = []
        solutions = []
        mask_counts = []
        for ex in examples:
            prompt, solution, clue_mask, masked_positions = board_prompt(ex)
            ids = codec.encode_text(prompt)
            ids = ids + [pad_id] * (seq_len_pad - len(ids))
            full_rows.append(ids)
            solution_rows.append([codec.char_to_id[ch] for ch in solution])
            masked_rows.append(masked_positions)
            clue_masks.append(clue_mask)
            masked_boards.append(ex.masked_board)
            solutions.append(ex.solution)
            mask_counts.append(int(ex.mask_count))
        input_ids = torch.tensor(full_rows, dtype=torch.long, device=device)
        solution_ids = torch.tensor(solution_rows, dtype=torch.long, device=device)
        masked_mask = torch.tensor(masked_rows, dtype=torch.bool, device=device)
        clue_mask_t = torch.tensor(clue_masks, dtype=torch.bool, device=device)
        return input_ids, solution_ids, masked_mask, clue_mask_t, masked_boards, solutions, mask_counts

    def forward_hidden(input_ids: torch.Tensor):
        hidden, _ = model(
            input_ids,
            future_seed=(fs_cfg.mode == "prompt_fs"),
            fs_alpha=alpha,
            fs_alpha_head=alpha_head,
            seed_scale=fs_cfg.seed_scale,
            fs_layer_start=fs_cfg.fs_layer_start,
            fs_alpha_schedule=fs_cfg.fs_alpha_schedule,
            fs_alpha_min=fs_cfg.fs_alpha_min,
            fs_alpha_max=fs_cfg.fs_alpha_max,
            fs_norm=fs_cfg.fs_norm,
            fs_clip=fs_cfg.fs_clip,
            fs_detach=fs_cfg.fs_detach,
            return_states=False,
        )
        return hidden[:, board_start : board_start + BOARD_LEN, :]

    def argmax_digit_ids(logits: torch.Tensor) -> torch.Tensor:
        digit_logits = logits.index_select(dim=-1, index=digit_ids)
        return digit_ids[digit_logits.argmax(dim=-1)]

    def consistency_penalty(logits: torch.Tensor, masked_mask: torch.Tensor, solution_ids: torch.Tensor) -> torch.Tensor:
        if float(args.consistency_lambda) <= 0.0:
            return logits.new_zeros(())
        digit_logits = logits.index_select(dim=-1, index=digit_ids)
        p_model = torch.softmax(digit_logits, dim=-1)
        y_idx = torch.zeros_like(solution_ids)
        for di, token_id in enumerate(digit_ids.tolist()):
            y_idx = torch.where(solution_ids == int(token_id), torch.full_like(y_idx, di), y_idx)
        p_true = F.one_hot(y_idx, num_classes=9).to(p_model.dtype)
        p = torch.where(masked_mask.unsqueeze(-1), p_model, p_true)
        groups = []
        for row in range(9):
            groups.append([row * 9 + col for col in range(9)])
        for col in range(9):
            groups.append([row * 9 + col for row in range(9)])
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                groups.append([(br + dr) * 9 + (bc + dc) for dr in range(3) for dc in range(3)])
        cons = logits.new_zeros(())
        for group in groups:
            cnt = p[:, group, :].sum(dim=1)
            cons = cons + (cnt - 1.0).pow(2).mean()
        return cons / len(groups)

    def evaluate_examples(examples: Sequence[SudokuInplaceExample], refine_steps: int) -> dict[str, float]:
        if not examples:
            return {"board_exact": 0.0, "board_valid": 0.0, "clue_consistent": 0.0, "masked_acc": 0.0}
        model.eval()
        exact = 0
        valid = 0
        clue_ok = 0
        masked_scores = []
        with torch.no_grad():
            for start in range(0, len(examples), int(args.bsz)):
                batch = examples[start : start + int(args.bsz)]
                input_ids, solution_ids, masked_mask, clue_mask_t, masked_boards, solutions, _ = encode_examples(batch)
                current = input_ids.clone()
                for _ in range(int(refine_steps)):
                    board_hidden = forward_hidden(current)
                    logits = model.project(board_hidden)
                    pred_digit_ids = argmax_digit_ids(logits)
                    board_tokens = current[:, board_start : board_start + BOARD_LEN].clone()
                    board_tokens = torch.where(clue_mask_t, board_tokens, pred_digit_ids)
                    current[:, board_start : board_start + BOARD_LEN] = board_tokens
                final_board_ids = current[:, board_start : board_start + BOARD_LEN]
                for row_idx in range(final_board_ids.size(0)):
                    pred = codec.decode_ids(final_board_ids[row_idx].tolist())
                    masked_board = masked_boards[row_idx]
                    solution = solutions[row_idx]
                    masked_scores.append(masked_token_accuracy(masked_board, pred, solution))
                    if is_clue_consistent(masked_board, pred):
                        clue_ok += 1
                    if is_valid_solution(string_to_board(pred)):
                        valid += 1
                    if pred == solution:
                        exact += 1
        model.train()
        total = len(examples)
        return {
            "board_exact": float(exact / total),
            "board_valid": float(valid / total),
            "clue_consistent": float(clue_ok / total),
            "masked_acc": float(sum(masked_scores) / len(masked_scores)),
        }

    def evaluate_manifest(groups: dict[int, list[SudokuInplaceExample]], examples_per_mask: int, refine_steps: int) -> dict[str, float]:
        out: dict[str, float] = {}
        focus_exact = []
        focus_valid = []
        focus_clue = []
        focus_masked = []
        for mask_count in eval_masks:
            bucket = groups.get(int(mask_count), [])[: int(examples_per_mask)]
            metrics = evaluate_examples(bucket, refine_steps=refine_steps)
            for name, value in metrics.items():
                out[f"{name}_{mask_count}"] = float(value)
            if mask_count in focus_masks:
                focus_exact.append(metrics["board_exact"])
                focus_valid.append(metrics["board_valid"])
                focus_clue.append(metrics["clue_consistent"])
                focus_masked.append(metrics["masked_acc"])
        out["focus_board_exact_mean"] = float(statistics.mean(focus_exact)) if focus_exact else 0.0
        out["focus_board_valid_mean"] = float(statistics.mean(focus_valid)) if focus_valid else 0.0
        out["focus_clue_consistent_mean"] = float(statistics.mean(focus_clue)) if focus_clue else 0.0
        out["focus_masked_acc_mean"] = float(statistics.mean(focus_masked)) if focus_masked else 0.0
        return out

    best_focus = float("-inf")
    best_record: dict[str, object] | None = None
    t0 = time.time()
    step = 0
    while True:
        if int(args.max_steps) > 0 and step >= int(args.max_steps):
            break
        if int(args.time_budget_sec) > 0 and (time.time() - t0) >= float(args.time_budget_sec):
            break

        batch = train_pool.sample(int(args.bsz))
        input_ids, solution_ids, masked_mask, _clue_mask_t, _masked_boards, _solutions, mask_counts = encode_examples(batch)
        opt.zero_grad(set_to_none=True)

        current = input_ids.clone()
        step_losses = []
        step_cons = []
        logits = None
        vocab_size = None
        targets = solution_ids.clone()
        targets[~masked_mask] = -100
        for refine_idx in range(max(1, int(args.refine_steps_train))):
            board_hidden = forward_hidden(current)
            logits = model.project(board_hidden)
            vocab_size = logits.size(-1)
            step_losses.append(F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100))
            step_cons.append(consistency_penalty(logits, masked_mask, solution_ids))
            if refine_idx + 1 < int(args.refine_steps_train):
                pred_digit_ids = argmax_digit_ids(logits).detach()
                board_tokens = current[:, board_start : board_start + BOARD_LEN].clone()
                board_tokens = torch.where(masked_mask, pred_digit_ids, board_tokens)
                current[:, board_start : board_start + BOARD_LEN] = board_tokens
        loss = torch.stack(step_losses).mean()
        cons = torch.stack(step_cons).mean()
        total_loss = loss + float(args.consistency_lambda) * cons
        total_loss.backward()
        opt.step()

        if (step % int(args.eval_every)) == 0:
            val_metrics = evaluate_manifest(val_by_mask, int(args.eval_examples_per_mask), refine_steps=int(args.refine_steps_eval))
            rec: dict[str, object] = {
                "t": round(time.time() - t0, 2),
                "step": int(step),
                "train_loss": float(total_loss),
                "train_mask_mean": float(statistics.mean(mask_counts)),
                "consistency_lambda": float(args.consistency_lambda),
                "refine_steps_train": int(args.refine_steps_train),
                "refine_steps": int(args.refine_steps_eval),
                "alpha_mean": float(torch.sigmoid(alpha[1:]).mean()) if alpha.numel() > 1 else float(torch.sigmoid(alpha).mean()),
                "alpha_head_mean": float(torch.sigmoid(alpha_head[1:]).mean()) if alpha_head is not None and alpha_head.size(0) > 1 else (float(torch.sigmoid(alpha_head).mean()) if alpha_head is not None else None),
            }
            for key, value in val_metrics.items():
                rec[f"val_{key}"] = float(value)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if best_record is None or record_rank_key(rec) > record_rank_key(best_record):
                best_focus = float(val_metrics["focus_board_exact_mean"])
                best_record = dict(rec)

        step += 1

    final_summary: dict[str, object] = {
        "tag": str(args.tag),
        "mode": str(args.mode),
        "train_masks": train_masks,
        "focus_masks": focus_masks,
        "eval_masks": eval_masks,
        "steps_ran": int(step),
        "best_val_focus_board_exact_mean": best_focus if best_record is not None else None,
        "best_record": best_record,
        "refine_steps_train": int(args.refine_steps_train),
        "refine_steps_eval": int(args.refine_steps_eval),
        "consistency_lambda": float(args.consistency_lambda),
    }

    if int(args.final_eval_examples_per_mask) > 0 and test_by_mask:
        final_test = evaluate_manifest(test_by_mask, int(args.final_eval_examples_per_mask), refine_steps=int(args.refine_steps_eval))
        final_summary["final_test"] = final_test

    (run_root / "summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(str(run_root))


if __name__ == "__main__":
    main()
