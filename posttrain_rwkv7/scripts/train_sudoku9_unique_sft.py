#!/usr/bin/env python3
"""9x9 unique-solution Sudoku mainline benchmark for Future-Seed.

This benchmark differs from the older `train_sudoku_sft.py` probe:
- puzzles are 9x9 unique-solution Sudokus
- loss is only applied on originally blank cells
- evaluation is autoregressive in-place fill with clue forcing
- headline metrics are exact solve / validity / clue consistency / blank accuracy
"""

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
from sudoku9_unique import (
    SudokuExample,
    blank_token_accuracy,
    generate_example,
    grouped_by_clue,
    is_clue_consistent,
    is_valid_solution,
    load_manifest,
    string_to_board,
    validate_example,
)

PROMPT_PREFIX = "SUDOKU9|G="
PROMPT_SUFFIX = "|A="
ANSWER_LEN = 81
ALLOWED_CHARS = set("SUDOKU9|GA=0123456789")


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


class TrainPool:
    def __init__(self, *, clues: Sequence[int], cache_per_clue: int, seed: int):
        self.clues = [int(c) for c in clues]
        self.cache_per_clue = int(cache_per_clue)
        self.rng = random.Random(int(seed))
        self.bank: dict[int, list[SudokuExample]] = {c: [] for c in self.clues}
        self.next_index: dict[int, int] = {c: 0 for c in self.clues}

    def _top_up(self, clue: int) -> None:
        bank = self.bank[clue]
        while len(bank) < self.cache_per_clue:
            item_seed = self.rng.randrange(1 << 61)
            item_idx = self.next_index[clue]
            bank.append(generate_example(split="train", clue_target=clue, seed=item_seed, index=item_idx))
            self.next_index[clue] = item_idx + 1

    def sample(self, batch_size: int) -> list[SudokuExample]:
        out: list[SudokuExample] = []
        for _ in range(int(batch_size)):
            clue = self.rng.choice(self.clues)
            self._top_up(clue)
            bank = self.bank[clue]
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


def prompt_and_answer(example: SudokuExample) -> tuple[str, str, list[bool], list[bool]]:
    validate_example(example)
    prompt = f"{PROMPT_PREFIX}{example.puzzle}{PROMPT_SUFFIX}"
    answer = example.solution
    clue_mask = [ch != "0" for ch in example.puzzle]
    blank_mask = [not x for x in clue_mask]
    return prompt, answer, clue_mask, blank_mask


def self_test() -> None:
    ex = generate_example(split="self_test", clue_target=32, seed=1234, index=0)
    prompt, answer, clue_mask, blank_mask = prompt_and_answer(ex)
    assert prompt.startswith(PROMPT_PREFIX)
    assert prompt.endswith(PROMPT_SUFFIX)
    assert len(answer) == ANSWER_LEN
    assert len(clue_mask) == ANSWER_LEN
    assert len(blank_mask) == ANSWER_LEN
    assert sum(clue_mask) == ex.clue_count
    assert sum(blank_mask) == ANSWER_LEN - ex.clue_count
    print("sudoku9_unique_train_self_test_ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--mode", choices=["no_fs", "prompt_fs"], default="no_fs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=1234)
    ap.add_argument("--train_clues", type=str, default="40,36,32,28")
    ap.add_argument("--focus_clues", type=str, default="32,28")
    ap.add_argument("--eval_clues", type=str, default="40,36,32,28,24")
    ap.add_argument("--train_cache_per_clue", type=int, default=32)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--time_budget_sec", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--eval_examples_per_clue", type=int, default=64)
    ap.add_argument("--final_eval_examples_per_clue", type=int, default=0)
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

    train_clues = parse_int_list(args.train_clues)
    eval_clues = parse_int_list(args.eval_clues)
    focus_clues = parse_int_list(args.focus_clues)

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

    val_examples = load_manifest(Path(args.val_manifest))
    val_by_clue = grouped_by_clue(val_examples)
    test_by_clue = grouped_by_clue(load_manifest(Path(args.test_manifest))) if args.test_manifest else {}

    run_root = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S") / "sudoku9_unique_sft" / str(args.tag)
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
    train_pool = TrainPool(clues=train_clues, cache_per_clue=int(args.train_cache_per_clue), seed=int(args.seed))

    def pad_left_list(seqs: Sequence[list[int]], *, multiple: int = 16) -> tuple[torch.Tensor, int]:
        max_len = max(len(s) for s in seqs)
        max_len_pad = round_up(max_len, multiple)
        out = []
        for s in seqs:
            out.append([pad_id] * (max_len_pad - len(s)) + s)
        return torch.tensor(out, dtype=torch.long, device=device), max_len

    def pad_right_tensor(t: torch.Tensor, *, multiple: int = 16) -> tuple[torch.Tensor, int]:
        seq_len = int(t.size(1))
        if seq_len == 0:
            return t, 0
        seq_len_pad = round_up(seq_len, multiple)
        if seq_len_pad == seq_len:
            return t, seq_len
        pad = torch.full((int(t.size(0)), seq_len_pad - seq_len), pad_id, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=1), seq_len

    def encode_examples(examples: Sequence[SudokuExample]):
        prompt_rows = []
        answer_rows = []
        blank_masks = []
        clue_masks = []
        puzzles = []
        solutions = []
        clue_counts = []
        for ex in examples:
            prompt, answer, clue_mask, blank_mask = prompt_and_answer(ex)
            prompt_rows.append(codec.encode_text(prompt))
            answer_rows.append(codec.encode_text(answer))
            blank_masks.append(blank_mask)
            clue_masks.append(clue_mask)
            puzzles.append(ex.puzzle)
            solutions.append(ex.solution)
            clue_counts.append(int(ex.clue_count))
        prompt_ids, prompt_len = pad_left_list(prompt_rows, multiple=16)
        answer_ids = torch.tensor(answer_rows, dtype=torch.long, device=device)
        blank_mask_t = torch.tensor(blank_masks, dtype=torch.bool, device=device)
        clue_mask_t = torch.tensor(clue_masks, dtype=torch.bool, device=device)
        return prompt_ids, prompt_len, answer_ids, blank_mask_t, clue_mask_t, puzzles, solutions, clue_counts

    def prompt_forward(prompt_ids: torch.Tensor):
        return model(
            prompt_ids,
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
            return_states=True,
        )

    def argmax_digit_ids(logits: torch.Tensor) -> torch.Tensor:
        digit_logits = logits.index_select(dim=-1, index=digit_ids)
        return digit_ids[digit_logits.argmax(dim=-1)]

    def evaluate_examples(examples: Sequence[SudokuExample]) -> dict[str, float]:
        if not examples:
            return {"exact": 0.0, "valid": 0.0, "clue": 0.0, "blank_acc": 0.0}
        model.eval()
        exact = 0
        valid = 0
        clue_ok = 0
        blank_scores = []
        with torch.no_grad():
            for start in range(0, len(examples), int(args.bsz)):
                batch = examples[start : start + int(args.bsz)]
                prompt_ids, prompt_len, answer_ids, _blank_mask, clue_mask, puzzles, solutions, _ = encode_examples(batch)
                prompt_hidden, states = prompt_forward(prompt_ids)
                assert states is not None
                generated = torch.empty_like(answer_ids)
                logits = model.project(prompt_hidden[:, prompt_len - 1, :])
                next_tok = argmax_digit_ids(logits)
                generated[:, 0] = torch.where(clue_mask[:, 0], answer_ids[:, 0], next_tok)
                for pos in range(1, ANSWER_LEN):
                    prefix_pad, prefix_len = pad_right_tensor(generated[:, :pos], multiple=16)
                    hidden, _ = model(
                        prefix_pad,
                        seed_states=states,
                        future_seed=False,
                        fs_alpha=None,
                        fs_alpha_head=None,
                        seed_scale=1.0,
                        return_states=False,
                    )
                    logits = model.project(hidden[:, prefix_len - 1, :])
                    next_tok = argmax_digit_ids(logits)
                    generated[:, pos] = torch.where(clue_mask[:, pos], answer_ids[:, pos], next_tok)

                for row_idx in range(generated.size(0)):
                    pred = codec.decode_ids(generated[row_idx].tolist())
                    puzzle = puzzles[row_idx]
                    solution = solutions[row_idx]
                    blank_scores.append(blank_token_accuracy(puzzle, pred, solution))
                    if is_clue_consistent(puzzle, pred):
                        clue_ok += 1
                    if is_valid_solution(string_to_board(pred)):
                        valid += 1
                    if pred == solution:
                        exact += 1
        model.train()
        total = len(examples)
        return {
            "exact": float(exact / total),
            "valid": float(valid / total),
            "clue": float(clue_ok / total),
            "blank_acc": float(sum(blank_scores) / len(blank_scores)),
        }

    def evaluate_manifest(groups: dict[int, list[SudokuExample]], examples_per_clue: int) -> dict[str, float]:
        out: dict[str, float] = {}
        focus_exact = []
        focus_valid = []
        focus_clue = []
        focus_blank = []
        for clue in eval_clues:
            bucket = groups.get(int(clue), [])[: int(examples_per_clue)]
            metrics = evaluate_examples(bucket)
            for name, value in metrics.items():
                out[f"{name}_{clue}"] = float(value)
            if clue in focus_clues:
                focus_exact.append(metrics["exact"])
                focus_valid.append(metrics["valid"])
                focus_clue.append(metrics["clue"])
                focus_blank.append(metrics["blank_acc"])
        out["focus_exact_mean"] = float(statistics.mean(focus_exact)) if focus_exact else 0.0
        out["focus_valid_mean"] = float(statistics.mean(focus_valid)) if focus_valid else 0.0
        out["focus_clue_mean"] = float(statistics.mean(focus_clue)) if focus_clue else 0.0
        out["focus_blank_acc_mean"] = float(statistics.mean(focus_blank)) if focus_blank else 0.0
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
        prompt_ids, prompt_len, answer_ids, blank_mask, _clue_mask, _puzzles, _solutions, clue_counts = encode_examples(batch)
        opt.zero_grad(set_to_none=True)

        prompt_hidden, prompt_states = prompt_forward(prompt_ids)
        assert prompt_states is not None

        prompt_last_hidden = prompt_hidden[:, prompt_len - 1, :]
        vocab_size = model.project(prompt_last_hidden).size(-1)
        loss0 = torch.tensor(0.0, device=device)
        if bool(blank_mask[:, 0].any()):
            logits0 = model.project(prompt_last_hidden)
            loss0 = F.cross_entropy(logits0[blank_mask[:, 0]], answer_ids[:, 0][blank_mask[:, 0]])

        ans_in = answer_ids[:, :-1].contiguous()
        ans_in_pad, ans_in_len = pad_right_tensor(ans_in, multiple=16)
        ans_tgt = answer_ids[:, 1:].clone()
        ans_tgt[~blank_mask[:, 1:]] = -100
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
        loss_rest = F.cross_entropy(logits.view(-1, vocab_size), ans_tgt[:, :ans_in_len].contiguous().view(-1), ignore_index=-100)
        loss = loss0 + loss_rest
        loss.backward()
        opt.step()

        if (step % int(args.eval_every)) == 0:
            val_metrics = evaluate_manifest(val_by_clue, int(args.eval_examples_per_clue))
            rec: dict[str, object] = {
                "t": round(time.time() - t0, 2),
                "step": int(step),
                "train_loss": float(loss),
                "train_clue_mean": float(statistics.mean(clue_counts)),
                "alpha_mean": float(torch.sigmoid(alpha[1:]).mean()) if alpha.numel() > 1 else float(torch.sigmoid(alpha).mean()),
                "alpha_head_mean": float(torch.sigmoid(alpha_head[1:]).mean()) if alpha_head is not None and alpha_head.size(0) > 1 else (float(torch.sigmoid(alpha_head).mean()) if alpha_head is not None else None),
            }
            for key, value in val_metrics.items():
                rec[f"val_{key}"] = float(value)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if float(val_metrics["focus_exact_mean"]) > best_focus:
                best_focus = float(val_metrics["focus_exact_mean"])
                best_record = dict(rec)

        step += 1

    final_summary: dict[str, object] = {
        "tag": str(args.tag),
        "mode": str(args.mode),
        "train_clues": train_clues,
        "focus_clues": focus_clues,
        "eval_clues": eval_clues,
        "steps_ran": int(step),
        "best_val_focus_exact_mean": best_focus if best_record is not None else None,
        "best_record": best_record,
    }

    if int(args.final_eval_examples_per_clue) > 0 and test_by_clue:
        test_metrics = evaluate_manifest(test_by_clue, int(args.final_eval_examples_per_clue))
        final_summary["final_test"] = test_metrics

    (run_root / "summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(str(run_root))


if __name__ == "__main__":
    main()
