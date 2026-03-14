#!/usr/bin/env python3
"""Utilities for 9x9 Sudoku solved-board in-place repair."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from sudoku9_unique import (
    CELL_COUNT,
    blank_token_accuracy,
    board_to_string,
    is_clue_consistent,
    is_valid_solution,
    randomized_solved_board,
    string_to_board,
)

MASK_CHAR = "0"


@dataclass(frozen=True)
class SudokuInplaceExample:
    split: str
    mask_count: int
    item_id: str
    masked_board: str
    solution: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True)


def count_masks(masked_board: str) -> int:
    return sum(ch == MASK_CHAR for ch in masked_board)


def apply_random_mask(solution: str, *, mask_count: int, rng: random.Random) -> str:
    if len(solution) != CELL_COUNT:
        raise ValueError(f"expected solution length {CELL_COUNT}, got {len(solution)}")
    if mask_count < 1 or mask_count >= CELL_COUNT:
        raise ValueError(f"mask_count must be in [1, {CELL_COUNT - 1}]")
    board = list(solution)
    for idx in rng.sample(list(range(CELL_COUNT)), k=int(mask_count)):
        board[idx] = MASK_CHAR
    return "".join(board)


def generate_example(split: str, mask_target: int, seed: int, index: int) -> SudokuInplaceExample:
    rng = random.Random((int(seed) * 1_000_003) + (int(mask_target) * 10_007) + int(index))
    solution = board_to_string(randomized_solved_board(rng))
    masked_board = apply_random_mask(solution, mask_count=int(mask_target), rng=rng)
    item_id = f"{split}_m{int(mask_target)}_{int(index):05d}"
    return SudokuInplaceExample(
        split=split,
        mask_count=int(mask_target),
        item_id=item_id,
        masked_board=masked_board,
        solution=solution,
    )


def write_manifest(path: Path, examples: Iterable[SudokuInplaceExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")


def load_manifest(path: Path) -> list[SudokuInplaceExample]:
    out: list[SudokuInplaceExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(SudokuInplaceExample(**json.loads(line)))
    return out


def grouped_by_mask(examples: Iterable[SudokuInplaceExample]) -> dict[int, list[SudokuInplaceExample]]:
    out: dict[int, list[SudokuInplaceExample]] = {}
    for ex in examples:
        out.setdefault(int(ex.mask_count), []).append(ex)
    return out


def validate_example(example: SudokuInplaceExample) -> None:
    if len(example.masked_board) != CELL_COUNT or len(example.solution) != CELL_COUNT:
        raise ValueError(f"bad example length for {example.item_id}")
    if count_masks(example.masked_board) != int(example.mask_count):
        raise ValueError(f"mask count mismatch for {example.item_id}")
    if not is_clue_consistent(example.masked_board, example.solution):
        raise ValueError(f"solution violates fixed clues for {example.item_id}")
    if not is_valid_solution(string_to_board(example.solution)):
        raise ValueError(f"invalid solution for {example.item_id}")


def masked_token_accuracy(masked_board: str, prediction: str, solution: str) -> float:
    return blank_token_accuracy(masked_board, prediction, solution)


def self_test(seed: int = 1234) -> None:
    for mask_count in (28, 32, 36, 40):
        ex = generate_example(split="self_test", mask_target=mask_count, seed=seed, index=mask_count)
        validate_example(ex)
        assert count_masks(ex.masked_board) == mask_count
        assert masked_token_accuracy(ex.masked_board, ex.solution, ex.solution) == 1.0
    print("sudoku9_inplace_self_test_ok")


if __name__ == "__main__":
    self_test()
