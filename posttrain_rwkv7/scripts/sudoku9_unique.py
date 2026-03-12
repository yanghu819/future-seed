#!/usr/bin/env python3
"""Utilities for 9x9 unique-solution Sudoku generation and evaluation."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

SIDE = 9
BOX = 3
CELL_COUNT = SIDE * SIDE
DIGITS = tuple(range(1, SIDE + 1))
DIGIT_MASK = sum(1 << d for d in DIGITS)
BOARD_CHARS = "0123456789"


@dataclass(frozen=True)
class SudokuExample:
    split: str
    clue_count: int
    item_id: str
    puzzle: str
    solution: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True)


def board_to_string(board: Sequence[int]) -> str:
    if len(board) != CELL_COUNT:
        raise ValueError(f"expected {CELL_COUNT} cells, got {len(board)}")
    return "".join(str(v) for v in board)


def string_to_board(text: str) -> list[int]:
    if len(text) != CELL_COUNT:
        raise ValueError(f"expected length {CELL_COUNT}, got {len(text)}")
    if any(ch not in BOARD_CHARS for ch in text):
        raise ValueError("board text must contain only digits 0-9")
    return [int(ch) for ch in text]


def clue_count(puzzle: str) -> int:
    return sum(ch != "0" for ch in puzzle)


def box_index(row: int, col: int) -> int:
    return (row // BOX) * BOX + (col // BOX)


def iter_bits(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def randomized_solved_board(rng: random.Random) -> list[int]:
    def pattern(row: int, col: int) -> int:
        return (BOX * (row % BOX) + row // BOX + col) % SIDE

    def shuffled(values: Sequence[int]) -> list[int]:
        out = list(values)
        rng.shuffle(out)
        return out

    base = list(range(BOX))
    rows = [g * BOX + r for g in shuffled(base) for r in shuffled(base)]
    cols = [g * BOX + c for g in shuffled(base) for c in shuffled(base)]
    nums = shuffled(list(range(1, SIDE + 1)))
    return [nums[pattern(r, c)] for r in rows for c in cols]


class _SudokuState:
    def __init__(self, board: Sequence[int]):
        self.board = list(board)
        self.row_used = [0] * SIDE
        self.col_used = [0] * SIDE
        self.box_used = [0] * SIDE
        for idx, value in enumerate(self.board):
            if value == 0:
                continue
            row, col = divmod(idx, SIDE)
            bit = 1 << value
            box = box_index(row, col)
            if self.row_used[row] & bit or self.col_used[col] & bit or self.box_used[box] & bit:
                raise ValueError("invalid Sudoku board")
            self.row_used[row] |= bit
            self.col_used[col] |= bit
            self.box_used[box] |= bit

    def allowed_mask(self, idx: int) -> int:
        row, col = divmod(idx, SIDE)
        used = self.row_used[row] | self.col_used[col] | self.box_used[box_index(row, col)]
        return DIGIT_MASK & ~used

    def place(self, idx: int, value: int) -> None:
        row, col = divmod(idx, SIDE)
        bit = 1 << value
        box = box_index(row, col)
        self.board[idx] = value
        self.row_used[row] |= bit
        self.col_used[col] |= bit
        self.box_used[box] |= bit

    def remove(self, idx: int, value: int) -> None:
        row, col = divmod(idx, SIDE)
        bit = 1 << value
        box = box_index(row, col)
        self.board[idx] = 0
        self.row_used[row] ^= bit
        self.col_used[col] ^= bit
        self.box_used[box] ^= bit


def count_solutions(board: Sequence[int], *, limit: int = 2) -> tuple[int, list[int] | None]:
    state = _SudokuState(board)
    first_solution: list[int] | None = None
    found = 0

    def backtrack() -> None:
        nonlocal found, first_solution
        if found >= limit:
            return

        best_idx = -1
        best_mask = 0
        best_count = 10
        for idx, value in enumerate(state.board):
            if value != 0:
                continue
            mask = state.allowed_mask(idx)
            count = mask.bit_count()
            if count == 0:
                return
            if count < best_count:
                best_idx = idx
                best_mask = mask
                best_count = count
                if count == 1:
                    break

        if best_idx < 0:
            found += 1
            if first_solution is None:
                first_solution = state.board[:]
            return

        for value in iter_bits(best_mask):
            state.place(best_idx, value)
            backtrack()
            state.remove(best_idx, value)
            if found >= limit:
                return

    backtrack()
    return found, first_solution


def is_valid_solution(board: Sequence[int]) -> bool:
    try:
        state = _SudokuState(board)
    except ValueError:
        return False
    if any(value == 0 for value in state.board):
        return False
    need = set(DIGITS)
    for row in range(SIDE):
        if set(state.board[row * SIDE : (row + 1) * SIDE]) != need:
            return False
    for col in range(SIDE):
        if {state.board[col + SIDE * row] for row in range(SIDE)} != need:
            return False
    for box_row in range(0, SIDE, BOX):
        for box_col in range(0, SIDE, BOX):
            cells = []
            for row in range(box_row, box_row + BOX):
                for col in range(box_col, box_col + BOX):
                    cells.append(state.board[row * SIDE + col])
            if set(cells) != need:
                return False
    return True


def is_clue_consistent(puzzle: str, board: str) -> bool:
    if len(puzzle) != CELL_COUNT or len(board) != CELL_COUNT:
        return False
    return all(p == "0" or p == b for p, b in zip(puzzle, board))


def blank_token_accuracy(puzzle: str, prediction: str, solution: str) -> float:
    total = 0
    correct = 0
    for p, pred, tgt in zip(puzzle, prediction, solution):
        if p != "0":
            continue
        total += 1
        if pred == tgt:
            correct += 1
    return float(correct / total) if total else 1.0


def generate_unique_puzzle(
    rng: random.Random,
    *,
    clues: int,
    max_restarts: int = 256,
) -> tuple[str, str]:
    if clues < 17 or clues > CELL_COUNT:
        raise ValueError("clues must be in [17, 81]")

    for _ in range(max_restarts):
        solved = randomized_solved_board(rng)
        puzzle = solved[:]
        positions = list(range(CELL_COUNT))
        rng.shuffle(positions)
        filled = CELL_COUNT
        for idx in positions:
            if filled <= clues:
                break
            saved = puzzle[idx]
            puzzle[idx] = 0
            count, _ = count_solutions(puzzle, limit=2)
            if count != 1:
                puzzle[idx] = saved
            else:
                filled -= 1
        if filled == clues:
            return board_to_string(puzzle), board_to_string(solved)
    raise RuntimeError(f"failed to generate unique puzzle with {clues} clues after {max_restarts} restarts")


def generate_example(split: str, clue_target: int, seed: int, index: int) -> SudokuExample:
    rng = random.Random((seed * 1_000_003) + (clue_target * 10_007) + index)
    puzzle, solution = generate_unique_puzzle(rng, clues=clue_target)
    item_id = f"{split}_c{clue_target}_{index:05d}"
    return SudokuExample(
        split=split,
        clue_count=clue_target,
        item_id=item_id,
        puzzle=puzzle,
        solution=solution,
    )


def write_manifest(path: Path, examples: Iterable[SudokuExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")


def load_manifest(path: Path) -> list[SudokuExample]:
    out: list[SudokuExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(SudokuExample(**row))
    return out


def grouped_by_clue(examples: Iterable[SudokuExample]) -> dict[int, list[SudokuExample]]:
    out: dict[int, list[SudokuExample]] = {}
    for ex in examples:
        out.setdefault(int(ex.clue_count), []).append(ex)
    return out


def validate_example(example: SudokuExample) -> None:
    if len(example.puzzle) != CELL_COUNT or len(example.solution) != CELL_COUNT:
        raise ValueError(f"bad example length for {example.item_id}")
    if clue_count(example.puzzle) != int(example.clue_count):
        raise ValueError(f"clue count mismatch for {example.item_id}")
    if not is_clue_consistent(example.puzzle, example.solution):
        raise ValueError(f"solution violates puzzle clues for {example.item_id}")
    if not is_valid_solution(string_to_board(example.solution)):
        raise ValueError(f"invalid solution for {example.item_id}")
    count, solved = count_solutions(string_to_board(example.puzzle), limit=2)
    if count != 1:
        raise ValueError(f"puzzle is not unique for {example.item_id}: count={count}")
    if solved is None or board_to_string(solved) != example.solution:
        raise ValueError(f"solver solution mismatch for {example.item_id}")


def self_test(seed: int = 1234) -> None:
    rng = random.Random(seed)
    for clues in (40, 36, 32, 28, 24):
        puzzle, solution = generate_unique_puzzle(rng, clues=clues, max_restarts=128)
        ex = SudokuExample(split="self_test", clue_count=clues, item_id=f"self_test_{clues}", puzzle=puzzle, solution=solution)
        validate_example(ex)
    print("sudoku9_unique_self_test_ok")


if __name__ == "__main__":
    self_test()
