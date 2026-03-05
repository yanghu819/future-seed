#!/usr/bin/env python3

"""NP-style logic probes for Future-Seed post-training.

Six synthetic-but-hard tasks are exposed under one trainer:
  - sat3: 3-SAT satisfiable? (Y/N)
  - tsp_mask: fill one masked city id in an optimal TSP tour template
  - zebra_mini: mini Zebra puzzle, predict house index for target pet
  - countdown: arithmetic reachability (Y/N)
  - graph_color: K-colorability decision (Y/N)
  - nqueens: partial N-Queens completion decision (Y/N)

Both tasks are converted to short-answer classification so they plug directly
into the existing quick->med orchestrator and report `val_tok_acc`.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

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


def _lit_to_text(lit: int) -> str:
    v = abs(lit)
    name = f"x{v:02d}"
    return name if lit > 0 else f"~{name}"


def _gen_sat3_formula(rng: random.Random, n_vars: int, n_clauses: int) -> List[Tuple[int, int, int]]:
    clauses: List[Tuple[int, int, int]] = []
    for _ in range(n_clauses):
        vars3 = rng.sample(range(1, n_vars + 1), 3)
        lits = []
        for v in vars3:
            lit = v if rng.random() < 0.5 else -v
            lits.append(lit)
        clauses.append((lits[0], lits[1], lits[2]))
    return clauses


def _clause_sat(clause: Tuple[int, int, int], assign_bits: int) -> bool:
    for lit in clause:
        idx = abs(lit) - 1
        val = ((assign_bits >> idx) & 1) == 1
        if lit < 0:
            val = not val
        if val:
            return True
    return False


def _is_sat_bruteforce(clauses: Sequence[Tuple[int, int, int]], n_vars: int) -> bool:
    for bits in range(1 << n_vars):
        ok = True
        for c in clauses:
            if not _clause_sat(c, bits):
                ok = False
                break
        if ok:
            return True
    return False


def _sat_prompt(
    clauses: Sequence[Tuple[int, int, int]],
    *,
    q_first: bool,
) -> str:
    cnf = " & ".join(["(" + " | ".join(_lit_to_text(x) for x in c) + ")" for c in clauses])
    if q_first:
        return (
            "Task: Determine whether this 3-SAT formula is satisfiable.\n"
            "Output Y (satisfiable) or N (unsatisfiable).\n\n"
            f"Formula:\n{cnf}\n\n"
            "Answer:"
        )
    return (
        f"3-SAT Formula:\n{cnf}\n\n"
        "Task: Determine whether the formula is satisfiable.\n"
        "Output Y or N.\n"
        "Answer:"
    )


def _build_sat_examples(
    *,
    tok: RWKVWorldTokenizer,
    n: int,
    seed: int,
    n_vars: int,
    n_clauses: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    q_first: bool,
    balance_labels: bool,
) -> List[Tuple[List[int], int]]:
    rng = random.Random(seed)
    out: List[Tuple[List[int], int]] = []
    pos_target = n // 2
    neg_target = n - pos_target
    pos = 0
    neg = 0
    tries = 0
    max_tries = max(10_000, n * 200)

    while len(out) < n and tries < max_tries:
        tries += 1
        clauses = _gen_sat3_formula(rng, n_vars=n_vars, n_clauses=n_clauses)
        sat = _is_sat_bruteforce(clauses, n_vars=n_vars)

        if balance_labels:
            if sat and pos >= pos_target:
                continue
            if (not sat) and neg >= neg_target:
                continue

        prompt = _sat_prompt(clauses, q_first=q_first)
        p_ids = tok.encode(prompt)
        if len(p_ids) > max_prompt_tokens:
            p_ids = p_ids[-max_prompt_tokens:]
        if len(p_ids) < min_prompt_tokens:
            continue

        ans = "Y" if sat else "N"
        a_ids = tok.encode(ans)
        if not a_ids:
            continue

        out.append((p_ids, int(a_ids[0])))
        if sat:
            pos += 1
        else:
            neg += 1

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} SAT examples (wanted {n}).")
    return out


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _held_karp_tour(coords: Sequence[Tuple[float, float]]) -> List[int]:
    n = len(coords)
    if n < 3:
        raise ValueError("n must be >= 3")
    dist = [[_dist(coords[i], coords[j]) for j in range(n)] for i in range(n)]

    # DP key: (mask, last), where mask encodes subset over nodes 1..n-1.
    dp: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for k in range(1, n):
        dp[(1 << (k - 1), k)] = (dist[0][k], 0)

    for sz in range(2, n):
        for subset in itertools.combinations(range(1, n), sz):
            bits = 0
            for x in subset:
                bits |= 1 << (x - 1)
            for k in subset:
                prev_bits = bits & ~(1 << (k - 1))
                best_cost = float("inf")
                best_prev = -1
                for m in subset:
                    if m == k:
                        continue
                    c = dp[(prev_bits, m)][0] + dist[m][k]
                    if c < best_cost:
                        best_cost = c
                        best_prev = m
                dp[(bits, k)] = (best_cost, best_prev)

    full = (1 << (n - 1)) - 1
    best_end = -1
    best_total = float("inf")
    for k in range(1, n):
        c = dp[(full, k)][0] + dist[k][0]
        if c < best_total:
            best_total = c
            best_end = k

    # Reconstruct 0 -> ... -> best_end -> 0
    rev = [best_end]
    bits = full
    cur = best_end
    while bits:
        _, prev = dp[(bits, cur)]
        bits = bits & ~(1 << (cur - 1))
        if prev == 0:
            break
        rev.append(prev)
        cur = prev
    rev.reverse()
    return [0] + rev + [0]


def _tsp_prompt(
    coords: Sequence[Tuple[float, float]],
    tour: Sequence[int],
    mask_pos: int,
    *,
    q_first: bool,
) -> Tuple[str, str]:
    templ = [str(x) for x in tour]
    ans = templ[mask_pos]
    templ[mask_pos] = "_"

    coord_lines = [f"{i}:({xy[0]:.3f},{xy[1]:.3f})" for i, xy in enumerate(coords)]
    coord_block = "\n".join(coord_lines)
    templ_text = " ".join(templ)
    n = len(coords)

    if q_first:
        prompt = (
            f"Task: Fill the missing city id in the optimal TSP tour (cities 0..{n-1}).\n"
            "Tour starts/ends at 0.\n\n"
            f"City coordinates:\n{coord_block}\n\n"
            f"Tour template:\n{templ_text}\n\n"
            "Answer:"
        )
    else:
        prompt = (
            f"City coordinates:\n{coord_block}\n\n"
            f"Tour template (optimal, starts/ends at 0):\n{templ_text}\n\n"
            f"Task: Fill '_' with the missing city id (0..{n-1}).\n"
            "Answer:"
        )
    return prompt, ans


def _build_tsp_examples(
    *,
    tok: RWKVWorldTokenizer,
    n: int,
    seed: int,
    n_cities: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    q_first: bool,
) -> List[Tuple[List[int], int]]:
    if n_cities > 10:
        # Keep answer token as single digit class for stable val_tok_acc.
        raise ValueError("n_cities must be <= 10 for single-token city ids.")

    rng = random.Random(seed)
    out: List[Tuple[List[int], int]] = []
    tries = 0
    max_tries = max(5000, n * 20)

    while len(out) < n and tries < max_tries:
        tries += 1
        coords = [(rng.random(), rng.random()) for _ in range(n_cities)]
        tour = _held_karp_tour(coords)
        if len(tour) != n_cities + 1:
            continue
        mask_pos = rng.randint(1, n_cities - 1)
        prompt, ans = _tsp_prompt(coords, tour, mask_pos, q_first=q_first)

        p_ids = tok.encode(prompt)
        if len(p_ids) > max_prompt_tokens:
            p_ids = p_ids[-max_prompt_tokens:]
        if len(p_ids) < min_prompt_tokens:
            continue

        a_ids = tok.encode(ans)
        if not a_ids:
            continue
        out.append((p_ids, int(a_ids[0])))

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} TSP examples (wanted {n}).")
    return out


def _zebra_pet_to_house(pet_at_house: Sequence[int], pet_idx: int) -> int:
    for h, p in enumerate(pet_at_house):
        if p == pet_idx:
            return h
    raise ValueError("pet not found")


def _zebra_color_to_house(color_at_house: Sequence[int], color_idx: int) -> int:
    for h, c in enumerate(color_at_house):
        if c == color_idx:
            return h
    raise ValueError("color not found")


def _zebra_sat(
    clue: Tuple[str, int, int],
    color_at_house: Sequence[int],
    pet_at_house: Sequence[int],
) -> bool:
    k, a, b = clue
    if k == "HC":
        # House a has color b.
        return color_at_house[a] == b
    if k == "HP":
        # House a has pet b.
        return pet_at_house[a] == b
    if k == "CP":
        # Color a and pet b are in the same house.
        return _zebra_color_to_house(color_at_house, a) == _zebra_pet_to_house(pet_at_house, b)
    if k == "PL":
        # Pet a is in a house strictly left of pet b.
        return _zebra_pet_to_house(pet_at_house, a) < _zebra_pet_to_house(pet_at_house, b)
    if k == "PA":
        # Pet a is adjacent to pet b.
        return abs(_zebra_pet_to_house(pet_at_house, a) - _zebra_pet_to_house(pet_at_house, b)) == 1
    raise ValueError(f"unknown clue kind: {k}")


def _zebra_clue_text(clue: Tuple[str, int, int], colors: Sequence[str], pets: Sequence[str]) -> str:
    k, a, b = clue
    if k == "HC":
        return f"House {a} has color {colors[b]}."
    if k == "HP":
        return f"House {a} has pet {pets[b]}."
    if k == "CP":
        return f"The {colors[a]} house keeps pet {pets[b]}."
    if k == "PL":
        return f"Pet {pets[a]} is left of pet {pets[b]}."
    if k == "PA":
        return f"Pet {pets[a]} is adjacent to pet {pets[b]}."
    raise ValueError(f"unknown clue kind: {k}")


def _zebra_prompt(
    *,
    clue_lines: Sequence[str],
    target_pet: str,
    colors: Sequence[str],
    pets: Sequence[str],
    n_houses: int,
    q_first: bool,
) -> str:
    rule_block = (
        f"Mini Zebra Puzzle. Houses are indexed 0..{n_houses - 1} from left to right.\n"
        f"Colors: {', '.join(colors)}.\n"
        f"Pets: {', '.join(pets)}.\n"
    )
    clues = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(clue_lines))
    q = f"Question: Which house index has pet {target_pet}? Output one digit."
    if q_first:
        return f"{q}\n\n{rule_block}\nClues:\n{clues}\n\nAnswer:"
    return f"{rule_block}\nClues:\n{clues}\n\n{q}\nAnswer:"


def _build_zebra_examples(
    *,
    tok: RWKVWorldTokenizer,
    n: int,
    seed: int,
    n_houses: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    q_first: bool,
) -> List[Tuple[List[int], int]]:
    if not (3 <= n_houses <= 5):
        raise ValueError("zebra_n_houses must be in [3,5]")
    if n_houses > 10:
        raise ValueError("zebra_n_houses must be <= 10 for single-digit answers")

    rng = random.Random(seed)
    out: List[Tuple[List[int], int]] = []
    tries = 0
    max_tries = max(20_000, n * 150)

    colors = list("RGBYO")[:n_houses]
    pets = list("CDFZH")[:n_houses]
    target_pet_idx = pets.index("Z") if "Z" in pets else (n_houses - 1)
    target_pet = pets[target_pet_idx]

    perms = list(itertools.permutations(range(n_houses)))
    worlds = [(tuple(c), tuple(p)) for c in perms for p in perms]

    while len(out) < n and tries < max_tries:
        tries += 1
        color_at_house = tuple(rng.choice(perms))
        pet_at_house = tuple(rng.choice(perms))

        clue_cands: List[Tuple[str, int, int]] = []
        for h in range(n_houses):
            clue_cands.append(("HC", h, int(color_at_house[h])))
            clue_cands.append(("HP", h, int(pet_at_house[h])))
        for c in range(n_houses):
            h = _zebra_color_to_house(color_at_house, c)
            clue_cands.append(("CP", c, int(pet_at_house[h])))
        for a in range(n_houses):
            for b in range(n_houses):
                if a == b:
                    continue
                ha = _zebra_pet_to_house(pet_at_house, a)
                hb = _zebra_pet_to_house(pet_at_house, b)
                if ha < hb:
                    clue_cands.append(("PL", a, b))
                if a < b and abs(ha - hb) == 1:
                    clue_cands.append(("PA", a, b))

        rng.shuffle(clue_cands)
        alive = worlds
        chosen: List[Tuple[str, int, int]] = []
        for clue in clue_cands:
            new_alive = [w for w in alive if _zebra_sat(clue, w[0], w[1])]
            if len(new_alive) < len(alive):
                chosen.append(clue)
                alive = new_alive
            if len(alive) == 1:
                break

        if len(alive) != 1:
            continue

        clue_lines = [_zebra_clue_text(c, colors, pets) for c in chosen]
        prompt = _zebra_prompt(
            clue_lines=clue_lines,
            target_pet=target_pet,
            colors=colors,
            pets=pets,
            n_houses=n_houses,
            q_first=q_first,
        )
        p_ids = tok.encode(prompt)
        if len(p_ids) > max_prompt_tokens:
            p_ids = p_ids[-max_prompt_tokens:]
        if len(p_ids) < min_prompt_tokens:
            continue

        ans = str(_zebra_pet_to_house(pet_at_house, target_pet_idx))
        a_ids = tok.encode(ans)
        if not a_ids:
            continue
        out.append((p_ids, int(a_ids[0])))

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} Zebra examples (wanted {n}).")
    return out


def _frac_key(nums: Sequence[Fraction]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((x.numerator, x.denominator) for x in nums))


def _countdown_reachable(nums: Sequence[int], target: int) -> bool:
    start = tuple(Fraction(int(x), 1) for x in nums)
    tgt = Fraction(int(target), 1)

    @functools.lru_cache(maxsize=200_000)
    def dfs(key: Tuple[Tuple[int, int], ...]) -> bool:
        vals = [Fraction(a, b) for (a, b) in key]
        if len(vals) == 1:
            return vals[0] == tgt
        m = len(vals)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = vals[i], vals[j]
                rest = [vals[k] for k in range(m) if k != i and k != j]
                cands = [a + b, a - b, b - a, a * b]
                if b != 0:
                    cands.append(a / b)
                if a != 0:
                    cands.append(b / a)
                seen = set()
                for c in cands:
                    ck = (c.numerator, c.denominator)
                    if ck in seen:
                        continue
                    seen.add(ck)
                    next_key = _frac_key(rest + [c])
                    if dfs(next_key):
                        return True
        return False

    return dfs(_frac_key(start))


def _countdown_prompt(
    nums: Sequence[int],
    target: int,
    *,
    q_first: bool,
) -> str:
    num_text = " ".join(str(int(x)) for x in nums)
    rules = (
        "Rules: use each number exactly once with + - * / and parentheses.\n"
        "Intermediate values may be negative or fractional.\n"
        "Output Y if target can be reached, else N."
    )
    q = "Question: Is this instance solvable?"
    if q_first:
        return f"{q}\n{rules}\n\nNumbers: {num_text}\nTarget: {target}\nAnswer:"
    return f"Numbers: {num_text}\nTarget: {target}\n\n{rules}\n{q}\nAnswer:"


def _build_countdown_examples(
    *,
    tok: RWKVWorldTokenizer,
    n: int,
    seed: int,
    n_nums: int,
    num_max: int,
    target_min: int,
    target_max: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    q_first: bool,
    balance_labels: bool,
) -> List[Tuple[List[int], int]]:
    if n_nums < 3:
        raise ValueError("count_n_nums must be >= 3")
    rng = random.Random(seed)
    out: List[Tuple[List[int], int]] = []
    tries = 0
    max_tries = max(20_000, n * 300)
    pos_target = n // 2
    neg_target = n - pos_target
    pos = 0
    neg = 0

    while len(out) < n and tries < max_tries:
        tries += 1
        nums = [rng.randint(1, num_max) for _ in range(n_nums)]
        target = rng.randint(target_min, target_max)
        sat = _countdown_reachable(nums, target)

        if balance_labels:
            if sat and pos >= pos_target:
                continue
            if (not sat) and neg >= neg_target:
                continue

        prompt = _countdown_prompt(nums, target, q_first=q_first)
        p_ids = tok.encode(prompt)
        if len(p_ids) < min_prompt_tokens:
            # Countdown prompts can be short; append deterministic rule recap
            # to satisfy orchestrator prompt-length floors.
            filler = (
                "\nRule recap: apply + - * / with parentheses, use each given number exactly once, "
                "and answer Y if target is reachable, else N."
            )
            for _ in range(12):
                if len(p_ids) >= min_prompt_tokens:
                    break
                prompt += filler
                p_ids = tok.encode(prompt)
        if len(p_ids) > max_prompt_tokens:
            p_ids = p_ids[-max_prompt_tokens:]
        if len(p_ids) < min_prompt_tokens:
            continue
        ans = "Y" if sat else "N"
        a_ids = tok.encode(ans)
        if not a_ids:
            continue
        out.append((p_ids, int(a_ids[0])))
        if sat:
            pos += 1
        else:
            neg += 1

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} Countdown examples (wanted {n}).")
    return out


def _graph_colorable(
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
    n_colors: int,
) -> bool:
    if n_colors <= 0:
        return False
    if n_nodes <= 0:
        return False
    adj = [set() for _ in range(n_nodes)]
    for u, v in edges:
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            return False
        if u == v:
            return False
        adj[u].add(v)
        adj[v].add(u)

    order = sorted(range(n_nodes), key=lambda x: len(adj[x]), reverse=True)
    color = [-1 for _ in range(n_nodes)]

    def dfs(idx: int) -> bool:
        if idx >= n_nodes:
            return True
        node = order[idx]
        used = {color[v] for v in adj[node] if color[v] >= 0}
        for c in range(n_colors):
            if c in used:
                continue
            color[node] = c
            if dfs(idx + 1):
                return True
            color[node] = -1
        return False

    return dfs(0)


def _graph_color_prompt(
    *,
    n_nodes: int,
    n_colors: int,
    edges: Sequence[Tuple[int, int]],
    q_first: bool,
) -> str:
    edge_text = ", ".join(f"({u},{v})" for (u, v) in edges) if edges else "(none)"
    rules = (
        f"Graph has nodes 0..{n_nodes - 1}. Undirected edges: {edge_text}.\n"
        f"Can nodes be colored with {n_colors} colors so every edge has different endpoint colors?\n"
        "Output Y if possible, else N."
    )
    if q_first:
        return f"Task: Graph coloring decision.\n{rules}\nAnswer:"
    return f"{rules}\nTask: graph-colorability decision.\nAnswer:"


def _build_graph_color_examples(
    *,
    tok: RWKVWorldTokenizer,
    n: int,
    seed: int,
    n_nodes: int,
    n_colors: int,
    edge_prob_min: float,
    edge_prob_max: float,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    q_first: bool,
    balance_labels: bool,
) -> List[Tuple[List[int], int]]:
    if n_nodes < 4:
        raise ValueError("gc_n_nodes must be >= 4")
    if n_colors < 2:
        raise ValueError("gc_n_colors must be >= 2")
    pmin = max(0.02, min(float(edge_prob_min), 0.98))
    pmax = max(0.02, min(float(edge_prob_max), 0.98))
    if pmax < pmin:
        pmin, pmax = pmax, pmin

    rng = random.Random(seed)
    out: List[Tuple[List[int], int]] = []
    tries = 0
    max_tries = max(30_000, n * 250)
    pos_target = n // 2
    neg_target = n - pos_target
    pos = 0
    neg = 0

    while len(out) < n and tries < max_tries:
        tries += 1
        p = rng.uniform(pmin, pmax)
        edges: List[Tuple[int, int]] = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < p:
                    edges.append((i, j))

        sat = _graph_colorable(n_nodes=n_nodes, edges=edges, n_colors=n_colors)
        if balance_labels:
            if sat and pos >= pos_target:
                continue
            if (not sat) and neg >= neg_target:
                continue

        prompt = _graph_color_prompt(
            n_nodes=n_nodes,
            n_colors=n_colors,
            edges=edges,
            q_first=q_first,
        )
        p_ids = tok.encode(prompt)
        if len(p_ids) > max_prompt_tokens:
            p_ids = p_ids[-max_prompt_tokens:]
        if len(p_ids) < min_prompt_tokens:
            continue

        ans = "Y" if sat else "N"
        a_ids = tok.encode(ans)
        if not a_ids:
            continue
        out.append((p_ids, int(a_ids[0])))
        if sat:
            pos += 1
        else:
            neg += 1

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} GraphColor examples (wanted {n}).")
    return out


def _nqueens_satisfiable(
    *,
    n: int,
    fixed: Dict[int, int],
) -> bool:
    if n < 4:
        return False
    for r, c in fixed.items():
        if not (0 <= r < n and 0 <= c < n):
            return False

    used_cols = set()
    used_d1 = set()
    used_d2 = set()

    def dfs(row: int) -> bool:
        if row >= n:
            return True
        cols = [fixed[row]] if row in fixed else list(range(n))
        for c in cols:
            d1 = row - c
            d2 = row + c
            if c in used_cols or d1 in used_d1 or d2 in used_d2:
                continue
            used_cols.add(c)
            used_d1.add(d1)
            used_d2.add(d2)
            if dfs(row + 1):
                return True
            used_cols.remove(c)
            used_d1.remove(d1)
            used_d2.remove(d2)
        return False

    return dfs(0)


def _nqueens_prompt(
    *,
    n: int,
    fixed: Dict[int, int],
    q_first: bool,
) -> str:
    fixed_lines = [f"row {r} -> col {fixed[r]}" for r in sorted(fixed.keys())]
    fixed_block = "; ".join(fixed_lines) if fixed_lines else "(none)"
    rules = (
        f"N-Queens on {n}x{n} board (rows/cols 0..{n - 1}).\n"
        "Need exactly one queen per row and per column, and no shared diagonal.\n"
        f"Pre-placed queens: {fixed_block}\n"
        "Can this be completed to a full valid arrangement? Output Y or N."
    )
    if q_first:
        return f"Task: N-Queens completion decision.\n{rules}\nAnswer:"
    return f"{rules}\nTask: decide satisfiable completion.\nAnswer:"


def _build_nqueens_examples(
    *,
    tok: RWKVWorldTokenizer,
    n: int,
    seed: int,
    nq_n: int,
    nq_fixed_min: int,
    nq_fixed_max: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    q_first: bool,
    balance_labels: bool,
) -> List[Tuple[List[int], int]]:
    if nq_n < 4:
        raise ValueError("nq_n must be >= 4")
    fmin = max(1, int(nq_fixed_min))
    fmax = max(fmin, int(nq_fixed_max))
    fmax = min(fmax, nq_n)

    rng = random.Random(seed)
    out: List[Tuple[List[int], int]] = []
    tries = 0
    max_tries = max(30_000, n * 250)
    pos_target = n // 2
    neg_target = n - pos_target
    pos = 0
    neg = 0

    while len(out) < n and tries < max_tries:
        tries += 1
        k = rng.randint(fmin, fmax)
        rows = rng.sample(range(nq_n), k)
        fixed = {r: rng.randrange(nq_n) for r in rows}
        sat = _nqueens_satisfiable(n=nq_n, fixed=fixed)

        if balance_labels:
            if sat and pos >= pos_target:
                continue
            if (not sat) and neg >= neg_target:
                continue

        prompt = _nqueens_prompt(n=nq_n, fixed=fixed, q_first=q_first)
        p_ids = tok.encode(prompt)
        if len(p_ids) < min_prompt_tokens:
            # N-Queens prompts are naturally short; extend with explicit rule recap
            # so orchestrator min-prompt gates do not starve data generation.
            filler = (
                "\nRule recap: exactly one queen per row; exactly one queen per column; "
                "no two queens can share a diagonal."
            )
            for _ in range(12):
                if len(p_ids) >= min_prompt_tokens:
                    break
                prompt += filler
                p_ids = tok.encode(prompt)
        if len(p_ids) > max_prompt_tokens:
            p_ids = p_ids[-max_prompt_tokens:]
        if len(p_ids) < min_prompt_tokens:
            continue

        ans = "Y" if sat else "N"
        a_ids = tok.encode(ans)
        if not a_ids:
            continue
        out.append((p_ids, int(a_ids[0])))
        if sat:
            pos += 1
        else:
            neg += 1

    if len(out) < n:
        raise RuntimeError(f"Only built {len(out)} NQueens examples (wanted {n}).")
    return out


def _build_examples(
    *,
    task: str,
    tok: RWKVWorldTokenizer,
    n: int,
    seed: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    q_first: bool,
    sat_n_vars: int,
    sat_n_clauses: int,
    sat_balance_labels: bool,
    tsp_n_cities: int,
    zebra_n_houses: int,
    count_n_nums: int,
    count_num_max: int,
    count_target_min: int,
    count_target_max: int,
    count_balance_labels: bool,
    gc_n_nodes: int,
    gc_n_colors: int,
    gc_edge_prob_min: float,
    gc_edge_prob_max: float,
    gc_balance_labels: bool,
    nq_n: int,
    nq_fixed_min: int,
    nq_fixed_max: int,
    nq_balance_labels: bool,
) -> List[Tuple[List[int], int]]:
    if task == "sat3":
        return _build_sat_examples(
            tok=tok,
            n=n,
            seed=seed,
            n_vars=sat_n_vars,
            n_clauses=sat_n_clauses,
            max_prompt_tokens=max_prompt_tokens,
            min_prompt_tokens=min_prompt_tokens,
            q_first=q_first,
            balance_labels=sat_balance_labels,
        )
    if task == "tsp_mask":
        return _build_tsp_examples(
            tok=tok,
            n=n,
            seed=seed,
            n_cities=tsp_n_cities,
            max_prompt_tokens=max_prompt_tokens,
            min_prompt_tokens=min_prompt_tokens,
            q_first=q_first,
        )
    if task == "zebra_mini":
        return _build_zebra_examples(
            tok=tok,
            n=n,
            seed=seed,
            n_houses=zebra_n_houses,
            max_prompt_tokens=max_prompt_tokens,
            min_prompt_tokens=min_prompt_tokens,
            q_first=q_first,
        )
    if task == "countdown":
        return _build_countdown_examples(
            tok=tok,
            n=n,
            seed=seed,
            n_nums=count_n_nums,
            num_max=count_num_max,
            target_min=count_target_min,
            target_max=count_target_max,
            max_prompt_tokens=max_prompt_tokens,
            min_prompt_tokens=min_prompt_tokens,
            q_first=q_first,
            balance_labels=count_balance_labels,
        )
    if task == "graph_color":
        return _build_graph_color_examples(
            tok=tok,
            n=n,
            seed=seed,
            n_nodes=gc_n_nodes,
            n_colors=gc_n_colors,
            edge_prob_min=gc_edge_prob_min,
            edge_prob_max=gc_edge_prob_max,
            max_prompt_tokens=max_prompt_tokens,
            min_prompt_tokens=min_prompt_tokens,
            q_first=q_first,
            balance_labels=gc_balance_labels,
        )
    if task == "nqueens":
        return _build_nqueens_examples(
            tok=tok,
            n=n,
            seed=seed,
            nq_n=nq_n,
            nq_fixed_min=nq_fixed_min,
            nq_fixed_max=nq_fixed_max,
            max_prompt_tokens=max_prompt_tokens,
            min_prompt_tokens=min_prompt_tokens,
            q_first=q_first,
            balance_labels=nq_balance_labels,
        )
    raise ValueError(task)


@torch.no_grad()
def cls_acc_from_logits(logits: torch.Tensor, tgt: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == tgt).float().mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["no_fs", "prompt_fs"], default="no_fs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_data_seed", type=int, default=0)
    ap.add_argument("--val_data_seed", type=int, default=1234)

    ap.add_argument("--task", choices=["sat3", "tsp_mask", "zebra_mini", "countdown", "graph_color", "nqueens"], default="sat3")
    ap.add_argument("--q_first", action="store_true", help="Instruction before main content.")

    ap.add_argument("--sat_n_vars", type=int, default=12)
    ap.add_argument("--sat_n_clauses", type=int, default=50)
    ap.add_argument("--sat_balance_labels", action="store_true")

    ap.add_argument("--tsp_n_cities", type=int, default=10)
    ap.add_argument("--zebra_n_houses", type=int, default=4)
    ap.add_argument("--count_n_nums", type=int, default=5)
    ap.add_argument("--count_num_max", type=int, default=10)
    ap.add_argument("--count_target_min", type=int, default=10)
    ap.add_argument("--count_target_max", type=int, default=80)
    ap.add_argument("--count_balance_labels", action="store_true")
    ap.add_argument("--gc_n_nodes", type=int, default=8)
    ap.add_argument("--gc_n_colors", type=int, default=3)
    ap.add_argument("--gc_edge_prob_min", type=float, default=0.25)
    ap.add_argument("--gc_edge_prob_max", type=float, default=0.55)
    ap.add_argument("--gc_balance_labels", action="store_true")
    ap.add_argument("--nq_n", type=int, default=8)
    ap.add_argument("--nq_fixed_min", type=int, default=2)
    ap.add_argument("--nq_fixed_max", type=int, default=4)
    ap.add_argument("--nq_balance_labels", action="store_true")

    ap.add_argument("--fs_variant", choices=["scalar", "head"], default="scalar")
    ap.add_argument("--alpha_head_lr", type=float, default=None)
    ap.add_argument("--alpha_head_init", type=float, default=None)

    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--max_prompt_tokens", type=int, default=1536)
    ap.add_argument("--min_prompt_tokens", type=int, default=256)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = RWKVWorldTokenizer(args.vocab)
    pad_id = tok.eot_id

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_meta = {
        "task": args.task,
        "q_first": bool(args.q_first),
        "sat_n_vars": int(args.sat_n_vars),
        "sat_n_clauses": int(args.sat_n_clauses),
        "sat_balance_labels": bool(args.sat_balance_labels),
        "tsp_n_cities": int(args.tsp_n_cities),
        "zebra_n_houses": int(args.zebra_n_houses),
        "count_n_nums": int(args.count_n_nums),
        "count_num_max": int(args.count_num_max),
        "count_target_min": int(args.count_target_min),
        "count_target_max": int(args.count_target_max),
        "count_balance_labels": bool(args.count_balance_labels),
        "gc_n_nodes": int(args.gc_n_nodes),
        "gc_n_colors": int(args.gc_n_colors),
        "gc_edge_prob_min": float(args.gc_edge_prob_min),
        "gc_edge_prob_max": float(args.gc_edge_prob_max),
        "gc_balance_labels": bool(args.gc_balance_labels),
        "nq_n": int(args.nq_n),
        "nq_fixed_min": int(args.nq_fixed_min),
        "nq_fixed_max": int(args.nq_fixed_max),
        "nq_balance_labels": bool(args.nq_balance_labels),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "min_prompt_tokens": int(args.min_prompt_tokens),
        "train_data_seed": int(args.train_data_seed),
        "val_data_seed": int(args.val_data_seed),
        "vocab": str(args.vocab),
    }
    cache_key = hashlib.md5(json.dumps(cache_meta, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / f"np_{args.task}_tok_{cache_key}.pt"

    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu")
        train_ex = data["train_ex"]
        val_ex = data["val_ex"]
        print(f"Loaded cache: {cache_path}")
    else:
        print("Building synthetic data...")
        train_ex = _build_examples(
            task=args.task,
            tok=tok,
            n=int(args.n_train),
            seed=int(args.train_data_seed),
            max_prompt_tokens=int(args.max_prompt_tokens),
            min_prompt_tokens=int(args.min_prompt_tokens),
            q_first=bool(args.q_first),
            sat_n_vars=int(args.sat_n_vars),
            sat_n_clauses=int(args.sat_n_clauses),
            sat_balance_labels=bool(args.sat_balance_labels),
            tsp_n_cities=int(args.tsp_n_cities),
            zebra_n_houses=int(args.zebra_n_houses),
            count_n_nums=int(args.count_n_nums),
            count_num_max=int(args.count_num_max),
            count_target_min=int(args.count_target_min),
            count_target_max=int(args.count_target_max),
            count_balance_labels=bool(args.count_balance_labels),
            gc_n_nodes=int(args.gc_n_nodes),
            gc_n_colors=int(args.gc_n_colors),
            gc_edge_prob_min=float(args.gc_edge_prob_min),
            gc_edge_prob_max=float(args.gc_edge_prob_max),
            gc_balance_labels=bool(args.gc_balance_labels),
            nq_n=int(args.nq_n),
            nq_fixed_min=int(args.nq_fixed_min),
            nq_fixed_max=int(args.nq_fixed_max),
            nq_balance_labels=bool(args.nq_balance_labels),
        )
        val_ex = _build_examples(
            task=args.task,
            tok=tok,
            n=int(args.n_val),
            seed=int(args.val_data_seed),
            max_prompt_tokens=int(args.max_prompt_tokens),
            min_prompt_tokens=int(args.min_prompt_tokens),
            q_first=bool(args.q_first),
            sat_n_vars=int(args.sat_n_vars),
            sat_n_clauses=int(args.sat_n_clauses),
            sat_balance_labels=bool(args.sat_balance_labels),
            tsp_n_cities=int(args.tsp_n_cities),
            zebra_n_houses=int(args.zebra_n_houses),
            count_n_nums=int(args.count_n_nums),
            count_num_max=int(args.count_num_max),
            count_target_min=int(args.count_target_min),
            count_target_max=int(args.count_target_max),
            count_balance_labels=bool(args.count_balance_labels),
            gc_n_nodes=int(args.gc_n_nodes),
            gc_n_colors=int(args.gc_n_colors),
            gc_edge_prob_min=float(args.gc_edge_prob_min),
            gc_edge_prob_max=float(args.gc_edge_prob_max),
            gc_balance_labels=bool(args.gc_balance_labels),
            nq_n=int(args.nq_n),
            nq_fixed_min=int(args.nq_fixed_min),
            nq_fixed_max=int(args.nq_fixed_max),
            nq_balance_labels=bool(args.nq_balance_labels),
        )
        torch.save({"train_ex": train_ex, "val_ex": val_ex, "meta": cache_meta}, cache_path)
        print(f"Saved cache: {cache_path}")

    run_root = Path(args.run_dir) / time.strftime("%Y%m%d-%H%M%S") / f"np_{args.task}_sft" / args.mode
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

    def sample_batch(
        examples: List[Tuple[List[int], int]],
        *,
        rng: random.Random,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ps, ys = [], []
        for _ in range(int(args.bsz)):
            p_ids, y = rng.choice(examples)
            ps.append(p_ids)
            ys.append(y)
        p = pad_left(ps, pad_id=pad_id, multiple=16).to(device)
        y = torch.tensor(ys, device=device, dtype=torch.long)
        return p, y

    t0 = time.time()
    step = 0
    while True:
        if time.time() - t0 > float(args.time_budget_sec):
            break
        if args.max_steps and step >= int(args.max_steps):
            break

        opt.zero_grad(set_to_none=True)
        prompt_ids, y = sample_batch(train_ex, rng=train_rng)
        use_fs = args.mode == "prompt_fs"
        hidden, _ = model(
            prompt_ids,
            future_seed=use_fs,
            fs_alpha=(alpha if use_fs else None),
            fs_alpha_head=(alpha_head if use_fs else None),
            seed_scale=float(args.seed_scale),
            fs_layer_start=int(args.fs_layer_start),
            fs_alpha_schedule=str(args.fs_alpha_schedule),
            fs_alpha_min=float(args.fs_alpha_min),
            fs_alpha_max=float(args.fs_alpha_max),
            fs_norm=bool(args.fs_norm),
            fs_clip=float(args.fs_clip),
            fs_detach=bool(args.fs_detach),
            return_states=False,
        )
        logits = model.project(hidden[:, -1, :])
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        if (step % int(args.eval_every)) == 0:
            model.eval()
            with torch.no_grad():
                val_losses: List[float] = []
                val_accs: List[float] = []
                for _ in range(int(args.val_batches)):
                    vp, vy = sample_batch(val_ex, rng=val_rng)
                    vhidden, _ = model(
                        vp,
                        future_seed=(args.mode == "prompt_fs"),
                        fs_alpha=(alpha if args.mode == "prompt_fs" else None),
                        fs_alpha_head=(alpha_head if args.mode == "prompt_fs" else None),
                        seed_scale=float(args.seed_scale),
                        fs_layer_start=int(args.fs_layer_start),
                        fs_alpha_schedule=str(args.fs_alpha_schedule),
                        fs_alpha_min=float(args.fs_alpha_min),
                        fs_alpha_max=float(args.fs_alpha_max),
                        fs_norm=bool(args.fs_norm),
                        fs_clip=float(args.fs_clip),
                        fs_detach=bool(args.fs_detach),
                        return_states=False,
                    )
                    vlogits = model.project(vhidden[:, -1, :])
                    vloss = F.cross_entropy(vlogits, vy)
                    vacc = cls_acc_from_logits(vlogits, vy)
                    val_losses.append(float(vloss))
                    val_accs.append(float(vacc))

                v_acc = sum(val_accs) / len(val_accs)
                rec = {
                    "t": round(time.time() - t0, 2),
                    "step": step,
                    "train_loss": float(loss),
                    "val_loss": sum(val_losses) / len(val_losses),
                    "val_tok_acc": v_acc,
                    "val_cls_acc": v_acc,
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
