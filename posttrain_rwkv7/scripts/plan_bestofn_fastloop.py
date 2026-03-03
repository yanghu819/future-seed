#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load_queue_builder_module():
    mod_path = ROOT / "scripts" / "rebuild_fastloop_queues_broad.py"
    spec = importlib.util.spec_from_file_location("rebuild_fastloop_queues_broad", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed loading {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _task_key(task_name: str) -> str:
    return task_name.split("_seed")[0]


def _iter_record_rows(records_dir: Path, round_from: int, round_to: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
    for fp in sorted(records_dir.glob("_round*_fastdiscover_records.jsonl")):
        m = re.search(r"_round(\d+)_", fp.name)
        if not m:
            continue
        rid = int(m.group(1))
        if rid < round_from or rid > round_to:
            continue
        for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
            except Exception:
                continue
            yield rid, row


def _build_task_stats(records_dir: Path, round_from: int, round_to: int, promote_pp: float) -> Dict[str, Dict[str, float]]:
    rows = list(_iter_record_rows(records_dir, round_from, round_to))

    quick_base: Dict[Tuple[int, str], float] = {}
    quick_best: Dict[Tuple[int, str], float] = {}
    quick_fail_base: Dict[str, int] = defaultdict(int)
    quick_base_total: Dict[str, int] = defaultdict(int)

    med_base: Dict[Tuple[int, str], float] = {}
    med_fs: Dict[Tuple[int, str], List[float]] = defaultdict(list)

    for rid, r in rows:
        stage = r.get("stage")
        status = r.get("status")
        task = str(r.get("task", ""))
        if not task:
            continue
        tkey = _task_key(task)
        cfg = str(r.get("config", ""))

        if stage == "quick" and cfg == "baseline":
            quick_base_total[tkey] += 1
            if status in {"ok", "skip_reuse"} and r.get("best_val_tok_acc") is not None:
                quick_base[(rid, task)] = float(r["best_val_tok_acc"])
            elif status in {"fail"}:
                quick_fail_base[tkey] += 1

        if stage == "quick" and cfg != "baseline" and status in {"ok", "skip_reuse"} and r.get("best_val_tok_acc") is not None:
            key = (rid, task)
            acc = float(r["best_val_tok_acc"])
            if key not in quick_best or acc > quick_best[key]:
                quick_best[key] = acc

        if stage == "med" and cfg == "baseline" and status in {"ok", "skip_reuse"} and r.get("best_val_tok_acc") is not None:
            med_base[(rid, task)] = float(r["best_val_tok_acc"])

        if stage == "med" and cfg != "baseline" and status in {"ok", "skip_reuse"} and r.get("best_val_tok_acc") is not None:
            med_fs[(rid, task)].append(float(r["best_val_tok_acc"]))

    quick_deltas: Dict[str, List[float]] = defaultdict(list)
    for key, b in quick_base.items():
        if key in quick_best:
            task = key[1]
            tkey = _task_key(task)
            quick_deltas[tkey].append((quick_best[key] - b) * 100.0)

    med_deltas: Dict[str, List[float]] = defaultdict(list)
    for key, b in med_base.items():
        vals = med_fs.get(key)
        if not vals:
            continue
        task = key[1]
        tkey = _task_key(task)
        med_deltas[tkey].append((max(vals) - b) * 100.0)

    tasks = sorted(set(quick_base_total) | set(quick_deltas) | set(med_deltas))
    out: Dict[str, Dict[str, float]] = {}
    for t in tasks:
        qv = quick_deltas.get(t, [])
        mv = med_deltas.get(t, [])
        qn = len(qv)
        mn = len(mv)
        out[t] = {
            "quick_n": float(qn),
            "quick_promote_rate": (sum(1 for x in qv if x >= promote_pp) / qn) if qn else 0.0,
            "quick_mean_pp": (sum(qv) / qn) if qn else 0.0,
            "med_n": float(mn),
            "med_pos_rate": (sum(1 for x in mv if x > 0) / mn) if mn else 0.0,
            "med_mean_pp": (sum(mv) / mn) if mn else 0.0,
            "med_max_pp": max(mv) if mv else 0.0,
            "quick_base_fail_rate": (
                quick_fail_base.get(t, 0) / quick_base_total[t]
            )
            if quick_base_total.get(t, 0)
            else 0.0,
        }
    return out


def _task_score(stat: Dict[str, float]) -> float:
    # Weighted for fast conversion to useful med-positive tasks.
    score = (
        2.4 * stat.get("med_mean_pp", 0.0)
        + 3.2 * stat.get("med_pos_rate", 0.0)
        + 1.2 * stat.get("quick_promote_rate", 0.0)
        + 0.4 * stat.get("quick_mean_pp", 0.0)
        - 2.8 * stat.get("quick_base_fail_rate", 0.0)
    )
    med_n = int(stat.get("med_n", 0.0))
    quick_n = int(stat.get("quick_n", 0.0))
    med_mean = stat.get("med_mean_pp", 0.0)
    quick_promote = stat.get("quick_promote_rate", 0.0)
    if med_n >= 2 and med_mean < 0:
        score -= 4.0
    if quick_n >= 6 and quick_promote == 0.0:
        score -= 3.0
    if med_n == 0 and quick_n >= 6:
        score -= 2.0
    return score


def _score_profile(
    profile_name: str,
    profiles: Dict[str, Dict[str, Any]],
    task_stats: Dict[str, Dict[str, float]],
    kill_tasks: List[str],
) -> Dict[str, Any]:
    cycle = profiles[profile_name]["cycle"]
    weighted_sum = 0.0
    weighted_n = 0.0
    task_breakdown = []

    for novelty_tier, pair in cycle:
        w = 1.0 if novelty_tier == "new" else 0.65
        for t in pair:
            st = task_stats.get(
                t,
                {
                    "quick_promote_rate": 0.0,
                    "quick_mean_pp": 0.0,
                    "med_pos_rate": 0.0,
                    "med_mean_pp": 0.0,
                    "quick_base_fail_rate": 0.0,
                    "med_n": 0.0,
                    "quick_n": 0.0,
                },
            )
            s = _task_score(st)
            if t in kill_tasks:
                s -= 8.0
            weighted_sum += w * s
            weighted_n += w
            task_breakdown.append({"task": t, "novelty_tier": novelty_tier, "task_score": s, **st})

    profile_score = weighted_sum / weighted_n if weighted_n else -1e9
    return {
        "profile": profile_name,
        "score": profile_score,
        "task_breakdown": task_breakdown,
        "search_mix": profiles[profile_name].get("search_mix", ""),
    }


def _infer_next_seed_start(results_dir: Path, seed_defaults: Dict[str, int]) -> Dict[str, int]:
    mx: Dict[str, int] = {k: v - 1 for k, v in seed_defaults.items()}
    for fp in sorted(results_dir.glob("_search_queue_round*_fastloop.json")):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for rd in payload.get("rounds", []):
            for t in rd.get("tasks", []):
                name = str(t.get("name", ""))
                if "_seed" not in name:
                    continue
                key = _task_key(name)
                if key not in mx:
                    continue
                try:
                    seed = int(t.get("seed", -1))
                except Exception:
                    continue
                if seed > mx[key]:
                    mx[key] = seed
    return {k: mx[k] + 1 for k in mx}


def _run_queue_builder(
    profile: str,
    start_round: int,
    end_round: int,
    block_size: int,
    seed_start: Dict[str, int],
) -> None:
    cmd = [
        "python3",
        "scripts/rebuild_fastloop_queues_broad.py",
        "--start_round",
        str(start_round),
        "--end_round",
        str(end_round),
        "--block_size",
        str(block_size),
        "--profile",
        profile,
        "--seed_arc_mc",
        str(seed_start["arc_mc"]),
        "--seed_protein_ss",
        str(seed_start["protein_ss"]),
        "--seed_hotpot",
        str(seed_start["hotpot"]),
        "--seed_mbpp",
        str(seed_start["mbpp"]),
        "--seed_mbpp_longctx",
        str(seed_start["mbpp_longctx"]),
        "--seed_squad",
        str(seed_start["squad"]),
        "--seed_wiki",
        str(seed_start["wiki"]),
        "--seed_protein_contact",
        str(seed_start["protein_contact"]),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_round", type=int, required=True)
    ap.add_argument("--end_round", type=int, required=True)
    ap.add_argument("--block_size", type=int, default=8)
    ap.add_argument("--history_from", type=int, default=409)
    ap.add_argument("--history_to", type=int, default=99999)
    ap.add_argument("--records_dir", type=str, default=str(RESULTS))
    ap.add_argument(
        "--candidates",
        type=str,
        default="ruthless_v6,roi_explore_v5,kernel_v3",
        help="comma-separated profiles",
    )
    ap.add_argument(
        "--kill_tasks",
        type=str,
        default="protein_contact,hotpot",
        help="comma-separated task keys to aggressively suppress",
    )
    ap.add_argument("--promote_pp", type=float, default=0.8)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    mod = _load_queue_builder_module()
    profiles: Dict[str, Dict[str, Any]] = mod.PROFILES
    seed_defaults: Dict[str, int] = mod.SEED_START

    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
    for c in candidates:
        if c not in profiles:
            raise ValueError(f"unknown profile: {c}")

    records_dir = Path(args.records_dir)
    stats = _build_task_stats(records_dir, args.history_from, args.history_to, args.promote_pp)
    kill_tasks = [x.strip() for x in args.kill_tasks.split(",") if x.strip()]

    scored = [_score_profile(c, profiles, stats, kill_tasks) for c in candidates]
    scored.sort(key=lambda x: x["score"], reverse=True)
    chosen = scored[0]

    seed_start = _infer_next_seed_start(RESULTS, seed_defaults)

    plan = {
        "model": "gpt-5.3-codex",
        "window": {"start_round": args.start_round, "end_round": args.end_round, "block_size": args.block_size},
        "history_window": {"from": args.history_from, "to": args.history_to},
        "promote_pp": args.promote_pp,
        "candidates": candidates,
        "kill_tasks": kill_tasks,
        "scores": [
            {
                "profile": s["profile"],
                "score": round(float(s["score"]), 6),
                "search_mix": s["search_mix"],
            }
            for s in scored
        ],
        "chosen_profile": chosen["profile"],
        "seed_start": seed_start,
    }

    out_plan = RESULTS / f"_bestofn_plan_round{args.start_round}_{args.end_round}.json"
    out_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_plan)
    print(f"chosen_profile={chosen['profile']} score={chosen['score']:.4f}")

    if not args.dry_run:
        _run_queue_builder(chosen["profile"], args.start_round, args.end_round, args.block_size, seed_start)


if __name__ == "__main__":
    main()
