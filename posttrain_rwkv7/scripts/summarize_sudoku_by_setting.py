#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def parse_best(run_dir: Path) -> dict[str, float]:
    best_loss = None
    best_acc = None
    with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "val_loss" in r:
                v = float(r["val_loss"])
                if best_loss is None or v < best_loss:
                    best_loss = v
            if "val_tok_acc" in r:
                v = float(r["val_tok_acc"])
                if best_acc is None or v > best_acc:
                    best_acc = v
    if best_loss is None or best_acc is None:
        raise RuntimeError(f"missing metrics in {run_dir}")
    return {"best_val_loss": float(best_loss), "best_val_tok_acc": float(best_acc)}


def match(cfg: dict[str, Any], flt: dict[str, Any]) -> bool:
    defaults = {
        "fs_layer_start": 6,
        "fs_alpha_schedule": "none",
        "fs_alpha_min": 1.0,
        "fs_alpha_max": 1.0,
        "fs_norm": False,
        "fs_clip": 1.0,
        "fs_detach": False,
    }
    for k, v in flt.items():
        if k not in cfg and k in defaults:
            if defaults[k] != v:
                return False
            continue
        if cfg.get(k, None) != v:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--time_budget_sec", type=int, default=240)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--val_batches", type=int, default=16)

    ap.add_argument("--base", type=int, default=3)
    ap.add_argument("--mask_count", type=int, default=40)
    ap.add_argument("--mask_region", choices=["prefix", "suffix", "random"], default="prefix")

    ap.add_argument("--alpha_init", type=float, default=-2.0)
    ap.add_argument("--alpha_lr", type=float, default=0.0)
    ap.add_argument("--fs_variant", choices=["scalar", "head"], default="scalar")
    ap.add_argument("--alpha_head_init", type=float, default=None)
    ap.add_argument("--alpha_head_lr", type=float, default=None)
    ap.add_argument("--model_lr", type=float, default=3e-5)
    ap.add_argument("--seed_scale", type=float, default=1.0)
    ap.add_argument("--fs_layer_start", type=int, default=6)
    ap.add_argument("--fs_alpha_schedule", choices=["none", "linear", "cosine"], default="none")
    ap.add_argument("--fs_alpha_min", type=float, default=1.0)
    ap.add_argument("--fs_alpha_max", type=float, default=1.0)
    ap.add_argument("--fs_norm", action="store_true")
    ap.add_argument("--fs_clip", type=float, default=1.0)
    ap.add_argument("--fs_detach", action="store_true")
    args = ap.parse_args()

    if args.alpha_head_init is None:
        args.alpha_head_init = float(args.alpha_init)
    if args.alpha_head_lr is None:
        args.alpha_head_lr = float(args.alpha_lr)

    seeds = {int(s) for s in args.seeds.split(",") if s.strip()}
    root = Path("runs")

    flt = {
        "base": int(args.base),
        "mask_count": int(args.mask_count),
        "mask_region": str(args.mask_region),
        "bsz": int(args.bsz),
        "time_budget_sec": int(args.time_budget_sec),
        "max_steps": int(args.max_steps),
        "eval_every": int(args.eval_every),
        "val_batches": int(args.val_batches),
        "model_lr": float(args.model_lr),
        "alpha_init": float(args.alpha_init),
        "alpha_lr": float(args.alpha_lr),
        "fs_variant": str(args.fs_variant),
        "alpha_head_init": float(args.alpha_head_init),
        "alpha_head_lr": float(args.alpha_head_lr),
        "seed_scale": float(args.seed_scale),
        "fs_layer_start": int(args.fs_layer_start),
        "fs_alpha_schedule": str(args.fs_alpha_schedule),
        "fs_alpha_min": float(args.fs_alpha_min),
        "fs_alpha_max": float(args.fs_alpha_max),
        "fs_norm": bool(args.fs_norm),
        "fs_clip": float(args.fs_clip),
        "fs_detach": bool(args.fs_detach),
    }

    runs = []
    for cfg_path in root.glob("**/sudoku_sft/*/config.json"):
        cfg = load_json(cfg_path)
        if cfg.get("mode") not in {"no_fs", "prompt_fs"}:
            continue
        seed = cfg.get("seed")
        if seed not in seeds:
            continue
        if not match(cfg, flt):
            continue
        run_dir = cfg_path.parent
        if not (run_dir / "metrics.jsonl").exists():
            continue
        runs.append((cfg, run_dir))

    if not runs:
        print("No matched Sudoku runs yet.")
        print("filter:", flt)
        return

    latest: dict[tuple[int, str], Path] = {}
    for cfg, run_dir in runs:
        seed = int(cfg["seed"])
        mode = str(cfg["mode"])
        k = (seed, mode)
        if k not in latest or str(run_dir) > str(latest[k]):
            latest[k] = run_dir

    rows: list[dict[str, Any]] = []
    for seed in sorted(seeds):
        a = latest.get((seed, "no_fs"))
        b = latest.get((seed, "prompt_fs"))
        if a is None or b is None:
            continue
        ma = parse_best(a)
        mb = parse_best(b)
        rows.append(
            {
                "seed": seed,
                "no_fs_loss": ma["best_val_loss"],
                "fs_loss": mb["best_val_loss"],
                "d_loss": mb["best_val_loss"] - ma["best_val_loss"],
                "no_fs_acc": ma["best_val_tok_acc"],
                "fs_acc": mb["best_val_tok_acc"],
                "d_acc": mb["best_val_tok_acc"] - ma["best_val_tok_acc"],
            }
        )

    if not rows:
        print("No complete no_fs/prompt_fs seed pairs yet.")
        print("filter:", flt)
        return

    import statistics as st

    d_acc = [r["d_acc"] for r in rows]
    d_loss = [r["d_loss"] for r in rows]
    n_pos = sum(1 for x in d_acc if x > 0)
    n_neg = sum(1 for x in d_acc if x < 0)

    print("filter:", flt)
    print("=" * 90)
    print(
        f"pairs={len(rows)} mean d_acc={st.mean(d_acc):+.4f} std={st.pstdev(d_acc):.4f} | mean d_loss={st.mean(d_loss):+.4f} std={st.pstdev(d_loss):.4f} | sign(+/0/-)={n_pos}/{len(rows)-n_pos-n_neg}/{n_neg}"
    )
    for r in sorted(rows, key=lambda x: x["seed"]):
        print(
            "  seed={seed} d_acc={d_acc:+.4f} (no={no_fs_acc:.4f} fs={fs_acc:.4f}) d_loss={d_loss:+.4f}".format(
                **r
            )
        )


if __name__ == "__main__":
    main()

