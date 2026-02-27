#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


CANDIDATES: Dict[str, List[str]] = {
    "scalar_l8_train1e4": [
        "--alpha_lr",
        "1e-4",
        "--alpha_init",
        "-2",
        "--fs_variant",
        "scalar",
        "--fs_layer_start",
        "8",
        "--fs_norm",
        "--fs_detach",
        "--fs_clip",
        "1.0",
    ],
    "scalar_l8_train8e5": [
        "--alpha_lr",
        "8e-5",
        "--alpha_init",
        "-2",
        "--fs_variant",
        "scalar",
        "--fs_layer_start",
        "8",
        "--fs_norm",
        "--fs_detach",
        "--fs_clip",
        "1.0",
    ],
    "scalar_l8_sched_cos": [
        "--alpha_lr",
        "0",
        "--alpha_init",
        "-2",
        "--fs_variant",
        "scalar",
        "--fs_layer_start",
        "8",
        "--fs_norm",
        "--fs_detach",
        "--fs_clip",
        "1.0",
        "--fs_alpha_schedule",
        "cosine",
        "--fs_alpha_min",
        "0.4",
        "--fs_alpha_max",
        "1.0",
    ],
    "head_l8": [
        "--alpha_lr",
        "0",
        "--alpha_init",
        "-3",
        "--fs_variant",
        "head",
        "--alpha_head_init",
        "-3",
        "--alpha_head_lr",
        "5e-4",
        "--fs_layer_start",
        "8",
        "--fs_norm",
        "--fs_detach",
        "--fs_clip",
        "1.0",
    ],
    "head_l10_strong": [
        "--alpha_lr",
        "0",
        "--alpha_init",
        "-2",
        "--fs_variant",
        "head",
        "--alpha_head_init",
        "-2",
        "--alpha_head_lr",
        "1e-3",
        "--fs_layer_start",
        "10",
        "--fs_norm",
        "--fs_detach",
        "--fs_clip",
        "1.0",
    ],
}


def cand_pack(names: List[str]) -> List[dict]:
    return [{"name": n, "args": deepcopy(CANDIDATES[n])} for n in names]


def mk_arc(seed: int) -> dict:
    return {
        "name": f"arc_mc_seed{seed}_discovery",
        "seed": seed,
        "trainer": "train_arc_mc_sft.py",
        "ds": "ai2_arc",
        "ds_cfg": "ARC-Challenge",
        "base_args": [
            "--ds",
            "ai2_arc",
            "--ds_cfg",
            "ARC-Challenge",
            "--train_split",
            "train",
            "--val_split",
            "validation",
            "--n_train",
            "1000",
            "--n_val",
            "200",
            "--max_prompt_tokens",
            "512",
            "--max_answer_tokens",
            "8",
            "--bsz",
            "8",
        ],
        "quick_budget": 130,
        "quick_steps": 160,
        "quick_eval_every": 20,
        "quick_val_batches": 3,
        "med_budget": 240,
        "med_steps": 320,
        "med_eval_every": 20,
        "med_val_batches": 3,
        "candidates": cand_pack(["scalar_l8_train1e4", "scalar_l8_sched_cos", "scalar_l8_train8e5"]),
    }


def mk_protein_ss(seed: int) -> dict:
    return {
        "name": f"protein_ss_seed{seed}_discovery",
        "seed": seed,
        "trainer": "train_protein_ss_spot_sft.py",
        "ds": "lamm-mit/protein_secondary_structure_from_PDB",
        "ds_cfg": "",
        "base_args": [
            "--ds",
            "lamm-mit/protein_secondary_structure_from_PDB",
            "--split",
            "train",
            "--n_train",
            "1200",
            "--n_val",
            "240",
            "--max_seq_len",
            "512",
            "--min_seq_len",
            "96",
            "--num_queries",
            "48",
            "--query_region",
            "random",
            "--fill_notes_to_max",
            "--note_pool_size",
            "2048",
            "--max_note_seq_len",
            "256",
            "--max_prompt_tokens",
            "2048",
            "--min_prompt_tokens",
            "1024",
            "--max_answer_tokens",
            "128",
            "--bsz",
            "2",
        ],
        "quick_budget": 140,
        "quick_steps": 160,
        "quick_eval_every": 20,
        "quick_val_batches": 4,
        "med_budget": 260,
        "med_steps": 320,
        "med_eval_every": 20,
        "med_val_batches": 4,
        "candidates": cand_pack(["head_l8", "scalar_l8_train1e4", "scalar_l8_sched_cos"]),
    }


def mk_hotpot(seed: int) -> dict:
    return {
        "name": f"hotpot_seed{seed}_discovery",
        "seed": seed,
        "trainer": "train_punc_restore_sft.py",
        "ds": "hotpot_qa",
        "ds_cfg": "distractor",
        "base_args": [
            "--ds",
            "hotpot_qa",
            "--ds_cfg",
            "distractor",
            "--train_split",
            "train",
            "--val_split",
            "validation",
            "--n_train",
            "800",
            "--n_val",
            "160",
            "--min_chars",
            "48",
            "--max_chars",
            "220",
            "--fill_notes_to_max",
            "--note_pool_size",
            "1024",
            "--max_prompt_tokens",
            "1536",
            "--min_prompt_tokens",
            "512",
            "--max_answer_tokens",
            "128",
            "--bsz",
            "2",
        ],
        "quick_budget": 120,
        "quick_steps": 160,
        "quick_eval_every": 20,
        "quick_val_batches": 3,
        "med_budget": 240,
        "med_steps": 360,
        "med_eval_every": 20,
        "med_val_batches": 3,
        "candidates": cand_pack(["scalar_l8_train1e4", "head_l8", "scalar_l8_sched_cos"]),
    }


def mk_mbpp(seed: int, novelty_tier: str) -> dict:
    suffix = "anchor" if novelty_tier == "anchor" else "headprobe"
    return {
        "name": f"mbpp_seed{seed}_{suffix}",
        "seed": seed,
        "trainer": "train_punc_restore_sft.py",
        "ds": "mbpp",
        "ds_cfg": "",
        "base_args": [
            "--ds",
            "mbpp",
            "--ds_cfg",
            "",
            "--train_split",
            "train",
            "--val_split",
            "test",
            "--n_train",
            "340",
            "--n_val",
            "100",
            "--min_chars",
            "24",
            "--max_chars",
            "360",
            "--fill_notes_to_max",
            "--note_pool_size",
            "384",
            "--max_prompt_tokens",
            "1408",
            "--min_prompt_tokens",
            "384",
            "--max_answer_tokens",
            "160",
            "--bsz",
            "12",
        ],
        "quick_budget": 160,
        "quick_steps": 160,
        "quick_eval_every": 20,
        "quick_val_batches": 2,
        "med_budget": 260,
        "med_steps": 320,
        "med_eval_every": 20,
        "med_val_batches": 3,
        "candidates": cand_pack(["head_l10_strong", "scalar_l8_sched_cos", "scalar_l8_train1e4"]),
    }


def mk_mbpp_longctx(seed: int) -> dict:
    return {
        "name": f"mbpp_longctx_seed{seed}_repair",
        "seed": seed,
        "trainer": "train_mbpp_longctx_sft.py",
        "ds": "mbpp",
        "ds_cfg": "",
        "base_args": [
            "--ds",
            "mbpp",
            "--ds_cfg",
            "",
            "--train_split",
            "train",
            "--val_split",
            "test",
            "--n_train",
            "320",
            "--n_val",
            "80",
            "--fill_notes_to_max",
            "--note_pool_size",
            "768",
            "--max_prompt_tokens",
            "2560",
            "--min_prompt_tokens",
            "1024",
            "--max_answer_tokens",
            "160",
            "--bsz",
            "1",
        ],
        "quick_budget": 160,
        "quick_steps": 160,
        "quick_eval_every": 20,
        "quick_val_batches": 3,
        "med_budget": 260,
        "med_steps": 320,
        "med_eval_every": 20,
        "med_val_batches": 3,
        "candidates": cand_pack(["head_l8", "scalar_l8_train1e4", "scalar_l8_sched_cos"]),
    }


def mk_squad(seed: int, novelty_tier: str) -> dict:
    suffix = "anchor" if novelty_tier == "anchor" else "discovery"
    return {
        "name": f"squad_seed{seed}_{suffix}",
        "seed": seed,
        "trainer": "train_punc_restore_sft.py",
        "ds": "squad",
        "ds_cfg": "",
        "base_args": [
            "--ds",
            "squad",
            "--ds_cfg",
            "",
            "--train_split",
            "train",
            "--val_split",
            "validation",
            "--n_train",
            "1200",
            "--n_val",
            "160",
            "--min_chars",
            "64",
            "--max_chars",
            "260",
            "--fill_notes_to_max",
            "--note_pool_size",
            "1024",
            "--max_prompt_tokens",
            "1536",
            "--min_prompt_tokens",
            "512",
            "--max_answer_tokens",
            "128",
            "--bsz",
            "12",
        ],
        "quick_budget": 150,
        "quick_steps": 160,
        "quick_eval_every": 20,
        "quick_val_batches": 2,
        "med_budget": 260,
        "med_steps": 320,
        "med_eval_every": 20,
        "med_val_batches": 3,
        "candidates": cand_pack(["scalar_l8_sched_cos", "scalar_l8_train1e4", "head_l8"]),
    }


def mk_wiki(seed: int, novelty_tier: str) -> dict:
    suffix = "anchor" if novelty_tier == "anchor" else "discovery"
    return {
        "name": f"wiki_seed{seed}_{suffix}",
        "seed": seed,
        "trainer": "train_punc_restore_sft.py",
        "ds": "wikitext",
        "ds_cfg": "wikitext-2-raw-v1",
        "base_args": [
            "--ds",
            "wikitext",
            "--ds_cfg",
            "wikitext-2-raw-v1",
            "--train_split",
            "train",
            "--val_split",
            "validation",
            "--n_train",
            "1200",
            "--n_val",
            "200",
            "--min_chars",
            "48",
            "--max_chars",
            "260",
            "--fill_notes_to_max",
            "--note_pool_size",
            "1024",
            "--max_prompt_tokens",
            "1536",
            "--min_prompt_tokens",
            "512",
            "--max_answer_tokens",
            "160",
            "--bsz",
            "4",
        ],
        "quick_budget": 120,
        "quick_steps": 160,
        "quick_eval_every": 20,
        "quick_val_batches": 2,
        "med_budget": 220,
        "med_steps": 320,
        "med_eval_every": 20,
        "med_val_batches": 3,
        "candidates": cand_pack(["head_l8", "scalar_l8_train1e4", "scalar_l8_sched_cos"]),
    }


BUILDERS = {
    "arc_mc": mk_arc,
    "protein_ss": mk_protein_ss,
    "hotpot": mk_hotpot,
    "mbpp": mk_mbpp,
    "mbpp_longctx": mk_mbpp_longctx,
    "squad": mk_squad,
    "wiki": mk_wiki,
}


SEED_START = {
    "arc_mc": 51,
    "protein_ss": 63,
    "hotpot": 45,
    "mbpp": 48,
    "mbpp_longctx": 15,
    "squad": 6,
    "wiki": 2,
}


CYCLE_BROAD_V1: List[Tuple[str, Tuple[str, str]]] = [
    ("new", ("arc_mc", "protein_ss")),
    ("new", ("mbpp", "hotpot")),
    ("new", ("squad", "wiki")),
    ("anchor", ("mbpp_longctx", "protein_ss")),
    ("new", ("arc_mc", "hotpot")),
    ("new", ("mbpp", "squad")),
    ("anchor", ("wiki", "protein_ss")),
    ("new", ("arc_mc", "mbpp_longctx")),
]


CYCLE_REBALANCE_V2: List[Tuple[str, Tuple[str, str]]] = [
    ("new", ("protein_ss", "arc_mc")),
    ("new", ("mbpp", "protein_ss")),
    ("new", ("squad", "wiki")),
    ("anchor", ("mbpp_longctx", "protein_ss")),
    ("new", ("arc_mc", "mbpp")),
    ("new", ("protein_ss", "hotpot")),
    ("anchor", ("squad", "mbpp_longctx")),
    ("new", ("protein_ss", "arc_mc")),
]


PROFILES: Dict[str, Dict[str, object]] = {
    "broad_v1": {
        "cycle": CYCLE_BROAD_V1,
        "search_mix": "75_new_25_anchor_broad",
    },
    "rebalance_v2": {
        "cycle": CYCLE_REBALANCE_V2,
        "search_mix": "75_new_25_anchor_rebalance_v2",
    },
}


def build_task(task_key: str, seed: int, novelty_tier: str) -> dict:
    fn = BUILDERS[task_key]
    if task_key in {"mbpp", "squad", "wiki"}:
        return fn(seed, novelty_tier)  # type: ignore[misc]
    return fn(seed)  # type: ignore[misc]


def generate(start_round: int, end_round: int, block_size: int, profile: str, seed_start: Dict[str, int]) -> None:
    profile_conf = PROFILES[profile]
    cycle: List[Tuple[str, Tuple[str, str]]] = profile_conf["cycle"]  # type: ignore[assignment]
    search_mix = str(profile_conf["search_mix"])
    seed_ctr = dict(seed_start)
    for block_start in range(start_round, end_round + 1, block_size):
        block_end = min(block_start + block_size - 1, end_round)
        rounds = []
        for rid in range(block_start, block_end + 1):
            novelty_tier, pair = cycle[(rid - start_round) % len(cycle)]
            tasks = []
            for task_key in pair:
                seed = seed_ctr[task_key]
                seed_ctr[task_key] += 1
                tasks.append(build_task(task_key, seed, novelty_tier))
            rounds.append({"round": rid, "novelty_tier": novelty_tier, "tasks": tasks})

        payload = {
            "strategy": {
                "search_mix": search_mix,
                "quick_promote_pp": 0.8,
                "quick_prune_pp": -0.5,
                "strong_negative_quick_pp": -1.0,
                "strong_negative_med_pp": -0.5,
                "cooldown_rounds": 8,
                "useful_rule": "single_positive_med",
            },
            "rounds": rounds,
        }
        out = RESULTS / f"_search_queue_round{block_start}_{block_end}_fastloop.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_round", type=int, default=233)
    ap.add_argument("--end_round", type=int, default=400)
    ap.add_argument("--block_size", type=int, default=8)
    ap.add_argument("--profile", choices=sorted(PROFILES.keys()), default="broad_v1")
    ap.add_argument("--seed_arc_mc", type=int, default=SEED_START["arc_mc"])
    ap.add_argument("--seed_protein_ss", type=int, default=SEED_START["protein_ss"])
    ap.add_argument("--seed_hotpot", type=int, default=SEED_START["hotpot"])
    ap.add_argument("--seed_mbpp", type=int, default=SEED_START["mbpp"])
    ap.add_argument("--seed_mbpp_longctx", type=int, default=SEED_START["mbpp_longctx"])
    ap.add_argument("--seed_squad", type=int, default=SEED_START["squad"])
    ap.add_argument("--seed_wiki", type=int, default=SEED_START["wiki"])
    args = ap.parse_args()
    seed_start = {
        "arc_mc": int(args.seed_arc_mc),
        "protein_ss": int(args.seed_protein_ss),
        "hotpot": int(args.seed_hotpot),
        "mbpp": int(args.seed_mbpp),
        "mbpp_longctx": int(args.seed_mbpp_longctx),
        "squad": int(args.seed_squad),
        "wiki": int(args.seed_wiki),
    }
    generate(args.start_round, args.end_round, args.block_size, args.profile, seed_start)


if __name__ == "__main__":
    main()
