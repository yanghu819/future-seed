# RWKV7 Future-Seed Post-Training

This directory is the local archival snapshot of the Future-Seed post-training campaign on a single 4090.

## Final Status (2026-03-06, through round808)

Current state:
- search is intentionally closed through `round808`
- latest local snapshot artifacts live in `runs/`
- current work is documentation, table cleanup, and repo hygiene, not more BFS under the same recipe
- legacy AutoDL-era scripts and chronology remain for provenance only; see [`LEGACY_AUTODL.md`](LEGACY_AUTODL.md)

## Historical Archive Summary Used By The Paper

This table is the curated paper-side family summary from the broader internal post-training archive.
It is not raw-recomputed from the shipped `runs/` subset in this repository snapshot.

| Task family | Best med gain | Med median | Positive med count | Current judgment |
|---|---:|---:|---:|---|
| `protein_ss_spot` | `+8.14pp` | `+2.18pp` | `103/126` | strongest repeatable |
| `hotpot_text_restore` | `+4.20pp` | `+0.54pp` | `35/53` | repeatable positive |
| `mbpp_longctx_probe` | `+10.00pp` | `+0.91pp` | `53/87` | promising, unconfirmed |
| `arc_mc_probe` | `+20.83pp` | `+4.17pp` | `70/116` | high upside, unstable |
| `squad_text_restore` | `+7.31pp` | `+0.55pp` | `24/35` | mixed, not locked |
| `punc_restore` | `+2.16pp` | `+0.36pp` | `16/24` | historical positive pocket |
| `graph_color` | `+8.33pp` | `+0.00pp` | `4/10` | useful diagnostic task only |
| `tsp_mask` | `+25.00pp` | `+2.08pp` | `2/4` | appendix-only spike |

## Shipped Snapshot-Backed Closure/Breadth Evidence

This table is the part that the released `runs/` snapshot can directly recompute from raw JSONL records
for rounds `783-788` and `799-808`.

| Task family | Shipped best med gain | Shipped med median | Shipped positive med count | Released reading |
|---|---:|---:|---:|---|
| protein_ss_spot | n/a | n/a | 0/0 | near-gate quick positives only in the shipped breadth window |
| hotpot_text_restore | +1.00pp | +0.55pp | 2/2 | released positive breadth signal |
| mbpp_longctx_probe | +5.01pp | +0.54pp | 3/5 | mixed released closure evidence |
| arc_mc_probe | +9.38pp | +3.12pp | 3/5 | mixed released closure evidence |
| squad_text_restore | n/a | n/a | 0/0 | quick-only near-gate signal in the shipped breadth window |
| punc_restore | n/a | n/a | 0/0 | historical only; not present in the shipped closure/breadth subset |

Low-ROI or negative families under the current recipe:
- `protein_contact`
- `wiki`
- `hotpot_longctx`
- `countdown`
- `nqueens`
- `zebra`
- `sat3`

## Final Closure And Breadth Pivot

### Shipped closure subset (`round783-788` and `round799-802`)

- `mbpp_longctx_probe` stayed promising but did not pass strict final confirmation:
  - shipped exploit positives: `round785 +5.01pp`, `round786 +0.54pp`, `round788 +0.97pp`
  - strict confirms failed to promote:
    - `round799`: quick `+0.38pp`
    - `round800`: quick `+0.60pp`
- `arc_mc_probe` produced large positives but stayed unstable:
  - shipped positives: `round785 +3.12pp`, `round787 +9.38pp`, `round788 +3.12pp`
  - held-out confirm failed:
    - `round801`: med `-1.56pp`
- `tsp_mask` did not confirm:
  - earlier spike: `round764 +25.00pp`
  - final confirm: `round802` quick `+0.00pp`

### Breadth pivot (`round803-808`)

- `round803-804` alt-confirm killed more spending on repeated `mbpp_longctx_probe` confirmation with the same `head_l8 / scalar_l8_*` family
- `round805-808` widened back out to `protein_ss_spot`, `hotpot_text_restore`, `squad_text_restore`, `hotpot_longctx`, and `wiki`
- new outcomes:
  - `round805 hotpot_seed95_breadth`: med `+0.11pp`
  - `round807 hotpot_seed96_breadth`: med `+1.00pp`
  - `round805 protein_ss_seed283_breadth`: quick `+0.58pp`, no promote
  - `round807 protein_ss_seed284_breadth`: quick `+0.40pp`, no promote
  - `round806 squad_seed104_breadth`: quick `+0.76pp`, no promote
  - `round808 squad_seed105_breadth`: quick `+0.76pp`, no promote
  - `round806 hotpot_longctx_seed12_breadth`: flat at `+0.00pp`
  - `round808 wiki_seed41_breadth`: baseline failed

## What The Repo Can Defend

1. Future-Seed clearly helps on toy and synthetic constraint-repair tasks.
2. In post-training, `protein_ss_spot` is the strongest repeatable family in the paper-side archive summary under the current recipe.
3. `hotpot_text_restore` has direct medium-stage positives in the released breadth window; `squad_text_restore` and `punc_restore` should be read as smaller historical positive pockets rather than current snapshot-backed lines.
4. `mbpp_longctx_probe`, `arc_mc_probe`, and `tsp_mask` remain mixed or exploratory rather than stable headline evidence.

## Sudoku Benchmark Boundary

- the canonical Sudoku benchmark is `scripts/train_sudoku9_unique_sft.py` plus `scripts/run_sudoku9_unique_maintrack.py`
- it uses 9x9 unique-solution puzzles, clue-forced in-place decode, and reports exact solve / validity / clue consistency / blank accuracy
- the older `scripts/train_sudoku_sft.py` remains an archive teacher-forced probe and should not be used as the main Sudoku headline
- smoke manifests for the canonical benchmark live in `assets/sudoku9_unique/`

## What The Repo Should Not Claim

1. real-task gains are already stable across held-out confirmation seeds
2. `mbpp_longctx_probe` was strictly confirmed by the final confirmation queue
3. `arc_mc_probe` is already robust enough for a clean stability claim
4. constraint-task wins alone prove the real-task story
5. the paper-side family counts can be raw-recomputed from the shipped `runs/` subset alone

## Quick Reproduction Entrypoints

```bash
cd posttrain_rwkv7
export FUTURE_SEED_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/future-seed"
python3 scripts/run_round77_82_fastdiscover.py --self_test
python3 scripts/run_round77_82_fastdiscover.py \
  --queue results/_search_queue_round805_808_breadth_roi.json \
  --round_from 805 --round_to 808 --dry_run
```

Notes:
- explicit `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, and `TORCH_EXTENSIONS_DIR` override `FUTURE_SEED_CACHE_ROOT`
- a `--dry_run` may still show baseline failures when the local dataset cache is empty; that indicates missing local data rather than a runner regression

Useful queue files:
- `results/_search_queue_round783_790_realtask_exploit_v3.json`
- `results/_search_queue_round799_802_final_confirm.json`
- `results/_search_queue_round803_804_mbpp_altconfirm.json`
- `results/_search_queue_round805_808_breadth_roi.json`

## Document Map

- `runs/`: local snapshot summaries and raw JSONL records
- `results/_rolling_round_log.md`: round-by-round operator log
- `paper/DETAILED_EXPERIMENT_LOG.md`: full success/failure ledger
- `ARCHIVE_ROUNDS.md`: archived round chronology previously in this README
- `assets/sudoku9_unique/`: smoke manifests and manifest format for the canonical 9x9 Sudoku benchmark
- `scripts/`: launchers, orchestrators, and trainers
- `LEGACY_AUTODL.md`: boundary note for preserved AutoDL-era history

## Historical Archive

The long round-by-round chronology has been moved to [`ARCHIVE_ROUNDS.md`](ARCHIVE_ROUNDS.md).

Use this README for the current state only. For historical forensics and search evolution, follow:

- [`ARCHIVE_ROUNDS.md`](ARCHIVE_ROUNDS.md)
- [`results/_rolling_round_log.md`](results/_rolling_round_log.md)
- [`paper/DETAILED_EXPERIMENT_LOG.md`](paper/DETAILED_EXPERIMENT_LOG.md)
