# Metrics Provenance

This file explains what `data/metrics.json` is and is not.

## What It Is

- `data/metrics.json` is the curated paper-side metrics snapshot used to render the LaTeX tables in `main.tex`.
- `render_tables.py` converts `data/metrics.json` into `tables/*.tex`.
- The file is version-controlled so the PDF can be rebuilt deterministically.
- `posttrain.main_summary` is a curated internal-archive summary used by the paper-side post-training table.
- `posttrain.snapshot_boundary_summary` is the subset that is directly recomputed from the shipped local snapshot artifacts for rounds `783-788` and `799-808`.
- `posttrain.closure_highlights` are representative release-backed rows inside that shipped subset.

## What It Is Not

- `posttrain.main_summary` is **not** currently regenerated from the full raw run archive by an automated script in this repository snapshot.
- It is **not** the same artifact as `posttrain_rwkv7/results/UNIFIED_EFFECTIVE_EXPERIMENTS.md`.
- The current release only ships a subset of late closure/breadth rounds, so it should be read as a curated internal-archive summary plus a release-backed boundary-audit subset, not as a raw public aggregator dump.

## Main-Table Row Mapping

| Paper row | Source family in curated metrics | Raw artifact family |
|---|---|---|
| `protein_ss_spot` | `posttrain.main_summary[0]` | `protein_ss_*` runs from `train_protein_ss_spot_sft.py` |
| `hotpot_text_restore` | `posttrain.main_summary[1]` | `hotpot_*` runs executed by `train_punc_restore_sft.py` on `hotpot_qa/distractor` |
| `mbpp_longctx_probe` | `posttrain.main_summary[2]` | `mbpp_longctx_*` runs from `train_mbpp_longctx_sft.py` |
| `arc_mc_probe` | `posttrain.main_summary[3]` | `arc_mc_*` runs from `train_arc_mc_sft.py` |
| `squad_text_restore` | `posttrain.main_summary[4]` | `squad_*` runs executed by `train_punc_restore_sft.py` on `squad` |
| `punc_restore` | `posttrain.main_summary[5]` | `punc_*` punctuation/case restoration runs from `train_punc_restore_sft.py` |

## Snapshot-Boundary Mapping

| Snapshot row | Source family in shipped runs | Directly recomputed from raw JSONL? |
|---|---|---|
| `protein_ss_spot` | `protein_ss_*` breadth rows in rounds `805-808` | yes; currently no medium-stage rows in the shipped subset |
| `hotpot_text_restore` | `hotpot_seed95_breadth`, `hotpot_seed96_breadth` | yes |
| `mbpp_longctx_probe` | `mbpp_longctx_seed37-41_depthmix_anchor` in rounds `784-788` | yes |
| `arc_mc_probe` | `arc_mc_seed258-261_depthmix_qfirst` and `arc_mc_seed271_finalconfirm` | yes |
| `squad_text_restore` | `squad_seed104/105_breadth` in rounds `806` and `808` | yes; currently quick-only, no medium-stage rows |
| `punc_restore` | no shipped closure/breadth rows in the current subset | yes; currently empty in the shipped subset |

## Synthetic Row Mapping

- `synthetic.main_summary` comes from the committed synthetic experiment summaries in `rwkv-diff-future-seed/`.
- `synthetic.permfill`, `synthetic.sudoku_phase`, `synthetic.sudoku_consistency`, and `synthetic.transformer_baselines` are paper-side summaries of the committed synthetic experiment outputs.

## Update Rule

If the paper tables change:

1. update `data/metrics.json`
2. explain the change here if the semantics of any row changed
3. run `python3 render_tables.py`
4. rebuild with `./build.sh submission`
