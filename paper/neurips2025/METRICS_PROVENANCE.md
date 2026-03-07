# Metrics Provenance

This file explains what `data/metrics.json` is and is not.

## What It Is

- `data/metrics.json` is the curated paper-side metrics snapshot used to render the LaTeX tables in `main.tex`.
- `render_tables.py` converts `data/metrics.json` into `tables/*.tex`.
- The file is version-controlled so the PDF can be rebuilt deterministically.
- The curated values are tied to the shipped local snapshot artifacts for rounds `783-788` and `799-808`.

## What It Is Not

- It is **not** currently regenerated from the raw run archive by an automated script in this repository snapshot.
- It is **not** the same artifact as `posttrain_rwkv7/results/UNIFIED_EFFECTIVE_EXPERIMENTS.md`.
- It should therefore be read as a curated internal-archive summary prepared for the paper, not as a raw public aggregator dump.

## Main-Table Row Mapping

| Paper row | Source family in curated metrics | Raw artifact family |
|---|---|---|
| `protein_ss_spot` | `posttrain.main_summary[0]` | `protein_ss_*` runs from `train_protein_ss_spot_sft.py` |
| `hotpot_text_restore` | `posttrain.main_summary[1]` | `hotpot_*` runs executed by `train_punc_restore_sft.py` on `hotpot_qa/distractor` |
| `mbpp_longctx_probe` | `posttrain.main_summary[2]` | `mbpp_longctx_*` runs from `train_mbpp_longctx_sft.py` |
| `arc_mc_probe` | `posttrain.main_summary[3]` | `arc_mc_*` runs from `train_arc_mc_sft.py` |
| `squad_text_restore` | `posttrain.main_summary[4]` | `squad_*` runs executed by `train_punc_restore_sft.py` on `squad` |
| `punc_restore` | `posttrain.main_summary[5]` | `punc_*` punctuation/case restoration runs from `train_punc_restore_sft.py` |

## Synthetic Row Mapping

- `synthetic.main_summary` comes from the committed synthetic experiment summaries in `rwkv-diff-future-seed/`.
- `synthetic.permfill`, `synthetic.sudoku_phase`, `synthetic.sudoku_consistency`, and `synthetic.transformer_baselines` are paper-side summaries of the committed synthetic experiment outputs.

## Update Rule

If the paper tables change:

1. update `data/metrics.json`
2. explain the change here if the semantics of any row changed
3. run `python3 render_tables.py`
4. rebuild with `./build.sh submission`
