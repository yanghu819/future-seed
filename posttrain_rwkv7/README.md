# RWKV7 Future-Seed Post-Training Backup

This folder is a backup snapshot of the current Future-Seed post-training work on AutoDL.

## Contents

- `scripts/`: training, summarization, and run scripts used in the current iteration.
- `results/`: exported summary text files from completed runs.
- `paper/`: current paper/progress note.
  - `paper/FS_POSTTRAIN_PROGRESS_2026-02-19.md`: short status note.
  - `paper/DETAILED_EXPERIMENT_LOG.md`: full success/failure experiment record.
  - `paper/FS_POSTTRAIN_PAPER_DRAFT.md`: current paper draft synced with latest experiments.

## Key Findings (latest snapshot)

- ARC-Challenge options-first: stable positive gain with FS (`_summary_arc_optionsfirst_stabilized_r2.txt`), but schedule variant was weaker (`_summary_arc_optionsfirst_stabilized_r4_sched_linear.txt`).
- HotpotQA L=4096:
  - R6 baseline near zero mean with large seed variance (`_summary_hotpot_qafter_stabilized_len4096_r6_s012.txt`).
  - R9/R10 did not improve mean accuracy.
  - R11 grid showed either near-zero/no-op behavior (`lstart=12`) or mixed gains with regressions (`lstart=10`).
- MBPP long-context:
  - q-after and q-first both regressed under FS (`_summary_mbpp_qafter_stabilized_len4096_r1_s012.txt`, `_summary_mbpp_qfirst_stabilized_len4096_r1_s012.txt`).
- Sudoku:
  - 4x4 prefix: small consistent gain (`_summary_sudoku4_prefix_r1_s012.txt`).
  - 4x4 suffix: near neutral (`_summary_sudoku4_suffix_r1_s012.txt`).
  - 9x9 prefix: near neutral / unstable (`_summary_sudoku9_prefix_r1_s012.txt`).
  - 9x9 suffix: strong regression (`_summary_sudoku9_suffix_r1_s012.txt`).
- Protein real-task probes:
  - SS spot labeling (q-after/q-first): near-zero, unstable deltas.
  - Contact-pair QA:
    - r1/r2: exact tie on token/seq acc.
    - r3 balanced and r4 schedule: small negative mean deltas.
- Round12 (5-seed high-util stability check):
  - ARC options-first regressed (`_summary_arc_optionsfirst_stabilized_r5_s01234.txt`).
  - Hotpot q-after/q-first both show small positive means but mixed signs (`_summary_hotpot_qafter_stabilized_len4096_r12_lstart10_alpha-3_s01234.txt`, `_summary_hotpot_qfirst_stabilized_len4096_r12_lstart10_alpha-3_s01234.txt`).

## Latest Rapid-Iteration Update (2026-02-21, Round20/21)

### Round20 (single-seed, serial, immediate-prune)

- Summary: `results/_summary_round20_serial_earlystop_s0.txt`
- Full records: `results/_round20_serial_earlystop_records.jsonl`
- Policy:
  - single GPU serial execution
  - quick screen first
  - prune candidate immediately if quick `d_acc < +0.001` (i.e., `< +0.10pp`)

Main outcome:

- `hotpot`: **all tested FS variants pruned** (no positive quick gain).
- `mbpp`: baseline run failed under this setting (insufficient built examples; fixed in next round).
- `protein_contact`: baseline run failed under this setting (insufficient built examples; fixed in next round).
- `protein_ss`: clear positive regime found.
  - quick baseline: `24.66%`
  - best med-confirmed FS:
    - `scalar_l10_norm_node`: `32.69%` (**+8.02pp**)
    - `scalar_l10_trainable`: `32.38%` (**+7.72pp**)
    - `scalar_l10_sched_cos`: `32.23%` (**+7.57pp**)

Interpretation:

- FS is not broadly useful on every real task.
- FS can be strongly useful on selected protein sequence-labeling settings.

### Round21 (targeted follow-up, completed)

- Summary: `results/_summary_round21_targeted_search_s0.txt`
- Records: `results/_round21_targeted_search_records.jsonl`
- Goal:
  - repair previously failed task settings (`mbpp`, `protein_contact`)
  - keep serial + aggressive prune policy
  - retain only candidates with quick `d_acc >= +0.002` (`+0.20pp`)

Main outcome:

- `mbpp_fix`:
  - baseline quick: `10.46%`
  - med-confirmed `scalar_l8_trainable`: `24.07%` (**+13.61pp**)
- `protein_contact_fix`:
  - baseline quick: `98.83%`
  - all FS variants pruned (`+0.00pp`, no gain)
- `protein_ss_refine`:
  - baseline quick: `21.14%`
  - med-confirmed best:
    - `scalar_l10_sched_cos`: `34.45%` (**+13.31pp**)
    - `scalar_l10_trainable`: `33.95%` (**+12.82pp**)
    - `head_l10`: `33.48%` (**+12.35pp**)

## Notes

- This snapshot does not include full training logs/checkpoints due size.
- Full raw logs remain on AutoDL under:
  - `/root/autodl-tmp/future-seed-posttrain/runs/`

## Result Integrity Workflow

Run these checks before updating paper tables:

```bash
python3 scripts/summarize_all_results.py
python3 scripts/check_doc_summary_refs.py --strict
```

Generated artifacts:

- `results/_aggregate_results.jsonl`: one parsed row per summary metric line.
- `results/_aggregate_results.md`: table used for quick paper sync.
- `paper/exp_manifest.json`: canonical list of paper-facing experiments.

## Next Paper Iteration (execution order)

1. Re-run ARC options-first and Hotpot L4096 with 5 seeds under one fixed recipe.
2. Keep q-first controls in each domain (ARC/Hotpot/MBPP/protein) for causal-friendly sanity.
3. Extend MBPP eval from token-acc to executable pass rate (`exec_ok`, `tests_passed`).
4. Add failure appendix from `results/_aggregate_results.md` (negative and no-op regimes).
