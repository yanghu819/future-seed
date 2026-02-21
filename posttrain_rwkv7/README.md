# RWKV7 Future-Seed Post-Training Backup

This folder is a backup snapshot of the current Future-Seed post-training work on AutoDL.

## Contents

- `scripts/`: training, summarization, and run scripts used in the current iteration.
- `results/`: exported summary text files from completed runs.
- `paper/`: current paper/progress note.
  - `paper/FS_POSTTRAIN_PROGRESS_2026-02-19.md`: short status note.
  - `paper/DETAILED_EXPERIMENT_LOG.md`: full success/failure experiment record.

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

## Notes

- This snapshot does not include full training logs/checkpoints due size.
- Full raw logs remain on AutoDL under:
  - `/root/autodl-tmp/future-seed-posttrain/runs/`
