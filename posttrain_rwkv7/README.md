# RWKV7 Future-Seed Post-Training Backup

This folder is a backup snapshot of the current Future-Seed post-training work on AutoDL.

## Contents

- `scripts/`: training, summarization, and run scripts used in the current iteration.
- `results/`: exported summary text files from completed runs.
- `paper/`: current paper/progress note.

## Key Findings (current snapshot)

- ARC-Challenge options-first: stable positive gain with FS (`_summary_arc_optionsfirst_stabilized_r2.txt`).
- ARC-Challenge question-first: no meaningful gain (`_summary_arc_qfirst_stabilized_r3.txt`).
- HotpotQA:
  - L=2048: unstable / near-zero average gain.
  - L=4096: high variance; some seeds improve strongly, one seed regresses.

## Notes

- This snapshot does not include full training logs/checkpoints due size.
- Full raw logs remain on AutoDL under:
  - `/root/autodl-tmp/future-seed-posttrain/runs/`

