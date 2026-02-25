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

### Round22 (adaptive serial search, completed)

- Summary: `results/_summary_round22_adaptive_search_s0.txt`
- Records: `results/_round22_adaptive_search_records.jsonl`
- Outcome:
  - `mbpp_focus`: all tested FS variants regressed (`-0.92pp` to `-4.83pp`).
  - `protein_ss_expand`: small positive quick deltas (best `+1.50pp` quick), but mixed.
  - `sudoku4_refine`: strong gains; med best `+33.29pp`.
  - `sudoku9_probe`: small but consistent gains; med best `+1.51pp`.

### Round23 (real-task sweep, partially completed before network abort)

- Records: `results/_round23_real_task_sweep_records.jsonl`
- Launcher log: `results/_launcher_round23.log`
- Outcome (completed parts):
  - `mbpp_rt`: all FS quick variants regressed (`-1.28pp` to `-4.67pp`).
  - `hotpot_rt`: FS quick variants were exact ties to baseline (`+0.00pp`).
  - `punc_restore_rt`: aborted due HF connectivity while probing (not a model failure).

### Round24 (punc + protein continuation, completed)

- Summary: `results/_summary_round24_punc_protein_s0.txt`
- Records: `results/_round24_punc_protein_records.jsonl`
- Outcome:
  - `protein_ss_rt`: med `scalar_l10_sched_cos` reached `34.31%` (**+4.14pp** over quick baseline).
  - `punc_restore_rt`: baseline failed by OOM at `bsz=10` (configuration issue).

### Round25 (punc salvage with memory-safe config, completed)

- Summary: `results/_summary_round25_punc_salvage_s0.txt`
- Records: `results/_round25_punc_salvage_records.jsonl`
- Outcome:
  - memory-safe setting (`bsz=2`, shorter prompt/answer) removed OOM and produced stable gains.
  - quick best: `+0.80pp` (`scalar_l8_sched_cos`, `head_l8`).
  - med best: `head_l8` `12.64%` (**+3.45pp**), `scalar_l8_sched_cos` `11.90%` (**+2.71pp**).

### Round26 (low-throughput MBPP/Hotpot check, completed)

- Summary: `results/_summary_round26_mbpp_hotpot_lowthroughput_s0.txt`
- Records: `results/_round26_mbpp_hotpot_lowthroughput_records.jsonl`
- Outcome:
  - `mbpp_low`:
    - baseline quick: `10.46%`
    - quick `scalar_l8_trainable`: `+1.00pp`
    - med `scalar_l8_trainable`: `29.64%` (**+19.17pp**)
  - `hotpot_low`:
    - baseline quick: `14.34%`
    - `scalar_l10_trainable`: `+0.00pp`
    - `scalar_l10_sched_cos`: `+0.00pp`
    - `head_l10`: `-1.84pp`

### Round27 (seed robustness check for positive regimes, completed)

- Summary: `results/_summary_round27_seedcheck_positive_s012.txt`
- Records: `results/_round27_seedcheck_positive_s012_records.jsonl`
- Outcome:
  - `mbpp_low + scalar_l8_trainable` (quick):
    - s0: `+1.00pp`, s1: `+0.32pp`, s2: `-0.82pp`
    - mean: `+0.17pp`, positive seeds: `2/3`
  - `punc_restore + head_l8` (quick):
    - s0: `+0.80pp`, s1: `+0.58pp`, s2: `+2.20pp`
    - mean: `+1.19pp`, positive seeds: `3/3`

### Round28 (MBPP throughput sweep, completed)

- Summary: `results/_summary_round28_mbpp_bsz_sweep_s0.txt`
- Records: `results/_round28_mbpp_bsz_sweep_s0_records.jsonl`
- Outcome:
  - `bsz=2`:
    - baseline quick: `10.46%`
    - FS quick: `+1.00pp`
    - FS med: `25.39%` (**+14.92pp**)
  - `bsz=4`: FS quick `-2.05pp`
  - `bsz=6`: FS quick `-2.08pp`
  - `bsz=8`: baseline OOM/fail
  - key conclusion: MBPP FS gain is concentrated in low-throughput (`bsz=2`) regime.

### Round29 (punc seed-5 stability, completed)

- Summary: `results/_summary_round29_punc_seed5_s01234.txt`
- Records: `results/_round29_punc_seed5_s01234_records.jsonl`
- Outcome (quick):
  - seed deltas: `+0.80pp`, `+0.58pp`, `+2.20pp`, `+2.41pp`, `+0.64pp`
  - mean: **+1.33pp**, positive seeds: **5/5**
- Interpretation:
  - confirms punc restoration as the most stable non-synthetic positive FS regime in current search.

### Round30 (embedding smoke: Hotpot retrieval, completed)

- Summary: `results/_summary_round30_embedding_hotpot_s0.txt`
- Records: `results/_round30_embedding_hotpot_s0_records.jsonl`
- Setup:
  - retrieval-style contrastive probe on Hotpot pairs (`question -> context`)
  - frozen RWKV backbone, train lightweight embedding head
  - compare `baseline` vs `fs` (`fs_layer_start=8`, scalar gate trainable)
- Outcome:
  - baseline: `R@1=1.17%`, `R@5=3.12%`, `MRR@10=1.93%`
  - fs: `R@1=0.78%`, `R@5=3.12%`, `MRR@10=1.69%`
  - delta (FS - baseline): `d_R@1=-0.39pp`, `d_MRR@10=-0.24pp`
  - interpretation: this first embedding smoke does **not** show FS benefit.

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

1. Convert MBPP eval from token-acc to executable pass metrics (`exec_ok`, `tests_passed`).
2. Keep MBPP at low-throughput (`bsz=2`) for FS-positive branch; treat high-throughput branch as separate negative regime.
3. Keep punc-restore multi-seed tracking as a stable positive real-text regime.
4. Retest Hotpot with matched low-throughput recipe before any FS variant expansion.
5. For embedding direction: try asymmetry-aware objectives / longer-doc pooling before claiming FS helps embeddings.

## Latest Rapid-Iteration Update (2026-02-25, Round60/61/62)

### Round60 (strict-quick, seed0, completed)

- Script: `scripts/run_round60_strictquick_s0.py`
- Summary: `results/_summary_round60_strictquick_s0.txt`
- Records: `results/_round60_strictquick_s0_records.jsonl`

Main outcomes:

- `mbpp_strict`:
  - quick baseline `42.70%`
  - quick best `head_l10_strong`: `43.54%` (`+0.84pp`)
  - med baseline: `48.39%`
  - med FS (`head_l10_strong`): `49.94%` (**`+1.55pp` vs med baseline**)
- `squad_strict`:
  - quick baseline `14.17%`
  - quick best `scalar_l8_sched_cos`: `15.45%` (`+1.28pp`)
  - med baseline: `17.27%`
  - med FS (`scalar_l8_sched_cos`): `17.47%` (**`+0.20pp` vs med baseline**)

Interpretation:

- strict recipe recovered positive med gains on both real tasks at seed0.
- MBPP strict is currently the strongest post-training positive evidence branch.

### Round61 (strict frontier, seed1, completed)

- Script: `scripts/run_round61_strict_seed1_frontier.py`
- Summary: `results/_summary_round61_strict_seed1_frontier.txt`
- Records: `results/_round61_strict_seed1_frontier_records.jsonl`

Search policy:

- single-GPU serial execution
- quick stage first (`150s`)
- med stage only when quick best `>= +0.30pp`

Main outcomes:

- `squad_strict_seed1`:
  - quick baseline `14.83%`
  - best FS quick `scalar_l8_train1e4`: `14.58%` (`-0.25pp`)
  - all tested FS quick variants were negative (`-0.25pp`, `-0.50pp`, `-0.75pp`)
  - med was pruned (`quick_d_acc=-0.25pp < +0.30pp`)
- `mbpp_strict_seed1`:
  - quick baseline `40.43%`
  - quick candidates:
    - `head_l10_strong`: `43.86%` (`+3.43pp`)
    - `head_l10_midlr`: `42.33%` (`+1.90pp`)
    - `head_l10_sched_cos`: `41.86%` (`+1.43pp`)
  - med baseline: `46.67%`
  - med FS (`head_l10_strong`): `47.01%` (**`+0.35pp` vs med baseline**)

Interpretation:

- MBPP strict remains positive on seed1, but gain magnitude is reduced vs seed0.
- SQuAD strict does not yet show seed-stable gain under current strict recipe.

### Round62 (3h finishpack, completed)

- Script: `scripts/run_round62_3h_finishpack.py`
- Summary: `results/_summary_round62_3h_finishpack.txt`
- Records: `results/_round62_3h_finishpack_records.jsonl`

Search policy:

- hard 3-hour execution window
- single-GPU serial quick->med
- immediate prune when quick drop below baseline exceeds `0.50pp`

Main outcomes:

- `mbpp_strict_seed2_confirm`:
  - quick baseline `41.36%`
  - quick `head_l10_strong`: `42.10%` (`+0.74pp`)
  - med baseline: `47.15%`
  - med FS (`head_l10_strong`): `45.17%` (**`-1.98pp` vs med baseline**)
  - `head_l10_midlr` quick pruned: `-1.26pp`
- `squad_strict_seed1_rescue`:
  - quick baseline `14.83%`
  - best rescue quick `scalar_l8_sched_ultra`: `14.82%` (`-0.01pp`)
  - other rescue candidates: `-0.50pp`, `-1.00pp`
  - med pruned (`best_quick=-0.01pp < +0.20pp gate`)
- `punc_restore_seed0_scout`:
  - quick baseline `8.44%`
  - quick `head_l8`: `10.10%` (`+1.66pp`)
  - med baseline: `10.59%`
  - med `head_l8`: `11.09%` (**`+0.51pp` vs med baseline**)

Interpretation:

- MBPP strict is not stable across seeds under current recipe (seed2 flipped negative).
- SQuAD strict rescue did not recover positive gain.
- PUNC restore remains a practical positive branch in this budget regime.

### Round63 (useful-followup, completed)

- Script: `scripts/run_round63_useful_followup.py`
- Summary: `results/_summary_round63_useful_followup.txt`
- Records: `results/_round63_useful_followup_records.jsonl`

Search policy:

- continue single-GPU serial quick->med
- focus on usefulness verification:
  - `punc_restore` seed1/seed2 confirmation
  - `mbpp` seed2 regularization rescue

Main outcomes:

- `mbpp_seed2_regrescue`:
  - quick baseline `39.11%`
  - quick best `head_l10_clip07`: `42.52%` (`+3.41pp`)
  - med baseline: `46.69%`
  - med FS (`head_l10_clip07`): `48.97%` (**`+2.28pp` vs med baseline**)
- `punc_restore_seed1_confirm`:
  - quick baseline `7.92%`
  - quick best `scalar_l8_sched_cos`: `10.34%` (`+2.42pp`)
  - med baseline: `12.20%`
  - med FS (`scalar_l8_sched_cos`): `14.32%` (**`+2.12pp` vs med baseline**)
- `punc_restore_seed2_confirm`:
  - quick baseline `6.74%`
  - quick best `scalar_l8_sched_cos`: `7.13%` (`+0.39pp`)
  - med baseline: `12.78%`
  - med FS (`scalar_l8_sched_cos`): `10.24%` (**`-2.55pp` vs med baseline**)
  - `head_l8` was quick-pruned (`-0.61pp`)

Interpretation:

- MBPP shows a viable rescue path on seed2 when FS is regularized (`fs_clip=0.7`).
- PUNC still has seed-level variance: strong positive on seed1, strong negative on seed2.

Current bottom line (real-task branch):

- `mbpp`:
  - original strict recipe was unstable on seed2 (`-1.98pp` in Round62),
  - but rescue recipe (`head_l10_clip07`) recovered to `+2.28pp` in Round63.
  - currently classified as promising but recipe-sensitive.
- `punc_restore`:
  - seed1 confirm is strongly positive (`+2.12pp` med),
  - seed2 confirm is negative (`-2.55pp` med).
  - currently classified as seed-variant, still needs one more seed to settle.
- `squad_strict`:
  - no stable positive signal under current budget/regimes.

Next round plan:

1. MBPP: promote `head_l10_clip07` into seed1/seed0 replay to test whether rescue recipe is consistently better than old strict winner.
2. PUNC: run one additional seed confirmation (`seed3`) on `scalar_l8_sched_cos` to decide keep-or-freeze.
