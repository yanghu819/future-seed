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

### Round64 (mbpp+punc multiseed, completed)

- Script: `scripts/run_round64_mbpp_punc_multiseed.py`
- Summary: `results/_summary_round64_mbpp_punc_multiseed.txt`
- Records: `results/_round64_mbpp_punc_multiseed_records.jsonl`

Main outcomes:

- `mbpp_seed0_clip07_confirm`:
  - quick baseline `40.69%`
  - `head_l10_clip07`: `40.71%` (`+0.02pp`)
  - `head_l10_strong`: `40.71%` (`+0.02pp`)
  - med skipped (`best_quick +0.02pp < +0.20pp gate`)
- `mbpp_seed1_clip07_confirm`:
  - quick baseline `40.86%`
  - `head_l10_clip07`: `39.64%` (`-1.22pp`, pruned)
  - `head_l10_strong`: `39.64%` (`-1.22pp`, pruned)
  - med skipped
- `punc_seed3_scalar_confirm`:
  - quick baseline `11.76%`
  - `scalar_l8_sched_cos`: `10.71%` (`-1.04pp`, pruned)
  - `head_l8`: `9.96%` (`-1.80pp`, pruned)
  - med skipped

Interpretation:

- MBPP rescue recipe did not transfer to seed1 and was near-neutral on seed0.
- PUNC branch remained unstable and was fully pruned on seed3.

### Round65 (mbpp+squad seed2/seed3, completed)

- Script: `scripts/run_round65_mbpp_squad_seed23.py`
- Summary: `results/_summary_round65_mbpp_squad_seed23.txt`
- Records: `results/_round65_mbpp_squad_seed23_records.jsonl`

Main outcomes:

- `mbpp_seed3_regrescue`:
  - quick baseline `43.11%`
  - `head_l10_strong`: `42.35%` (`-0.76pp`)
  - `head_l10_clip07`: `40.22%` (`-2.89pp`)
  - med skipped
- `squad_strict_seed2`:
  - quick baseline `14.58%`
  - `scalar_l8_train1e4`: `15.59%` (`+1.01pp`)
  - `scalar_l8_sched_cos`: `14.07%` (`-0.51pp`, pruned)
  - med baseline `18.33%`
  - med FS (`scalar_l8_train1e4`) `19.04%` (**`+0.71pp` vs med baseline**)

Interpretation:

- SQuAD regained a clear positive branch with `scalar_l8_train1e4` on seed2.
- MBPP remained seed-sensitive and failed on seed3 under rescue setting.

### Round66 (squad+mbpp frontier, completed)

- Script: `scripts/run_round66_squad_mbpp_frontier.py`
- Summary: `results/_summary_round66_squad_mbpp_frontier.txt`
- Records: `results/_round66_squad_mbpp_frontier_records.jsonl`

Main outcomes:

- `squad_seed0_train1e4_confirm`:
  - quick baseline `13.33%`
  - quick `scalar_l8_train1e4`: `14.05%` (`+0.72pp`)
  - quick `scalar_l8_sched_cos`: `13.80%` (`+0.47pp`)
  - med baseline `17.27%`
  - med FS (`scalar_l8_train1e4`) `19.00%` (**`+1.73pp` vs med baseline**)
- `mbpp_strict_seed3_alt`:
  - quick baseline `40.90%`
  - quick `head_l10_strong`: `41.85%` (`+0.95pp`)
  - quick `head_l8_nodetach`: `41.45%` (`+0.56pp`)
  - med baseline `48.23%`
  - med FS (`head_l10_strong`) `46.76%` (**`-1.47pp` vs med baseline**)

Interpretation:

- SQuAD `scalar_l8_train1e4` now shows repeatable med positives on seed0 and seed2.
- MBPP still exhibits quick-positive/med-negative reversal risk on seed3.

Current bottom line (real-task branch, after Round66):

- `squad`:
  - seed0 med `+1.73pp`
  - seed2 med `+0.71pp`
  - seed1 remained negative in prior strict frontier
  - currently the strongest practical branch.
- `mbpp`:
  - still recipe- and seed-sensitive, with recent seed3 med regression (`-1.47pp`).
  - requires additional seed-level reconfirmation before any stable-gain claim.
- `punc_restore`:
  - high variance across seeds (strong positives and strong negatives), currently deprioritized.

### Round67 (squad3+mbpp0, completed)

- Script: `scripts/run_round67_squad3_mbpp0.py`
- Summary: `results/_summary_round67_squad3_mbpp0.txt`
- Records: `results/_round67_squad3_mbpp0_records.jsonl`

Main outcomes:

- `squad_seed3_train1e4_frontier`:
  - quick baseline `14.67%`
  - `scalar_l8_train1e4`: `14.06%` (`-0.61pp`, pruned)
  - `scalar_l8_sched_cos`: `13.83%` (`-0.84pp`, pruned)
  - med skipped
- `mbpp_seed0_reconfirm_strict`:
  - quick baseline `40.60%`
  - `head_l10_strong`: `42.88%` (`+2.27pp`)
  - `head_l8_nodetach`: `41.10%` (`+0.50pp`)
  - med baseline `48.39%`
  - med FS (`head_l10_strong`) `49.94%` (**`+1.54pp` vs med baseline**)

Interpretation:

- MBPP seed0 reconfirmed a strong med gain with `head_l10_strong`.
- SQuAD train1e4 showed clear seed split (positive on seed0/2, negative on seed3).

Current bottom line (real-task branch, after Round67):

- `mbpp`:
  - seed0 med `+1.55pp` (Round60) and reconfirm `+1.54pp` (Round67),
  - seed1 med `+0.35pp` (Round61),
  - seed2 med `-1.98pp` (Round62), seed3 med `-1.47pp` (Round66).
  - conclusion: useful on part of seeds but not yet seed-stable.
- `squad`:
  - seed0 med `+1.73pp` (Round66), seed2 med `+0.71pp` (Round65),
  - seed1/seed3 quick both negative and pruned.
  - conclusion: currently the strongest practical branch, but still seed-split.
- `punc_restore`:
  - retained as high-variance auxiliary signal; not used as main proof branch.

### Round68 (squad head rescue, completed)

- Script: `scripts/run_round68_squad_head_rescue.py`
- Summary: `results/_summary_round68_squad_head_rescue.txt`
- Records: `results/_round68_squad_head_rescue_records.jsonl`

Main outcomes:

- `squad_seed1_head_rescue`:
  - quick baseline `14.83%`
  - `head_l8_nodetach`: `14.58%` (`-0.25pp`)
  - `head_l8`: `14.07%` (`-0.76pp`, pruned)
  - med skipped (`best_quick -0.25pp < med_gate -0.20pp`)
- `squad_seed3_head_rescue`:
  - quick baseline `14.67%`
  - `head_l8_nodetach`: `13.90%` (`-0.76pp`, pruned)
  - `head_l8`: `13.32%` (`-1.35pp`, pruned)
  - med skipped

Interpretation:

- head-based rescue did not recover SQuAD on seed1/seed3.
- this branch is currently low value under the present budget regime.

### Round69 (mbpp+squad rescue, completed)

- Script: `scripts/run_round69_mbpp_squad_rescue.py`
- Summary: `results/_summary_round69_mbpp_squad_rescue.txt`
- Records: `results/_round69_mbpp_squad_rescue_records.jsonl`

Main outcomes:

- `mbpp_seed3_dualmed_rescue`:
  - quick baseline `40.90%`
  - quick `head_l10_strong`: `41.85%` (`+0.95pp`)
  - quick `head_l8_nodetach`: `41.45%` (`+0.56pp`)
  - med baseline `48.23%`
  - med `head_l8_nodetach`: `48.28%` (**`+0.05pp` vs med baseline**)
  - med `head_l10_strong`: `46.76%` (**`-1.47pp` vs med baseline**)
- `squad_seed1_scalar_micro`:
  - quick baseline `14.83%`
  - quick best `scalar_l8_train1e4`/`scalar_l8_train8e5`: `14.58%` (`-0.25pp`)
  - `scalar_l8_train1e4_clip07`: `14.41%` (`-0.42pp`)
  - med skipped (`best_quick -0.25pp < med_gate -0.10pp`)

Interpretation:

- MBPP seed3 can be repaired from clear negative to near-neutral (`+0.05pp`) with `head_l8_nodetach`, but still not strong gain.
- SQuAD seed1 remained negative after scalar micro tuning.

Current bottom line (real-task branch, after Round69):

- `squad`:
  - stable strong positives seen on seed0 (`+1.73pp`) and seed2 (`+0.71pp`);
  - seed1/seed3 remained negative across head/scalar rescues.
  - currently useful but seed-split.
- `mbpp`:
  - strong positives on seed0/seed1, strong negatives on seed2, and near-neutral repaired result on seed3.
  - currently classified as recipe- and seed-sensitive, not yet universally stable.

### Round70 (squad3+mbpp2, completed)

- Script: `scripts/run_round70_squad3_mbpp2.py`
- Summary: `results/_summary_round70_squad3_mbpp2.txt`
- Records: `results/_round70_squad3_mbpp2_records.jsonl`

Main outcomes:

- `squad_seed3_scalar_micro`:
  - quick baseline `14.67%`
  - best quick `scalar_l8_train8e5`: `14.41%` (`-0.26pp`)
  - other candidates: `-0.59pp`, `-0.61pp`
  - med skipped (`best_quick -0.26pp < med_gate -0.10pp`)
- `mbpp_seed2_dualmed_recheck`:
  - quick baseline `39.11%`
  - quick `head_l10_clip07`: `42.52%` (`+3.41pp`)
  - quick `head_l10_strong`: `42.52%` (`+3.41pp`)
  - med baseline `46.69%`
  - med `head_l10_clip07`: `48.97%` (**`+2.28pp` vs med baseline**)
  - med `head_l10_strong`: `48.97%` (**`+2.28pp` vs med baseline**)

Interpretation:

- MBPP seed2 rescue signal is reproducible and strong under current quick->med protocol.
- SQuAD seed3 remained negative even after scalar micro tuning.

Current bottom line (real-task branch, after Round70):

- `mbpp`:
  - confirmed strong med gains on seed0 (`+1.54pp`) and seed2 (`+2.28pp`),
  - seed3 currently near-neutral best (`+0.05pp`) under rescue,
  - still seed-sensitive but high-value.
- `squad`:
  - positive on seed0/seed2, persistently negative on seed1/seed3 under both head/scalar rescue.
  - currently useful but not seed-stable.

### Round71 (mbpp3+squad2, completed)

- Script: `scripts/run_round71_mbpp3_squad2.py`
- Summary: `results/_summary_round71_mbpp3_squad2.txt`
- Records: `results/_round71_mbpp3_squad2_records.jsonl`

Main outcomes:

- `mbpp_seed3_rescue384`:
  - quick baseline `43.11%`
  - best quick `head_l10_strong` `42.35%` (`-0.76pp`)
  - med skipped
- `squad_seed2_scalar_reconfirm`:
  - quick baseline `14.58%`
  - quick `scalar_l8_train1e4` `15.59%` (`+1.01pp`)
  - med baseline `18.33%`
  - med FS (`scalar_l8_train1e4`) `19.04%` (**`+0.71pp` vs med baseline**)

Interpretation:

- SQuAD seed2 positive med gain is reproducible.
- MBPP seed3 remained negative in this recipe neighborhood.

### Round72 (mbpp3+squad1 rescue, completed)

- Script: `scripts/run_round72_mbpp3_squad1_rescue.py`
- Summary: `results/_summary_round72_mbpp3_squad1_rescue.txt`
- Records: `results/_round72_mbpp3_squad1_rescue_records.jsonl`

Main outcomes:

- `mbpp_seed3_nodetach_rescue384`:
  - quick baseline `43.11%`
  - best quick `head_l10_nodetach_clip07` `41.35%` (`-1.76pp`)
  - med skipped
- `squad_seed1_lowpressure_rescue`:
  - quick baseline `14.83%`
  - quick `scalar_l8_train1e4` `14.83%` (`-0.00pp`)
  - med baseline `19.38%`
  - med FS `19.20%` (**`-0.18pp` vs med baseline**)

Interpretation:

- MBPP seed3 and SQuAD seed1 rescue both remained non-positive at med.

### Round73 (mbpp1+squad0 recheck, completed)

- Script: `scripts/run_round73_mbpp1_squad0_recheck.py`
- Summary: `results/_summary_round73_mbpp1_squad0_recheck.txt`
- Records: `results/_round73_mbpp1_squad0_recheck_records.jsonl`

Main outcomes:

- `mbpp_seed1_strict_recheck`:
  - med baseline `46.67%`
  - med FS (`head_l10_strong`) `47.01%` (**`+0.35pp`**)
- `squad_seed0_frontier_recheck`:
  - med baseline `17.27%`
  - med FS (`scalar_l8_train1e4`) `19.00%` (**`+1.73pp`**)

Interpretation:

- dual-task med positives reproduced cleanly under strict quick->med gating.

### Round74 (punc1+squad2 frontier, completed)

- Script: `scripts/run_round74_punc1_squad2_frontier.py`
- Summary: `results/_summary_round74_punc1_squad2_frontier.txt`
- Records: `results/_round74_punc1_squad2_frontier_records.jsonl`

Main outcomes:

- `punc_seed1_frontier_recheck`:
  - quick baseline `7.92%`
  - quick `scalar_l8_train1e4` `13.04%` (`+5.12pp`)
  - med baseline `12.20%`
  - med FS `13.04%` (**`+0.84pp`**)
- `squad_seed2_frontier_recheck`:
  - quick baseline `14.58%`
  - quick `scalar_l8_train1e4` `15.59%` (`+1.01pp`)
  - med baseline `18.33%`
  - med FS `19.04%` (**`+0.71pp`**)

Interpretation:

- `scalar_l8_train1e4` is a strong high-value branch across SQuAD/PUNC in current budget regime.

Current bottom line (real-task branch, after Round74):

- `squad`:
  - repeated med positives on seed0 (`+1.73pp`) and seed2 (`+0.71pp`),
  - seed1/seed3 remain negative in tested rescue neighborhoods.
- `mbpp`:
  - repeated positives on seed0 (`+1.54pp`), seed1 (`+0.35pp`), seed2 (`+2.28pp`),
  - seed3 remains unresolved/mostly negative.
- `punc_restore`:
  - still variance-sensitive, but round74 added a positive med confirmation (`+0.84pp`).

### Round75 (mbpp3+punc0 targeted, completed)

- Script: `scripts/run_round75_mbpp3_punc0_targeted.py`
- Summary: `results/_summary_round75_mbpp3_punc0_targeted.txt`
- Records: `results/_round75_mbpp3_punc0_targeted_records.jsonl`

Main outcomes:

- `mbpp_seed3_l8_refine`:
  - med baseline `48.23%`
  - med FS (`head_l8_nodetach`) `48.28%` (**`+0.05pp`**)
- `punc_seed0_frontier_recheck`:
  - quick `scalar_l8_train1e4` severe drop (`-4.17pp`, pruned)
  - med baseline `10.59%`
  - med FS (`head_l8`) `11.09%` (**`+0.51pp`**)

Interpretation:

- MBPP seed3 stays near-neutral under current budget and recipe.
- PUNC seed0 keeps a small positive path with `head_l8`; scalar train1e4 is seed-sensitive.

### Round76 (punc1+squad0 dualmed, completed)

- Script: `scripts/run_round76_punc1_squad0_dualmed.py`
- Summary: `results/_summary_round76_punc1_squad0_dualmed.txt`
- Records: `results/_round76_punc1_squad0_dualmed_records.jsonl`

Main outcomes:

- `punc_seed1_dualmed_compare`:
  - med baseline `10.88%`
  - med `scalar_l8_train1e4` `13.04%` (**`+2.16pp`**)
  - med `scalar_l8_sched_cos` `10.47%` (**`-0.41pp`**)
- `squad_seed0_dualmed_compare`:
  - med baseline `17.27%`
  - med `scalar_l8_train1e4` `19.00%` (**`+1.73pp`**)
  - med `scalar_l8_sched_cos` `17.47%` (**`+0.20pp`**)

Interpretation:

- `scalar_l8_train1e4` is now the most reliable high-value branch across current SQuAD/PUNC settings.
- dual-med protocol proved useful by exposing quick/med ranking reversals.

### Round77-82 (fastdiscover broad search, completed)

- Script: `scripts/run_round77_82_fastdiscover.py`
- Queue: `results/_search_queue_round77_82.json`
- Summaries:
  - `results/_summary_round77_fastdiscover.txt`
  - `results/_summary_round78_fastdiscover.txt`
  - `results/_summary_round79_fastdiscover.txt`
  - `results/_summary_round80_fastdiscover.txt`
  - `results/_summary_round81_fastdiscover.txt`
  - `results/_summary_round82_fastdiscover.txt`
- Records:
  - `results/_round77_fastdiscover_records.jsonl`
  - `results/_round78_fastdiscover_records.jsonl`
  - `results/_round79_fastdiscover_records.jsonl`
  - `results/_round80_fastdiscover_records.jsonl`
  - `results/_round81_fastdiscover_records.jsonl`
  - `results/_round82_fastdiscover_records.jsonl`

Main outcomes:

- Round77 (`hotpot_seed0_discovery`, `arc_seed0_discovery`):
  - `arc_seed0_discovery`: med `scalar_l8_train8e5` **`+0.29pp`**.
  - `hotpot_seed0_discovery`: med `scalar_l8_train8e5` **`-0.12pp`**.
  - prune: `hotpot scalar_l8_train1e4` quick **`-4.17pp`**.
- Round78 (`wiki_seed0_discovery`, `protein_ss_seed0_discovery`):
  - `protein_ss_seed0_discovery`: med `scalar_l8_train1e4` **`+3.60pp`**.
  - `wiki_seed0_discovery`: baseline failed (offline dataset not cached).
  - prune: `protein_ss scalar_l8_train8e5` quick **`-1.64pp`**.
- Round79 (`hotpot_seed1_discovery`, `protein_contact_seed0_discovery`):
  - `hotpot_seed1_discovery`: med `scalar_l8_train1e4` **`+2.16pp`**.
  - `protein_contact_seed0_discovery`: baseline failed (`n_val=200` exceeded buildable examples).
- Round80 (`arc_seed1_discovery`, `wiki_seed1_discovery`):
  - `arc_seed1_discovery`: best quick `scalar_l8_sched_cos` **`+0.41pp`**, med skipped (`< +0.80pp gate`).
  - `wiki_seed1_discovery`: baseline failed (offline dataset not cached).
  - prune: `arc scalar_l8_train1e4` **`-1.45pp`**, `scalar_l8_train8e5` **`-1.16pp`**.
- Round81 (`squad_seed2_anchor`, `punc_seed1_anchor`):
  - `squad_seed2_anchor`: med `scalar_l8_train1e4` **`+0.71pp`**.
  - `punc_seed1_anchor`: med `scalar_l8_train1e4` **`+2.16pp`**.
  - prune: `squad scalar_l8_sched_cos` quick **`-0.51pp`**.
- Round82 (`squad_seed0_anchor`, `mbpp_seed2_anchor`):
  - `mbpp_seed2_anchor`: med `scalar_l8_sched_cos` **`+0.89pp`**.
  - `squad_seed0_anchor`: best quick `scalar_l8_train1e4` **`+0.72pp`**, med skipped (`< +0.80pp gate`).

Fastdiscover bottom line (after Round82):

- Highest new-task gain in this pack: `protein_ss_seed0_discovery` **`+3.60pp`** (med).
- Additional useful-task entries (`med > baseline`): `hotpot_seed1_discovery` **`+2.16pp`**, `punc_seed1_anchor` **`+2.16pp`**, `mbpp_seed2_anchor` **`+0.89pp`**, `squad_seed2_anchor` **`+0.71pp`**, `arc_seed0_discovery` **`+0.29pp`**.
- Main blockers to fix next:
  - `wikitext` blocked by offline-mode cache miss.
  - `protein_contact` blocked by over-large `n_val` target for available data.

### Round89-94 (fastdiscover continuation, completed)

- Script: `scripts/run_round77_82_fastdiscover.py`
- Queues:
  - `results/_search_queue_round89_94_nowiki.json` (active continuation queue)
- Summaries:
  - `results/_summary_round89_fastdiscover.txt`
  - `results/_summary_round90_fastdiscover.txt`
  - `results/_summary_round91_fastdiscover.txt`
  - `results/_summary_round92_fastdiscover.txt`
  - `results/_summary_round93_fastdiscover.txt`
  - `results/_summary_round94_fastdiscover.txt`

Main outcomes:

- Round89-90 (`protein_contact` seed0/seed1 fix):
  - baseline and all FS quick were exact ties (`+0.00pp`), med skipped by gate.
  - interpretation: current contact-pair setup is saturated; low search value.
- Round91 (`hotpot_seed2_discovery`, `arc_seed2_discovery`):
  - `hotpot_seed2_discovery`: quick best `+1.10pp`, but med `-2.74pp` (failed confirmation).
  - `arc_seed2_discovery`: all quick candidates negative (`-0.51pp` to `-1.65pp`), all pruned.
- Round92 (`protein_ss_seed1_discovery`, `arc_easy_seed0_discovery`):
  - `protein_ss_seed1_discovery`: med `head_l8` **`+2.39pp`**.
  - `arc_easy_seed0_discovery`: baseline failed (dataset split incompatibility in current trainer/data path).
- Round93 (`squad_seed3_anchor`, `mbpp_seed0_anchor`):
  - `mbpp_seed0_anchor`: med `scalar_l8_train8e5` **`+2.03pp`**.
  - `squad_seed3_anchor`: quick best `-0.26pp`, med skipped.
- Round94 (`squad_seed1_anchor`, `punc_seed0_anchor`):
  - `punc_seed0_anchor`: quick best `+1.92pp`, med `-0.12pp` (failed confirmation).
  - `squad_seed1_anchor`: quick best `-0.25pp`, med skipped.

Fastdiscover continuation bottom line (after Round94):

- New useful-task gain found: `protein_ss_seed1_discovery` **`+2.39pp`** (med, `head_l8`).
- Anchor gain reconfirmed: `mbpp_seed0_anchor` **`+2.03pp`** (med, `scalar_l8_train8e5`).
- Low-value branches to deprioritize:
  - `protein_contact` (quick tie ceiling).
  - `arc_seed2`/`squad_seed1|3` (consistent quick negatives).
  - `hotpot_seed2` and `punc_seed0` (quick-positive but med-negative reversal).

### Round95-98 (focus queue, completed)

- Queue: `results/_search_queue_round95_98_focus.json`
- Summaries:
  - `results/_summary_round95_fastdiscover.txt`
  - `results/_summary_round96_fastdiscover.txt`
  - `results/_summary_round97_fastdiscover.txt`
  - `results/_summary_round98_fastdiscover.txt`

Main outcomes:

- Round95:
  - `mbpp_seed1_anchor`: quick best `+2.25pp`, but med `-1.51pp` (failed confirmation).
  - `protein_ss_seed2_discovery`: all quick non-positive, med skipped.
- Round96:
  - `protein_ss_seed3_discovery`: med `scalar_l8_train1e4` **`+1.68pp`**.
  - `mbpp_seed3_anchor`: all quick negative, med skipped.
- Round97:
  - `squad_seed2_anchor_recheck`: med `scalar_l8_train1e4` **`+0.71pp`** (reconfirmed).
  - `hotpot_seed3_discovery`: all quick negative, med skipped.
- Round98:
  - `punc_seed1_anchor_recheck`: med `scalar_l8_train1e4` **`+2.16pp`** (reconfirmed).
  - `arc_seed3_discovery`: all quick negative, med skipped.

Latest bottom line (after Round98):

- High-yield useful tasks:
  - `protein_ss`: seed0 **`+3.60pp`**, seed1 **`+2.39pp`**, seed3 **`+1.68pp`**.
  - `punc_seed1`: **`+2.16pp`** (reconfirmed).
  - `mbpp_seed0`: **`+2.03pp`** (anchor reconfirmed).
  - `squad_seed2`: **`+0.71pp`** (reconfirmed).
- Prune/freeze candidates:
  - `arc` seed2/seed3 branches (consistent quick negatives).
  - `hotpot` seed2/seed3 branches (quick->med instability or quick negatives).
  - `mbpp` seed1/seed3 branches (quick-positive but med-negative / quick-negative split).

### Round99-102 (focus2 queue, in progress)

- Queue:
  - `results/_search_queue_round99_102_focus2.json`
- Completed:
  - `results/_summary_round99_fastdiscover.txt`
  - `results/_summary_round100_fastdiscover.txt`
  - `results/_summary_round101_fastdiscover.txt`
  - `results/_summary_round102_fastdiscover.txt`
- Running:
  - none (queue finished)

Round99 outcomes:

- `protein_ss_seed4_discovery`:
  - quick baseline `30.30%`
  - med baseline `30.30%`
  - `scalar_l8_train8e5` med `35.20%` (**`+4.90pp`**)
  - quick-pruned: `scalar_l8_train1e4` (`-1.16pp`), `head_l8` (`-1.01pp`)
- `mbpp_seed2_headprobe`:
  - quick baseline `39.11%`
  - med baseline `46.69%`
  - `head_l10_strong` med `48.97%` (**`+2.28pp`**)

Round100 outcomes:

- `hotpot_seed1_headprobe`:
  - quick baseline `7.92%`
  - med baseline `10.88%`
  - `scalar_l8_train1e4` med `13.04%` (**`+2.16pp`**)
- `arc_seed0_headprobe`:
  - quick baseline `10.36%`
  - med baseline `12.33%`
  - `scalar_l8_train8e5` med `12.63%` (**`+0.29pp`**)

Round101 outcomes:

- `punc_seed1_anchor_recheck2`:
  - quick baseline `7.92%`
  - med baseline `10.88%`
  - `scalar_l8_train1e4` med `13.04%` (**`+2.16pp`**)
- `squad_seed2_anchor_recheck2`:
  - quick baseline `14.58%`
  - med baseline `18.33%`
  - `scalar_l8_train1e4` med `19.04%` (**`+0.71pp`**)

Round102 outcomes:

- `protein_ss_seed1_anchor_recheck`:
  - quick baseline `25.23%`
  - med baseline `32.23%`
  - `head_l8` med `34.62%` (**`+2.39pp`**)
- `mbpp_seed0_anchor_recheck2`:
  - quick baseline `40.69%`
  - med baseline `48.07%`
  - `scalar_l8_sched_cos` med `49.26%` (**`+1.20pp`**)

### Round103-104 (expand queue, in progress)

- Queue:
  - `results/_search_queue_round103_104_expand.json`
- Completed:
  - `results/_summary_round103_fastdiscover.txt`
  - `results/_summary_round104_fastdiscover.txt`
- Running:
  - none (queue finished)

Round103 outcomes:

- `protein_ss_seed5_discovery`:
  - quick baseline `23.08%`
  - med baseline `26.68%`
  - `head_l8` med `28.30%` (**`+1.62pp`**)
- `hotpot_seed4_discovery`:
  - quick baseline `8.78%`
  - quick best `scalar_l8_train1e4` `9.57%` (`+0.79pp`)
  - med skipped (`+0.79pp < +0.80pp gate`)

Round104 outcomes:

- `arc_seed4_discovery`:
  - quick baseline `9.07%`
  - med baseline `10.34%`
  - `scalar_l8_train1e4` med `13.07%` (**`+2.73pp`**)
- `mbpp_seed4_headprobe`:
  - quick baseline `40.61%`
  - med baseline `47.53%`
  - `scalar_l8_sched_cos` med `48.82%` (**`+1.29pp`**)

### Round105-106 (expand2 queue, completed)

- Queue:
  - `results/_search_queue_round105_106_expand2.json`
- Completed:
  - `results/_summary_round105_fastdiscover.txt`
  - `results/_summary_round106_fastdiscover.txt`

Round105 outcomes:

- `protein_ss_seed6_discovery`:
  - quick baseline `31.37%`
  - quick best `scalar_l8_train1e4` `31.82%` (`+0.44pp`)
  - med skipped (`+0.44pp < +0.80pp gate`)
- `arc_seed5_discovery`:
  - quick baseline `10.72%`
  - best quick `scalar_l8_train1e4` `8.54%` (`-2.19pp`)
  - all candidates quick-pruned / med skipped

Round106 outcomes:

- `mbpp_seed5_headprobe`:
  - quick baseline `41.36%`
  - quick best `head_l10_strong` `40.35%` (`-1.01pp`)
  - all candidates quick-pruned, med skipped
- `hotpot_seed5_discovery`:
  - quick baseline `8.41%`
  - quick best `scalar_l8_train1e4` `7.13%` (`-1.28pp`)
  - all candidates quick-pruned, med skipped

### Round107-112 (new-task expansion queue, in progress)

- Queue:
  - `results/_search_queue_round107_112_newtasks.json`
- Completed:
  - `results/_summary_round107_fastdiscover.txt`
  - `results/_summary_round108_fastdiscover.txt`
  - `results/_summary_round109_fastdiscover.txt`
- Running:
  - round `110` (`mbpp_seed6_headprobe`, `squad_seed4_anchor`)
- Auto-next queued:
  - `results/_search_queue_round113_120_iter.json`
  - remote chainer: waits for `107-112` completion, then launches `113-120` automatically

Round107 outcomes:

- `hotpot_longctx_seed0_discovery`:
  - quick baseline `8.89%`
  - all FS quick ties (`+0.00pp`)
  - med skipped (`+0.00pp < +0.80pp gate`)
- `arc_mc_seed0_discovery`:
  - quick baseline `37.50%`
  - all FS quick `33.33%` (`-4.17pp`)
  - all candidates quick-pruned / med skipped

Round108 outcomes:

- `mbpp_longctx_seed0_discovery`:
  - quick baseline failed
  - reason: insufficient buildable examples for current long-context filter (`built 374 < wanted 600`)
  - med skipped
- `protein_ss_seed7_discovery`:
  - quick baseline `32.91%`
  - quick best `head_l8` `30.35%` (`-2.56pp`)
  - all candidates quick-pruned / med skipped

Round109 outcomes:

- `hotpot_seed6_discovery`:
  - quick baseline `7.10%`
  - quick best `scalar_l8_train1e4` `9.96%` (`+2.86pp`)
  - med baseline `9.01%`
  - med FS `scalar_l8_train1e4` `12.59%` (**`+3.59pp`**)
- `arc_seed6_discovery`:
  - quick baseline `9.02%`
  - quick best `head_l8` `9.26%` (`+0.24pp`)
  - med skipped (`+0.24pp < +0.80pp gate`)

Planned expansion highlights:

- Introduce new trainer/task families:
  - `train_hotpot_longctx_sft.py` (`hotpot_longctx_seed0/1`)
  - `train_mbpp_longctx_sft.py` (`mbpp_longctx_seed0/1`)
  - `train_arc_mc_sft.py` (`arc_mc_seed0/1`)
- Keep fresh-seed discovery on known real-task winners:
  - `protein_ss_seed7_discovery`, `mbpp_seed6_headprobe`, `hotpot_seed6_discovery`, `arc_seed6_discovery`
- Keep anchor calibration light:
  - `squad_seed4_anchor`, `punc_seed2_anchor`
