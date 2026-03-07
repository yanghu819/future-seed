# RWKV7 Future-Seed Post-Training

This directory is the local archival snapshot of the Future-Seed post-training campaign on a single 4090.

## Final Status (2026-03-06, through round808)

Current state:
- search is intentionally closed through `round808`
- latest local snapshot artifacts live in `runs/`
- current work is documentation, table cleanup, and repo hygiene, not more BFS under the same recipe
- legacy AutoDL-era scripts and chronology remain for provenance only; see [`LEGACY_AUTODL.md`](LEGACY_AUTODL.md)

## Current Best Evidence

| Task family | Best med gain | Med median | Positive med count | Current judgment |
|---|---:|---:|---:|---|
| `protein_ss_spot` | `+8.14pp` | `+2.18pp` | `103/126` | strongest repeatable real-task family |
| `hotpot_text_restore` | `+4.20pp` | `+0.54pp` | `35/53` | small but repeatable positive line |
| `mbpp_longctx_probe` | `+10.00pp` | `+0.91pp` | `53/87` | promising, but strict confirmation failed |
| `arc_mc_probe` | `+20.83pp` | `+4.17pp` | `70/116` | high upside, high variance |
| `squad_text_restore` | `+7.31pp` | `+0.55pp` | `24/35` | mixed, not locked |
| `punc_restore` | `+2.16pp` | `+0.36pp` | `16/24` | small support signal only |
| `graph_color` | `+8.33pp` | `+0.00pp` | `4/10` | useful diagnostic task only |
| `tsp_mask` | `+25.00pp` | `+2.08pp` | `2/4` | appendix-only spike |

Low-ROI or negative families under the current recipe:
- `protein_contact`
- `wiki`
- `hotpot_longctx`
- `countdown`
- `nqueens`
- `zebra`
- `sat3`

## Final Closure And Breadth Pivot

### Closure window (`round783-802`)

- `mbpp_longctx_probe` stayed promising but did not pass strict final confirmation:
  - exploit positives: `round785 +5.01pp`, `round786 +0.54pp`, `round788 +0.97pp`, `round789 +1.87pp`
  - strict confirms failed to promote:
    - `round799`: quick `+0.38pp`
    - `round800`: quick `+0.60pp`
- `arc_mc_probe` produced large positives but stayed unstable:
  - positives: `round785 +3.12pp`, `round787 +9.38pp`, `round788 +3.12pp`, `round790 +6.25pp`
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
2. In post-training, `protein_ss_spot` is the strongest repeatable real-task family under the current recipe.
3. `hotpot_text_restore`, `squad_text_restore`, and `punc_restore` are supporting positive signals.
4. `mbpp_longctx_probe`, `arc_mc_probe`, and `tsp_mask` remain mixed or exploratory rather than stable headline evidence.

## What The Repo Should Not Claim

1. real-task gains are already stable across held-out confirmation seeds
2. `mbpp_longctx_probe` was strictly confirmed by the final confirmation queue
3. `arc_mc_probe` is already robust enough for a clean stability claim
4. constraint-task wins alone prove the real-task story

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
- `scripts/`: launchers, orchestrators, and trainers
- `LEGACY_AUTODL.md`: boundary note for preserved AutoDL-era history

## Older Notes

The historical per-round notes below are kept as archive chronology. They may still mention legacy remote or AutoDL-era paths; see [`LEGACY_AUTODL.md`](LEGACY_AUTODL.md) for the boundary between supported local usage and preserved history.

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

### Round107-112 (new-task expansion queue, completed)

- Queue:
  - `results/_search_queue_round107_112_newtasks.json`
- Completed:
  - `results/_summary_round107_fastdiscover.txt`
  - `results/_summary_round108_fastdiscover.txt`
  - `results/_summary_round109_fastdiscover.txt`
- `results/_summary_round110_fastdiscover.txt`
- `results/_summary_round111_fastdiscover.txt`
- `results/_summary_round112_fastdiscover.txt`

Execution continuity:
  - `results/_search_queue_round113_120_iter.json`
  - remote chainer triggered after round112 and launched `113-120`

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

Round110 outcomes:

- `mbpp_seed6_headprobe`:
  - quick baseline `42.61%`
  - quick best `scalar_l8_sched_cos` `42.54%` (`-0.07pp`)
  - med skipped (`-0.07pp < +0.80pp gate`)
- `squad_seed4_anchor`:
  - quick baseline failed (run error), med skipped

Round111 outcomes:

- `punc_seed2_anchor`:
  - quick baseline `9.16%`
  - all candidates negative (`-0.63pp` to `-0.70pp`)
  - all quick-pruned / med skipped
- `arc_mc_seed1_discovery`:
  - quick baseline `33.33%`
  - all candidates tie (`+0.00pp`)
  - med skipped

Round112 outcomes:

- `hotpot_longctx_seed1_discovery`:
  - quick baseline `6.67%`
  - quick best `scalar_l8_train1e4` `8.89%` (`+2.22pp`)
  - med baseline `6.67%`
  - med FS `scalar_l8_train1e4` `8.89%` (**`+2.22pp`**)
- `mbpp_longctx_seed1_discovery`:
  - quick baseline failed (same longctx buildability issue), med skipped

### Round113-120 (iter queue, in progress)

- Queue:
  - `results/_search_queue_round113_120_iter.json`
- Completed:
  - `results/_summary_round113_fastdiscover.txt`
- Running:
  - round `114` (`mbpp_seed7_headprobe`, `arc_seed7_discovery`)

Round113 outcomes:

- `hotpot_seed7_discovery`: all quick ties (`+0.00pp`), med skipped.
- `protein_ss_seed8_discovery`: all quick ties (`+0.00pp`), med skipped.

Planned expansion highlights:

- Introduce new trainer/task families:
  - `train_hotpot_longctx_sft.py` (`hotpot_longctx_seed0/1`)
  - `train_mbpp_longctx_sft.py` (`mbpp_longctx_seed0/1`)
  - `train_arc_mc_sft.py` (`arc_mc_seed0/1`)
- Keep fresh-seed discovery on known real-task winners:
  - `protein_ss_seed7_discovery`, `mbpp_seed6_headprobe`, `hotpot_seed6_discovery`, `arc_seed6_discovery`
- Keep anchor calibration light:
  - `squad_seed4_anchor`, `punc_seed2_anchor`

### Round121-128 (fastloop queue, running)

- Queue:
  - `results/_search_queue_round121_128_fastloop.json`
- Completed:
  - `results/_summary_round121_fastdiscover.txt`
- Running:
  - round `122` (`hotpot_seed14_discovery`, `protein_ss_seed11_discovery`)

Round121 outcomes:

- `hotpot_seed13_discovery`:
  - quick baseline `5.56%`
  - med baseline `9.40%`
  - `scalar_l8_train1e4` med `10.97%` (**`+1.57pp`**)
- `protein_ss_seed10_discovery`:
  - quick baseline `23.93%`
  - med baseline `27.66%`
  - `scalar_l8_train8e5` med `33.62%` (**`+5.96pp`**)

### Round129-136 (next fastloop queue, chained)

- Queue prepared:
  - `results/_search_queue_round129_136_fastloop.json`
- Remote chaining:
  - auto-chainer waits for round128 completion then launches round129-136 automatically
- Search intent:
  - broaden tasks beyond repeated lines (`arc_mc`, `protein_contact`, `hotpot`, `protein_ss`, `mbpp_headprobe`) while keeping light anchors (`punc`, `mbpp`).

### Round122-136 (fastloop queues, completed)

Key outcomes (med vs baseline, pp):
- `arc_mc_seed3_discovery`: `scalar_l8_sched_cos` **`+12.50pp`** (new best)
- `mbpp_seed11_anchor`: `scalar_l8_sched_cos` **`+3.68pp`**
- `mbpp_longctx_seed2_repair`: `scalar_l8_train1e4` **`+1.32pp`**
- `punc_seed7_anchor`: `scalar_l8_train1e4` **`+1.13pp`**
- `punc_seed4_anchor`: `scalar_l8_train8e5` **`+0.80pp`**

Main pruned/negative patterns:
- `hotpot_longctx` seeds mostly `+0.00pp` (failed promote gate)
- `protein_ss` seeds `11/12/13/14/15` mostly negative quick (med skipped)
- several `hotpot` fresh seeds stayed below promote gate (`<= +0.78pp`)

### Round137-144 (fastloop queue, running)

- Queue:
  - `results/_search_queue_round137_144_fastloop.json`
- Running:
  - round `137` (`arc_mc_seed4_discovery`, `mbpp_seed12_anchor`)
- Search focus:
  - leverage new high-value branch (`arc_mc + scalar_l8_sched_cos`) while keeping broad discovery (`protein_ss/hotpot/mbpp_longctx/arc`).

### Round151-158 (fastloop queues, completed)

Key outcomes (med vs baseline, pp):
- `arc_mc_seed13_discovery`: `scalar_l8_train8e5` **`+20.83pp`** (new global best)
- `protein_ss_seed25_discovery`: `scalar_l8_train8e5` **`+6.04pp`**
- `hotpot_seed26_discovery`: `scalar_l8_train8e5` **`+4.20pp`**
- `protein_ss_seed24_discovery`: `scalar_l8_train8e5` **`+3.92pp`**
- `mbpp_longctx_seed4_repair`: `scalar_l8_train1e4` **`+3.00pp`**
- `mbpp_seed18_headprobe`: `scalar_l8_sched_cos` **`+2.95pp`**

Follow-up status:
- queue `results/_search_queue_round153_160_fastloop.json` completed through round160.
- chained queue `results/_search_queue_round161_168_fastloop.json` launched after round160.

### Round159-162 (fastloop queues, completed)

Key outcomes (med vs baseline, pp):
- `arc_mc_seed14_discovery`: `scalar_l8_train1e4` **`+8.33pp`**
- `mbpp_seed20_headprobe`: `head_l10_strong` **`+2.87pp`**
- `protein_ss_seed27_discovery`: `scalar_l8_train1e4` **`+1.95pp`**
- `punc_seed14_anchor`: `head_l8` **`+0.17pp`**

Main pruned/negative patterns:
- `hotpot_seed27_discovery`: best quick `-0.26pp`, failed promote gate (`+0.80pp`), med skipped.
- `hotpot_seed27_discovery`: `scalar_l8_train1e4` (`-2.94pp`) and `head_l8` (`-3.08pp`) quick-pruned.
- `mbpp_seed21_headprobe`: quick passed (`+0.92pp`) but med regressed to **`-0.67pp`** vs baseline (cooldown applied).

Running/chaining:
- active queue: `results/_search_queue_round161_168_fastloop.json` (currently in round163+)
- next queue prepared: `results/_search_queue_round169_176_fastloop.json` (auto-start after round168)
- next-next queue prepared: `results/_search_queue_round177_184_fastloop.json` (auto-start after round176)

### Round163 (fastdiscover, completed)

Key outcomes (med vs baseline, pp):
- `arc_mc_seed16_discovery`: `scalar_l8_sched_cos` **`+16.67pp`**
- `protein_ss_seed28_discovery`: `scalar_l8_train1e4` **`+1.40pp`**

Current status:
- active queue has moved to `round164`.
- continuation queue prepared: `results/_search_queue_round185_192_fastloop.json` (auto-start after round184).

### Round164 (fastdiscover, completed)

Key outcomes (med vs baseline, pp):
- `mbpp_seed22_anchor`: `head_l10_strong` quick `+2.15pp` but med **`-0.58pp`** (regression)
- `punc_seed15_anchor`: best quick `+0.07pp`, below promote gate; med skipped

Current status:
- active queue has moved to `round165` (`arc_mc_seed17_discovery`, `hotpot_seed28_discovery`).
- auto-chain is armed for `169-176`, `177-184`, `185-192` with no idle gap.

### Round165 (fastdiscover, completed)

Key outcomes (med vs baseline, pp):
- `hotpot_seed28_discovery`: `scalar_l8_train8e5` **`+0.57pp`** (med positive but below `+0.80pp` quick promote target)
- `arc_mc_seed17_discovery`: all FS candidates quick **`-8.33pp`** vs baseline; med skipped

Current status:
- active queue has moved into `round166` (`protein_ss_seed29_discovery`, `mbpp_longctx_seed6_repair`).
- chained queues remain active and waiting: `169-176`, `177-184`, `185-192`, `193-200`.

### Round166-176 (fastloop queues, completed)

Key outcomes (med vs baseline, pp):
- `arc_mc_seed18_discovery`: `scalar_l8_train1e4` **`+8.33pp`**
- `arc_mc_seed20_discovery`: `scalar_l8_train1e4` **`+8.33pp`**
- `arc_mc_seed19_discovery`: `scalar_l8_train8e5` **`+4.17pp`**
- `protein_ss_seed33_discovery`: `scalar_l8_train1e4` **`+1.42pp`**
- `punc_seed18_anchor`: `head_l8` **`+1.22pp`**

Main pruned/negative patterns:
- `arc_mc_seed17`/`22` showed strong baseline spike and FS collapse (multiple `-8.33pp` to `-16.67pp` quick drops).
- `protein_ss_seed30/34` mostly quick negative and med skipped.
- `mbpp_seed23/25/26` and `mbpp_longctx_seed6` had quick positives but med regressions (`-1.14pp` to `-4.68pp`).

Current status:
- `169-176` completed and auto-chained into active queue `177-184`.
- waiting chains remain active for `185-192`, `193-200`, `201-208`.

### Round177 (fastdiscover, completed)

Key outcomes (med vs baseline, pp):
- `arc_mc_seed23_discovery`: best quick `+0.00pp`, med skipped
- `protein_ss_seed35_discovery`: best quick `-0.47pp`, med skipped

Current status:
- active queue is now in `round178` (`mbpp_seed27_headprobe`, `hotpot_seed31_discovery`).
- downstream chains remain active: `185-192`, `193-200`, `201-208`, `209-216`.

### Round178 (fastdiscover, completed)

Key outcomes (med vs baseline, pp):
- `hotpot_seed31_discovery`: quick best `+1.40pp` but med **`-0.55pp`** vs baseline
- `mbpp_seed27_headprobe`: best quick `+0.04pp`, med skipped

Current status:
- active queue moved to `round179` (`arc_mc_seed24_discovery`, `protein_ss_seed36_discovery`).
- chained queues active: `185-192`, `193-200`, `201-208`, `209-216`, `217-224`, `225-232`, `233-240`.

### Round179 (fastdiscover, completed)

Key outcomes (med vs baseline, pp):
- `arc_mc_seed24_discovery`: `scalar_l8_train1e4` med **`+0.00pp`** (flat)
- `protein_ss_seed36_discovery`: all quick candidates **`+0.00pp`**, med skipped

Current status:
- active queue moved to `round180` (`mbpp_seed28_anchor`, `punc_seed19_anchor`).
- queued auto-chain remains continuous through `240`.

### Round180-207 (fastloop queues, completed)

Key outcomes (med vs baseline, pp):
- `arc_mc_seed31_discovery`: `scalar_l8_sched_cos` **`+8.33pp`**
- `arc_mc_seed32_discovery`: `scalar_l8_sched_cos` **`+8.33pp`**
- `arc_mc_seed28_discovery`: `scalar_l8_train8e5` **`+4.17pp`**
- `protein_ss_seed45_discovery`: `scalar_l8_train1e4` **`+3.58pp`**
- `mbpp_seed37_anchor`: `scalar_l8_sched_cos` **`+2.55pp`**
- `protein_ss_seed48_discovery`: `scalar_l8_train8e5` **`+2.46pp`**

Main pruned/negative patterns:
- `arc_mc` has intermittent baseline spikes: some seeds jump strongly positive while others collapse to quick negatives (e.g. `-8.33pp` / `-16.67pp`).
- several `hotpot/mbpp/longctx` rounds show quick gains that fail to transfer to med (flat or negative med deltas).
- many rounds from `199-202` were med-skipped by promote gate (weak quick deltas).

Current status:
- active queue moved to `209-216` (now running).
- chained queues remain armed and waiting: `217-224`, `225-232`, `233-240`, `241-248`.

### Nonstop Extension

- Added pre-queued fastloop blocks through `round368` (`249-256` ... `361-368`).
- All corresponding remote auto-chainer scripts are started in waiting mode to keep serial execution continuous.

### Round208 (fastdiscover, completed)

Key outcomes (med vs baseline, pp):
- `arc_mc_seed38_discovery`: best quick `+0.00pp`, med skipped
- `protein_ss_seed50_discovery`: all quick candidates `+0.00pp`, med skipped

### Round209-214 (fastloop queues, completed)

Key outcomes (med vs baseline, pp):
- `arc_mc_seed39_discovery`: `scalar_l8_train1e4` **`+8.33pp`**
- `mbpp_longctx_seed12_repair`: `scalar_l8_train8e5` **`+4.37pp`**
- `arc_mc_seed40_discovery`: `scalar_l8_sched_cos` **`+4.17pp`**
- `arc_mc_seed41_discovery`: `scalar_l8_sched_cos` **`+4.17pp`**
- `hotpot_seed39_discovery`: `scalar_l8_train1e4` **`+3.07pp`**

Main pruned/negative patterns:
- `hotpot_seed40_discovery` candidates all quick negative (up to `-3.20pp`), med skipped.
- `mbpp_seed39_headprobe` and several anchor tasks showed quick gains but med regression/skip.
- `protein_ss_seed51/53` mostly failed promote gate (near-flat quick).

Current status:
- active queue remains `209-216`, currently in `round215`.
- downstream chain is armed continuously through `round248`.

### Round215-227 (fastloop queues, completed)

Key outcomes (med vs baseline, pp):
- `mbpp_longctx_seed13_repair`: `scalar_l8_train8e5` **`+10.00pp`**
- `protein_ss_seed60_discovery`: `head_l8` **`+5.42pp`**
- `protein_ss_seed56_discovery`: `scalar_l8_train1e4` **`+4.75pp`**
- `mbpp_seed45_headprobe`: `head_l10_strong` **`+4.64pp`**
- `arc_mc_seed45_discovery`: `scalar_l8_train1e4` **`+4.17pp`**
- `arc_mc_seed47_discovery`: `scalar_l8_sched_cos` **`+4.17pp`**

Main pruned/negative patterns:
- multiple anchor rounds (`215/217/218/220`) failed promote gate and were med-skipped.
- `hotpot` remains unstable, with many rounds not converting to med-positive gains.

Current status:
- active queue has moved to `225-232` and is currently in `round228`.
- downstream chain remains armed through `round248` (and staged templates continue beyond).

### Round228-229 (fastloop queues, completed)

Key outcomes (med vs baseline, pp):
- `hotpot_seed44_discovery`: `scalar_l8_train1e4` **`+0.69pp`**
- `arc_mc_seed49_discovery`: best quick `+0.00pp`, med skipped

Main pruned/negative patterns:
- `mbpp_seed46_anchor`: all candidates quick negative (`-1.39pp` to `-4.35pp`), med skipped.
- `punc_seed31_anchor`: all candidates quick non-positive (`-0.46pp` to `-1.53pp`), med skipped.
- `arc_mc_seed49_discovery`: no candidate passed promote gate (`+0.80pp`), med skipped.

Current status:
- active queue is in `round230` (`protein_ss_seed61_discovery` path running).
- chained queues remain active through `round248`.

### Chain Tail Extension

- Added queued fastloop blocks through `round400` (`369-376`, `377-384`, `385-392`, `393-400`).
- Remote auto-chainer scripts for these blocks are started and waiting on predecessor summaries.

### Queue Retrofit (2026-02-27, Round233+)

- Reworked dedup in `scripts/run_round77_82_fastdiscover.py` to use canonical signatures (ignore task alias) and include `trainer` + `base_args_fp`, reducing repeated reruns of equivalent configs.
- Rebuilt fastloop queues from `round233` to `round400` with a broader 75/25 mix (`new`/`anchor`) instead of repeated anchor-heavy loops.
- Expanded active task mix to include `squad` and `wikitext` discovery lines alongside `arc_mc`, `protein_ss`, `hotpot`, `mbpp`, and `mbpp_longctx`.
- Hot-swapped queue files in-place (`results/_search_queue_round233_240_fastloop.json` ... `_round393_400_fastloop.json`) so existing remote autochain scripts continue without restart.

### Queue Rebalance (2026-02-27, Round241+)

- Applied a second-pass allocation rebalance for `round241-400`: higher share on `protein_ss` + `arc_mc`, reduced low-yield loops, and retained periodic `squad/wikitext` calibration.
- New queue profile tag: `75_new_25_anchor_rebalance_v2`.
- Seed continuity was explicitly advanced (`arc:54`, `protein_ss:66`, `mbpp:50`, `hotpot:47`, `mbpp_longctx:17`, `squad:8`, `wiki:4`) to avoid overlap with active `round233-240` block.

### Round231-233 Status (2026-02-27)

- `round231` completed: both tasks (`mbpp_seed47_headprobe`, `punc_seed32_anchor`) failed promote gate; no med stage.
- `round232` completed: `arc_mc_seed50_discovery` flat (`+0.00pp` quick best), `protein_ss_seed62_discovery` near-gate (`+0.65pp`) but still med-skipped.
- `round233` running under new queue: `arc_mc_seed51_discovery` already reached **`+4.17pp`** med vs baseline (provisional; round not finished).

### Round401-403 Status (latest)

- `round401` completed with med-positive gains:
  - `arc_mc_seed114_discovery`: `scalar_l8_train1e4` **`+4.17pp`**
  - `protein_ss_seed166_discovery`: `scalar_l8_sched_cos` **`+0.16pp`**
- `round402` completed with mixed outcomes:
  - `mbpp_seed90_headprobe`: quick best `+2.67pp` but med **`-1.85pp`** (flagged strong negative)
  - `protein_ss_seed167_discovery`: best quick `+0.38pp`, below promote gate; med skipped
- `round403` completed:
  - `squad_seed48_discovery`: `scalar_l8_sched_cos` med **`+0.73pp`**
  - `wiki_seed24_discovery`: quick baseline failed (`rc_nonzero`), task skipped

Current execution continues in `round401-408` queue and will auto-chain into `round409-416`.

### Kernel-Value Priority Update (2026-03-01, Round417+)

- Prioritized broader kernel-value exploration by introducing `protein_contact` into the active search mix.
- New queue profile: `85_new_15_anchor_kernel_v3` with heavier focus on `protein_ss / protein_contact / arc_mc` and reduced dependency on unstable `wiki` runs.
- Generated and deployed:
  - `results/_search_queue_round417_424_fastloop.json`
  - `results/_search_queue_round425_432_fastloop.json`
- Remote autochain scripts are armed:
  - `runs/autochain_round417_424.sh` (trigger: `_summary_round416_fastdiscover.txt`)
  - `runs/autochain_round425_432.sh` (trigger: `_summary_round424_fastdiscover.txt`)

Recent round signal snapshot:
- Positive med wins: `protein_ss_seed174 +4.58pp`, `protein_ss_seed173 +2.98pp`, `protein_ss_seed171 +2.49pp`, `mbpp_seed92 +1.89pp`.
- Negative/unstable lanes: multiple `arc_mc` hard drops (quick `-4pp` to `-12pp`), and repeated `wiki` baseline failures.

### Nonstop Extension (2026-03-02, Round433+)

- Continued nonstop execution after `round432` completion by launching:
  - `results/_search_queue_round433_440_fastloop.json`
  - `results/_search_queue_round441_448_fastloop.json`
- Profile kept as `85_new_15_anchor_kernel_v3` to prioritize kernel-value exploration (`protein_ss`, `protein_contact`, `arc_mc`).
- Remote status:
  - active: `run_round77_82_fastdiscover.py --queue results/_search_queue_round433_440_fastloop.json`
  - waiting chainer: `runs/autochain_round441_448.sh` (trigger on `_summary_round440_fastdiscover.txt`)

### ROI-Explore v5 Extension (2026-03-02, Round473+)

- Based on rounds `409-433` med/quick conversion, `protein_contact` was de-prioritized (quick promote rate ~`0%`, mostly flat `+0.00pp`).
- Added profile `roi_explore_v5` in `scripts/rebuild_fastloop_queues_broad.py`:
  - focus: `protein_ss`, `arc_mc`, `mbpp`, `mbpp_longctx`
  - exploration slots: `squad`, `hotpot`, `wiki`
  - search mix: `80_new_20_anchor_roi_explore_v5`
- Generated and deployed queues:
  - `results/_search_queue_round473_480_fastloop.json`
  - `results/_search_queue_round481_488_fastloop.json`
  - `results/_search_queue_round489_496_fastloop.json`
  - `results/_search_queue_round497_504_fastloop.json`
  - `results/_search_queue_round505_512_fastloop.json`
  - `results/_search_queue_round513_520_fastloop.json`
- Deployed and started remote autochains:
  - `runs/autochain_round473_480.sh`
  - `runs/autochain_round481_488.sh`
  - `runs/autochain_round489_496.sh`
  - `runs/autochain_round497_504.sh`
  - `runs/autochain_round505_512.sh`
  - `runs/autochain_round513_520.sh`
- Current live run remains `round433-440` (single-GPU serial), with seamless handoff now armed through `round520`.

### Ruthless Prune + Immediate Restart (2026-03-02, Round449+)

- Aggressive kill of low-value lanes (based on `409-445` conversion):
  - removed from main search cycle: `protein_contact` (quick promote `0%`, mostly `+0.00pp`)
  - removed from main search cycle: `hotpot` (med transfer unstable/negative in recent window)
- Added profile `ruthless_v6` in `scripts/rebuild_fastloop_queues_broad.py`:
  - core: `protein_ss`, `arc_mc`, `mbpp`, `mbpp_longctx`
  - exploration: `squad`, `wiki`
  - mix: `85_new_15_anchor_ruthless_v6`
- Rebuilt and deployed queues `round441-560` with new seeds and no `protein_contact/hotpot` in cycle.
- Restarted active execution immediately on new queue:
  - active runner: `--queue results/_search_queue_round449_456_fastloop.json --round_from 449 --round_to 456`
  - round449 now running on `protein_ss_seed232_discovery` / `arc_mc_seed164_discovery`.

### GPT-5.3 Codex Policy + Best-of-N Replan (2026-03-02, Round561+)

- Added strategy policy file (Spark disabled):
  - `results/_codex53_team_policy.json`
  - `model: gpt-5.3-codex` with fixed gates:
    - quick promote `+0.80pp`
    - quick prune `< -0.50pp`
    - strong negative cooldown: quick `<= -1.00pp`, med `<= -0.50pp`
- Updated orchestrator:
  - `scripts/run_round77_82_fastdiscover.py` now supports `--policy` and writes active policy in round summary header.
- Added Best-of-N planner:
  - `scripts/plan_bestofn_fastloop.py`
  - scores candidate profiles from recent history and applies hard suppression on killed tasks.
- Replanned and generated queues `round561-640` using Best-of-N:
  - plan artifact: `results/_bestofn_plan_round561_640.json`
  - selected profile: `ruthless_v6`
  - generated: `_search_queue_round561_568_fastloop.json` ... `_search_queue_round633_640_fastloop.json`
- Remote chain extension deployed:
  - `runs/autochain_round561_568.sh` ... `runs/autochain_round633_640.sh`
  - each launcher uses `--policy results/_codex53_team_policy.json`.

### Latest Progress Snapshot (2026-03-02, Round455-457)

- `round455` produced a small positive:
  - `squad_seed81_anchor` / `scalar_l8_train1e4` med **`+0.69pp`**
- `round456` produced a positive on protein and a negative on long-context MBPP:
  - `protein_ss_seed235_discovery` / `scalar_l8_train1e4` med **`+1.33pp`**
  - `mbpp_longctx_seed86_repair` / `scalar_l8_sched_cos` med **`-1.46pp`** (flagged)
- `round457` exposed high-variance ARC behavior:
  - `arc_mc_seed167_discovery` quick all **`+4.17pp`** but med **`-12.50pp`**
  - `protein_ss_seed236_discovery` best quick **`+0.00pp`**, med skipped
- Current live run has moved to `round458` (`mbpp_seed129_headprobe`, `wiki_seed36_discovery`) under `ruthless_v6`.

### BFS-First Switch (2026-03-02, Round459+)

- Per request, switched from repeated high-ROI lanes to breadth-first exploration first.
- Added queue profile `bfs_v7` in `scripts/rebuild_fastloop_queues_broad.py`:
  - search mix: `70_new_30_anchor_bfs_v7`
  - cycle covers all active tasks in rotation:
    - `arc_mc`, `mbpp`, `protein_ss`, `squad`, `wiki`, `hotpot`, `mbpp_longctx`, `protein_contact`
- Rebuilt and deployed BFS queues for `round457-640`.
- Hard cutover executed:
  - stopped old `457-464` runner
  - relaunched from `round459` with policy:
    - `--queue results/_search_queue_round457_464_fastloop.json`
    - `--round_from 459 --round_to 464`
    - `--policy results/_codex53_team_policy.json`
- Current BFS round sequence (`459-464`) is:
  - `459: wiki + hotpot`
  - `460: mbpp_longctx + protein_contact`
  - `461: arc_mc + squad`
  - `462: protein_ss + mbpp`
  - `463: wiki + mbpp_longctx`
  - `464: hotpot + protein_contact`

### NT + SAT/TSP Exploratory Update (2026-03-03, Round701-708)

- Added new task adapters:
  - `scripts/train_nt_seqcls_sft.py` for Nucleotide Transformer downstream tasks (`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`)
  - `scripts/train_np_sat_tsp_sft.py` for SAT/TSP probes (`sat3`, `tsp_mask`)
- Added queues:
  - `results/_search_queue_round701_704_nt_bfs.json`
  - `results/_search_queue_round705_708_sat_tsp.json`

Round outcomes:
- `round701` (NT seed0 start):
  - `nt_splice_sites_all_seed0`: quick best **`+0.00pp`**, med skipped
  - `nt_enhancers_types_seed0`: baseline failed (initial prompt-length gate too strict), fixed in later queue patch
- `round702` (NT seed0 continue):
  - `nt_h3k4me3_seed0`: quick baseline hit **`100.00%`**, all FS variants `<= +0.00pp`, med skipped
  - `nt_promoter_all_seed0`: quick best **`+5.56pp`**, but med collapsed to **`+0.00pp`**
- `round705` (SAT/TSP seed0):
  - `sat3_seed0_discovery`: quick best **`+0.00pp`**, med skipped
  - `tsp_mask_seed0_discovery`: med **`+4.17pp`** (FS `25.00%` vs baseline `20.83%`)
- `round706` (SAT/TSP seed1):
  - `sat3_seed1_discovery`: quick `+12.50pp`, med **`+0.00pp`**
  - `tsp_mask_seed1_discovery`: quick best **`+0.00pp`**, med skipped
- `round707` (anchor seed2):
  - `sat3_seed2_anchor`: quick best **`+0.00pp`**, med skipped
  - `tsp_mask_seed2_anchor`: quick best **`+0.00pp`**, med skipped
- `round708` (`q_first` controls):
  - `sat3_seed0_qfirst_control`: quick `+18.75pp`, med **`+0.00pp`**
  - `tsp_mask_seed0_qfirst_control`: quick best **`+0.00pp`**, med skipped

Current takeaways:
- NT line currently shows strong ceiling/instability effects (not a stable positive branch).
- SAT line shows quick gains but does not transfer to med under current recipe.
- TSP line produced one meaningful med gain (`+4.17pp`, seed0) but has not yet shown cross-seed stability.
- Next high-ROI follow-up is TSP-focused stabilization (same task, tighter recipe sweep) rather than broad SAT expansion.

### TSP Stabilization Sweep (2026-03-03, Round709-710)

- Added queue:
  - `results/_search_queue_round709_710_tsp_targeted.json`
- Purpose:
  - verify whether the earlier `tsp_mask` positive (`+4.17pp`) can be stabilized across seeds,
  - test `head_l8/head_l10` against scalar baseline under larger train/val budget.

Results:
- `round709`:
  - `tsp_mask_seed0_targeted`: quick all candidates **`-6.25pp`** (all pruned)
  - `tsp_mask_seed1_targeted`: quick best `+6.25pp`, but med **`+0.00pp`**
- `round710`:
  - `tsp_mask_seed2_targeted_anchor`: quick best **`+0.00pp`**, med skipped
  - `tsp_mask_seed0_qfirst_targeted`: quick all candidates **`-6.25pp`** (all pruned)

Conclusion:
- Current TSP gain is not yet stable under seed/control sweeps.
- `head` variants did not rescue stability in this pass.
- Keep `tsp_mask` as a tentative candidate task, but do not claim robust FS gain yet.

### Fail-Backup Cut (2026-03-04)

- Per stability policy, unstable experiment families were removed from active search paths and archived:
  - unstable: `nt / sat / tsp`
  - stable whitelist kept active: `protein_ss / mbpp / squad / punc`
- Archived location:
  - `fail-backup/scripts/`
  - `fail-backup/results/`
  - `fail-backup/runs/`
  - manifest: `fail-backup/manifests/ARCHIVE_NOTES.md`
- This cut includes:
  - scripts: `train_nt_seqcls_sft.py`, `train_np_sat_tsp_sft.py`
  - queues: `_search_queue_round701_704_nt_bfs.json`, `_search_queue_round705_708_sat_tsp.json`, `_search_queue_round709_710_tsp_targeted.json`
  - run artifacts: `round701/702/705/706/707/708/709/710` summaries + records

### Bi-Encoder Fastdiscover Burst (2026-03-04, Round711-714)

Scope (new tasks only):
- `squad` (extractive QA)
- `glue/mrpc` (sentence-pair classification)
- `glue/stsb` (semantic similarity binning)
- `google/code_x_glue_cc_defect_detection` (code understanding)

Execution policy:
- quick promote: `+0.80pp`
- quick prune: `< -0.50pp`
- useful task rule: `med > baseline` (pp)

Engineering updates for this burst:
- Added `scripts/train_hf_pair_cls_sft.py` (generic pair/text classification probe).
- Updated `scripts/train_hotpot_longctx_sft.py` to support SQuAD (`answers.text`) + empty `ds_cfg`.
- Added queue/policy:
  - `results/_search_queue_round711_714_bi_encoder.json`
  - `results/_policy_round711_bi.json`

Round highlights:
- `round711`
  - `mrpc_seed0_bicls`: quick best `+0.00pp`, med skipped.
  - `squad_seed0_biqa`: first pass failed at sample build gate (`min_prompt_tokens` too strict), then fixed in later rounds.
- `round712`
  - `stsb_seed0_bisim`: quick best `+4.17pp`, but med `+0.00pp`.
  - `code_defect_seed0_bicode`: quick best `+0.00pp`, med skipped.
- `round713`
  - `squad_seed1_biqa`:
    - quick best `+10.32pp` (`scalar_l8_train8e5`)
    - med **`+0.73pp`** (`10.86%` vs baseline `10.13%`)
  - `mrpc_seed1_bicls`: quick best `+0.00pp`; strong negatives pruned (`-4.17pp`, `-12.50pp`); med skipped.
- `round714`
  - `stsb_seed1_bisim`: quick best `+8.33pp`, but med `+0.00pp`.
  - `code_defect_seed1_bicode`: quick best `+0.00pp`, med skipped.

Current takeaway:
- In this bi-encoder burst, only `squad_seed1_biqa` converted to a positive med gain (**`+0.73pp`**).
- `stsb` shows consistent quick gains but fails to hold advantage at med.
- `mrpc` and `code_defect` remain flat or negative under current FS settings.

### Constraint BFS Quick-Attack (2026-03-05, Round735-742, running)

Goal:
- switch to breadth-first constraint tasks quickly, keep strict quick->med gates, and avoid repeated low-yield confirmations.

Code/infra updates applied:
- `scripts/train_np_sat_tsp_probe_sft.py`
  - added new tasks: `graph_color` (k-colorability Y/N), `nqueens` (partial completion Y/N)
  - extended parser/cache/build dispatch for new task args and metadata
  - hotfix: N-Queens prompt auto-recap when `min_prompt_tokens` is high (avoid sample-build starvation)
- `scripts/train_sft.py`
  - added orchestrator-compatible args: `--train_data_seed`, `--val_data_seed`
  - keeps backward behavior while enabling seed-controlled data generation from fastdiscover
- new queue: `results/_search_queue_round735_742_constraint_bfs.json`
- strict policy file: `results/_codex53_team_policy_strict_08_05.json`
  - quick promote `+0.80pp`
  - quick prune `< -0.50pp`
  - cooldown `8` rounds

Live run:
- launcher: `runs/_launcher_round735_742_constraint_bfs.log`
- records:
  - `runs/_round735_fastdiscover_records.jsonl` (completed)
  - `runs/_round736_fastdiscover_records.jsonl` (in progress)

Round735 outcomes:
- `graph_color_seed0_bfs`
  - quick baseline: `100.00%`
  - FS quick candidates: `81.25%`, `75.00%`, `81.25%` (all heavily negative, pruned)
  - med skipped by gate
- `nqueens_seed0_bfs`
  - baseline failed in first pass due prompt-length gate starvation (`Only built 0 NQueens examples`)
  - fix applied immediately in trainer (prompt recap expansion) for subsequent N-Queens rounds

Round736 (partial while running):
- `sat3_seed1_balanced`
  - quick best: `+25.00pp` (`87.50%` vs baseline `62.50%`)
  - med: `+0.00pp` (no transfer)
- `tsp_mask_seed1_balanced`
  - quick currently flat (`+0.00pp` best so far)

Current interpretation:
- SAT keeps showing a recurring pattern: strong quick uplift, weak med transfer.
- Graph-color baseline can saturate under current setting (needs harder instance regime in next queue revision).
- N-Queens pipeline is now unblocked after prompt-length hotfix; next evidence comes from upcoming rounds (`738/741`).

### Constraint BFS v2 Continuation (2026-03-05, Round743-750, running)

Completed outcomes from `round736-742`:
- Positive med gains:
  - `graph_color_seed3_qfirst`: **`+4.17pp`**
  - `graph_color_seed5_dense`: **`+4.17pp`**
  - `mbpp_longctx_seed20_qfirst_anchor`: **`+1.84pp`**
  - `punc_seed38_anchor`: **`+0.41pp`**
- Negative/unstable:
  - `nqueens_seed3_qfirst`: `-16.67pp` med
  - `nqueens_seed5_hard`: `-4.17pp` med
  - `mbpp_longctx_seed19_anchor`: `-3.22pp` med
- Flat lanes:
  - `sat3_seed1_balanced`: quick `+25.00pp` but med `+0.00pp`
  - `tsp_mask_seed1_balanced`, `zebra_seed2_bfs`, `arc_mc_seed238_qfirst_anchor`: no med conversion
- Engineering fix:
  - `countdown` build starvation fixed by prompt-recap expansion in `train_np_sat_tsp_probe_sft.py` (same class fix as N-Queens).

New queue launched:
- `results/_search_queue_round743_750_constraint_bfs_v2.json`
- launcher: `runs/_launcher_round743_750_constraint_bfs_v2.log`
- policy unchanged: quick promote `+0.80pp`, quick prune `< -0.50pp`, cooldown `8`.

Live signal (`round743` in progress):
- `graph_color_seed6_phase` quick baseline `87.50%`
- FS quick candidates currently flat (`+0.00pp` best so far)
- next task in round: `countdown_seed3_retry` (post-fix retry)

### GitHub Full Sync (2026-03-05, Round736-744)

This section supersedes earlier partial notes and lists all newly synced artifacts in this push.

Synced artifacts:
- summaries: `results/_summary_round736_fastdiscover.txt` ... `results/_summary_round743_fastdiscover.txt`
- records: `results/_round736_fastdiscover_records.jsonl` ... `results/_round744_fastdiscover_records.jsonl`
- queues/policy:
  - `results/_search_queue_round735_742_constraint_bfs.json`
  - `results/_search_queue_round743_750_constraint_bfs_v2.json`
  - `results/_codex53_team_policy_strict_08_05.json`
- launcher log:
  - `results/_launcher_round743_750_constraint_bfs_v2.log`

Completed round outcomes:
- round736:
  - `sat3_seed1_balanced`: quick `+25.00pp`, med `+0.00pp`
  - `tsp_mask_seed1_balanced`: quick `+0.00pp` best, med skipped
- round737:
  - `zebra_seed2_bfs`: quick ties baseline, med skipped
  - `countdown_seed2_bfs`: baseline failed (build starvation)
- round738:
  - `graph_color_seed3_qfirst`: med **`+4.17pp`**
  - `nqueens_seed3_qfirst`: med **`-16.67pp`**
- round739:
  - `arc_mc_seed237_anchor`: no promote
  - `mbpp_longctx_seed19_anchor`: med **`-3.22pp`**
- round740:
  - `punc_seed38_anchor`: med **`+0.41pp`**
  - `squad_seed11_anchor`: no promote
- round741:
  - `graph_color_seed5_dense`: med **`+4.17pp`**
  - `nqueens_seed5_hard`: med **`-4.17pp`**
- round742:
  - `arc_mc_seed238_qfirst_anchor`: no promote
  - `mbpp_longctx_seed20_qfirst_anchor`: med **`+1.84pp`**
- round743:
  - `graph_color_seed6_phase`: quick all `+0.00pp`, med skipped
  - `countdown_seed3_retry`: quick `+18.75pp` best, med **`-4.17pp`**

In-progress (round744, partial from records):
- `graph_color_seed7_qfirst_phase` already med-positive:
  - baseline med `91.67%`
  - FS med (`scalar_l8_train8e5`) `95.83%` (**`+4.17pp`**)
- `tsp_mask_seed2_qfirst_retry` is still running at sync time.

Current practical takeaway:
- `graph_color` has become the strongest new constraint task in this window (multiple med-positive confirmations at `+4.17pp`).
- `nqueens` and `countdown` currently show quick-stage uplift but unstable or negative med transfer.

### Incremental Sync (2026-03-05, Round744-746)

Newly synced artifacts:
- `results/_summary_round744_fastdiscover.txt`
- `results/_summary_round745_fastdiscover.txt`
- `results/_round745_fastdiscover_records.jsonl`
- `results/_round746_fastdiscover_records.jsonl` (running snapshot)

Round744 (completed):
- `graph_color_seed7_qfirst_phase`:
  - quick best `+6.25pp`
  - med **`+4.17pp`** (`95.83%` vs baseline `91.67%`)
- `tsp_mask_seed2_qfirst_retry`:
  - quick best **`-6.25pp`** (all variants negative)
  - med skipped; all FS variants pruned

Round745 (completed):
- `graph_color_seed8_k4_dense`:
  - quick best `+6.25pp`
  - med **`-4.17pp`** (transfer failed)
- `sat3_seed3_harder`:
  - quick all `+0.00pp`
  - med skipped

Round746 (partial while running):
- `zebra_seed4_qfirst_retry`:
  - baseline quick `50.00%`
  - FS quick best currently `+0.00pp` (one variant `-18.75pp` pruned)
  - med currently skipped by gate
- next subtask: `countdown_seed4_retry` in same round.

### Final Closure Update (2026-03-06, round783-802 completed)

Artifacts:
- `results/_summary_round783_fastdiscover.txt` ... `results/_summary_round790_fastdiscover.txt`
- `results/_summary_round799_fastdiscover.txt` ... `results/_summary_round802_fastdiscover.txt`
- `results/_round783_fastdiscover_records.jsonl` ... `results/_round790_fastdiscover_records.jsonl`
- `results/_round799_fastdiscover_records.jsonl` ... `results/_round802_fastdiscover_records.jsonl`

Final closure outcomes:
- `round783`
  - `arc_mc_seed256_depthmix_qfirst`: quick tie (`+0.00pp`), med skipped
  - `mbpp_longctx_seed36_depthmix_anchor`: all FS variants quick-negative, hard-pruned
- `round784`
  - `arc_mc_seed257_depthmix_qfirst`: all quick-negative, hard-pruned
  - `mbpp_longctx_seed37_depthmix_anchor`: med **`-1.69pp`**
- `round785`
  - `arc_mc_seed258_depthmix_qfirst` / `scalar_l10_train1e4`: med **`+3.12pp`**
  - `mbpp_longctx_seed38_depthmix_anchor` / `scalar_l6_train1e4`: med **`+5.01pp`**
- `round786`
  - `arc_mc_seed259_depthmix_qfirst` / `scalar_l8_train1e4`: med **`-1.56pp`**
  - `mbpp_longctx_seed39_depthmix_anchor` / `scalar_l10_train1e4`: med **`+0.54pp`**
- `round787`
  - `arc_mc_seed260_depthmix_qfirst` / `scalar_l6_train1e4`: med **`+9.38pp`**
  - `mbpp_longctx_seed40_depthmix_anchor` / `scalar_l6_train1e4`: med **`-0.63pp`**
- `round788`
  - `arc_mc_seed261_depthmix_qfirst` / `scalar_l6_train1e4`: med **`+3.12pp`**
  - `mbpp_longctx_seed41_depthmix_anchor` / `scalar_l6_train1e4`: med **`+0.97pp`**
- `round789`
  - `arc_mc_seed262_depthmix_qfirst`: quick-negative, med skipped
  - `mbpp_longctx_seed42_depthmix_anchor` / `scalar_l10_train1e4`: med **`+1.87pp`**
- `round790`
  - `arc_mc_seed263_depthmix_qfirst` / `scalar_l10_train1e4`: med **`+6.25pp`**
  - `mbpp_longctx_seed43_depthmix_anchor`: best quick `-0.47pp`, med skipped
- `round799`
  - `mbpp_longctx_seed51_finalconfirm` / `scalar_l6_train1e4`: quick **`+0.38pp`**, below promote gate, med skipped
- `round800`
  - `mbpp_longctx_seed52_finalconfirm` / `scalar_l6_train1e4`: quick **`+0.60pp`**, below promote gate, med skipped
- `round801`
  - `arc_mc_seed271_finalconfirm` / `scalar_l6_train1e4`: med **`-1.56pp`**
- `round802`
  - `tsp_mask_seed13_finalconfirm` / `scalar_l10_train1e4`: quick **`+0.00pp`**, med skipped

Final readout:
- `mbpp_longctx` is still the best real-task lead, but the correct wording is "promising and repeatedly positive in exploitation," not "strictly confirmed stable gain."
- `arc_mc` produced several strong positives in the exploit block, but the held-out confirm seed failed, so it remains mixed evidence.
- `tsp_mask` should stay in appendix only.
- broad expansion is stopped at `round802`; remaining work is documentation, paper tables, and repo cleanup instead of more search.
