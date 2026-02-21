# Future-Seed for RWKV Post-Training: Where It Works and Where It Fails

Status: draft for repo tracking (updated 2026-02-21).

## Abstract

We study whether Future-Seed (FS), a cross-layer terminal-state seeding mechanism for RWKV, helps post-training on real tasks. We run controlled no-FS vs FS probes under fixed budgets, then apply a serial immediate-prune search protocol to reduce compute waste and surface working regimes quickly. Across ARC/Hotpot/MBPP/Sudoku/protein and real-text restoration probes, FS is not a universal gain switch. However, under the same single-GPU budget, we repeatedly find strong positive regimes on protein secondary-structure spot labeling (up to +13.31pp in targeted runs; +4.14pp in later stability runs) and a memory-safe punctuation/case restoration probe (+3.45pp). In contrast, Hotpot remains non-improving in the tested protocols, and MBPP gains are recipe-sensitive (positive in one targeted setup, negative in several high-throughput sweeps). Our conclusion is that FS should be treated as a targeted mechanism whose utility depends on task structure and training regime.

## 1. Introduction

RWKV-style recurrent models are efficient for long context, but causal state updates can struggle when useful constraints are distributed far across the prompt. Future-Seed attempts to improve this by feeding the previous layer's terminal state into deeper layers at prompt time, while keeping the model scan itself left-to-right.

The core question in this work is practical: does FS help post-training on realistic tasks? We focus on small-budget experiments on a single RTX 4090 and emphasize fast iteration with explicit pruning of non-working variants.

## 2. Method

### 2.1 Base setting

- Base checkpoint: `rwkv7-g1d-0.1b-20260129-ctx8192.pth`
- Comparison: `mode=no_fs` vs `mode=prompt_fs`
- FS variants explored:
  - scalar gate
  - head-wise gate
  - scheduled alpha (cos/linear)
  - trainable alpha
  - with/without norm/detach

### 2.2 Search protocol

We use a serial immediate-prune policy:

1. Probe max feasible batch size per task.
2. Run quick baseline.
3. Run quick FS variant; prune immediately if `d_acc` below threshold.
4. Only medium-confirm survivors.

This keeps all runs single-GPU serial and shifts budget from non-working settings to exploration.

## 3. Experimental Summary

## 3.1 Prior multi-seed rounds (R1-R12)

From the earlier logged summaries:

- ARC options-first had positive regimes in some protocols, but sign flipped under higher-throughput R12.
- Hotpot long-context remained unstable with mixed signs and frequent near-no-op behavior.
- MBPP and protein contact mostly showed no reliable improvements in the early settings.

See:

- `paper/FS_POSTTRAIN_PROGRESS_2026-02-19.md`
- `paper/DETAILED_EXPERIMENT_LOG.md`
- `results/_aggregate_results.md`

### 3.2 Round20: serial immediate-prune (single seed)

Main artifact:

- `results/_summary_round20_serial_earlystop_s0.txt`

Findings:

- **Hotpot**:
  - Baseline quick accuracy: `6.17%`
  - All tested FS variants pruned (no positive quick delta).
- **MBPP (old setting)**:
  - Baseline failed due sample-construction constraints.
- **Protein contact (old setting)**:
  - Baseline failed due sample-construction constraints.
- **Protein secondary-structure spot labeling**:
  - Baseline quick: `24.66%`
  - Best medium-confirmed FS:
    - `scalar_l10_norm_node`: `32.69%` (**+8.02pp**)
    - `scalar_l10_trainable`: `32.38%` (**+7.72pp**)
    - `scalar_l10_sched_cos`: `32.23%` (**+7.57pp**)
    - `head_l10`: `31.97%` (**+7.31pp**)

This is the strongest real-task signal in the current repository.

### 3.3 Round21 targeted follow-up (completed)

Artifacts:

- `results/_summary_round21_targeted_search_s0.txt`
- `results/_round21_targeted_search_records.jsonl`

Results:

- `mbpp_fix`:
  - quick baseline: `10.46%`
  - medium `scalar_l8_trainable`: `24.07%` (**+13.61pp**)
- `protein_contact_fix`:
  - quick baseline: `98.83%`
  - all FS variants pruned (`+0.00pp`)
- `protein_ss_refine`:
  - quick baseline: `21.14%`
  - medium best:
    - `scalar_l10_sched_cos`: `34.45%` (**+13.31pp**)
    - `scalar_l10_trainable`: `33.95%` (**+12.82pp**)
    - `head_l10`: `33.48%` (**+12.35pp**)

### 3.4 Round22 adaptive search (completed)

Artifacts:

- `results/_summary_round22_adaptive_search_s0.txt`
- `results/_round22_adaptive_search_records.jsonl`

Results:

- `mbpp_focus`: all FS quick variants regressed (`-0.92pp` to `-4.83pp`).
- `protein_ss_expand`: small mixed quick gains (best `+1.50pp`).
- `sudoku4_refine`: strong positive med gains (best `+33.29pp`).
- `sudoku9_probe`: modest positive med gains (best `+1.51pp`).

### 3.5 Round23 real-task sweep (partial)

Artifacts:

- `results/_round23_real_task_sweep_records.jsonl`
- `results/_launcher_round23.log`

Completed results:

- `mbpp_rt`: quick FS variants all negative.
- `hotpot_rt`: FS quick variants exact-tie baseline (`+0.00pp`).
- punctuation run aborted due HF connectivity (infra issue).

### 3.6 Round24/25 continuation

Artifacts:

- `results/_summary_round24_punc_protein_s0.txt`
- `results/_summary_round25_punc_salvage_s0.txt`

Results:

- `protein_ss_rt` (Round24):
  - baseline quick: `30.17%`
  - med `scalar_l10_sched_cos`: `34.31%` (**+4.14pp**)
- `punc_restore_rt`:
  - Round24 baseline failed by OOM at `bsz=10` (configuration failure).
  - Round25 memory-safe rerun (`bsz=2`) recovered:
    - baseline quick: `9.18%`
    - med `head_l8`: `12.64%` (**+3.45pp**)
    - med `scalar_l8_sched_cos`: `11.90%` (**+2.71pp**)

### 3.7 Round26 low-throughput check (completed)

Artifacts:

- `results/_summary_round26_mbpp_hotpot_lowthroughput_s0.txt`
- `results/_round26_mbpp_hotpot_lowthroughput_records.jsonl`

Results:

- `mbpp_low`:
  - quick baseline: `10.46%`
  - quick `scalar_l8_trainable`: `+1.00pp`
  - medium `scalar_l8_trainable`: `29.64%` (**+19.17pp**)
- `hotpot_low`:
  - quick baseline: `14.34%`
  - `scalar_l10_trainable`: `+0.00pp`
  - `scalar_l10_sched_cos`: `+0.00pp`
  - `head_l10`: `-1.84pp`

### 3.8 Round27 seedcheck for positive regimes (completed)

Artifacts:

- `results/_summary_round27_seedcheck_positive_s012.txt`
- `results/_round27_seedcheck_positive_s012_records.jsonl`

Results:

- `mbpp_low + scalar_l8_trainable` (quick):
  - seed0 `+1.00pp`, seed1 `+0.32pp`, seed2 `-0.82pp`
  - mean `+0.17pp`, positive seeds `2/3`
- `punc_restore + head_l8` (quick):
  - seed0 `+0.80pp`, seed1 `+0.58pp`, seed2 `+2.20pp`
  - mean `+1.19pp`, positive seeds `3/3`

### 3.9 Round28 MBPP throughput sweep (completed)

Artifacts:

- `results/_summary_round28_mbpp_bsz_sweep_s0.txt`
- `results/_round28_mbpp_bsz_sweep_s0_records.jsonl`

Results:

- `bsz=2`:
  - baseline quick `10.46%`
  - FS quick `+1.00pp`
  - FS med `25.39%` (**+14.92pp**)
- `bsz=4`:
  - baseline quick `11.71%`
  - FS quick `-2.05pp`
- `bsz=6`:
  - baseline quick `14.50%`
  - FS quick `-2.08pp`
- `bsz=8`:
  - baseline failed (OOM)

## 4. Failure Analysis

We repeatedly observe four failure modes:

1. **Regime dependence**: batch/steps/eval cadence can flip sign.
2. **Task dependence**: gains do not transfer uniformly across tasks.
3. **No-op collapse**: some deep-start settings produce near-zero deltas.
4. **Data-build bottlenecks**: null conclusions can be caused by insufficient constructed examples.
5. **Throughput sensitivity**: MBPP can flip sign across throughput recipes.
6. **Seed instability**: MBPP positive regime is weaker than punc on seed robustness.
7. **Resource cliff**: MBPP crosses an OOM cliff at higher batch sizes in current setup.

The Round20/21 protocol directly addresses (4) by making build failures explicit and then retesting.

## 5. Main Takeaways

1. FS is **not** a universal post-training improvement.
2. FS can be **strongly useful** in specific real-task regimes (protein SS strongest; punc restore positive after memory correction).
3. MBPP gains are **recipe-sensitive** and weakly seed-stable in current setup.
4. A serial immediate-prune workflow is effective for quickly finding viable FS regimes on one 4090.
5. Punctuation restoration is currently the most reproducible non-synthetic FS-positive regime.
6. MBPP currently requires low-throughput operation (`bsz=2`) to retain FS gains.

## 6. What Is Included in This Repo

- Continuous logs:
  - `paper/DETAILED_EXPERIMENT_LOG.md`
  - `paper/FS_POSTTRAIN_PROGRESS_2026-02-19.md`
- Machine-readable experiment registry:
  - `paper/exp_manifest.json`
  - `results/_aggregate_results.jsonl`
- Human-readable result table:
  - `results/_aggregate_results.md`
- Latest search records:
  - `results/_summary_round20_serial_earlystop_s0.txt`
  - `results/_round20_serial_earlystop_records.jsonl`
  - `results/_summary_round21_targeted_search_s0.txt`
  - `results/_round21_targeted_search_records.jsonl`
  - `results/_summary_round22_adaptive_search_s0.txt`
  - `results/_round22_adaptive_search_records.jsonl`
  - `results/_summary_round24_punc_protein_s0.txt`
  - `results/_summary_round25_punc_salvage_s0.txt`
  - `results/_summary_round26_mbpp_hotpot_lowthroughput_s0.txt`
  - `results/_summary_round27_seedcheck_positive_s012.txt`
  - `results/_summary_round28_mbpp_bsz_sweep_s0.txt`

## 7. Limitations and Next Steps

Limitations:

- Some conclusions are still single-seed (search phase).
- Not all tasks have completion-level metrics (e.g., executable MBPP pass rate in this round).

Next:

1. Add completion-level MBPP metric (pass@k / executable tests), not just token accuracy.
2. Shift MBPP branch to low-throughput-only until executable metrics are added.
3. Keep punc/head_l8 as a rolling stability benchmark (multi-seed every round).
4. Build harder non-saturated protein-contact formulation.
