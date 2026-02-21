# Future-Seed for RWKV Post-Training: Where It Works and Where It Fails

Status: draft for repo tracking (updated 2026-02-21).

## Abstract

We study whether Future-Seed (FS), a cross-layer terminal-state seeding mechanism for RWKV, helps post-training on real tasks. We run controlled no-FS vs FS probes under fixed budgets, then add a serial immediate-prune search protocol to reduce compute waste and surface task-specific gains quickly. Across ARC/Hotpot/MBPP/Sudoku/protein tasks, FS is not a universal gain switch. However, under the same single-GPU budget, we find a strong positive regime on protein secondary-structure spot labeling, with up to +8.02 percentage points over no-FS. In contrast, Hotpot remains non-improving in the tested protocol, and several earlier null results are partly explained by sample-construction bottlenecks. Our conclusion is that FS should be treated as a targeted mechanism whose utility depends on task structure and training regime.

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

### 3.3 Round21 targeted follow-up (ongoing)

Partial artifacts:

- `results/_round21_targeted_search_records.partial.jsonl`
- `results/_log_round21_targeted_search_s0.20260221_193225.log.partial`

Current partial signal:

- MBPP after data-build fixes now runs; `scalar_l8_trainable` passes quick stage (`+1.00pp`) and is promoted.
- Protein contact after data-build fixes now runs; screening is ongoing.

These partial logs are tracked as in-progress evidence and are not final claims.

## 4. Failure Analysis

We repeatedly observe four failure modes:

1. **Regime dependence**: batch/steps/eval cadence can flip sign.
2. **Task dependence**: gains do not transfer uniformly across tasks.
3. **No-op collapse**: some deep-start settings produce near-zero deltas.
4. **Data-build bottlenecks**: null conclusions can be caused by insufficient constructed examples.

The Round20/21 protocol directly addresses (4) by making build failures explicit and then retesting.

## 5. Main Takeaways

1. FS is **not** a universal post-training improvement.
2. FS can be **strongly useful** in specific real-task regimes (currently strongest on protein SS spot).
3. A serial immediate-prune workflow is effective for quickly finding viable FS regimes on one 4090.

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
  - `results/_round21_targeted_search_records.partial.jsonl`

## 7. Limitations and Next Steps

Limitations:

- Some conclusions are still single-seed (search phase).
- Not all tasks have completion-level metrics (e.g., executable MBPP pass rate in this round).

Next:

1. Finalize Round21 and export completed summary.
2. Re-run top protein SS settings with small multi-seed confirmation.
3. Add one completion-level metric for MBPP in the same serial prune framework.

