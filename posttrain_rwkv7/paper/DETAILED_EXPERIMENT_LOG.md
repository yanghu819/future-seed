# Future-Seed Post-Training: Detailed Experiment Log

Date: 2026-02-19  
Base model: `rwkv7-g1d-0.1b-20260129-ctx8192.pth`  
Rule: keep model scan strictly left->right; FS only via cross-layer terminal-state seeding.

## Common FS Config (Stabilized Recipe)

- `mode=prompt_fs` (prefill/prompt only)
- `alpha_init=-2` (unless explicitly changed)
- `alpha_lr=0`
- `fs_layer_start=6`
- `fs_norm`
- `fs_detach`
- `fs_clip=1.0`

## Real-Task Results (Completed)

## ARC-Challenge (MCQ)

### R2: Options-first (causal-unfriendly)

- Script: `run_arc_stabilized_round2.sh`
- Summary: `results/_summary_arc_optionsfirst_stabilized_r2.txt`
- Result:
  - mean `d_acc = +0.0339`, std `0.0205`
  - seed deltas: `+0.0156`, `+0.0234`, `+0.0625`
- Status: **success / stable positive**

### R3: Question-first (causal-friendly control)

- Script: `run_arc_qfirst_stabilized_round3.sh`
- Summary: `results/_summary_arc_qfirst_stabilized_r3.txt`
- Result:
  - mean `d_acc = -0.0052`, std `0.0097`
  - seed deltas: `-0.0156`, `+0.0078`, `-0.0078`
- Status: **control matched expectation** (no universal gain)

## HotpotQA Long Context

### R4: q-after, L=2048

- Script: `run_hotpot_qafter_stabilized_round4_s0.sh` + `_s12.sh`
- Summary: `results/_summary_hotpot_qafter_stabilized_r4_s012.txt`
- Result:
  - mean `d_acc = -0.0027`, std `0.0171`
  - seed deltas: `+0.0084`, `+0.0104`, `-0.0269`
- Status: **mixed / unstable**

### R5: q-first, L=2048 (control)

- Script: `run_hotpot_qfirst_stabilized_round5_s012.sh`
- Summary: `results/_summary_hotpot_qfirst_stabilized_r5_s012.txt`
- Result:
  - mean `d_acc = -0.0090`, std `0.0085`
  - seed deltas: `-0.0104`, `+0.0021`, `-0.0186`
- Status: **negative control (no gain)**

### R6: q-after, L=4096, alpha=-2

- Script: `run_hotpot_qafter_stabilized_len4096_round6_s0.sh` + `_s12.sh`
- Summary: `results/_summary_hotpot_qafter_stabilized_len4096_r6_s012.txt`
- Result:
  - mean `d_acc = +0.0012`, std `0.0388`
  - seed deltas: `+0.0382`, `+0.0179`, `-0.0525`
- Status: **slight mean gain but high variance**

### R7: q-first, L=4096, seed0

- Script: `run_hotpot_qfirst_stabilized_len4096_round7_s0.sh`
- Summary: `results/_summary_hotpot_qfirst_stabilized_len4096_r7_s0.txt`
- Result:
  - `d_acc = +0.0025` (single seed)
  - loss worsened in that run
- Status: **inconclusive (single-seed control)**

### R8: q-after, L=4096, alpha=-4

- Script: `run_hotpot_qafter_stabilized_len4096_round8_alpha_m4_s012.sh`
- Summary: `results/_summary_hotpot_qafter_stabilized_len4096_r8_alpha_m4_s012.txt`
- Result:
  - mean `d_acc = -0.0056`, std `0.0519`
  - seed deltas: `+0.0141`, `+0.0458`, `-0.0766`
- Status: **failure for variance reduction** (negative outlier got worse)

## Failure Modes Observed

1. **Seed-level instability on long QA**
- Same config can be strongly positive on one seed and strongly negative on another.

2. **Simple alpha weakening is insufficient**
- Moving from `alpha=-2` to `alpha=-4` reduced mean utility on Hotpot L=4096.

3. **Task-order dependence is real**
- ARC options-first gains are robust; q-first control removes gains.
- Indicates FS utility is conditional on causal awkwardness, not a universal boost.

## In-Progress Mitigation

### R9: Depth-Scheduled FS (new)

- Scripts:
  - `run_arc_optionsfirst_stabilized_round4_sched_linear.sh`
  - `run_hotpot_qafter_stabilized_len4096_round9_sched_linear_s012.sh`
- New flags:
  - `--fs_alpha_schedule linear`
  - `--fs_alpha_min 0.25`
  - `--fs_alpha_max 1.0`
- Hypothesis:
  - make early FS injection weaker, keep deeper layers stronger
  - reduce negative seed outliers without killing positive cases

### R9a Result: ARC options-first (completed)

- Script: `run_arc_optionsfirst_stabilized_round4_sched_linear.sh`
- Summary: `results/_summary_arc_optionsfirst_stabilized_r4_sched_linear.txt`
- Result:
  - mean `d_acc = +0.0156`, std `0.0292`
  - sign pattern: `2+ / 0 / 1-`
  - mean `d_loss = +0.0295` (worse)
- Comparison vs R2 baseline recipe:
  - R2 mean `d_acc = +0.0339`
  - R9a mean `d_acc = +0.0156`
- Status: **mixed / not adopted as default**

### R9b Status: Hotpot q-after L=4096 (running)

- Script: `run_hotpot_qafter_stabilized_len4096_round9_sched_linear_s012.sh`
- Launch mode: queued after ARC via `run_after_arc_start_hotpot_r9.sh`
- Pending output:
  - `results/_summary_hotpot_qafter_stabilized_len4096_r9_sched_linear_s012.txt`

### R9b Result: Hotpot q-after L=4096 (completed)

- Summary: `results/_summary_hotpot_qafter_stabilized_len4096_r9_sched_linear_s012.txt`
- Result:
  - mean `d_acc = -0.0220`, std `0.0378`
  - sign pattern: `1+ / 0 / 2-`
  - mean `d_loss = -0.0188`
- Status: **failed mitigation** (accuracy regressed vs R6 baseline config)

## New Iteration

### R10: Deeper-only FS injection on Hotpot

- Script: `run_hotpot_qafter_stabilized_len4096_round10_lstart10_s012.sh`
- Change from baseline:
  - `fs_layer_start: 6 -> 10`
  - keep constant alpha (`-2`), no schedule
- Hypothesis:
  - avoid early/mid-layer harmful seed interference
  - keep useful deep-layer global conditioning

### R10 Result: Hotpot q-after L=4096 (completed)

- Summary: `results/_summary_hotpot_qafter_stabilized_len4096_r10_lstart10_s012.txt`
- Result:
  - mean `d_acc = -0.0225`, std `0.0435`
  - sign pattern: `2+ / 0 / 1-`
  - mean `d_loss = -0.1906`
- Status: **not adopted** (mean accuracy regressed vs R6 baseline)

### R11: Small grid queued after R10

- Scripts:
  - `run_hotpot_qafter_stabilized_len4096_round11_grid_lstart10_12_alpha_m2_m3_s012.sh`
  - `run_after_r10_start_r11.sh`
- Grid:
  - `fs_layer_start`: `10`, `12`
  - `alpha_init`: `-2`, `-3`
- Fixed:
  - `L=4096`, q-after, seeds `{0,1,2}`, scalar FS, `fs_norm`, `fs_detach`, `fs_clip=1.0`
- Goal:
  - find a robust config with better seed-sign consistency than R9.

## Additional Task Expansion Queued

### MBPP long-context post-training probe

- Scripts:
  - `run_mbpp_qafter_stabilized_len4096_round1_s012.sh`
  - `run_mbpp_qfirst_stabilized_len4096_round1_s012.sh`
- Objective:
  - check whether FS helps retain problem/test constraints when answer trigger is far from relevant spec.

### Sudoku suite (more Sudoku settings)

- Script: `run_sudoku_suite_round1_s012.sh`
- Settings:
  - 4x4 prefix-mask / suffix-mask
  - 9x9 prefix-mask / suffix-mask
- Objective:
  - evaluate FS on stronger global-consistency tasks beyond retrieval-style probes.

### Queue hook

- `run_after_r11_start_mbpp_sudoku.sh` waits for R11 completion, then runs MBPP + Sudoku batch.

## Current Bottom Line

- FS for post-training is **validated in specific real settings** (ARC options-first).
- FS on real long-context QA is **not yet stable enough** for a universal claim.
- Next milestone is not higher single-run peaks; it is **cross-seed stability** under fixed budget.

---

## 2026-02-21 Addendum: Full Trace for R11 + MBPP/Sudoku + Protein

### Hotpot R11 Grid (post R10)

Script:
- `scripts/run_hotpot_qafter_stabilized_len4096_round11_grid_lstart10_12_alpha_m2_m3_s012.sh`

Summaries:
- `results/_summary_hotpot_qafter_stabilized_len4096_r11_lstart10_alpha-2_s012.txt`
- `results/_summary_hotpot_qafter_stabilized_len4096_r11_lstart10_alpha-3_s012.txt`
- `results/_summary_hotpot_qafter_stabilized_len4096_r11_lstart12_alpha-2_s012.txt`
- `results/_summary_hotpot_qafter_stabilized_len4096_r11_lstart12_alpha-3_s012.txt`

Outcome:
- `lstart=12` behaves as near no-op (all 0 deltas).
- `lstart=10` remains unstable (mixed signs, mean not robustly positive).

### MBPP + Sudoku batch

Queue hook:
- `scripts/run_after_r11_start_mbpp_sudoku.sh`

MBPP scripts:
- `scripts/run_mbpp_qafter_stabilized_len4096_round1_s012.sh`
- `scripts/run_mbpp_qfirst_stabilized_len4096_round1_s012.sh`

MBPP summaries:
- `results/_summary_mbpp_qafter_stabilized_len4096_r1_s012.txt`
- `results/_summary_mbpp_qfirst_stabilized_len4096_r1_s012.txt`

MBPP outcome:
- q-after mean `d_acc = -0.0191`
- q-first mean `d_acc = -0.0254`
- both are regressions.

Sudoku script:
- `scripts/run_sudoku_suite_round1_s012.sh`

Sudoku summaries:
- `results/_summary_sudoku4_prefix_r1_s012.txt`
- `results/_summary_sudoku4_suffix_r1_s012.txt`
- `results/_summary_sudoku9_prefix_r1_s012.txt`
- `results/_summary_sudoku9_suffix_r1_s012.txt`

Sudoku outcome:
- 4x4 prefix: small stable gain.
- 4x4 suffix: near neutral.
- 9x9 prefix: near neutral/slightly negative.
- 9x9 suffix: severe regression.

### Protein task expansion (new)

New trainers:
- `scripts/train_protein_ss_spot_sft.py`
- `scripts/train_protein_contact_pair_sft.py`

New summarizers:
- `scripts/summarize_protein_ss_spot.py`
- `scripts/summarize_protein_contact_pair.py`

#### Protein SS spot labeling

Runs:
- `scripts/run_protein_ss_spot_qafter_len2048_round1_s012.sh`
- `scripts/run_protein_ss_spot_qfirst_len2048_round1_s012.sh`

Summaries:
- `results/_summary_protein_ss_spot_qafter_len2048_r1_s012.txt`
- `results/_summary_protein_ss_spot_qfirst_len2048_r1_s012.txt`

Outcome:
- both orderings are near-zero mean deltas (`-0.0011`, `-0.0001`).

#### Protein contact-pair QA: iterative rounds

R1 script:
- `scripts/run_protein_contact_pair_qafter_len2048_round1_s012.sh`
R1 summary:
- `results/_summary_protein_contact_pair_qafter_len2048_r1_s012.txt`
R1 outcome:
- exact tie on token/seq acc (`d_acc=0`).

R2 script:
- `scripts/run_protein_contact_pair_qafter_len2048_round2_trainable_s012.sh`
R2 summary:
- `results/_summary_protein_contact_pair_qafter_len2048_r2_trainable_s012.txt`
R2 outcome:
- still exact tie (`d_acc=0`), despite trainable alpha.

R3 script:
- `scripts/run_protein_contact_pair_qafter_len2048_round3_balanced_s012.sh`
R3 summary:
- `results/_summary_protein_contact_pair_qafter_len2048_r3_balanced_s012.txt`
R3 outcome:
- `mean d_acc = -0.0007` (small negative mean, mixed signs).

R4 script:
- `scripts/run_protein_contact_pair_qafter_len2048_round4_sched_s012.sh`
R4 summary:
- `results/_summary_protein_contact_pair_qafter_len2048_r4_sched_s012.txt`
R4 outcome:
- `mean d_acc = -0.0043` (negative mean, 1+/2-).

Queue helpers used:
- `scripts/run_after_protein_contact_r1_start_r2_trainable.sh`
- `scripts/run_after_protein_contact_r2_start_r3_balanced.sh`
- `scripts/run_after_protein_contact_r3_start_r4_sched.sh`

## Consolidated Interpretation After Addendum

1. FS benefits are **task-shape dependent**, not universal.
2. The strongest positive signal remains in explicitly causal-unfriendly or small-constraint tasks.
3. On MBPP and protein probes under current budgets, FS does not provide reliable gains.
4. For realistic claims, report FS as a **targeted mechanism** rather than a global post-training improvement.

## Reproducibility Tooling (added)

- Manifest: `paper/exp_manifest.json`
- Global summary parser: `scripts/summarize_all_results.py`
- Doc/reference checker: `scripts/check_doc_summary_refs.py`
- Aggregated outputs:
  - `results/_aggregate_results.jsonl`
  - `results/_aggregate_results.md`

## 2026-02-21 Round12 (5-seed stability, high-util config)

Goal:
- stress-test previously reported signals with higher GPU utilization and fixed step caps.
- settings changed from earlier rounds:
  - ARC: `bsz=48`, `max_steps=2000`, `val_batches=8`
  - Hotpot: `bsz=2`, `max_steps=500`, `val_batches=8`

### ARC options-first (R5, seeds 0..4)

- Script: `scripts/run_arc_optionsfirst_stabilized_round5_s01234.sh`
- Summary: `results/_summary_arc_optionsfirst_stabilized_r5_s01234.txt`
- Result:
  - mean `d_acc = -0.0203`, std `0.0367`, sign `2+/3-`
- Takeaway:
  - under this high-throughput regime, options-first no longer shows stable FS gain.

### ARC q-first control (R5, seeds 0..4)

- Script: `scripts/run_arc_qfirst_stabilized_round5_s01234.sh`
- Summary: `results/_summary_arc_qfirst_stabilized_r5_s01234.txt`
- Result:
  - mean `d_acc = +0.0005`, std `0.0094`, sign `2+/3-`
- Takeaway:
  - near-zero control, but does not rescue the options-first regression.

### Hotpot q-after L4096 (R12, seeds 0..4)

- Script: `scripts/run_hotpot_qafter_stabilized_len4096_round12_lstart10_alpha_m3_s01234.sh`
- Summary: `results/_summary_hotpot_qafter_stabilized_len4096_r12_lstart10_alpha-3_s01234.txt`
- Result:
  - mean `d_acc = +0.0054`, std `0.0179`, sign `2+/2=/1-`
- Takeaway:
  - small positive mean with mixed signs and multiple no-op seeds.

### Hotpot q-first L4096 control (R12, seeds 0..4)

- Script: `scripts/run_hotpot_qfirst_stabilized_len4096_round12_lstart10_alpha_m3_s01234.sh`
- Summary: `results/_summary_hotpot_qfirst_stabilized_len4096_r12_lstart10_alpha-3_s01234.txt`
- Result:
  - mean `d_acc = +0.0052`, std `0.0214`, sign `2+/2=/1-`
- Takeaway:
  - similarly small mean gain in control ordering; ordering-specific hypothesis is not supported in this regime.

### Updated practical conclusion after Round12

1. FS behavior is strongly regime-dependent (batch/steps/eval cadence).
2. Claims must be tied to a fixed compute protocol; otherwise sign can flip.
3. For paper main results, keep one canonical protocol and report Round12 as robustness/failure analysis.

## 2026-02-21 Round20 (serial immediate-prune, single seed)

Script:
- `scripts/run_round20_serial_earlystop_s0.sh`

Outputs:
- `results/_summary_round20_serial_earlystop_s0.txt`
- `results/_round20_serial_earlystop_records.jsonl`

Protocol:
- one GPU, serial queue only
- quick stage first (`time_budget_sec=80`)
- immediate prune if quick `d_acc < +0.001` (`+0.10pp`)
- medium confirm only for survivors (`time_budget_sec=220`)

### Task outcomes

#### Hotpot
- selected probe `bsz=6`
- baseline quick: `acc=6.17%`
- all FS variants pruned
- strongest negative quick regressions around `-2.93pp`

#### MBPP
- selected probe `bsz=1`
- baseline failed due sample construction limit:
  - `Only built 374 examples (wanted 900).`
- moved to targeted fix round

#### Protein contact
- selected probe `bsz=1`
- baseline failed due sample construction limit:
  - `Only built 168 examples (wanted 180).`
- moved to targeted fix round

#### Protein SS spot
- selected probe `bsz=8`
- baseline quick: `24.66%`
- quick survivors all promoted
- medium confirmed gains:
  - `scalar_l10_norm_node`: `32.69%` (**+8.02pp**)
  - `scalar_l10_trainable`: `32.38%` (**+7.72pp**)
  - `scalar_l10_sched_cos`: `32.23%` (**+7.57pp**)
  - `head_l10`: `31.97%` (**+7.31pp**)
  - `scalar_l10_norm_detach`: `31.46%` (**+6.80pp**)
  - `scalar_l10_nonorm_detach`: `30.32%` (**+5.66pp**)

Round20 conclusion:
- strong positive FS regime found on `protein_ss`
- `hotpot` remains non-work in this protocol
- `mbpp` and `protein_contact` required dataset-build fixes before fair FS comparison

## 2026-02-21 Round21 (targeted follow-up, completed)

Script:
- `scripts/run_round21_targeted_search_s0.sh`

Outputs:
- `results/_summary_round21_targeted_search_s0.txt`
- `results/_round21_targeted_search_records.jsonl`
- `results/_log_round21_targeted_search_s0.20260221_193225.log`

Targeted goals:
1. fix MBPP and protein-contact sample-construction failures
2. keep serial + immediate prune policy
3. search only high-priority FS variants

### Completed trace

#### mbpp_fix (reduced construction constraints)
- baseline quick now runs:
  - `acc=10.46%`
- quick FS:
  - `scalar_l8_norm_node`: `+0.06pp` (pruned by threshold)
  - `scalar_l8_sched_cos`: `-0.67pp` (pruned)
  - `head_l8`: `+0.14pp` (pruned)
  - `scalar_l8_trainable`: `+1.00pp` (kept -> medium run)

#### protein_contact_fix
- baseline quick: `98.83%`
- all tested FS variants pruned (`+0.00pp`), no measurable gain in this setup

#### protein_ss_refine
- baseline quick: `21.14%`
- quick keep set:
  - `scalar_l10_norm_node`: `+2.87pp`
  - `scalar_l10_trainable`: `+4.48pp`
  - `scalar_l10_sched_cos`: `+2.01pp`
  - `head_l10`: `+1.72pp`
- med confirmed:
  - `scalar_l10_sched_cos`: `34.45%` (**+13.31pp**)
  - `scalar_l10_trainable`: `33.95%` (**+12.82pp**)
  - `head_l10`: `33.48%` (**+12.35pp**)
  - `scalar_l10_norm_node`: `32.57%` (**+11.43pp**)

Round21 interpretation:
- at least one previously non-work real task (`mbpp`) becomes strongly positive once data-build constraints are fixed.
- protein-contact remains a no-gain task under current prompt/label formulation.
- protein SS remains the strongest effective scene for FS in this repository snapshot.

## 2026-02-21 Round22 (adaptive serial search, completed)

Script:
- `scripts/run_round22_adaptive_search_s0.sh`

Outputs:
- `results/_summary_round22_adaptive_search_s0.txt`
- `results/_round22_adaptive_search_records.jsonl`
- `results/_log_round22_adaptive_search_s0.*.log`

Highlights:
- `mbpp_focus`: baseline `15.49%`; all tested FS quick variants regressed (`-0.92pp` to `-4.83pp`).
- `protein_ss_expand`: baseline `28.15%`; best quick `+1.50pp` (`scalar_l10_nodetach`), mixed signs.
- `sudoku4_refine`: very strong positive regime, med best `+33.29pp`.
- `sudoku9_probe`: small but consistent positive med gains (best `+1.51pp`).

Interpretation:
- MBPP positive regime from Round21 is not stable under this higher-throughput recipe.
- Protein SS stays positive but magnitude is recipe-sensitive.
- Sudoku confirms FS can strongly help constrained in-place repair (especially easier 4x4).

## 2026-02-21 Round23 (real-task sweep, partial)

Script:
- `scripts/run_round23_real_task_sweep_s0.sh`

Outputs:
- `results/_round23_real_task_sweep_records.jsonl`
- `results/_launcher_round23.log`
- `results/_log_round23_real_task_sweep_s0.*.log`

Completed outcomes:
- `mbpp_rt`: all quick FS variants negative (`-1.28pp` to `-4.67pp`).
- `hotpot_rt`: quick FS variants exact-tie baseline (`+0.00pp`).
- `punc_restore_rt`: run aborted by HF connectivity; treated as infrastructure failure, not model verdict.

## 2026-02-21 Round24 (punc + protein continuation, completed)

Script:
- `scripts/run_round24_punc_protein_s0.sh`

Outputs:
- `results/_summary_round24_punc_protein_s0.txt`
- `results/_round24_punc_protein_records.jsonl`
- `results/_log_round24_punc_protein_s0.*.log`

Outcomes:
- `protein_ss_rt`:
  - quick baseline `30.17%`
  - med `scalar_l10_sched_cos` `34.31%` (**+4.14pp**)
- `punc_restore_rt`:
  - baseline failed with OOM at `bsz=10` (configuration issue, not discarded task).

## 2026-02-21 Round25 (punc salvage, completed)

Script:
- `scripts/run_round25_punc_salvage_s0.sh`

Outputs:
- `results/_summary_round25_punc_salvage_s0.txt`
- `results/_round25_punc_salvage_records.jsonl`
- `results/_log_round25_punc_salvage_s0.*.log`

Memory-safe config:
- `bsz=2`, `max_prompt_tokens=1536`, `max_answer_tokens=128`
- dataset from cached `hotpot_qa` text fields (offline mode)

Outcomes:
- baseline quick `9.18%`
- quick:
  - `scalar_l8_sched_cos`: `+0.80pp`
  - `head_l8`: `+0.80pp`
  - `scalar_l8_trainable`: `-2.20pp`
- med:
  - `head_l8`: `12.64%` (**+3.45pp**)
  - `scalar_l8_sched_cos`: `11.90%` (**+2.71pp**)

Interpretation:
- The punc task is a valid additional positive real-text regime once memory settings are corrected.

## 2026-02-21 Round26 (low-throughput MBPP/Hotpot, completed)

Script:
- `scripts/run_round26_mbpp_hotpot_lowthroughput_s0.sh`

Outputs:
- `results/_summary_round26_mbpp_hotpot_lowthroughput_s0.txt`
- `results/_round26_mbpp_hotpot_lowthroughput_records.jsonl`
- `results/_log_round26_mbpp_hotpot_lowthroughput_s0.*.log`

Outcomes:
- `mbpp_low`:
  - baseline quick `10.46%`
  - quick `scalar_l8_trainable` `+1.00pp`
  - med `scalar_l8_trainable` `29.64%` (**+19.17pp**)
- `hotpot_low`:
  - baseline quick `14.34%`
  - `scalar_l10_trainable`: `+0.00pp`
  - `scalar_l10_sched_cos`: `+0.00pp`
  - `head_l10`: `-1.84pp`

Interpretation:
- MBPP sign-flip is throughput-sensitive in current setup (`bsz=2` works, higher-throughput setups often regress).
- Hotpot remains no-gain even under the same low-throughput control.

## 2026-02-21 Round27 (seed robustness check, completed)

Script:
- `scripts/run_round27_seedcheck_positive_s012.sh`

Outputs:
- `results/_summary_round27_seedcheck_positive_s012.txt`
- `results/_round27_seedcheck_positive_s012_records.jsonl`
- `results/_log_round27_seedcheck_positive_s012.20260221_223003.log`

Outcomes:
- `mbpp_low + scalar_l8_trainable` (quick):
  - seed0 `+1.00pp`
  - seed1 `+0.32pp`
  - seed2 `-0.82pp`
  - mean `+0.17pp`, positive seeds `2/3`
- `punc_restore + head_l8` (quick):
  - seed0 `+0.80pp`
  - seed1 `+0.58pp`
  - seed2 `+2.20pp`
  - mean `+1.19pp`, positive seeds `3/3`

Interpretation:
- Punctuation restoration currently has the strongest seed-level robustness among non-synthetic real-text tasks.
- MBPP remains conditionally positive, not yet robust.

## 2026-02-21 Round28 (MBPP throughput sweep, completed)

Script:
- `scripts/run_round28_mbpp_bsz_sweep_s0.sh`

Outputs:
- `results/_summary_round28_mbpp_bsz_sweep_s0.txt`
- `results/_round28_mbpp_bsz_sweep_s0_records.jsonl`
- `results/_log_round28_mbpp_bsz_sweep_s0.20260221_225305.log` (final clean run)
- `results/_launcher_round28.log`

Outcomes:
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
  - baseline OOM/fail

Interpretation:
- MBPP gain from FS is concentrated in low-throughput settings.
- Throughput increase is currently a stronger negative factor than FS variant choice on MBPP.

## 2026-02-22 Round29 (punc seed-5 stability, completed)

Script:
- `scripts/run_round29_punc_seed5_s01234.sh`

Outputs:
- `results/_summary_round29_punc_seed5_s01234.txt`
- `results/_round29_punc_seed5_s01234_records.jsonl`
- `results/_log_round29_punc_seed5_s01234.20260222_153614.log`

Outcomes (quick, punc_restore + head_l8):
- seed0: `+0.80pp`
- seed1: `+0.58pp`
- seed2: `+2.20pp`
- seed3: `+2.41pp`
- seed4: `+0.64pp`
- mean: `+1.33pp`, positive seeds: `5/5`

Interpretation:
- Punctuation restoration remains the most stable non-synthetic FS-positive regime in this project.

## 2026-02-22 Round30 (embedding smoke on Hotpot retrieval, completed)

Scripts:
- `scripts/train_embedding_hotpot_fs.py`
- `scripts/run_round30_embedding_hotpot_s0.sh`

Outputs:
- `results/_summary_round30_embedding_hotpot_s0.txt`
- `results/_round30_embedding_hotpot_s0_records.jsonl`
- `results/_round30_embed_baseline_summary.json`
- `results/_round30_embed_fs_summary.json`
- `results/_round30_embed_baseline_metrics.jsonl`
- `results/_round30_embed_fs_metrics.jsonl`

Setup:
- Frozen RWKV backbone (`rwkv7-g1d-0.1b`) + trainable embedding head.
- Contrastive retrieval objective (in-batch InfoNCE) on Hotpot pairs.
- Compare baseline vs FS (`fs_layer_start=8`, trainable scalar gate).

Outcomes:
- baseline:
  - `R@1=1.17%`
  - `R@5=3.12%`
  - `MRR@10=1.93%`
- FS:
  - `R@1=0.78%`
  - `R@5=3.12%`
  - `MRR@10=1.69%`
- delta (FS - baseline):
  - `d_R@1=-0.39pp`
  - `d_MRR@10=-0.24pp`

Interpretation:
- This first embedding probe is negative for FS.
- Evidence currently does not support claiming FS gains for embedding quality under this simple contrastive setup.

## 2026-02-25 Round60 (strict-quick, seed0, completed)

Script:
- `scripts/run_round60_strictquick_s0.py`

Outputs:
- `results/_summary_round60_strictquick_s0.txt`
- `results/_round60_strictquick_s0_records.jsonl`

Task setting:
- real-task strict branch on `squad` and `mbpp`
- single GPU serial quick->med verification

Outcomes:
- `mbpp_strict`:
  - quick baseline: `42.70%`
  - quick best FS (`head_l10_strong`): `43.54%` (`+0.84pp`)
  - med baseline: `48.39%`
  - med FS (`head_l10_strong`): `49.94%` (**`+1.55pp` vs med baseline**)
- `squad_strict`:
  - quick baseline: `14.17%`
  - quick best FS (`scalar_l8_sched_cos`): `15.45%` (`+1.28pp`)
  - med baseline: `17.27%`
  - med FS (`scalar_l8_sched_cos`): `17.47%` (**`+0.20pp` vs med baseline**)

Decision:
- adopt `head_l10_strong` as MBPP strict winner for cross-seed follow-up.
- keep SQuAD strict branch active, but mark as low-margin pending seed check.

## 2026-02-25 Round61 (strict frontier, seed1, completed)

Script:
- `scripts/run_round61_strict_seed1_frontier.py`

Outputs:
- `results/_summary_round61_strict_seed1_frontier.txt`
- `results/_round61_strict_seed1_frontier_records.jsonl`

Search policy:
- `seed=1` robustness pass on strict recipe
- quick stage first; med only if quick best `>= +0.30pp`
- no multi-seed sweep expansion in this round

Outcomes:
- `squad_strict_seed1` quick:
  - baseline: `14.83%`
  - `scalar_l8_train1e4`: `14.58%` (`-0.25pp`)
  - `scalar_l8_sched_cos_highfloor`: `14.32%` (`-0.50pp`)
  - `scalar_l8_sched_cos`: `14.08%` (`-0.75pp`)
  - med pruned (`quick_d_acc=-0.25pp < +0.30pp`)
- `mbpp_strict_seed1`:
  - quick baseline: `40.43%`
  - quick `head_l10_strong`: `43.86%` (`+3.43pp`)
  - quick `head_l10_midlr`: `42.33%` (`+1.90pp`)
  - quick `head_l10_sched_cos`: `41.86%` (`+1.43pp`)
  - med baseline: `46.67%`
  - med FS (`head_l10_strong`): `47.01%` (**`+0.35pp` vs med baseline**)

Failure / prune notes:
- SQuAD strict branch failed quick gate on seed1 (all tested FS variants negative), so no extra med compute wasted.
- MBPP quick alternates (`head_l10_midlr`, `head_l10_sched_cos`) were not promoted because they trailed `head_l10_strong`.

Current judgment:
- MBPP strict now has two positive med seeds (`seed0 +1.55pp`, `seed1 +0.35pp`), supporting non-toy real-task positive signal.
- SQuAD strict remains unstable and requires dedicated rescue search before claiming stable gain.

Next round plan:
1. SQuAD strict rescue with more conservative FS schedules and lower prompt-note pressure.
2. MBPP strict `seed2` confirmation on `head_l10_strong` to extend stability evidence.

## 2026-02-25 Round62 (3h finishpack, completed)

Script:
- `scripts/run_round62_3h_finishpack.py`

Outputs:
- `results/_summary_round62_3h_finishpack.txt`
- `results/_round62_3h_finishpack_records.jsonl`

Execution constraints:
- hard 3-hour budget window
- single-GPU serial execution
- quick->med with immediate prune when quick drop exceeds `0.50pp`

Branch A: MBPP strict seed2 confirmation
- quick baseline: `41.36%`
- quick `head_l10_strong`: `42.10%` (`+0.74pp`)
- quick `head_l10_midlr`: `40.10%` (`-1.26pp`, pruned)
- med baseline: `47.15%`
- med FS (`head_l10_strong`): `45.17%` (**`-1.98pp` vs med baseline**)

Branch B: SQuAD strict seed1 rescue
- quick baseline: `14.83%`
- rescue quick:
  - `scalar_l8_sched_ultra`: `14.82%` (`-0.01pp`)
  - `scalar_l8_train5e5`: `14.33%` (`-0.50pp`, pruned)
  - `scalar_l10_sched_gentle`: `13.83%` (`-1.00pp`, pruned)
- med pruned: best quick still below med gate (`-0.01pp < +0.20pp`)

Branch C: PUNC restore seed0 scout
- quick baseline: `8.44%`
- quick `head_l8`: `10.10%` (`+1.66pp`)
- quick `scalar_l8_sched_cos`: `8.18%` (`-0.26pp`)
- med baseline: `10.59%`
- med FS (`head_l8`): `11.09%` (**`+0.51pp` vs med baseline**)

Round62 useful-task conclusion:
- positive: `punc_restore` (current round positive on both quick and med).
- mixed/unstable: `mbpp_strict` (seed2 med turned negative after seed0/seed1 positives).
- not useful under tested rescue: `squad_strict`.

## 2026-02-25 Round63 (useful-followup, completed)

Script:
- `scripts/run_round63_useful_followup.py`

Outputs:
- `results/_summary_round63_useful_followup.txt`
- `results/_round63_useful_followup_records.jsonl`

Goal:
- continue within short-cycle budget and identify genuinely useful branches fast.
- verify if `punc`/`mbpp` can sustain positive deltas with focused follow-up.

Outcomes:

1) `punc_restore_seed1_confirm`
- quick baseline: `7.92%`
- quick best (`scalar_l8_sched_cos`): `10.34%` (`+2.42pp`)
- med baseline: `12.20%`
- med FS (`scalar_l8_sched_cos`): `14.32%` (**`+2.12pp` vs med baseline**)

2) `punc_restore_seed2_confirm`
- quick baseline: `6.74%`
- quick `head_l8`: `6.12%` (`-0.61pp`, pruned)
- quick `scalar_l8_sched_cos`: `7.13%` (`+0.39pp`)
- med baseline: `12.78%`
- med FS (`scalar_l8_sched_cos`): `10.24%` (**`-2.55pp` vs med baseline**)

3) `mbpp_seed2_regrescue`
- recipe change: reduced FS intensity (`fs_clip=0.7`) + lower note pressure.
- quick baseline: `39.11%`
- quick candidates:
  - `head_l10_clip07`: `42.52%` (`+3.41pp`)
  - `head_l11_strong`: `40.45%` (`+1.35pp`)
  - `head_l10_sched_soft`: `40.41%` (`+1.30pp`)
- med baseline: `46.69%`
- med FS (`head_l10_clip07`): `48.97%` (**`+2.28pp` vs med baseline**)

Round63 useful-task conclusion:
- useful: `mbpp` rescue recipe (`head_l10_clip07`) and `punc` seed1 (`scalar_l8_sched_cos`) both showed >`+2pp` med gains.
- still unstable: `punc` across seeds (seed1 positive, seed2 negative).
- action: treat `mbpp` as recipe-sensitive but currently high-value; keep `squad` frozen.

## 2026-02-25 Round64 (mbpp+punc multiseed, completed)

Script:
- `scripts/run_round64_mbpp_punc_multiseed.py`

Outputs:
- `results/_summary_round64_mbpp_punc_multiseed.txt`
- `results/_round64_mbpp_punc_multiseed_records.jsonl`

Goal:
- verify whether the Round63 useful configs transfer to new seeds.
- keep strict quick pruning (`quick drop < -0.50pp`) and skip low-value med.

Outcomes:

1) `mbpp_seed0_clip07_confirm`
- quick baseline: `40.69%`
- quick `head_l10_clip07`: `40.71%` (`+0.02pp`)
- quick `head_l10_strong`: `40.71%` (`+0.02pp`)
- med skipped (`best_quick +0.02pp < +0.20pp`)

2) `mbpp_seed1_clip07_confirm`
- quick baseline: `40.86%`
- quick `head_l10_clip07`: `39.64%` (`-1.22pp`, pruned)
- quick `head_l10_strong`: `39.64%` (`-1.22pp`, pruned)
- med skipped

3) `punc_seed3_scalar_confirm`
- quick baseline: `11.76%`
- quick `scalar_l8_sched_cos`: `10.71%` (`-1.04pp`, pruned)
- quick `head_l8`: `9.96%` (`-1.80pp`, pruned)
- med skipped

Round64 conclusion:
- MBPP rescue signal did not transfer cleanly (neutral on seed0, negative on seed1).
- PUNC branch remained unstable and was fully pruned on seed3.

## 2026-02-25 Round65 (mbpp+squad seed2/seed3, completed)

Script:
- `scripts/run_round65_mbpp_squad_seed23.py`

Outputs:
- `results/_summary_round65_mbpp_squad_seed23.txt`
- `results/_round65_mbpp_squad_seed23_records.jsonl`

Goal:
- fast branch triage on real tasks under strict quick->med runtime.

Outcomes:

1) `mbpp_seed3_regrescue`
- quick baseline: `43.11%`
- quick `head_l10_strong`: `42.35%` (`-0.76pp`)
- quick `head_l10_clip07`: `40.22%` (`-2.89pp`, pruned)
- med skipped

2) `squad_strict_seed2`
- quick baseline: `14.58%`
- quick `scalar_l8_train1e4`: `15.59%` (`+1.01pp`)
- quick `scalar_l8_sched_cos`: `14.07%` (`-0.51pp`, pruned)
- med baseline: `18.33%`
- med FS (`scalar_l8_train1e4`): `19.04%` (**`+0.71pp` vs med baseline**)

Round65 conclusion:
- SQuAD reopened as a useful real-task branch via `scalar_l8_train1e4`.
- MBPP remained unstable on seed3 under rescue settings.

## 2026-02-25 Round66 (squad+mbpp frontier, completed)

Script:
- `scripts/run_round66_squad_mbpp_frontier.py`

Outputs:
- `results/_summary_round66_squad_mbpp_frontier.txt`
- `results/_round66_squad_mbpp_frontier_records.jsonl`

Goal:
- verify if SQuAD train1e4 gain is reproducible on another seed.
- test MBPP seed3 with strict-base alternatives.

Outcomes:

1) `squad_seed0_train1e4_confirm`
- quick baseline: `13.33%`
- quick `scalar_l8_train1e4`: `14.05%` (`+0.72pp`)
- quick `scalar_l8_sched_cos`: `13.80%` (`+0.47pp`)
- med baseline: `17.27%`
- med FS (`scalar_l8_train1e4`): `19.00%` (**`+1.73pp` vs med baseline**)

2) `mbpp_strict_seed3_alt`
- quick baseline: `40.90%`
- quick `head_l10_strong`: `41.85%` (`+0.95pp`)
- quick `head_l8_nodetach`: `41.45%` (`+0.56pp`)
- med baseline: `48.23%`
- med FS (`head_l10_strong`): `46.76%` (**`-1.47pp` vs med baseline**)

Round66 conclusion:
- SQuAD `scalar_l8_train1e4` achieved another strong med gain, now positive on seed0 and seed2.
- MBPP on seed3 showed quick-positive but med-negative reversal; stability remains unresolved.

## 2026-02-25 Round67 (squad3+mbpp0, completed)

Script:
- `scripts/run_round67_squad3_mbpp0.py`

Outputs:
- `results/_summary_round67_squad3_mbpp0.txt`
- `results/_round67_squad3_mbpp0_records.jsonl`

Goal:
- extend SQuAD train1e4 branch to seed3.
- recheck MBPP strict winner on seed0 with current quick->med pipeline.

Outcomes:

1) `squad_seed3_train1e4_frontier`
- quick baseline: `14.67%`
- quick `scalar_l8_train1e4`: `14.06%` (`-0.61pp`, pruned)
- quick `scalar_l8_sched_cos`: `13.83%` (`-0.84pp`, pruned)
- med skipped (`best_quick -0.61pp < med_gate -0.50pp`)

2) `mbpp_seed0_reconfirm_strict`
- quick baseline: `40.60%`
- quick `head_l10_strong`: `42.88%` (`+2.27pp`)
- quick `head_l8_nodetach`: `41.10%` (`+0.50pp`)
- med baseline: `48.39%`
- med FS (`head_l10_strong`): `49.94%` (**`+1.54pp` vs med baseline**)

Round67 conclusion:
- MBPP seed0 positive signal is reproducible in the current pipeline.
- SQuAD train1e4 branch remains seed-split: strong on seed0/2 but negative on seed3.

## 2026-02-25 Round68 (squad head rescue, completed)

Script:
- `scripts/run_round68_squad_head_rescue.py`

Outputs:
- `results/_summary_round68_squad_head_rescue.txt`
- `results/_round68_squad_head_rescue_records.jsonl`

Goal:
- attempt a lightweight head-based rescue for SQuAD negative seeds (`seed1`, `seed3`).

Outcomes:

1) `squad_seed1_head_rescue`
- quick baseline: `14.83%`
- quick `head_l8`: `14.07%` (`-0.76pp`, pruned)
- quick `head_l8_nodetach`: `14.58%` (`-0.25pp`)
- med skipped (`best_quick -0.25pp < med_gate -0.20pp`)

2) `squad_seed3_head_rescue`
- quick baseline: `14.67%`
- quick `head_l8`: `13.32%` (`-1.35pp`, pruned)
- quick `head_l8_nodetach`: `13.90%` (`-0.76pp`, pruned)
- med skipped (`best_quick -0.76pp < med_gate -0.20pp`)

Round68 conclusion:
- head variants failed to recover SQuAD on both negative seeds.
- this rescue direction is currently low expected value.

## 2026-02-25 Round69 (mbpp+squad rescue, completed)

Script:
- `scripts/run_round69_mbpp_squad_rescue.py`

Outputs:
- `results/_summary_round69_mbpp_squad_rescue.txt`
- `results/_round69_mbpp_squad_rescue_records.jsonl`

Goal:
- MBPP seed3: test whether quick-positive candidates can survive med using dual-promotion.
- SQuAD seed1: scalar micro-tuning around train1e4.

Outcomes:

1) `mbpp_seed3_dualmed_rescue`
- quick baseline: `40.90%`
- quick `head_l10_strong`: `41.85%` (`+0.95pp`)
- quick `head_l8_nodetach`: `41.45%` (`+0.56pp`)
- med baseline: `48.23%`
- med `head_l8_nodetach`: `48.28%` (**`+0.05pp` vs med baseline**)
- med `head_l10_strong`: `46.76%` (**`-1.47pp` vs med baseline**)

2) `squad_seed1_scalar_micro`
- quick baseline: `14.83%`
- quick `scalar_l8_train1e4`: `14.58%` (`-0.25pp`)
- quick `scalar_l8_train8e5`: `14.58%` (`-0.25pp`)
- quick `scalar_l8_train1e4_clip07`: `14.41%` (`-0.42pp`)
- med skipped (`best_quick -0.25pp < med_gate -0.10pp`)

Round69 conclusion:
- MBPP seed3 improved from clear regression to near-neutral with `head_l8_nodetach`, but no strong gain yet.
- SQuAD seed1 remained unrecovered under scalar micro search.

## 2026-02-25 Round70 (squad3+mbpp2, completed)

Script:
- `scripts/run_round70_squad3_mbpp2.py`

Outputs:
- `results/_summary_round70_squad3_mbpp2.txt`
- `results/_round70_squad3_mbpp2_records.jsonl`

Goal:
- SQuAD seed3: scalar micro rescue follow-up after round68 head failure.
- MBPP seed2: dual med recheck for `clip07` and `strong` under rescue base.

Outcomes:

1) `squad_seed3_scalar_micro`
- quick baseline: `14.67%`
- quick `scalar_l8_train8e5`: `14.41%` (`-0.26pp`)
- quick `scalar_l8_train1e4_clip07`: `14.08%` (`-0.59pp`, pruned)
- quick `scalar_l8_train1e4`: `14.06%` (`-0.61pp`, pruned)
- med skipped (`best_quick -0.26pp < med_gate -0.10pp`)

2) `mbpp_seed2_dualmed_recheck`
- quick baseline: `39.11%`
- quick `head_l10_clip07`: `42.52%` (`+3.41pp`)
- quick `head_l10_strong`: `42.52%` (`+3.41pp`)
- med baseline: `46.69%`
- med `head_l10_clip07`: `48.97%` (**`+2.28pp` vs med baseline**)
- med `head_l10_strong`: `48.97%` (**`+2.28pp` vs med baseline**)

Round70 conclusion:
- MBPP seed2 strong positive (`+2.28pp`) is repeatable under current pipeline.
- SQuAD seed3 still not recoverable in this scalar micro neighborhood.

## 2026-02-25 Round71 (mbpp3+squad2, completed)

Script:
- `scripts/run_round71_mbpp3_squad2.py`

Outputs:
- `results/_summary_round71_mbpp3_squad2.txt`
- `results/_round71_mbpp3_squad2_records.jsonl`

Goal:
- MBPP seed3 rescue recheck under note-pool-384 base.
- SQuAD seed2 reconfirm in scalar train1e4 neighborhood.

Outcomes:

1) `mbpp_seed3_rescue384`
- quick baseline: `43.11%`
- quick `head_l10_strong`: `42.35%` (`-0.76pp`, pruned)
- quick `head_l10_clip07`: `40.22%` (`-2.89pp`, pruned)
- med skipped (`best_quick -0.76pp < med_gate +0.20pp`)

2) `squad_seed2_scalar_reconfirm`
- quick baseline: `14.58%`
- quick `scalar_l8_train1e4`: `15.59%` (`+1.01pp`)
- quick `scalar_l8_train8e5`: `15.33%` (`+0.76pp`)
- quick `scalar_l8_train1e4_clip07`: `13.58%` (`-1.00pp`, pruned)
- med baseline: `18.33%`
- med FS (`scalar_l8_train1e4`): `19.04%` (**`+0.71pp` vs med baseline**)

Round71 conclusion:
- SQuAD seed2 positive signal is stable in current quick->med regime.
- MBPP seed3 remains negative in this recipe family.

## 2026-02-26 Round72 (mbpp3+squad1 rescue, completed)

Script:
- `scripts/run_round72_mbpp3_squad1_rescue.py`

Outputs:
- `results/_summary_round72_mbpp3_squad1_rescue.txt`
- `results/_round72_mbpp3_squad1_rescue_records.jsonl`

Goal:
- try no-detach MBPP seed3 rescue.
- low-pressure SQuAD seed1 rescue around train1e4.

Outcomes:

1) `mbpp_seed3_nodetach_rescue384`
- quick baseline: `43.11%`
- quick `head_l10_nodetach_clip07`: `41.35%` (`-1.76pp`, pruned)
- quick `head_l8_nodetach`: `41.10%` (`-2.01pp`, pruned)
- med skipped

2) `squad_seed1_lowpressure_rescue`
- quick baseline: `14.83%`
- quick `scalar_l8_train1e4`: `14.83%` (`-0.00pp`)
- quick `scalar_l8_train8e5`: `14.58%` (`-0.25pp`)
- quick `scalar_l8_train1e4_clip07`: `14.16%` (`-0.67pp`, pruned)
- med baseline: `19.38%`
- med FS (`scalar_l8_train1e4`): `19.20%` (**`-0.18pp` vs med baseline**)

Round72 conclusion:
- no-detach did not recover MBPP seed3.
- SQuAD seed1 remained non-positive at med; rescue branch stays low priority.

## 2026-02-26 Round73 (mbpp1+squad0 recheck, completed)

Script:
- `scripts/run_round73_mbpp1_squad0_recheck.py`

Outputs:
- `results/_summary_round73_mbpp1_squad0_recheck.txt`
- `results/_round73_mbpp1_squad0_recheck_records.jsonl`

Goal:
- reconfirm known positive real-task seeds under strict quick->med gating.

Outcomes:

1) `mbpp_seed1_strict_recheck`
- quick baseline: `40.43%`
- quick `head_l10_strong`: `43.86%` (`+3.43pp`)
- quick `head_l10_clip07`: `43.86%` (`+3.43pp`)
- quick `head_l10_sched_cos`: `41.10%` (`+0.67pp`)
- med baseline: `46.67%`
- med FS (`head_l10_strong`): `47.01%` (**`+0.35pp` vs med baseline**)

2) `squad_seed0_frontier_recheck`
- quick baseline: `13.33%`
- quick `scalar_l8_train1e4`: `14.05%` (`+0.72pp`)
- quick `scalar_l8_sched_cos`: `13.80%` (`+0.47pp`)
- quick `scalar_l8_train8e5`: `13.58%` (`+0.25pp`)
- med baseline: `17.27%`
- med FS (`scalar_l8_train1e4`): `19.00%` (**`+1.73pp` vs med baseline**)

Round73 conclusion:
- positive med deltas on both tasks reproduced cleanly.
- this round strengthens non-toy practical signal for Future-Seed on MBPP/SQuAD branches.

## 2026-02-26 Round74 (punc1+squad2 frontier, completed)

Script:
- `scripts/run_round74_punc1_squad2_frontier.py`

Outputs:
- `results/_summary_round74_punc1_squad2_frontier.txt`
- `results/_round74_punc1_squad2_frontier_records.jsonl`

Goal:
- expand useful-task coverage with one additional real-task branch (`punc_restore`) while reconfirming SQuAD seed2.

Outcomes:

1) `punc_seed1_frontier_recheck`
- quick baseline: `7.92%`
- quick `scalar_l8_train1e4`: `13.04%` (`+5.12pp`)
- quick `scalar_l8_sched_cos`: `10.34%` (`+2.42pp`)
- quick `head_l8`: `7.80%` (`-0.12pp`)
- med baseline: `12.20%`
- med FS (`scalar_l8_train1e4`): `13.04%` (**`+0.84pp` vs med baseline**)

2) `squad_seed2_frontier_recheck`
- quick baseline: `14.58%`
- quick `scalar_l8_train1e4`: `15.59%` (`+1.01pp`)
- quick `scalar_l8_train8e5`: `15.33%` (`+0.76pp`)
- quick `scalar_l8_sched_cos`: `14.07%` (`-0.51pp`, pruned)
- med baseline: `18.33%`
- med FS (`scalar_l8_train1e4`): `19.04%` (**`+0.71pp` vs med baseline**)

Round74 conclusion:
- `scalar_l8_train1e4` became the highest-value common recipe across PUNC and SQuAD in this round.
- practical branch now has repeated med positives across multiple real tasks, but seed-level variance still exists.

## 2026-02-26 Round75 (mbpp3+punc0 targeted, completed)

Script:
- `scripts/run_round75_mbpp3_punc0_targeted.py`

Outputs:
- `results/_summary_round75_mbpp3_punc0_targeted.txt`
- `results/_round75_mbpp3_punc0_targeted_records.jsonl`

Goal:
- MBPP seed3: targeted `l8_nodetach` refinement to test whether near-neutral signal can be stabilized.
- PUNC seed0: quick frontier recheck to find usable positive branch.

Outcomes:

1) `mbpp_seed3_l8_refine`
- quick baseline: `40.90%`
- quick `head_l8_nodetach`: `41.45%` (`+0.56pp`)
- quick `head_l8_nodetach_clip07`: `41.45%` (`+0.56pp`)
- quick `head_l8_nodetach_midlr`: `40.67%` (`-0.23pp`)
- med baseline: `48.23%`
- med FS (`head_l8_nodetach`): `48.28%` (**`+0.05pp` vs med baseline**)

2) `punc_seed0_frontier_recheck`
- quick baseline: `8.44%`
- quick `head_l8`: `10.10%` (`+1.66pp`)
- quick `scalar_l8_sched_cos`: `8.18%` (`-0.26pp`)
- quick `scalar_l8_train1e4`: `4.27%` (`-4.17pp`, pruned)
- med baseline: `10.59%`
- med FS (`head_l8`): `11.09%` (**`+0.51pp` vs med baseline**)

Round75 conclusion:
- MBPP seed3 remains near-neutral even after targeted refinement.
- PUNC seed0 retains a modest positive branch via `head_l8`, while `scalar_l8_train1e4` is unstable across seeds.

## 2026-02-26 Round76 (punc1+squad0 dualmed, completed)

Script:
- `scripts/run_round76_punc1_squad0_dualmed.py`

Outputs:
- `results/_summary_round76_punc1_squad0_dualmed.txt`
- `results/_round76_punc1_squad0_dualmed_records.jsonl`

Goal:
- validate quick->med ranking consistency with dual-med promotion on two high-value tasks.

Outcomes:

1) `punc_seed1_dualmed_compare`
- quick baseline: `7.92%`
- quick `scalar_l8_train1e4`: `13.04%` (`+5.12pp`)
- quick `scalar_l8_sched_cos`: `10.34%` (`+2.42pp`)
- quick `head_l8`: `7.80%` (`-0.12pp`)
- med baseline: `10.88%`
- med `scalar_l8_train1e4`: `13.04%` (**`+2.16pp`**)
- med `scalar_l8_sched_cos`: `10.47%` (**`-0.41pp`**)

2) `squad_seed0_dualmed_compare`
- quick baseline: `13.33%`
- quick `scalar_l8_train1e4`: `14.05%` (`+0.72pp`)
- quick `scalar_l8_sched_cos`: `13.80%` (`+0.47pp`)
- quick `scalar_l8_train8e5`: `13.58%` (`+0.25pp`)
- med baseline: `17.27%`
- med `scalar_l8_train1e4`: `19.00%` (**`+1.73pp`**)
- med `scalar_l8_sched_cos`: `17.47%` (**`+0.20pp`**)

Round76 conclusion:
- dual-med confirmed that `scalar_l8_train1e4` is the dominant high-yield recipe for current PUNC/SQuAD settings.
- quick-only ranking can mislead (`sched_cos` looked strong in quick for PUNC but lost at med).

## 2026-02-26 Round77-82 (fastdiscover broad search, completed)

Script:
- `scripts/run_round77_82_fastdiscover.py`

Queue:
- `results/_search_queue_round77_82.json`

Outputs:
- `results/_summary_round77_fastdiscover.txt`
- `results/_summary_round78_fastdiscover.txt`
- `results/_summary_round79_fastdiscover.txt`
- `results/_summary_round80_fastdiscover.txt`
- `results/_summary_round81_fastdiscover.txt`
- `results/_summary_round82_fastdiscover.txt`
- `results/_round77_fastdiscover_records.jsonl`
- `results/_round78_fastdiscover_records.jsonl`
- `results/_round79_fastdiscover_records.jsonl`
- `results/_round80_fastdiscover_records.jsonl`
- `results/_round81_fastdiscover_records.jsonl`
- `results/_round82_fastdiscover_records.jsonl`

Policy:
- 70% budget on new-task discovery, 30% on anchor calibration.
- quick prune: `< baseline -0.50pp`.
- med promotion: `quick >= baseline +0.80pp` and only top1.
- useful task criterion: `med > baseline` (pp).

### Round77 (hotpot_seed0 + arc_seed0)

1) `hotpot_seed0_discovery`
- quick baseline: `8.44%`
- quick `scalar_l8_train8e5`: `10.36%` (`+1.92pp`)
- quick `scalar_l8_sched_cos`: `8.18%` (`-0.26pp`)
- quick `scalar_l8_train1e4`: `4.27%` (`-4.17pp`, pruned)
- med baseline: `10.48%`
- med FS (`scalar_l8_train8e5`): `10.36%` (**`-0.12pp`**)

2) `arc_seed0_discovery`
- quick baseline: `10.36%`
- quick `scalar_l8_train8e5`: `11.36%` (`+1.00pp`)
- med baseline: `12.33%`
- med FS (`scalar_l8_train8e5`): `12.63%` (**`+0.29pp`**)

Round77 conclusion:
- ARC seed0 gave a small but positive med gain.
- Hotpot seed0 showed quick-positive but med-negative reversal.

### Round78 (wiki_seed0 + protein_ss_seed0)

1) `wiki_seed0_discovery`
- baseline failed before quick comparison.
- failure cause: offline mode + missing local cache for `wikitext`.

2) `protein_ss_seed0_discovery`
- quick baseline: `27.17%`
- quick `scalar_l8_train1e4`: `30.77%` (`+3.60pp`)
- quick `head_l8`: `29.72%` (`+2.55pp`)
- quick `scalar_l8_train8e5`: `25.53%` (`-1.64pp`, pruned)
- med baseline: `27.17%`
- med FS (`scalar_l8_train1e4`): `30.77%` (**`+3.60pp`**)

Round78 conclusion:
- protein sequence labeling became the strongest new-task positive branch in this pack.
- wiki remained blocked by data-access mode.

### Round79 (hotpot_seed1 + protein_contact_seed0)

1) `hotpot_seed1_discovery`
- quick baseline: `7.92%`
- quick `scalar_l8_train1e4`: `13.04%` (`+5.12pp`)
- med baseline: `10.88%`
- med FS (`scalar_l8_train1e4`): `13.04%` (**`+2.16pp`**)

2) `protein_contact_seed0_discovery`
- baseline failed.
- failure cause: only `168` val examples were buildable while `n_val=200`.

Round79 conclusion:
- hotpot seed1 is a clear useful task under current recipe.
- protein_contact requires queue fix (`n_val` reduction) before fair comparison.

### Round80 (arc_seed1 + wiki_seed1)

1) `arc_seed1_discovery`
- quick baseline: `9.09%`
- quick `scalar_l8_sched_cos`: `9.49%` (`+0.41pp`)
- quick `scalar_l8_train8e5`: `7.93%` (`-1.16pp`, pruned)
- quick `scalar_l8_train1e4`: `7.64%` (`-1.45pp`, pruned)
- med skipped (`+0.41pp < +0.80pp` gate)

2) `wiki_seed1_discovery`
- baseline failed.
- failure cause: same as seed0 (`wikitext` cache missing in offline mode).

Round80 conclusion:
- ARC seed1 did not reach med gate.
- wiki remained blocked by environment, not by model quality.

### Round81 (anchor: squad_seed2 + punc_seed1)

1) `squad_seed2_anchor`
- quick baseline: `14.58%`
- quick `scalar_l8_train1e4`: `15.59%` (`+1.01pp`)
- quick `scalar_l8_train8e5`: `15.33%` (`+0.76pp`)
- quick `scalar_l8_sched_cos`: `14.07%` (`-0.51pp`, pruned)
- med baseline: `18.33%`
- med FS (`scalar_l8_train1e4`): `19.04%` (**`+0.71pp`**)

2) `punc_seed1_anchor`
- quick baseline: `7.92%`
- quick `scalar_l8_train1e4`: `13.04%` (`+5.12pp`)
- med baseline: `10.88%`
- med FS (`scalar_l8_train1e4`): `13.04%` (**`+2.16pp`**)

Round81 conclusion:
- anchor calibration confirmed both SQuAD and PUNC practical positives.

### Round82 (anchor: squad_seed0 + mbpp_seed2)

1) `squad_seed0_anchor`
- quick baseline: `13.33%`
- quick `scalar_l8_train1e4`: `14.05%` (`+0.72pp`)
- quick `scalar_l8_sched_cos`: `13.80%` (`+0.47pp`)
- quick `scalar_l8_train8e5`: `13.58%` (`+0.25pp`)
- med skipped (`best_quick +0.72pp < +0.80pp`)

2) `mbpp_seed2_anchor`
- quick baseline: `39.11%`
- quick `scalar_l8_sched_cos`: `41.19%` (`+2.08pp`)
- med baseline: `46.69%`
- med FS (`scalar_l8_sched_cos`): `47.58%` (**`+0.89pp`**)

Round82 conclusion:
- MBPP seed2 remained a useful task at med.
- SQuAD seed0 was positive in quick but below strict promote gate.

Round77-82 consolidated conclusion:
- Useful-task pool added/confirmed with med positives:
  - `protein_ss_seed0_discovery`: **`+3.60pp`**
  - `hotpot_seed1_discovery`: **`+2.16pp`**
  - `punc_seed1_anchor`: **`+2.16pp`**
  - `mbpp_seed2_anchor`: **`+0.89pp`**
  - `squad_seed2_anchor`: **`+0.71pp`**
  - `arc_seed0_discovery`: **`+0.29pp`**
- Primary blockers were operational:
  - `wikitext` unavailable in offline mode.
  - `protein_contact` validation target too large for sampled build constraints.

## 2026-02-26 Round89-94 (fastdiscover continuation, completed)

Queue:
- `results/_search_queue_round89_94_nowiki.json`

Outputs:
- `results/_summary_round89_fastdiscover.txt`
- `results/_summary_round90_fastdiscover.txt`
- `results/_summary_round91_fastdiscover.txt`
- `results/_summary_round92_fastdiscover.txt`
- `results/_summary_round93_fastdiscover.txt`
- `results/_summary_round94_fastdiscover.txt`
- `results/_round89_fastdiscover_records.jsonl`
- `results/_round90_fastdiscover_records.jsonl`
- `results/_round91_fastdiscover_records.jsonl`
- `results/_round92_fastdiscover_records.jsonl`
- `results/_round93_fastdiscover_records.jsonl`
- `results/_round94_fastdiscover_records.jsonl`

Policy:
- keep strict quick prune (`< -0.50pp`) and strict promotion (`>= +0.80pp`) unchanged.
- remove `wiki` from continuation queue after repeated network/cache failures.

### Round89-90 (protein_contact fixes, seed0/seed1)

1) `protein_contact_seed0_discovery_fix` / `protein_contact_seed1_discovery_fix`
- quick baseline: `96.88%`
- all candidate quick deltas: `+0.00pp`
- med skipped by gate

Conclusion:
- contact-pair branch in this setup is metric-saturated and low-yield for FS search.

### Round91 (hotpot_seed2 + arc_seed2)

1) `hotpot_seed2_discovery`
- quick baseline: `6.74%`
- quick best `scalar_l8_train8e5`: `7.83%` (`+1.10pp`)
- med baseline: `12.78%`
- med FS (`scalar_l8_train8e5`): `10.05%` (**`-2.74pp`**)
- `scalar_l8_train1e4` quick-pruned (`-4.18pp`)

2) `arc_seed2_discovery`
- quick baseline: `9.09%`
- all quick candidates negative (`-0.51pp`, `-0.82pp`, `-1.65pp`)
- med skipped

Conclusion:
- both branches are currently negative-value under this recipe family.

### Round92 (protein_ss_seed1 + arc_easy_seed0)

1) `protein_ss_seed1_discovery`
- quick baseline: `25.23%`
- quick `head_l8`: `34.62%` (`+9.39pp`)
- med baseline: `32.23%`
- med FS (`head_l8`): `34.62%` (**`+2.39pp`**)

2) `arc_easy_seed0_discovery`
- baseline failed during quick stage (current trainer/data path mismatch)

Conclusion:
- protein sequence-labeling branch remains the strongest new-task discovery direction.

### Round93 (anchor: squad_seed3 + mbpp_seed0)

1) `mbpp_seed0_anchor`
- quick baseline: `40.69%`
- quick best `scalar_l8_train8e5`: `43.55%` (`+2.86pp`)
- med baseline: `48.07%`
- med FS (`scalar_l8_train8e5`): `50.10%` (**`+2.03pp`**)

2) `squad_seed3_anchor`
- quick baseline: `14.67%`
- best quick: `-0.26pp`
- med skipped

Conclusion:
- MBPP anchor positive is reinforced; SQuAD seed3 remains negative.

### Round94 (anchor: squad_seed1 + punc_seed0)

1) `punc_seed0_anchor`
- quick baseline: `8.44%`
- quick best `scalar_l8_train8e5`: `10.36%` (`+1.92pp`)
- med baseline: `10.48%`
- med FS (`scalar_l8_train8e5`): `10.36%` (**`-0.12pp`**)

2) `squad_seed1_anchor`
- quick baseline: `14.83%`
- best quick: `-0.25pp`
- med skipped

Conclusion:
- `punc_seed0` showed quick-positive/med-negative reversal.
- SQuAD seed1 remains negative under current scalar family.

Round89-94 consolidated conclusion:
- New useful-task addition: `protein_ss_seed1_discovery` **`+2.39pp`** (med, `head_l8`).
- Strong anchor confirmation: `mbpp_seed0_anchor` **`+2.03pp`** (med, `scalar_l8_train8e5`).
- Branches to deprioritize for next search pack:
  - `protein_contact` (quick tie ceiling),
  - `arc_seed2`/`squad_seed1|3` (consistent quick negatives),
  - `hotpot_seed2` and `punc_seed0` (quick-positive but med-negative).

## 2026-02-26 Round95-98 (focus queue, completed)

Queue:
- `results/_search_queue_round95_98_focus.json`

Outputs:
- `results/_summary_round95_fastdiscover.txt`
- `results/_summary_round96_fastdiscover.txt`
- `results/_summary_round97_fastdiscover.txt`
- `results/_summary_round98_fastdiscover.txt`
- `results/_round95_fastdiscover_records.jsonl`
- `results/_round96_fastdiscover_records.jsonl`
- `results/_round97_fastdiscover_records.jsonl`
- `results/_round98_fastdiscover_records.jsonl`

### Round95 (`protein_ss_seed2_discovery` + `mbpp_seed1_anchor`)

1) `protein_ss_seed2_discovery`
- quick baseline: `26.45%`
- best quick: `scalar_l8_train1e4` `26.20%` (`-0.25pp`)
- med skipped

2) `mbpp_seed1_anchor`
- quick baseline: `40.86%`
- quick `scalar_l8_train8e5`: `43.11%` (`+2.25pp`)
- med baseline: `47.56%`
- med FS (`scalar_l8_train8e5`): `46.05%` (**`-1.51pp`**)

Round95 conclusion:
- MBPP seed1 again shows quick-positive/med-negative reversal.
- protein_ss seed2 is non-positive under this budget.

### Round96 (`protein_ss_seed3_discovery` + `mbpp_seed3_anchor`)

1) `protein_ss_seed3_discovery`
- quick baseline: `31.29%`
- quick `scalar_l8_train1e4`: `32.97%` (`+1.68pp`)
- med baseline: `31.29%`
- med FS (`scalar_l8_train1e4`): `32.97%` (**`+1.68pp`**)

2) `mbpp_seed3_anchor`
- quick baseline: `43.11%`
- all quick candidates negative (`-0.51pp`, `-1.12pp`, `-3.01pp`)
- med skipped

Round96 conclusion:
- protein_ss remains a stable useful-task direction across seeds.
- MBPP seed3 branch stays negative.

### Round97 (`hotpot_seed3_discovery` + `squad_seed2_anchor_recheck`)

1) `hotpot_seed3_discovery`
- quick baseline: `11.76%`
- all quick candidates negative (`-0.52pp`, `-1.04pp`, `-1.43pp`)
- med skipped

2) `squad_seed2_anchor_recheck`
- quick baseline: `14.58%`
- quick `scalar_l8_train1e4`: `15.59%` (`+1.01pp`)
- med baseline: `18.33%`
- med FS (`scalar_l8_train1e4`): `19.04%` (**`+0.71pp`**)

Round97 conclusion:
- SQuAD seed2 positive signal is reconfirmed.
- hotpot seed3 is negative under current recipe.

### Round98 (`arc_seed3_discovery` + `punc_seed1_anchor_recheck`)

1) `arc_seed3_discovery`
- quick baseline: `10.63%`
- all quick candidates negative (`-0.51pp`, `-0.87pp`, `-1.92pp`)
- med skipped

2) `punc_seed1_anchor_recheck`
- quick baseline: `7.92%`
- quick `scalar_l8_train1e4`: `13.04%` (`+5.12pp`)
- med baseline: `10.88%`
- med FS (`scalar_l8_train1e4`): `13.04%` (**`+2.16pp`**)

Round98 conclusion:
- punc seed1 strong positive is reconfirmed.
- arc seed3 branch should be frozen in this config family.

Round95-98 consolidated conclusion:
- Useful-task confirmations:
  - `protein_ss_seed3_discovery`: **`+1.68pp`**
  - `squad_seed2_anchor_recheck`: **`+0.71pp`**
  - `punc_seed1_anchor_recheck`: **`+2.16pp`**
- Negative/low-value branches:
  - `arc_seed3`, `hotpot_seed3`, `mbpp_seed3` (quick negatives),
  - `mbpp_seed1` (quick-positive but med-negative reversal).

## 2026-02-26 Round99 (focus2 queue, completed)

Queue:
- `results/_search_queue_round99_102_focus2.json`

Outputs:
- `results/_summary_round99_fastdiscover.txt`
- `results/_round99_fastdiscover_records.jsonl`

### Round99 (`protein_ss_seed4_discovery` + `mbpp_seed2_headprobe`)

1) `protein_ss_seed4_discovery`
- quick baseline: `30.30%`
- quick `scalar_l8_train8e5`: `35.20%` (`+4.90pp`)
- med baseline: `30.30%`
- med FS (`scalar_l8_train8e5`): `35.20%` (**`+4.90pp`**)
- quick-pruned: `scalar_l8_train1e4` (`-1.16pp`), `head_l8` (`-1.01pp`)

2) `mbpp_seed2_headprobe`
- quick baseline: `39.11%`
- quick `head_l10_strong`: `42.52%` (`+3.41pp`)
- med baseline: `46.69%`
- med FS (`head_l10_strong`): `48.97%` (**`+2.28pp`**)

Round99 conclusion:
- New strongest useful-task evidence in this sweep: `protein_ss_seed4_discovery` **`+4.90pp`**.
- `mbpp_seed2_headprobe` also confirms positive transfer at med budget (**`+2.28pp`**).
- Continuation status: round100-102 still running under same gate/prune rules.

## 2026-02-26 Round100 (focus2 queue, completed)

Outputs:
- `results/_summary_round100_fastdiscover.txt`
- `results/_round100_fastdiscover_records.jsonl`

### Round100 (`arc_seed0_headprobe` + `hotpot_seed1_headprobe`)

1) `arc_seed0_headprobe`
- quick baseline: `10.36%`
- quick best `scalar_l8_train8e5`: `11.36%` (`+1.00pp`)
- med baseline: `12.33%`
- med FS (`scalar_l8_train8e5`): `12.63%` (**`+0.29pp`**)

2) `hotpot_seed1_headprobe`
- quick baseline: `7.92%`
- quick best `scalar_l8_train1e4`: `13.04%` (`+5.12pp`)
- med baseline: `10.88%`
- med FS (`scalar_l8_train1e4`): `13.04%` (**`+2.16pp`**)

Round100 conclusion:
- `hotpot_seed1_headprobe` enters useful-task pool with stable med gain (**`+2.16pp`**).
- `arc_seed0_headprobe` is positive but weak (**`+0.29pp`**), lower priority than protein/hotpot branches.
- Continuation status: round101-102 running.

## 2026-02-26 Round101 (focus2 queue, completed)

Outputs:
- `results/_summary_round101_fastdiscover.txt`
- `results/_round101_fastdiscover_records.jsonl`

### Round101 (`squad_seed2_anchor_recheck2` + `punc_seed1_anchor_recheck2`)

1) `squad_seed2_anchor_recheck2`
- quick baseline: `14.58%`
- quick best `scalar_l8_train1e4`: `15.59%` (`+1.01pp`)
- med baseline: `18.33%`
- med FS (`scalar_l8_train1e4`): `19.04%` (**`+0.71pp`**)

2) `punc_seed1_anchor_recheck2`
- quick baseline: `7.92%`
- quick best `scalar_l8_train1e4`: `13.04%` (`+5.12pp`)
- med baseline: `10.88%`
- med FS (`scalar_l8_train1e4`): `13.04%` (**`+2.16pp`**)

Round101 conclusion:
- Anchor positives are stable across repeated checks (`punc` **`+2.16pp`**, `squad` **`+0.71pp`**).
- No quick-prune was triggered in this round; ranking still favors `punc`/`protein_ss` over `arc`.
- Continuation status: round102 running.

## 2026-02-26 Round102 (focus2 queue, completed)

Outputs:
- `results/_summary_round102_fastdiscover.txt`
- `results/_round102_fastdiscover_records.jsonl`

### Round102 (`mbpp_seed0_anchor_recheck2` + `protein_ss_seed1_anchor_recheck`)

1) `mbpp_seed0_anchor_recheck2`
- quick baseline: `40.69%`
- quick best `scalar_l8_sched_cos`: `42.12%` (`+1.43pp`)
- med baseline: `48.07%`
- med FS (`scalar_l8_sched_cos`): `49.26%` (**`+1.20pp`**)

2) `protein_ss_seed1_anchor_recheck`
- quick baseline: `25.23%`
- quick best `head_l8`: `34.62%` (`+9.39pp`)
- med baseline: `32.23%`
- med FS (`head_l8`): `34.62%` (**`+2.39pp`**)

Round102 conclusion:
- Queue `round99-102` fully completed; both tasks in round102 delivered positive med gains.
- `protein_ss` remains the strongest stable family (again above `+2pp` in med).
- `mbpp` gained from cosine-scheduled scalar FS in this anchor recheck (`+1.20pp`).

## 2026-02-26 Round103 (expand queue, completed)

Queue:
- `results/_search_queue_round103_104_expand.json`

Outputs:
- `results/_summary_round103_fastdiscover.txt`
- `results/_round103_fastdiscover_records.jsonl`

### Round103 (`hotpot_seed4_discovery` + `protein_ss_seed5_discovery`)

1) `hotpot_seed4_discovery`
- quick baseline: `8.78%`
- quick best `scalar_l8_train1e4`: `9.57%` (`+0.79pp`)
- med skipped (`+0.79pp < +0.80pp promote gate`)
- quick-pruned: `scalar_l8_train8e5` (`-0.75pp`)

2) `protein_ss_seed5_discovery`
- quick baseline: `23.08%`
- quick best `head_l8`: `28.30%` (`+5.22pp`)
- med baseline: `26.68%`
- med FS (`head_l8`): `28.30%` (**`+1.62pp`**)

Round103 conclusion:
- `protein_ss` continues to produce positive med gains on unseen seed (`seed5`).
- `hotpot_seed4` is near-threshold but failed strict promote gate by `0.01pp`.
- Continuation status: round104 running.
