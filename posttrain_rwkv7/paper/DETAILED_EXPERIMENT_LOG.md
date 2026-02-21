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
