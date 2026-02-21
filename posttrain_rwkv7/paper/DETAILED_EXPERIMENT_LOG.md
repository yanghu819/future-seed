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
