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
- Summary: `runs/_summary_arc_optionsfirst_stabilized_r2.txt`
- Result:
  - mean `d_acc = +0.0339`, std `0.0205`
  - seed deltas: `+0.0156`, `+0.0234`, `+0.0625`
- Status: **success / stable positive**

### R3: Question-first (causal-friendly control)

- Script: `run_arc_qfirst_stabilized_round3.sh`
- Summary: `runs/_summary_arc_qfirst_stabilized_r3.txt`
- Result:
  - mean `d_acc = -0.0052`, std `0.0097`
  - seed deltas: `-0.0156`, `+0.0078`, `-0.0078`
- Status: **control matched expectation** (no universal gain)

## HotpotQA Long Context

### R4: q-after, L=2048

- Script: `run_hotpot_qafter_stabilized_round4_s0.sh` + `_s12.sh`
- Summary: `runs/_summary_hotpot_qafter_stabilized_r4_s012.txt`
- Result:
  - mean `d_acc = -0.0027`, std `0.0171`
  - seed deltas: `+0.0084`, `+0.0104`, `-0.0269`
- Status: **mixed / unstable**

### R5: q-first, L=2048 (control)

- Script: `run_hotpot_qfirst_stabilized_round5_s012.sh`
- Summary: `runs/_summary_hotpot_qfirst_stabilized_r5_s012.txt`
- Result:
  - mean `d_acc = -0.0090`, std `0.0085`
  - seed deltas: `-0.0104`, `+0.0021`, `-0.0186`
- Status: **negative control (no gain)**

### R6: q-after, L=4096, alpha=-2

- Script: `run_hotpot_qafter_stabilized_len4096_round6_s0.sh` + `_s12.sh`
- Summary: `runs/_summary_hotpot_qafter_stabilized_len4096_r6_s012.txt`
- Result:
  - mean `d_acc = +0.0012`, std `0.0388`
  - seed deltas: `+0.0382`, `+0.0179`, `-0.0525`
- Status: **slight mean gain but high variance**

### R7: q-first, L=4096, seed0

- Script: `run_hotpot_qfirst_stabilized_len4096_round7_s0.sh`
- Summary: `runs/_summary_hotpot_qfirst_stabilized_len4096_r7_s0.txt`
- Result:
  - `d_acc = +0.0025` (single seed)
  - loss worsened in that run
- Status: **inconclusive (single-seed control)**

### R8: q-after, L=4096, alpha=-4

- Script: `run_hotpot_qafter_stabilized_len4096_round8_alpha_m4_s012.sh`
- Summary: `runs/_summary_hotpot_qafter_stabilized_len4096_r8_alpha_m4_s012.txt`
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
- Summary: `runs/_summary_arc_optionsfirst_stabilized_r4_sched_linear.txt`
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
  - `runs/_summary_hotpot_qafter_stabilized_len4096_r9_sched_linear_s012.txt`

### R9b Result: Hotpot q-after L=4096 (completed)

- Summary: `runs/_summary_hotpot_qafter_stabilized_len4096_r9_sched_linear_s012.txt`
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

## Current Bottom Line

- FS for post-training is **validated in specific real settings** (ARC options-first).
- FS on real long-context QA is **not yet stable enough** for a universal claim.
- Next milestone is not higher single-run peaks; it is **cross-seed stability** under fixed budget.
