# Future-Seed Post-Training Progress (2026-02-19)

This note summarizes the current status of **Future-Seed (FS)** in post-training on real tasks.

## Setup

- Base model: `rwkv7-g1d-0.1b-20260129-ctx8192.pth`
- Training style: short SFT probes with fixed wall-clock budgets.
- FS usage: prompt/prefill only (`mode=prompt_fs`), causal decode unchanged.
- Stable FS recipe used in latest runs:
  - `alpha_init=-2`
  - `fs_layer_start=6`
  - `fs_norm`
  - `fs_detach`
  - `fs_clip=1.0`

## Main Results So Far

### ARC-Challenge (real MCQ)

- **Options-first (causal-unfriendly)**: FS helps.
  - File: `runs/_summary_arc_optionsfirst_stabilized_r2.txt`
  - Mean delta token-acc: **+0.0339** (3 seeds, all positive)
- **Question-first (causal-friendly control)**: FS does not help.
  - File: `runs/_summary_arc_qfirst_stabilized_r3.txt`
  - Mean delta token-acc: **-0.0052**

Interpretation: FS gains appear in orderings where useful evidence is placed in a causally awkward position.

### HotpotQA Long Context (real QA)

- Baseline stabilized run (`L=2048`, q-after):
  - File: `runs/_summary_hotpot_qafter_stabilized_r4_s012.txt`
  - Mean delta token-acc: **-0.0027** (high variance)
- Control (`L=2048`, q-first):
  - File: `runs/_summary_hotpot_qfirst_stabilized_r5_s012.txt`
  - Mean delta token-acc: **-0.0090**

Interpretation: at `L=2048`, FS is not stable on this probe.

### HotpotQA Longer Prompt (`L=4096`)

- q-after, `alpha=-2`:
  - File: `runs/_summary_hotpot_qafter_stabilized_len4096_r6_s012.txt`
  - Mean delta token-acc: **+0.0012**
  - Per-seed deltas: `+0.0382`, `+0.0179`, `-0.0525` (still high variance)
- q-first control, seed0:
  - File: `runs/_summary_hotpot_qfirst_stabilized_len4096_r7_s0.txt`
  - Delta token-acc: **+0.0025**, but loss worsens.

Interpretation: longer context can show larger gains on some seeds, but instability remains.

## Latest Update (Round 8)

- `L=4096`, q-after, stabilized FS with weaker seed injection (`alpha_init=-4`)
- File: `runs/_summary_hotpot_qafter_stabilized_len4096_r8_alpha_m4_s012.txt`
- Mean delta token-acc: **-0.0056**
- Per-seed deltas: `+0.0141`, `+0.0458`, `-0.0766`

Interpretation: simply weakening fixed seed injection (`alpha=-4`) did **not** fix variance; seed2 remained a strong negative outlier.

## Current In-Progress (Round 9)

- New method: **depth-scheduled FS injection** (still single-pass left->right per layer).
  - `fs_alpha_schedule=linear`
  - `fs_alpha_min=0.25`, `fs_alpha_max=1.0`
  - keeps stabilized stack: `fs_layer_start=6`, `fs_norm`, `fs_detach`, `fs_clip=1.0`
- Scripts:
  - `run_arc_optionsfirst_stabilized_round4_sched_linear.sh`
  - `run_hotpot_qafter_stabilized_len4096_round9_sched_linear_s012.sh`
- Goal: reduce seed variance without losing the causal-unfriendly gains.

### First Round-9 Result (ARC options-first, scheduled FS)

- File: `runs/_summary_arc_optionsfirst_stabilized_r4_sched_linear.txt`
- Config delta vs R2:
  - `fs_alpha_schedule=linear`
  - `fs_alpha_min=0.25`
  - `fs_alpha_max=1.0`
- Result:
  - mean delta token-acc: **+0.0156** (vs R2 `+0.0339`)
  - sign pattern: **2 positive / 1 negative**
  - mean delta loss: **+0.0295** (worse)

Interpretation: this linear depth schedule is not a free win; it reduced ARC gains and worsened loss despite modest acc lift.

### Round-9 Hotpot Status

- Script running: `run_hotpot_qafter_stabilized_len4096_round9_sched_linear_s012.sh`
- Triggered automatically after ARC via:
  - `run_after_arc_start_hotpot_r9.sh`
- Expected summary file:
  - `runs/_summary_hotpot_qafter_stabilized_len4096_r9_sched_linear_s012.txt`

### Round-9 Hotpot Result (completed)

- File: `runs/_summary_hotpot_qafter_stabilized_len4096_r9_sched_linear_s012.txt`
- Result:
  - mean delta token-acc: **-0.0220**
  - sign pattern: **1 positive / 2 negative**
  - mean delta loss: **-0.0188**

Interpretation: linear depth schedule did not stabilize Hotpot; it regressed average accuracy versus the no-schedule baseline.

## Current In-Progress (Round 10)

- New mitigation: push FS injection to deeper layers only.
  - `fs_layer_start=10` (from 6)
  - keep `alpha_init=-2`, `fs_norm`, `fs_detach`, `fs_clip=1.0`
- Script:
  - `run_hotpot_qafter_stabilized_len4096_round10_lstart10_s012.sh`
- Goal:
  - reduce negative outlier seeds while preserving the positive long-context seeds.

## Next Queued (Round 11)

- Script:
  - `run_hotpot_qafter_stabilized_len4096_round11_grid_lstart10_12_alpha_m2_m3_s012.sh`
- Auto-launch hook:
  - `run_after_r10_start_r11.sh`
- Grid:
  - `fs_layer_start in {10,12}`
  - `alpha_init in {-2,-3}`
- Purpose:
  - check whether deeper-only injection + weaker alpha gives a better mean/variance tradeoff than R10.

## Current Conclusion

1. FS is **useful in specific causal-unfriendly prompt orderings** (clear on ARC options-first).
2. FS is **not universally positive** in post-training; on Hotpot it remains high-variance.
3. The key open problem is **stability across seeds** on long-context QA.
4. The next lever is **schedule/structure** (not just smaller constant alpha).
