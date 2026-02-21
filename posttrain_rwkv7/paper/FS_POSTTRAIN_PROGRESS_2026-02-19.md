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

## Repro/Bookkeeping Update (2026-02-21)

- Canonical manifest added: `paper/exp_manifest.json`
- Aggregate parser added: `scripts/summarize_all_results.py`
- Doc/file consistency checker added: `scripts/check_doc_summary_refs.py`
- Auto-generated table:
  - `results/_aggregate_results.md`

## Main Results So Far

### ARC-Challenge (real MCQ)

- **Options-first (causal-unfriendly)**: FS helps.
  - File: `results/_summary_arc_optionsfirst_stabilized_r2.txt`
  - Mean delta token-acc: **+0.0339** (3 seeds, all positive)
- **Question-first (causal-friendly control)**: FS does not help.
  - File: `results/_summary_arc_qfirst_stabilized_r3.txt`
  - Mean delta token-acc: **-0.0052**

Interpretation: FS gains appear in orderings where useful evidence is placed in a causally awkward position.

### HotpotQA Long Context (real QA)

- Baseline stabilized run (`L=2048`, q-after):
  - File: `results/_summary_hotpot_qafter_stabilized_r4_s012.txt`
  - Mean delta token-acc: **-0.0027** (high variance)
- Control (`L=2048`, q-first):
  - File: `results/_summary_hotpot_qfirst_stabilized_r5_s012.txt`
  - Mean delta token-acc: **-0.0090**

Interpretation: at `L=2048`, FS is not stable on this probe.

### HotpotQA Longer Prompt (`L=4096`)

- q-after, `alpha=-2`:
  - File: `results/_summary_hotpot_qafter_stabilized_len4096_r6_s012.txt`
  - Mean delta token-acc: **+0.0012**
  - Per-seed deltas: `+0.0382`, `+0.0179`, `-0.0525` (still high variance)
- q-first control, seed0:
  - File: `results/_summary_hotpot_qfirst_stabilized_len4096_r7_s0.txt`
  - Delta token-acc: **+0.0025**, but loss worsens.

Interpretation: longer context can show larger gains on some seeds, but instability remains.

## Latest Update (Round 8)

- `L=4096`, q-after, stabilized FS with weaker seed injection (`alpha_init=-4`)
- File: `results/_summary_hotpot_qafter_stabilized_len4096_r8_alpha_m4_s012.txt`
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

- File: `results/_summary_arc_optionsfirst_stabilized_r4_sched_linear.txt`
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
  - `results/_summary_hotpot_qafter_stabilized_len4096_r9_sched_linear_s012.txt`

### Round-9 Hotpot Result (completed)

- File: `results/_summary_hotpot_qafter_stabilized_len4096_r9_sched_linear_s012.txt`
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

### Round-10 Result (completed)

- File: `results/_summary_hotpot_qafter_stabilized_len4096_r10_lstart10_s012.txt`
- Result:
  - mean delta token-acc: **-0.0225**
  - sign pattern: **2 positive / 1 negative**
  - mean delta loss: **-0.1906**

Interpretation: delaying FS start to deeper layers (`fs_layer_start=10`) did not recover mean accuracy; still worse than R6 baseline.

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

## New Queued Track: MBPP + More Sudoku

- MBPP real code task:
  - `run_mbpp_qafter_stabilized_len4096_round1_s012.sh`
  - `run_mbpp_qfirst_stabilized_len4096_round1_s012.sh`
  - with `fill_notes_to_max` long-context setting.
- Sudoku suite:
  - `run_sudoku_suite_round1_s012.sh`
  - settings:
    - `sudoku4_prefix`, `sudoku4_suffix`
    - `sudoku9_prefix`, `sudoku9_suffix`
- Queue hook:
  - `run_after_r11_start_mbpp_sudoku.sh`
  - runs automatically after R11 finishes.

## Current Conclusion

1. FS is **useful in specific causal-unfriendly prompt orderings** (clear on ARC options-first).
2. FS is **not universally positive** in post-training; on Hotpot it remains high-variance.
3. The key open problem is **stability across seeds** on long-context QA.
4. The next lever is **schedule/structure** (not just smaller constant alpha).

---

## Update: 2026-02-21 (R11 + MBPP + Sudoku + Protein)

### Hotpot R11 Grid (L=4096, q-after)

- `lstart=10, alpha=-2`:
  - File: `results/_summary_hotpot_qafter_stabilized_len4096_r11_lstart10_alpha-2_s012.txt`
  - Mean `d_acc = -0.0225` (2+/1-), mean `d_loss = -0.2831`
- `lstart=10, alpha=-3`:
  - File: `results/_summary_hotpot_qafter_stabilized_len4096_r11_lstart10_alpha-3_s012.txt`
  - Mean `d_acc = +0.0058` (2+/1-), mean `d_loss = +0.1495`
- `lstart=12, alpha=-2`:
  - File: `results/_summary_hotpot_qafter_stabilized_len4096_r11_lstart12_alpha-2_s012.txt`
  - Mean `d_acc = +0.0000` (3 zero deltas), near no-op
- `lstart=12, alpha=-3`:
  - File: `results/_summary_hotpot_qafter_stabilized_len4096_r11_lstart12_alpha-3_s012.txt`
  - Mean `d_acc = +0.0000` (3 zero deltas), exact no-op

Interpretation: deeper start at layer 12 collapses to no-op; layer 10 variants remain unstable.

### MBPP Long-Context (real code generation)

- q-after:
  - `results/_summary_mbpp_qafter_stabilized_len4096_r1_s012.txt`
  - Mean `d_acc = -0.0191` (0+/3-), mean `d_loss = +0.0293`
- q-first:
  - `results/_summary_mbpp_qfirst_stabilized_len4096_r1_s012.txt`
  - Mean `d_acc = -0.0254` (0+/3-), mean `d_loss = +0.1034`

Interpretation: FS regressed MBPP in both orderings.

### Sudoku Suite

- 4x4 prefix:
  - `results/_summary_sudoku4_prefix_r1_s012.txt`
  - Mean `d_acc = +0.0029` (3+/0-)
- 4x4 suffix:
  - `results/_summary_sudoku4_suffix_r1_s012.txt`
  - Mean `d_acc = +0.0003` (1+/1-/1=0)
- 9x9 prefix:
  - `results/_summary_sudoku9_prefix_r1_s012.txt`
  - Mean `d_acc = -0.0009` (1+/2-)
- 9x9 suffix:
  - `results/_summary_sudoku9_suffix_r1_s012.txt`
  - Mean `d_acc = -0.2861` (0+/3-), mean `d_loss = +1.5500`

Interpretation: FS helps weakly on easy 4x4 constraints; does not scale to harder 9x9 suffix setting.

### Protein Real-Task Probes

#### Secondary-structure spot labeling

- q-after:
  - `results/_summary_protein_ss_spot_qafter_len2048_r1_s012.txt`
  - Mean `d_acc = -0.0011` (2+/1-), `d_seq = 0`
- q-first:
  - `results/_summary_protein_ss_spot_qfirst_len2048_r1_s012.txt`
  - Mean `d_acc = -0.0001` (2+/1-), `d_seq = 0`

#### Contact-pair QA

- R1 (baseline settings):
  - `results/_summary_protein_contact_pair_qafter_len2048_r1_s012.txt`
  - Mean `d_acc = 0.0000` (all ties)
- R2 (trainable alpha, `alpha_lr=5e-4`, `seed_scale=0.5`, `lstart=8`):
  - `results/_summary_protein_contact_pair_qafter_len2048_r2_trainable_s012.txt`
  - Mean `d_acc = 0.0000` (all ties)
- R3 (balanced harder pairs):
  - `results/_summary_protein_contact_pair_qafter_len2048_r3_balanced_s012.txt`
  - Mean `d_acc = -0.0007` (2+/1-), `d_seq = 0`
- R4 (balanced + linear FS schedule):
  - `results/_summary_protein_contact_pair_qafter_len2048_r4_sched_s012.txt`
  - Mean `d_acc = -0.0043` (1+/2-), `d_seq = 0`

Interpretation: protein probes show near-zero to slight-negative mean effects for FS under current compute budget.

## Revised Bottom Line

1. **Supported**: FS can help in selected causal-unfriendly settings (ARC options-first, weak 4x4 Sudoku).
2. **Not supported**: broad post-training gains across MBPP, Hotpot, and protein tasks.
3. **Observed failure mode**: either high seed variance or effective no-op behavior when FS is pushed too deep.
4. **Practical guidance**: use FS as a targeted mechanism for constraint-style contexts, not as a universal post-training switch.

## Update: 2026-02-21 (Round12 5-seed stability check)

Round12 uses a higher-throughput training regime (`ARC bsz=48`, `Hotpot bsz=2`, fixed `max_steps`) to test seed stability under stronger hardware utilization.

### ARC (5 seeds, options-first / q-first)

- options-first:
  - `results/_summary_arc_optionsfirst_stabilized_r5_s01234.txt`
  - mean `d_acc = -0.0203` (2+/3-)
- q-first:
  - `results/_summary_arc_qfirst_stabilized_r5_s01234.txt`
  - mean `d_acc = +0.0005` (2+/3-)

Interpretation: the previous ARC positive signal is not stable under this optimization regime.

### Hotpot L=4096 (5 seeds, lstart=10 alpha=-3)

- q-after:
  - `results/_summary_hotpot_qafter_stabilized_len4096_r12_lstart10_alpha-3_s01234.txt`
  - mean `d_acc = +0.0054` (2+/2=/1-)
- q-first:
  - `results/_summary_hotpot_qfirst_stabilized_len4096_r12_lstart10_alpha-3_s01234.txt`
  - mean `d_acc = +0.0052` (2+/2=/1-)

Interpretation: gains are small and appear in both orderings; this weakens the claim that FS gain is tied to causal-unfriendly ordering in this setup.
