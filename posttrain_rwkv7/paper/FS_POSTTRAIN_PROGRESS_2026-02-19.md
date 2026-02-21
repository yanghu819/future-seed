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

## Update: 2026-02-21 (Round20 serial immediate-prune)

Protocol:

- one GPU, strictly serial execution
- quick-stage first, then immediate prune
- prune threshold: `quick d_acc < +0.001` (i.e., `< +0.10pp`)

Artifacts:

- `results/_summary_round20_serial_earlystop_s0.txt`
- `results/_round20_serial_earlystop_records.jsonl`

Round20 outcomes:

1. `hotpot`:
   - baseline quick acc: `6.17%`
   - all FS variants pruned (no positive quick delta)
2. `mbpp` (old setting):
   - baseline build failed due sample construction constraints
3. `protein_contact` (old setting):
   - baseline build failed due sample construction constraints
4. `protein_ss`:
   - baseline quick acc: `24.66%`
   - med-confirmed best:
     - `scalar_l10_norm_node`: `32.69%` (**+8.02pp**)
     - `scalar_l10_trainable`: `32.38%` (**+7.72pp**)
     - `scalar_l10_sched_cos`: `32.23%` (**+7.57pp**)

Interpretation:

- FS is not broadly effective across all tested real tasks.
- FS can be strongly effective on specific protein sequence-labeling setups.

## Update: 2026-02-21 (Round21 targeted follow-up, completed)

Goal:

- recover previously non-work tasks by fixing data-build bottlenecks
- keep serial + immediate-prune policy

Artifacts:

- `results/_summary_round21_targeted_search_s0.txt`
- `results/_round21_targeted_search_records.jsonl`

Round21 outcomes:

1. `mbpp_fix`:
   - quick baseline: `10.46%`
   - med-confirmed `scalar_l8_trainable`: `24.07%` (**+13.61pp**)
2. `protein_contact_fix`:
   - quick baseline: `98.83%`
   - all tested FS variants pruned (`+0.00pp`, no gain)
3. `protein_ss_refine`:
   - quick baseline: `21.14%`
   - med-confirmed best:
     - `scalar_l10_sched_cos`: `34.45%` (**+13.31pp**)
     - `scalar_l10_trainable`: `33.95%` (**+12.82pp**)
     - `head_l10`: `33.48%` (**+12.35pp**)
     - `scalar_l10_norm_node`: `32.57%` (**+11.43pp**)

Interpretation:

- fixing data-build constraints uncovered a clear positive MBPP regime.
- protein-contact remains saturated/no-gain in current formulation.
- protein SS gains become stronger under targeted search than in Round20.

## Update: 2026-02-21 (Round22 adaptive serial search, completed)

Artifacts:

- `results/_summary_round22_adaptive_search_s0.txt`
- `results/_round22_adaptive_search_records.jsonl`

Outcomes:

1. `mbpp_focus`:
   - quick baseline: `15.49%`
   - all tested FS variants regressed (`-0.92pp` to `-4.83pp`)
2. `protein_ss_expand`:
   - quick baseline: `28.15%`
   - best quick: `scalar_l10_nodetach` `+1.50pp`
   - mixed sign, weaker than Round21 targeted regime
3. `sudoku4_refine`:
   - quick baseline: `58.89%`
   - med best: `scalar_l6_trainable` `92.18%` (**+33.29pp**)
4. `sudoku9_probe`:
   - quick baseline: `1.35%`
   - med best: `scalar_l6_trainable` `2.86%` (**+1.51pp**)

## Update: 2026-02-21 (Round23 real-task sweep, partial)

Artifacts:

- `results/_round23_real_task_sweep_records.jsonl`
- `results/_launcher_round23.log`

Outcomes (completed tasks before abort):

1. `mbpp_rt`: all quick FS variants regressed (`-1.28pp` to `-4.67pp`).
2. `hotpot_rt`: FS quick variants tied baseline (`+0.00pp`).
3. `punc_restore_rt`: run aborted by HF connectivity (infrastructure/network issue).

## Update: 2026-02-21 (Round24/25 continuation, completed)

Artifacts:

- `results/_summary_round24_punc_protein_s0.txt`
- `results/_round24_punc_protein_records.jsonl`
- `results/_summary_round25_punc_salvage_s0.txt`
- `results/_round25_punc_salvage_records.jsonl`

Round24 outcomes:

1. `protein_ss_rt`:
   - quick baseline: `30.17%`
   - med `scalar_l10_sched_cos`: `34.31%` (**+4.14pp**)
2. `punc_restore_rt`:
   - baseline failed by OOM at `bsz=10` (configuration issue)

Round25 outcomes (memory-safe punc config):

1. baseline quick: `9.18%`
2. quick best:
   - `scalar_l8_sched_cos`: `+0.80pp`
   - `head_l8`: `+0.80pp`
3. med confirmed:
   - `head_l8`: `12.64%` (**+3.45pp**)
   - `scalar_l8_sched_cos`: `11.90%` (**+2.71pp**)

Interpretation:

- FS remains non-universal on MBPP/Hotpot under this recipe.
- FS remains consistently useful on protein SS.
- Punctuation/case restoration becomes a new positive real-text regime once memory config is corrected.

## Update: 2026-02-21 (Round26 low-throughput MBPP/Hotpot, completed)

Artifacts:

- `results/_summary_round26_mbpp_hotpot_lowthroughput_s0.txt`
- `results/_round26_mbpp_hotpot_lowthroughput_records.jsonl`

Round26 outcomes:

1. `mbpp_low` (`bsz=2`):
   - quick baseline: `10.46%`
   - quick `scalar_l8_trainable`: `+1.00pp`
   - med `scalar_l8_trainable`: `29.64%` (**+19.17pp**)
2. `hotpot_low` (`bsz=2`):
   - quick baseline: `14.34%`
   - `scalar_l10_trainable`: `+0.00pp`
   - `scalar_l10_sched_cos`: `+0.00pp`
   - `head_l10`: `-1.84pp`

Interpretation:

- MBPP FS gain is throughput-sensitive: low-throughput restored a strong positive regime.
- Hotpot remains no-gain in the matched low-throughput setup.

## Update: 2026-02-21 (Round27 seed robustness check, completed)

Artifacts:

- `results/_summary_round27_seedcheck_positive_s012.txt`
- `results/_round27_seedcheck_positive_s012_records.jsonl`

Round27 outcomes:

1. `mbpp_low + scalar_l8_trainable` (quick, seeds 0/1/2):
   - s0 `+1.00pp`, s1 `+0.32pp`, s2 `-0.82pp`
   - mean `+0.17pp`, positive seeds `2/3`
2. `punc_restore + head_l8` (quick, seeds 0/1/2):
   - s0 `+0.80pp`, s1 `+0.58pp`, s2 `+2.20pp`
   - mean `+1.19pp`, positive seeds `3/3`

Interpretation:

- Punctuation restoration is currently the most stable positive real-text FS regime.
- MBPP retains mean-positive signal but shows seed instability, so it should be treated as conditional rather than robust.

## Update: 2026-02-21 (Round28 MBPP throughput sweep, completed)

Artifacts:

- `results/_summary_round28_mbpp_bsz_sweep_s0.txt`
- `results/_round28_mbpp_bsz_sweep_s0_records.jsonl`

Round28 outcomes:

1. `bsz=2`:
   - baseline quick: `10.46%`
   - FS quick: `+1.00pp`
   - FS med: `25.39%` (**+14.92pp**)
2. `bsz=4`:
   - baseline quick: `11.71%`
   - FS quick: `9.66%` (**-2.05pp**)
3. `bsz=6`:
   - baseline quick: `14.50%`
   - FS quick: `12.42%` (**-2.08pp**)
4. `bsz=8`:
   - baseline failed (OOM)

Interpretation:

- MBPP FS-positive behavior is sharply throughput-dependent.
- For post-training search, MBPP should be split into two regimes:
  - low-throughput (`bsz=2`, FS-positive candidate)
  - high-throughput (`bsz>=4`, FS-negative under current recipe).
