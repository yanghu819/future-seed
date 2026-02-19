# Detailed Experiment Log: RWKV7 Future-Seed Post-Training

Last updated: **2026-02-19**

This document records all major real-task post-training experiments, including positive results, failures, and interpretation.

## 1. Objective

Goal: test whether **Future-Seed (FS)** improves post-training (SFT-style) on real tasks, not only synthetic right-copy style tasks.

Primary hypothesis:
- FS should help more when prompt order is **causally unfriendly** (critical evidence appears in positions that are hard for strict left-to-right compression).

## 2. Common Setup

- Model: `rwkv7-g1d-0.1b-20260129-ctx8192.pth`
- FS usage: enabled on prompt/prefill (`mode=prompt_fs`), decode remains causal.
- Metrics:
  - `val_tok_acc` (token accuracy on supervised answer region)
  - `val_loss`
- Compare `no_fs` vs `prompt_fs`; report delta as `FS - no_fs`.

Stable FS recipe used in later rounds:
- `alpha_init=-2`
- `alpha_lr=0` (fixed gate in most runs)
- `fs_layer_start=6`
- `fs_norm`
- `fs_detach`
- `fs_clip=1.0`

## 3. ARC-Challenge (Multiple-Choice, Real Task)

### 3.1 Input-adaptive gate attempt (mixed / weak)
- Summary: `results/_summary_arc_optionsfirst_inputgate_r1.txt`
- Setting: options-first, `fs_in_gate` on, `fs_w_lr=fs_b_lr=1e-3`
- Result:
  - mean `Δacc = +0.0052` (std `0.0184`)
  - mean `Δloss = +0.0139`
  - 2 seeds negative, 1 seed positive
- Verdict: weak and unstable.

### 3.2 Stabilized FS without input gate (clear positive)
- Summary: `results/_summary_arc_optionsfirst_stabilized_r2.txt`
- Setting: options-first, stable FS recipe, no `fs_in_gate`
- Result:
  - mean `Δacc = +0.0339` (std `0.0205`)
  - mean `Δloss = -0.0182`
  - 3/3 seeds positive
- Verdict: **successful**.

### 3.3 Question-first control (near-zero / slight negative)
- Summary: `results/_summary_arc_qfirst_stabilized_r3.txt`
- Same config as 3.2 but `q_first=True`
- Result:
  - mean `Δacc = -0.0052`
  - mean `Δloss = +0.0003`
- Verdict: FS gain is tied to order difficulty, not universal.

## 4. HotpotQA (Long-Context QA, Real Task)

### 4.1 Early baseline-like runs (negative)
- Summaries:
  - `results/_summary_hotpot_qafter.txt`
  - `results/_summary_hotpot_qfirst.txt`
- Result:
  - q-after mean `Δacc = -0.0196`
  - q-first mean `Δacc = -0.0175`
- Verdict: FS harmed average performance under early setup.

### 4.2 Stabilized recipe at L=2048 (still unstable)
- Summaries:
  - `results/_summary_hotpot_qafter_stabilized_r4_s012.txt`
  - `results/_summary_hotpot_qfirst_stabilized_r5_s012.txt`
- Result:
  - q-after mean `Δacc = -0.0027` (close to zero, high variance)
  - q-first mean `Δacc = -0.0090`
- Verdict: stabilization helps compared to early negative runs, but no robust positive gain at 2048.

### 4.3 Longer prompt L=4096 with stable FS (`alpha=-2`)
- Summaries:
  - `results/_summary_hotpot_qafter_stabilized_len4096_r6_s012.txt`
  - `results/_summary_hotpot_qfirst_stabilized_len4096_r7_s0.txt`
- Result:
  - q-after: mean `Δacc = +0.0012` (2 positive seeds, 1 strong negative seed)
  - q-first seed0: `Δacc = +0.0025`, but `Δloss` worsened (+0.0873)
- Verdict: some long-context upside exists, but seed variance remains a central issue.

## 5. Failure Cases and Diagnostics

## 5.1 Permission/script robustness issues
- Some runs ended without summary due to executable-bit mismatch (e.g., `summarize_*.py` permission denied).
- Fix: call summarizers via `./.venv/bin/python summarize_*.py` inside run scripts.

## 5.2 Data sparsity at long length
- At `L=4096` with `min_prompt_tokens=2048`, Hotpot could not build enough validation examples.
- Observed error: *Only built 213 examples (wanted 512)*.
- Fix: use `n_train=1000`, `n_val=200`, `min_prompt_tokens=1536` for L=4096 sweeps.

## 5.3 High variance across seeds (main unresolved blocker)
- Hotpot `q_after`, L=4096, `alpha=-2`:
  - seed0 +0.0382
  - seed1 +0.0179
  - seed2 -0.0525
- This single negative outlier dominates the mean.

## 6. Current Ongoing Experiment

- Script: `scripts/run_hotpot_qafter_stabilized_len4096_round8_alpha_m4_s012.sh`
- Change vs previous best-known setup: only `alpha_init` from `-2` to `-4` (weaker FS injection).
- Purpose: reduce harmful outlier seeds while preserving gains in hard-order long-context setting.
- Runtime status should be checked via:
  - AutoDL log: `runs/run_hotpot_qafter_stabilized_len4096_round8_alpha_m4_s012.log`
  - Expected summary: `runs/_summary_hotpot_qafter_stabilized_len4096_r8_alpha_m4_s012.txt`

## 7. Practical Takeaways So Far

1. FS can be useful for post-training on real tasks, but **conditional on ordering**:
   - strong on ARC options-first
   - weak/negative on easy-order controls
2. FS is not yet a universally safe drop-in for long-context QA.
3. Main research problem now is **stability across seeds** rather than “is there any gain”.

## 8. Repro Pointers

Core scripts:
- `scripts/train_arc_mc_sft.py`
- `scripts/train_hotpot_longctx_sft.py`
- `scripts/rwkv7_g1d.py`

Main run scripts:
- `scripts/run_arc_stabilized_round2.sh`
- `scripts/run_arc_qfirst_stabilized_round3.sh`
- `scripts/run_hotpot_qafter_stabilized_round4_s12.sh`
- `scripts/run_hotpot_qfirst_stabilized_round5_s012.sh`
- `scripts/run_hotpot_qafter_stabilized_len4096_round6_s12.sh`
- `scripts/run_hotpot_qafter_stabilized_len4096_round8_alpha_m4_s012.sh`

