# Future-Seed Post-Training Progress (2026-02-19)

This note summarizes the current status of **Future-Seed (FS)** in post-training on real tasks.

For a full chronological record (including failed runs and diagnostics), see:
- `DETAILED_EXPERIMENT_LOG.md`

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

## Ongoing Run

- `L=4096`, q-after, stabilized FS with weaker seed injection (`alpha_init=-4`)
- Script: `run_hotpot_qafter_stabilized_len4096_round8_alpha_m4_s012.sh`
- Log: `runs/run_hotpot_qafter_stabilized_len4096_round8_alpha_m4_s012.log`
- Goal: reduce negative outlier seeds while keeping positive long-context gains.

## Current Conclusion

1. FS is **useful in specific causal-unfriendly prompt orderings** (clear on ARC options-first).
2. FS is **not universally positive** in post-training; on Hotpot it remains high-variance.
3. The key open problem is **stability across seeds** on long-context QA.
