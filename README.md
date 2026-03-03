# Future-Seed (RWKV Diffusion + Post-Training)

This repository now has **two tracks**:

- `rwkv-diff-future-seed/`: core Future-Seed method, toy stress tests, and prefix-infill benchmarks.
- `posttrain_rwkv7/`: real-task post-training search (ARC / protein / MBPP / others) with full round logs.

If you only remember one thing: choose one track first, then follow that track's README.

## Start Here (10 minutes)

### Path A: quick method sanity (toy tasks)

```bash
bash run.sh
```

Outputs:
- `rwkv-diff-future-seed/logs/rightcopy_base_big.log`
- `rwkv-diff-future-seed/logs/rightcopy_future_seed_big.log`
- `rwkv-diff-future-seed/logs/constr_base_big.log`
- `rwkv-diff-future-seed/logs/constr_future_seed_big.log`

### Path B: real-data prefix infill (WikiText / MBPP)

Build byte-level bins first:

```bash
python tools/build_hf_bins.py --dataset wikitext --config wikitext-2-raw-v1 \
  --train_split train --val_split validation --fields text --out_dir data/wikitext2_bytes

python tools/build_hf_bins.py --dataset mbpp \
  --train_split train --val_split test --fields code --out_dir data/mbpp_bytes
```

Run:

```bash
bash rwkv-diff-future-seed/run_wikitext_prefix.sh /abs/path/to/data/wikitext2_bytes
bash rwkv-diff-future-seed/run_mbpp_prefix.sh /abs/path/to/data/mbpp_bytes
```

Outputs:
- `rwkv-diff-future-seed/exp/wikitext2_prefix_*.log`
- `rwkv-diff-future-seed/exp/mbpp_prefix_*.log`

### Path C: post-training search (ARC/protein/MBPP)

```bash
cd posttrain_rwkv7
bash scripts/repro_doctor.sh
bash scripts/run_repropack_569_574.sh
bash scripts/sync_runs_to_results.sh --round-from 569 --round-to 574
```

Then read:
- `posttrain_rwkv7/README.md`
- `posttrain_rwkv7/results/README_RESULTS.md`
- `posttrain_rwkv7/scripts/README_SCRIPTS.md`

## Quick Navigation

- project onboarding: `GETTING_STARTED.md`
- task-to-script map: `TASK_INDEX.md`
- main method results: `RESULTS.md`
- paper/report assets: `paper/`
- implementation notes: `future-seed.md`

## Core Idea

Future-Seed passes previous layer final state `s_T` to next layer initial state `s_0`.

```text
baseline:    s=0              ; for t: s=f(s, x[t])
future-seed: s=prev_layer_s_T ; for t: s=f(s, x[t])
```

In non-causal diffusion-style denoising, this acts like cross-depth re-reading and strengthens future-context usage.

## Current Snapshot (high level)

From `RESULTS.md` and `posttrain_rwkv7/README.md`:

- toy stress tasks show clear gains on future-aware in-place repair settings.
- prefix-infill scripts exist for real text/code datasets:
  - `rwkv-diff-future-seed/run_wikitext_prefix.sh`
  - `rwkv-diff-future-seed/run_mbpp_prefix.sh`
- post-training track contains large-scale ARC/protein/MBPP search history and auditable ledgers.

## Where specific scripts live

- WikiText/MBPP prefix infill scripts: `rwkv-diff-future-seed/`
- ARC/protein/MBPP post-training scripts: `posttrain_rwkv7/scripts/`
- top-level helper entrypoints: `run.sh`, `run_qa.sh`

This split is intentional: core-method experiments and post-training campaigns are kept separate to keep each workflow reproducible.

## Practical Principles Used For This Repo Layout

Inspired by the "andrej-karpathy-skills" style:

- goal-first navigation: pick task goal, then one command path.
- simple defaults: finite packets and explicit outputs.
- verify before scale: doctor/self-test before long runs.
- keep artifacts auditable: command -> output file path is explicit.
