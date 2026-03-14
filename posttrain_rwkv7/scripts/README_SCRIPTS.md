# Scripts Guide

The `scripts/` folder contains both active and legacy runners.

## 1) Start from these scripts

- `run_round77_82_fastdiscover.py`
  - main orchestrator for queue-based quick->med search
- `run_sudoku9_inplace_maintrack.py`
  - fixed orchestrator for the canonical 9x9 in-place repair Sudoku benchmark
- `rebuild_fastloop_queues_broad.py`
  - queue generator for profile-based search cycles
- `plan_bestofn_fastloop.py`
  - profile selector / planner from historical records
- `generate_fastdiscover_audit.py`
  - regenerate full good/bad audit from JSONL records
- `run_repropack_569_574.sh`
  - one-command finite reproduction packet
- `repro_doctor.sh`
  - preflight check + orchestrator self-test
- `sync_runs_to_results.sh`
  - sync new records/summaries from `runs/` into tracked `results/`

## 2) Canonical Sudoku benchmark

The active Sudoku benchmark is the 9x9 solved-board in-place repair task:

- `sudoku9_inplace.py`
  - solved-board masker, manifest helpers, and repair metrics
- `build_sudoku9_inplace_manifests.py`
  - deterministic `val/test` manifest builder for `mask28/32/36/40`
- `train_sudoku9_inplace_refine.py`
  - canonical trainer with full-board masked repair loss and iterative refine eval
- `summarize_sudoku9_inplace.py`
  - aggregate runner summaries across `runs/`
- `run_sudoku9_inplace_maintrack.py`
  - fixed two-phase main-track launcher

The older `train_sudoku9_unique_sft.py` remains in the tree as a transfer / archive line.
`train_sudoku_sft.py` remains the older teacher-forced probe.

## 3) Sudoku-RWKV Baseline + Future-Seed Bridge

For the CoT-style `Sudoku-RWKV` baseline from the external repo, use:

- `sudoku_rwkv_official.py`
  - fetches the pinned upstream snapshot and checkpoint into the local cache root
- `sudoku_rwkv_future_seed.py`
  - dynamic wrapper around the upstream `RWKV-v6` inference model with optional Future-Seed attention-state injection
- `run_sudoku_rwkv_eval.py`
  - evaluates either the untouched official baseline or the Future-Seed-augmented variant on our Sudoku manifests

Smoke:

```bash
cd posttrain_rwkv7
python3 scripts/sudoku_rwkv_official.py --skip-checkpoint --self_test
python3 scripts/run_sudoku_rwkv_eval.py --self_test
```

Example baseline / FS probe:

```bash
cd posttrain_rwkv7
./.venv/bin/python scripts/run_sudoku_rwkv_eval.py \
  --manifest assets/sudoku9_unique/val_smoke.jsonl \
  --limit 1 \
  --max_tokens 50000 \
  --strategy "cuda fp16"

./.venv/bin/python scripts/run_sudoku_rwkv_eval.py \
  --manifest assets/sudoku9_unique/val_smoke.jsonl \
  --limit 1 \
  --max_tokens 50000 \
  --strategy "cuda fp16" \
  --future_seed \
  --fs_layer_start 2 \
  --seed_scale 0.5 \
  --fs_norm \
  --fs_clip 0.5
```

## 4) Recommended execution order

```bash
# preflight
bash scripts/repro_doctor.sh

# run finite packet
bash scripts/run_repropack_569_574.sh

# sync raw outputs into tracked snapshot
bash scripts/sync_runs_to_results.sh --round-from 569 --round-to 574

# rebuild audit
python3 scripts/generate_fastdiscover_audit.py \
  --results_dir results \
  --round_from 77 \
  --round_to 569 \
  --out_prefix results/_audit_round77_569
```

Sudoku smoke / dry-run:

```bash
cd posttrain_rwkv7
python3 scripts/build_sudoku9_inplace_manifests.py --out_dir assets/sudoku9_inplace --smoke
python3 scripts/train_sudoku9_inplace_refine.py --self_test
python3 scripts/run_sudoku9_inplace_maintrack.py --self_test --dry_run
```

## 5) Orchestrator quick reference

- self-test (no training):

```bash
./.venv/bin/python scripts/run_round77_82_fastdiscover.py --self_test
```

- dry-run queue (no training, emits dry-run records):

```bash
./.venv/bin/python scripts/run_round77_82_fastdiscover.py \
  --queue results/_search_queue_round569_576_fastloop.json \
  --round_from 569 \
  --round_to 574 \
  --policy results/_codex53_team_policy.json \
  --dry_run
```

## 6) Supporting scripts

- `supervise_gapless_457_640.sh`
  - watchdog style auto-restart launcher for long unattended runs
- `summarize_*.py`
  - task-specific summary scripts from older workflows

## 7) Legacy scripts

Files like `run_round2X_*`, `run_round3X_*`, ... `run_round7X_*` are historical launchers kept for traceability.
They are useful for forensics, but not required for new reproduction.

## 8) Notes for contributors

1. Read root `README.md` and `results/README_RESULTS.md` first.
2. Use finite packets before editing strategy files.
3. For Sudoku, prefer `sudoku9_inplace` over `sudoku9_unique` and the older `train_sudoku_sft.py` probe.
4. For CoT Sudoku baselines, prefer `run_sudoku_rwkv_eval.py` over ad-hoc copies of the external repo.
5. Sync `runs/` to `results/` before updating README or logs.
6. Only then modify queue strategy or policy files.
