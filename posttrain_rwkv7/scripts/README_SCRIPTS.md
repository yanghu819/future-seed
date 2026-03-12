# Scripts Guide

The `scripts/` folder contains both active and legacy runners.

## 1) Start from these scripts

- `run_round77_82_fastdiscover.py`
  - main orchestrator for queue-based quick->med search
- `run_sudoku9_unique_maintrack.py`
  - fixed orchestrator for the canonical 9x9 unique-solution Sudoku benchmark
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

The active Sudoku benchmark is the 9x9 unique-solution in-place task:

- `sudoku9_unique.py`
  - puzzle generator, uniqueness checker, and metric helpers
- `build_sudoku9_unique_manifests.py`
  - deterministic `val/test` manifest builder
- `train_sudoku9_unique_sft.py`
  - canonical trainer with clue-forced decode and exact/valid/clue metrics
- `summarize_sudoku9_unique.py`
  - aggregate runner summaries across `runs/`
- `run_sudoku9_unique_maintrack.py`
  - fixed two-phase main-track launcher

The older `train_sudoku_sft.py` remains in the tree as an archive probe. It is not the canonical 9x9 benchmark.

## 3) Recommended execution order

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
python3 scripts/build_sudoku9_unique_manifests.py --out_dir assets/sudoku9_unique --smoke
python3 scripts/train_sudoku9_unique_sft.py --self_test
python3 scripts/run_sudoku9_unique_maintrack.py --self_test --dry_run
```

## 4) Orchestrator quick reference

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

## 5) Supporting scripts

- `supervise_gapless_457_640.sh`
  - watchdog style auto-restart launcher for long unattended runs
- `summarize_*.py`
  - task-specific summary scripts from older workflows

## 6) Legacy scripts

Files like `run_round2X_*`, `run_round3X_*`, ... `run_round7X_*` are historical launchers kept for traceability.
They are useful for forensics, but not required for new reproduction.

## 7) Notes for contributors

1. Read root `README.md` and `results/README_RESULTS.md` first.
2. Use finite packets before editing strategy files.
3. For Sudoku, prefer `sudoku9_unique` over the older `train_sudoku_sft.py` probe.
4. Sync `runs/` to `results/` before updating README or logs.
5. Only then modify queue strategy or policy files.
