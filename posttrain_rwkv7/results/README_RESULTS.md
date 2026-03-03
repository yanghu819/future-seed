# Results Directory Guide

This folder stores curated experiment snapshots for reproducibility and paper writing.

## 1) Data flow (source of truth)

- run-time outputs are produced in `runs/`
- after each round block, copy into this folder:

```bash
bash scripts/sync_runs_to_results.sh --round-from 569 --round-to 574
```

- this `results/` folder is the tracked snapshot used by README, audit, and Git history

## 2) Core file types

- `_summary_roundXXX_fastdiscover.txt`
  - human-readable per-round summary
  - includes baseline vs candidate deltas in `pp`
- `_roundXXX_fastdiscover_records.jsonl`
  - machine-readable event log
  - full events across `quick`, `med`, pruning, and skips
- `_search_queue_roundA_B_fastloop.json`
  - queue definition consumed by orchestrator
  - task list, budgets, candidate configs, metadata

## 3) Policy and planning artifacts

- `_codex53_team_policy.json`
  - gate policy used by `run_round77_82_fastdiscover.py`
- `_bestofn_plan_round561_640.json`
  - profile selection artifact from planner

## 4) Audit artifacts (recommended entry point)

- `_audit_round77_569_analysis.md`
  - full good/bad analysis summary
- `_audit_round77_569_med_ledger.csv`
  - all med rows (positive/flat/negative)
- `_audit_round77_569_quick_ledger.csv`
  - all quick rows
- `_audit_round77_569_task_combo.csv`
  - joined quick/med rows for reversal analysis

## 5) Closure and chronology

- `_summary_round545_568_sprint_closure.txt`
  - sprint-window closure report
- `_rolling_round_log.md`
  - chronological running log across many rounds

## 6) Minimal verification checklist

For a newly finished round `N`, check all items:

- `_roundN_fastdiscover_records.jsonl` exists
- `_summary_roundN_fastdiscover.txt` exists
- both files are synced from `runs/` to `results/`
- if the round belongs to an audit window, audit regeneration succeeds

## 7) Common commands

Regenerate full audit tables from raw records:

```bash
python3 scripts/generate_fastdiscover_audit.py \
  --results_dir results \
  --round_from 77 \
  --round_to 569 \
  --out_prefix results/_audit_round77_569
```

Run finite repro packet:

```bash
bash scripts/run_repropack_569_574.sh
```

List latest summary files:

```bash
ls -1 results/_summary_round*_fastdiscover.txt | sort -V | tail -n 10
```

Count synced round records:

```bash
find results -maxdepth 1 -type f -name '_round*_fastdiscover_records.jsonl' | wc -l
```
