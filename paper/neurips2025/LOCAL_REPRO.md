# Local Reproduction Boundary

This note defines what a reader can and cannot reproduce from the current local repository snapshot.

## What Works Locally

1. Paper rebuild

```bash
cd paper/neurips2025
python3 verify_metrics_snapshot.py
python3 verify_submission_layout.py
./build.sh submission
./build.sh preprint
```

2. Repository health check

```bash
bash check_repo_health.sh
```

Useful fast paths:

```bash
bash check_repo_health.sh --skip-paper
bash check_repo_health.sh --skip-dry-run
```

3. Fastdiscover orchestration sanity

```bash
cd posttrain_rwkv7
export FUTURE_SEED_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/future-seed"
python3 scripts/run_round77_82_fastdiscover.py --self_test
python3 scripts/run_round77_82_fastdiscover.py \
  --queue results/_search_queue_round805_808_breadth_roi.json \
  --round_from 805 --round_to 808 --dry_run
```

## What The Snapshot Contains

- paper sources and bibliography
- curated paper-side metrics in `data/metrics.json`
- a paper-side verification script in `verify_metrics_snapshot.py`
- a paper-side layout/page-budget checker in `verify_submission_layout.py`
- queue files for the final exploit / confirm / breadth windows
- local snapshot summaries and JSONL records for rounds `783-788` and `799-808`
- active trainers and orchestrators used by the paper-facing post-training audit

## What The Snapshot Does Not Contain

- the referenced RWKV7 checkpoint blob
- a pinned full training environment lockfile
- a turnkey end-to-end rerun of the whole post-training search campaign
- every historical dataset cache used during the original search

## How To Interpret Dry-Run Failures

`--dry_run` validates queue parsing, dedup logic, path resolution, and summary generation without mutating tracked artifacts.

If a task shows `quick baseline failed` during `--dry_run`, the usual cause is missing local data or cache state, not a regression in the orchestrator itself.

## Cache And Environment Defaults

Active local scripts now support a shared cache root:

```bash
export FUTURE_SEED_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/future-seed"
```

The following variables still take precedence if set explicitly:

- `HF_HOME`
- `HF_DATASETS_CACHE`
- `TRANSFORMERS_CACHE`
- `TORCH_EXTENSIONS_DIR`

## Paper Table Boundary

The paper tables are regenerated from committed `data/metrics.json`, not from a full raw-archive replay inside this snapshot.

That means:

- the PDF build is deterministic inside the repo
- the tables are auditable against the committed snapshot
- the shipped closure-window highlights are directly checked against committed round records by `verify_metrics_snapshot.py`
- the built PDFs are checked against the NeurIPS content-page budget by `verify_submission_layout.py`
- the build is not a guarantee that the entire historical search can be rerun from scratch

For row semantics and provenance, see:

- [`TASK_MATRIX.md`](TASK_MATRIX.md)
- [`METRICS_PROVENANCE.md`](METRICS_PROVENANCE.md)
- [`ARTIFACT_GUIDE.md`](ARTIFACT_GUIDE.md)
- [`SUPPLEMENTARY_MANIFEST.md`](SUPPLEMENTARY_MANIFEST.md)
