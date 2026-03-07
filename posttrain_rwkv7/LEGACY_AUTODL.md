# Legacy AutoDL Notes

This repository snapshot is no longer organized around the historical AutoDL workflow.

Current supported path:
- paper-first reading through `README.md`, `PAPER.md`, and `paper/neurips2025/`
- local artifact inspection through `posttrain_rwkv7/runs/` and `posttrain_rwkv7/results/`
- local sanity checks through `python3 scripts/run_round77_82_fastdiscover.py --self_test` and `--dry_run`

What remains in the tree for provenance only:
- older launch scripts that hard-code remote paths
- historical round chronology in `README.md`
- internal operator logs in `results/_rolling_round_log.md`
- the full audit ledger in `paper/DETAILED_EXPERIMENT_LOG.md`

Those files are kept to preserve the search history and paper traceability. They are not the supported workflow for a new reader.
