# Reproducibility Matrix

This matrix separates what is fully reproducible from the anonymous snapshot from what is only auditable or partially reproducible.

| Paper claim / artifact | Primary source files | Rebuild path in snapshot | Status |
|---|---|---|---|
| Submission and preprint PDFs | `main.tex`, `main_preprint.tex`, `appendix.tex`, `checklist.tex`, `references.bib`, `tables/*.tex` | `cd paper/neurips2025 && ./build.sh submission && ./build.sh preprint` | **Fully reproducible** |
| Main and appendix table rendering | `data/metrics.json`, `render_tables.py` | `cd paper/neurips2025 && python3 render_tables.py` | **Fully reproducible** |
| Metrics consistency between paper tables and the shipped snapshot boundary | `verify_metrics_snapshot.py`, `data/metrics.json`, shipped `runs/_summary_*.txt`, shipped `runs/_round*_records.jsonl` | `python3 paper/neurips2025/verify_metrics_snapshot.py` | **Fully reproducible** for historical-summary consistency plus release-backed closure/breadth recomputation |
| NeurIPS page-budget and appendix/checklist boundaries | built logs in `paper/neurips2025/build/` and `verify_submission_layout.py` | `python3 paper/neurips2025/verify_submission_layout.py` | **Fully reproducible** |
| Source anonymity and ZIP anonymity checks | `verify_anonymity_snapshot.py`, curated supplementary file set | `python3 paper/neurips2025/verify_anonymity_snapshot.py` and `python3 paper/neurips2025/verify_anonymity_snapshot.py --zip paper/neurips2025/dist/future-seed-neurips2025-supplementary.zip` | **Fully reproducible** |
| Anonymous supplementary ZIP assembly | `artifact_manifest.py`, `package_submission_bundle.py` | `python3 paper/neurips2025/package_submission_bundle.py` | **Fully reproducible** |
| Fastdiscover orchestrator sanity | `posttrain_rwkv7/scripts/run_round77_82_fastdiscover.py` plus the shipped queue files | `python3 posttrain_rwkv7/scripts/run_round77_82_fastdiscover.py --self_test` and `--dry_run` on the shipped breadth queue | **Fully reproducible** for parser/orchestrator sanity |
| Synthetic evidence as rendered in the paper | `rwkv-diff-future-seed/` plus paper-side `data/metrics.json` summary | inspect code/logs and rebuild the paper | **Code/log reviewable from the committed snapshot**; full rerun depends on local environment |
| Curated post-training main table | `data/metrics.json`, `TASK_MATRIX.md`, shipped queue files, shipped round summaries and records | inspect the committed historical summary, then inspect the shipped closure/breadth subset | **Partially auditable**: the paper table is rebuilt from a committed historical summary, while only the closure/breadth subset is directly recomputed from shipped raw records |
| End-to-end retraining of the reported post-training probe families | active trainer scripts, datasets, omitted checkpoint, omitted environment lockfile | not turnkey from this snapshot alone | **Not fully reproducible** |

## Practical Reading

- If the goal is to rebuild the paper, audit the committed historical summary, and recompute the released closure/breadth boundary subset, the snapshot is sufficient.
- If the goal is to rerun the entire post-training campaign from scratch, the snapshot is insufficient because the default checkpoint blob and a pinned end-to-end environment are not bundled.
- This is why the paper keeps a conservative distinction between `fully reproducible`, `partially auditable`, and `not fully reproducible`.
