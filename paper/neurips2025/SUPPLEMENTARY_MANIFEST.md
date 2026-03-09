# Supplementary Manifest

This file defines the intended anonymous supplementary snapshot for the NeurIPS submission.

## Purpose

The supplementary package is designed to support three things only:
- paper rebuild
- artifact inspection
- local sanity checks for the post-training runner

It is not a turnkey rerun package for the entire historical post-training campaign.

## Included Components

### Paper package
- `paper/neurips2025/main.tex`
- `paper/neurips2025/main_preprint.tex`
- `paper/neurips2025/appendix.tex`
- `paper/neurips2025/checklist.tex`
- `paper/neurips2025/references.bib`
- `paper/neurips2025/neurips_2025.sty`
- `paper/neurips2025/render_tables.py`
- `paper/neurips2025/verify_metrics_snapshot.py`
- `paper/neurips2025/verify_submission_layout.py`
- `paper/neurips2025/data/metrics.json`
- `paper/neurips2025/tables/`

### Paper-side documentation
- `paper/neurips2025/README.md`
- `paper/neurips2025/ARTIFACT_GUIDE.md`
- `paper/neurips2025/LOCAL_REPRO.md`
- `paper/neurips2025/TASK_MATRIX.md`
- `paper/neurips2025/METRICS_PROVENANCE.md`
- `paper/neurips2025/REFERENCE_AUDIT.md`
- `paper/neurips2025/SUPPLEMENTARY_MANIFEST.md`

### Method and synthetic-task code
- `rwkv-diff-future-seed/`

### Post-training orchestrators and queues
- `posttrain_rwkv7/scripts/`
- `posttrain_rwkv7/results/_search_queue_round783_790_realtask_exploit_v3.json`
- `posttrain_rwkv7/results/_search_queue_round799_802_final_confirm.json`
- `posttrain_rwkv7/results/_search_queue_round803_804_mbpp_altconfirm.json`
- `posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json`

### Shipped post-training artifacts
- `posttrain_rwkv7/runs/_summary_round783_fastdiscover.txt` through `_summary_round788_fastdiscover.txt`
- `posttrain_rwkv7/runs/_summary_round799_fastdiscover.txt` through `_summary_round808_fastdiscover.txt`
- `posttrain_rwkv7/runs/_round783_fastdiscover_records.jsonl` through `_round788_fastdiscover_records.jsonl`
- `posttrain_rwkv7/runs/_round799_fastdiscover_records.jsonl` through `_round808_fastdiscover_records.jsonl`

## Explicitly Not Included

- the referenced RWKV7 checkpoint blob
- a pinned end-to-end training environment lockfile
- the full historical post-training archive outside the shipped rounds
- any remote-machine state or AutoDL-era runtime environment
- any claim that all historical experiments can be rerun from this snapshot alone

## Required Verification Commands

### Paper-side verification
```bash
cd paper/neurips2025
python3 verify_metrics_snapshot.py
python3 verify_submission_layout.py
./build.sh submission
./build.sh preprint
```

### Repo-side verification
```bash
bash check_repo_health.sh
```

### Post-training runner sanity
```bash
cd posttrain_rwkv7
python3 scripts/run_round77_82_fastdiscover.py --self_test
python3 scripts/run_round77_82_fastdiscover.py \
  --queue results/_search_queue_round805_808_breadth_roi.json \
  --round_from 805 --round_to 808 --dry_run
```

## Interpretation Boundary

The paper tables are curated paper-side summaries generated from `paper/neurips2025/data/metrics.json`.

Those summaries are constrained by two automated checks:
- `verify_metrics_snapshot.py` checks them against the shipped README scoreboard and selected shipped round records
- `verify_submission_layout.py` checks that the built PDFs stay within the NeurIPS content-page budget and that bibliography / appendix / checklist boundaries remain well-formed

The real-task rows in the paper remain teacher-forced token-accuracy probes, not benchmark end metrics such as EM, F1, or pass@k.

## Canonical Reading Order

1. `paper/neurips2025/README.md`
2. `paper/neurips2025/main.tex`
3. `paper/neurips2025/ARTIFACT_GUIDE.md`
4. `paper/neurips2025/LOCAL_REPRO.md`
5. `paper/neurips2025/SUPPLEMENTARY_MANIFEST.md`
6. `paper/neurips2025/METRICS_PROVENANCE.md`
7. `paper/neurips2025/REFERENCE_AUDIT.md`
