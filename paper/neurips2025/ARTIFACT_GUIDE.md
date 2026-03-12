# Anonymous Artifact Guide

This guide describes the **curated anonymous supplementary boundary** for the paper-facing snapshot.

## What The Supplementary ZIP Contains

The ZIP built by `package_submission_bundle.py` contains four classes of material:

1. Paper package
- `paper/neurips2025/main.tex`
- `paper/neurips2025/main_preprint.tex`
- `paper/neurips2025/appendix.tex`
- `paper/neurips2025/checklist.tex`
- `paper/neurips2025/references.bib`
- `paper/neurips2025/neurips_2025.sty`
- `paper/neurips2025/render_tables.py`
- `paper/neurips2025/verify_metrics_snapshot.py`
- `paper/neurips2025/verify_submission_layout.py`
- `paper/neurips2025/verify_anonymity_snapshot.py`
- `paper/neurips2025/package_submission_bundle.py`
- `paper/neurips2025/artifact_manifest.py`
- `paper/neurips2025/data/metrics.json`
- `paper/neurips2025/tables/*.tex`

2. Paper-side documentation
- `paper/neurips2025/README.md`
- `paper/neurips2025/LOCAL_REPRO.md`
- `paper/neurips2025/TASK_MATRIX.md`
- `paper/neurips2025/METRICS_PROVENANCE.md`
- `paper/neurips2025/REFERENCE_AUDIT.md`
- `paper/neurips2025/SUPPLEMENTARY_MANIFEST.md`
- `paper/neurips2025/COMPUTE_ACCOUNTING.md`
- `paper/neurips2025/ASSET_LICENSE_MATRIX.md`
- `paper/neurips2025/REPRO_MATRIX.md`

3. Method and synthetic-task source snapshot
- `rwkv-diff-future-seed/`

4. Curated post-training snapshot
- `posttrain_rwkv7/scripts/run_round77_82_fastdiscover.py`
- `posttrain_rwkv7/scripts/cache_defaults.py`
- `posttrain_rwkv7/scripts/rwkv7_g1d.py`
- `posttrain_rwkv7/scripts/train_protein_ss_spot_sft.py`
- `posttrain_rwkv7/scripts/train_punc_restore_sft.py`
- `posttrain_rwkv7/scripts/train_mbpp_longctx_sft.py`
- `posttrain_rwkv7/scripts/train_arc_mc_sft.py`
- `posttrain_rwkv7/scripts/train_hotpot_longctx_sft.py`
- `posttrain_rwkv7/scripts/README_SCRIPTS.md`
- `posttrain_rwkv7/results/_codex53_team_policy.json`
- `posttrain_rwkv7/results/_search_queue_round783_790_realtask_exploit_v3.json`
- `posttrain_rwkv7/results/_search_queue_round799_802_final_confirm.json`
- `posttrain_rwkv7/results/_search_queue_round803_804_mbpp_altconfirm.json`
- `posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json`
- shipped summaries and JSONL records for rounds `783-788` and `799-808`

The ZIP also contains a generated `MANIFEST.json` at its root.

## What The Supplementary ZIP Does Not Contain

- the default RWKV7 checkpoint blob
- raw dataset caches
- a pinned end-to-end training environment lockfile
- the full historical post-training archive
- legacy AutoDL-era launchers or remote-machine state

## Why The Snapshot Is Curated

The repository contains older historical runners and archive notes that are valuable for provenance but not suitable for an anonymous paper package.
The supplementary ZIP therefore ships only the files that support:

- paper rebuild
- paper-side historical-summary consistency checks plus boundary audit of the shipped closure/breadth subset
- local sanity checks of the active fastdiscover orchestrator

It is not intended to be a turnkey rerun bundle for the entire project history.

## Minimal Verification Commands

Paper side:

```bash
cd paper/neurips2025
python3 verify_metrics_snapshot.py
python3 verify_submission_layout.py
./build.sh submission
./build.sh preprint
```

Supplementary packaging:

```bash
cd paper/neurips2025
python3 verify_anonymity_snapshot.py
python3 package_submission_bundle.py
python3 verify_anonymity_snapshot.py --zip dist/future-seed-neurips2025-supplementary.zip
```

Repository-level smoke test:

```bash
bash check_repo_health.sh
```

## Checkpoint Policy

The package script supports three modes:

- `omit`: current default; no checkpoint is bundled
- `bundle`: include a checkpoint file via `--checkpoint-path`
- `link`: record an anonymous external retrieval URL via `--checkpoint-url`

Because the current anonymous snapshot does not ship a checkpoint blob or anonymous checkpoint URL, the paper keeps conservative checklist answers for reproducibility and open access.
