# Paper Hub

This repository is the unified code, artifact, and paper snapshot for Future-Seed.

## Primary Paper Entry

- anonymous paper package: [`paper/neurips2025/README.md`](paper/neurips2025/README.md)
- submission source: [`paper/neurips2025/main.tex`](paper/neurips2025/main.tex)
- appendix and checklist: [`paper/neurips2025/appendix.tex`](paper/neurips2025/appendix.tex), [`paper/neurips2025/checklist.tex`](paper/neurips2025/checklist.tex)
- artifact layout: [`paper/neurips2025/ARTIFACT_GUIDE.md`](paper/neurips2025/ARTIFACT_GUIDE.md)
- reference audit: [`paper/neurips2025/REFERENCE_AUDIT.md`](paper/neurips2025/REFERENCE_AUDIT.md)

## What The Paper Claims

- strong synthetic evidence for in-place constraint repair
- strongest repeatable real-task family under the current recipe: `protein_ss`
- supporting but smaller real-task positives: `hotpot`, `squad`, `punc`
- mixed evidence only: `mbpp_longctx`, `arc_mc`
- no stable held-out real-task confirmation claim

## Where The Numbers Come From

- unified result index: [`RESULTS.md`](RESULTS.md)
- post-training final status: [`posttrain_rwkv7/README.md`](posttrain_rwkv7/README.md)
- rolling round log: [`posttrain_rwkv7/results/_rolling_round_log.md`](posttrain_rwkv7/results/_rolling_round_log.md)
- full experiment ledger: [`posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md`](posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md)
- synced summaries and JSONL records: [`posttrain_rwkv7/runs/`](posttrain_rwkv7/runs)

## Minimal Build

```bash
cd paper/neurips2025
./build.sh submission
./build.sh preprint
```

## Minimal Repro Checks

```bash
cd posttrain_rwkv7
python3 scripts/run_round77_82_fastdiscover.py --self_test
python3 scripts/run_round77_82_fastdiscover.py \
  --queue results/_search_queue_round805_808_breadth_roi.json \
  --round_from 805 --round_to 808 --dry_run
```
