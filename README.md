# Future-Seed

This repository is a unified Future-Seed paper-and-artifact snapshot with two active experimental tracks:

- `rwkv-diff-future-seed/`: core Future-Seed method, toy stress tests, and prefix-infill experiments
- `posttrain_rwkv7/`: single-GPU post-training search on real tasks, with auditable round logs

If you are new here, choose one track first and follow that path end-to-end.

## Current Snapshot

The post-training campaign is closed through `round808`.

Current judgment:
- toy and synthetic constraint tasks clearly support Future-Seed
- strongest repeatable real-task family under the current recipe: `protein_ss`
- smaller positive real-task signals: `hotpot`, `mbpp_longctx`, `squad`, `punc`
- high-upside but high-variance: `arc_mc`
- low-ROI or negative under the current recipe: `protein_contact`, `wiki`, `hotpot_longctx`, `countdown`, `nqueens`, `zebra`, `sat3`

## Start Here

1. paper hub: [`PAPER.md`](PAPER.md)
2. global results: [`RESULTS.md`](RESULTS.md)
3. quick onboarding: [`GETTING_STARTED.md`](GETTING_STARTED.md)
4. task-to-script map: [`TASK_INDEX.md`](TASK_INDEX.md)
5. post-training final status: [`posttrain_rwkv7/README.md`](posttrain_rwkv7/README.md)
6. full experiment ledger: [`posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md`](posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md)
7. NeurIPS paper package: [`paper/neurips2025/README.md`](paper/neurips2025/README.md)

## Most Important Results

| Bucket | Task | Best gain |
|---|---|---:|
| real-task, repeatable | `protein_ss` | `+8.14pp` |
| real-task, small positive breadth signal | `hotpot` | `+4.20pp` |
| real-task, promising but not locked | `mbpp_longctx` | `+10.00pp` |
| real-task, high variance | `arc_mc` | `+20.83pp` |
| diagnostic constraint task | `graph_color` | `+8.33pp` |
| appendix-only spike | `tsp_mask` | `+25.00pp` |

## Quick Paths

### Path A: toy and method sanity

```bash
bash run.sh
bash run_qa.sh
```

### Path B: prefix infill on real data

```bash
python tools/build_hf_bins.py --dataset wikitext --config wikitext-2-raw-v1 \
  --train_split train --val_split validation --fields text --out_dir data/wikitext2_bytes
python tools/build_hf_bins.py --dataset mbpp \
  --train_split train --val_split test --fields code --out_dir data/mbpp_bytes
bash rwkv-diff-future-seed/run_wikitext_prefix.sh /abs/path/to/data/wikitext2_bytes
bash rwkv-diff-future-seed/run_mbpp_prefix.sh /abs/path/to/data/mbpp_bytes
```

### Path C: post-training runner sanity

```bash
cd posttrain_rwkv7
python3 scripts/run_round77_82_fastdiscover.py --self_test
python3 scripts/run_round77_82_fastdiscover.py \
  --queue results/_search_queue_round805_808_breadth_roi.json \
  --round_from 805 --round_to 808 --dry_run
```

## Repository Map

- [`PAPER.md`](PAPER.md): paper-first entry for the unified repo
- [`rwkv-diff-future-seed/`](rwkv-diff-future-seed): method code and toy experiments
- [`posttrain_rwkv7/`](posttrain_rwkv7): post-training experiments, logs, and queues
- [`RESULTS.md`](RESULTS.md): unified results page
- [`paper/neurips2025/README.md`](paper/neurips2025/README.md): anonymous NeurIPS 2025 submission package
- [`paper/neurips2025/REFERENCE_AUDIT.md`](paper/neurips2025/REFERENCE_AUDIT.md): source audit for all citations used in the paper
- [`posttrain_rwkv7/runs/`](posttrain_rwkv7/runs): synced latest summaries and raw JSONL records
