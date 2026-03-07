# Getting Started

Choose exactly one workflow first.

## 0. Whole-repo health check

```bash
bash check_repo_health.sh
```

This runs link validation, the fastdiscover self-test, a non-mutating dry-run, and a submission paper build.

## 1. Workflow choice

- choose `rwkv-diff-future-seed/` if your goal is method verification on toy and prefix-infill settings
- choose `posttrain_rwkv7/` if your goal is the real-task post-training campaign

## 2. Read in this order

1. [`README.md`](README.md)
2. [`RESULTS.md`](RESULTS.md)
3. [`TASK_INDEX.md`](TASK_INDEX.md)
4. [`posttrain_rwkv7/README.md`](posttrain_rwkv7/README.md)
5. [`posttrain_rwkv7/ARCHIVE_ROUNDS.md`](posttrain_rwkv7/ARCHIVE_ROUNDS.md)
6. [`posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md`](posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md)

## 3. Fast sanity commands

### Method sanity

```bash
bash run.sh
bash run_qa.sh
```

### Prefix infill

```bash
python tools/build_hf_bins.py --dataset wikitext --config wikitext-2-raw-v1 \
  --train_split train --val_split validation --fields text --out_dir data/wikitext2_bytes
python tools/build_hf_bins.py --dataset mbpp \
  --train_split train --val_split test --fields code --out_dir data/mbpp_bytes
bash rwkv-diff-future-seed/run_wikitext_prefix.sh /abs/path/to/data/wikitext2_bytes
bash rwkv-diff-future-seed/run_mbpp_prefix.sh /abs/path/to/data/mbpp_bytes
```

### Post-training runner sanity

```bash
cd posttrain_rwkv7
python3 scripts/run_round77_82_fastdiscover.py --self_test
python3 scripts/run_round77_82_fastdiscover.py \
  --queue results/_search_queue_round805_808_breadth_roi.json \
  --round_from 805 --round_to 808 --dry_run
```

Optional local cache override:

```bash
export FUTURE_SEED_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/future-seed"
```

Notes:
- explicit `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, and `TORCH_EXTENSIONS_DIR` still override the shared root
- `--dry_run` validates orchestration and paths; if local datasets are absent, task baselines may still fail without changing tracked artifacts

## 4. Latest finished artifacts to inspect

```bash
cd posttrain_rwkv7
ls runs/_summary_round78{3,4,5,6,7,8}_fastdiscover.txt
sed -n '1,220p' runs/_summary_round808_fastdiscover.txt
```

## 5. Good finite queues to reuse

- [`posttrain_rwkv7/results/_search_queue_round783_790_realtask_exploit_v3.json`](posttrain_rwkv7/results/_search_queue_round783_790_realtask_exploit_v3.json)
- [`posttrain_rwkv7/results/_search_queue_round799_802_final_confirm.json`](posttrain_rwkv7/results/_search_queue_round799_802_final_confirm.json)
- [`posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json`](posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json)

## 6. What not to rerun under the same recipe

- `countdown`
- `nqueens`
- `zebra`
- `sat3`
- `wiki`
- repeated `mbpp_longctx_probe` confirmation with the same `head_l8 / scalar_l8_*` family after `round803-804`
