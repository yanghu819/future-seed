# Getting Started

This repo has two independent workflows. Choose exactly one first.

## 1) Workflow choice

- choose `rwkv-diff-future-seed/` if your goal is method verification (toy + prefix-infill).
- choose `posttrain_rwkv7/` if your goal is real-task post-training search and ROI tracking.

## 2) Minimal prerequisites

- Python 3.10+
- PyTorch installed for your platform
- for CUDA experiments: a CUDA-capable environment and GPU drivers

## 3) Fast path commands

### Method sanity (toy)

```bash
bash run.sh
```

### QA variant

```bash
bash run_qa.sh
```

### Prefix infill on real datasets

```bash
python tools/build_hf_bins.py --dataset wikitext --config wikitext-2-raw-v1 \
  --train_split train --val_split validation --fields text --out_dir data/wikitext2_bytes

python tools/build_hf_bins.py --dataset mbpp \
  --train_split train --val_split test --fields code --out_dir data/mbpp_bytes

bash rwkv-diff-future-seed/run_wikitext_prefix.sh /abs/path/to/data/wikitext2_bytes
bash rwkv-diff-future-seed/run_mbpp_prefix.sh /abs/path/to/data/mbpp_bytes
```

### Post-training packet (ARC/protein/MBPP search)

```bash
cd posttrain_rwkv7
bash scripts/repro_doctor.sh
bash scripts/run_repropack_569_574.sh
bash scripts/sync_runs_to_results.sh --round-from 569 --round-to 574
python3 scripts/generate_fastdiscover_audit.py \
  --results_dir results --round_from 77 --round_to 569 \
  --out_prefix results/_audit_round77_569
```

## 4) If something is "missing"

Most confusion comes from directory split:

- prefix scripts are in `rwkv-diff-future-seed/`
- ARC/protein posttrain scripts are in `posttrain_rwkv7/scripts/`

Use:

```bash
rg --files | rg 'run_wikitext_prefix|run_mbpp_prefix|arc|protein'
```

## 5) Recommended reading order

1. `README.md`
2. `TASK_INDEX.md`
3. `RESULTS.md`
4. `posttrain_rwkv7/README.md`
