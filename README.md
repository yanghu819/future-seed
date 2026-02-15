# Future‑Seed — Diffusion RWKV

A minimal, reproducible prototype that shows Future‑Seed (vertical state non‑causal initialization) can dramatically improve future‑aware tasks in a diffusion‑style RWKV7 model.

## Why it matters
Future‑Seed passes the previous layer’s final state `s_T` as the next layer’s initial state. In diffusion LM (non‑causal), this behaves like “re‑reading” the sequence across depth, strengthening future context instead of one‑pass online updates.

## Results
See `RESULTS.md` and `paper/future-seed-report.pdf`.
English report: `paper/future-seed-report-en.pdf`.

Quick highlights (single GPU 4090, RWKV7 `cuda_wind`):
| task | FUTURE_SEED=0 | FUTURE_SEED=1 |
|---|---:|---:|
| rightcopy acc (3 seeds, mean) | 0.1046 | 0.1550 |
| constr acc (3 seeds, mean) | 0.0977 | 0.2114 |
| kvsort exact (keys36, n=20) | 0.0 | 1.0 |
| sudoku solve (holes=12) | 0.0 | 0.5510 |

Logs:
- `rwkv-diff-future-seed/logs/rightcopy_base_big.log`
- `rwkv-diff-future-seed/logs/rightcopy_future_seed_big.log`
- `rwkv-diff-future-seed/logs/constr_base_big.log`
- `rwkv-diff-future-seed/logs/constr_future_seed_big.log`

## Minimal setup
- Python 3.10+
- torch (CPU or MPS)

## Pseudocode (one‑look intuition)
```
baseline:   s=0;           for t: s=f(s,x[t])
Future‑Seed: s=prev_layer_sT; for t: s=f(s,x[t])
```

Simple: baseline starts each layer from 0; Future‑Seed starts from previous layer’s final state `s_T`.

## One‑command Mac reproduction
Assumes Python + torch are installed.

```bash
bash run.sh
```

Or inline:

```bash
bash -lc 'cd rwkv-diff-future-seed && mkdir -p logs && \
PYTHONUNBUFFERED=1 TRAIN=1 FUTURE_SEED=0 RIGHTCOPY_TASK=1 RIGHTCOPY_LEN=16 RIGHTCOPY_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_future_seed.py | tee logs/rightcopy_base_big.log && \
PYTHONUNBUFFERED=1 TRAIN=1 FUTURE_SEED=1 FUTURE_SEED_ALPHA_INIT=-2 RIGHTCOPY_TASK=1 RIGHTCOPY_LEN=16 RIGHTCOPY_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_future_seed.py | tee logs/rightcopy_future_seed_big.log && \
PYTHONUNBUFFERED=1 TRAIN=1 FUTURE_SEED=0 CONSTR_TASK=1 CONSTR_LEN=16 CONSTR_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_future_seed.py | tee logs/constr_base_big.log && \
PYTHONUNBUFFERED=1 TRAIN=1 FUTURE_SEED=1 FUTURE_SEED_ALPHA_INIT=-2 CONSTR_TASK=1 CONSTR_LEN=16 CONSTR_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_future_seed.py | tee logs/constr_future_seed_big.log'
```

## QA 双向任务（Q->A / A->Q）
新增了 `QA_TASK`，同一条样本格式为 `Q:...|A:...`，可按 `QA_MODE` 控制掩码方向：
- `QA_MODE=qa`：给问题补答案
- `QA_MODE=aq`：给答案反推问题
- `QA_MODE=both`：两种方向混合

关键开关：
- `FUTURE_SEED=1`：开启 Future‑Seed
- `FS_MASK_ONLY=1`：只训练 `future_seed_alpha`（state 参数）和 `mask_emb`（mask 参数）
- `QA_FILE=/path/to/qa.tsv`：可选；每行 `question<TAB>answer`。不提供则自动用合成加法 QA。

运行：
```bash
bash run_qa.sh
```

## Log readability
Each evaluation prints:
- `IN/GT/PR` around the masked span
- mask range and token‑level accuracy

## NeurIPS-Style Baselines (WIP)
New switches (all optional):
- `MODEL=rwkv|transformer|transformer_causal`: RWKV7 vs Transformer-MLM vs Transformer-Causal (same task generator).
- `DECODE=argmax|hungarian`: (Permutation tasks) optional Hungarian decoding for uniqueness.
- `REFINE_STEPS>0`: iterative refinement for in-place fill (Mask-Predict style; used by Sudoku/KVSORT/PERMFILL when `DECODE!=hungarian`).
- `BIN_MASK_MODE=prefix`: future-dependent prefix infill for `DATA_BIN` experiments.
- `ATTN_FS=1`: (Transformer-Causal) attention-side "future-seed" via cross-layer global token.
- `ATTN_FS_COLLECTOR=zero|learned`: (Transformer-Causal) suffix collector token init.
- `ATTN_FS_GATING=1`: (Transformer-Causal) learn per-layer gate on the prefix memory token.
- `ATTN_FS_ALPHA_INIT=...`: (Transformer-Causal) gate init value.
- `ATTN_FS_K=...`: (Transformer-Causal) number of memory/collector tokens (capacity knob).
- `PERM_SINKHORN=1`: (Permutation tasks) add Sinkhorn assignment loss (stronger structured baseline).

Scripts (CUDA, logs under `rwkv-diff-future-seed/exp/`):
- `rwkv-diff-future-seed/run_kvsort_baselines.sh`
- `rwkv-diff-future-seed/run_kvsort_attn_fs.sh`
- `rwkv-diff-future-seed/run_permfill_anchor_sweep.sh`
- `rwkv-diff-future-seed/run_permfill_attn_fs.sh`
- `rwkv-diff-future-seed/run_wikitext_prefix.sh`
- `rwkv-diff-future-seed/run_mbpp_prefix.sh`

Helpers:
- `tools/build_hf_bins.py`: build byte-level `train.bin` / `val.bin` from HF datasets.
- `tools/summarize_jsonl.py`: summarize `LOG_JSONL=...` runs into a Markdown table.

## Log example (baseline vs Future‑Seed)
baseline (FUTURE_SEED=0):
```
mask[21:37] len=16
IN[0:77]: P=5506062696895811|M=________________|R=06########################
GT[21:37]: 1060606060606060
PR[21:37]: 1888888888888888
```

Future‑Seed (FUTURE_SEED=1):
```
mask[21:37] len=16
IN[0:77]: P=5506062696895811|M=________________|R=06########################
GT[21:37]: 1060606060606060
PR[21:37]: 1060606066666660
```
