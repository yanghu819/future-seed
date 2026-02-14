# Future‑Seed — Diffusion RWKV

A minimal, reproducible prototype that shows Future‑Seed (vertical state non‑causal initialization) can dramatically improve future‑aware tasks in a diffusion‑style RWKV7 model.

## Why it matters
Future‑Seed passes the previous layer’s final state `s_T` as the next layer’s initial state. In diffusion LM (non‑causal), this behaves like “re‑reading” the sequence across depth, strengthening future context instead of one‑pass online updates.

## Results (1000 steps, tiny model)
| task | FUTURE_SEED=0 acc | FUTURE_SEED=1 acc |
|---|---:|---:|
| rightcopy (LEN=16) | 0.1075 | 0.6341 |
| constr (LEN=16) | 0.1628 | 0.7384 |

## New: In-place Sudoku (4x4) Constraint Repair
We add a hard in-place constraint repair benchmark: 4x4 Sudoku masked infill (`SUDOKU_TASK=1`), with `solve` defined as satisfying row/col/2x2-block constraints.

Phase curve (curriculum holes∈[4,12], trials=2000):
| holes | FS=0 solve | FS=1 solve |
|---:|---:|---:|
| 10 | 0.0095 | 0.9510 |
| 12 | 0.0000 | 0.5510 |

See `RESULTS.md` and `paper/future-seed-sudoku.pdf`.

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
