# Future‑Seed — Diffusion RWKV

A minimal, reproducible prototype that shows Future‑Seed (vertical state non‑causal initialization) can dramatically improve future‑aware tasks in a diffusion‑style RWKV7 model.

## Why it matters
Future‑Seed passes the previous layer’s final state `s_T` as the next layer’s initial state. In diffusion LM (non‑causal), this behaves like “re‑reading” the sequence across depth, strengthening future context instead of one‑pass online updates.

## Results (1000 steps, tiny model)
| task | FUTURE_SEED=0 acc | FUTURE_SEED=1 acc |
|---|---:|---:|
| rightcopy (LEN=16) | 0.1075 | 0.5787 |
| constr (LEN=16) | 0.1628 | 0.8281 |

Logs:
- `rwkv-diff-future-seed/logs/rightcopy_base_big.log`
- `rwkv-diff-future-seed/logs/rightcopy_future_seed_big.log`
- `rwkv-diff-future-seed/logs/constr_base_big.log`
- `rwkv-diff-future-seed/logs/constr_future_seed_big.log`

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
