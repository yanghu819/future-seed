# RWKVDLLM — VSNI for Diffusion LLM

A minimal, reproducible prototype that shows VSNI (vertical state non‑causal initialization) can dramatically improve future‑aware tasks in a diffusion‑style RWKV7 model.

## Why it matters
VSNI passes the previous layer’s final state `s_T` as the next layer’s initial state. In diffusion LM (non‑causal), this behaves like “re‑reading” the sequence across depth, strengthening future context instead of one‑pass online updates.

## Results (1000 steps, tiny model)
| task | VSNI=0 acc | VSNI=1 acc |
|---|---:|---:|
| rightcopy (LEN=16) | 0.1075 | 0.5787 |
| constr (LEN=16) | 0.1628 | 0.8281 |

Logs:
- `rwkv-diff-vsni/logs/rightcopy_base_big.log`
- `rwkv-diff-vsni/logs/rightcopy_vsni_big.log`
- `rwkv-diff-vsni/logs/constr_base_big.log`
- `rwkv-diff-vsni/logs/constr_vsni_big.log`

## One‑command Mac reproduction
Assumes Python + torch are installed.

```bash
bash -lc 'cd rwkv-diff-vsni && mkdir -p logs && \
PYTHONUNBUFFERED=1 TRAIN=1 VSNI=0 RIGHTCOPY_TASK=1 RIGHTCOPY_LEN=16 RIGHTCOPY_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_vsni.py | tee logs/rightcopy_base_big.log && \
PYTHONUNBUFFERED=1 TRAIN=1 VSNI=1 VSNI_ALPHA_INIT=-2 RIGHTCOPY_TASK=1 RIGHTCOPY_LEN=16 RIGHTCOPY_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_vsni.py | tee logs/rightcopy_vsni_big.log && \
PYTHONUNBUFFERED=1 TRAIN=1 VSNI=0 CONSTR_TASK=1 CONSTR_LEN=16 CONSTR_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_vsni.py | tee logs/constr_base_big.log && \
PYTHONUNBUFFERED=1 TRAIN=1 VSNI=1 VSNI_ALPHA_INIT=-2 CONSTR_TASK=1 CONSTR_LEN=16 CONSTR_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_vsni.py | tee logs/constr_vsni_big.log'
```

## Log readability
Each evaluation prints:
- `IN/GT/PR` around the masked span
- mask range and token‑level accuracy

No weights are kept in the repo.
