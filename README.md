# RWKVDLLM — DW‑JRT for Diffusion LLM

A minimal, reproducible prototype that shows DW‑JRT (vertical state non‑causal initialization) can dramatically improve global‑consistency tasks in a diffusion‑style RWKV7 model.

## Why it matters
DW‑JRT passes the previous layer’s final state `s_T` as the next layer’s initial state. In diffusion LM (non‑causal), this behaves like “re‑reading” the sequence across depth, strengthening global constraints instead of one‑pass online updates.

## Results (1000 steps, tiny model)
| task | DW_JRT=0 acc | DW_JRT=1 acc |
|---|---:|---:|
| rightcopy (LEN=16) | 0.1075 | 0.5787 |
| constr (LEN=16) | 0.1628 | 0.8281 |

Logs:
- `rwkv-diff-dw-jrt/logs/rightcopy_dw0_big.log`
- `rwkv-diff-dw-jrt/logs/rightcopy_dw1_big.log`
- `rwkv-diff-dw-jrt/logs/constr_dw0_big.log`
- `rwkv-diff-dw-jrt/logs/constr_dw1_big.log`

## One‑command Mac reproduction
Assumes Python + torch are installed.

```bash
bash -lc 'cd rwkv-diff-dw-jrt && mkdir -p logs && \
PYTHONUNBUFFERED=1 TRAIN=1 DW_JRT=0 RIGHTCOPY_TASK=1 RIGHTCOPY_LEN=16 RIGHTCOPY_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_dw_jrt.py | tee logs/rightcopy_dw0_big.log && \
PYTHONUNBUFFERED=1 TRAIN=1 DW_JRT=1 DW_JRT_ALPHA_INIT=-2 RIGHTCOPY_TASK=1 RIGHTCOPY_LEN=16 RIGHTCOPY_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_dw_jrt.py | tee logs/rightcopy_dw1_big.log && \
PYTHONUNBUFFERED=1 TRAIN=1 DW_JRT=0 CONSTR_TASK=1 CONSTR_LEN=16 CONSTR_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_dw_jrt.py | tee logs/constr_dw0_big.log && \
PYTHONUNBUFFERED=1 TRAIN=1 DW_JRT=1 DW_JRT_ALPHA_INIT=-2 CONSTR_TASK=1 CONSTR_LEN=16 CONSTR_EVAL=1 \
MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3 BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128 \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 LOG_WIN=40 LOG_SAMPLE=1 LOG_OUTPUT=0 \
python rwkv_diff_dw_jrt.py | tee logs/constr_dw1_big.log'
```

## Log readability
Each evaluation prints:
- `IN/GT/PR` around the masked span
- mask range and token‑level accuracy

No weights are kept in the repo.
