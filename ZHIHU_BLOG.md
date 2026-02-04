# Future‑Seed：让 DiffusionLLM 更优雅地看到未来

## 结论先行
在 RWKV7 + DiffusionLM 上，Future‑Seed（vertical state non‑causal initialization）能显著提升“未来上下文约束”任务：
- rightcopy：0.1075 → 0.5787
- constr：0.1628 → 0.8281

## 机制一句话
把上一层末状态 s_T 作为下一层初始状态，让深层一开始就带着“未来上下文摘要”，加速全局一致解的收敛。

## 最小任务（机制最清晰）
- **rightcopy**：`L=...|M=...|R=...`，mask `M`，目标 `M=R`
- **constr**：`P=...|M=...|R=dd`，mask `M`，目标 `M = last(P) + (d1,d2交替)`

指标：只统计 mask 段 token acc。

## 结果（1000 step）
| task | FUTURE_SEED=0 acc | FUTURE_SEED=1 acc |
|---|---:|---:|
| rightcopy（LEN=16） | 0.1075 | 0.5787 |
| constr（LEN=16） | 0.1628 | 0.8281 |

日志：
- `rwkv-diff-future-seed/logs/rightcopy_base_big.log`
- `rwkv-diff-future-seed/logs/rightcopy_future_seed_big.log`
- `rwkv-diff-future-seed/logs/constr_base_big.log`
- `rwkv-diff-future-seed/logs/constr_future_seed_big.log`

## 读日志示例（直接看懂）
```
mask[21:37] len=16
IN[0:77]: P=5506062696895811|M=________________|R=06########################
GT[21:37]: 1060606060606060
PR[21:37]: 1060606066666660
```

## 一键 Mac 复现
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
