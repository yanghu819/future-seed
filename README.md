# rwkvdllm

## 目标
- 用 RWKV7 作为 backbone，在离散 diffusion LLM 设定下评估 DW‑JRT 是否有收益
- 对比纯 causal vs 加 DW‑JRT（不做 prompt/target 分段）

## 结构
- `rwkv-diff-dw-jrt/`：单文件实现与实验日志
- `modded-nanogpt-rwkv/`：RWKV7 参考实现与数据脚本
- `tiny-diffusion/`：diffusionLLM 参考实现
- `dw-jrt.md`：方案对齐与记录

## 方法简述
- 训练：随机 mask token，仅在 mask 位置做交叉熵
- 生成：并行解码（confidence‑based），与 tiny‑diffusion 风格一致
- DW‑JRT：上一层末尾 state 作为下一层初始 state（`DW_JRT_SCALE` 缩放）

## 论文摘要（arXiv:2601.22031）
CARD 提出在严格 causal attention 下重写 diffusion 过程，实现单次前向的密集 per‑token 监督；为稳定训练引入 soft‑tailed masking 与基于信噪比的重加权，并支持基于置信度的动态并行解码；在保持 ARM 训练效率的同时获得 diffusion 的高吞吐推理。citeturn2view0

论文链接：
```
https://arxiv.org/abs/2601.22031
```

## 实验
### 大规模（fineweb‑edu sample‑10BT, bin）
配置（两组一致）：
- `SEQ_LEN=128, N_LAYER=2, N_EMBD=128, HEAD_SIZE=32`
- `BATCH_SIZE=64, DEVICE_BSZ=16, MAX_ITERS=1000`
- `EVAL_INTERVAL=200, EVAL_ITERS=50, WARMDOWN_ITERS=200`

结果（来自 `rwkv-diff-dw-jrt/logs`）：
| step | DW_JRT=0 train | DW_JRT=0 val | DW_JRT=1 (scale=20) train | DW_JRT=1 val |
|---:|---:|---:|---:|---:|
| 0   | 10.8259 | 10.8259 | 10.8259 | 10.8259 |
| 200 | 6.5481  | 8.4659  | 6.4915  | 8.5465  |
| 400 | 5.1953  | 9.1386  | 5.1669  | 9.1670  |
| 600 | 4.1939  | 9.6602  | 4.2244  | 9.6515  |
| 800 | 3.7682  | 10.1131 | 3.7630  | 10.1012 |
| 999 | 3.3028  | 10.3396 | 3.3093  | 10.2167 |

简短分析：
- 训练 loss 接近
- 验证 loss 后期 DW‑JRT 略低，但差距很小；整体有过拟合趋势（val 上升）

### 小规模过拟合（overfit.txt, char）
配置：
- `SEQ_LEN=64, N_LAYER=2, N_EMBD=128, HEAD_SIZE=32`
- `BATCH_SIZE=32, DEVICE_BSZ=8, MAX_ITERS=200`

结果（step 199）：
- DW_JRT=0: train 0.5793 / val 0.6434
- DW_JRT=1 (scale=20): train 0.4348 / val 0.4952

简短分析：
- 小数据过拟合上 DW‑JRT 收敛更好

## 复现实验
### 大规模对比
```
cd rwkv-diff-dw-jrt
PYTHONUNBUFFERED=1 \
DATA_BIN=../modded-nanogpt-rwkv/data/fineweb_edu10B/fineweb_train_*.bin \
DATA_VAL_BIN=../modded-nanogpt-rwkv/data/fineweb_edu10B/fineweb_val_*.bin \
VOCAB_SIZE=50304 TRAIN=1 GEN_TOKENS=1 \
SEQ_LEN=128 N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 \
BATCH_SIZE=64 DEVICE_BSZ=16 MAX_ITERS=1000 EVAL_INTERVAL=200 EVAL_ITERS=50 WARMDOWN_ITERS=200 \
DW_JRT=0 WEIGHTS_PATH=weights/diffusion_dw0_v3.pt \
python -u rwkv_diff_dw_jrt.py

PYTHONUNBUFFERED=1 \
DATA_BIN=../modded-nanogpt-rwkv/data/fineweb_edu10B/fineweb_train_*.bin \
DATA_VAL_BIN=../modded-nanogpt-rwkv/data/fineweb_edu10B/fineweb_val_*.bin \
VOCAB_SIZE=50304 TRAIN=1 GEN_TOKENS=1 \
SEQ_LEN=128 N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 \
BATCH_SIZE=64 DEVICE_BSZ=16 MAX_ITERS=1000 EVAL_INTERVAL=200 EVAL_ITERS=50 WARMDOWN_ITERS=200 \
DW_JRT=1 DW_JRT_SCALE=20 WEIGHTS_PATH=weights/diffusion_dw1_v3.pt \
python -u rwkv_diff_dw_jrt.py
```

### 小规模过拟合
```
cd rwkv-diff-dw-jrt
PYTHONUNBUFFERED=1 DATA_PATH=overfit.txt TRAIN=1 GEN_TOKENS=1 \
SEQ_LEN=64 N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 \
BATCH_SIZE=32 DEVICE_BSZ=8 MAX_ITERS=200 EVAL_INTERVAL=50 EVAL_ITERS=20 WARMDOWN_ITERS=50 \
DW_JRT=0 WEIGHTS_PATH=weights/overfit_dw0.pt \
python -u rwkv_diff_dw_jrt.py

PYTHONUNBUFFERED=1 DATA_PATH=overfit.txt TRAIN=1 GEN_TOKENS=1 \
SEQ_LEN=64 N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 \
BATCH_SIZE=32 DEVICE_BSZ=8 MAX_ITERS=200 EVAL_INTERVAL=50 EVAL_ITERS=20 WARMDOWN_ITERS=50 \
DW_JRT=1 DW_JRT_SCALE=20 WEIGHTS_PATH=weights/overfit_dw1.pt \
python -u rwkv_diff_dw_jrt.py
```
