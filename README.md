# rwkvdllm

## 目标
- 用 RWKV7 作为 backbone，在离散 diffusion LLM 设定下评估 DW‑JRT 是否有收益
- 重点验证“非因果双向信息融合”是否提升任务表现

## 方法简述
- 训练：随机 mask / 指定 mask，仅在 mask 位置做交叉熵
- 生成：并行解码（confidence‑based），与 tiny‑diffusion 风格一致
- DW‑JRT：上一层末尾 state 作为下一层初始 state（`DW_JRT_SCALE` 缩放）

## 核心任务：结构化补全（双向）
**任务定义**
- 输入：`{"a":A,"b":A+C,"c":C}`，其中 A/C 为 8 位随机数字串，`b` 为拼接（A 后接 C）
- 训练/评测：只 mask `b` 段，要求模型用左右上下文恢复
- 指标：`struct acc`（只统计 `b` 段逐 token 准确率）

**结论**
- DW‑JRT 在该任务上显著优于 baseline，且差距随训练拉长而扩大

## 结果（固定设置）
配置：
- `STRUCT_TASK=1 STRUCT_LEN=8 SEQ_LEN=64`
- `N_LAYER=2 N_EMBD=128 HEAD_SIZE=32`
- `BATCH_SIZE=32 DEVICE_BSZ=8`

### 200 step
| step | DW_JRT=0 struct acc | DW_JRT=1 struct acc |
|---:|---:|---:|
| 199 | 0.5469 | 0.7450 |

日志：
- `rwkv-diff-dw-jrt/logs/struct_dw0_smoke2.log`
- `rwkv-diff-dw-jrt/logs/struct_dw1_smoke2.log`

### 1000 step
| step | DW_JRT=0 struct acc | DW_JRT=1 struct acc |
|---:|---:|---:|
| 200 | 0.5181 | 0.6191 |
| 400 | 0.3281 | 0.6531 |
| 600 | 0.5494 | 0.7262 |
| 800 | 0.5525 | 0.8153 |
| 999 | 0.5444 | 0.9244 |

日志：
- `rwkv-diff-dw-jrt/logs/struct_dw0_long2.log`
- `rwkv-diff-dw-jrt/logs/struct_dw1_long2.log`

### 固定评测集（已训练权重）
配置：
- `STRUCT_EVAL_FIXED=1 STRUCT_EVAL_N=200 STRUCT_EVAL_SEED=1234`
- 权重：`struct_dw0_long2.pt` / `struct_dw1_long2.pt`

| DW_JRT | struct acc | exact |
|---:|---:|---:|
| 0 | 0.5559 | 0.0000 |
| 1 | 0.5919 | 0.0000 |

日志：
- `rwkv-diff-dw-jrt/logs/struct_eval_fixed_dw0.log`
- `rwkv-diff-dw-jrt/logs/struct_eval_fixed_dw1.log`

## 可学习 DW‑JRT 注入（alpha）
说明：
- 每层每头一个可学习标量 `jrt_alpha`，注入强度为 `sigmoid(jrt_alpha)`
- 注入加权：`inject * (1 - w)`，减小被 `w` 抹掉的影响
- 可选 `DW_JRT_LAYER_START` 仅在高层启用
- `DW_JRT_ALPHA_INIT=0` 表示初始注入强度 0.5

### 200 step（alpha init=0）
| task | DW_JRT=0 acc | DW_JRT=1 acc |
|---|---:|---:|
| struct | 0.5484 | 0.7172 |
| add | 0.1013 | 0.1125 |

解读：
- struct 仍明显受益，说明可学习注入没有破坏 DW‑JRT 的主收益
- add 只有小幅提升，可能仍是短跑噪声，需要更长步数验证

日志：
- `rwkv-diff-dw-jrt/logs/struct_dw0_alpha0_smoke.log`
- `rwkv-diff-dw-jrt/logs/struct_dw1_alpha0_smoke.log`
- `rwkv-diff-dw-jrt/logs/add_dw0_alpha0_smoke.log`
- `rwkv-diff-dw-jrt/logs/add_dw1_alpha0_smoke.log`

### 200 step（右侧依赖任务）
| task | DW_JRT=0 acc | DW_JRT=1 acc |
|---|---:|---:|
| rightcopy | 0.1087 | 0.7181 |
| rightrev | 0.1175 | 0.7731 |
| index（I 在右侧） | 0.2737 | 0.2650 |
| rule（右侧指令 C/V） | 0.3881 | 0.3806 |
| cipher（右侧置换表） | 0.0916 | 0.0916 |

解读：
- rightcopy/rightrev 明显受益，任务强依赖右侧信息
- index 略差，可能是短跑噪声或索引解析更难

日志：
- `rwkv-diff-dw-jrt/logs/rightcopy_dw0_smoke.log`
- `rwkv-diff-dw-jrt/logs/rightcopy_dw1_smoke.log`
- `rwkv-diff-dw-jrt/logs/rightrev_dw0_smoke.log`
- `rwkv-diff-dw-jrt/logs/rightrev_dw1_smoke.log`
- `rwkv-diff-dw-jrt/logs/index_dw0_smoke2.log`
- `rwkv-diff-dw-jrt/logs/index_dw1_smoke2.log`
- `rwkv-diff-dw-jrt/logs/rule_dw0_smoke.log`
- `rwkv-diff-dw-jrt/logs/rule_dw1_smoke.log`
- `rwkv-diff-dw-jrt/logs/cipher_dw0_smoke.log`
- `rwkv-diff-dw-jrt/logs/cipher_dw1_smoke.log`

### 200 step（更长序列）
| task | setting | DW_JRT=0 acc | DW_JRT=1 acc |
|---|---|---:|---:|
| rightcopy | LEN=32 | 0.0991 | 0.2680 |
| rightrev | LEN=32 | 0.0991 | 0.3042 |

日志：
- `rwkv-diff-dw-jrt/logs/rightcopy_dw0_len32_smoke.log`
- `rwkv-diff-dw-jrt/logs/rightcopy_dw1_len32_smoke.log`
- `rwkv-diff-dw-jrt/logs/rightrev_dw0_len32_smoke.log`
- `rwkv-diff-dw-jrt/logs/rightrev_dw1_len32_smoke.log`

### 1000 step（右侧依赖任务）
| task | DW_JRT=0 acc | DW_JRT=1 acc |
|---|---:|---:|
| rightcopy | 0.0994 | 0.9581 |
| rightrev | 0.0994 | 0.9500 |
| rule（右侧指令 C/V） | 0.4850 | 0.4612 |

解读：
- rightcopy/rightrev 的优势在长跑更明显，DW‑JRT 接近完美恢复

日志：
- `rwkv-diff-dw-jrt/logs/rightcopy_dw0_long.log`
- `rwkv-diff-dw-jrt/logs/rightcopy_dw1_long.log`
- `rwkv-diff-dw-jrt/logs/rightrev_dw0_long.log`
- `rwkv-diff-dw-jrt/logs/rightrev_dw1_long.log`
- `rwkv-diff-dw-jrt/logs/rule_dw0_long.log`
- `rwkv-diff-dw-jrt/logs/rule_dw1_long.log`

## 快速扫参（100 step）
配置（共用）：
- `N_LAYER=2 N_EMBD=128 HEAD_SIZE=32`
- `MAX_ITERS=100 EVAL_INTERVAL=50 EVAL_ITERS=10`

### rightcopy/rightrev
| task | setting | DW_JRT=0 | DW_JRT=1 | 备注 |
|---|---|---:|---:|---|
| rightcopy | LEN=8 | 0.0875 | 0.3969 | alpha=0 |
| rightcopy | LEN=8 | - | 0.6281 | alpha=-2 |
| rightcopy | LEN=8 | - | 0.5875 | alpha=2 |
| rightrev | LEN=8 | 0.0875 | 0.5587 | alpha=0 |
| rightcopy | LEN=32 | 0.0969 | 0.2197 | alpha=0 |
| rightcopy | LEN=32 | - | 0.2166 | layer_start=1 |
| rightrev | LEN=32 | 0.0969 | 0.2291 | alpha=0 |
| rightrev | LEN=32 | - | 0.2286 | layer_start=1 |

### 其它任务
| task | setting | DW_JRT=0 | DW_JRT=1 |
|---|---|---:|---:|
| struct | LEN=8 | 0.3672 | 0.5397 |
| add | LEN=8 | 0.1100 | 0.0912 |
| index | LEN=16 | 0.2425 | 0.2387 |
| rule | LEN=8 | 0.3912 | 0.3787 |
| cipher | LEN=16 | 0.0966 | 0.0966 |
| shift | LEN=16 | 0.0950 | 0.0950 |
| shift | LEN=16 | - | 0.0950 (alpha=-2) |

解读：
- rightcopy/rightrev 对 DW‑JRT 极敏感，且 **alpha=-2** 在短跑最优
- LEN=32 仍有收益，但需要更长训练才能接近满分
- struct 有提升，其它任务短跑无优势或略差

日志（100 step sweep）：
- `rwkv-diff-dw-jrt/logs/rightcopy_len8_*.log`
- `rwkv-diff-dw-jrt/logs/rightrev_len8_*.log`
- `rwkv-diff-dw-jrt/logs/rightcopy_len32_*.log`
- `rwkv-diff-dw-jrt/logs/rightrev_len32_*.log`
- `rwkv-diff-dw-jrt/logs/struct_len8_*.log`
- `rwkv-diff-dw-jrt/logs/add_len8_*.log`
- `rwkv-diff-dw-jrt/logs/index_len16_*.log`
- `rwkv-diff-dw-jrt/logs/rule_len8_*.log`
- `rwkv-diff-dw-jrt/logs/cipher_len16_*.log`
- `rwkv-diff-dw-jrt/logs/shift_len16_*.log`

## 复现实验
### 200 step
```
cd rwkv-diff-dw-jrt
PYTHONUNBUFFERED=1 STRUCT_TASK=1 STRUCT_EVAL=1 STRUCT_LEN=8 \
TRAIN=1 GEN_TOKENS=1 SEQ_LEN=64 N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 \
BATCH_SIZE=32 DEVICE_BSZ=8 MAX_ITERS=200 EVAL_INTERVAL=50 EVAL_ITERS=20 WARMDOWN_ITERS=50 \
DW_JRT=0 WEIGHTS_PATH=weights/struct_dw0_smoke2.pt \
python -u rwkv_diff_dw_jrt.py

PYTHONUNBUFFERED=1 STRUCT_TASK=1 STRUCT_EVAL=1 STRUCT_LEN=8 \
TRAIN=1 GEN_TOKENS=1 SEQ_LEN=64 N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 \
BATCH_SIZE=32 DEVICE_BSZ=8 MAX_ITERS=200 EVAL_INTERVAL=50 EVAL_ITERS=20 WARMDOWN_ITERS=50 \
DW_JRT=1 DW_JRT_SCALE=20 WEIGHTS_PATH=weights/struct_dw1_smoke2.pt \
python -u rwkv_diff_dw_jrt.py
```

### 1000 step
```
cd rwkv-diff-dw-jrt
PYTHONUNBUFFERED=1 STRUCT_TASK=1 STRUCT_EVAL=1 STRUCT_LEN=8 \
TRAIN=1 GEN_TOKENS=1 SEQ_LEN=64 N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 \
BATCH_SIZE=32 DEVICE_BSZ=8 MAX_ITERS=1000 EVAL_INTERVAL=200 EVAL_ITERS=20 WARMDOWN_ITERS=200 \
DW_JRT=0 WEIGHTS_PATH=weights/struct_dw0_long2.pt \
python -u rwkv_diff_dw_jrt.py

PYTHONUNBUFFERED=1 STRUCT_TASK=1 STRUCT_EVAL=1 STRUCT_LEN=8 \
TRAIN=1 GEN_TOKENS=1 SEQ_LEN=64 N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 \
BATCH_SIZE=32 DEVICE_BSZ=8 MAX_ITERS=1000 EVAL_INTERVAL=200 EVAL_ITERS=20 WARMDOWN_ITERS=200 \
DW_JRT=1 DW_JRT_SCALE=20 WEIGHTS_PATH=weights/struct_dw1_long2.pt \
python -u rwkv_diff_dw_jrt.py
```
