# rwkvdllm

## 目标
- 用 RWKV7 作为 backbone，在离散 diffusion LLM 设定下验证 DW‑JRT
- 只保留“对 DW‑JRT 有优势”的最小实验

## 日志怎么读
- `xxx acc`：只统计被 mask 段的 token 准确率（和 GT 对比）
- 每次评测打印 `IN/GT/PR`（mask 附近窗口），方便直观看
- `LOG_WIN=80` 控制窗口大小（默认 80）
- 不保留权重（跑完即删）

## 极简实验（200 step）
配置（共用）：
- `N_LAYER=2 N_EMBD=128 HEAD_SIZE=32`
- `BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128`
- `MAX_ITERS=200 EVAL_INTERVAL=200 EVAL_ITERS=3`

任务：
- rightcopy：`L=...|M=...|R=...`，mask `M`，目标 `M=R`
- constr：`P=...|M=...|R=dd`，mask `M`，目标 `M = last(P) + (d1,d2交替)`

结果：
| task | DW_JRT=0 acc | DW_JRT=1 acc |
|---|---:|---:|
| rightcopy（LEN=8, alpha=-2） | 0.0956 | 0.2031 |
| constr（LEN=16, alpha=-2） | 0.1066 | 0.2766 |

日志：
- `rwkv-diff-dw-jrt/logs/rightcopy_dw0_min.log`
- `rwkv-diff-dw-jrt/logs/rightcopy_dw1_min.log`
- `rwkv-diff-dw-jrt/logs/constr_dw0_min.log`
- `rwkv-diff-dw-jrt/logs/constr_dw1_min.log`
