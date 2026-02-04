# rwkvdllm

## 目标
- 用 RWKV7 作为 backbone，在离散 diffusion LLM 设定下评估 DW‑JRT 是否有收益
- 重点验证“非因果双向信息融合”是否提升任务表现

## 方法
- 训练：随机/指定 mask，仅在 mask 位置做交叉熵
- 生成：并行解码（confidence‑based），与 tiny‑diffusion 风格一致
- DW‑JRT：上一层末尾 state 作为下一层初始 state（`DW_JRT_SCALE` 缩放）

## 日志怎么读
- `xxx acc`：只统计被 mask 段的 token 准确率（和 GT 对比）
- 默认每次评测会打印 `IN/GT/PR`（可读窗口），由 `LOG_SAMPLE=1` 控制
- 窗口大小：`LOG_WIN=80`（只展示 mask 附近）
- 需要整段 GT/PR：运行时加 `MEM_CHECK=1`
- 额外生成输出：`LOG_OUTPUT=1`（默认关闭）

## 主实验任务（双向依赖）
- rightcopy：`L=...|M=...|R=...`，mask `M`，目标 `M=R`
- rightrev：`L=...|M=...|R=...`，mask `M`，目标 `M=reverse(R)`
- constr：`P=...|M=...|R=dd`，mask `M`，目标 `M = last(P) + (d1,d2交替)`
- struct：`{"a":A,"b":A+C,"c":C}`，mask `b`

## 主结果（1000 step，小模型约 0.56M params）
| task | DW_JRT=0 acc | DW_JRT=1 acc |
|---|---:|---:|
| rightcopy | 0.0994 | 0.9581 |
| rightrev | 0.0994 | 0.9500 |
| constr（LEN=16, alpha=-2） | 0.1584 | 0.6828 |
| struct | 0.5444 | 0.9244 |

日志（仅保留主实验）：
- `rwkv-diff-dw-jrt/logs/rightcopy_dw0_long.log`
- `rwkv-diff-dw-jrt/logs/rightcopy_dw1_long.log`
- `rwkv-diff-dw-jrt/logs/rightrev_dw0_long.log`
- `rwkv-diff-dw-jrt/logs/rightrev_dw1_long.log`
- `rwkv-diff-dw-jrt/logs/constr_len16_dw0_long.log`
- `rwkv-diff-dw-jrt/logs/constr_len16_dw1_aN2_long.log`
- `rwkv-diff-dw-jrt/logs/struct_dw0_long2.log`
- `rwkv-diff-dw-jrt/logs/struct_dw1_long2.log`

说明：其它探索性日志已删除；需要可复现时直接按脚本参数重跑即可。
