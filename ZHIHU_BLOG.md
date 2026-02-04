# Future‑Seed：在 DiffusionLLM 中用“垂直状态非因果初始化”更优雅地看到未来

## 一句话摘要
在 RWKV7 + 离散扩散式语言建模（DiffusionLM）中，我引入 Future‑Seed：把上一层的末状态 s_T 作为下一层的初始状态，从而在“深度维”实现对整段序列的再读与校正。最小实验显示：在跨段复制与未来上下文约束任务上，Future‑Seed 的提升非常明显（rightcopy: 0.10 → 0.58，constr: 0.16 → 0.83），这支持“多轮优化同一序列”的机制直觉。

---

## 1. 背景与动机
因果 LM 的本质是“一眼过”：状态只向前流动，在线更新，一次读完就结束。DiffusionLM 则不同，它对同一序列进行多轮去噪，等价于多次修正同一个全局解。于是我思考：能否把“再读一遍”的能力引入 RWKV 的层间结构？

Future‑Seed 的核心就是：**把上一层的末状态 s_T 作为下一层的初始状态**。这相当于把“全序列摘要”直接提供给更深层，让它从一开始就带着全局记忆，特别适合非因果、需要“看到未来”的任务。

---

## 2. 方法：Future‑Seed 的实现细节
在 `rwkv-diff-future-seed/rwkv_diff_future_seed.py` 中，Future‑Seed 的做法非常克制：

- baseline：每层初始状态 s_0 = 0
- Future‑Seed：每层初始状态 s_0 = 上一层末状态 s_T
- 为避免“被 w 抹掉”，注入被写成：
  - `inject = base * sigmoid(alpha)`
  - `s = s + inject * (1 - w)`

这里 `alpha` 是每层每头一个可学习标量（默认 `FUTURE_SEED_ALPHA_INIT=-2`，表示初始注入较弱）。

**核心观点**：这不是更强采样，而是“更快进入全局一致解”的初始化方式。

---

## 3. 与 DiffusionLM 的关系
DiffusionLM 的训练/生成逻辑是：
- 训练：随机 mask，**只对 mask 段计算 loss**
- 生成：并行去噪、多轮填补 mask

Future‑Seed 适合 DiffusionLM 的原因：
- 非因果：不会出现“信息泄露”问题
- 多轮修正：s_T 提供未来上下文摘要，利于跨段约束

这正好契合“多轮优化同一序列，克服一眼过”的直觉。

---

## 4. 任务定义（合成但机制清晰）
为了验证“更优雅地看到未来”的优势，我使用两个强依赖全局信息的任务：

### rightcopy
- 格式：`L=...|M=...|R=...`
- 训练时 mask `M`
- 目标：`M = R`

### constr
- 格式：`P=...|M=...|R=dd`
- 训练时 mask `M`
- 目标：`M = last(P) + (d1,d2交替)`

两者都是“跨字段一致性”任务，最容易检验 Future‑Seed 的有效性。

---

## 5. 实验设置（极简、Mac 可跑）
共用设置：
- `N_LAYER=2 N_EMBD=128 HEAD_SIZE=32`
- `BATCH_SIZE=32 DEVICE_BSZ=8 SEQ_LEN=128`
- `MAX_ITERS=1000 EVAL_INTERVAL=500 EVAL_ITERS=3`

对比：
- baseline：`FUTURE_SEED=0`
- Future‑Seed：`FUTURE_SEED=1 FUTURE_SEED_ALPHA_INIT=-2`

日志默认打印 `IN / GT / PR`（mask 附近窗口），可直接人工检查。

---

## 6. 结果
| task | FUTURE_SEED=0 acc | FUTURE_SEED=1 acc |
|---|---:|---:|
| rightcopy（LEN=16） | 0.1075 | 0.5787 |
| constr（LEN=16） | 0.1628 | 0.8281 |

日志文件：
- `rwkv-diff-future-seed/logs/rightcopy_base_big.log`
- `rwkv-diff-future-seed/logs/rightcopy_future_seed_big.log`
- `rwkv-diff-future-seed/logs/constr_base_big.log`
- `rwkv-diff-future-seed/logs/constr_future_seed_big.log`

**解读：** baseline 几乎不学会规则；Future‑Seed 能明显恢复正确结构，且 PR 接近 GT。

---

## 7. 读日志示例（直观）
日志每次评测会输出：
- mask 区间
- IN（含 mask）
- GT（真实答案）
- PR（模型预测）

例如：
```
mask[21:37] len=16
IN[0:77]: P=5506062696895811|M=________________|R=06########################
GT[21:37]: 1060606060606060
PR[21:37]: 1060606066666660
```
肉眼就能看出是否满足规则。

---

## 8. 为什么这是“有意义”的改动
Future‑Seed 的本质是：**把“再读一遍”从时间维搬到深度维**。
在扩散式生成里，这意味着：
- 早期就注入未来上下文摘要，减少随机游走
- 更快满足未来上下文约束
- 多轮去噪不是简单采样，而是结构性修正

这解释了为何在 rightcopy / constr 上提升如此显著。

---

## 9. 局限与下一步
局限：
- 目前只有合成任务，没有真实世界验证
- 单次运行，无多种 seed 稳定性分析

下一步（最小成本方向）：
- 真实文本上的结构化抽取/一致性生成
- 多段检索 + 结构化汇总
- 量化“平均去噪步数”是否下降

---

## 10. 一键 Mac 复现
假设已安装 Python 与 torch：

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

如果你要更“顶会级”的证据链，我可以基于这些最小实验扩展为：
- 多 seed 稳定性
- 更长序列
- 真任务基准
- 解码效率/收敛速度分析
