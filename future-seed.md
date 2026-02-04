Future‑Seed × RWKV7 DiffusionLLM（对齐版）

已对齐（确认项）
- RWKV7 代码来源：`/Users/huyang/Downloads/gdn_jrt/modded-nanogpt-rwkv`，核心实现完整复制到单文件。
- diffusion 训练/推理/生成流程：对齐 `tiny-diffusion`。
- gdn_exp 目录重命名为 `rwkv-diff-future-seed`，作为完整项目。
- 训练为 DiffusionLLM，不做 prompt/target 分段。
- 对比实验：纯 causal vs 加入 Future‑Seed。
- mask token：沿用 tiny‑diffusion 的方式，mask id = vocab_size（扩词表）。
- 训练超参：按 rwkv7 体系配置。
- 优化器：完全照搬 `train_rwkv7.py` 的优化器体系；CPU/MPS 下 fused=False。
- 生成参数：沿用 tiny‑diffusion 默认值；prompt_len=16（固定）。
- 本地：CPU/MPS 先 smoke；序列长度 T=1024。
- 评估：loss + sample。
- Future‑Seed 落地路径：先做纯 torch 显式 state（CPU/MPS），再改 CUDA kernel（wkv7g 默认路径）。
- logits：保留 train_rwkv7 的 tanh cap。

Future‑Seed 算法理解（全序列、无分段）
核心想法
- 把“再读一遍”从时间维度搬到深度维度。
- 每一层因果扫完整序列得到末状态 s_T。
- 下一层在读序列第一个 token 前，用上一层的 s_T 初始化。
- 除了这个“层间初始状态”外，其余结构完全不变（x0 shortcut / v1 / LN / MLP / 输出头一致）。

形式化（对齐 RWKV 递归视角）
- 把 RWKV7 的注意力核视为递归：
  s_t = F(s_{t-1}, x_t; θ_l)
  h_t = G(s_t, x_t; θ_l)
- baseline：每层 s_0 = 0。
- Future‑Seed：
  第 1 层：s_0^(1)=0 → 得到 s_T^(1)
  第 2 层：s_0^(2)=s_T^(1)
  …
  第 L 层：s_0^(L)=s_T^(L-1)

工程上怎么理解“state”
- RWKV7 的真正递归 state 在 CUDA kernel 内部更新（head 维度的 C×C 状态）。
- Future‑Seed 需要“可读可写”的 s_0 / s_T：
  - CPU/MPS：纯 torch 实现显式维护 state，便于调试与对齐。
  - CUDA：修改 wkv7g kernel 接口，传入 s0、返回 sT。
- v1 是跨层的 value 参考，不是递归 state，本方案不改它。

Future‑Seed 与 baseline 的唯一区别
- baseline：每层 RWKV 注意力核从零状态开始。
- Future‑Seed：每层 RWKV 注意力核从上一层末状态开始。
- 其余所有计算完全一致。

训练（tiny‑diffusion 对齐）
- 输入完整序列。
- 每个样本随机 mask 一部分 token。
- 预测原 token，只在 mask 位置计算 loss。
- logits 用 tanh cap。

推理/生成（tiny‑diffusion 对齐）
- 并行去噪，多轮填补 mask。
- 置信度驱动只更新低置信度位置。
- block 生成方式沿用 tiny‑diffusion。

下一步实现顺序
1) 纯 torch RWKV7 + diffusion（CPU/MPS 跑通）。
2) 加 Future‑Seed（层间 s_T 传递）。
3) CUDA kernel 改造（wkv7g 路径），对齐 torch 行为。
