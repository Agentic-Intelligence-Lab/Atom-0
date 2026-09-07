# π0.7 第二步：MEM（多尺度记忆）框架复现方案

## Context

我们正在 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) 基础上**逐模块**复现 π0.7（[arXiv:2604.15483](https://arxiv.org/abs/2604.15483)）。第一步 KI（Knowledge Insulation, [arXiv:2505.23705](https://arxiv.org/abs/2505.23705)）假设已完成。本方案聚焦**第二步 MEM**（[arXiv:2603.03596](https://arxiv.org/abs/2603.03596) / [Mem.pdf](https://www.pi.website/download/Mem.pdf)，2026-03 PI 发布）。

**约束**：
- 没有 PI 预训练量级的私有数据，仅依赖开源数据集（DROID / LIBERO / OXE 子集）。
- 复现和验证**不能太复杂**——优先 MVP，后做完整版。
- 算力 >8 卡（H100/A100），不是瓶颈。

**目标**：在 openpi 上加上"多尺度记忆"使模型能基于历史观察决策，并通过专门的记忆 benchmark 量化收益。

---

## 1. MEM 论文核心要点（精炼）

| 维度 | 论文设计 |
|---|---|
| 短期记忆 | 视频编码器（ViT 衍生）+ space-time separable attention，**16 帧 / 54 秒**稠密视觉 token |
| 长期记忆 | **自然语言摘要**（如 "I placed a plate in the cabinet…"），由**同一个 backbone LLM** 周期性生成 |
| 整体上下文 | 短期视觉 + 长期摘要拼接，可达 **15 分钟** |
| 训练阶段 | 预训练 6 帧（5 历史+当前，间隔 1s）；后训练扩到 18 帧 / 54s |
| 推理 | 每步 forward；摘要每 ~30s 由同模型重新生成（chain-of-thought） |
| 关键挑战 | **训练-推理分布偏移**：训练用近最优 demo，推理需重试失败子任务 → 需混合训练数据 |
| 论文 ablation | 关掉视频记忆 → 长任务"卡顿"；关掉文本摘要 → 食谱步骤无法追踪；筷子可变高度 +11%、冰箱未知铰链 +62% |

**关键事实（与本方案有关）**：
- 摘要不是另起一个 LM，是同 backbone 自回归生成 → 不增加额外参数。
- ablation 显示 **两个组件都关键**，但短期视觉单独已能在中等长度任务上获得显著收益 → MVP 仅做短期视觉的合理性。

---

## 2. 社区现状

- **没有任何 pi0.7 MEM 的开源复现**（截至 2026-04）。
- 相关可借鉴工作：
  - [MemoryVLA](https://shihao1895.github.io/MemoryVLA/) — perceptual + cognitive memory，12 真实任务 84% 成功率（+26pt）。
  - ReMem-VLA（[arXiv:2603.12942](https://arxiv.org/abs/2603.12942)）— 帧级+块级双递归 memory。
  - [VLA-Cache](https://arxiv.org/html/2502.02175v1)、Compressor-VLA — 推理优化方向。
- **openpi 当前完全单帧**，无 history / context window flag。

---

## 3. openpi 改造点（已定位）

| 模块 | 当前状态 | 需改动 |
|---|---|---|
| `Observation` dataclass | [src/openpi/models/model.py:83-130](src/openpi/models/model.py#L83-L130)，单帧无时序维度 | 加 `T` 时序维度：`images: dict[str, "*b T h w c"]` |
| `embed_prefix` | [src/openpi/models/pi0.py:106-137](src/openpi/models/pi0.py#L106-L137)，单帧逐 cam 喂 SigLIP | 多帧并行喂 SigLIP，concat 后加 temporal pos embed |
| transforms | [src/openpi/transforms.py:185-289](src/openpi/transforms.py#L185-L289) | 新增 `StackHistoryFrames` transform，从 LeRobot dataset 滚动取 N 帧 |
| 配置 | [src/openpi/models/pi0_config.py:27,38-39](src/openpi/models/pi0_config.py#L27)，仅有 `max_token_len` | 新增 `history_length: int = 1`、`history_stride: int = 30`（@30Hz≈1s） |
| Tokenizer | [src/openpi/models/tokenizer.py](src/openpi/models/tokenizer.py)，`max_len=200`（pi05） | MVP 不动；阶段 B 长期摘要时增大到 ≥512 |
| 数据加载 | LeRobot 单帧 | 改 LeRobot dataset 配置 `delta_timestamps` 取 N 帧（LeRobot 原生支持，无需写新代码） |

**核心可复用资产**：
- `pi0_fast.py` 的自回归生成循环（阶段 B 摘要生成会用到，参考 [pi0_fast.py:198](src/openpi/models/pi0_fast.py#L198)）。
- LeRobot dataset 的 `delta_timestamps` 接口直接支持取历史帧 → 训练数据侧改动最小。

---

## 4. 推荐实施路线

### 阶段 A：短期视觉记忆 MVP（**优先做，3-4 周**）

#### A.1 设计选择
- **history_length = 6**（预训练 ≈ 论文 6 帧 / 1s 间隔；后续可扩到 16 帧）。
- **必须实现 space-time separable attention**——这是 pi0.7 论文宣称的核心算法贡献，不能跳过。
- **Temporal positional embedding**：加 `T` 维 learnable embed（`time_pos_embed[0..T-1]`），与 SigLIP 输出按 frame index 相加。
- **保持单帧前向兼容**：`history_length=1` 时空间-时间路径自动退化（temporal block 短路），回归测试用。

#### A.1.1 Space-Time Separable Attention 设计

**为什么不能用普通 full attention**：
- 6 帧 × 3 cam × 256 spatial token = **4608 视觉 token**；扩到论文的 16 帧后是 **12288 token**。
- Full attention cost 是 O((T·N)²)，4608² ≈ 21M、12288² ≈ 150M ops/head/layer，对 50Hz 控制循环不现实。
- Separable 把 cost 拆成 O(T·N²) spatial + O(T²·N) temporal = 21M → **1.5M + 0.4M ≈ 2M**，**降 ~10×**（16 帧场景 ~30×）。

**采用 TimeSformer 风格的 "divided space-time attention"**（[arXiv:2102.05095](https://arxiv.org/abs/2102.05095)），也是 ViViT / VideoMAE 的标准做法。**Late fusion**：SigLIP 仍按帧独立做 spatial attention（保留预训练权重），输出后接 K 层独立的 `TemporalAttention` block 做跨帧聚合。

```
[input] T frames × 3 cams × (h×w) patches
   │
   ▼ (并行喂入 SigLIP, frozen 或 fine-tune)
SigLIP per-frame spatial attention            # 复用预训练，O(T·N²)
   │
   ▼ output: (B, T, num_cam·N, D)
Add learnable temporal_pos_embed[0..T-1]      # 沿 T 维加 pos
   │
   ▼
K × TemporalBlock:                            # K=2-4，新增
  for each spatial position p:
    x[:, :, p, :] = MultiHeadAttention(       # attend across T
        queries=x[:, :, p, :], 
        keys=x[:, :, p, :], 
        values=x[:, :, p, :])
  + FFN + residual + RMSNorm
   │
   ▼ flatten (B, T·num_cam·N, D) → concat 给 Gemma backbone 做 cross-modal
```

**实现路径**：
- 新增模块 `src/openpi/models/temporal_attention.py`（~150 行 Flax NNX）：
  - `TemporalBlock`：标准 attention block，但 attention 沿 `T` 轴（reshape 后 batch 包含空间位置）。
  - `SpaceTimeSeparableEncoder`：堆叠 K 层 TemporalBlock + temporal pos embed。
  - 复用 `src/openpi/models/gemma.py` 里现成的 `Attention` / `RMSNorm` / `FeedForward` 组件，不重新实现。
- 配置新增字段：
  - `num_temporal_layers: int = 2`
  - `temporal_attn_heads: int = 8`
  - `temporal_attn_dim: int` 默认与 SigLIP 输出 dim 对齐
- PyTorch 路径同步实现到 `src/openpi/models_pytorch/temporal_attention_pytorch.py`。

**为什么不直接改 SigLIP 内部 attention 层**：
- 会破坏 PaliGemma 预训练权重对齐，论文也未声称这么做。
- Late fusion + 独立 temporal block 是 video-ViT 主流方案，便于初始化（temporal block 可零初始化使初值等价无 history）。

**初始化技巧**：
- TemporalBlock 的最后一个 dense / FFN 输出层用 **零初始化**（BERT / ViViT 标准 trick），训练开始时 temporal block 是恒等映射，等价于 history=1 的 baseline → **训练 stability + 回归保护一举两得**。

#### A.2 改动清单（按文件）

```
src/openpi/models/
├── model.py:83-130
│     Observation: images dict 内 shape 由 (b,h,w,c) 改 (b,T,h,w,c)
│     wrist_image / base_image 加 T 维
├── pi0_config.py:18-118
│     新增字段：
│       history_length: int = 1
│       history_stride: int = 30        # @30Hz≈1s
│       num_temporal_layers: int = 2
│       temporal_attn_heads: int = 8
├── temporal_attention.py（新文件，~150 行）
│     TemporalBlock: 沿 T 轴的 multi-head attention + FFN，零初始化
│     SpaceTimeSeparableEncoder: temporal_pos_embed + K 层 TemporalBlock
│     复用 gemma.py 的 Attention / RMSNorm / FeedForward
├── pi0.py:106-137
│     embed_prefix 改造：
│       1) images[name] reshape (b,T,h,w,c) → (b·T,h,w,c) 喂 SigLIP（per-frame spatial attention）
│       2) 输出 reshape 回 (b, T, num_patches, dim)
│       3) 过 SpaceTimeSeparableEncoder（per-spatial-position temporal attention）
│       4) flatten (b, T·num_cam·N, dim) → concat 文本 token 给 Gemma
│     注意：attention mask 要扩到 T·num_patches·num_cam 长度
└── pi0_pytorch.py + temporal_attention_pytorch.py（PyTorch 路径）
     同步改

src/openpi/transforms.py:185-289
     新增 StackHistoryFrames transform（如果 LeRobot delta_timestamps 不够用）

src/openpi/training/config.py
     新增 pi05_mem_libero / pi05_mem_droid named configs

src/openpi/policies/libero_policy.py、droid_policy.py
     LiberoInputs / DroidInputs：image 字段从 (h,w,c) 改 (T,h,w,c)
     从 LeRobot dataset 用 delta_timestamps=[-5, -4, -3, -2, -1, 0] 拿 6 帧
```

#### A.3 训练数据
- **LIBERO（全量 130 任务 + Long 子集）**：仿真，免费，**主要训练源**。
- **DROID 子集**：CC-BY 4.0，76k episode 中取 5-10k 子集，提供真实场景多样性。
- **OXE 子集**（可选）：BridgeV2、Fractal20—增加跨平台数据。
- 起点 ckpt：`pi05_base`（KI 已训过的版本）→ 直接 fine-tune，不从 PaliGemma 起。
- 训练步数：30k–60k step，bf16，FSDP，~16 卡 H100，2-4 天可跑完。

#### A.4 关键风险
- 显存：6 帧 × 3 cam × 256 token = 4608 视觉 token + 200 prompt。**Separable attention 把 self-attn cost 降回到与 spatial-only 同量级**，主要显存增量在激活值（线性于 T，~6×），需开 gradient checkpointing。
- 跨模态阶段（Gemma backbone 处理 4608 视觉 + 200 文本 token）仍是 full attention，~50× 增加 → flash attention 必开，必要时降 history 到 4。
- 数据 IO：6 帧加载 IO 增 6×，建议预先用 LeRobot 的 `image_transforms` 缓存。
- TemporalBlock 训练 stability：用零初始化策略保证训练前 1k step 等价 baseline，避免梯度爆炸。

---

### 阶段 B：长期语言记忆（给出方案，2-3 周，**优先级低于 A**）

#### B.1 设计选择
- **生成器**：复用 backbone Gemma（或 KI 训练好的同模型），不引入新模型。
- **触发频率**：每 30 秒（30 step @ 1Hz 或论文里的"语义切换 / Δ=4s"，MVP 用固定 30s 简化）调用一次自回归生成，类似 `pi0_fast.py:sample_actions` 但生成的是文字。
- **摘要 prompt**：`"Task: <ell>. Recent observations: <短期视觉 token>. History summary so far: <prev summary>. New summary:"` → 自回归生成 ≤ 64 token。
- **拼接位置**：摘要 token 放 prefix 最前面（task instruction 之前），保证长 horizon 信息一定看到。
- **训练**：摘要部分用 cross-entropy，与 KI 的 FAST CE loss 共享一套 next-token-prediction 实现。**摘要监督信号怎么来**是关键问题（见 §B.3）。

#### B.2 改动清单

```
src/openpi/models/pi0.py
  - 新增 generate_summary(obs, prev_summary, max_new_tokens=64)
    复制 pi0_fast.py:217-270 的自回归循环，仅生成文字
  - compute_loss 加 CE 分支：summary tokens 监督

src/openpi/transforms.py
  - 新增 SummarySupervisionTransform：从 demo 标注或自动生成摘要 GT

src/openpi/serving/
  - 推理时维护 summary state，按时间触发更新
  - 异步更新（不阻塞 50Hz 动作生成），用单独线程
```

#### B.3 摘要监督信号来源（**最难的工程问题**）
论文用 PI 内部 coaching 数据 + 子任务标注，我们没有。三种替代：

| 来源 | 成本 | 质量 |
|---|---|---|
| **方案 1：用 GPT-4o / Gemini 离线给 demo 视频生成摘要 GT**（推荐 MVP） | 低，~$0.001/episode × 10k = $10 | 中等，可能与机器人视角语义不一致 |
| 方案 2：用 LIBERO/DROID 的 task_description 拼现成的 sub-task tag 模拟摘要 | 极低 | 低，缺乏时序累积语义 |
| 方案 3：完全自监督——next-state token prediction 间接训摘要 | 中 | 不确定，可能学到无意义 token |

**推荐**：阶段 B 起步用方案 1，1 万条 episode × 8 个时间点 × 1 句摘要 = 8 万条 GT，~$80。

#### B.4 训练-推理分布偏移
论文核心难点。MVP 简化：
- 训练时 50% 概率用 GT 摘要、50% 用模型自己上一步生成的摘要（teacher forcing → student forcing 混合）。
- 不实现论文的"混合最优 + 失败 demo"——开源数据没有失败 demo。明确放弃此 ablation 项，文档里说明。

---

## 5. 评测方案

### 5.1 主评测 A：RoboMME（推荐先验证仓库可用，再用作主评测）

**简介**（来自调研，需亲自验证）：
- 论文：[arXiv:2603.04639](https://arxiv.org/abs/2603.04639)，2026-03。
- 仓库：https://github.com/RoboMME/robomme_benchmark（Apache 2.0）。
- **设计上正好对应 MEM 的 4 类记忆维度**：

| Suite | 测什么 | MEM 哪个组件主要负责 |
|---|---|---|
| **Counting Suite** | 重复操作 N 次后报数 | 长期文本摘要（计数无法靠视觉帧累积） |
| **Permanence Suite** | 物体被遮挡后回忆位置 | 短期视觉记忆 |
| **Reference Suite** | 跨时间识别物体身份 | 短期视觉 + 长期摘要联合 |
| **Imitation Suite** | 复现演示过的动作序列 | 短期视觉 |

**用法**：
1. clone 仓库 → 看 `inference_api/` 是否暴露标准 VLA 接口（observation in, action out）。
2. 写一个 adapter：openpi `serving/websocket_policy_server.py` ↔ RoboMME 评测 client。
3. 跑三组：(a) 单帧 baseline (history=1)，(b) MVP 阶段 A (history=6)，(c) 阶段 B 完整（如果做了）。
4. **对每个 Suite 单独报 success rate**——这是 MEM 真实有效性的最强信号。

**预期收益**（论文未公开 RoboMME 数字，按 MEM 论文趋势推断）：
- Permanence Suite：history=6 vs history=1 应有 **+15-30%**（物体遮挡场景）。
- Counting Suite：阶段 A 几乎无收益、阶段 B 才会显著提升 → **可作为 A 与 B 必要性的判别实验**。
- Reference Suite：阶段 A +10-15%，阶段 B 再 +10%。
- Imitation Suite：阶段 A +20%。

**风险**：仓库时间太新（2026-03），可能维护不完整。建议**评测开工前先派 1 天验证**：clone、跑通 README quickstart、确认 16 任务都能 reset & step。

### 5.2 主评测 B：RMBench（次选 / 对照）

**简介**：
- 论文：[arXiv:2603.01229](https://arxiv.org/abs/2603.01229)。
- 仓库：https://github.com/RoboTwin-Platform/RMBench。
- 9 个任务跨**记忆复杂度等级**，基于 RoboTwin 2.0。
- 自带 Mem-0 等参考实现 → **可以直接和它的 baseline 比对，不用自己跑 baseline**。

**用法**：
- 同样写 RoboTwin 2.0 ↔ openpi serving adapter。
- 报告 9 任务的成功率分级曲线（横轴：记忆复杂度，纵轴：success rate）。
- 主要价值：**对照其 Mem-0 baseline，验证 openpi+MEM 是否达到或超过显式 memory 模块**。

### 5.3 备用方案：自建 LIBERO Memory Probe（**强烈建议同时做，作为 sanity test**）

无论 RoboMME/RMBench 是否跑通，**都自建 2-3 个轻量 probe**，1 周可搭好：

| Probe | 设计 | 测什么 |
|---|---|---|
| **P1: 遮挡-取物** | LIBERO `pick_up_X`，episode 起始 0-2s 物体可见，3-5s 视野被一个大屏幕遮挡，6s 后机械臂启动取物 | 物体永存 (object permanence) |
| **P2: 多步指令记忆** | LIBERO-Long 改装，先口头说 "remember which drawer is open"，然后执行 30s 干扰任务，最后说 "now go to that drawer" | 长期跨任务记忆 |
| **P3: 计数任务** | "pick up 3 blocks one by one" → 模型必须知道已拿了几个 | 计数 / 长期符号记忆（仅阶段 B 才能做好） |

实现：复用 LIBERO env，改 task config / reset 逻辑，无需新仿真。

### 5.4 补充评测：MIKASA-Robo（成熟、稳妥）

- 论文：[arXiv:2502.10550](https://arxiv.org/abs/2502.10550)（ICLR 2025，**已知成熟**，不像 RoboMME 是 2026-03 新论文）。
- 仓库：https://github.com/CognitiveAISystems/MIKASA-Robo。
- 32 个桌面任务，专测 partial observability + memory capacity。
- 用法：直接当成单独的 evaluation suite 跑，挑 5-10 个最 memory-intensive 的任务报数即可。
- **价值**：作为 RoboMME / RMBench 出问题时的"保底评测"，时间成熟、社区已验证。

### 5.5 评测优先级建议

```
Day 0-1:   验证 RoboMME / RMBench 仓库能 clone + quickstart 跑通
Day 1-7:   搭 LIBERO 自建 P1/P2/P3 probe（无论上面结果如何）
Day 7-14:  跑 RoboMME 全 16 任务（baseline + MVP）
Day 14-21: 跑 RMBench + MIKASA-Robo 子集
```

**最终汇报指标**：
- LIBERO baseline（确保 MEM 不退化普通任务，回归保护）。
- LIBERO 自建 probe（P1-P3）：success rate baseline vs MEM。
- RoboMME 4 个 Suite 分别的成功率。
- RMBench 9 任务的复杂度曲线。
- MIKASA-Robo 5-10 任务子集（用作交叉验证）。

---

## 6. 数据需求

| 数据 | 阶段 A 需要 | 阶段 B 需要 | 来源 |
|---|---|---|---|
| LIBERO 全量 demo | ✅ | ✅ | 公开 |
| DROID 子集（5-10k ep） | ✅（可选） | ✅ | CC-BY 4.0 |
| OXE BridgeV2 / Fractal | 可选 | 可选 | 公开 |
| **摘要 GT 标注** | ❌ | ✅（用 GPT-4o 离线生成，~$80） | 自生成 |
| 失败 demo（论文方法） | ❌ | ❌（明确放弃） | — |

**结论**：阶段 A 完全使用现有公开数据；阶段 B 唯一新增的是 ~$80 的 LLM 摘要标注。**不需要 PI 量级私有数据**。

---

## 7. 算力估算

| 阶段 | 参数 | 显存（batch=1） | 推荐集群 | 训练步数 | H100-hr |
|---|---|---|---|---|---|
| 阶段 A（history=6） | ~3.5B（pi0.5 KI 起点） | ~70 GB | 16×H100 FSDP | 30-60k | 200-500 |
| 阶段 B（+ summary） | 同 | ~85 GB（长 prompt） | 16-32×H100 | 50-100k | 800-1500 |
| 评测推理 | — | 单卡 H100 即可 | 1-4×H100 | — | 真机不需要 |

**总 budget**：阶段 A+B+评测全部跑通，~3000 H100-hr 内。

---

## 8. 验证策略（端到端）

每个阶段必须满足：

1. **forward 单元测试**：history=1 退化等价 openpi 现行为（数值精度内）。
2. **零初始化退化测试**：TemporalBlock 零初始化时（训练前），history=6 输出与 history=1 数值等价（separable attention 的恒等映射性质）。
3. **梯度路径测试**：history=6 时 6 帧的视觉编码梯度都能反传到 SigLIP；temporal block 参数也接收非零梯度。
4. **算子等价测试**：自实现的 separable attention 在玩具数据上与 full attention 在 T=1 时数值等价；T>1 时计算复杂度比例符合 O(T·N²+T²·N) vs O(T²·N²)。
5. **小数据 overfit 测试**：单 batch 训 1k step 能 overfit（loss → 0）。
6. **LIBERO 回归保护**：MEM 训完，普通 LIBERO success rate **不退化超过 -2%**。
7. **memory probe**：P1-P3 上 baseline vs MEM 必须有显著正向 gap（否则 MEM 实现有问题）。
8. **RoboMME 4-suite 报数**：作为对外发布的核心数字。

---

## 9. 风险与待定

1. **RoboMME / RMBench 仓库可用性**——2026-03 新论文，仓库可能不成熟。**第一周必须验证**，不能假定可用。
2. **训练显存**：6 帧 × 3 cam 视觉 token 量级膨胀，需提前测显存上限，可能要降到 history=4 或减少 cam 数。
3. **LeRobot delta_timestamps 是否覆盖所有训练数据**：若 LIBERO/DROID 默认配置没暴露此接口，需新写 history loader。
4. **阶段 B 摘要 GT 噪声**：GPT-4o 生成的摘要可能与机器人视角语义不齐，需做小规模人审。
5. **history=6 是否够**：论文后训练扩到 18 帧。如果 RoboMME Counting Suite 里要求 >5s 历史，可能需要先把 history 调到 12-16。

---

## 10. 实施时间线

```
Week 1:  实现 temporal_attention.py + 改 Observation / pi0.py / pi0_config
         单元测试（forward 等价 / 零初始化退化 / 算子等价）
         同步验证 RoboMME / RMBench 仓库可用
Week 2:  改 transforms / policies (LiberoInputs / DroidInputs)
         自建 LIBERO probe (P1/P2/P3) 搭好
         小数据 overfit + 显存压测
Week 3:  阶段 A 训练（LIBERO + DROID 子集，30-60k step）
Week 4:  阶段 A 评测：LIBERO 回归 + 自建 probe + RoboMME / RMBench / MIKASA-Robo
         发布"短期视觉 MEM" 阶段性 ckpt + blog
Week 5-6: 阶段 B 实现（generate_summary + 摘要监督） + GPT-4o 摘要 GT 生成
Week 7-8: 阶段 B 训练 + 评测
Week 9:   联合 ablation 报告（A only / B only / A+B），写技术报告
```

---

## 11. 关键参考资料

- MEM 论文：[arXiv:2603.03596](https://arxiv.org/html/2603.03596) / [Mem.pdf](https://www.pi.website/download/Mem.pdf)
- π0.7 论文：[arXiv:2604.15483](https://arxiv.org/abs/2604.15483) / [blog](https://www.pi.website/blog/pi07)
- RoboMME：https://github.com/RoboMME/robomme_benchmark
- RMBench：https://github.com/RoboTwin-Platform/RMBench
- MIKASA-Robo：https://github.com/CognitiveAISystems/MIKASA-Robo
- MemoryVLA（社区相关 VLA 记忆工作）：https://shihao1895.github.io/MemoryVLA/
- openpi 关键文件：
  - [src/openpi/models/pi0.py](src/openpi/models/pi0.py)（embed_prefix 在 106-137）
  - [src/openpi/models/model.py](src/openpi/models/model.py)（Observation 在 83-130）
  - [src/openpi/models/pi0_config.py](src/openpi/models/pi0_config.py)
  - [src/openpi/transforms.py](src/openpi/transforms.py)
  - [src/openpi/policies/libero_policy.py](src/openpi/policies/libero_policy.py)