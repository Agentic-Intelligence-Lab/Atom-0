# π0.7 第三步：Diverse Context Conditioning + Dropout 复现方案

## Context

我们正在基于 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) **逐模块**复现 π0.7（[arXiv:2604.15483](https://arxiv.org/abs/2604.15483)）。**第一步 KI**（[arXiv:2505.23705](https://arxiv.org/abs/2505.23705)）和**第二步 MEM** 已完成；本方案聚焦**第三步：Diversifying the Prompt + per-component Dropout**——π0.7 论文 §V 的核心训练改造，是 pi0.7 在多模态 prompt（subtask / subgoal image / metadata / control mode）下实现 steerability 与混杂数据鲁棒性的关键。

**约束**：
- 没有 PI 私有 web-scale 数据；只用 LIBERO + 可选 DROID 子集 + GPT-4 辅助标注。
- **不做大规模"预训练"**——在已有 KI+MEM 起点 ckpt 上做 fine-tune 即可（这一节本质是输入分布改造，不是新知识注入，§1.2 决策依据）。
- 不依赖 World Model 训练（用 trajectory GT 未来帧作为 subgoal，与论文 §VI-C 做法一致）。
- 算力 >8 卡（H100/A100），不是瓶颈。

**目标**：在 openpi 上实现 §V 的多组件 prompt + per-component dropout，**严格证明** pi0.7 的 steerability claim（speed/strategy 可控）+ 混杂数据鲁棒性 claim，给社区第一份开源参考实现。

---

## 1. π0.7 §V 核心要点（精炼）

### 1.1 五种 prompt 组件

pi0.7 把 context $\mathcal{C}_t$ 拆成五段，全部以 token 形式拼进 VLM 的 prefix：

| 组件 | 符号 | 内容 | 来源 |
|---|---|---|---|
| 总任务指令 | $\ell_t$ | "Make me a sandwich" | 数据集自带 |
| 子任务指令 | $\hat{\ell}_t$ | "open the fridge door" | 高层策略 / coaching 标注 |
| 子目标图像 | $\mathbf{g}_t$ | 多视角未来帧（4s 后场景） | 训练用 GT 真实帧；推理用 World Model |
| Episode metadata | $m$ | speed / quality(1–5) / mistake labels | 自动统计 + 人工标注 |
| Control mode | $c$ | "joint" / "end-effector" | 数据集元信息 |

### 1.2 Per-component Dropout 配方（§V-E）

各组件**独立、不同概率**地随机丢掉：

| 组件 | 训练时保留率 | 备注 |
|---|---|---|
| 子目标图像 $\mathbf{g}$ | **25%** 样本含 subgoal | 论文："action prediction essentially becomes an inverse dynamics problem" 当有 subgoal 时 |
| 子任务 $\hat{\ell}$ | 当 subgoal 在场时 70% 保留（30% drop） | 防止模型退化成"只看图忽略文字" |
| Metadata $m$ | 整体 85% 保留；保留时每个子字段再独立 95% 保留 | **嵌套 dropout** |
| Control mode $c$ | **不做 dropout** | 否则歧义太大 |
| 语言指令 $\ell$ | 不 drop | base 信号 |

### 1.3 与 KI/MEM 的关系

**这一阶段不是独立训练阶段**——它是数据 transform 层 + prompt 构造方式的改造。同一个 flow-matching + KI loss 不变，只是 prefix 内容更丰富、训练时分布更随机。论文把它和 §VI 的混杂数据训练**合在一起**作为 pi0.7 主训练配方。

→ **本方案在 KI+MEM 起点 ckpt 上加 30k step fine-tune 即可。**

### 1.4 核心 claim

| Claim | 论文实证 |
|---|---|
| **Steerability**：metadata.speed → 真实速度可控；$\hat{\ell}$ → 行为路径可控 | §V, §VIII |
| **Mixed-quality data robustness**：metadata 让模型能区分干净 vs 脏 demo | §IX-E, Figure 18 左 |
| **Diversity ingestion**：dropout 让模型能消化高度异质数据 | §IX-E, Figure 18 右 |
| **Subgoal accelerates training**：含 subgoal 的样本 loss 下降快（inverse dynamics 结构） | §V-B |

---

## 2. 社区现状

- **没有任何 open-source 复现** pi0.7 的 diverse context + per-component dropout。
- openpi `pi05_base` 仅支持单一 prompt（`tokenized_prompt`），无 metadata / subgoal image / 多组件 dropout。
- 相关可借鉴：classifier-free guidance（CFG）的 conditioning dropout 已在 diffusion 文献中通用，但**多组件独立 dropout + 嵌套 dropout** 在 VLA 领域是 pi0.7 首次系统化。

→ 本方案产出将是**社区第一份开源 diverse context + per-component dropout 参考实现**。

---

## 3. openpi 改造点（已定位）

### 3.1 模块差异表

| 模块 | 当前状态 | 需改动 |
|---|---|---|
| `Pi0Config` | [src/openpi/models/pi0_config.py](src/openpi/models/pi0_config.py) | 新增 `use_subgoal_image / use_metadata / use_control_mode / dropout_rates: dict` 字段 |
| `Observation` | [src/openpi/models/model.py](src/openpi/models/model.py) | 新增可选字段 `subgoal_images / subgoal_image_masks / metadata_tokens / control_mode_tokens / subtask_tokens` |
| `embed_prefix` | [src/openpi/models/pi0.py:106-137](src/openpi/models/pi0.py#L106) | 在现有 prefix 后追加：subgoal-image SigLIP tokens（带 type embedding）+ subtask tokens + metadata tokens + control_mode token |
| `compute_loss` | [src/openpi/models/pi0.py:189-214](src/openpi/models/pi0.py#L189) | **不动**——dropout 在 transform 层做完了，forward 路径透明 |
| `sample_actions` | [src/openpi/models/pi0.py:217-279](src/openpi/models/pi0.py#L217) | **不动**（推理时 prompt 由用户/上层提供） |
| `LiberoInputs` | [src/openpi/policies/libero_policy.py:29-83](src/openpi/policies/libero_policy.py#L29) | 新增 `subgoal_image / metadata / control_mode / subtask` 字段透传 |
| `transforms.py` | [src/openpi/transforms.py](src/openpi/transforms.py) | 新增 `BuildDiverseContext`（拼 prompt 字符串/token） + `DropoutContext`（独立按概率 mask 各组件） |
| 配置注册 | [src/openpi/training/config.py](src/openpi/training/config.py) | 新增 `pi05_div_libero` named config（KI+MEM 起点 + diverse context + dropout） |
| 数据准备脚本 | 新建 `scripts/prepare_libero_subtasks.py` 等 | LIBERO 子任务标注脚本（GPT-4 + 人 review） |

### 3.2 Prefix 拼接顺序（关键）

KI+MEM 现有 prefix（自上而下）：

```
[history images (T frames × cam)] [state token] [language prompt tokens]
```

新版 prefix（diverse context）：

```
[history images (MEM)]
[state token]
[<SUBGOAL_BEGIN>] [subgoal images (cams) | <empty>] [<SUBGOAL_END>]
[<META_BEGIN>] [speed token] [quality token] [mistake tokens | <empty>] [<META_END>]
[<CTRL_BEGIN>] [control mode token] [<CTRL_END>]
[<TASK_BEGIN>] [language prompt tokens] [<TASK_END>]
[<SUBTASK_BEGIN>] [subtask tokens | <empty>] [<SUBTASK_END>]
```

**约束**：
- **prefix 长度恒定**——所有 dropout 字段被 drop 时用 `<empty>` sentinel token 占位（不变长，避免 batching 复杂化和 nn.scan 重编译）。
- **type embedding**：subgoal image 走和 obs image 一样的 SigLIP，但**追加一段可学习 type embedding**（`type_id ∈ {obs, subgoal}`）以区分时序角色。论文 §V-B 提到这点。
- **metadata 词表**：约束在受限离散词表内（`speed ∈ {fast, normal, slow}`、`quality ∈ {1, 2, 3, 4, 5}`、`mistake ∈ 预定义短语集合`），用 PaliGemma tokenizer 直接 tokenize 即可，无需新词。

### 3.3 Dropout 在 transform 层做的优势

- **forward 路径透明**：模型代码完全不知道 dropout 存在，回归保护强（forward 与 KI+MEM 阶段位级一致，当所有 dropout=0）。
- **便于 ablation**：dropout_rates 是配置项，B 系列 ablation 改一个 dict 即可。
- **可缓存**：subtask 标注、metadata 等字段一次性 tokenize 后写入 LeRobot dataset cache，每 epoch 只 sample dropout mask。

---

## 4. 数据构建：在 LIBERO 上没有 web-scale 怎么办

**结论**：**完全不需要 web-scale 预训练**——这一节本质是**输入分布改造**而非知识注入。所需新标注少，多数字段从 trajectory 自动派生。

### 4.1 五个组件在 LIBERO 上的来源

| 组件 | LIBERO 上怎么拿 | 工作量 |
|---|---|---|
| 总任务 $\ell$ | LIBERO 自带 task description | 0 |
| 子任务 $\hat{\ell}$ | **新标注**：GPT-4 / Gemini 切段写子任务（参考 LIBERO sub-instruction 工作；或人工切 keyframe） | 中：130 任务 × 5 demo × ~3 子任务 ≈ 2k 条标注，~1 GPU 天 + 0.5 人天 review |
| Subgoal 图像 $\mathbf{g}$ | **训练用 trajectory[t + Δ] 真实帧**（论文 §VI-C 做法） | 0（纯 transform） |
| Metadata $m$ | LIBERO demo 大多 expert+success，metadata 信息量低。建议：<br>• `quality`：固定 5（demo 都是 expert）<br>• `speed`：从 action norm 自动分箱成 fast/normal/slow<br>• `mistake_labels`：跳过 / 全 0<br>• 真正想验证 metadata 价值需要再搭配 noisy demos / RL rollout（见 §6.4） | 低 |
| Control mode $c$ | LIBERO 是 EEF 控制，标 `"end-effector"` | 0 |

### 4.2 是否需要大规模预训练

**不需要**。理由：

1. 论文这一节的核心是"学会**忽略**缺失组件、并在有组件时**用上**它"——是表示层 routing 能力，**小数据上一样能验证**。
2. 起点 ckpt（KI+MEM 训出来）已具备 VLA 能力；只需让模型适应"prefix 多模态、字段可缺"这个新输入分布。
3. PI 用 web-scale 数据是为了让 metadata 这种语义在自然语境下被理解；LIBERO 用受限词表当离散 token 处理就够。

### 4.3 起点 ckpt

**KI+MEM 阶段的 ckpt**（你已有）。**不要从 PaliGemma 直接起**，否则上一阶段的工作浪费了。

---

## 5. 是否需要世界模型

**这一阶段不需要**。明确分两件事：

| 用途 | 需要 World Model 吗 |
|---|---|
| **训练** subgoal-conditioned VLA | ❌ 直接用 trajectory 里 t+Δ 的真实帧。论文 §VI-C 自己也是真实+生成混合，且训练时大头是真实帧 |
| **推理时**没有 GT 未来帧，要现场生成 subgoal | ✅ 需要 World Model（BAGEL 14B），但**这是下一阶段** |

**Dropout 设计的妙处**：subgoal 缺席本身就是训练分布的一部分（75% 样本无 subgoal）→ 推理时即便 World Model 没接上，模型也能 fallback 到无 subgoal 模式正常工作。这给了我们**把 World Model 推迟到独立 milestone**的合法性。

---

## 6. 推荐实施路线

### 阶段 A：核心 diverse context + dropout 实现（**优先做，2-3 周**）

#### A.1 设计选择

- **Prefix 拼接顺序**：固定如 §3.2，sentinel token 占位保证恒定长度。
- **Dropout 在 transform 层**：`DropoutContext` 转换器按 config dict 独立采样 mask。
- **Subgoal 选取策略**：trajectory 里采 t + Δ 帧，Δ 从 [2s, 6s] 均匀采样（贴近论文 4s）。如果 t + Δ 超出 episode，clip 到末帧。
- **Metadata 词表**：`speed ∈ {fast, normal, slow}`（按 action L2 norm 三分位自动标）；`quality ∈ {1..5}`（LIBERO demo 全标 5；noisy 实验时按规则降级）。
- **不引入新参数**：subgoal type embedding 是新参数，但**单独初始化为 0** 保证起点等价 KI+MEM ckpt。

#### A.2 改动清单（按文件）

```
src/openpi/models/pi0_config.py
│     新增字段：
│       use_subgoal_image: bool = False
│       use_metadata: bool = False
│       use_control_mode: bool = False
│       dropout_rates: dict = field(default_factory=dict)
│         例如 {"subgoal": 0.75, "subtask": 0.30, "metadata_all": 0.15,
│                "metadata_field": 0.05, "control_mode": 0.0}
│       subgoal_horizon_sec: tuple[float, float] = (2.0, 6.0)
│
src/openpi/models/model.py:Observation
│     新增可选字段：
│       subgoal_images: dict[str, jax.Array] | None
│       subgoal_image_masks: dict[str, jax.Array] | None
│       subtask_tokens / subtask_mask
│       metadata_tokens / metadata_mask
│       control_mode_tokens / control_mode_mask
│
src/openpi/models/pi0.py:embed_prefix (line 106-137)
│     按 §3.2 顺序追加各段 embedding；mask=False 段用 sentinel embedding（一个可学习
│     的 <empty> token，单独初始化为 0）
│     subgoal image：复用 self.PaliGemma.img + 一段新可学习 type_emb（init=0）
│
src/openpi/transforms.py
│     新增 BuildDiverseContext: dict → 各组件字符串/张量
│     新增 DropoutContext: 按 dropout_rates 独立 sample mask
│       - subgoal 整体：75% 概率丢弃（mask=False）
│       - subtask 在 subgoal 在场时 30% 丢弃
│       - metadata 整体 15% 丢弃；保留时每个子字段再 5% 独立丢弃
│       - control_mode：保留率 100%
│
src/openpi/policies/libero_policy.py:LiberoInputs
│     新增字段透传 subgoal_image / metadata / control_mode / subtask
│
src/openpi/training/config.py
│     新增 named configs:
│       pi05_div_libero        (起点 KI+MEM ckpt, 全开)
│       pi05_div_libero_no_meta (ablation: 关 metadata)
│       pi05_div_libero_no_sg   (ablation: 关 subgoal)
│       pi05_div_libero_no_drop (ablation: 关所有 dropout)
│
scripts/prepare_libero_subtasks.py (新)
│     用 GPT-4 / Gemini 对 LIBERO trajectory 生成子任务序列；写入 LeRobot
│     dataset 的 episode-level 字段缓存
│
scripts/prepare_libero_metadata.py (新)
│     从 actions 自动统计 speed 分箱；quality 默认 5；mistake 默认空
```

#### A.3 训练数据

| 数据 | 用途 | 来源 |
|---|---|---|
| **LIBERO 全量**（130 任务 + Long） | 主训练源 + 主 eval | 公开 |
| **LIBERO subtask 标注** | $\hat{\ell}$ 字段 | GPT-4 辅助标注，0.5 人天 review |
| **LIBERO metadata 缓存** | $m$ 字段 | 自动生成 |
| **DROID 子集 5k ep**（可选） | 增加 control_mode 多样性 | 公开 |

**起点 ckpt**：你的 KI+MEM 阶段 ckpt（必须）。

**训练步数**：30k step，bf16，FSDP，~16 卡 H100，~2 天。

#### A.4 关键风险

- **子任务标注质量**：GPT-4 切段可能粗糙，需要采样人工 review。建议先标 30 个任务做 sanity，再批量。
- **subgoal type embedding 初始化为非零**会破坏起点 ckpt 兼容；**必须 init=0**。
- **metadata 词表如果用自然语言**（如 "fast"），与 LIBERO 训练数据中的语言指令可能歧义；建议用**特殊 token**（如 `<speed_fast>`）添加到 PaliGemma vocab。
- **dropout mask 必须在 batch 内 per-sample 独立**，不要 jit 时被错误广播——这是常见 bug。

---

### 阶段 B：完整论文 ablation（**优先级中，2 周**）

#### B.1 设计选择

| 实验 | 配置 | 验证什么 |
|---|---|---|
| **B.1** baseline | KI+MEM 不动，单 prompt | 现有水平 |
| **B.2** 全 prompt 无 dropout | 全 5 组件给，不 dropout | 训练 success 高，**推理鲁棒性差**（缺字段就崩） |
| **B.3** 全 prompt + dropout（论文配方） | 阶段 A 完整版 | 训练略低于 B.2，**推理任意字段子集鲁棒** |
| **B.4** no-metadata | B.3 但去掉 metadata 字段 | 复刻 Figure 18 左：含 noisy demo 时 B.3 > B.4 |
| **B.5** no-subgoal | B.3 但 subgoal 永远空 | 验证 §V-B "inverse dynamics 加速收敛" |
| **B.6** dropout 率扫描 | subgoal dropout ∈ {0.5, 0.75, 0.9} | 找最优 |

#### B.2 实现要点

- 所有 ablation 共享同一份 `pi0_config.py` 字段，改 dict 即可切换。
- log per-component dropout 实际命中率到 wandb，方便诊断。

---

## 7. 验证矩阵（每项有明确通过条件）

### 7.1 DC-V1 — Dropout 统计单元测试（必须先过，0.5 天）

跑 1k batch，统计每个组件实际丢弃率，与配置 ±2% 内一致。

实现位置：新增 `src/openpi/transforms_dc_test.py`。

### 7.2 DC-V2 — Forward 等价性回归（必须先过，几小时）

设所有 dropout=0、所有组件传 `<empty>`、subgoal type_emb 为 0 → forward 路径数值与 KI+MEM 阶段**完全一致**（fp 精度内）。保护新代码不破坏 baseline。

### 7.3 DC-V3 — Prefix 长度恒定测试（0.5 天）

任意 dropout 组合下 prefix 序列长度严格恒定（sentinel 占位生效）。否则 nn.scan 编译会重复触发，训练吞吐崩溃。

### 7.4 DC-V4 — Steerability 测试（核心学术 claim，~30 GPU-hr）

在阶段 A 训完的 ckpt 上设计三种可控性测试：

| 测试 | 设计 | 通过条件 |
|---|---|---|
| **Speed 控制** | 固定任务，metadata.speed ∈ {fast, normal, slow}，量化 action L2 norm | 三种条件下 norm 单调变化，p<0.05 |
| **Strategy 控制** | LIBERO-Goal 多解任务，喂不同 $\hat{\ell}$ 看执行路径 | 不同子任务下成功率均 >50%，且执行路径明显不同 |
| **Subgoal 控制** | 手动给一张目标场景图，看动作是否朝向该状态 | subgoal-conditioned vs unconditioned 的 trajectory 分布显著差异 |

### 7.5 DC-V5 — 混杂数据鲁棒性（最值得发的实验，~50 GPU-hr）

LIBERO demo 太干净，需**人工注入低质量 demo** 模拟论文 §IX-E：

- 取 30% LIBERO demo，加噪声/抖动/打错抓取等"mistake"，标 `quality=2 mistake_labels=["dropped object"]`。
- 训练 B.3（含 metadata）vs B.4（无 metadata）。
- **期望**：B.3 在 hold-out clean demo 上 success rate ≥ baseline；B.4 显著下降——**证明 metadata 让模型能区分干净/脏数据**。

→ 这是阶段性**最值得发的实验**：在公开数据上首次复现 metadata 的 mixed-quality data robustness。

### 7.6 DC-V6 — Subgoal 加速收敛（~20 GPU-hr）

训练曲线对比 B.3（含 subgoal）vs B.5（无 subgoal）：

- **期望**：B.3 在前 10k step flow loss 下降明显更快（论文 §V-B "inverse dynamics" claim）。
- 长期 success rate 可能持平或 B.3 略高。

### 7.7 DC-V7 — 推理时缺字段鲁棒性（~10 GPU-hr）

阶段 A ckpt，推理时**任意删除一个组件**测 success rate：

- 期望：B.3 在缺 metadata / 缺 subgoal / 缺 subtask 各情况下 success rate 衰减 < 10%；B.2（无 dropout 训）衰减 > 30%。
- → 直接证明 dropout 训练带来的输入鲁棒性。

---

## 8. 数据需求

| 数据 | 阶段 A 需要 | 阶段 B 需要 | 来源 | 标注成本 |
|---|---|---|---|---|
| LIBERO 全量 demo | ✅ | ✅ | 公开 | 0 |
| LIBERO 子任务标注 | ✅ | ✅ | GPT-4 + 人 review | 0.5 人天 + ~$50 API |
| LIBERO metadata 缓存 | ✅ | ✅ | 脚本自动 | 0 |
| LIBERO noisy demo 注入 | ❌ | ✅（DC-V5） | 脚本生成 | 0 |
| DROID 子集 5k ep | 可选 | ✅ | CC-BY 4.0 | 0 |
| **新人工标注** | ❌（GPT-4 替代） | ❌ | — | **基本无需** |

**结论**：阶段 A+B 全程使用现有公开数据 + GPT-4 辅助标注，**几乎无标注成本**，**完全不需要 PI 量级私有数据**。

---

## 9. 算力估算

| 阶段 | 参数 | 显存（batch=1） | 推荐集群 | 训练步数 | H100-hr |
|---|---|---|---|---|---|
| DC-V1/V2/V3 单元测试 | 3.5B | <40 GB | 单卡 H100 | — | ≤5 |
| 阶段 A 主训练（B.3） | 3.5B | ~60 GB（subgoal 多图 + 多组件） | 16×H100 FSDP | 30k | ~150-300 |
| DC-V4 steerability 评测 | 3.5B inference only | <40 GB | 单卡 | — | ~30 |
| DC-V5 混杂数据（B.3 vs B.4） | 3.5B | ~60 GB | 16×H100 | 30k × 2 | ~400 |
| DC-V6 subgoal 加速（B.3 vs B.5） | 3.5B | ~60 GB | 16×H100 | 10k × 2（前期对比足够） | ~100 |
| 阶段 B 完整 ablation（B.1-B.6 共 6 条） | 3.5B | ~60 GB | 16×H100 | 30k × 6 | ~1000 |

**总 budget**：阶段 A ~300 H100-hr；阶段 A+B 完整 ~1500 H100-hr，比 KI 阶段（~2000）小。

---

## 10. 验证策略（端到端）

每次 PR / 每个阶段必须满足：

1. **DC-V1 dropout 统计**：实际命中率与配置一致。
2. **DC-V2 forward 等价**：所有 dropout=0、type_emb=0 时 forward 与 KI+MEM 位级一致。
3. **DC-V3 prefix 长度恒定**：任意 mask 组合下序列长度不变。
4. **小数据 overfit 测试**：单 batch 训 1k step，flow_loss → 0。
5. **回归保护**：dropout=0 时训练 30k step loss 曲线与 KI+MEM 阶段重合。
6. **DC-V4 steerability**：speed/strategy/subgoal 三测均显著。
7. **DC-V5 混杂数据**：B.3 ≥ baseline，B.4 显著差。
8. **DC-V7 推理鲁棒性**：缺字段 success rate 衰减 < 10%。

---

## 11. 风险与待定

1. **GPT-4 子任务标注的对齐质量**：模型给的子任务可能与 trajectory 实际语义错位（"opening door" 标到了 "approaching door" 的帧上）。预案：用一个固定模板（"based on frames [t1, t2], what subtask is being performed?"）+ 人工 review 30 任务做 sanity，再批量。
2. **Subgoal type embedding 初始化**：必须严格 init=0，否则起点 ckpt 加载后 forward 数值就偏。建议在 ckpt-load hook 里硬置零。
3. **prefix 长度膨胀**：MEM 已让 prefix 较长（多帧 obs），再加 subgoal images（多 cam）+ metadata + subtask，**显存和 attention 计算 O(N²) 都会膨胀**。需 gradient checkpointing + 可能降 batch size。
4. **Metadata 词表的 token 化**：如果当作普通自然语言 token，可能与 LIBERO prompt 中的形容词冲突（"fast" 在 prompt 中也出现）；建议加特殊 token 到 vocab，或者用 PaliGemma 已有的 unused token 占位（更简单）。
5. **LIBERO 上 metadata 信号弱**：所有 demo 都是 expert+success，metadata 几乎无变化 → DC-V5 必须自己注入 noisy demo 才能验证 claim。
6. **BuildDiverseContext 与 transform pipeline 顺序**：需要在 ImageTransform / Tokenize 之前完成 prefix 字段构造；transform list 顺序需仔细 review。
7. **Dropout 与 jit/scan 兼容**：dropout mask 是 per-sample 的随机量；transform 在 dataloader 端运行（CPU），不入 jit 编译，但**rng 必须正确传**（否则每 epoch mask 完全相同）。
8. **PyTorch 路径同步**：JAX 改完后 PyTorch 路径要重写一遍（[src/openpi/models_pytorch/pi0_pytorch.py](src/openpi/models_pytorch/pi0_pytorch.py)）；建议阶段 A 仅 JAX，PyTorch 滞后一周作为后续 PR。

---

## 12. 实施时间线

```
Week 1: 数据侧
  - LIBERO 子任务标注脚本（GPT-4）+ 30 任务 sanity review
  - metadata 自动生成脚本（speed 分箱）
  - subgoal frame 缓存到 LeRobot dataset 字段
Week 2: 模型侧
  - Pi0Config 新字段 + Observation 字段 + embed_prefix 改造
  - BuildDiverseContext / DropoutContext transforms
  - DC-V1/V2/V3 单元测试通过
  - 小数据 overfit + 显存压测
Week 3: 阶段 A 主训练（B.3）
  - 30k step on LIBERO（KI+MEM 起点）
  - 同步开发 PyTorch 路径
Week 4: 阶段 A 评测
  - DC-V4 steerability 测试套件
  - DC-V7 推理鲁棒性测试
  - 发布"diverse context"阶段性 ckpt + 简短 blog
Week 5-6: 阶段 B ablation
  - DC-V5 混杂数据实验（noisy demo 注入 + B.3 vs B.4）
  - DC-V6 subgoal 加速对比（B.3 vs B.5）
  - B.6 dropout 率扫描
  - 完整 ablation 报告
Week 7: PyTorch 路径完整 PR + 上游回馈
```

**Week 4 末为阶段性可交付里程碑**：能发布社区第一份开源 diverse-context-conditioning 训练实现 + steerability demo。
**Week 6 末为完整 ablation 里程碑**：可写技术报告。

---

## 13. 关键参考资料

**论文**
- π0.7 论文：[arXiv:2604.15483](https://arxiv.org/abs/2604.15483)（特别是 §V "Diversifying the Prompt"、§VI 训练细节、§IX-E 混杂数据 ablation）
- π0.7 blog：[pi.website/blog/pi07](https://www.pi.website/blog/pi07)
- π0.7 PDF：[pi.website/download/pi07.pdf](https://www.pi.website/download/pi07.pdf)
- KI 论文：[arXiv:2505.23705](https://arxiv.org/abs/2505.23705)
- pi0.5 论文：[arXiv:2504.16054](https://arxiv.org/abs/2504.16054)

**openpi 关键文件**
- [src/openpi/models/pi0.py](src/openpi/models/pi0.py)（embed_prefix 在 106-137；compute_loss 在 189-214；sample_actions 在 217-279）
- [src/openpi/models/pi0_config.py](src/openpi/models/pi0_config.py)
- [src/openpi/models/model.py](src/openpi/models/model.py)（Observation dataclass）
- [src/openpi/transforms.py](src/openpi/transforms.py)
- [src/openpi/policies/libero_policy.py](src/openpi/policies/libero_policy.py)
- [src/openpi/training/config.py](src/openpi/training/config.py)

**前置阶段 plan**
- 第一步 KI：[openpi-pi0-7-step1-knowledge-insulation.md](openpi-pi0-7-step1-knowledge-insulation.md)
- 第二步 MEM：（参考 Mem 论文 + MEM 阶段产出 ckpt）

**根 plan**
- [pi0-7-pi0-5-pi0-6-pi-0-6-pi0-7-pi0-7-cozy-minsky.md](pi0-7-pi0-5-pi0-6-pi-0-6-pi0-7-pi0-7-cozy-minsky.md)
- [pi07_highlevel_subtask_design.md](../../Research%20Exploration/Foundation%20Model/openpi/docs/pi07_highlevel_subtask_design.md)