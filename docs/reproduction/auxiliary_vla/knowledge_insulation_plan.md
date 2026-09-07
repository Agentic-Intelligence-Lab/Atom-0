# π0.7 第一步：KI（Knowledge Insulation）复现方案

## Context

我们正在 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) 基础上**逐模块**复现 π0.7（[arXiv:2604.15483](https://arxiv.org/abs/2604.15483)）。**第一步聚焦 KI**（Knowledge Insulation, [arXiv:2505.23705](https://arxiv.org/abs/2505.23705)，NeurIPS 2025）——这是 π0.5/0.6/0.7 共有的训练时机制，也是 openpi 已发布权重 `pi05_base` 实际使用的训练方法、但**官方未开源训练代码**（[issue #735](https://github.com/Physical-Intelligence/openpi/issues/735)）。

**约束**：
- 没有 PI 预训练量级私有数据，仅依赖开源数据集（DROID / LIBERO / OXE 子集）。
- 复现和验证**不能太复杂**——先 MVP（FAST 副 loss + stop_gradient），再做完整版（VQA 探针、多源 mix）。
- 算力 >8 卡（H100/A100），不是瓶颈。
- **本方案不做大规模"预训练"**——直接在 openpi 现有 fine-tune loop 上加 KI loss，参考 §1.2 决策依据。

**目标**：在 openpi 上加上 KI 训练机制，并在公开数据上**严格证明** KI 的三大 claim（train fast / run fast / generalize better）+ "VLM 表示不被破坏"，给社区一份首个开源的 KI 训练参考实现。

---

## 1. KI 论文核心要点（精炼）

### 1.1 KI 解决的问题 + 机制

| 维度 | 论文设计 |
|---|---|
| 问题 | action expert 的 flow-matching 梯度回传 → 污染 VLM backbone 的预训练表示 → web 知识 catastrophic forgetting → 泛化能力下降 |
| 机制 1：副信号 | 训练时在 VLM 端额外预测 **FAST 离散动作 token**（[π0-FAST](https://arxiv.org/abs/2501.09747) 的 tokenizer），加 next-token-prediction CE loss |
| 机制 2：梯度阻断 | action expert 经 cross-attention 与 VLM 交互时，**梯度不回传** VLM 的 K/V 投影 |
| 三大 claim | **train fast**：FAST 副信号让 VLM 早期表示更稳；**run fast**：推理仍用连续动作（FAST 路径关闭）；**generalize better**：VLM 知识保留 → 新场景泛化 |
| **关键事实** | KI 是**训练时**机制，**推理时是 no-op**——sample_actions 路径完全不变 |
| 论文实证 | 训练吞吐 vs 无 KI ↑；VQA 探针保留 ≥90%；LIBERO 长任务泛化 +X% |

### 1.2 关键事实（与本方案有关）

- **不引入新参数**：KI 既不加新模块也不改架构，是训练 loss 和反向梯度路径的修改 → ckpt 加载与现有 `pi05_base` 完全兼容。
- **`pi05_base` 是 KI 训出来的**——但 PI **没有开源训练代码**。所以"加载 `pi05_base` 跑推理"**不能**验证你的 KI 实现正确（推理完全等价 baseline，见 §3.3 关键澄清）。
- **不需要 PI 量级数据**：KI 现象在小规模、单一数据源上**反而更显著**（数据越少 VLM 表示崩塌越快、KI 救得越多）。论文 ablation 也是小对照设置上跑的。
- **FAST tokenizer 已在 openpi 中**：[`tokenizer.py:51`](src/openpi/models/tokenizer.py#L51) 的 `FASTTokenizer` 直接可复用，无需移植。
- **next-token CE loss 已在 pi0_fast 中实现**：[`pi0_fast.py:198-233`](src/openpi/models/pi0_fast.py#L198-L233)，可作为 KI 副 loss 的参考实现复用。

---

## 2. 社区现状

- **没有任何 KI 训练流程的开源复现**（截至 2026-04），即使 [qrafty-ai/pi-openpi](https://github.com/qrafty-ai/pi-openpi)、[exla-ai/openpie-0.6](https://huggingface.co/exla-ai/openpie-0.6) 这两个声称做了 RECAP 的项目也没碰 KI（它们直接基于 `pi05_base` 起步）。
- 相关可借鉴工作：
  - [allenzren/open-pi-zero](https://github.com/allenzren/open-pi-zero)：PyTorch pi0 reimpl，有 freeze VLM 选项但非 KI（freeze ≠ insulate，freeze 完全切断 forward 也切了，KI 只切反向）。
  - [lucidrains/pi-zero-pytorch](https://github.com/lucidrains/pi-zero-pytorch)：单文件参考，无 KI。
- **openpi 自身**：`pi0.py:compute_loss` 仅有 flow-matching MSE，无 FAST 副 loss、无 stop_gradient。

→ 本方案产出将是**社区第一份开源 KI 训练实现**，价值高。

---

## 3. openpi 改造点（已定位）

### 3.1 模块差异表

| 模块 | 当前状态 | 需改动 |
|---|---|---|
| `Pi0Config` | [src/openpi/models/pi0_config.py:18-48](src/openpi/models/pi0_config.py#L18-L48) | 新增 `ki_enabled`、`ki_alpha`、`ki_fast_max_len` 三个字段 |
| `compute_loss` | [src/openpi/models/pi0.py:188-214](src/openpi/models/pi0.py#L188-L214)，仅 flow-matching MSE | 加 KI 副 loss 分支（next-token CE on FAST tokens），返回字典 `{"flow": ..., "ki_fast": ...}` |
| `embed_prefix` | [src/openpi/models/pi0.py:105-137](src/openpi/models/pi0.py#L105-L137) | KI 模式下在 prefix 末尾追加 FAST action tokens（teacher-forcing 输入） |
| `gemma.Attention` | [src/openpi/models/gemma.py:158-249](src/openpi/models/gemma.py#L158-L249)，两 expert q/k/v concat 后做 cross attention | 拆 attention 为两段，给 action-expert query 看的 VLM K/V 加 `stop_gradient` |
| `gemma.Block` | [src/openpi/models/gemma.py:284-340](src/openpi/models/gemma.py#L284-L340) | 透传 `ki_insulate: bool` 到 Attention |
| `gemma.Module.__call__` | [src/openpi/models/gemma.py:388-411](src/openpi/models/gemma.py#L388-L411) | 接受新 kwarg `ki_insulate`，scan 时透传 |
| `FASTTokenizer` | [src/openpi/models/tokenizer.py:51-139](src/openpi/models/tokenizer.py#L51-L139)，已实现 | ✅ 直接复用 |
| `TokenizeFASTInputs` transform | [src/openpi/transforms.py:269-288](src/openpi/transforms.py#L269-L288)，已实现 | ✅ 直接复用 |
| 训练入口 | [scripts/train.py:146-191](scripts/train.py#L146-L191) | 接受字典形式的 loss，按 `ki_alpha` 加权 |
| 配置注册 | [src/openpi/training/config.py](src/openpi/training/config.py) | 新增 `pi05_ki_libero`、`pi05_ki_droid` 两个 named config |
| Sampling 路径 | [src/openpi/models/pi0.py:217-279](src/openpi/models/pi0.py#L217-L279) | **不动**（KI 推理 no-op，回归保护必须） |
| PyTorch 路径 | [src/openpi/models_pytorch/pi0_pytorch.py](src/openpi/models_pytorch/pi0_pytorch.py)、[gemma_pytorch.py](src/openpi/models_pytorch/gemma_pytorch.py) | 同步实现（M1 优先 JAX，PyTorch 滞后一周） |

### 3.2 Pi0 双 expert 架构与梯度通道（关键技术背景）

[`gemma.py`](src/openpi/models/gemma.py) 的 `Module` 是双 expert 设计：`configs: Sequence[Config]` 包含 PaliGemma + action expert，对应 `embedded: Sequence[...]` 即 `[prefix, suffix]`。两个 expert 在每层 `Block` 内**共享 attention**（cross-expert KV 沿 axis=1 concat，[gemma.py:201](src/openpi/models/gemma.py#L201)）但有独立的 q/k/v projection 和 FFN 权重。

这意味着 cross attention 是**梯度泄露的关键通道**：每一层中 action expert 的 query 与 VLM 的 K/V 做 attention，反向传播时 action loss 会通过 dL/dK_vlm、dL/dV_vlm 的链路污染 VLM 参数。**KI 的 stop_gradient 必须在每一层 attention 内部注入**，而不是在外层切断。

### 3.3 关键澄清：KI 是训练时机制

**只看 forward，KI 版 pi0.5 和无 KI 版 pi0.5 是同一份代码**。差异只在：
- 训练时：多算一个 FAST 副 loss；反向梯度被切。
- 推理时：完全等价。

→ "用 `pi05_base` ckpt 加载并跑 LIBERO 推理"**只能验证 forward 路径实现正确**（属于 M0 sanity test），**不能**验证 KI 训练 trick 本身。KI 必须从训练时的梯度和 loss 行为去验证（见 §5）。

---

## 4. 推荐实施路线

### 阶段 A：核心 KI 训练机制（**优先做，2-3 周**）

#### A.1 设计选择

- **副 loss 类型**：next-token CE，复用 `pi0_fast.py:compute_loss` 的实现思路（line 209-233）。
- **副 loss 权重 α**：默认 `ki_alpha=1.0`（论文默认）。提供配置项允许调小做 ablation。
- **stop_gradient 注入点**：在每层 Attention 内，区分 VLM query 段和 action expert query 段，单独计算 attention（详见 A.1.1）。
- **保持 sample_actions 不变**：纯训练时机制，inference 完全 no-op。
- **保持 ckpt 兼容**：不引入新参数，可加载现有 `pi05_base` 起训。

#### A.1.1 Stop-Gradient 在 Attention 中的实现细节

**当前 [Attention.__call__](src/openpi/models/gemma.py#L164)**（精简）：

```python
qkvs = []
for i, (x, config) in enumerate(zip(xs, self.configs)):
    # 每个 expert 计算自己的 q, k, v
    qkvs.append((q_i, k_i, v_i))

# 沿 sequence 维度拼接
q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs))

# 共享 attention：每个 query 都能 attend 全部 K/V
logits = einsum("BTKGH,BSKH->BKGTS", q, k)
probs = softmax(masked_logits)
encoded = einsum("BKGTS,BSKH->BTKGH", probs, v)
```

**KI 模式改造（伪代码）**：

```python
if ki_insulate and len(qkvs) == 2 and all(x is not None for x in xs):
    q_vlm, k_vlm, v_vlm = qkvs[0]
    q_act, k_act, v_act = qkvs[1]

    sg = jax.lax.stop_gradient

    # VLM query 看完整 KV，但 action 端的 KV 反传被切（避免 action loss 污染 VLM）
    k_for_vlm = jnp.concatenate([k_vlm, sg(k_act)], axis=1)
    v_for_vlm = jnp.concatenate([v_vlm, sg(v_act)], axis=1)

    # action query 看完整 KV，但 VLM 端的 KV 反传被切（KI 的核心 insulation）
    k_for_act = jnp.concatenate([sg(k_vlm), k_act], axis=1)
    v_for_act = jnp.concatenate([sg(v_vlm), v_act], axis=1)

    # 分两段计算 attention（mask 也要分两段切片）
    attn_vlm = compute_attention(q_vlm, k_for_vlm, v_for_vlm, mask[:, :len_vlm, :])
    attn_act = compute_attention(q_act, k_for_act, v_for_act, mask[:, len_vlm:, :])

    encoded = jnp.concatenate([attn_vlm, attn_act], axis=1)
else:
    # 原始路径
    ...
```

**为什么要双向 stop_gradient**：
- `sg(k_vlm), sg(v_vlm)` 用于 action query → 切断 action loss 反传到 VLM K/V。这是 KI 的主要意图。
- `sg(k_act), sg(v_act)` 用于 VLM query → 切断 FAST 副 loss 反传到 action expert 参数（次要，但保证两个 expert 真正独立训练，符合论文"insulation"含义）。

**实现路径**：
- 修改 [gemma.py:158](src/openpi/models/gemma.py#L158) 的 `Attention.__call__`，加 `ki_insulate: bool = False` 参数。
- [gemma.py:284 `Block`](src/openpi/models/gemma.py#L284)、[gemma.py:340 `Module`](src/openpi/models/gemma.py#L340) 透传该参数。
- nn.scan 的 `static_argnums` 需要把新参数加入静态参数列表（[gemma.py:362](src/openpi/models/gemma.py#L362)）。

**为什么不在外层做**：双 expert 是 layer-wise 共享 attention 的，每层都会 cross attend，外层做不到细粒度切断。**必须在 Attention 内部**。

**初始化等价性 / 回归保护**：当 `ki_insulate=False` 时所有路径走原 concat 分支，行为与现有 openpi 完全一致——这是回归测试的基础。

#### A.1.2 FAST 副 loss 的 forward 路径

**输入端**（修改 [pi0.py:106 embed_prefix](src/openpi/models/pi0.py#L106)）：

KI 模式下的 prefix 由三段组成（顺序）：
1. 图像 token（per cam，SigLIP 编码后）
2. 任务 prompt + state token（pi0.5 现有逻辑）
3. **新增：FAST action tokens**（teacher-forcing 输入，对应当前 batch 的 GT actions）

第 3 段使用 `FASTTokenizer.tokenize(prompt, state, actions)`（[tokenizer.py:67](src/openpi/models/tokenizer.py#L67)）已经能产出 `tokens / token_mask / ar_mask / loss_mask`——直接复用。

**关键 token mask 约定**：
- `loss_mask`：仅在 FAST action token 位置为 True，prefix 段为 False（[tokenizer.py:96](src/openpi/models/tokenizer.py#L96)）。
- `ar_mask`：FAST tokens 段为 1（causal），其他为 0（双向）。

**输出端**（修改 [pi0.py:188 compute_loss](src/openpi/models/pi0.py#L188)）：

```python
# 现有 flow-matching 路径（保留）
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

if not self.config.ki_enabled:
    return flow_loss

# KI 副 loss：next-token CE on FAST tokens, 仅 VLM 输出端
# prefix_out 是 VLM expert 输出（行 209-211）
# 在 FAST tokens 位置取 logits（用 LLM 的 unembed 即 self.PaliGemma.llm.module.embedder.decode）
fast_logits = self.PaliGemma.llm(pre_logits=prefix_out[:, fast_token_slice])  # 复用 pi0_fast 的接口
targets = obs.tokenized_prompt_for_ki[:, 1:]  # next-token shift
ce = -jnp.sum(one_hot(targets) * log_softmax(fast_logits), axis=-1)
ki_loss = jnp.sum(ce * obs.token_loss_mask[:, 1:]) / jnp.sum(obs.token_loss_mask[:, 1:])

# 训练循环按 ki_alpha 加权（在 train.py 而非这里加权，便于 ablation logging）
return {"flow": flow_loss, "ki_fast": ki_loss}
```

**注意**：当前 `pi0.py` 的 prefix 不含 FAST tokens，所以 `prefix_tokens` 序列长度会变长——`max_token_len` 需要相应增大（pi05 默认 200，KI 模式建议 ≥ 256，对应 `Pi0FASTConfig` 默认 250）。

#### A.2 改动清单（按文件）

```
src/openpi/models/
├── pi0_config.py:18-48
│     新增字段：
│       ki_enabled: bool = False
│       ki_alpha: float = 1.0
│       ki_fast_max_len: int = 256        # KI 模式下 max_token_len 上限
│     __post_init__ 中：当 ki_enabled=True 时把 max_token_len bump 到 ≥256
│
├── gemma.py
│   ├── Attention (line 158-249)
│   │     新参数 ki_insulate: bool = False
│   │     当 True 且双 expert 全在场时走 §A.1.1 的双 stop_gradient 分支
│   ├── Block (line 284-)
│   │     透传 ki_insulate
│   ├── Module.__call__ (line 388-411)
│   │     新 kwarg ki_insulate；scan 把它加入 broadcast 静态参数
│   └── nn.scan static_argnums 调整（line 362）
│
├── pi0.py
│   ├── __init__ (line 67-103)
│   │     缓存 self.ki_enabled / self.ki_alpha / 暴露 unembed
│   ├── embed_prefix (line 105-137)
│   │     KI 模式：把 obs.tokenized_prompt 中 FAST tokens 段也 embed 进 prefix
│   │     已被 FASTTokenizer 在 transform 端拼好，无需在 forward 重新拼接
│   ├── compute_loss (line 188-214)
│   │     增 KI 副 loss 分支，返回字典 {"flow": ..., "ki_fast": ...}
│   │     使用 prefix_out 在 FAST token 位置算 next-token CE
│   └── sample_actions (line 217-279)
│         **不动**（回归保护）
│
src/openpi/transforms.py
│     已有 TokenizeFASTInputs (line 269-288) 可直接复用
│     新增极薄 wrapper KITokenize：内部调用 FASTTokenizer 但保留 PaliGemmaTokenizer
│     的 200-token prefix 与 FAST 段拼接（论文格式）
│
src/openpi/models/model.py:83-130
│     Observation dataclass 加可选字段 token_ar_mask、token_loss_mask
│     （pi0_fast.py 已用过这两个字段，复用即可）
│
scripts/train.py:146-191
│     train_step 接受 dict 形式 loss
│     总 loss = flow + ki_alpha * ki_fast
│     wandb 分别 log 两个分项 + 它们的梯度范数
│
src/openpi/training/config.py
│     新增 named configs：
│       pi05_ki_libero (起点 PaliGemma; ki_enabled=True; LIBERO 全量)
│       pi05_ki_droid  (起点 PaliGemma; ki_enabled=True; DROID 子集)
│       pi05_no_ki_libero (对照组，KI 关闭)
│       pi05_no_ki_droid  (对照组)
│
src/openpi/models_pytorch/pi0_pytorch.py + gemma_pytorch.py
│     M1 阶段后期同步实现（晚 JAX 一周）
```

#### A.3 训练数据

| 数据 | 用途 | 来源 |
|---|---|---|
| **LIBERO 全量**（130 任务 + Long 子集） | 主训练源 + LIBERO eval | 公开免费 |
| **DROID 子集**（5-10k episode） | 真实场景多样性 | CC-BY 4.0 |
| **VQA-v2 子集**（200 样本） | KI-V4 VLM 知识探针 | 公开 |

**起点 ckpt**：**PaliGemma**（不要从 `pi05_base` 起，否则 ckpt 已完成 KI 训练，对照失效——这是 §3.3 的逻辑必要要求）。

**训练步数**：
- 单条对照 30k step，bf16，FSDP，~16 卡 H100。
- A vs B（无 KI vs KI）总训练量 ~60k step，2-3 天。

#### A.4 关键风险

- **stop_gradient 拆分 attention 的实现**易错：mask 切片、双向 sg 顺序、KV 拼接顺序都可能写错。**必须先用 §5 的 KI-V1 单元测试卡住**。
- **nn.scan 静态参数**：把 `ki_insulate` 加入 `static_argnums` 后，scan 编译可能出错；必要时改成 `length` 维度外的标志而非 layer-wise 条件分支。
- **`max_token_len` 增大** → 显存膨胀：从 200 → 256 prefix 长度增加 28%，跨模态 attention 是 O(N²) → 全局 ~60% 显存增加。需开 gradient checkpointing。
- **FAST tokenizer 调用慢**：`AutoProcessor.from_pretrained("physical-intelligence/fast")` 在每个 dataloader worker 都要 init 一次，建议预先 tokenize 一遍存盘。

---

### 阶段 B：完整论文 ablation（**优先级中，1-2 周**）

#### B.1 设计选择

阶段 A 完成后，用同一套训练 + 评测代码做完整 ablation：

| 实验 | 配置 | 验证什么 |
|---|---|---|
| **B.1** 纯 flow-matching baseline | `ki_enabled=False` | 回归基线 |
| **B.2** 仅 FAST 副 loss（无 stop_gradient） | `ki_enabled=True; ki_insulate=False`（新增独立开关） | 副信号本身的贡献 |
| **B.3** 仅 stop_gradient（无 FAST 副 loss） | `ki_alpha=0; ki_insulate=True` | insulation 本身的贡献 |
| **B.4** 完整 KI | `ki_enabled=True; ki_insulate=True; ki_alpha=1.0` | 论文方法 |
| **B.5** α 扫描 | `ki_alpha ∈ {0.1, 0.5, 1.0, 2.0}` | 副 loss 强度敏感性 |

**B.2 / B.3 是论文未做的 ablation**——拆开两个机制单独看贡献，可作为本项目的开源增量贡献。

#### B.2 实现要点

- 新增 `Pi0Config.ki_insulate: bool = True`（默认随 `ki_enabled` 联动），单独可关。
- 训练循环加分项 metric：`grad_norm_vlm_from_flow_loss`、`grad_norm_vlm_from_ki_loss`，确认 stop_gradient 真的工作（即 KI 模式下前者应为 0）。

---

## 5. 验证矩阵（KI 实现正确性，详见根 plan §14）

每个验证项有明确通过条件：

### 5.1 KI-V1 — 梯度路径单元测试（**必须先过**，1 天）

构造 toy batch（B=2, 任意 LIBERO sample），跑一次 `compute_loss + jax.grad`，断言：

| 测试 | 通过条件 |
|---|---|
| **5.1.1**：仅 flow-matching loss 反传，KI 模式开 | VLM backbone 参数梯度 `g_vlm` **全为 0**（stop_gradient 切断成功） |
| **5.1.2**：仅 FAST 副 loss 反传，KI 模式开 | `g_vlm` **非 0**（副信号确实进入 VLM） |
| **5.1.3**：仅 flow-matching loss 反传，KI 模式开 | action expert 参数梯度**非 0**（flow loss 仍正常更新 action 端） |
| **5.1.4**：仅 FAST 副 loss 反传，KI 模式开 | action expert 参数梯度**全为 0**（副 loss 不污染 action 端） |
| **5.1.5**：KI 模式关 | 行为与原 openpi 一致（数值精度内） |

实现位置：新增 `src/openpi/models/pi0_ki_test.py`。

**这是验证 KI 实现正确性的最强、最便宜手段**——直接断言两个机制都工作。

### 5.2 KI-V2 — α=0 退化等价性（回归保护，几小时）

设 `ki_alpha=0`、`ki_insulate=False`，跑 1k step 训练。loss 曲线**与原 openpi `pi0.py` 训练完全一致**（fp 精度内）。保护新代码不破坏 baseline。

### 5.3 KI-V3 — 训练动力学对照（≤100 GPU-hr）

LIBERO 全量上从 PaliGemma 起，跑两条线 30k step：

- **A**：纯 flow-matching（无 KI）
- **B**：flow-matching + 完整 KI

对照指标（log 到 wandb）：

| 指标 | 期望 |
|---|---|
| 训练初期 VLM 表示稳定性（每 1k step freeze VLM 跑 VQA-v2 200 样本） | A 快速崩坏；B 保留 |
| LIBERO success rate（hold-out task suite） | B 显著优于 A，长任务/新场景尤甚 |
| 相同 GPU-hours 下达到同 flow loss 的 step 数 | B 更少（论文 "train fast"） |

### 5.4 KI-V4 — VLM 知识保留度（最有学术价值，~50 GPU-hr）

- 训练前在 MMMU / MMBench / VQA-v2 跑 PaliGemma 评测，记基线。
- A、B 各 30k step 训练后再跑同一套 VLM 评测。
- **期望**：A 大幅下降（catastrophic forgetting）；B 保留 ≥90%。

这是最直接量化"insulation"的实证，**也是论文的核心 ablation**。

### 5.5 KI-V5 — `pi05_base` ckpt 加载推理（forward sanity，无训练）

加载官方 `pi05_base` ckpt 跑 LIBERO eval。期望 success rate **复现 openpi README 数字（90%+）**。

⚠️ **此项与 KI 无关**——它验证的是 forward / 模型类 / sampling 实现是否与官方一致（属 M0 sanity test）。但必须先过，否则 KI-V3/V4 forward 数值可能从一开始就错。

---

## 6. 数据需求

| 数据 | 阶段 A 需要 | 阶段 B 需要 | 来源 | 标注 |
|---|---|---|---|---|
| LIBERO 全量 demo | ✅ | ✅ | 公开 | 自带 task prompt |
| DROID 子集（5-10k ep） | ✅ | ✅ | CC-BY 4.0 | 自带 |
| VQA-v2 子集（200-2000 样本） | ✅（KI-V3/V4） | ✅ | 公开 | 自带 |
| MMMU / MMBench | 可选（KI-V4） | ✅ | 公开 | 自带 |
| FAST tokenizer 预 tokenize 缓存 | ✅（性能） | ✅ | 用 `physical-intelligence/fast` 离线生成 | 自动 |
| **新标注** | ❌ | ❌ | — | **完全无需新标注** |

**结论**：阶段 A+B 全程使用现有公开数据，**完全不需要标注成本**。**不需要 PI 量级私有数据**。

---

## 7. 算力估算

| 阶段 | 参数 | 显存（batch=1） | 推荐集群 | 训练步数 | H100-hr |
|---|---|---|---|---|---|
| KI-V1 单元测试 | 3.5B | <40 GB | 单卡 H100 | — | ≤5 |
| KI-V2 α=0 回归 | 3.5B | ~50 GB | 8×H100 | 1k | ~10 |
| KI-V3 A vs B 对照 | 3.5B | ~55 GB（FAST tokens 拉长 prefix） | 16×H100 FSDP | 30k × 2 | 200-500 |
| KI-V4 VLM 探针评测 | 3.5B forward only | <40 GB | 单卡 | — | ~20 |
| 阶段 B 完整 ablation（B.1-B.5，5 条线） | 3.5B | ~55 GB | 16×H100 | 30k × 5 | 800-1500 |

**总 budget**：阶段 A 全部跑通 ~500 H100-hr；阶段 A+B 完整 ablation ~2000 H100-hr 内，远小于 MEM 阶段的预算。

---

## 8. 验证策略（端到端）

每次 PR / 每个阶段必须满足：

1. **forward 单元测试**（`pi0_ki_test.py`）：`ki_enabled=False` 时 forward 数值与原 openpi 完全一致。
2. **KI-V1 梯度路径单元测试**：5.1.1-5.1.5 全部通过。
3. **算子等价测试**：`ki_insulate=True` 时 forward 数值与 `ki_insulate=False` 完全一致（stop_gradient 不影响 forward）。
4. **小数据 overfit 测试**：单 batch 训 1k step，flow_loss 和 ki_fast_loss 都能 → 0。
5. **`pi05_base` 加载 sanity**（KI-V5）：forward 路径复现 LIBERO 数字。
6. **训练对照**（KI-V3）：B 在 hold-out 任务上显著优于 A。
7. **VLM 探针**（KI-V4）：B 保留 ≥90% PaliGemma 原 VQA 能力。
8. **回归保护**：KI 模式开但 `ki_alpha=0`、`ki_insulate=False` 时，30k step 训练 loss 曲线与 baseline 重合。

---

## 9. 风险与待定

1. **stop_gradient attention 拆分**的实现复杂度——nn.scan + 双段 attention 可能与现有 `nn.remat` checkpoint policy 冲突。预案：先用非 scan 版本的 Attention 实现 KI，性能优化后置。
2. **`max_token_len` 增大**带来的显存膨胀（200 → 256+），需开 gradient checkpointing；可能需要降 batch size。
3. **`pi05_base` ckpt 起点 vs PaliGemma 起点的对照设计**：必须从 PaliGemma 起，否则 ckpt 已 KI 过；但 PaliGemma → 30k step 在 LIBERO 上可能不足以完全收敛，**需要先做 baseline 收敛性 sanity**。
4. **FAST tokenizer 性能**：每 worker 每次 tokenize 较慢，必须预先离线缓存到 LeRobot dataset 字段。
5. **VLM 探针所需 `embedder.decode` 接口**：当前 `gemma.py` 通过 `Embedder.decode` 接口（[gemma.py:153](src/openpi/models/gemma.py#L153)）支持 logits 输出，但 pi0.py 路径未走过这条路；需要确认 decode 路径在 `pi05` 配置（adaRMSNorm + adarms_cond）下正确工作。
6. **NNX bridge 的 stop_gradient**：openpi 用 `nnx_bridge.ToNNX` wrap Linen Gemma，`jax.lax.stop_gradient` 在该桥接层下行为需要测试（可能要在 Linen 内部插入而非外层）。
7. **PyTorch 路径同步**：JAX 改完后 PyTorch 路径要重写一遍 stop_gradient 逻辑（PyTorch 的 `.detach()`），易遗漏。建议 M1 仅对 JAX 路径完整支持，PyTorch 滞后一周作为后续 PR。

---

## 10. 实施时间线

```
Week 1:  实现 gemma.py Attention 的 ki_insulate 分支 + Pi0Config 新字段
         单元测试：KI-V1 五项断言全部通过；ki_insulate=False 时 forward 数值等价
         pi0.py compute_loss 双 loss 字典输出
Week 2:  改 transforms 让 FAST tokens 进入 prefix；改 train.py 接受 dict loss
         小数据 overfit + 显存压测；KI-V2 α=0 退化等价测试
         FAST tokenizer 预缓存脚本（写到 LeRobot dataset 缓存字段）
Week 3:  阶段 A KI-V3 对照训练（A vs B 各 30k step on LIBERO）
         同步开发 PyTorch 路径
Week 4:  KI-V3 评测 + KI-V4 VLM 探针（VQA-v2 / MMMU / MMBench）
         发布"KI 训练" 阶段性 ckpt + blog（首份开源 KI 训练实现）
Week 5-6: 阶段 B 完整 ablation（B.1-B.5 共 5 条线，含 α 扫描）
         联合 ablation 报告，技术报告草稿（拆分 stop_gradient 与副 loss 贡献）
Week 7:  PyTorch 路径完整 PR + 上游回馈（向 openpi 主仓提 PR，对应 issue #735）
```

**Week 4 末为阶段性可交付里程碑**：能够发布社区第一份开源 KI 训练实现 + 可复现的 ablation 数字。

---

## 11. 关键参考资料

**论文**
- KI 论文：[arXiv:2505.23705](https://arxiv.org/abs/2505.23705)（NeurIPS 2025）
- KI research page：[pi.website/research/knowledge_insulation](https://www.pi.website/research/knowledge_insulation)
- pi0 论文：[arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- pi0-FAST 论文：[arXiv:2501.09747](https://arxiv.org/abs/2501.09747)（FAST tokenizer 设计）
- pi0.5 论文：[arXiv:2504.16054](https://arxiv.org/abs/2504.16054)
- pi0.7 论文：[arXiv:2604.15483](https://arxiv.org/abs/2604.15483)

**openpi 关键文件**
- [src/openpi/models/pi0.py](src/openpi/models/pi0.py)（compute_loss 在 188-214；embed_prefix 在 105-137；sample_actions 在 217-279）
- [src/openpi/models/pi0_config.py](src/openpi/models/pi0_config.py)
- [src/openpi/models/gemma.py](src/openpi/models/gemma.py)（Attention 在 158-249；Block 在 284；Module 在 340）
- [src/openpi/models/pi0_fast.py](src/openpi/models/pi0_fast.py)（FAST CE loss 实现在 198-233，可参考）
- [src/openpi/models/tokenizer.py](src/openpi/models/tokenizer.py)（FASTTokenizer 在 51-139）
- [src/openpi/transforms.py](src/openpi/transforms.py)（TokenizeFASTInputs 在 269-288）
- [scripts/train.py](scripts/train.py)（train_step 在 146-191）

**相关 issue**
- [openpi #735（请求开源不含 KI 的 pi0.5 权重）](https://github.com/Physical-Intelligence/openpi/issues/735)——本方案的直接受众与回应

**相关复现工作**（用于借鉴 / 对比）
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)（基线）
- [allenzren/open-pi-zero](https://github.com/allenzren/open-pi-zero)（freeze VLM 选项，非 KI）
- [lucidrains/pi-zero-pytorch](https://github.com/lucidrains/pi-zero-pytorch)（参考实现）

**根 plan**
- [pi0-7-pi0-5-pi0-6-pi-0-6-pi0-7-pi0-7-cozy-minsky.md](pi0-7-pi0-5-pi0-6-pi-0-6-pi0-7-pi0-7-cozy-minsky.md) §14（KI 验证矩阵）+ §15（关于"openpi 不含预训练代码"的应对策略）