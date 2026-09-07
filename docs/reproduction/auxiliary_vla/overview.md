# π0.7 复现整体方案（基于 openpi）

> 汇报版整体方案。整合此前对 5 个子模块（KI / MEM / Diverse Context Conditioning / World Model / High-Level Policy）的独立详细方案。
>
> 子模块详细方案：
> - [第一步 KI](openpi-pi0-7-step1-knowledge-insulation.md)
> - [第二步 MEM](openpi-pi0-7-pi0-5-knowledge-insulation-luminous-locket.md)
> - [第三步 Diverse Context + Dropout](openpi-pi0-7-step3-diverse-context-dropout.md)
> - [第四步 World Model (BAGEL)](openpi-pi0-7-ki-mem-dicerse-context-con-proud-reef.md)
> - [第五步 High-Level Policy](openpi-pi0-7-ki-mem-dicerse-context-con-zany-floyd.md)
> - [初始整体调研](pi0-7-pi0-5-pi0-6-pi-0-6-pi0-7-pi0-7-cozy-minsky.md)

---

## 0. Context（为什么做这件事）

我们的目标是在 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)（已开源 π₀ / π₀-FAST / π₀.₅）的基础上，**逐模块**复现 **π₀.₇**（[arXiv:2604.15483](https://arxiv.org/abs/2604.15483)），为社区提供一份高性能、可训练、可推理、可验证的开源 codebase。

**项目分工与约束**：
- **本团队**负责代码实现与每个模块的有效性验证。**数据准备**由另一个小团队负责。
- 不进行 web-scale 预训练；所有模块的有效性都基于在 `pi05_base` 上用**公开仿真数据**（LIBERO、RoboTwin、必要时 DROID 子集）做 fine-tune + 在公开 benchmark 上评估。
- 主代码路径是 **openpi 的 JAX 代码**（`src/openpi/models/`），仅在不得已（BAGEL 世界模型）时使用 PyTorch 子系统。
- π₀.₆ 的 RECAP 不在本期复现范围（专注 π₀.₇ 的 KI / MEM / DCC / WM / HL 五件套 + RTC）。
- backbone 升级（Gemma3 4B + 860M action expert）放在所有子模块验证完之后再做（理由见 §5）。

**关于 KI 的一个关键决策（影响整个项目优先级）**：

`pi05_base` **本身就是 KI 训出来的**（详见 §1.2）。这意味着我们 fine-tune 它做 M3-M6 时，VLM 已经被"绝缘保护"过。KI 的最大价值在 **pretraining 阶段**防止 VLM 知识被 action 梯度污染——而我们不做 pretraining。因此本项目对 KI 复现采取**分层策略**：

| 层 | 内容 | 是否做 |
|---|---|---|
| L1 | KI 机制实现（`stop_gradient` + 副 loss）+ 单元测试（KI-V1/V2） | ✅ M1 必做 |
| L2 | 开源代码（首份 KI 训练代码，回应 [issue #735](https://github.com/Physical-Intelligence/openpi/issues/735)） | ✅ M1 必做 |
| L3 | 复现 KI 论文 claim 的大型对照实验（KI-V3 train fast / KI-V4 VLM retention，从 PaliGemma 起 30k step × 2 条线） | ⏸️ **延后到 M7**（M7 必须从 PaliGemma 起重训，是天然窗口） |

→ M3-M6 fine-tune 时**默认开启 KI 开关**（cost 接近 0），但**不专门验证 KI claim**。节省 ~500-2000 H100-hr 给后续模块。

---

## 1. openpi 现状 vs π₀.₅ 论文 vs π₀.₇ 新增

### 1.1 openpi 当前 codebase 有什么

- **JAX 主路径**（`src/openpi/models/`）：`pi0.py`（flow-matching VLA）、`pi0_fast.py`（FAST 离散动作 + AR）、`gemma.py`（双 expert 共享 attention）、`siglip.py`、`lora.py`、`tokenizer.py`（含 `FASTTokenizer`）、`vit.py`。配套 `pi0_test.py` / `model_test.py`。
- **PyTorch 副路径**（`src/openpi/models_pytorch/`）：仅 `pi0_pytorch.py` (~462 行) + `gemma_pytorch.py` (~280 行)，是 JAX 路径的子集（无 π₀-FAST、无 LoRA、无 EMA、FSDP 仅单节点、几乎无测试）。
- **训练入口**：`scripts/train.py`（JAX，FSDP）、`scripts/train_pytorch.py`（单节点）。
- **Policies / transforms**：[`libero_policy.py`](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/policies/libero_policy.py)、[`droid_policy.py`](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/policies/droid_policy.py)、[`transforms.py`](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/transforms.py)。
- **已发布权重**：`pi05_base`（KI 训出来的）、`pi05_libero`、`pi0_fast` 等。

### 1.2 与 π₀.₅ 论文的关键出入

**时间线还原**：

| 时间 | 事件 |
|---|---|
| 2025-04 | pi0.5 论文（[arXiv:2504.16054](https://arxiv.org/abs/2504.16054)）发布。论文方法 = 多机器人 + web 数据 + 子任务标注的异质 co-training + flow-matching MSE only + 高层子任务路由 |
| 2025-05 | KI 论文（[arXiv:2505.23705](https://arxiv.org/abs/2505.23705)，NeurIPS 2025）发布。本质是 pi0.5 训练方法的**改进版**：在原 flow-matching loss 基础上加 FAST token 副 loss + `stop_gradient` |
| 2025-09 | openpi 发布 `pi05_base` 权重——**用的是 KI 版方法训出来的**，不是 pi0.5 原 paper 方法 |

openpi README 直接明示："the π₀.₅ model... an upgraded version of π₀ with better open-world generalization **trained with knowledge insulation**"；issue #735 标题就是请求 "KI-free pi05 weights"。两者交叉验证。

**因此实际现状是**：

| 东西 | 现状 |
|---|---|
| pi0.5 论文原版方法（无 KI）训出来的权重 | ❌ **从未公开**（issue #735 在请求） |
| KI 方法训出来的权重 (= `pi05_base`) | ✅ openpi 已发布 |
| **KI 训练流程的代码** | ❌ **未开源**（这是我们 M1 要补的） |
| pi0.5 论文里的"异质 co-training 数据 pipeline" | ❌ 未开源 |
| pi0.5 论文里的"高层子任务推理" | ❌ 未开源（[issue #664](https://github.com/Physical-Intelligence/openpi/issues/664)） |

→ **`pi05_base` ≠ pi0.5 paper weights**。它是"用 KI 方法 + PI 私有数据训出来、forward 架构与 pi0.5 论文一致"的 ckpt。这点对 §0 末尾的 KI 决策有直接含义：因为 `pi05_base` 已经过 KI 训练，所以从它起步做 fine-tune 时 KI 的边际价值大幅下降；KI 的真正用武之地是 M7（重新预训练）。

### 1.3 π₀.₇ 相对 π₀.₅/0.6 新增的能力（本项目的复现 scope）

| 能力 | 论文出处 | openpi 现状 | 本项目要做 |
|---|---|---|---|
| **KI (Knowledge Insulation) 训练流程** | [arXiv:2505.23705](https://arxiv.org/abs/2505.23705) | ❌ 仅有权重 | ✅ M1 |
| **RTC (Real-Time Chunking) 推理** | [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) | ❌（同步阻塞） | ✅ M2（独立轨道） |
| **MEM 多尺度记忆**（短期视觉 + 长期文本摘要） | [arXiv:2603.03596](https://arxiv.org/html/2603.03596) | ❌（窗口=1） | ✅ M3 |
| **Diverse Context Conditioning + per-component Dropout** | π0.7 §V | ❌ | ✅ M4 |
| **BAGEL 世界模型生成视觉子目标** | π0.7 §V-B + Appendix C | ❌ | ✅ M5（PyTorch） |
| **同架构高层策略（自动子任务）** | π0.7 §IV / Algorithm 1 | ❌ | ✅ M6 |
| **Gemma 3 4B + 860M action expert** | π0.6 model card | ❌ | ⏸️ M7（推迟） |
| 多节点训练 | — | ⚠️ 单节点 | ✅ M0 时打通 |

**π₀.₇ 总参数估算**：4B Gemma3 + 400M SigLIP + 860M expert ≈ 5B（不含 14B BAGEL 与同架构高层策略副本）。

---

## 2. 已有开源工作盘点（保留分析，决定哪些可直接复用）

### 2.1 可直接复用的开源资产

| 资产 | License | 用途 |
|---|---|---|
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | Apache 2.0 | **基线**；所有改动在此之上 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot)（含 RTC PR [#1698](https://github.com/huggingface/lerobot/pull/1698)） | Apache 2.0 | **唯一开源含 RTC 推理**实现，M2 移植参考 |
| [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) | Apache 2.0 | M5 世界模型权重 |
| FAST tokenizer（`physical-intelligence/fast`） | Apache 2.0 | M1 KI 副 loss 直接用 |
| LeRobot `delta_timestamps` 接口 | Apache 2.0 | M3 多帧加载零成本 |
| [peft](https://github.com/huggingface/peft) | Apache 2.0 | M5 BAGEL LoRA |

### 2.2 已有的 π0 系列复现工作（仅供参考，不作为基线）

| 项目 | 范围 | 评估 |
|---|---|---|
| [allenzren/open-pi-zero](https://github.com/allenzren/open-pi-zero) | π₀ PyTorch fine-tune；含 freeze VLM 选项（**非 KI**） | 单数据集 fine-tune 参考；多图历史未实现 |
| [lucidrains/pi-zero-pytorch](https://github.com/lucidrains/pi-zero-pytorch) | π₀ 单文件 PyTorch reimpl | 概念参考 |
| [qrafty-ai/pi-openpi](https://github.com/qrafty-ai/pi-openpi) | 声称含 π₀.₆ + RECAP | 1★/0fork，未验证；不依赖 |
| [exla-ai/openpie-0.6](https://huggingface.co/exla-ai/openpie-0.6) | 声称 π0.6 + RECAP | 仅单仿真任务，只发权重不发代码；概念验证级 |
| [RLinf/RLinf](https://github.com/RLinf/RLinf) | RECAP / DSRL / IQL on π0/π0.5 | 3.2k★，**唯一有分量的 RL 后训练框架**；本期不用（不复现 RECAP），以后可对接 |
| [MemoryVLA](https://shihao1895.github.io/MemoryVLA/) / ReMem-VLA | VLA + memory | M3 设计参考，非直接 fork |
| [RoboMME](https://github.com/RoboMME/robomme_benchmark) / [RMBench](https://github.com/RoboTwin-Platform/RMBench) / [MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo) | 记忆 benchmark | M3 评估 |

### 2.3 π0.7 复现的空白

**截至 2026-04，全网无任何 π0.7 复现**。本项目五个模块产出的均为社区首份开源参考实现。

---

## 3. JAX vs PyTorch 策略

**默认主路径：JAX**，仅以下情况使用 PyTorch：

| 模块 | 语言 | 理由 |
|---|---|---|
| M1 KI / M2 RTC / M3 MEM / M4 DCC / M6 HL / M7 backbone 升级 | **JAX** | 全部是 policy 内部改动，与官方 `pi05_base` ckpt 1:1 对齐 |
| M5 World Model (BAGEL) | **PyTorch** 独立服务 | BAGEL 只有 PyTorch 权重和 modeling 代码；通过 numpy/共享内存的 `BagelInferenceServer` 与 JAX policy 通信 |

**关键技术约束**：
- openpi 用 Flax `nnx`，新代码必须保持 NNX 风格（避免与 Linen 混用导致 ckpt 加载断裂）。
- JAX trace 错误较 PyTorch 难读：每个新模块先 `jax.disable_jit()` 跑通再 jit 化。
- M5 之前必须设计好 policy ↔ 世界模型之间的 IPC 协议（决策为线程内共享 numpy slot + lock，不引入 RPC 复杂度）。

---

## 4. 整体复现路线图（多轨道并行）

### 4.1 完整工作清单（含所有 phase / stage）

先把每个模块的**所有阶段**列全，避免遗漏：

| 模块 | 阶段 | 必做 / 可选 | 备注 |
|---|---|---|---|
| M0 | 基线对齐 + embed_prefix RFC | 必做 | 全员 |
| M1 KI | L1 机制 + L2 单元测试 (KI-V1/V2) | **必做** | scope 已缩减（§M1） |
| M1 KI | L3 论文级 ablation (KI-V3/V4) | **延后到 M7** | 复用 M7 PaliGemma 重训窗口 |
| M2 RTC | 推理移植 + ALOHA sim eval | 必做 | 完全独立 |
| M3 MEM | 阶段 A：短期视觉记忆（separable attention） | **必做** | 30-60k step on LIBERO |
| M3 MEM | 阶段 B：长期文本摘要（同 backbone 自回归） | **可选**（推荐做） | 需 GPT-4o 标注摘要 GT (~$80)；Counting Suite 唯一能验证 B 价值的 benchmark |
| M4 DCC | Stage A：核心实现 + B.3 训练（含 dropout 完整配方） | **必做** | 30k step on LIBERO |
| M4 DCC | Stage B：完整 ablation (B.1–B.6 共 6 条线) | **可选**（论文级） | 含 α 扫描 + steerability + 混杂数据；~1000 H100-hr |
| M5 WM | Phase A：BAGEL LoRA 训练 + 闭环 | **必做** | 30k step on 8×H100 |
| M5 WM | Phase B："gen images in context" 续训 | **条件触发**（仅 WM-V-6 不达标时） | 5k step |
| M6 HL | LoRA 训练 + 离线 eval + 闭环 (HL-V-5) | **必做** | 2×H100 极快 |
| M6 HL | HL-V-7 ablation (history / cold-start / full-FT) | **可选**（论文级） | 3 条件各 4k step |
| M7 | Backbone 升级 + 多源 mixture loader | **延后**（与数据团队联动） | ~3 个月 |

### 4.2 依赖关系矩阵

| Pair | 代码冲突 | Ckpt 依赖 | 数据依赖 | 结论 |
|---|---|---|---|---|
| M1 vs M3-A | 几乎无（不同文件） | 无（M3-A 直接从 `pi05_base` 起） | 无 | ✅ **强并行** |
| M1 vs M4 | 无 | 无 | 无 | ✅ **强并行** |
| M3-A vs M4 | ⚠️ `embed_prefix`/`Observation` 重叠 | 无（都可从 `pi05_base` 起） | 无 | 🔶 **代码串行 / 数据准备并行** |
| M3-A vs M3-B | M3-B 改 `pi0.py` 加 `generate_summary` + serving | M3-B 续 M3-A ckpt | M3-B 需 GPT-4o 摘要 GT | 🟡 **A 收尾后 B 开始** |
| M2 RTC vs 任何 | 完全无 | 无 | 无 | ✅ **完全独立** |
| M5 vs M6 | M5=PyTorch / M6=JAX 子系统 | 都从 M4 ckpt 起 | M6 复用 M4 的 GPT-4 标注 | ✅ **强并行**，合并点在 HL-V-5 |
| M4 主训练 vs M4 Stage B ablation | 同代码不同 config | Stage B 续 Stage A | 无 | 🟡 **可滚动并行**（不同 GPU pool） |
| M5 Phase A vs Phase B | 同代码 | B 续 A | 无 | 🔴 **B 续 A，且 B 是条件触发** |

### 4.3 多轨道并行甘特图（按周）

假设 **3-4 名工程师**（A/B/C/D），共 ~22 周（5.5 个月）走到 M5+M6 三组件闭环里程碑；M7 在此之后。

```
                W1   W2   W3   W4   W5   W6   W7   W8   W9   W10  W11  W12  W13  W14  W15  W16  W17  W18  W19  W20  W21  W22
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Phase 0        [M0 + RFC]
全员 (1 人周)
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Track A       │ M1 KI 机制+单测 │ M5 setup + BAGEL  │ M5 Phase A LoRA 训练       │ M5 离线 eval │ M5 闭环 + WM-V-5/6/7  │ M5 ablat │
(KI → WM)     │ KI-V1/V2        │ vendored + 数据   │ 30k step on 8×H100         │ FID/LPIPS    │                       │ + Phase B│
              │ ~50 H100-hr     │ prep + FSDP+peft  │ ~200 H100-hr               │ + VLM judge  │                       │ (条件触发)│
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Track B       │ M3-A 实现           │ M3-A 训练 30-60k │ M3-A 评测：自建 probe   │ M3-B 实现+ │ M3-B 训练 + Counting Suite eval  │
(MEM A → B)   │ temporal_attn       │ step on LIBERO+  │ + RoboMME + RMBench     │ GPT-4o GT  │ (可选；若不做则 Track B 转支援   │
              │ + Observation T     │ DROID 子集       │ + MIKASA-Robo           │ 标注       │  Track A/C 的评测 / 写报告)      │
              │                     │ ~300 H100-hr     │ → 发 M3-A blog          │            │                                  │
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Track C       │ M2 RTC 移植 │ M4 数据准备：GPT-4 子任务  │ ⏸ 等 M3-A     │ M4 实现：多组件prefix│ M4 主训练 Stage A 30k step + DC-V4/V5/V7 │
(RTC → DCC)   │ → ALOHA eval│ 标注 + metadata 缓存 +     │ embed_prefix  │ + DropoutContext     │ ~300 H100-hr → 发 M4 blog                 │
              │ ~10 H100-hr │ subgoal frame 缓存          │  接口稳定     │                      │                                            │
              │ → 发 RTC demo│                            │              │                      │ ┌──── M4 Stage B 完整 ablation ────────┐ │
              │              │                            │              │                      │ │ (B.1-B.6 共 6 条线，~1000 H100-hr)   │ │
              │              │                            │              │                      │ │ 与 Track A/D 不同 GPU pool 并行       │ │
              │              │                            │              │                      │ └────────────────────────────────────────┘ │
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Track D                                                                                  │ M6 数据 │ M6 LoRA   │ HL-V-4 │ HL-V-5 闭环（4 条件 sweep）│
(HL，W14 加入)                                                                            │ prep    │ 训练 8k   │ VLM    │ ★ 三组件闭环里程碑：首份  │
                                                                                          │ (复用   │ step on   │ judge  │   开源 π0.7 (HL+WM+VLA)   │
                                                                                          │  M4 标注)│ 2×H100    │ + V-6  │   完整系统 ★              │
                                                                                          │         │ ~6 H100-hr│latency │                            │
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Phase 5                                                                                                                          │ M7 启动 │
(W22+)                                                                                                                           │ + KI L3 │
                                                                                                                                  │ ablation│
```

**关键合并点 / 决策门**：
- **W4-5**：M1 + M3-A 代码 merge → main 分支稳定，M4 可以基于稳定的 `embed_prefix` 接口开始扩展。
- **W11**：M3-A 评测完成 → 决定 M3-B 是否做（取决于 RoboMME Counting Suite 的实际表现 + 算力余量）。
- **W14**：M4 主训练 Stage A 完成 + ckpt 落地 → M5 / M6 同时起跑（Track A 和 Track D 都基于这个 ckpt）。
- **W17**：M5 离线 eval 完成 → 决定是否触发 Phase B（看 WM-V-6 train-test gap stress 结果）。
- **W20**：HL-V-5 条件 d 跑通 → ★ 三组件闭环里程碑 ★，发 release blog + 技术报告草稿。
- **W22+**：与数据团队联动启动 M7（前提：M0-M6 全部稳定 + 数据团队能提供多源混合数据）。

### 4.4 阶段交付物（Deliverables）

| 时间点 | 可发布产出 | 对外形式 |
|---|---|---|
| W4 | KI 训练代码（首份开源）+ M2 RTC demo | GitHub PR + blog |
| W11 | M3-A 短期视觉 MEM ckpt + 多 benchmark 数字 | HF ckpt + blog |
| W14 | M4 DCC ckpt（含 steerability demo） | HF ckpt + blog |
| W17 | M5 BAGEL-VLA 闭环 demo（首份开源 BAGEL+VLA） | HF ckpt + blog |
| W20 | **三组件闭环系统 release** | HF ckpt + 技术报告草稿 |
| W22+ | M7 v2 release（Gemma3 4B 升级版）+ KI 论文级 ablation | 完整技术报告 |
| 滚动 | M3-B（可选）/ M4 Stage B（可选）/ M6 HL-V-7（可选） | ablation 补丁式发布 |

### 4.5 关键设计前置：`embed_prefix` 重构 RFC

Phase 0（W1）必须先开"`embed_prefix` 重构 RFC"，由一人主笔接口，明确未来要支持：

- (a) 可选 FAST token 段（M1 用）
- (b) 时序帧维 T（M3-A 用）
- (c) 长期摘要 token 段（M3-B 用）
- (d) 多组件 sentinel 占位（M4 用）
- (e) 子目标图像走和 obs image 一样的 SigLIP（M4 用，M5 推理时填充）

让所有轨道都基于同一接口扩展，避免后期大规模 rebase 冲突。

### 4.6 团队规模灵活性

| 团队规模 | 可行的并行度 | 总时间预估 |
|---|---|---|
| 1-2 人 | 只能跑主线（M0→M1→M3-A→M4→M5/M6 半串行） | 8-10 个月 |
| 3 人 | Track A/B/C 全开但 Track D 由其他人轮岗 | 6-7 个月 |
| **4 人**（推荐） | 全部 4 轨道 + 可选阶段（M3-B / M4 Stage B / M6 HL-V-7） | **5-6 个月** |
| 5+ 人 | 上述 + M7 提前与数据团队对接 | 4-5 个月 |

每个 milestone 都设计为**独立可量化验证**：单元测试（forward / 梯度路径）+ 小数据 overfit + LIBERO/RoboTwin 主评测 + 模块特异性指标。

---

## 5. 各子模块复现方案

> 每个模块只讲：**原理 / 训练数据 / 评估方法**。代码实现细节见各自的子模块详细方案文件。

### M0 — 基线对齐与基础设施（前置工作，~1 周）

**做什么**：
- 把 `pi05_base` ckpt 加载推理路径打通；复现 [openpi README](../../Research%20Exploration/Foundation%20Model/openpi/README.md) 的 LIBERO 数字（90%+）作为 forward sanity test。
- 打通 `train.py` 多节点 FSDP；搭好评估 harness（LIBERO + 可选 RoboTwin）。
- 确认评测 wandb / ckpt 流水线 + CI 回归测试框架。

**注意**：成功跑通 `pi05_base` 推理 **不等于** 验证 KI——见 §M1。

---

### M1 — Knowledge Insulation（KI）训练流程（**缩减 scope**）

#### 原理

KI 是**训练时**机制，**推理时是 no-op**：

1. **副信号**：训练时在 VLM 端额外预测 FAST 离散动作 token，加 next-token-prediction CE loss。
2. **梯度阻断**：action expert 经 cross-attention 与 VLM 交互时，**梯度不回传** VLM 的 K/V 投影（[`gemma.py:158-249`](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/models/gemma.py#L158-L249) 内部加 `jax.lax.stop_gradient`）。
3. 推理路径完全等价无 KI 版本。

**三大 claim**：train fast / run fast / generalize better，本质是"VLM 预训练知识不被 action 梯度污染"。

#### 本期 scope 决策（与原方案的差异）

由于 `pi05_base` 已经过 KI 训练（§1.2），M3-M6 都基于它 fine-tune，**KI 在 fine-tune 阶段的边际收益很小**。本项目把 KI 工作分两段：

| 阶段 | 内容 | H100-hr | 时间 |
|---|---|---|---|
| **M1（本期，必做）** | L1 机制实现 + L2 开源代码 + KI-V1/V2 单元测试 + M3-M6 fine-tune 时**默认开启 KI 开关** | ~50 | 1-2 周 |
| **M7 期复用（延后）** | L3 论文级 ablation（KI-V3 train fast + KI-V4 VLM retention，从 PaliGemma 起 30k step × 2 条线） | 500-2000 | 与 M7 同步做 |

**理由**：KI-V3/V4 必须从 PaliGemma 起跑（不能从 `pi05_base` 起，否则对照失效）；M7 backbone 升级时本来就需要从头训，是天然合适的窗口。在 M1 阶段单独做 KI-V3/V4，性价比不如把算力投到 M3/M4/M5。

#### 训练数据（M1 本期）

- **toy 数据**（任意 LIBERO sample 一个 batch）做 KI-V1/V2 单元测试。
- **不需要**大规模训练数据。
- **起点 ckpt**：单元测试不需要；M3-M6 fine-tune 默认从 `pi05_base` 起 + KI 开关 on。

#### 评估方法（M1 本期）

| 验证 | 在验证什么 | 通过条件 |
|---|---|---|
| **KI-V1 梯度路径单元测试**（最关键） | KI 两个机制都启动 | 仅 flow loss 反传时 `g_vlm = 0`；仅 FAST loss 反传时 `g_vlm ≠ 0` 且 `g_act = 0` |
| **KI-V2 α=0 退化等价** | 不破坏 baseline | KI 关闭时 loss 曲线与原 openpi 位级一致 |
| **KI-V5 forward sanity** | 模型类与 sampling 一致 | `pi05_base` 加载推理 LIBERO 数字 ≥ 90%（属 M0） |

#### 评估方法（M7 期，与 backbone 升级合并）

| 验证 | 在验证什么 | 通过条件 |
|---|---|---|
| **KI-V3 训练动力学 A vs B** | 复现 train fast / generalize better | LIBERO hold-out 上 B 显著优于 A；同 GPU-hours 下 B 收敛更快 |
| **KI-V4 VLM 知识保留** | 复现 insulation | 训练后 B 在 VQA-v2/MMMU/MMBench 上保留 ≥90% PaliGemma 基线；A 显著下降 |

> **关键澄清**：KI-V5（加载 `pi05_base` 跑推理）只能验证 forward 实现，与 KI 训练机制无关——无论代码里有没有 stop_gradient，推理结果都一样。**KI 训练机制本身的正确性靠 KI-V1 守住**（M1 已能做到）；论文 claim 的复现靠 KI-V3/V4（延后到 M7）。

---

### M2 — Real-Time Chunking（RTC）推理（独立轨道）

#### 原理

flow-matching denoise 时把"已发出的 action chunk"当 inpainting 锚点冻结，对剩余位置加 inpainting guidance term，实现异步无停顿、平滑过渡。论文 [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) 报告执行时长 -20%、jerk 显著降低。

#### 训练数据

**无需训练**——纯推理改造。直接复用任意 ckpt。

#### 评估方法

- **ALOHA sim cube transfer** 上：相同任务执行时长比同步推理 **-15% 以上**。
- **运动平滑度**：jerk 指标显著低于同步基线。
- **回归保护**：`rtc_enabled=False` 时与现 inference 路径数值一致。

> 本步骤**算力成本接近零**、**对外宣传力强**（论文级 demo，无需训练）。可作为整个项目的早期对外里程碑。直接 port LeRobot PR #1698 的 PyTorch 实现到 openpi JAX `serving/websocket_policy_server.py`。

---

### M3 — MEM（多尺度记忆）

#### 原理

| 组件 | 设计 |
|---|---|
| **短期视觉记忆** | 6–18 帧（间隔 1s/3s）通过 SigLIP 独立 spatial encode → 加可学习 temporal positional embedding → 经 K=2-4 层 **TimeSformer 风格 space-time separable attention**（沿 T 轴做 attention，per-spatial-position）→ 串接喂 Gemma backbone |
| **长期文本记忆** | **同 backbone Gemma 周期性自回归生成自然语言摘要**（每 ~30s 一次），摘要 token 拼到 prefix 最前 |

**为什么不用 full 4D attention**：6 帧 × 3 cam × 256 = 4608 视觉 token，full attention 是 O((TN)²) ≈ 21M ops/head/layer；separable 把 cost 降到 O(TN² + T²N) ≈ 2M，约 **10×** 加速。

**关键 trick**：TemporalBlock 的输出层**零初始化**（ViViT/BERT 标准做法）→ 训练初值等价 history=1 baseline，回归保护一举两得。

#### 训练数据

- **LIBERO 全量** + **DROID 子集**（5-10k ep）。
- 起点 ckpt：**直接从 `pi05_base` 起**（KI 已 baked in），fine-tune 时开 KI 开关（M1 已实现）。
- **阶段 A（必做）**：仅短期视觉，30-60k step。
- **阶段 B（可选，优先级低）**：加长期摘要。GT 摘要由 GPT-4o 离线对 demo 视频标注（~$80 / 1 万 episode），**不**实现"混合最优 + 失败 demo"（开源数据无失败 demo，明确放弃）。

#### 评估方法

| Benchmark | 测什么 | 通过条件 |
|---|---|---|
| **LIBERO baseline** | 回归保护：MEM 不退化普通任务 | success rate 不退化超过 -2% |
| **自建 LIBERO Memory Probe**（P1 遮挡-取物 / P2 多步指令记忆 / P3 计数任务） | 物体永存 / 长期记忆 / 计数 | baseline (history=1) vs MEM 必须有显著正向 gap |
| **[RoboMME](https://github.com/RoboMME/robomme_benchmark)** 4 个 Suite（Counting / Permanence / Reference / Imitation） | 记忆四象限 | 阶段 A：Permanence +15-30%、Reference +10-15%、Imitation +20%；Counting Suite 仅阶段 B 显著 |
| **[RMBench](https://github.com/RoboTwin-Platform/RMBench)** 9 任务 | 跨复杂度 | 与其自带 Mem-0 baseline 比对，验证是否达到或超过显式 memory 模块 |
| **[MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo)** 5-10 任务子集 | 成熟保底 | 交叉验证 |

> **风险**：RoboMME / RMBench 是 2026-03 新发布仓库，第一周必须验证 clone + quickstart 跑通；不能假定可用。MIKASA-Robo 是 ICLR 2025 已成熟仓库，作为保底。

---

### M4 — Diverse Context Conditioning + Per-Component Dropout

#### 原理

把 prefix 从单 prompt 扩到**五段多模态 context**（论文 §V），训练时**每段独立按概率 dropout**：

| 组件 | 内容 | 训练时保留率 |
|---|---|---|
| 总任务 ℓ | "Make me a sandwich" | 100%（base 信号） |
| 子任务 ℓ̂ | "open the fridge door" | subgoal 在场时 70% |
| 子目标图像 g | 多视角未来帧（训练用 trajectory[t+Δ] GT 真实帧；Δ ∈ [2,6]s 均匀采样） | 25% |
| Metadata m | speed bin / quality 1-5 / mistake bool | 整体 85%；保留时每个子字段再 95%（**嵌套 dropout**） |
| Control mode c | "joint" / "end-effector" | 100%（不 dropout） |

**Prefix 长度恒定**：dropout 段用 `<empty>` sentinel token 占位（避免 nn.scan 重编译）。**Subgoal 图像走和 obs image 一样的 SigLIP**，但加一段可学习 type embedding（`type_id ∈ {obs, subgoal}`，**初始化为 0** 保证起点 ckpt 兼容）。

**Dropout 在 transform 层做**——forward 路径完全透明，回归保护强、便于 ablation。

**关键 claim**：steerability（speed/strategy 可控）+ mixed-quality data robustness（metadata 让模型区分干净 vs 脏 demo）+ subgoal 加速收敛（inverse dynamics 结构）。

#### 训练数据

- **LIBERO 全量** + **DROID 子集** + **RoboTwin**（增加 control_mode 多样性）。
- 起点 ckpt：M3 阶段产出的 MEM 版本（含 KI 开关）。如 M3 未完成可临时退化用 `pi05_base`。
- 五个组件在 LIBERO 上的来源：
  - 总任务 ℓ：LIBERO 自带。
  - 子任务 ℓ̂：**新标注**——GPT-4 / Gemini 切段写子任务（130 任务 × 5 demo × ~3 子任务 ≈ 2k 条标注，~$50 API + 0.5 人天 review）。
  - Subgoal g：trajectory[t + Δ] 真实帧，**纯 transform，零标注**。
  - Metadata m：`speed` 由 action L2 norm 三分位自动分箱；`quality=5`（demo 都是 expert）；`mistake=空`。**DC-V5 实验时手动注入 30% noisy demo** 模拟 mixed quality。
  - Control mode c：LIBERO=`end-effector`。
- 训练步数 30k step，bf16 + FSDP，~16×H100，~2 天。

#### 评估方法

| 验证 | 在验证什么 | 通过条件 |
|---|---|---|
| **DC-V1 dropout 统计** | 实际命中率与配置一致 | ±2% 内 |
| **DC-V2 forward 等价** | 不破坏 baseline | 全 dropout=0 + type_emb=0 时与 KI+MEM 位级一致 |
| **DC-V3 prefix 长度恒定** | 不触发 nn.scan 重编译 | 任意 mask 组合下序列长度不变 |
| **DC-V4 Steerability**（核心 claim） | speed / strategy / subgoal 可控 | speed bin 三档下 action norm 单调变化 (p<0.05)；不同 ℓ̂ 下执行路径明显不同；subgoal-conditioned vs unconditioned 轨迹分布显著差异 |
| **DC-V5 混杂数据鲁棒性**（最值得发的实验） | metadata 让模型区分干净 vs 脏数据 | 30% LIBERO demo 加噪后训练，B（含 metadata）≥ baseline；B'（无 metadata）显著下降 |
| **DC-V6 Subgoal 加速收敛** | §V-B "inverse dynamics" claim | 含 subgoal 训练前 10k step flow loss 下降明显更快 |
| **DC-V7 推理时缺字段鲁棒性** | dropout 训练带来的输入鲁棒性 | 缺任意字段成功率衰减 < 10%；无 dropout 训的对照衰减 > 30% |

---

### M5 — World Model（BAGEL）

#### 原理

为 M4 中"推理时缺失的 subgoal 图像"提供生成器：根据当前观测 + 子任务文本 + metadata，**异步**生成 4 秒后的多视角未来帧（FLUX VAE latent 上的 conditional flow matching）。

| 要素 | 决策 |
|---|---|
| **Backbone** | `ByteDance-Seed/BAGEL-7B-MoT`（PyTorch，~14B 总参 / ~7B 激活，MoT = Qwen2.5-7B LLM + 图像生成 expert + FLUX VAE + SigLIP2）；论文原版选型 |
| **Adaptation** | **LoRA only on gen-expert**（r=16, α=32），SigLIP2 / LLM expert / FLUX VAE 全冻；**不**做 full FT（14B + LIBERO 小数据 = 必然过拟合） |
| **分辨率** | 训练 256×256（VAE 16 倍数约束），推理时下采到 224 喂第三步 VLA |
| **多视角** | **联合生成**（两视角 latent token 串接到同一 `<gen_begin>...<gen_end>`，共享 attention 保一致性） |
| **推理刷新** | 4s OR subtask 字符串变化时，`BagelInferenceServer` 独立线程（threading + lock-protected slot），VLA 主线程拿 `latest()` |
| **训练目标采样** | 25% 取 segment 末帧；75% 在 0–4s 内均匀采样（论文 §V-B）；clip-to-end + clip-to-subtask-end 处理 LIBERO 短 episode |

#### 训练数据

- **LIBERO 全量** + 第三步 GPT-4 子任务标注（复用，零额外成本）+ 第三步 metadata 缓存（复用）。
- BAGEL HF 权重（公开）。
- 训练步数 30k step（Phase A），8×H100 ~25h wall。
- **Phase B（论文 §V-B "generated images in context"）**：仅当 WM-V-6 不达标时启动，5k step 续训以 50% 概率把当前帧替换为 WM 自采样图，缓解 train-test gap。

#### 评估方法（三模态）

| 验证 | 在验证什么 | 通过条件 |
|---|---|---|
| **WM-V-1 forward + LoRA 等价** | adapter init=0 时与 base bit-exact | FSDP × 2 GPU 不 OOM |
| **WM-V-2 单 batch overfit** | 视觉收敛 | 500 step 后 CFM val loss < 0.05；目视生成可识别为 GT 未来状态 |
| **WM-V-3 离线图像质量** | 生成质量 | 5 LIBERO holdout × ~5k pair：**FID ≤ 35**, **LPIPS ≤ 0.30**, **SSIM ≥ 0.55**（双视角平均）。基线：BAGEL 不 fine-tune 直接跑 LIBERO 大概率 FID ≥ 100 |
| **WM-V-4 VLM-judge 语义正确** | subgoal 是否对应 subtask | 200 holdout 三元组 GPT-4o 0–5 打分，**mean ≥ 3.5 且 ≥70% sample ≥ 3** |
| **WM-V-5 闭环 LIBERO 成功率**（核心） | 真实增益 | 4 suite × 50 trial × 3 条件：(a) 第三步 VLA 无 subgoal（下界）, (b) +oracle GT subgoal（上界）, (c) +WM 生成。**c ≥ b − 5pp 且 c > a + 3pp on Long** |
| **WM-V-6 train-test gap stress** | 是否需要 Phase B | 每步强制 refresh 时成功率下降 ≤ 8pp（不达标 → 启动 Phase B） |
| **WM-V-7 Async timing** | 推理可用 | median `generate()` < 4.0s, p95 < 5.0s on 1×H100；VLA 主循环 stall ≤ baseline +5% |
| **WM-V-8 ablation** | 消融三因素 | 去 metadata / per-view independent / LLM-expert 也加 LoRA：报 ΔFID/Δsuccess |

---

### M6 — High-Level Policy（自动子任务生成）

#### 原理

把 M4 中 `subtask` 字符串本身也变成可学习的产物。**独立模型**（π0.7 §IV / Algorithm 1，与 π0.5 KI 联合训练不同），从 `(当前观察, 主任务, 历史 subtask)` 自回归生成下一个子任务，每 4s 或语义变化时异步刷新。

| 要素 | 决策 |
|---|---|
| **Backbone** | 复用 openpi 原生 `paligemma_gemma_2b`（**不**新增 Gemma 3 4B），从 M4 ckpt 的 PaliGemma 部分**热启动**，**冻结 SigLIP**，**LoRA 注入 LLM**（r=16） |
| **架构** | mirror `Pi0FAST` 但**去掉 action expert**：纯文字 next-token CE，AR 解码 |
| **Prompt 模板** | `"Task: {ℓ}. Done so far: {history}. Next subtask:|{label}<eos>"` |
| **推理** | `HLInferenceServer`（threading），triggers: 4s 超时 OR history 变化；与 M5 `BagelInferenceServer` 同进程不同线程 |
| **接入 VLA** | `LiberoInputs` 拉 `hl_server.latest()` → 注入 subtask；HL 输出变化 → WM 立即重生 subgoal（论文 Algorithm 1 line 7） |

#### 训练数据

- **完全复用 M4 的 GPT-4 子任务标注**（episode-level segment + subtask string），反过来当 HL 训练标签。**零新标注成本**。
- 数据规模：LIBERO 全量 ~500 episode × 平均 3-5 segment × 8 采样点 ≈ **15k 样本**；加 boundary + switch 增强 → **25k 样本**。
- LoRA r=16 + 25k 样本 + ~10 epoch on 2×H100 ≈ **3 小时收敛**。
- 起点 ckpt：M4 `pi05_div_libero` 的 PaliGemma 部分热启动（丢弃 action expert）。**不要** cold-start。

#### 评估方法（三层）

| 验证 | 在验证什么 | 通过条件 |
|---|---|---|
| **HL-V-1** LoRA 等价 + smoke | adapter init=0 时与 base bit-exact | 2 GPU 不 OOM |
| **HL-V-2** 单 batch overfit | 500 step 后 greedy decode 完全匹配 label | suffix CE < 0.05 |
| **HL-V-3** 离线 token-level（20 holdout episode → ~5k 样本） | 离线准确率与切换检测 | (a) **Top-1 exact-match ≥ 60%** on regular；(b) **Subtask switch detection F1 ≥ 0.55**；(c) CE ≤ 0.4 |
| **HL-V-4** VLM-judge 语义合理性 | 生成 subtask 是否合理 | 200 holdout (current_image, main_task, history, generated ℓ̂) → GPT-4o 0-5 打分，**mean ≥ 3.7 且 ≥75% sample ≥ 3** |
| **HL-V-5** 闭环 LIBERO 替换 oracle（核心） | 替代人工 subtask 是否有效 | 4 suite × 50 trial × **4 条件**：(a) M4 VLA 无 subtask, (b) +oracle subtask（上界）, (c) +HL 生成 subtask, (d) +HL + WM（**完整 π0.7 三组件**）。**c ≥ b − 4pp，d ≥ b − 6pp**，Long suite 上 d > a + 3pp |
| **HL-V-6** Async timing | 推理不阻塞 VLA | median `generate_subtask()` < 300ms；VLA 主循环 stall ≤ baseline +3% |
| **HL-V-7** Ablation | 消融 history / 起点 / LoRA vs full FT | Δ exact-match / Δ success |
| **HL-V-8** 与 M5 联评一致性 | HL 输出不让 WM 失效 | HL ℓ̂ 喂 WM 时 FID 漂移 ≤ 5 |

> **HL-V-5 的条件 d 是整个项目的最终里程碑**——首份开源 π0.7 完整三组件（HL + WM + VLA）闭环系统。

---

### M7 — Backbone 升级（Gemma 3 4B + 860M action expert，**推迟到最后**）

#### 原理

把 VLM 升到 Gemma 3 4B、action expert 加宽到 860M，匹配 π₀.₆ / π₀.₇ 实际规格。新写 `models/gemma3.py`（Flax NNX，~500 行），HF safetensors → orbax 权重转换脚本。

#### 为什么放在最后

- 改变模型大小后**无法继续使用 `pi05_base` 这个 pretrained 权重**作为起点。
- 我们目前**没有适合自己 pretrain 的数据规模**——任何 backbone 升级后的训练都只能从 PaliGemma → Gemma3 转换权重起，效果不如沿用 `pi05_base`。
- 因此 M7 的合理位置是：**前 6 个模块都用 `pi05_base` 验证完毕之后**，再统一升级 backbone 重训一轮，作为 v2 release。
- M7 的训练涉及多源 mixture sampling（写 `MixtureDataset` wrapper）、OXE 子集加载（参考 Octo / open-pi-zero）——这些都是预训练侧的工作，**与数据团队联动**。

#### 训练数据

需数据团队提供 OXE 子集 / 多机器人混合配方；本团队仅写 mixture loader。

#### 评估方法

- 在 LIBERO / RoboTwin / DROID 上重训，与"M1-M6 在 `pi05_base` 上"的指标比较，期望 success rate +3-5%（按 PI model card 趋势）。
- 验证 KI/MEM/DCC/WM/HL 五个模块在新 backbone 下数值行为一致。

---

## 6. 算力规模估算

| Milestone | 参数 | 显存 (batch=1, bf16) | 推荐集群 | 训练步数 | H100-hr |
|---|---|---|---|---|---|
| M0 sanity | 3.5B | ~50 GB | 8×H100 | — | ~20 |
| **M1 KI（缩减）**：仅 L1 机制 + KI-V1/V2 单元测试 | 3.5B | <40 GB（toy） | 单卡 H100 | — | **~50** |
| M2 RTC | — | — | — | 0（纯推理） | ~10（评测） |
| M3 MEM 阶段 A | 3.5B | ~70 GB（6 帧视觉 + temporal） | 16×H100 | 30-60k | 200-500 |
| M3 MEM 阶段 B（可选） | 3.5B | ~85 GB（长 prompt） | 16-32×H100 | 50-100k | 800-1500 |
| M4 DCC 阶段 A | 3.5B | ~60 GB（多组件 prefix） | 16×H100 | 30k | ~150-300 |
| M4 DCC 完整 ablation（B.1-B.6） | 3.5B | ~60 GB | 16×H100 | 30k × 6 | ~1000 |
| M5 WM Phase A | 14B (LoRA) | ~120 GB / GPU（FSDP frozen base） | 8×H100 | 30k | ~200 |
| M5 WM 闭环 + ablation | 14B | — | 8×H100 | — | ~150 |
| M6 HL（LoRA） | 2B (LoRA) | <40 GB | 2×H100 | 8k | ~6 |
| M6 HL 闭环 + ablation | 2B | — | 2×H100 | — | ~40 |
| **M7 backbone 升级 + KI L3 论文级 ablation 合并** | ~5B | ~80 GB | 16-32×H100 | 100k+ | 1.5k-4k |

**总 budget**（M0-M6，不含 M7）：**~2000-4000 H100-hr**（相比原方案省 1000+ H100-hr：M1 KI L3 大型实验延后到 M7 合并做）。

M7 单独再 1.5k-4k H100-hr，与数据团队联动后再启动。

---

## 7. 数据需求总览

| 数据 | 用途 | 来源 | 标注成本 |
|---|---|---|---|
| LIBERO 全量 demo | 主训练 + LIBERO eval | 公开（MIT） | 0 |
| DROID 子集 5-10k ep | 真实场景多样性 | CC-BY 4.0 | 0 |
| RoboTwin 任务集 | M3/M4 评测 + control mode 多样性 | 公开 | 0 |
| **LIBERO subtask 标注**（M4/M6 共享） | subtask 字段 + HL 标签 | GPT-4 + 人 review | 0.5 人天 + ~$50 API |
| LIBERO metadata 缓存 | M4 metadata 字段 | 脚本自动 | 0 |
| LIBERO noisy demo 注入（M4 DC-V5） | mixed-quality 实验 | 脚本生成 | 0 |
| Subgoal 训练 pair | M5 (current, future) | trajectory 自动派生 | 0 |
| BAGEL HF 权重 | M5 起点 | HF 公开 | 0 |
| GPT-4o 摘要 GT（M3 阶段 B 可选） | 长期摘要监督 | OpenAI | ~$80 |
| GPT-4o VLM judge（M5/M6 评测用） | 语义评分 | OpenAI Batch API | ~$15/次 |
| VQA-v2 / MMMU / MMBench | M1 KI VLM 探针 | 公开 | 0 |
| RoboMME / RMBench / MIKASA-Robo | M3 评测 | 公开（Apache/MIT） | 0 |
| **新人工标注** | — | — | **基本无需** |
| OXE 多源 mixture（M7 才需） | backbone 升级预训练 | 各子集独立 license | 与数据团队联动 |

→ M1-M6 全程使用现有公开数据 + GPT-4 辅助标注，**几乎无标注成本**，**完全不需要 PI 量级私有数据**。

---

## 8. 验证策略（贯穿所有 milestone 的统一原则）

每个 milestone 必须满足：

1. **forward 单元测试**：新模块开关 off 时与原 openpi 数值位级一致（回归保护）。
2. **小规模 overfit 测试**：单 batch 训 1k step，loss → 0。
3. **回归保护**：旧 milestone 的指标不退化超过 -2%。
4. **基准评测**：LIBERO 子集上的 success rate（最便宜 e2e）；模块特异性 benchmark（如 M3 的 RoboMME）。
5. **PR review 必过项**：单元测试 + α=0/init=0 退化等价测试。
6. **公开博客 + ckpt 发布**：每个 milestone 一份 release blog + HF ckpt，对外宣传持续可见。

---

## 9. 现有方案的潜在问题分析

> 以下是把 5 个子模块方案与整体路线放到一起后，识别出的潜在问题。按"可能影响发表 / 可能阻塞实施 / 仅需关注"分级。

### 9.1 可能影响发表 / 论文级 claim 的潜在问题

| # | 问题 | 影响 | 应对建议 |
|---|---|---|---|
| **P1** | **`pi05_base` 已经过 KI 训练**——M3-M6 都从它起步，会让"加 KI vs 不加 KI"的对照在 M3 之后无法严格做。M1 的 KI-V3/V4 必须严格从 PaliGemma 起，但 M3+ 的对照基线就只能是"M1 训出来的我们自己的 KI ckpt"，**而不是论文 KI 版本**。 | M1 之后所有 ablation 都基于自训 KI ckpt，与论文数字不可直接对齐；reviewer 可能质疑"你的 baseline 不是 paper 的 baseline"。 | 在每个 milestone 报"自训 KI baseline" + "`pi05_base` baseline"两条；明确说明差异源于训练数据规模而非方法。 |
| **P2** | **GPT-4 子任务标注的语义对齐**：M4 / M6 都依赖这套标注，但 GPT-4 切段可能与 trajectory 实际语义错位（"opening door" 标到 "approaching door" 帧）。M6 HL 训练对此尤其敏感（labels 即标注本身）。 | 标注噪声会让 HL-V-3 token accuracy 上限被卡死；HL-V-5 闭环可能因 subtask 切换时机不对而跑挂。 | 30 任务先做人工 review；标注脚本固定 prompt 模板；HL-V-3 加 "exact-match-with-main-task" 退化检测（>30% 视为失败）。 |
| **P3** | **LIBERO demo 全是 expert + success → metadata 信号弱**：M4 的 metadata 在 LIBERO 上几乎无变化（speed 自动分箱、quality 全 5、mistake 全空），DC-V5 必须**手动注入 noisy demo** 才能验证 mixed-quality robustness claim。 | 注入策略不严谨会被质疑（"你的 noisy demo 是合成的"）。 | DC-V5 用真实失败案例：从早期 LIBERO 训练 ckpt rollout 收集失败 episode，标 quality=2 + mistake_label。比纯合成噪声更可信。 |
| **P4** | **MEM benchmark 仓库不成熟**：RoboMME（2026-03）和 RMBench 都是新发布，可能维护不完整。M3 评测重度依赖。 | 评测无法跑 → M3 模块 claim 无外部 benchmark 支撑。 | M3 第一周 spike 验证仓库可用；自建 LIBERO P1/P2/P3 probe 作为强制保底；MIKASA-Robo（ICLR 2025）作为成熟交叉验证。 |
| **P5** | **MEM 的"训练-推理分布偏移"未解决**：论文核心难点是训练用近最优 demo、推理需重试失败子任务，需混合训练数据。开源数据无失败 demo，方案明确放弃此 ablation。 | MEM 在长 horizon 任务上可能 underperform 论文数字。 | 文档明确说明这是 known limitation；作为 future work；P3 注入策略可部分缓解。 |
| **P6** | **三组件闭环（HL + WM + VLA）的稳定性未充分验证**：M5 + M6 各自评估通过不代表三者拼起来稳定。HL 输出抖动 → WM 频繁重生 → 计算资源争用、subgoal 频繁变化 → VLA 行为不稳。 | HL-V-5 条件 d 可能成功率比 c 还低，反向结果。 | HL-V-8 单独验 HL 输出对 WM 的 FID 漂移；增加 HL 输出去重 + 防抖（已设计）；记录三组件之间的 timing trace。 |
| **P7** | **WM-V-3 FID 阈值（≤35）是经验估计**，无 LIBERO 上 BAGEL 的公开数字校准。如果 BAGEL 在 LIBERO 风格图像上原生 FID 已经 < 35，门槛太低；如果 > 100，门槛过严。 | M5 阶段 A 训完才能知道阈值是否合理。 | W3 训练前先跑一次 BAGEL 不 fine-tune 的 LIBERO FID baseline，再据此调阈值；阈值设计为相对值（FID 比 zero-shot 降 60%）而非绝对值。 |

### 9.2 可能阻塞实施 / 工程级问题

| # | 问题 | 应对建议 |
|---|---|---|
| **P8** | **JAX 与 PyTorch 并存的运维负担**：M5 (BAGEL PyTorch) 与 M1-M4/M6 (JAX) 同进程双线程跑；CUDA context 共享、显存竞争、双线程死锁、jaxlib + torch 版本兼容是实操坑。 | 单卡 ≥80GB（H100）；BAGEL 钉 cuda:0 / VLA 钉 cuda:1 多卡部署作主推荐；首调 `jax.block_until_ready` + BAGEL 显式预热；CI 加双线程 smoke test。 |
| **P9** | **`max_token_len` 在每个 milestone 都在膨胀**：M0=200 → M1（+FAST tokens）≥256 → M3（+T 帧视觉 token，4608+）→ M4（+subgoal images + metadata + subtask + sentinel）。每次都 O(N²) attention 增长。 | 每个 milestone 必须强制开 gradient checkpointing + flash attention；prefix 长度恒定（dropout 用 sentinel 占位）的设计必须严格执行；可能需要在 M3 之后降 batch size。 |
| **P10** | **`nn.scan` + `static_argnums` 兼容性**：M1 的 `ki_insulate: bool` 参数加入 scan 静态参数后，scan 编译可能与现有 `nn.remat` checkpoint policy 冲突。M3/M4 同样需要透传新 kwarg。 | M1 实施时先用非 scan 版本 Attention，确认逻辑正确再做 scan；M3/M4 增量改 scan 时跑全量回归测试。 |
| **P11** | **多节点训练支持**：openpi 当前 `train.py` JAX 路径有 FSDP 但**单节点**。M3+ 数据量会逼到多节点。 | M0 阶段必须打通多节点 FSDP；优先用 SLURM + JAX 自带 distributed init，避免引入 DeepSpeed / Ray 复杂栈。 |
| **P12** | **FAST tokenizer 性能**：M1 副 loss 需 tokenize FAST action token，每 dataloader worker init `AutoProcessor` 慢；M5 / M6 LoRA 预处理也涉及 tokenize。 | 离线一次性 tokenize 写入 LeRobot dataset 缓存字段，训练期不重算。 |
| **P13** | **Subgoal type embedding 初始化**：M4 引入 type embedding 后，加载 KI+MEM ckpt 时如果 type_emb 非零会破坏 forward 数值。 | ckpt-load hook 强制 type_emb 置零；DC-V2 forward 等价测试守住。 |
| **P14** | **PyTorch 路径滞后**：openpi PyTorch 副路径功能子集严重，每个新模块都要"JAX 完成 + 一周后 PyTorch 跟上"。M5 因 BAGEL 必须 PyTorch；其他模块的 PyTorch 路径可能长期不同步。 | 明确文档化"PyTorch 路径是 second-class citizen"；M1-M4/M6 优先 JAX，PyTorch 滞后跟进作为 community PR；M5 唯一 PyTorch 主线。 |
| **P15** | **HL 训练用 GT history、推理用模型自累积 history → exposure bias**：M6 训练时 history 是数据集 GT，推理时是模型自己上一步预测。 | 已设计 Phase B 缓解：训练时 50% 概率把 history 替换为前一步预测（schedule sampling）。仅当 HL-V-5 不达标时启动。 |

### 9.3 仅需关注 / 长期跟踪问题

| # | 问题 | 备注 |
|---|---|---|
| **P16** | **PI 是否会突然开源 π₀.₆/π₀.₇** | 多个 issue 未回应（[#789](https://github.com/Physical-Intelligence/openpi/issues/789)/[#791](https://github.com/Physical-Intelligence/openpi/issues/791)/[#793](https://github.com/Physical-Intelligence/openpi/issues/793)），假设短期不会；如果突然开源，本项目重心切换为"对照与 ablation"。 |
| **P17** | **KI 论文细节缺失**：stop-gradient 边界、辅助 loss 权重、FAST token 在 VLM 输入里的位置——需要从 NeurIPS supp / OpenReview 找补充材料。 | M1 实施时若细节模糊，按 §M1 的 KI-V1 单元测试断言为准（机制成立即可），不必死磕论文措辞。 |
| **P18** | **MEM 论文细节缺失**：长期摘要的具体生成方式、压缩比、是否需要专门微调摘要 LM——待精读全文。 | M3 阶段 B 优先级低，本期可先完成阶段 A。 |
| **P19** | **数据团队的 metadata 标注 pipeline** | M4 / M5 / M6 都依赖；建议提前与数据团队对齐 schema。 |
| **P20** | **模型升级（M7）的时机判断** | 必须等前 6 个模块都验证完毕；M7 启动前需确认数据团队能提供足够规模的多源混合数据，否则 M7 收益不如直接发 M1-M6 在 `pi05_base` 上的版本。 |

---

## 10. 时间线总览

详细的轨道分配 / 周历甘特图 / 决策门 / 阶段交付物已在 **§4 整体复现路线图**中给出（§4.3 的甘特图覆盖全部 22 周；§4.4 列出每个时间点的 deliverables；§4.6 给出不同团队规模下的总时间）。

**一句话节奏**（4 人团队，5.5 个月主线）：

```
Month 1     M0 + M1 + M2 RTC + M3-A 启动 + M4 数据准备             ─► W4: KI 代码 + RTC demo
Month 2-3   M3-A 训练评测 + M4 实现                                ─► W11: M3-A blog
Month 3-4   M4 主训练 + Stage A 评测；M3-B (可选) 启动             ─► W14: M4 DCC blog
Month 4-5   M5 BAGEL 训练 + 离线 eval；M6 HL kickoff               ─► W17: BAGEL-VLA demo
Month 5     M5 闭环 + M6 闭环 + 三组件联评                         ─► W20: ★ 三组件闭环里程碑 ★
Month 6+    M7 backbone 升级 + KI L3 ablation（与数据团队联动）   ─► 完整技术报告 + workshop 投稿
```

每 1-1.5 个月一个可发布的 ckpt + blog，对外宣传持续可见。

---

## 11. 关键参考资料

**论文**
- π₀ [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- π₀-FAST [arXiv:2501.09747](https://arxiv.org/abs/2501.09747)
- π₀.₅ [arXiv:2504.16054](https://arxiv.org/abs/2504.16054)
- KI [arXiv:2505.23705](https://arxiv.org/abs/2505.23705)（NeurIPS 2025）
- RTC [arXiv:2506.07339](https://arxiv.org/abs/2506.07339)
- MEM [arXiv:2603.03596](https://arxiv.org/html/2603.03596)
- π₀.₇ [arXiv:2604.15483](https://arxiv.org/abs/2604.15483)
- BAGEL [arXiv:2505.14683](https://arxiv.org/abs/2505.14683)

**关键 openpi 文件**
- [src/openpi/models/pi0.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/models/pi0.py)（compute_loss 188-214、embed_prefix 105-137、sample_actions 217-279）
- [src/openpi/models/pi0_fast.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/models/pi0_fast.py)（FAST CE loss 模板）
- [src/openpi/models/gemma.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/models/gemma.py)（双 expert 共享 attention 158-249）
- [src/openpi/models/tokenizer.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/models/tokenizer.py)（FASTTokenizer）
- [src/openpi/transforms.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/transforms.py)
- [src/openpi/policies/libero_policy.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/policies/libero_policy.py)
- [scripts/train.py](../../Research%20Exploration/Foundation%20Model/openpi/scripts/train.py)

**子模块详细方案（执行时各自参照）**
- 第一步 KI：[openpi-pi0-7-step1-knowledge-insulation.md](openpi-pi0-7-step1-knowledge-insulation.md)
- 第二步 MEM：[openpi-pi0-7-pi0-5-knowledge-insulation-luminous-locket.md](openpi-pi0-7-pi0-5-knowledge-insulation-luminous-locket.md)
- 第三步 Diverse Context + Dropout：[openpi-pi0-7-step3-diverse-context-dropout.md](openpi-pi0-7-step3-diverse-context-dropout.md)
- 第四步 World Model：[openpi-pi0-7-ki-mem-dicerse-context-con-proud-reef.md](openpi-pi0-7-ki-mem-dicerse-context-con-proud-reef.md)
- 第五步 High-Level Policy：[openpi-pi0-7-ki-mem-dicerse-context-con-zany-floyd.md](openpi-pi0-7-ki-mem-dicerse-context-con-zany-floyd.md)
- 初始整体调研：[pi0-7-pi0-5-pi0-6-pi-0-6-pi0-7-pi0-7-cozy-minsky.md](pi0-7-pi0-5-pi0-6-pi-0-6-pi0-7-pi0-7-cozy-minsky.md)
- 高层设计文档：[docs/pi07_highlevel_subtask_design.md](../../Research%20Exploration/Foundation%20Model/openpi/docs/pi07_highlevel_subtask_design.md)

**仓库**
- 主仓：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- RTC 参考：[lerobot PR #1698](https://github.com/huggingface/lerobot/pull/1698)
- 世界模型权重：[ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
- 记忆 benchmark：[RoboMME](https://github.com/RoboMME/robomme_benchmark) / [RMBench](https://github.com/RoboTwin-Platform/RMBench) / [MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo)