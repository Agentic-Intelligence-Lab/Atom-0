# π0.7 第五步：High-Level Policy（高层子任务推理）复现方案

## Context

我们正在基于 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) **逐模块**复现 π0.7（[arXiv:2604.15483](https://arxiv.org/abs/2604.15483)）。**第一步 KI**、**第二步 MEM**、**第三步 Diverse Context Conditioning + Dropout**、**第四步 World Model (BAGEL)** 已完成；本方案聚焦**第五步：High-Level Policy** —— π0.7 §IV/§V.A/Algorithm 1 的最后一块拼图。

第三步把 `subtask` 字符串引入 VLA prefix，但默认靠**人工或脚本提供**；第四步给出了 subgoal image 的生成器。本步要做的是：**把 subtask 字符串本身也变成可学习的产物**，由一个独立模型从 `(当前观察, 主任务, 历史 subtask)` 自回归生成，每 4 秒（或语义变化时）异步刷新一次 —— 完全实现 π0.7 论文 Algorithm 1 line 6–13。

**π0.5/π*0.6 vs π0.7 的关键差异**（[docs/pi07_highlevel_subtask_design.md](../../Research%20Exploration/Foundation%20Model/openpi/docs/pi07_highlevel_subtask_design.md)）：
- π0.5/π*0.6：高层与低层在**同一 PaliGemma 主干**里 KI 联合训练（CE + FM）。
- π0.7：高层是**独立模型**，分开训练，异步推理。本方案完全走 π0.7 路线。

**已锁定决策（用户已选）**：
- **Backbone**：复用 openpi 原生 `paligemma_gemma_2b`（**不**新增 Gemma 3 4B），从第三步 `pi05_div_libero` ckpt 的 PaliGemma 部分**热启动**，冻结 SigLIP，LoRA 注入 LLM。
- **数据**：复用第三步 [scripts/prepare_libero_subtasks.py](../../Research%20Exploration/Foundation%20Model/openpi/scripts/prepare_libero_subtasks.py) 产出的 `(episode, segment, subtask_string)` 标注，**反过来**当 HL 训练标签：`(o_t, ℓ_main, history) → ℓ̂_next`。零新标注成本。
- **评测**：三层 —— (a) 离线 token 准确率 + 切换边界 F1，(b) VLM-judge (GPT-4o) 语义合理性，(c) 闭环 LIBERO 在第三步 VLA 上替换 oracle subtask。
- **算力**：≤2×H100 即可（LoRA 训练 + 短上下文）。

**目标**：在 openpi 上构建第一份**开源 π0.7 风格 HL Policy**，并通过三层评估证明它能在闭环 LIBERO 上替代人工 subtask 输入，与第四步 World Model + 第三步 VLA 拼接出完整三组件 Algorithm 1。

---

## 1. π0.7 论文核心要点（精炼）

| 要素 | 论文做法（§IV / §V.A / §VII / §IX.D） |
|---|---|
| **架构** | SigLIP + Gemma 3 4B，**无 action expert**；纯自回归文字生成 |
| **输入** | 多视角当前观察 + 主任务 ℓ + 历史 subtask + episode metadata |
| **输出** | 下一个 subtask 字符串 ℓ̂_t（如 "pick up the knife"） |
| **训练 loss** | 标准 next-token CE，`E_D[log π_HL(ℓ̂_t \| o_t, ℓ, history)]` |
| **数据** | Human coaching data —— 人在演示时口述 subtask；私有，规模未公开 |
| **更新频率** | 每 4 秒 OR ℓ̂ 语义变化（Algorithm 1 line 7） |
| **推理模式** | 异步、非阻塞，VLA 拿"上一次输出" |
| **可替代性** | 论文明确说可被人工 / coaching 输入直接替代（§VII） |
| **与 VLA 关系** | 完全分离训练；推理时 HL 输出 ℓ̂ → 喂给 World Model 与 VLA |

**论文未明确的点**（我们必须自己定）：context 长度、token 预算、LR/batch/steps、是否 hindsight relabeling、coaching 数据规模。本方案对每条都做了显式选择（见 §6）。

**论文也未做**：HL 自身的离线指标（无 BLEU、F1、accuracy 报告）—— 这是社区可补的空白。

---

## 2. 社区现状

- **没有任何 open-source 复现** π0.7 高层策略；openpi 完全没有 subtask 生成基础设施（确认：grep "high_level"/"coaching"/"subtask generation" 无命中）。
- 最近的可借鉴是 [pi0_fast.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/models/pi0_fast.py) 的 `sample_actions` AR 解码循环 —— 它是 token-level 自回归生成，把 action token 换成文字 token 即是 HL 的骨架。
- π0.5/π*0.6 风格的"统一模型 KI 联合训练"也无开源版本，但**那不是 π0.7 路线**，本方案不走。

→ 本方案产出将是**社区首份开源 π0.7 风格独立 HL Policy + 与 VLA 异步闭环系统**。

---

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────┐
│ HighLevelPolicy（JAX，Pi0HL 类）                          │
│  SigLIP（frozen）+ Gemma 2B（LoRA）                       │
│  输入: 多视角 obs_t + main_task ℓ + history of ℓ̂        │
│  输出: 下一个 subtask 字符串（AR 生成 ≤ 32 token）        │
│  延迟: ~150ms / 调用（单卡 bf16）                         │
└─────────────────────┬────────────────────────────────────┘
                      │ ℓ̂_t 字符串
                      ▼
┌──────────────────────────────────────────────────────────┐
│ HLInferenceServer（Python 线程，独立于 VLA 主循环）        │
│  触发: (4s 超时) OR (上次输出与当前不一致)                 │
│  输出 ℓ̂_t → 同时投喂第四步 BagelInferenceServer           │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│ 第四步 BagelInferenceServer → subgoal image g_t            │
│ 第三步 VLA prefix 消费 (ℓ̂_t, g_t, metadata, mode)         │
└──────────────────────────────────────────────────────────┘
```

**进程模型**：单 Python 进程，**三线程**（VLA 主线程 + HL 线程 + WM 线程）。
- HL 在 JAX 主线程之外用 `threading.Thread`；JAX jit 后的解码函数 GIL 友好。
- VLA、HL、WM 都 GPU-bound，CUDA kernel 释放 GIL。
- 多卡：HL 与 VLA 同卡（HL 推理便宜），WM 钉到另一卡。

**接口契约**（与第四步对称）：

```python
class HighLevelPolicy:
    def __init__(self, ckpt_path: str): ...
    def generate_subtask(
        self,
        current_views: dict[str, np.ndarray],     # 同 VLA: base_0_rgb / left_wrist_0_rgb
        main_task: str,                           # 大任务指令
        history_subtasks: list[str],              # 已完成的子任务（从老到新）
        max_new_tokens: int = 32,
        temperature: float = 0.0,                 # greedy 默认
    ) -> str: ...

class HLInferenceServer:
    def start(self): ...
    def submit(self, current_views, main_task, history): ...   # 非阻塞
    def latest(self) -> str | None: ...                        # 最新已完成的 ℓ̂
    def changed_since_last_call(self) -> bool: ...             # 让 WM 据此判断是否要刷新
    def stop(self): ...
```

**与第四步 + 第三步的拼装**：
```python
# 在 LiberoInputs.__call__ 末段
hl_server.submit(views, main_task, history)
hat_ell = hl_server.latest() or main_task   # 冷启动 fallback
if hl_server.changed_since_last_call():
    wm_server.submit(views, hat_ell, metadata)
inputs["subtask"] = hat_ell
inputs["subgoal_images"] = wm_server.latest()    # 第四步逻辑
```

---

## 4. openpi 改造点

### 4.1 模块差异表

| 模块 | 当前状态 | 需改动 |
|---|---|---|
| `src/openpi/models/pi0_high_level.py`（**新建**） | 无 | 新建 `Pi0HL` 类（mirror `Pi0FAST` 但去掉 action expert，prefix 加 history） |
| `src/openpi/models/pi0_high_level_config.py`（**新建**） | 无 | `Pi0HLConfig`：`paligemma_variant="gemma_2b_lora"`, `max_history_subtasks=4`, `max_subtask_len=32` |
| `src/openpi/models/tokenizer.py` | `FASTTokenizer` 处理 action token | 加 `HLTokenizer`：纯文本 prompt 构造 + label tokenize（无 FAST action） |
| `scripts/prepare_libero_hl_data.py`（**新建**） | 无 | 读第三步的 subtask 标注 parquet → 转 (o_t, main_task, history) → next_subtask 样本（详 §5） |
| `scripts/train_high_level.py`（**新建**） | 无 | 镜像 `scripts/train.py`（JAX 主线），跑 CE loss |
| `scripts/eval_high_level.py`（**新建**） | 无 | 离线 token-level + VLM-judge（详 §8） |
| `src/openpi/policies/libero_policy.py` | 第三步加了 `subtask` / `subgoal_images` 字段 | 加可选 `hl_server` hook：eval 时拉 `latest()` 注入 subtask |
| `src/openpi/training/config.py` | 第三步注册了 `pi05_div_libero` | 注册 `pi0_hl_libero_lora_r16` HL 训练 config |
| `pyproject.toml` | 第四步加了 `[world_model]` 组 | 新增 `[high_level]` optional group：`openai`（VLM judge），其余复用 |
| 第三步 / 第四步 | 不动 | 完全只读复用 |

### 4.2 `Pi0HL` 模型骨架（mirror Pi0FAST，关键差异）

```
Pi0FAST                                Pi0HL（本步）
─────────────────────────────────────────────────────────────────
输入: 多视角 obs + state + task          输入: 多视角 obs + main_task + history
prefix: 图像 + state + "Task: …"        prefix: 图像 + main_task + 拼接 history
suffix: action FAST tokens (CE)         suffix: subtask 字符串 tokens (CE)
sample_actions: AR 出 action token       generate_subtask: AR 出文字直到 EOS
PaliGemma 2B + (FAST action head)       PaliGemma 2B（去掉 action 相关）
```

**Prompt 模板**（HLTokenizer 构造）：
```
Task: {main_task}.
Done so far: {", ".join(history_subtasks) or "<none>"}.
Next subtask:|{label_tokens}<eos>
```
`|` 之前是 prefix（双向 attention，loss=False），`|` 之后是 suffix（causal，loss=True）。

**LoRA**：复用 openpi 已有 `gemma_2b_lora` 变体（[src/openpi/models/gemma.py:55](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/models/gemma.py)），rank=16，仅在 LLM 块；SigLIP frozen。

### 4.3 与第三步 ckpt 的接口

**热启动**：从第三步 `pi05_div_libero` ckpt 加载 `PaliGemma`（SigLIP + Gemma）权重，丢弃 action expert / action_in_proj / time_mlp（HL 不需要）。LoRA 适配器 init=0 → forward 与 base 等价（HL-V-1 校验）。

**为什么热启动**：第三步 VLA 已经在 LIBERO 视觉 + 语言条件下"看过"了完整轨迹分布；HL 只是换了输出头（文字而非动作）。Cold-start PaliGemma 收敛慢且容易 hallucinate。

---

## 5. 数据构建

### 5.1 数据流：第三步标注 → HL 训练样本

第三步 [scripts/prepare_libero_subtasks.py](../../Research%20Exploration/Foundation%20Model/openpi/scripts/prepare_libero_subtasks.py) 产出每个 episode 的 `[(s_i, e_i, subtask_i), …]` segment 列表。HL 数据生成（`scripts/prepare_libero_hl_data.py`）：

对每个 episode，每个 subtask 段 $i$（含 $s_i$ 到 $e_i$）：
1. **正样本采样点**：在 $[s_i, e_i]$ 内均匀采 $K=8$ 个时间点 $t$。
2. 对每个 $t$：
   - `current_views` = obs_t 多视角
   - `main_task` = episode 顶层指令（LIBERO 自带）
   - `history_subtasks` = 已完成的 subtasks `[subtask_0, ..., subtask_{i-1}]`
   - **label** = `subtask_i`（**当前正在执行的**子任务文字 —— 等价于"接下来要做"）
3. **边界增强**（关键）：在每个 segment 末端 1s 内额外采 $K_b=4$ 点，标签**仍是 `subtask_i`**（不是 $i+1$，避免模型学到"末尾切换"的捷径）。
4. **切换样本**：在 $e_i$ 之后的第 1 步采一个样本，标签改为 `subtask_{i+1}` —— 教会模型识别切换。

**输出**：`data/libero_hl_pairs.parquet`，列：`episode_id, t, view_paths, main_task, history_json, label_subtask, sample_kind ∈ {regular, boundary, switch}`。

### 5.2 数据规模估算

- LIBERO 全套 ~500 episode × 平均 3-5 subtask 段 × 8 采样点 ≈ **15k 训练样本**。
- 加 boundary + switch 增强 → ~25k 样本。
- LoRA r=16 + 25k 样本 + ~10 epoch 在 A100 上 ~3 小时收敛。

**数据量是否够**：对于 LIBERO 这种**封闭词表**（约 50–100 个独立 subtask 短语，跨任务族）和**短指令**（≤ 8 词），25k 远超 NLP 经验下的 LoRA 收敛要求（通常 1–5k 即可，参考 instruction-tuning 文献）。**完全不需要 web-scale**。如果 HL 在 LIBERO 内泛化失败（HL-V-3 不达标），优先排查的是数据分布 / 模板偏置，而非数据量。

### 5.3 Holdout 划分

- 每个 LIBERO 任务族（Spatial / Object / Goal / Long）各保留 5 episode，共 **20 episode**（与第四步同一 holdout 集，便于联评）。
- Holdout 同时提供：(a) 离线 (current, history, label) 三元组用于 HL-V-3，(b) 闭环 rollout 起点用于 HL-V-5。

### 5.4 数据需求总览

| 数据 | 是否必需 | 来源 | 标注成本 |
|---|---|---|---|
| LIBERO 全量 demo | ✅ | 公开 | 0 |
| 第三步子任务标注 | ✅ | 第三步已生成 | 0（复用） |
| **新人工标注** | ❌ | — | **0** |
| GPT-4o（HL-V-4 评测用） | 仅 eval | OpenAI Batch API | ~$5/次 |

→ **不需要大规模数据集**。整套 HL 复现完全在第三步已有数据基础上完成。

---

## 6. 训练计划

### 6.1 超参（论文未给，本方案选择）

| Knob | Value | 理由 |
|---|---|---|
| Loss | Next-token CE on suffix only | 标准 instruction-tuning |
| Backbone | PaliGemma 2B（gemma_2b_lora） | §决策已锁定 |
| 冻结 | SigLIP 全冻；Gemma 2B 全冻；只训 LoRA | 防止 LIBERO 把通用语言能力蚀掉 |
| LoRA | r=16, alpha=32, dropout=0.05；attention q/k/v/o + MLP | openpi 现成变体 |
| Optimizer | AdamW(β1=0.9, β2=0.95, wd=0.01) | openpi 标准 |
| LR | 2e-4（LoRA 标准） | LoRA 经验值 |
| Schedule | 200 warmup → cosine decay → 1e-5 | |
| Batch | global 64（8/GPU × 2 H100） | 短样本，显存富裕 |
| Steps | **8k**（约 ~10 epoch on 25k 样本） | 小数据集，再多过拟 |
| Max history len | 4 个 subtask（拼成单字符串） | LIBERO 最长 ~4 个 segment |
| Max label len | 32 token | LIBERO subtask 短语典型 ≤ 8 词 ≈ 16 token，留 buffer |
| Temperature | 训练 N/A；推理默认 0（greedy） | 决定性输出便于复现与评测 |
| Mixed precision | bf16；LoRA 参数 fp32 master | |

### 6.2 算力预算

| 阶段 | H100-hr |
|---|---|
| W1 数据 + smoke test | ~2 |
| W2 LoRA 训练（8k step on 2×H100，~3h wall × 2 GPU） | ~6 |
| W3 离线 eval（HL-V-3 / V-4） | ~3 |
| W4 闭环 LIBERO sweep（4 suite × 50 trial × 4 condition） | ~30 |
| W5 ablation（HL-V-7） | ~10 |
| **合计** | **~50** |

比第四步（~360）小一个数量级 —— HL 是整个 π0.7 复现里**最便宜**的模块。

### 6.3 训练入口

`scripts/train_high_level.py` 镜像 `scripts/train.py`（JAX 主线）：
- 复用 `openpi.training.data_loader.create_data_loader` 但接 `LiberoHLDataset`（读 §5.1 parquet）。
- 复用 `init_wandb` / `Checkpointer`。
- 主 loss = `Pi0HL.compute_loss`（CE on suffix tokens）。
- 保存：LoRA adapter + tokenizer + config（小，<100MB）。

---

## 7. 推理 Runtime（异步）

### 7.1 线程设计（mirror 第四步）

```python
class HLInferenceServer:
    def __init__(self, hl: HighLevelPolicy, refresh_seconds: float = 4.0):
        self._hl = hl
        self._lock = threading.Lock()
        self._latest: str | None = None
        self._latest_changed = False
        self._req_event = threading.Event()
        self._req_payload = None
        self._stop = False
    # producer:
    #   wait → snapshot payload → hl.generate_subtask() → 写 _latest（与上次比，set _latest_changed）
    # 触发条件: (≥4s 自上次完成) OR （history 字段发生变化，例如外部传入新 done subtask）
```

**正确性约束**（论文 Algorithm 1 line 6–13）：
1. HL 第一次完成前，`latest()` 返回 `None` → VLA 走第三步 dropout sentinel（无 subtask 路径）。
2. HL 输出与上次不同时 set `_latest_changed=True` → 第四步 WM **立即**重新 submit（论文 line 7）。
3. HL 与 WM 总刷新时机解耦：HL 4s 周期，WM 紧跟 HL 切换或自身 4s 周期。

### 7.2 Hook 进 LIBERO eval

修改 `LiberoInputs.__call__`（在第三步、第四步基础上递增）：

```python
if self.hl_server is not None:
    self.hl_server.submit(views, main_task, self._history)
    hat_ell = self.hl_server.latest()
    if hat_ell is not None:
        inputs["subtask"] = hat_ell
        if self.hl_server.changed_since_last_call() and self.wm_server is not None:
            self.wm_server.submit(views, hat_ell, metadata)
        # 维护 history（去重 + 防抖）：
        if hat_ell != (self._history[-1] if self._history else None):
            self._history.append(hat_ell)
            self._history = self._history[-4:]
    # else 走第三步 fallback
```

**训练路径不变**（用 GT subtask 标注，§5.1）。

### 7.3 Latency 预算

目标：单卡 H100 上 median `generate_subtask()` < 300ms（max_new_tokens=32，bf16，greedy）。
- 实际预期 100–200ms（2B 模型 + KV cache + 短 suffix）。
- **远小于** 4s 刷新周期，不会成为瓶颈。

降级路径（基本不会触发）：
1. 限制 max_new_tokens=16。
2. 把 history 缩到 2 个。
3. 多卡分配（不太必要）。

---

## 8. 验证矩阵（mirror 第四步 WM-V 格式）

| ID | 测试 | 通过条件 |
|---|---|---|
| **HL-V-1** | LoRA 等价 + forward smoke | LoRA init=0 时 prefix logits 与 base PaliGemma bit-exact；2 GPU FSDP forward 不 OOM |
| **HL-V-2** | 单 batch overfit（1 sample × 500 step） | suffix CE < 0.05；目视 greedy decode 输出与 label 完全一致 |
| **HL-V-3** | 离线 token-level（20 holdout episode → ~5k 样本） | (a) **Top-1 exact-match ≥ 60%** 在常规样本上；(b) **Subtask switch detection F1 ≥ 0.55**（switch 样本上能识别"换到下一个"）；(c) Prefix-token CE 单调下降至 ≤ 0.4 |
| **HL-V-4** | VLM-judge 语义合理性 | 200 holdout (current_image, main_task, history, generated ℓ̂) → GPT-4o 0–5 打分。**通过：mean ≥ 3.7 且 ≥ 75% 样本 ≥ 3** |
| **HL-V-5** | 闭环 LIBERO 替换 oracle | 4 suite × 50 trial × **4 条件**：(a) 第三步 VLA 无 subtask（下界），(b) +oracle subtask（上界），(c) +HL 生成 subtask（本步成果），(d) +HL 生成 + 第四步 WM 生成 subgoal（**完整 π0.7 三组件**）。**通过：c ≥ b − 4pp，且 d ≥ b − 6pp**，且 d 与 c 在 Long suite 上的差异显示 WM 有正贡献 |
| **HL-V-6** | Async timing | median `generate_subtask()` < 300ms；VLA 主循环 step time variance ≤ baseline +3% |
| **HL-V-7** | Ablation（每条 4k step） | (i) 去 history（只看当前帧 + main task），(ii) cold start（不用第三步 ckpt），(iii) full-FT vs LoRA。报 ΔExact-match / ΔSuccess |
| **HL-V-8** | 与第四步联评一致性 | HL ℓ̂ 输出稳定后再喂 WM；WM 生成图与"HL ℓ̂ + GT 未来帧"的 FID 差距 ≤ 5（保证 HL 不引入显著 OOD prompt 让 WM 失效） |

### 8.1 VLM-judge rubric（HL-V-4，固定 in `vlm_judge.py`）

```
You are evaluating a robot high-level policy that proposes the next subtask.

Inputs:
- current observation: <img>
- main task: "{main_task}"
- already completed subtasks: {history}
- proposed next subtask: "{generated_subtask}"

Score 0–5:
  5 — proposed subtask is exactly what a competent robot should do next
  4 — reasonable next step, minor phrasing or sequencing issues
  3 — plausible but not optimal (e.g., out of order but feasible)
  2 — unrelated to current scene state
  1 — already-completed action restated, or contradicts main task
  0 — incoherent / wrong / not actionable

Output JSON: {"score": int, "reason": str}
```

OpenAI Batch API，单次 V-4 ~$5。

### 8.2 闭环成功率拆解（HL-V-5 是核心）

| 条件 | 含义 | 期待成功率（粗估，相对 oracle b） |
|---|---|---|
| a 无 subtask | 第三步 dropout sentinel 路径 | b − 10~15pp（Long 任务上差距最明显） |
| b oracle subtask | 数据集 GT 标注，**上界** | 100%（基线） |
| c HL 生成 subtask + GT 未来帧 subgoal | 本步纯 HL 贡献 | b − 0~4pp（目标） |
| d HL ℓ̂ + WM g* | 完整 π0.7 三组件 | b − 0~6pp（目标） |

c vs b 的 gap 是 **HL 自身的迁移损失**；d vs c 是 **WM 注入 subgoal 在 HL ℓ̂ 条件下的增益/损失**。Long suite 是关键（subtask 切换最频繁）。

---

## 9. 风险 & Mitigations

| 风险 | 应对 |
|---|---|
| LIBERO subtask 词表小（~50–100 短语）→ HL 退化为 lookup | LoRA 低 rank + 早停 + holdout 严格按 episode 切分（不漏数据） |
| 切换时机错误（HL 提前/滞后切换） → 闭环失败 | §5.1 boundary + switch 增强样本；HL-V-3 单独评 switch F1 |
| 训练用 GT history，推理用模型自己累积的 history → exposure bias | 训练时 50% 概率把 history 替换为"上一个采样点的预测"（teacher-forcing → schedule sampling）。**仅当 HL-V-5 不达标时启动**这个 Phase B |
| 第三步 ckpt 与 HL prefix 格式不兼容 | HLTokenizer 完全独立于 FASTTokenizer；只共享 PaliGemma 权重，不共享 prompt 模板 |
| LIBERO 主任务与 subtask 文字重叠 → HL 直接复读 main task 也能"看似对" | HL-V-3 检查 exact-match-with-main-task 比例；超过 30% 视为退化 |
| 异步双线程 jaxlib 状态污染 | HL JAX 线程独立 `jax.devices()` 句柄；首调 `jax.block_until_ready` 预热 |
| greedy 解码 in 高熵任务下卡死复读同一句 | repetition penalty=1.1（KV-cache 兼容）；HL-V-3 加 distinct-1/2 检查 |
| Coaching 数据量小 → HL 在新场景泛化差 | 本步明确范围限于 LIBERO；跨域泛化（DROID 等）作为 Future Work 而非本步目标 |
| HL 输出影响 WM → 联评不稳定 | HL-V-8 单独验：HL ℓ̂ 喂入 WM 时 FID 漂移可控 |

---

## 10. 文件级改动清单

```
新增（自包含）
├── src/openpi/models/
│   ├── pi0_high_level.py              # Pi0HL（mirror Pi0FAST，去掉 action expert）
│   ├── pi0_high_level_config.py       # Pi0HLConfig
│   └── tokenizer.py                   # 内部加 HLTokenizer 类（与 FASTTokenizer 并列）
├── src/openpi/inference/
│   └── hl_inference_server.py         # HLInferenceServer（线程模型）
├── scripts/
│   ├── prepare_libero_hl_data.py      # 第三步标注 → HL parquet
│   ├── train_high_level.py            # JAX 训练入口（mirror train.py）
│   └── eval_high_level.py             # 离线 V-3/V-4 + 闭环 V-5 联调
└── docs/
    └── pi07_high_level_eval_protocol.md   # HL-V-3..V-8 严格定义版

修改（精确触点）
├── src/openpi/policies/libero_policy.py
│   └── LiberoInputs 加 optional hl_server / 维护 self._history；
│       __call__ 末段：submit → latest() → 注入 subtask；
│       与第四步 wm_server hook 顺序耦合（HL 改 → WM 重生）
├── src/openpi/training/config.py
│   └── 注册 "pi0_hl_libero_lora_r16" config（指向 §6.1 超参）
└── pyproject.toml
    └── [project.optional-dependencies] 新增 high_level 组：openai（其余 numpy/jax 已有）

复用（不动）
├── scripts/prepare_libero_subtasks.py      # 第三步产物（数据来源）
├── 第三步 pi05_div_libero ckpt             # 用于 PaliGemma 热启动
├── 第三步 LiberoInputs 的 subtask 字段     # 接收 HL 输出的下游接口
└── 第四步 BagelInferenceServer + ckpt      # HL-V-5 条件 d / HL-V-8 联评
```

---

## 11. 端到端验证流程（每个 PR / 每个阶段必须满足）

1. **HL-V-1** LoRA 等价 + smoke：init=0 时与 base bit-exact。
2. **HL-V-2** overfit 单 batch：500 step 后 greedy decode 完全匹配。
3. **HL-V-3** 离线：top-1 exact-match ≥ 60%；switch F1 ≥ 0.55；CE ≤ 0.4。
4. **HL-V-4** VLM judge：mean ≥ 3.7 且 ≥ 75% 样本 ≥ 3。
5. **HL-V-5** 闭环 LIBERO：c ≥ b − 4pp 且 d ≥ b − 6pp；Long suite 上 d > a + 3pp。
6. **HL-V-6** Async latency：median < 300ms；VLA 主循环 stall ≤ baseline +3%。
7. **HL-V-7** Ablation：history / cold-start / full-FT 三条件 Δ 报告。
8. **HL-V-8** 与第四步联评：HL ℓ̂ 输入下 WM FID 漂移 ≤ 5。

---

## 12. 时间线

```
W1  数据 + 加载烟测                                   ~2 H100-h     → HL-V-1
    ├─ scripts/prepare_libero_hl_data.py（含 boundary/switch 增强）
    ├─ HLTokenizer + Pi0HL forward smoke
    └─ 从第三步 ckpt 加载 PaliGemma 权重路径打通

W2  LoRA 训练                                          ~6 H100-h     → HL-V-2, V-3
    ├─ scripts/train_high_level.py 跑通
    ├─ 8k step on 25k 样本（2×H100, ~3h wall）
    └─ 离线 token-level 评估

W3  VLM-judge + Async server                          ~3 H100-h     → HL-V-4, V-6
    ├─ vlm_judge.py（200 sample, GPT-4o）
    ├─ HLInferenceServer 实现
    └─ latency 测试

W4  闭环 LIBERO 联评                                   ~30 H100-h    → HL-V-5, V-8
    ├─ LiberoInputs hook（与第四步 wm_server 拼装）
    ├─ 4 suite × 50 trial × 4 condition
    └─ 与第四步 WM 联评一致性检查

W5  Ablation + report                                  ~10 H100-h    → HL-V-7
    ├─ 三个 ablation 各 4k step
    └─ 写技术报告 / blog
```

**W2 末**：里程碑 —— HL 离线指标达标，可独立发"open-source π0.7 高层策略 LoRA 复现"。
**W4 末**：**最终里程碑** —— **首份开源 π0.7 完整三组件 (HL + WM + VLA) 闭环系统**，Algorithm 1 完全跑通。
**W5 末**：完整 ablation + 技术报告 → 可投 workshop / 开源 release。

---

## 13. 关键参考资料

**论文**
- π0.7：[arXiv:2604.15483](https://arxiv.org/abs/2604.15483)（特别 §IV、§V.A、§VII、§IX.D、Algorithm 1）
- π0.7 blog：[pi.website/blog/pi07](https://www.pi.website/blog/pi07)
- π0.5（统一 KI 联合训练，对比基线）：[arXiv:2505.23705](https://arxiv.org/abs/2505.23705)

**openpi 关键文件**
- [src/openpi/models/pi0_fast.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/models/pi0_fast.py)（**Pi0HL 直接父模板**：AR 解码、CE loss）
- [src/openpi/models/tokenizer.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/models/tokenizer.py)（FASTTokenizer 模式参考）
- [src/openpi/models/gemma.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/models/gemma.py)（gemma_2b_lora 变体）
- [src/openpi/policies/libero_policy.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/policies/libero_policy.py)（VLA 输入端钩子点）
- [scripts/train.py](../../Research%20Exploration/Foundation%20Model/openpi/scripts/train.py)（JAX 训练模板）
- [src/openpi/training/config.py](../../Research%20Exploration/Foundation%20Model/openpi/src/openpi/training/config.py)（config 注册）

**前置阶段 plan**
- 第一步 KI：[openpi-pi0-7-step1-knowledge-insulation.md](openpi-pi0-7-step1-knowledge-insulation.md)
- 第二步 MEM：见根 plan
- 第三步 Diverse Context + Dropout：[openpi-pi0-7-step3-diverse-context-dropout.md](openpi-pi0-7-step3-diverse-context-dropout.md)（**HL 训练数据来源 + 热启动 ckpt**）
- 第四步 World Model：[openpi-pi0-7-ki-mem-dicerse-context-con-proud-reef.md](openpi-pi0-7-ki-mem-dicerse-context-con-proud-reef.md)（**HL-V-5 条件 d / HL-V-8 联评对象**）

**根 plan + 设计文档**
- [pi0-7-pi0-5-pi0-6-pi-0-6-pi0-7-pi0-7-cozy-minsky.md](pi0-7-pi0-5-pi0-6-pi-0-6-pi0-7-pi0-7-cozy-minsky.md)
- [docs/pi07_highlevel_subtask_design.md](../../Research%20Exploration/Foundation%20Model/openpi/docs/pi07_highlevel_subtask_design.md)（π0.5 vs π*0.6 vs π0.7 高层设计对比 + 路线 A/B/C 讨论；本方案走**路线 B**）