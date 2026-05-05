# π0.7 第四步：World Model（BAGEL）复现方案

## Context

我们正在基于 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) **逐模块**复现 π0.7（[arXiv:2604.15483](https://arxiv.org/abs/2604.15483)）。**第一步 KI**、**第二步 MEM**、**第三步 Diverse Context Conditioning + Dropout** 已完成；本方案聚焦**第四步：World Model**——为推理时缺失的 subgoal 图像 $\mathbf{g}_t$ 提供生成器。

**第三步已经在 `Observation` 中预留了 `subgoal_images` 槽位，并在 `embed_prefix` 中通过 SigLIP + type-embedding 消费它**。本步骤的目标就是**填这个槽位**：在推理时，根据当前观测 + 子任务文本 + metadata，异步生成 4 秒后的多视角未来帧。

**已锁定决策（用户已选）**：
- **Backbone**：`ByteDance-Seed/BAGEL-7B-MoT`（PyTorch，~14B 总参 / ~7B 激活，MoT = Qwen2.5-7B LLM expert + 图像生成 expert + FLUX VAE + SigLIP2 understanding tower）。论文原版选型。
- **Adaptation**：`peft` 库的 LoRA。**不做** full fine-tune（14B + LIBERO 小数据 = 必然过拟合）。
- **Inference**：异步 4 秒 refresh，独立线程，VLA 端 RTC buffering（论文 Algorithm 1）。
- **Evaluation**：三模态——(a) 闭环 LIBERO 成功率，(b) 图像质量 FID/LPIPS/SSIM，(c) VLM-judge（GPT-4V）语义正确性。
- **数据**：LIBERO + 第三步生成的子任务标注 +（可选）DROID 子集。**没有** web-scale。
- **算力**：≥8×H100，不是瓶颈。

**目标**：在 openpi 上构建第一份**开源 pi0.7 风格 BAGEL world model + async runner**，并通过三模态评估证明其在闭环 LIBERO 上的有效性。

---

## 1. π0.7 §V-B + Appendix C 核心要点（精炼）

| 要素 | 论文做法 |
|---|---|
| **架构** | 从 BAGEL 14B 初始化（image gen + understanding 一体的 MoT 模型） |
| **训练损失** | $L_{CFM}(g_t', g_o(o_t, \ell_t, m))$ ——VAE latent 上的 conditional flow matching |
| **输入** | 当前观测 $o_t$（多视角）+ subtask $\ell_t$ + metadata $m$ |
| **输出** | 多视角未来帧（论文 448×448，本方案 256→224 适配 LIBERO） |
| **训练目标采样** | 25% 取 segment 末帧；75% 在 0–4s 内均匀采样 |
| **真实+生成混合** | 用已训出的 WM 自采样图替换部分 context，缓解 train-test gap（§V-B 后段） |
| **推理刷新** | 每 4 秒 OR subtask 改变时刷新；独立线程，VLA RTC 容延迟 |
| **数据来源** | 机器人 demo + 自主 rollout + 人体第一视角视频 + web 图像编辑数据 |

**与第三步的接口**：第三步 VLA 已支持空 subgoal（dropout sentinel），所以 WM 启动前的 warmup 期间 VLA 能 fallback 到无 subgoal 模式；WM 上线后，VLA 自动获得"含 subgoal"的能力增益。

---

## 2. 社区现状

- **没有任何 open-source 复现** pi0.7 world model；BAGEL 自身有 HF 公开权重但没人把它接到 VLA 上。
- 最相近参考是 [SuSIE](https://github.com/kvablack/susie)（IP2P 风格 InstructPix2Pix subgoal 生成），但单视角、Stable Diffusion 1.5 量级、且不和 VLA 闭环。
- openpi 当前**完全没有** image generation / VAE / world model 基础设施——本步骤为 greenfield。

→ 本方案产出将是**社区首份开源 BAGEL-VLA 闭环复现**。

---

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│ Step-3 VLA (JAX, Flax NNX) — pi05_div_libero ckpt             │
│  embed_prefix 消费 subgoal_images 槽位（224×224×3）            │
└──────────▲─────────────────────────────────────┬─────────────┘
           │ latest_subgoal (numpy, 共享内存槽位)│ obs, subtask
           │ (lock-protected)                    │ (每 control step)
┌──────────┴─────────────────────────────────────▼─────────────┐
│ BagelInferenceServer (PyTorch, 独立线程)                       │
│  ─ 持有 frozen base BAGEL + LoRA adapter                       │
│  ─ 生成多视角 256×256 → 双线性下采样到 224                    │
│  ─ 触发条件：(4s 经过) OR (subtask 字符串变化)                │
└──────────────────────────────────────────────────────────────┘
```

**进程模型**：单 Python 进程，**双线程**。
- 选 threading 而非 multiprocessing：PyTorch CUDA context 共享 OK；224×224×3 numpy 数组 IPC 开销不必要；BAGEL forward 是 GPU-bound，CUDA kernel 释放 GIL。
- 单卡 H100（80GB）：BAGEL bf16 ~14GB + step-3 VLA <10GB，无内存竞争。
- 多卡评估时把 BAGEL 钉在 `cuda:0`、VLA 钉在 `cuda:1`。

**接口契约**（唯一公开 API）：

```python
class BagelWorldModel:
    def __init__(self, base_ckpt: str, lora_ckpt: str | None,
                 device: str, dtype: torch.dtype): ...
    @torch.no_grad()
    def generate(
        self,
        current_views: dict[str, np.ndarray],   # {"base_0_rgb": uint8 HWC, "left_wrist_0_rgb": uint8 HWC}
        subtask: str,
        metadata: dict | None = None,
        num_steps: int = 25, cfg_scale: float = 3.0,
    ) -> dict[str, np.ndarray]:                 # 同 keys, uint8 224x224x3

class BagelInferenceServer:
    def start(self): ...
    def submit(self, current_views, subtask, metadata): ...   # 非阻塞
    def latest(self) -> dict[str, np.ndarray] | None: ...     # 最新已完成结果
    def stop(self): ...
```

VLA 控制循环每步调用 `submit()`；`latest()` 在第一次生成完成前返回 `None`（VLA 此时走第三步的 dropout sentinel 路径）。

**分辨率决策：256×256 训练 / 224×224 输出**。
- BAGEL VAE 要求 16 倍数；256 是和 LIBERO 原生 224 最接近的合法尺寸。
- 推理时在 server 端双线性下采样 256→224，让第三步 VLA 的 SigLIP 输入维度不变。

---

## 4. openpi 改造点

### 4.1 模块差异表

| 模块 | 当前状态 | 需改动 |
|---|---|---|
| `models_pytorch/world_model/`（新包） | 无 | **新建**完整子包（详见 §4.2） |
| `scripts/prepare_libero_world_model_data.py` | 无 | 新建：构建 (current_t, future_t+Δ) pair |
| `scripts/train_world_model.py` | 无 | 新建：mirror `scripts/train_pytorch.py` 风格 |
| `scripts/eval_world_model.py` | 无 | 新建：FID/LPIPS/SSIM + VLM-judge |
| `scripts/serve_world_model.py` | 无 | 新建：standalone async server（latency 测试用） |
| `src/openpi/policies/libero_policy.py` | 第三步已有 subgoal_images 字段 | 加可选 `wm_server` hook：eval 时拉 `latest()` 注入 subgoal |
| `src/openpi/training/config.py` | 第三步已有 `pi05_div_libero` | 注册 `bagel_libero_lora_r16` WM 训练 config |
| `pyproject.toml` | 无 WM 依赖 | 新增可选组 `[world_model]`：`peft, accelerate, diffusers, clean-fid, lpips, scikit-image, openai, google-generativeai` |
| `third_party/bagel/`（vendored） | 无 | 锁定一个 BAGEL commit，vendor 只读副本（参考现有 `models_pytorch/transformers_replace/`） |
| 第三步 VLA path | 不动 | 训练用 GT 未来帧不变；只在 eval 时注入 WM 输出 |

### 4.2 新包 `src/openpi/models_pytorch/world_model/` 结构

```
world_model/
  __init__.py              # 导出 BagelWorldModel, BagelInferenceServer
  bagel_loader.py          # HF snapshot_download + state-dict 适配
  bagel_wrapper.py         # BagelWorldModel：preprocess + generate + LoRA hook
  lora_config.py           # peft.LoraConfig 工厂（target_modules 列表）
  flow_matching.py         # VAE latent 上的 CFM loss + train helpers
  data.py                  # WorldModelSample dataclass + collate_fn
  inference_server.py      # threading.Thread + lock-protected slot
  metrics.py               # FID/LPIPS/SSIM 包装
  vlm_judge.py             # GPT-4V/Gemini rubric 调用
```

**为什么选 `models_pytorch/` 而非新顶层包**：BAGEL 是 PyTorch 模型，与已有 `models_pytorch/pi0_pytorch.py` 同级最合理；和 JAX 主线隔离干净。

### 4.3 LoRA 配置（`lora_config.py`）

**冻结部分**（不动）：
- Qwen2.5-7B LLM expert——不改语言理解。
- SigLIP2 understanding tower——不改视觉感知。
- FLUX VAE encoder/decoder——latent diffusion fine-tune 标准做法。

**LoRA 注入**（image-generation expert 的所有 attention + MLP 投影）：

```python
target_modules = [
    r"gen_expert\..*\.q_proj", r"gen_expert\..*\.k_proj",
    r"gen_expert\..*\.v_proj", r"gen_expert\..*\.o_proj",
    r"gen_expert\..*\.gate_proj", r"gen_expert\..*\.up_proj",
    r"gen_expert\..*\.down_proj",
]
# 注：gen_expert 实际命名以 BAGEL state_dict 为准，第一次加载后用 print(model.named_modules()) 校准
```

**超参**：
- `r=16, alpha=32, dropout=0.05, bias="none"`。
- 理由：LIBERO 小+低多样性 → r=16（约 0.3% 可训参）安全；r=8 容量不足，r≥32 + 30k step 容易让 BAGEL 通用图像先验漂移。
- 最终 adapter 文件 <200 MB。

### 4.4 输入 prompt 格式（`bagel_wrapper.py`）

BAGEL 期望多模态 token 流。我们构造：

```
<bos>
[SigLIP2 tokens of current_view_base]
[SigLIP2 tokens of current_view_wrist]
<task> {subtask string} </task>
<meta> speed={speed} quality={quality} </meta>
<gen_begin>
  [N latent tokens for base_0_rgb]       ← 联合生成
  [N latent tokens for left_wrist_0_rgb] ← 共享 attention
<gen_end>
```

**多视角联合生成**（决策：**joint**，非 per-view）：两视角 latent token grid 串接到同一 `<gen_begin>...<gen_end>` 块——两视角共享 attention 保证一致性。这是 BAGEL 原生多图生成模式。

**缓存策略**：
- **缓存**：训练目标的 FLUX VAE latent 编码（每 (episode, t_future) 一份，写到 `data/libero_wm_cache/`）。砍掉 ~30% 步时。
- **不缓存**：当前帧的 SigLIP2 features（每步都换）；subtask 字符串 tokenize（cheap）。

---

## 5. 数据构建

### 5.1 (current, future) pair 采样规则

`scripts/prepare_libero_world_model_data.py` 输出 `data/libero_wm_pairs.parquet`：

对每个 LIBERO episode（长度 $T$），结合第三步的 subtask 标注 `[(s_i, e_i, subtask_i), ...]`：

对每个 $t_\text{curr} \in [0, T)$：
1. 找到包含 $t_\text{curr}$ 的 segment $(s, e, \text{subtask})$。
2. 决定未来偏移 $\Delta$：
   - **25%**：$\Delta = e - t_\text{curr}$（segment 末帧）。
   - **75%**：$\Delta \sim \text{Uniform}\big(1, \min(4\text{s 步数},\ e - t_\text{curr},\ T-1-t_\text{curr})\big)$。
3. **Clip-to-end policy**（LIBERO episode 短，必须）：
   - 不跨 subtask 边界（保持 $\le e$）。
   - 不超出 episode 末尾（保持 $\le T-1$）。
   - 若 $e - t_\text{curr} < 1$，**丢弃**该样本。
4. 记录 metadata（speed/quality，从第三步流水线复用）。

**Holdout**：每个 LIBERO 任务族（Spatial/Object/Goal/Long）保留 5 episode（共 ~50）作 V-3/V-4/V-5 评估。

### 5.2 Phase A（必做）vs Phase B（可选）

| Phase | 做法 | 何时启动 |
|---|---|---|
| **Phase A** | 纯真实 (current, future) pair 上 CFM 训练 | 默认 |
| **Phase B**（论文 §V-B "generated images in context"）| Phase A 收敛后，10% 步数继续训练，以概率 $p_\text{gen}=0.5$ 把**当前帧**替换为 WM 自采样的"上一步预测帧" | **仅当 V-6（train-test gap）不达标时启动** |

### 5.3 多视角

**联合训练**，loss = mean(per_view_CFM_loss)。监控 per-view val loss；wrist view 因更小 + 更 OOD，若 plateau 则把 wrist:base 权重从 1:1 调到 1.5:1。

### 5.4 数据需求

| 数据 | 是否必需 | 来源 | 标注成本 |
|---|---|---|---|
| LIBERO 全量 demo | ✅ | 公开 | 0 |
| 第三步子任务标注 | ✅ | 第三步已生成 | 0（复用） |
| 第三步 metadata 缓存 | ✅ | 第三步已生成 | 0（复用） |
| BAGEL HF 权重 | ✅ | HF 公开 | 0 |
| DROID 子集 5k ep | ❌（可选） | CC-BY 4.0 | 0 |
| **新人工标注** | ❌ | — | **0** |

→ **完全公开数据**，无新标注成本。

---

## 6. 训练计划

### 6.1 超参

| Knob | Value | 理由 |
|---|---|---|
| Loss | CFM in FLUX VAE latent (bf16) | BAGEL 原生目标，无需重训 head |
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.01) | 大 MoT LoRA 标准 |
| LR (LoRA params) | 1e-4 | peft 标准；full-FT 才用 1e-5 |
| Schedule | 1k warmup → cosine decay → 1e-5 | |
| Global batch | 64（8/GPU × 8 H100） | 受显存限 |
| Steps Phase A | 30k | LIBERO 规模再多就过拟合 |
| Steps Phase B（条件） | 5k | 续训补 train-test gap |
| Grad checkpointing | **开**（仅 gen-expert 块） | 省 ~16GB activation |
| Mixed precision | bf16；LoRA 参数 fp32 master | |
| Distributed | **FSDP HYBRID_SHARD** on gen-expert（frozen 部分），LoRA 参数 DDP | LoRA 极小，sharding 收益为零 |
| Timestep schedule | logit-normal | BAGEL 默认 |
| KL regularizer | None；仅当 V-8 显示先验漂移再加 | |

### 6.2 算力预算

| 阶段 | H100-hr |
|---|---|
| W1 数据 + smoke test | ~10 |
| W2 LoRA 包装 + overfit | ~10 |
| W3 Phase A 主训练（30k step × ~3s/step × 8 GPU ≈ 25h wall × 8） | ~200 |
| W4 离线 eval（FID/LPIPS/VLM judge） | ~10 |
| W5 闭环 LIBERO sweep（4 suite × 50 trial × 3 condition） | ~80 |
| W6 V-6/V-8 ablation + Phase B（如需） | ~50 |
| **合计** | **~360** |

比第三步的 ~1500 H100-hr 小很多——LoRA 训练成本低，主要预算在闭环 eval。

### 6.3 训练入口

`scripts/train_world_model.py` 镜像 `scripts/train_pytorch.py` 风格：
- `setup_ddp()`（torchrun，复用现有模式）→ FSDP wrap。
- 复用 `init_wandb`。
- DataLoader 跑 §5.1 的 parquet，custom `collate_fn`：图像加载 + 实时 VAE 编码（或 §4.4 缓存命中）+ subtask tokenize。
- 保存：peft adapter（每 `save_interval`）+ `wm_step_state.pt`（optimizer/scheduler）。

---

## 7. 推理 Runtime（异步）

### 7.1 线程设计（`inference_server.py`）

```python
class BagelInferenceServer:
    def __init__(self, wm: BagelWorldModel, refresh_seconds: float = 4.0):
        self._wm = wm
        self._lock = threading.Lock()
        self._latest = None              # dict[str, np.ndarray] | None
        self._latest_subtask = None
        self._req_event = threading.Event()
        self._req_payload = None         # (current_views, subtask, metadata)
        self._stop = False
    # producer loop:
    #   wait on _req_event → snapshot _req_payload → wm.generate() → 写入 _latest under lock
    # 客户端 API：
    #   submit(...): 写 _req_payload，set _req_event
    #     仅当 (subtask 改变) OR (距上次 ≥ 4s) 时才真正触发
    #   latest():    lock 内 snapshot
```

**正确性约束**：
1. Producer 启动 generate 时**取最新** payload（中间提交都丢弃）。
2. **subtask 字符串变化**绕过 4 秒门限，立即重算（论文 Algorithm 1 line 7）。
3. `latest()` 首生成前返回 `None`——VLA 必须 handle，走第三步 dropout sentinel。

### 7.2 Hook 进 LIBERO eval

修改 `LiberoInputs.__call__`：

```python
if self.wm_server is not None:
    self.wm_server.submit(inputs["image"], data.get("subtask",""), data.get("metadata",{}))
    sg = self.wm_server.latest()
    if sg is not None:
        inputs["subgoal_images"] = sg
        inputs["subgoal_image_masks"] = {k: True for k in sg}
    # else: 缺字段，第三步 dropout sentinel 路径接管
```

**训练路径不变**（仍用 GT 未来帧，第三步设计）。

### 7.3 Latency 预算

目标：单卡 H100 上 median `generate()` < 4s（25 sampling step，bf16，双视角）。

降级路径（按顺序触发）：
1. sampling step 25 → 20（FLUX 系一般 20–30 都 OK）。
2. LLM expert int8 量化（`bitsandbytes`），gen-expert 保持 bf16。
3. 独立 GPU（`cuda:1` 跑 BAGEL）。

---

## 8. 验证矩阵（mirror 第三步 DC-V 格式）

| ID | 测试 | 通过条件 |
|---|---|---|
| **WM-V-1** | 单元 forward + LoRA 应用 | LoRA-on 输出在 adapter init=0 时与 LoRA-off 完全 bit-exact；FSDP × 2 GPU 单 batch forward 不 OOM |
| **WM-V-2** | 单 batch overfit（1 sample × 500 step） | CFM val loss < 0.05；可视化生成的未来帧——目视可识别为 GT 未来状态（双视角都过） |
| **WM-V-3** | 图像质量（5 LIBERO holdout × ~5k pair） | **FID ≤ 35**，**LPIPS ≤ 0.30**，**SSIM ≥ 0.55**（双视角平均）。基线校准：BAGEL 不 fine-tune 直接跑 LIBERO 大概率 FID ≥ 100 |
| **WM-V-4** | VLM-judge 语义正确性 | 200 holdout (current, generated_subgoal, subtask) 三元组 → GPT-4o 0–5 打分（rubric 在 `vlm_judge.py` 版本化）。**通过：mean ≥ 3.5 且 ≥ 70% 样本 ≥ 3** |
| **WM-V-5** | 闭环 LIBERO 成功率 | 4 suite（Spatial/Object/Goal/Long）× 50 trial × 3 条件：(a) 第三步 VLA 无 subgoal（下界），(b) 第三步 VLA + GT 未来帧 oracle subgoal（上界），(c) 第三步 VLA + WM 生成 subgoal。**通过：c ≥ b − 5pp 平均 且 c > a + 3pp on Long** |
| **WM-V-6** | Train-test gap stress | 同 V-5(c) 但 WM 强制每步 refresh（最坏 compounding error 情况）。**通过：相对 4s refresh 成功率下降 ≤ 8pp**。否则启动 Phase B |
| **WM-V-7** | Async timing | median `generate()` < 4.0s，p95 < 5.0s on 1×H100。VLA 控制循环不 stall（VLA step time variance ≤ baseline +5%） |
| **WM-V-8** | Ablation（每条 10k step） | (i) 去 metadata 条件，(ii) per-view independent（无联合 attention），(iii) LLM expert 也加 LoRA。报 ΔFID / Δsuccess |

### 8.1 VLM-judge rubric（WM-V-4 用，固定 in `vlm_judge.py`）

```
You are evaluating a robot subgoal image generator.

Inputs:
- current frame: <img1>
- generated subgoal frame: <img2>
- subtask description: "{subtask}"

Score 0–5:
  5 — subgoal image clearly shows the subtask completed
  4 — subgoal shows substantial progress, minor inaccuracies
  3 — subgoal shows partial completion or correct direction
  2 — subgoal is plausible but unrelated to subtask
  1 — subgoal is implausible / artifacts dominate
  0 — irrelevant / wrong / corrupt

Output JSON: {"score": int, "reason": str}
```

OpenAI Batch API 跑，单次 V-4 ~$10。

---

## 9. 风险 & Mitigations

| 风险 | 应对 |
|---|---|
| LIBERO 轨迹短于 4s → "0–4s ahead" 未定义 | §5.1 clip-to-end + clip-to-subtask-end，丢弃 horizon < 1 步样本 |
| LIBERO 低多样性 → 过拟合 / 失去 BAGEL 通用图像先验 | 低 rank（r=16）+ 仅 gen-expert + ≤30k step + 不动 LLM expert。WM-V-8(iii) 验证。Fallback：加 0.01·KL(WM‖frozen base) on held-out 通用 prompt |
| 14B 推理 > 4s on 单卡 | 默认 bf16；降级路径 §7.3。WM-V-7 量化 |
| Train-test mismatch（训用真实，测用生成） | §5.2 Phase B，由 WM-V-6 触发 |
| BAGEL HF state_dict 格式坑 | Vendor `third_party/bagel/`，`bagel_loader.py` 显式 mapping，第一次加载即 WM-V-1 |
| 闭架构调试难 | 每 500 step 把 gen-expert 每个块 forward 激活统计上传 wandb；"drift detector"：固定一组非-LIBERO held-out prompt，每 1k step 算 cosine(base output, LoRA output)，<0.7 报警 |
| FSDP × peft 兼容性（peft 包装时机错会被 FSDP 拒） | `peft.get_peft_model` **必须在 FSDP wrap 之前**；LoRA 参数 `_fsdp_wrap=False`，replicate（DDP 风格）；frozen base 才 shard |
| GPT-4V cost | 每次 V-4 cap 200 sample（~$10），OpenAI Batch API |
| 双线程 CUDA 竞争 | producer 线程显式 `with torch.cuda.stream(...)`；VLA jaxlib 默认主流；保险起见 BAGEL 预热一次再交给 server |
| vendored BAGEL 升级不及时 | pin 一个 commit；新版 BAGEL 出来时手动 review + 重跑 WM-V-1/V-2 再升 |

---

## 10. 文件级改动清单

```
新增（自包含）
├── src/openpi/models_pytorch/world_model/
│   ├── __init__.py
│   ├── bagel_loader.py          # HF download + state-dict 适配
│   ├── bagel_wrapper.py         # BagelWorldModel
│   ├── lora_config.py           # peft.LoraConfig 工厂
│   ├── flow_matching.py         # CFM loss in VAE latent
│   ├── data.py                  # WorldModelSample + collate_fn
│   ├── inference_server.py      # BagelInferenceServer
│   ├── metrics.py               # FID/LPIPS/SSIM
│   └── vlm_judge.py             # GPT-4V rubric
├── third_party/bagel/           # vendored BAGEL modeling code (pinned commit)
├── scripts/prepare_libero_world_model_data.py
├── scripts/train_world_model.py
├── scripts/eval_world_model.py
└── scripts/serve_world_model.py # standalone latency tester

修改（精确触点）
├── src/openpi/policies/libero_policy.py
│   └── LiberoInputs 加 optional wm_server 参数；__call__ 末段注入 subgoal
├── src/openpi/training/config.py
│   └── 注册 "bagel_libero_lora_r16" config（路径/batch/LR）
└── pyproject.toml
    └── 新增 [project.optional-dependencies] world_model 组：
        peft, accelerate, diffusers, clean-fid, lpips, scikit-image,
        openai, google-generativeai

复用（不动）
├── scripts/prepare_libero_subtasks.py     # 第三步产物
├── scripts/prepare_libero_metadata.py     # 第三步产物
└── 第三步 pi05_div_libero ckpt            # 闭环 eval 用
```

---

## 11. 端到端验证流程（每个 PR / 每个阶段必须满足）

1. **WM-V-1** forward + LoRA 等价：adapter init=0 时与 base 输出 bit-exact。
2. **WM-V-2** overfit 单 batch：500 step 后视觉收敛。
3. **WM-V-3** 离线图像质量：FID ≤ 35 / LPIPS ≤ 0.30 / SSIM ≥ 0.55。
4. **WM-V-4** VLM judge：mean ≥ 3.5 且 ≥ 70% sample ≥ 3。
5. **WM-V-5** 闭环 LIBERO：c ≥ b − 5pp 且 c > a + 3pp on Long。
6. **WM-V-6** 每步 refresh stress：相对 4s refresh 下降 ≤ 8pp（不达标 → Phase B）。
7. **WM-V-7** async latency：median < 4s，VLA stall 率 ≤ baseline +5%。
8. **WM-V-8** ablation 报告：metadata / 联合视角 / LLM-expert LoRA 三个条件的 ΔFID/Δsuccess。

---

## 12. 时间线

```
W1  数据 + 加载烟测                                   ~10 H100-h     → WM-V-1
    ├─ scripts/prepare_libero_world_model_data.py（含 25/75 + clip-to-end）
    ├─ bagel_loader.py + 1-batch forward smoke
    └─ FLUX VAE latent 缓存生成

W2  LoRA + 训练 harness                               ~10 H100-h     → WM-V-2
    ├─ peft 包装（FSDP-before-peft 顺序）
    ├─ scripts/train_world_model.py 跑通
    └─ 单 batch overfit

W3  Phase A 主训练（8×H100, ~25h wall）                ~200 H100-h    训练曲线
    └─ 30k step on LIBERO，wandb + drift detector

W4  离线 eval                                         ~10 H100-h     → WM-V-3, V-4
    ├─ FID/LPIPS/SSIM on 5 holdout × 5k pair
    └─ VLM-judge GPT-4V on 200 sample

W5  Async server + 闭环                               ~80 H100-h     → WM-V-5, V-7
    ├─ inference_server.py 实现 + latency 测
    ├─ LiberoInputs hook
    └─ 4 suite × 50 trial × 3 condition

W6  Stress + ablation + report                        ~50 H100-h     → WM-V-6, V-8
    ├─ 每步 refresh stress（必要时 Phase B 5k step）
    ├─ V-8 三个 ablation 各 10k step
    └─ 写 blog / 技术报告
```

**W4 末**：里程碑——离线生成质量达标，可发"BAGEL-on-LIBERO LoRA 复现" demo。
**W5 末**：里程碑——首份 open-source pi0.7-style 闭环 WM-VLA 系统跑通。
**W6 末**：完整可发表 ablation 报告。

---

## 13. 关键参考资料

**论文**
- π0.7：[arXiv:2604.15483](https://arxiv.org/abs/2604.15483)（特别 §V-B "Subgoal images"、Algorithm 1、Appendix C "Training of the world model"）
- π0.7 blog：[pi.website/blog/pi07](https://www.pi.website/blog/pi07)
- BAGEL 论文：参考 π0.7 引用 [148]（ByteDance MoT）
- SuSIE：[arXiv:2310.10639](https://arxiv.org/abs/2310.10639)（subgoal image generation 思路源头）
- FLUX VAE / Flow Matching：[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- RTC（real-time action chunking）：π0.7 引用 [107]

**模型与权重**
- BAGEL HF：`ByteDance-Seed/BAGEL-7B-MoT`
- peft：[huggingface/peft](https://github.com/huggingface/peft)

**openpi 关键文件**
- [src/openpi/models_pytorch/pi0_pytorch.py](src/openpi/models_pytorch/pi0_pytorch.py)（PyTorch 风格参考）
- [scripts/train_pytorch.py](scripts/train_pytorch.py)（trainer 模板）
- [src/openpi/policies/libero_policy.py](src/openpi/policies/libero_policy.py)（VLA 输入端钩子点）
- [src/openpi/training/config.py](src/openpi/training/config.py)（config 注册）
- [src/openpi/models_pytorch/transformers_replace/](src/openpi/models_pytorch/transformers_replace/)（vendored 第三方代码模板）

**前置阶段 plan**
- 第一步 KI：[openpi-pi0-7-step1-knowledge-insulation.md](openpi-pi0-7-step1-knowledge-insulation.md)
- 第二步 MEM：见根 plan
- 第三步 Diverse Context + Dropout：[openpi-pi0-7-step3-diverse-context-dropout.md](openpi-pi0-7-step3-diverse-context-dropout.md)（**本步骤的 VLA 端 ckpt 与 subgoal_images 接口由它产出**）

**根 plan**
- [pi0-7-pi0-5-pi0-6-pi-0-6-pi0-7-pi0-7-cozy-minsky.md](pi0-7-pi0-5-pi0-6-pi-0-6-pi0-7-pi0-7-cozy-minsky.md)
- [pi07_highlevel_subtask_design.md](../../Research%20Exploration/Foundation%20Model/openpi/docs/pi07_highlevel_subtask_design.md)