# HPT（Heterogeneous Pre-trained Transformer）in Atom-0

基于 [HPT (NeurIPS 2024)](https://arxiv.org/abs/2409.20537) / [liruiw/HPT](https://github.com/liruiw/HPT) 的异构 stem + 共享 trunk 设计，在 Atom-0 中对接 **pi05 同款 cotrain RLDS 数据流**（统一 80D 动作空间、`is_ego` 域标签、多数据集 mixture）。

实现为 **PyTorch-only**（与 FastWAM 相同接入方式），训练入口：`scripts/train_hpt.py`。

---

## 1. 目标与数据

三类数据（根目录 `/mnt/bos/bo23lu`，与 pi05 cotrain 一致）：

| 类型 | 内容 | 相机 |
|------|------|------|
| piper 自采 | `realworld_piper_*` | 头 + 双腕 |
| robot | agibot / robocoin / robomind 等 | 大多三路 |
| ego | `EgoVerse_full`（full） | 仅头部 |

- 动作 / 状态空间：**统一 80D**（见 `openpi/cotrain/action_space.py`、`rlds_dataset.py`）
- piper 监督：**14 维 native joint**（12 delta 关节 + 2 absolute gripper）；80D 中其余维为 pad，经 `action_mask` 屏蔽 loss
- ego / robot 通过 `DispatchNormalize` 写入 `is_ego`（`dataset_id.startswith("egoverse")`）

---

## 2. 模型结构

```text
              Human Ego Image ──► Ego Stem ────────┐
                                                   │
Robot Head Image ──► Robot Stem ───────────────────┤
                                                   │
Robot Wrist(s) ──► Wrist Stem ─────────────────────┤──► Shared HPT Trunk
                                                   │         │
Robot/Ego State ──► State Stem ────────────────────┤    ┌────┴────┐
                                                   │    ▼         ▼
Prompt ──► T5 (frozen) ──► Language Stem ──────────┘ Action-DiT  World Head
                                                      (flow)      (DINO)
                                                         │            │
                                                   robot action   future DINO
```

### 编码器（训练时冻结）

| 模块 | 默认 | 说明 |
|------|------|------|
| **DINOv2** | `facebook/dinov2-base`，768d | `freeze_encoders=True`；`train_hpt.py` 调用 `freeze_encoders()` |
| **T5-base** | 768d | 同上；stem 可训，encoder 不更新 |

### Stem（可训练）

| Stem | 输入 | 编码 |
|------|------|------|
| Ego / Robot / Wrist | 图像 | **冻结 DINOv2** → MLP + CrossAttn → 固定 token 数 |
| State | 80D proprio | MLP + CrossAttn |
| Language | 文本 prompt | **冻结 T5-base** → MLP + CrossAttn |

默认 token 数：ego/robot/wrist/state=**16**，language=**8** → 单时刻 **72** obs tokens。

### Trunk（共享，对齐 hpt-base-lang 尺度）

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `embed_dim` | **256** | 与 `liruiw/hpt-base-lang` trunk 一致 |
| `num_blocks` | 16 | |
| `num_heads` | 8 | |
| `num_action_tokens` | **64** | learnable action query |
| `num_future_tokens` | **16** | learnable future query |

序列布局（4 帧历史在 stem 内压缩，**trunk 仍 152 tokens**）：

```text
[ ego(16) | robot(16) | wrist(16) | state(16) | lang(8) | action_q(64) | future_q(16) ]
  ←—————————————— 72 obs ——————————————→   ←—— 80 query ——→
  共 152 tokens → + pos_embed → 16-layer trunk self-attn
```

`action_tokens` / `future_tokens` 是 **learnable query**，不是观测本身。Action-DiT 以 trunk 输出的 action query 为 cross-attn KV。

### Heads

| Head | 监督 | 说明 |
|------|------|------|
| **Action Head** | Flow matching | 默认 **`action_head_type=dit`**：Action-DiT；可选 **`cross_transformer`**（EgoWAM 风格）；**`diffusion`**（官方 mean-pool + DDIM）；**`transformer_decoder`**（官方 path B，全 trunk context + cross-attn）；**`mlp`** 为 legacy FM |
| **World Head** | Future DINO | action chunk 末端帧 DINOv2 CLS，MSE 回归 |

Action-DiT 默认：6 blocks / 128 hidden / 4 heads；`action_horizon=50`，`num_inference_steps=50`。

可选 `action_head_type=mlp` / `cross_transformer` / `diffusion` / `transformer_decoder`（与 DiT 做 ablation 时只改 head；后两者为官方 HPT 范式，obs-only trunk condition，输出仍为 H=50×80D）。

| `action_head_type` | Condition | 训练 | 推理 |
|--|--|--|--|
| `dit` / `cross_transformer` / `mlp` | 64 action query tokens | Flow Matching | Euler 50 步 |
| `diffusion` | trunk obs **mean pool** → 全局向量 | DDPM epsilon loss | DDIM（默认 50 步） |
| `transformer_decoder` | trunk obs **全序列** `[L,D]` | Huber 直接回归 | 一步 decode |

### EMA

生产 config 默认 **关闭**（`ema_decay=None`）。`train_hpt.py` 仍支持开启：设 `ema_decay` / `eval_on_ema` 后会存 EMA 为 `model.safetensors`。

### 域路由（`is_ego`）

**前向：**

- ego：走 Ego Stem；Robot/Wrist Stem 输出被置零（并可截断梯度）
- robot：走 Robot + Wrist Stem；Ego Stem 置零

**损失（四路加权，可在 config 改）：**

```python
loss = λ_ego_action·L_ego_a + λ_robot_action·L_robot_a
     + λ_ego_world·L_ego_w  + λ_robot_world·L_robot_w
     + λ_action_smooth·(L_ego_smooth + L_robot_smooth)
```

默认：`ego_world=1, ego_action=0.5, robot_world=0.5, robot_action=1, action_smooth=0.1`。

平滑项在 **归一化 action 空间**：由 FM 反解 \(\hat a=\varepsilon-v\)，再匹配相邻步差分 \(\Delta\hat a\approx\Delta a\)。

### 训练模式

| `train_mode` | 可训练 |
|--------------|--------|
| `pretrain` | stem + trunk + heads + query tokens（**DINOv2/T5 始终冻结**） |
| `finetune` | **冻结 trunk + query tokens**，只训 stem + heads |

---

## 3. 观测历史（observation_horizon=4）与 trunk 输入

对齐官方 HPT：历史帧 **不会** 在 trunk 里变成 4× token。4 帧在 **stem 内** 被 cross-attn 压成每 modality 固定 token 数。

```text
RLDS 图像时间维 T=5:  [t-3, t-2, t-1, t=0, t=50]
                       ←—— obs history 4 ——→  future（World Head）

每个 modality:
  1. 编码 T=4 帧（图像: 冻结 DINOv2；state: 80D）
  2. sinusoidal time embedding（按 T × spatial 展平）
  3. Stem cross-attn：learnable query → 固定 16 tokens（language 仍 8，无历史）
  4. 拼接 72 obs tokens + 64 action_q + 16 future_q = 152 → trunk
```

Trunk 序列长度 **仍为 152**（不是 72×4）。训练时 `random_horizon_masking=True`：随机只用最近 1~4 帧（官方同款）。

episode 开头不足 4 帧时，RLDS `clip` 到第 0 帧（重复首帧）。

---

## 4. 从 hpt-base-lang  warm-start（只加载 trunk.pth）

整包 `model.pth` **不能**直接训（stem ResNet vs DINO、head MLP vs Action-DiT、无 64+16 query）。  
官方 pretrain 也是 **只共享 trunk**，domain stem/head 后接。

Atom-0 支持：

```bash
# 本地目录（含 trunk.pth）或 trunk.pth 文件路径
export PRETRAINED_TRUNK_PATH=/path/to/hpt-base-lang

# 或 CLI
uv run scripts/train_hpt.py hpt_cotrain_real_only \
  --model.pretrained-trunk-path=/path/to/hpt-base-lang \
  --exp-name=hpt-trunk-warmstart
```

`load_pretrained_trunk()` 会做 key remap（`norm_1`→`norm1`、`mlp.fc1`→`mlp.0` 等），跳过官方 `bias_k/bias_v`。  
要求 **`embed_dim=256, num_blocks=16, num_heads=8`** 与 checkpoint 一致（当前 `_UNIFIED_HPT_PRETRAIN` 已对齐）。

Warm-start 后仍 **随机初始化**：5 个 stem、Action-DiT、action/future query、`pos_embed`、world head。  
`train_mode=pretrain` 下 trunk **继续可训**；DINO/T5 **始终冻结**。

日志应出现 `mapped=... loaded=...`；若 `loaded=0` 检查路径与 embed_dim。

---

## 5. 数据流（对齐 pi05 cotrain）

```text
RLDS builders (/mnt/bos/bo23lu)
  → restructure（STD_RESTRUCTURE_FNS）
  → scatter 到 80D + action_mask（piper native 14 维有效）
  → action chunk [H=50, 80]
  → 图像窗口 T=5：offsets [-3,-2,-1,0,50]  + state_history [4, 80]
  → decode / resize 224
  → StandardizedInputs → delta → DispatchNormalize（写 is_ego）
  → ModelTransformFactory(HPT): prompt + ResizeImages + PadStatesAndActions
  → Observation + Actions
  → HPTPytorch.compute_loss
```

与 pi05 的差异：

- pi05：SigLIP + PaliGemma + 离散 state-in-prompt；**无 world head**
- HPT：DINO + T5 stems + 共享 trunk + **Action-DiT flow + world DINO**；用 `is_ego` 路由 stem

---

## 6. 文件目录

```text
Atom-0/
├── docs/HPT.md                          # 本文档
├── scripts/
│   ├── train_hpt.py                     # 训练入口（DDP / wandb / ckpt）
│   ├── train_hpt_baige.sh               # 百舸节点内入口
│   └── preflight_cotrain_baige.py       # 含 HPT config 验收
└── src/openpi/
    ├── models/
    │   ├── model.py                     # ModelType.HPT
    │   └── hpt_config.py                # HPTConfig
    ├── models_pytorch/
    │   ├── hpt_pytorch.py               # Observation ↔ HPT 适配
    │   └── hpt/
    │       ├── encoders.py              # Frozen DINOv2 / T5
    │       ├── modules.py               # Stem / Trunk / Heads
    │       └── model.py                 # HPTModel + trunk.pth remap
    ├── cotrain/
    │   ├── config.py                    # hpt_cotrain_* 注册
    │   ├── data_loader.py               # HPT video_num_frames=2
    │   └── hpt_eval.py                  # 简单 val loss
    └── training/
        ├── config.py                    # ModelTransformFactory(HPT)
        └── data_loader.py               # LeRobot 路径的 future 窗口（如用）

baige-cluster/
└── atom0_hpt_train_job.py               # 提交百舸 PyTorchJob
```

---

## 7. 已注册 Config（生产用两个）

| name | assets | 模式 | builder 分片 | val | 数据 |
|------|--------|------|--------------|-----|------|
| `hpt_cotrain_real_only` | `cotrain_real_only` | **pretrain** | 否 | **开**（每 1k step） | piper30 + piper2 |
| `hpt_cotrain_real_robot_ego_fix` | `cotrain_real_robot_ego_fix` | **pretrain** | 否 | 关 | real+robot+ego（43） |

默认 **stem/trunk/heads 全可训**（DINO/T5 冻结）。  
`hpt_cotrain_real_only` 学习率：**cosine 100k**，warmup **5k**（5%）→ peak **`1e-4`** → **`1e-5`**，weight decay **`1e-4`**。  
`hpt_cotrain_real_robot_ego_fix` 同一套，缩放到 **300k**（warmup **15k**，save 每 30k）。`train_hpt.py` 会按 step 更新 lr。  
可选 **`PRETRAINED_TRUNK_PATH`** 从 `liruiw/hpt-base-lang` warm-start trunk。

已有 **Atom-0 全量 checkpoint** 时，可对 real-only 开真微调：`TRAIN_MODE=finetune` + `PYTORCH_WEIGHT_PATH=.../model.safetensors`（冻 trunk + query）。

另有 `hpt_cotrain_smoke` 本地冒烟。norm stats 复用对应 `assets/<assets_name>/`。

**注意**：旧 ckpt（512d / horizon 32 / MLP head）与当前 256d / horizon 50 / Action-DiT **不兼容**。

---

## 8. 百度云（百舸）提交

| 文件 | 作用 |
|------|------|
| `baige-cluster/atom0_hpt_train_job.py` | 提交 PyTorchJob |
| `Atom-0/scripts/train_hpt_baige.sh` | 节点内 torchrun → `train_hpt.py` |

### Preflight

```bash
cd /data/zjyang/Atom-0
source scripts/atom0_env.sh
.venv/bin/python scripts/preflight_cotrain_baige.py hpt_cotrain_real_only
.venv/bin/python scripts/preflight_cotrain_baige.py hpt_cotrain_real_robot_ego_fix
```

### Smoke（1×8，关 wandb）

```bash
cd /data/zjyang/baige-cluster
export BOS_SOURCE=atom0-data/
export BOS_MOUNT_PATH=/mnt/bos/bo23lu

MODE=smoke CONFIG_NAME=hpt_cotrain_real_only EXP_NAME=hpt-real-smoke \
INSTANCES=1 GPU_PER_NODE=8 BATCH_SIZE=16 SMOKE_STEPS=20 \
pixi run python atom0_hpt_train_job.py
```

### 正式 1：real-only + trunk warm-start（推荐）

```bash
cd /data/zjyang/baige-cluster
export BOS_SOURCE=atom0-data/
export BOS_MOUNT_PATH=/mnt/bos/bo23lu
export WANDB_API_KEY=...
export PRETRAINED_TRUNK_PATH=/path/to/hpt-base-lang   # 含 trunk.pth 的目录

CONFIG_NAME=hpt_cotrain_real_only EXP_NAME=hpt-real-trunk-v1 \
MODE=train INSTANCES=1 GPU_PER_NODE=8 BATCH_SIZE=128 \
EVAL_INTERVAL=1000 VAL_BATCH_SIZE=32 NUM_VAL_BATCHES=10 \
WANDB_ENABLED=1 OVERWRITE=1 \
pixi run python atom0_hpt_train_job.py
```

### 正式 2：real+robot+ego pretrain

```bash
CONFIG_NAME=hpt_cotrain_real_robot_ego_fix EXP_NAME=hpt-mix-pretrain-v1 \
MODE=train INSTANCES=1 GPU_PER_NODE=8 BATCH_SIZE=128 \
WANDB_ENABLED=1 OVERWRITE=1 \
pixi run python atom0_hpt_train_job.py
```

已有 Atom-0 checkpoint 后再微调（冻 trunk）：

```bash
CONFIG_NAME=hpt_cotrain_real_only EXP_NAME=hpt-real-ft-v1 \
TRAIN_MODE=finetune \
PYTORCH_WEIGHT_PATH=/data/zjyang/Atom-0/checkpoints/.../model.safetensors \
... pixi run python atom0_hpt_train_job.py
```

---

## 9. 前向伪代码（与实现一致）

```python
# observation_horizon=4: stem 内压缩，trunk 仍 152 tokens
base_feat = dino(base_img)                      # [B, T=4, N, 768]
base_ctx  = flatten(base_feat) + sinusoid       # [B, T*N, 768]
ego_tokens   = ego_stem(base_ctx)               # [B, 16, 256], gated by is_ego
robot_tokens = robot_stem(base_ctx)             # [B, 16, 256], gated by ~is_ego
wrist_tokens = wrist_stem(dino(left)+dino(right) + sinusoid)
state_tokens = state_stem(state_history + sinusoid)   # [B, T=4, 80]
lang_tokens  = language_stem(t5(prompt))              # 无历史

obs_tokens = cat([ego, robot, wrist, state, lang], dim=1)   # [B, 72, 256]
tokens = cat([obs_tokens, action_queries(64), future_queries(16)], dim=1)  # [B, 152, 256]
tokens = trunk(tokens + pos_embed)

action_features = tokens[:, 72:136]
future_features = tokens[:, 136:152]
v_t = action_dit(x_t, t, action_features)
world = world_head(future_features)           # DINO CLS @ t=50
```

---

## 10. 与官方 HPT-base / pi05 / FastWAM 对照

| | HPT-base-lang 官方 | Atom-0 HPT（本实现） | pi05 cotrain | FastWAM |
|--|--|--|--|--|
| Trunk | 256 / 16 / 8 | **同** | PaliGemma | Wan MoT |
| 图像 stem | ResNet 512d 预提特征 | **DINOv2** 768d | SigLIP | Wan VAE |
| 语言 | T5 | **T5 stem** | PaliGemma | UMT5 |
| 观测历史 | **T=4，stem 内压缩** | **同（T=4 + random mask）** | 可配 MEM | 可配 |
| Action | MLP 256→128 | **Action-DiT FM**, H=50 | Flow matching | Action DiT FM |
| Trunk query | 无（mean pool） | **64 action + 16 future** | — | — |
| World | 无 | **Future DINO** | 无 | Video DiT |
| 数据 | MetaWorld 等 | **cotrain 80D RLDS** | 同左 | 同左 |
| `is_ego` | 无 | stem 路由 + 四路 loss | 仅标签 | 四路 loss |

---

## 11. 实现备注

1. **原始观测维度不必固定**：stem/head 按 80D 与相机槽位适配；进 trunk 的是固定 `embed_dim=256`。
2. **DINOv2 / T5 训练时冻结**（`freeze_encoders=True`），不计入 optimizer；首次运行从 HuggingFace 拉权重。
3. **缺失腕部相机**（ego）：空白图 + `image_mask=False`，wrist stem 输出被 mask 掉。
4. **Trunk warm-start**：`pretrained_trunk_path` 或 `PRETRAINED_TRUNK_PATH`；仅 `trunk.pth`，自动 key remap。
5. **旧 checkpoint 不兼容**：512d / 32 horizon / MLP 版权重勿用于当前 config。
6. **观测历史**：`observation_horizon=4`；RLDS 采 `[-3,-2,-1,0,50]`；stem 压缩后 trunk 仍 152 tokens。DINO 按 T 帧前向（约 4× 图像编码算力）。
