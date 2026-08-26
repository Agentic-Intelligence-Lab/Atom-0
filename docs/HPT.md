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
Human/Robot Head Image ──► Ego Stem (32) ──────┐
                                               │
Robot Wrist(s) ──► Wrist Stem (16, human mask)─┤──► Shared HPT Trunk (72)
                                               │         │
Proprio ──► State Stem (16) ───────────────────┤    ┌────┴────┐
                                               │    ▼         ▼
Prompt ──► T5 (frozen) ──► Language Stem (8) ──┘  tdec     World DiT
                                                      │        │
                                                 action chunk  future DINO patches (train only)
```

### 编码器（训练时冻结）

| 模块 | 默认 | 说明 |
|------|------|------|
| **DINOv2** | `facebook/dinov2-base`，768d | `freeze_encoders=True`；`train_hpt.py` 调用 `freeze_encoders()` |
| **T5-base** | 768d | 同上；stem 可训，encoder 不更新 |

### Stem（可训练）

| Stem | 输入 | 编码 |
|------|------|------|
| Ego | human/robot **当前** head/base 图 | **冻结 DINOv2** → MLP + CrossAttn → **32** tokens |
| Wrist | robot 双腕（human mask） | 同上 → **16** tokens |
| State | 80D proprio（当前帧） | MLP + CrossAttn → **16** tokens |
| Language | 文本 prompt | **冻结 T5-base** → MLP + CrossAttn → **8** tokens |

默认 **72** obs tokens：`32+16+16+8`。无 `robot_stem`，无 action/future query token。

### Trunk（共享，对齐 hpt-base-lang 尺度）

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `embed_dim` | **256** | 与 `liruiw/hpt-base-lang` trunk 一致 |
| `num_blocks` | 16 | |
| `num_heads` | 8 | |

序列布局（**仅当前帧**）：

```text
[ ego(32) | wrist(16) | state(16) | lang(8) ]  = 72
  → + pos_embed → 16-layer trunk self-attn
```

tdec 以这 72 个 trunk obs token 为 cross-attn context。World head 对 72 token **mean-pool** 成全局向量，不把 query 拼进 trunk。

### Heads

| Head | 监督 | 说明 |
|------|------|------|
| **Action Head** | 默认 **tdec** Huber 回归 | `transformer_decoder`：50 learnable queries × 72 obs context；可选 `dit` / `cross_transformer` / `mlp` / `diffusion` |
| **World Head** | Future DINO **patch** FM | 冻结 DINOv2-B **丢掉 CLS** 的 256×768；DiT 6×384/6h + wide Transformer 2×2048；**仅训练** |

| `action_head_type` | Condition | 训练 | 推理 |
|--|--|--|--|
| `transformer_decoder`（默认） | trunk obs **全序列** `[72,256]` | Huber | 一步 decode |
| `dit` / `cross_transformer` / `mlp` | 同上 72 obs tokens | Flow Matching | Euler 50 步 |
| `diffusion` | trunk obs **mean pool** | DDPM epsilon | DDIM |

### EMA

生产 config 默认 **关闭**（`ema_decay=None`）。`train_hpt.py` 仍支持开启：设 `ema_decay` / `eval_on_ema` 后会存 EMA 为 `model.safetensors`。

### 域路由（`is_ego`）

**前向：**

- human 与 robot 的 head/base 图都走 **Ego Stem**
- human：**Wrist Stem 置零**（并可截断梯度）
- robot：Ego + Wrist 都有效

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
| `pretrain` | stem + trunk + heads（**DINOv2/T5 始终冻结**） |
| `finetune` | **冻结 trunk + pos_embed + world_head**，只训 stem + action head（跳过 world loss） |

---

## 3. 当前帧观测（observation_horizon=1）与 trunk 输入

图像 / 状态 / 语言均为 **t=0**。`action_world` 时 RLDS 再拼一帧未来图（t=50）仅作 world GT。

```text
RLDS 图像时间维 T=2:  [t=0, t=50]
                       obs     future（World Head 目标）

obs_tokens = cat([ego(32), wrist(16), proprio(16), lang(8)])  # [B, 72, 256]
H_obs = trunk(obs_tokens)
actions = tdec(H_obs)
# train only:
z = dino_patches(future_ego)[:, 1:]   # drop CLS → [B, 256, 768]
û = world_dit(x_τ, τ, mean(H_obs))    # AdaLN, no cross-attn
```

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

Warm-start 后仍 **随机初始化**：stems、tdec / world DiT、`pos_embed`。  
`train_mode=pretrain` 下 trunk **继续可训**；DINO/T5 **始终冻结**。

日志应出现 `mapped=... loaded=...`；若 `loaded=0` 检查路径与 embed_dim。

---

## 5. 数据流（对齐 pi05 cotrain）

```text
RLDS builders (/mnt/bos/bo23lu)
  → restructure（STD_RESTRUCTURE_FNS）
  → scatter 到 80D + action_mask（piper native 14 维有效）
  → action chunk [H=50, 80]
  → 图像窗口 T=2：offsets [0, 50]  + 当前 state
  → decode / resize 224
  → StandardizedInputs → delta → DispatchNormalize（写 is_ego）
  → ModelTransformFactory(HPT): prompt + ResizeImages + PadStatesAndActions
  → Observation + Actions
  → HPTPytorch.compute_loss
```

与 pi05 的差异：

- pi05：SigLIP + PaliGemma + 离散 state-in-prompt；**无 world head**
- HPT：DINO + T5 stems + 共享 trunk + **tdec action + DINO World DiT（训练）**；用 `is_ego` 路由 wrist

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
| `hpt_cotrain_real_robot_ego_fix` | `cotrain_real_robot_ego_fix` | **pretrain** | 是 | 开（piper2/30） | real+robot+ego（47） |
| `hpt_cotrain_piper_ft` | `cotrain_real_only` | **finetune**（冻 trunk+world） | 否 | **开** | piper30 + piper2 |
| `hpt_cotrain_piper_ft_full` | `cotrain_real_only` | **pretrain**（全参） | 否 | **开** | piper30 + piper2 |
| `hpt_cotrain_hhz_robot_ft` | `cotrain_real_robot_ego_fix` | **finetune**（冻 trunk+world） | 否 | **开** | front_cam `aligned_hangzhou_robot_right` |

冻结微调 config 共用 **peak_lr=1.5e-4**（warmup 1k → cosine → 1e-6，20k steps）。

默认 **stem/trunk/heads 全可训**（DINO/T5 冻结）。  
`hpt_cotrain_real_only` 学习率：**cosine 100k**，warmup **5k**（5%）→ peak **`1e-4`** → **`1e-5`**，weight decay **`1e-4`**。  
`hpt_cotrain_real_robot_ego_fix` 同一套，缩放到 **300k**（warmup **15k**，save 每 30k）。`train_hpt.py` 会按 step 更新 lr。  
可选 **`PRETRAINED_TRUNK_PATH`** 从 `liruiw/hpt-base-lang` warm-start trunk。

已有 **Atom-0 全量 checkpoint** 时，用 **`hpt_cotrain_piper_ft`** 或任意 config 设 `TRAIN_MODE=finetune` + `PYTORCH_WEIGHT_PATH=.../model.safetensors`（冻 trunk + world_head + pos_embed，只训 stem + action head）。

另有 `hpt_cotrain_smoke` 本地冒烟。norm stats 复用对应 `assets/<assets_name>/`。

**注意**：旧 ckpt（含 4 帧历史 / robot_stem / action·future query / CLS world MLP）与当前结构 **不兼容**。

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

已有 Atom-0 checkpoint 后再微调 piper（冻 trunk + world head）：

```bash
CONFIG_NAME=hpt_cotrain_piper_ft EXP_NAME=hpt-piper-ft-v1 \
PYTORCH_WEIGHT_PATH=/data/zjyang/Atom-0/checkpoints/hpt_cotrain_real_robot_ego_fix/hpt-3*8/90000 \
MODE=train INSTANCES=1 GPU_PER_NODE=8 BATCH_SIZE=128 \
EVAL_INTERVAL=1000 WANDB_ENABLED=1 OVERWRITE=1 \
pixi run python atom0_hpt_train_job.py
```

或在 real-only 上手动开 finetune：

```bash
CONFIG_NAME=hpt_cotrain_real_only EXP_NAME=hpt-real-ft-v1 \
TRAIN_MODE=finetune \
PYTORCH_WEIGHT_PATH=/data/zjyang/Atom-0/checkpoints/.../model.safetensors \
... pixi run python atom0_hpt_train_job.py
```

---

## 9. 前向伪代码（与实现一致）

```python
# observation_horizon=1: current frame only; trunk = 72 obs tokens
base_feat = dino(base_img)                      # current head/ego camera
ego_tokens   = ego_stem(base_feat)              # [B, 32, 256], human+robot
wrist_tokens = wrist_stem(dino(left)+dino(right))  # [B, 16, 256], masked if human
state_tokens = state_stem(state)                # [B, 16, 256]
lang_tokens  = language_stem(t5(prompt))        # [B, 8, 256]

obs_tokens = cat([ego, wrist, state, lang], dim=1)   # [B, 72, 256]
H_obs = trunk(obs_tokens + pos_embed)
actions = tdec(H_obs)                           # train + infer

# train only — future ego/head DINO patches, drop CLS
z = dino(future_base)[:, 1:]                    # [B, 256, 768]
x_tau = (1 - tau) * z + tau * eps
u = eps - z
v_hat = world_dit(x_tau, tau, mean(H_obs))      # AdaLN; no trunk cross-attn
L_world = MSE(v_hat, u)
```

---

## 10. 与官方 HPT-base / pi05 / FastWAM 对照

| | HPT-base-lang 官方 | Atom-0 HPT（本实现） | pi05 cotrain | FastWAM |
|--|--|--|--|--|
| Trunk | 256 / 16 / 8 | **同** | PaliGemma | Wan MoT |
| 图像 stem | ResNet 512d 预提特征 | **DINOv2** 768d | SigLIP | Wan VAE |
| 语言 | T5 | **T5 stem** | PaliGemma | UMT5 |
| 观测历史 | **T=4，stem 内压缩** | **当前帧 T=1** | 可配 MEM | 可配 |
| Action | MLP 256→128 | **tdec Huber**, H=50 | Flow matching | Action DiT FM |
| Trunk query | 无（mean pool） | **无**（72 obs only） | — | — |
| World | 无 | **Future DINO patch FM** | 无 | Video DiT |
| 数据 | MetaWorld 等 | **cotrain 80D RLDS** | 同左 | 同左 |
| `is_ego` | 无 | stem 路由 + 四路 loss | 仅标签 | 四路 loss |

---

## 11. 实现备注

1. **原始观测维度不必固定**：stem/head 按 80D 与相机槽位适配；进 trunk 的是固定 `embed_dim=256`。
2. **DINOv2 / T5 训练时冻结**（`freeze_encoders=True`），不计入 optimizer；首次运行从 HuggingFace 拉权重。
3. **缺失腕部相机**（ego）：空白图 + `image_mask=False`，wrist stem 输出被 mask 掉。
4. **Trunk warm-start**：`pretrained_trunk_path` 或 `PRETRAINED_TRUNK_PATH`；仅 `trunk.pth`，自动 key remap。
5. **旧 checkpoint 不兼容**：512d / 32 horizon / MLP 版权重勿用于当前 config。
6. **观测**：`observation_horizon=1`；`action_world` 时 RLDS 采 `[0, 50]`。World 仅训练，推理只跑 trunk + tdec。
