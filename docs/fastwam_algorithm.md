# FastWAM 算法说明（Atom-0 × Cotrain RLDS）

本文说明在 **Atom-0** 中新增的 **FastWAM** 算法：将 FastWAM 的 **World Model（Video DiT）与 Action Model（Action DiT）并联 MoT** 接到你们已有的 **`src/openpi/cotrain/` RLDS 数据管线**上。

> 数据侧遵循 cotrain 设计原则：多数据集加权混采、运行时 restructure、native 动作 + pad、per-dataset delta/归一化。  
> 模型侧为 PyTorch MoT；训练入口 `scripts/train_fastwam.py`（不是 JAX `train_cotrain.py`）。

更完整的 cotrain 背景见 [`cotrain_技术文档.md`](./cotrain_技术文档.md)。

---

## 1. 算法概览

FastWAM（Fast World-Action Model）是 **Mixture-of-Transformers (MoT)**：

| 分支 | 角色 | 骨干 |
|------|------|------|
| **Video DiT** | World Model：latent 空间预测未来帧 | Wan2.2 TI2V-5B |
| **Action DiT** | Action Model：预测 action chunk | 同层数/头数的 Action Expert |
| **MoT** | 层内混合自注意力 | `models_pytorch/fastwam/wan22/mot.py` |

语言经 **UMT5** → `context`，以 cross-attn 注入两路 expert。训练对未来视频 latent 与 action 做 dual continuous flow matching；推理可用 **action-only + 首帧 video KV cache**。

基本（uncond）注意力：video `first_frame_causal`；action→video 仅首帧；video↛action。

---

## 2. 与 Cotrain RLDS 的对接（主路径）

```
RLDS (builder_dir / uid)
  → CotrainRldsDataset
       restructure 注册表 → 统一 schema
       pad state/action → model.action_dim
       chunk actions [H]
       ★ FastWAM: chunk 未来视频窗 [T_v]（编码 JPEG，shuffle 后再 decode）
       sample_from_datasets → shuffle → decode+resize224 → batch
  → StandardizedInputs + DispatchDelta + DispatchNormalize
  → FastWAM model transforms（InjectDefaultPrompt + ResizeImages，不用 PaliGemma）
  → Observation (+ prompt 旁路) / Actions
  → FastWAMPytorch.compute_loss / sample_actions
```

### 2.1 视频窗（相对 pi0/pi05 的增量）

在 `CotrainRldsDataset` 中，当 `video_num_frames` 非空时：

- 约束：`action_horizon == (video_num_frames - 1) * action_video_freq_ratio`  
  默认：`32 == (9 - 1) * 4`
- 在 traj 级对每个起点 `t` gather 编码帧：`t, t+r, …, t+(T_v-1)r`
- **仍在 shuffle 之后 decode**（与 cotrain 文档一致，避免 buffer 存 raw 像素 OOM）

### 2.2 Schema / 动作空间

沿用 cotrain，**不另做动作对齐**：

- 统一嵌套键：`image{base,left_wrist,right_wrist}` / `state` / `actions` / `prompt` / `dataset_id`
- native 维 pad 到 `FastWAMConfig.action_dim`（如 piper30 → **14**）
- `DispatchDeltaActions` / `DispatchNormalize` 按 `dataset_id`（uid）分派

### 2.3 代码位置

| 路径 | 作用 |
|------|------|
| `src/openpi/cotrain/rlds_dataset.py` | 视频窗 chunk + 多帧 decode |
| `src/openpi/cotrain/data_loader.py` | `framework="pytorch"`、从 model 读 `video_num_frames` |
| `src/openpi/cotrain/config.py` | `fastwam_cotrain_piper30` 等注册 |
| `src/openpi/models/fastwam_config.py` | FastWAMConfig |
| `src/openpi/models_pytorch/fastwam*` | MoT 实现 + Observation 适配器 |
| `scripts/train_fastwam.py` | PyTorch 训练（读 **cotrain** config registry） |

---

## 3. Observation → FastWAM sample

| Cotrain / Observation | FastWAM |
|----------------------|---------|
| `image[*]` `[B,T,H,W,C]`，`[-1,1]` | 水平拼接相机 → `video` `[B,3,T,H,2W]` |
| `state` `[B,D]`（已 pad） | `proprio`（前 `proprio_dim`） |
| `actions` `[B,H,D]` | `action` |
| `prompt` 或 `context`/`context_mask` | UMT5 / 预计算 embedding |

约束：`T % 4 == 1`；`H % (T-1) == 0`。

---

## 4. 训练

```bash
# 依赖 RLDS group（dlimp / tensorflow）
uv sync --group rlds
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints/fastwam"

# 逐数据集 norm（与 cotrain 相同工具链）
uv run --group rlds scripts/compute_cotrain_norm_stats.py --config-name fastwam_cotrain_piper30

# 训练（配置在 cotrain registry）
uv run --group rlds scripts/train_fastwam.py fastwam_cotrain_piper30 \
    --exp_name=fw_piper30 --overwrite
```

| Config | 说明 |
|--------|------|
| `fastwam_cotrain_piper30` | 真机 piper30 RLDS + 完整 FastWAM |
| `fastwam_cotrain_piper30_debug` | 同数据；跳过 DiT 预训练 / 无 T5（联调） |

- 冻结：VAE、Text Encoder  
- 训练：MoT + 可选 proprio_encoder  
- 损失（四路，按 `is_ego` 拆分）：  
  `λ_ego_v L_video_ego + λ_ego_a L_action_ego + λ_robot_v L_video_robot + λ_robot_a L_action_robot`  
  （`dataset_id` 以 `egoverse` 开头 → ego；权重见 `FastWAMConfig.loss`）

> 不要用 `scripts/train_cotrain.py` 训 FastWAM（那是 JAX π0.5）。  
> 不要用 openpi 主 registry 里的 `fastwam_libero` 作为主路径（LeRobot 遗留可选）。

---

## 5. 推理

`FastWAMPytorch.sample_actions(device, observation)` → `infer_action`（首帧 VAE + KV cache）。

Checkpoint 为 `model.safetensors`；可用 `serve_policy`（需对应 cotrain/openpi policy 配置，按部署再挂）。

---

## 6. 与 π0.5 cotrain / 纯 LeRobot 的差异

| | π0.5 cotrain | FastWAM cotrain | FastWAM LeRobot（遗留） |
|--|--------------|-----------------|-------------------------|
| 数据 | cotrain RLDS | **同一套 RLDS** | LeRobot repo |
| 观测 | 单帧 | **未来视频窗** | LeRobot delta_timestamps |
| 训练 | JAX `train_cotrain.py` | PyTorch `train_fastwam.py` | （旧）`framework=pytorch` |
| 语言 | PaliGemma | UMT5 | UMT5 |
| 世界模型 | 无 | Video DiT | Video DiT |

---

## 7. 扩展新 RLDS 数据集

与 cotrain 相同：

1. 写 / 复用 `STD_RESTRUCTURE_FNS` 中的 restructure  
2. 配一条 `CotrainRLDSDataset`（`builder_dir` + `uid` + delta mask + `action_dim`）  
3. 新建 `CotrainTrainConfig(name="fastwam_cotrain_...", model=FastWAMConfig(...), data=...)`  
4. 保证 `action_horizon` 与 `video_num_frames` / `ratio` 对齐  

多数据集混采时，`action_dim` 取混合物中最大 pad 宽度（与 π0.5 cotrain 一样）。

---

## 8. 参考

- Cotrain 技术文档：[`cotrain_技术文档.md`](./cotrain_技术文档.md)  
- 环境配置：[`fastwam_environment.md`](./fastwam_environment.md)  
- 上游 FastWAM：`../FastWAM/`
