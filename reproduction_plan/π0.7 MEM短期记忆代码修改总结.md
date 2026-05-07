# π0.7 MEM 短期记忆代码修改总结

## 目标

本次修改实现 MEM 第一阶段：短期视觉/状态记忆。设计目标是尽可能贴近 Physical Intelligence MEM 论文原文，而不是采用简单的 late-fusion 多帧拼接。

核心原则：

- 图像历史在 SigLIP/ViT 视觉编码器内部融合。
- 视觉 temporal attention 使用 space-time separable attention。
- 只把当前 timestep 的压缩视觉 tokens 传给后续 VLA backbone，避免 Gemma prefix token 数变成历史长度倍数。
- temporal position encoding 使用固定 sin/cos，并保证当前帧 `t=0` 的 temporal embedding 精确为 0。
- state memory 使用连续投影：每个历史 state 投影为一个 prefix token，不使用 π0.5 的分箱文本 tokenizer。

## 主要代码改动

### 1. MEM 配置

文件：`src/openpi/models/pi0_config.py`

新增字段：

- `history_length`
- `history_stride_seconds`
- `temporal_attention_every_n_layers`
- `mem_include_state_history`
- `long_memory_enabled`

默认 `history_length=1`，保持原始单帧行为。`long_memory_enabled=True` 时自动把 `max_token_len` 提升到至少 320，用于长期语言记忆接口。

### 2. 多帧 Observation 与预处理

文件：`src/openpi/models/model.py`

支持图像输入：

- 单帧：`[B, H, W, C]`
- 多帧：`[B, T, H, W, C]`

新增 `state_history` 字段，形状为 `[B, T, state_dim]`。

多帧图像预处理会先 flatten 成 `[B*T, H, W, C]` 做 resize/augmentation，再 reshape 回 `[B, T, H, W, C]`。训练增强时，同一个样本的 T 帧共享相同随机增强，避免破坏 temporal attention 的 patch 对应关系。

### 3. Faithful MEM Video Encoder

文件：`src/openpi/models/siglip.py`

新增：

- `posemb_sincos_1d_zero_current`
- `VideoEncoder1DBlock`

实现方式：

- 每帧先做原始 ViT spatial attention。
- 每 4 层对同一 spatial patch 位置做 causal temporal attention。
- temporal attention 复用同一层原有 attention 参数，不新增 temporal attention 权重。
- 多帧 encoder 输出后只保留当前帧 tokens：`x[:, -1]`。

这对应 MEM 原文中“video encoder compresses observation history and only passes current timestep representations to the VLA backbone”的设计。

### 4. Pi0 接入 MEM

文件：`src/openpi/models/pi0.py`

修改点：

- `history_length=1` 时继续使用原 scan SigLIP encoder。
- `history_length>1` 时使用非 scan SigLIP encoder，以便按层号插入每 4 层 temporal attention。
- 新增 `state_memory_proj`，把 `state_history [B,T,S]` 投影为 `[B,T,D]` prefix tokens。
- `embed_prefix` 中图像 mask 如果是 `[B,T]`，只取当前帧 mask，因为视频 encoder 已经把历史压缩到当前帧 tokens。

prefix 顺序：

```text
video-compressed visual tokens
+ state history tokens
+ prompt / optional memory summary tokens
+ KI FAST tokens
```

### 5. 数据加载历史帧

文件：

- `src/openpi/training/config.py`
- `src/openpi/training/data_loader.py`

新增 `DataConfig.observation_history_keys` 和 `DataConfig.state_history_key`。

LeRobot 数据加载时：

- action 仍取未来 action chunk。
- MEM observation history 取负向 `delta_timestamps`。
- `history_length=6`、`history_stride_seconds=1.0` 时为 `[-5, -4, -3, -2, -1, 0]` 秒。

LIBERO 默认加载：

- `image`
- `wrist_image`
- `state`

DROID LeRobot 路径加载：

- `exterior_image_1_left`
- `wrist_image_left`

RLDS DROID 第一阶段暂不接历史帧。

### 6. Policy 推理 history buffer

文件：

- `src/openpi/transforms.py`
- `src/openpi/policies/policy_config.py`

新增 `HistoryBufferTransform`。

推理端仍然只需要外部传当前帧，transform 会自动维护 ring buffer：

- 第一次调用用当前帧 warm start repeat 成 T 帧。
- 后续调用 append 当前帧，保留最近 T 帧。
- 输出 `[T,H,W,C]` 图像历史与 `[T,S]` state history。

### 7. Policy 输入适配

文件：

- `src/openpi/policies/libero_policy.py`
- `src/openpi/policies/droid_policy.py`

`_parse_image` 现在支持：

- `[H,W,C]`
- `[C,H,W]`
- `[T,H,W,C]`
- `[T,C,H,W]`

如果 dataset 返回历史 state，policy transform 会把最后一帧作为当前 `state`，完整序列作为 `state_history`。

### 8. KI tokenization 与 state memory

文件：`src/openpi/transforms.py`

`KITokenize` 现在尊重 `discrete_state_input`：

- `discrete_state_input=True` 时，普通 PaliGemma prompt 继续包含 π0.5 风格分箱 state。
- `discrete_state_input=False` 时，普通 prompt 不包含分箱 state。
- KI FAST 辅助 tokenization 仍保留自己的 state 输入路径。

MEM config 使用 `discrete_state_input=False`，因此 state memory 走连续投影 token，而不是 π0.5 的分箱文本 state。

### 9. Checkpoint 加载

文件：`src/openpi/training/weight_loaders.py`

`CheckpointWeightLoader` 新增 `missing_regex`。

MEM 使用非 scan SigLIP encoder，但官方 checkpoint 里的 SigLIP transformer 是 scan 参数。加载器新增 scan 参数展开逻辑：

```text
Transformer/encoderblock/<param>[layer]
→ Transformer/encoderblock_<layer>/<param>
```

MEM config 允许新增 `state_memory_proj` 参数保留初始化值。

### 10. PyTorch 路径

文件：`src/openpi/models_pytorch/pi0_pytorch.py`

第一阶段 MEM 只实现 JAX 路径。PyTorch 模型在 `history_length>1` 时显式抛出 `NotImplementedError`，避免静默错误。

## 新增 Config

文件：`src/openpi/training/config.py`

- `debug_pi05_mem`
- `pi05_mem_libero`
- `pi05_mem_long_interface`

## 新增验证

新增文件：

- `src/openpi/models/pi0_mem_test.py`
- `tests/mem/stage1_short_memory_forward.py`
- `tests/mem/stage2_policy_history_buffer.py`
- `tests/mem/stage3_train_step_smoke.py`
- `tests/mem/stage4_long_memory_interface.py`
- `tests/mem/stage5_single_batch_overfit.py`
- `scripts/validate_mem_short_term_local.sh`

新增测试重点：

- `history_length=1` 等价性：用相同 seed 初始化原始 pi05 path 和显式 `history_length=1` path，确认 loss 数值一致。
- temporal gradient：用当前帧视觉 tokens 的 loss 反传到输入视频，确认历史帧也有非零梯度，证明当前表示确实依赖历史视觉 token。
- 单 batch overfit：固定一个 6 帧 toy batch 训练 80 步，确认 MEM 模型 loss 能下降，覆盖 optimizer update 和 backward 路径。

建议运行：

```bash
python -m pytest src/openpi/models/model_test.py src/openpi/models/pi0_mem_test.py src/openpi/transforms_test.py -q
python tests/mem/stage1_short_memory_forward.py
python tests/mem/stage2_policy_history_buffer.py
python tests/mem/stage3_train_step_smoke.py
python tests/mem/stage4_long_memory_interface.py
python tests/mem/stage5_single_batch_overfit.py
```

或者：

```bash
bash scripts/validate_mem_short_term_local.sh
```

## 本地验证状态

已通过：

```bash
python -m py_compile ...
git diff --check
```

未能在当前机器完整运行 pytest/stage scripts，原因是当前 Python 环境缺少项目依赖：

- `flax`
- `pynvml`

同时当前 shell 找不到 `uv` 命令。安装依赖或进入项目虚拟环境后，可运行上面的验证命令。
