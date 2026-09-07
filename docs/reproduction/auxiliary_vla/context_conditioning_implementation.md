# π0.7 Diverse Context Conditioning 代码修改总结

## 目标

本次修改实现 π0.7 第三步的 V2：Diverse Context Conditioning + per-component Dropout 的代码闭环。目标不是直接复现论文中的大规模训练结果，而是在 KI + MEM 已完成的基础上，先把多组件 context 的输入路径、独立文本 token segment、subgoal 图像条件、dropout 采样、debug 训练 smoke test 全部打通。

核心闭环：

```text
InfiData / LIBERO 风格字段
→ DCC transform 独立 tokenize metadata / control / subtask segments
→ per-component dropout 修改文本字段和 subgoal mask
→ subgoal future image 作为视觉 prefix token 接入 Pi0
→ 原 flow loss + KI FAST CE loss 不变
→ debug batch 完整 forward / loss 验证
```

## 设计取舍

本阶段采用“独立文本 token segment + subgoal 图像 token”的 V2 方案：

- `task` 保持走现有 `tokenized_prompt`，兼容 KI FAST tokenization。
- `metadata / control_mode / subtask` 分别 tokenize 成独立固定长度 segment。
- `subgoal_image` 不放进文本，作为独立视觉字段走 SigLIP 编码，并拼入 VLM prefix。
- dropout 在 transform 层执行，模型 forward 路径只看已经处理好的 prompt 和 subgoal mask。
- flow matching、KI FAST CE、MEM 长短期记忆 loss 都不改。
- DCC 关闭时，现有 KI/MEM 行为保持原样。

这和原方案的核心目标一致：让模型在训练中见到不同 context 组件组合，并在 subgoal 缺失时仍能 fallback。当前版本已经完成独立 token segment；尚未实现的是显式 learned sentinel / special token 边界。

## 主要代码改动

### 1. DCC 配置

文件：`src/openpi/models/pi0_config.py`

新增字段：

- `diverse_context_enabled`
- `use_subgoal_image`
- `subgoal_delta_seconds`
- `subgoal_keep_prob`
- `subtask_drop_when_subgoal`
- `metadata_drop_prob`
- `metadata_field_drop_prob`
- `control_mode_drop_prob`

行为变化：

- `diverse_context_enabled=True` 时，`max_token_len` 自动提升到至少 384，避免主任务 prompt 被截断。
- 对所有 dropout 概率做 `[0, 1]` 合法性检查。
- `use_subgoal_image=True` 时，`inputs_spec()` 额外生成 `subgoal_images / subgoal_image_masks`。
- `diverse_context_enabled=True` 时，`inputs_spec()` 额外生成 `dcc_metadata_tokens / dcc_control_tokens / dcc_subtask_tokens` 及对应 masks。

### 2. Observation 支持 subgoal 图像

文件：`src/openpi/models/model.py`

新增可选字段：

- `subgoal_images`
- `subgoal_image_masks`

同步更新：

- `Observation.from_dict()`
- `Observation.to_dict()`
- `preprocess_observation()`

现在 subgoal 图像可以像普通 observation image 一样接受 `uint8` 输入，并被转成 `[-1, 1]` float 图像；如果尺寸不是 `224x224`，也会走 resize-with-pad。

### 3. Pi0 prefix 接入 subgoal visual tokens

文件：`src/openpi/models/pi0.py`

新增：

- `subgoal_type_embedding`
- `_embed_subgoal_images()`

实现方式：

1. `subgoal_images` 复用现有 `self.PaliGemma.img` / SigLIP 编码器。
2. 编码后给 subgoal tokens 加 zero-init `subgoal_type_embedding`，用于区分“当前观察图像”和“未来目标图像”。
3. `embed_prefix()` 在 MEM state history 后、语言 prompt 前追加 subgoal tokens。
4. 如果 `use_subgoal_image=True` 但当前样本没有 subgoal 图像，则用零图像 + 全 false mask 占位，保持 prefix 长度稳定。

prefix 顺序变为：

```text
obs / MEM-compressed image tokens
+ state history tokens
+ subgoal image tokens
+ metadata tokens
+ control-mode tokens
+ task prompt tokens
+ subtask tokens
+ KI FAST tokens
```

### 4. DCC transform

文件：`src/openpi/transforms.py`

新增：

- `SplitSubgoalFromHistory`
- `ApplyDiverseContextDropout`
- `TokenizeDiverseContextSegments`

`SplitSubgoalFromHistory`：

- 支持 data loader 为同一个 image key 读取 `[history frames..., future frame]`。
- 前 `T` 帧保留为 MEM history。
- 最后一帧拆成 `subgoal_image`。
- 同步拆分 `image_mask` 和 `subgoal_image_mask`。

`ApplyDiverseContextDropout`：

- subgoal：默认保留率 25%，drop 时只把 mask 置 false，不删除数组。
- subtask：仅在 subgoal 保留时，按 30% 概率置为 `none`。
- metadata：整体 15% dropout；保留时每个字段 5% 独立 dropout。
- control mode：默认不 dropout。

`TokenizeDiverseContextSegments`：

将文本类 context 拆成三个独立 token segment：

```text
Subtask: <subtask or none>
Metadata: quality=<...>; speed=<...>; mistake=<...>; success=<...>
Control: <control_mode>
```

输出字段：

- `dcc_metadata_tokens / dcc_metadata_mask`
- `dcc_control_tokens / dcc_control_mask`
- `dcc_subtask_tokens / dcc_subtask_mask`

主任务 `prompt` 保持不变，继续交给现有 `TokenizePrompt` / `KITokenize`。

### 5. LIBERO policy / data loader 接入

文件：

- `src/openpi/policies/libero_policy.py`
- `src/openpi/training/data_loader.py`
- `src/openpi/training/config.py`

`LiberoInputs` 新增透传字段：

- `subtask`
- `quality`
- `speed_bin`
- `mistake`
- `success`
- `control_mode`
- `subgoal_image`
- `subgoal_wrist_image`

`DataConfig` 新增：

- `subgoal_image_keys`

`create_torch_dataset()` 行为变化：

- `use_subgoal_image=True` 时，会给对应 image key 增加一个 future delta timestamp。
- 默认 future delta 使用 `subgoal_delta_seconds=2.0`。
- 对 MEM history + subgoal 的组合，loader 会读取 `[-5, -4, -3, -2, -1, 0, +2]` 秒形式的图像序列，再由 transform 拆分。

### 6. 配置注册

文件：`src/openpi/training/config.py`

新增 configs：

- `debug_pi05_dcc`
- `pi05_dcc_libero`
- `pi05_dcc_libero_no_subgoal`
- `pi05_dcc_libero_no_metadata`
- `pi05_dcc_libero_no_dropout`

`pi05_dcc_libero` 使用 KI + MEM 配置作为基础，并打开：

- `diverse_context_enabled=True`
- `use_subgoal_image=True`
- `history_length=6`
- `ki_enabled=True`
- `discrete_state_input=False`

checkpoint loader 的 `missing_regex` 放行新增参数：

```text
state_memory_proj
subgoal_type_embedding
```

### 7. PyTorch 路径保护

文件：`src/openpi/models_pytorch/pi0_pytorch.py`

当开启：

- `diverse_context_enabled`
- 或 `use_subgoal_image`

PyTorch path 显式抛出 `NotImplementedError`，避免静默走错路径。本阶段 DCC 只支持 JAX Pi0。

## 新增验证

新增文件：

- `tests/dcc/stage1_context_transforms.py`
- `tests/dcc/stage2_subgoal_split.py`
- `tests/dcc/stage3_train_step_smoke.py`
- `scripts/validate_dcc_local.sh`

### Stage 1：Context transform / dropout

验证内容：

- InfiData 风格字段能分别 tokenized 成 metadata / control / subtask segment。
- 强制 metadata dropout 后，metadata 字段变成 `none`。
- 强制 subgoal dropout 后，`subgoal_image_mask=False`。
- 统计 5000 次 `subgoal_keep_prob=0.25`，实际保留率接近 25%。

覆盖：

- `ApplyDiverseContextDropout`
- `TokenizeDiverseContextSegments`

### Stage 2：Subgoal split

验证内容：

- 输入 `[7, H, W, C]` 图像序列。
- 前 6 帧保留为 MEM history。
- 第 7 帧拆成 `subgoal_image`。
- mask 同步拆分。

覆盖：

- `SplitSubgoalFromHistory`

### Stage 3：Debug train-step smoke

验证内容：

- 使用 `debug_pi05_dcc` 创建 fake batch。
- 走完整 data loader → `Observation.from_dict()` → `Pi0.embed_prefix()` → `compute_loss()`。
- 检查 DCC 开启且 subgoal 缺失时 prefix length 仍稳定。
- 检查 loss 返回 `{"flow", "ki_fast"}`，且全部 finite。
- 修复了多卡服务器上 batch size 小于 device count 时的 sharding 问题：测试脚本显式使用 single-device sharding。

覆盖：

```text
FakeDataConfig
→ DCC Observation fields
→ subgoal visual prefix
→ MEM state/image history
→ KI FAST tokens
→ flow + KI CE loss
```

### 本地验证状态

已通过 V1 全部测试；V2 修改后新增独立 token segment 断言，需在完整依赖环境中重跑：

```bash
bash scripts/validate_dcc_local.sh
```

验证脚本包含：

```bash
python -m py_compile ...
python tests/dcc/stage1_context_transforms.py
python tests/dcc/stage2_subgoal_split.py
python tests/dcc/stage3_train_step_smoke.py
git diff --check
```

## 对照原复现方案：已完成内容

### 已完成

- `Pi0Config` 增加 DCC 开关、subgoal 开关和各组件 dropout 概率。
- `Observation` 增加 `subgoal_images / subgoal_image_masks`。
- `embed_prefix()` 接入 subgoal image SigLIP tokens。
- subgoal type embedding zero-init，保证新增参数不会一开始强扰动模型。
- dropout 在 transform 层执行，forward / loss 路径透明。
- `LiberoInputs` 透传 subtask / metadata / control mode / subgoal image。
- data loader 支持读取 future frame 作为训练用真实 subgoal。
- 新增主配置和 ablation 配置。
- `compute_loss()`、`sample_actions()` 保持不改。
- 新增验证矩阵，覆盖 dropout、独立 segment tokenization、subgoal split、prefix shape、debug train-step。

### 部分完成

- 原方案要求 prefix 中有 `<SUBGOAL_BEGIN>` / `<META_BEGIN>` 等 sentinel；V2 暂未加 special token，使用独立 segment 边界和文本标签。
- 原方案的 subgoal 采样是 `[2s, 6s]` 或 `[0, 4s]` 随机未来帧；V2 先用固定 `subgoal_delta_seconds=2.0`。
- 原方案希望 metadata speed 按 action norm 三分位自动生成；V2 已支持读取 `speed_bin`，但未实现自动生成脚本。
- 原方案提出 LIBERO/DROID 真实数据训练；V2 先完成 fake/debug 和 transform 验证。

### 尚未完成

- `scripts/prepare_libero_subtasks.py`：LIBERO 子任务分段 / 标注准备脚本。
- `scripts/prepare_libero_metadata.py`：speed / quality / mistake 自动 metadata 生成脚本。
- InfiData parquet / jsonl 到 openpi LeRobot pipeline 的完整生产级接入。
- special token / sentinel embedding 版本的严格 prefix packing。
- random subgoal horizon 策略。
- generated subgoal / World Model 推理时接入。
- PyTorch DCC 路径。
- 真实数据训练和 ablation。

## 后续验证计划

### DC-V4：Steerability

数据准备好并训练 DCC checkpoint 后，验证：

- 固定任务，改变 `metadata.speed`，action norm 是否单调变化。
- 固定主任务，改变 `subtask`，轨迹策略是否变化。
- 给不同 subgoal image，动作是否朝向对应未来状态。

### DC-V5：Mixed-quality data robustness

构造 noisy demo：

- clean demo：`quality=5, mistake=False`
- noisy demo：`quality=2, mistake=True`

比较：

- `pi05_dcc_libero`
- `pi05_dcc_libero_no_metadata`

目标是验证 metadata 能让模型区分干净 / 脏 demo。

### DC-V6：Subgoal 加速收敛

比较：

- `pi05_dcc_libero`
- `pi05_dcc_libero_no_subgoal`

观察前 10k step flow loss 是否含 subgoal 的版本下降更快。

### DC-V7：缺字段鲁棒性

训练后推理时分别删除：

- subgoal
- subtask
- metadata

验证 DCC + dropout 训练得到的模型是否比 no-dropout 版本更稳。

## 当前结论

第三步 DCC 的 V2 代码构建已经完成。V1 已通过基础验证；V2 已通过本地语法和 diff 检查，需要在完整依赖环境中重跑 `validate_dcc_local.sh`。当前实现已经具备：

- 独立 metadata / control / subtask token segment；
- per-component dropout；
- 真实未来帧 subgoal 图像条件；
- KI + MEM + DCC 联合 forward / loss 路径；
- debug 级 shape / prefix / loss 回归测试。

下一阶段重点不再是基础代码路径，而是数据准备和训练验证：先接入 InfiData / LIBERO 的真实 DCC 字段，再做小规模 fine-tune，最后推进 steerability、mixed-quality robustness 和 subgoal acceleration 等论文 claim 的实验复现。
