# π0.7 KI 代码修复总结

本次修复针对 KI 复现代码与论文机制的两个关键偏差：

## 1. 修复 action expert 可见 FAST teacher-forcing token 的泄漏

原实现把 `ki_fast_tokens` 直接追加到 VLM prefix，随后用统一的 `make_attn_mask` 连接 prefix 和 action suffix。由于 action suffix 可以 attend 到全部 prefix，这会让 flow-matching action expert 在训练时看到真实 FAST 动作 token，形成目标泄漏。

修复位置：
- `src/openpi/models/pi0.py`

修复内容：
- 在 `compute_loss` 中定位 FAST token slice：`[fast_start:fast_end]`。
- 将 action suffix query 到 FAST token key 的 attention 显式置为 `False`。
- 对 action suffix 的 RoPE `positions` 减去有效 FAST token 数，使 action expert 的位置编码等价于无 KI FAST token 的路径。

结果：
- FAST token 只服务于 VLM 侧 CE 辅助 loss。
- flow loss 的前向不再依赖 teacher-forced FAST token 内容。
- 推理路径 `sample_actions` 不变。

## 2. 修复 KI FAST tokenization 重复 prompt/state 的问题

原 `KITokenize` 复用了完整 `FASTTokenizer.tokenize(prompt, state, actions)`，导致 KI prefix 里除了正常 PaliGemma prompt 外，又额外出现一遍 `Task: ... State: ...; Action: ...`。这和论文中“FAST action tokens attend to prefix”的结构不一致，也放大了第 1 个泄漏问题。

修复位置：
- `src/openpi/models/tokenizer.py`
- `src/openpi/transforms.py`

修复内容：
- 新增 `FASTTokenizer.tokenize_action_tokens(actions)`。
- 该方法只产生 action-side FAST token 序列：`Action: <FAST action tokens> | <eos>`，不再包含任务 prompt 和 state。
- `KITokenize` 改为调用 action-only tokenizer。

结果：
- VLM prefix 只保留一份 prompt/state。
- KI 追加段只承载离散动作辅助目标。
- `token_loss_mask` 与 action-only FAST 段对齐。

## 3. 增加泄漏回归测试

修复位置：
- `src/openpi/models/pi0_ki_test.py`
- `tests/ki/stage1_gradient_paths.py`

新增测试：
- 改变 `ki_fast_tokens`，保持 observation/actions 其他部分不变。
- 断言 `flow_loss` 不发生变化。

该测试直接保护“action expert 不能看到 FAST teacher-forcing token”这一 KI 论文约束。

## 4. 关于 PaliGemma loader 的再分析

本地代码中 `pi05_ki_libero` 和 `pi05_no_ki_libero` 仍配置为：

```python
weight_loaders.CheckpointWeightLoader("gs://big_vision/paligemma/pt_224.params.npz")
```

从本地实现看，`CheckpointWeightLoader` 走 Orbax `PyTreeCheckpointer`，通常更适合 openpi/orbax checkpoint 目录；而仓库里已有 `PaliGemmaWeightLoader` 专门加载 PaliGemma `.npz` 权重。

如果云端实验已经成功加载，说明云端可能满足以下情况之一：
- 云端代码已经把 loader 改过；
- 云端实际跑的 config 不是这两个 PaliGemma 起点 config；
- 缓存路径/权重格式在云端和当前本地仓库不同。

本次没有修改 loader，以免影响你已经跑通的云端实验；但当前本地代码仍建议后续统一确认。
