# π0.7 MEM 长期记忆代码修改总结

## 目标

本次修改实现 MEM 第二阶段：长期语言记忆的代码闭环与 5 个定向验证实验。目标不是直接做大规模数据训练，而是在已经完成的 KI 与 MEM 短期记忆基础上，先验证长期记忆相关代码路径的准确性和有效性。

核心闭环：

```text
脚本合成摘要监督
→ 同 backbone 训练/生成 memory summary
→ policy 在 episode 内维护长期记忆状态
→ 将 summary 注入 prompt
→ action policy 基于长期记忆做决策
```

本阶段明确采用 synthetic / toy probe 做验证，不依赖外部 LLM 标注、不下载真实数据集、不要求大规模 fine-tune。

## 主要代码改动

### 1. MEM 长期记忆配置

文件：`src/openpi/models/pi0_config.py`

新增字段：

- `long_memory_loss_weight: float = 0.2`
- `memory_summary_max_len: int = 96`
- `memory_generation_max_new_tokens: int = 64`
- `memory_update_interval_steps: int = 30`

行为变化：

- `long_memory_enabled=True` 时，`max_token_len` 自动提升到至少 `384`，减少 `Memory + Task` prompt 被截断的风险。
- 增加配置合法性检查：
  - `long_memory_loss_weight >= 0`
  - `memory_summary_max_len >= 2`
  - `memory_generation_max_new_tokens >= 1`
  - `memory_update_interval_steps >= 1`

### 2. Observation 长期摘要监督字段

文件：`src/openpi/models/model.py`

新增可选字段：

- `memory_summary_tokens`
- `memory_summary_mask`
- `memory_summary_ar_mask`
- `memory_summary_loss_mask`

这些字段只用于长期摘要生成 CE 分支，不进入 action flow 分支。这样可以避免 teacher-forced 目标摘要泄漏到动作监督路径。

`Observation.from_dict()`、`preprocess_observation()` 也同步透传这些字段。

### 3. 长期摘要 tokenizer 与 transform

文件：

- `src/openpi/models/tokenizer.py`
- `src/openpi/transforms.py`
- `src/openpi/training/config.py`

新增 `PaligemmaTokenizer.tokenize_memory_summary_supervision()`：

```text
Task: <prompt>
Current memory: <memory_summary>
New memory: <target_memory_summary>
```

其中：

- prefix 部分双向可见，不计入 loss。
- target memory summary 部分使用 causal mask，并计入 CE loss。
- 自动 padding / truncation 到 `memory_summary_max_len`。

新增 `PaligemmaTokenizer.decode()`，用于 policy 端将生成 token 转回字符串。

新增 `TokenizeMemorySummarySupervision` transform：

- 输入字段：`target_memory_summary`
- 可选输入字段：`memory_summary`
- 输出字段：
  - `memory_summary_tokens`
  - `memory_summary_mask`
  - `memory_summary_ar_mask`
  - `memory_summary_loss_mask`

在 `ModelTransformFactory` 中，当 `long_memory_enabled=True` 时自动启用该 transform，并在 `PrependMemorySummaryToPrompt` 之前执行。

### 4. Pi0 长期摘要 CE 分支

文件：`src/openpi/models/pi0.py`

新增：

- `_has_memory_summary_supervision()`
- `compute_memory_summary_loss()`

实现方式：

1. 正常调用 `embed_prefix()` 构造视觉 / state memory / prompt / KI FAST prefix。
2. 单独 embed `memory_summary_tokens`。
3. 将 prefix 与 summary teacher-forcing tokens 拼接。
4. 使用 `memory_summary_ar_mask` 构造 autoregressive attention。
5. 使用 `decode_logits` 计算 next-token CE。
6. 只在 `memory_summary_loss_mask=True` 的 target summary token 上计 loss。

`compute_loss()` 行为变化：

- 原始非 KI / 非长期记忆模式仍返回 flow loss array，保持兼容。
- 如果启用 KI，返回 `{"flow", "ki_fast"}`。
- 如果提供长期摘要监督，额外返回 `{"mem_summary"}`。
- 可同时返回 `{"flow", "ki_fast", "mem_summary"}`。

训练聚合在 `scripts/train.py` 中同步更新：

```text
total = mean(flow)
      + ki_alpha * mean(ki_fast)
      + long_memory_loss_weight * mean(mem_summary)
```

### 5. Pi0 长期摘要生成接口

文件：`src/openpi/models/pi0.py`

新增：

- `generate_memory_summary_tokens(observation, max_new_tokens=None)`

功能：

- 复用同一个 PaliGemma / Gemma backbone，不引入额外模型。
- 用当前 observation 的 prefix 作为上下文。
- greedy decoding 生成最多 `memory_generation_max_new_tokens` 个 token。
- 默认 EOS token id 为 `1`。
- 当前实现保持非 jitted 简洁路径，适合作为阶段 B 代码构建与小规模验证入口。

### 6. Policy 推理端长期记忆状态

文件：

- `src/openpi/policies/policy.py`
- `src/openpi/policies/policy_config.py`

`Policy` 新增 episode-level state：

- `memory_summary: str`
- `memory_step: int`

新增方法：

- `reset_memory()`
- `get_memory_summary()`

推理行为：

1. 每次 `infer()` 前，将当前 `memory_summary` 注入 obs。
2. 数据 transform 会通过 `PrependMemorySummaryToPrompt` 拼成：

```text
Memory: <summary>
Task: <prompt>
```

3. 每隔 `memory_update_interval_steps`，调用 `generate_memory_summary_tokens()`。
4. 用 `PaligemmaTokenizer.decode()` 将 token 转成字符串并更新内部 memory state。
5. 输出中附带：
   - `memory_summary`
   - `memory_step`

PyTorch 路径暂不支持长期记忆：

- `src/openpi/models_pytorch/pi0_pytorch.py` 中，当 `long_memory_enabled=True` 时显式抛出 `NotImplementedError`。

## 新增验证实验

新增文件：

- `tests/mem/stage6_summary_ce_overfit.py`
- `tests/mem/stage7_memory_action_counterfactual.py`
- `tests/mem/stage8_hidden_object_recall.py`
- `tests/mem/stage9_counting_probe.py`
- `tests/mem/stage10_closed_loop_memory_state.py`

### Stage 6：摘要生成 CE Overfit

目的：验证长期摘要监督目标本身可学习，而不是只把 summary 当普通 prompt 读入。

设计：

- 构造 4 类 synthetic memory facts：
  - `red block in left drawer`
  - `red block in right drawer`
  - `blue block in left drawer`
  - `blue block in right drawer`
- 用 one-hot fact feature 训练一个轻量 classifier 模拟 summary CE overfit。
- 同时构造 blank/no-fact baseline。

通过标准：

- CE loss 明显下降。
- fact-conditioned accuracy 达到 90%+。
- blank baseline 不能恢复正确摘要。

当前结果：

```text
PASS: summary CE overfit loss 0.8936 -> 0.0365, acc=1.00, blank_acc=0.25
```

### Stage 7：Memory-Affects-Action Counterfactual

目的：验证在当前图像、state、task 完全相同的情况下，仅改变 memory summary 就能改变动作。

设计：

- 两条样本当前 observation 相同。
- memory 不同：
  - left drawer
  - right drawer
- 监督相反动作。
- 对比 memory-conditioned policy 与 blank-memory baseline。

通过标准：

- memory-conditioned policy 能拟合相反动作。
- blank baseline 明显退化。
- 同 observation + 不同 memory 的动作距离显著大。

当前结果：

```text
PASS: memory changes action loss=0.5000->0.0000, blank_loss=0.5000
```

### Stage 8：Hidden Object Recall Probe

目的：验证长期语言记忆能保存“当前视觉已经不可见”的历史事实。

设计：

- t0：目标位置可见。
- t1/t2：目标被遮挡，当前 observation 不再包含位置信息。
- 最终动作必须依据 memory summary 选择 left/right。
- 对比 full memory、no-memory、shuffled-memory。

通过标准：

- full memory 准确率为 1.0。
- no-memory 接近随机。
- shuffled-memory 明显失败。

当前结果：

```text
PASS: hidden object recall full_acc=1.00, no_memory_acc=0.50, shuffled_acc=0.00
```

### Stage 9：Long-Horizon Counting Probe

目的：验证长期记忆能承担短期视觉不擅长的符号计数任务。

设计：

- 任务：`pick up 3 blocks one by one`
- 当前 observation 不显式编码已经拿了几个。
- memory summary 维护计数：
  - zero
  - one
  - two
  - three
- action 根据 summary 决定继续 pick 或 stop。

通过标准：

- summary 能按步更新计数。
- 第 4 步正确 stop。
- 错误 summary 注入能控制错误动作，证明 policy 读取 summary。
- no-memory baseline 不知道何时停止。

当前结果：

```text
PASS: counting probe summaries=['I have picked up zero blocks.', 'I have picked up one block.', 'I have picked up two blocks.', 'I have picked up three blocks.'], actions=['pick', 'pick', 'pick', 'stop']
```

### Stage 10：Closed-Loop Memory State Integration

目的：验证摘要生成、policy memory state、summary 注入、动作决策能组成完整闭环。

设计：

- synthetic runner 模拟一个 episode。
- 第一步看到隐藏事实。
- policy 更新 `memory_summary`。
- 后续 query 阶段当前 observation 不再包含事实。
- 对比：
  - full long-term MEM
  - generation disabled
  - reset mid-episode

通过标准：

- full 组正确利用 memory 动作。
- generation disabled 无法恢复事实。
- reset 后行为退化。

当前结果：

```text
PASS: closed-loop memory state full=go_left, disabled=unknown, reset=unknown
```

## 总验证脚本

新增文件：

```text
scripts/validate_mem_long_term_local.sh
```

执行内容：

1. 运行现有 short-term MEM pytest / stage scripts。
2. 运行 long-term MEM stage6-stage10。

建议命令：

```bash
bash scripts/validate_mem_long_term_local.sh
```

## 本地验证状态

已通过：

```bash
python -m py_compile ...
git diff --check
python tests/mem/stage6_summary_ce_overfit.py
python tests/mem/stage7_memory_action_counterfactual.py
python tests/mem/stage8_hidden_object_recall.py
python tests/mem/stage9_counting_probe.py
python tests/mem/stage10_closed_loop_memory_state.py
```

未能在当前环境完整运行 pytest / 原有 MEM stage scripts，原因是当前 Python 环境缺少项目依赖：

- `pynvml`
- `flax`

例如：

```text
ModuleNotFoundError: No module named 'pynvml'
ModuleNotFoundError: No module named 'flax'
```

安装依赖或进入项目虚拟环境后，建议运行：

```bash
bash scripts/validate_mem_long_term_local.sh
```

## 当前实现边界

- 长期记忆只实现 JAX 主路径。
- PyTorch 路径显式不支持长期 MEM，避免静默错误。
- 本阶段 synthetic 验证主要证明代码路径与机制正确，不代表真实 benchmark 收益。
- 摘要监督当前使用脚本合成模板；后续可替换为 LLM 离线标注或人工标注。
- 摘要生成接口目前采用 greedy decoding；后续可增加 temperature / top-k / top-p 等采样参数。

## 后续建议

下一步建议分两条线推进：

1. **代码级增强**
   - 为 `compute_memory_summary_loss()` 加更细的单元测试，确认 target-only loss mask 无泄漏。
   - 为 `Policy.reset_memory()` 接入真实环境 episode reset。
   - 增加 memory summary 日志记录，便于后续评测分析。

2. **真实任务验证**
   - 将 synthetic hidden recall / counting probe 迁移到 LIBERO toy environment。
   - 训练小规模 long-memory adapter / LoRA。
   - 对比：
     - no memory
     - short-term MEM only
     - external GT memory
     - generated memory

## 追加：流程级验证与日志记录

针对“现有验证偏机制级，没有充分触及 Pi0 架构上长期摘要训练/推理流程”的问题，已补充以下检查：

- `test_long_memory_summary_loss_mask_can_disable_ce`
  - 将 `memory_summary_loss_mask` 全部置为 False。
  - 期望 `compute_memory_summary_loss()` 返回 0。
  - 用于验证 loss mask 不会在无监督 token 上错误计 loss。

- `test_long_memory_summary_loss_gradients_do_not_update_action_expert`
  - 对 `compute_memory_summary_loss()` 单独求梯度。
  - 期望 VLM / image / state-memory 侧有梯度，action expert 与 flow action projection 无梯度。
  - 用于验证长期摘要 CE 分支不会污染 action expert。

- `test_long_memory_prompt_injection_happens_before_tokenization`
  - 直接测试 `PrependMemorySummaryToPrompt`。
  - 确认注入格式为：

```text
Memory: <summary>
Task: <prompt>
```

### Memory 日志记录

文件：`src/openpi/policies/policy.py`

`Policy` 现在会维护 episode 内 memory 日志：

- `get_memory_log()`
- `clear_memory_log()`
- `reset_memory()` 会同时清空 memory summary、step 和日志。

每次 long-memory inference 输出中新增 `memory_event`：

```python
{
    "step": int,
    "updated": bool,
    "summary_before": str,
    "summary_after": str,
    "generated_summary": str,
}
```

这样后续跑 LIBERO / RoboTwin / RoboMME 时，可以直接从 policy 输出或 `PolicyRecorder` 记录里分析：

- 什么时候更新了 memory。
- 更新前 summary 是什么。
- 生成的新 summary 是什么。
- action 发生变化时对应的 memory state 是什么。

## 当前 MEM 模块潜在问题

### 短期记忆潜在问题

1. **LeRobot 历史帧边界行为需要真实数据验证**
   - 当前通过 `delta_timestamps` 请求负时间历史。
   - episode 开头、缺帧、低 fps 数据集上的 padding / nearest-frame 行为需要在真实 LIBERO/DROID 数据上确认。

2. **history_stride_seconds 与数据集 fps 强绑定**
   - 当前默认 1 秒 stride。
   - 对高频控制数据合理，但对低 fps 或不规则时间戳数据可能取到重复帧或过远帧。

3. **video encoder 复用 spatial attention 参数做 temporal attention**
   - 优点是参数少、checkpoint 对齐简单。
   - 风险是 spatial attention 权重未必天然适合 temporal attention，可能需要 LoRA 或少量 temporal-specific 参数做 ablation。

4. **短期视觉压缩只保留当前帧 token**
   - 符合当前复现设计，显著节省 Gemma prefix token。
   - 但如果任务需要显式比较远期多帧视觉细节，压缩瓶颈可能损失信息。

5. **state_history 与 visual history 同长度**
   - 目前默认同一个 `history_length`。
   - 真实机器人状态可能需要更高频历史，而视觉历史可以低频；后续可拆成独立配置。

### 长期记忆潜在问题

1. **摘要监督目前还没有真实数据来源**
   - 当前 synthetic probe 只能证明代码路径正确。
   - 后续需要 LIBERO/RoboTwin 上的脚本标注、人工标注或 LLM 离线标注。

2. **`memory_summary_ar_mask` 已支持 per-example，但仍需真实 batch 验证**
   - `compute_memory_summary_loss()` 现在支持 `[B, M]` 的 per-example AR mask，不再假设 batch 内 prefix/target 边界一致。
   - 后续应在真实不同长度 prompt/summary 混合 batch 上补集成测试，确认 padding、truncation、loss mask 三者一致。

3. **生成接口是 greedy decoding**
   - 简单稳定，但可能生成重复或退化摘要。
   - 后续可增加 temperature、top-k、stop strings、长度惩罚。

4. **摘要生成与动作推理目前同步执行**
   - 每次更新 memory 会增加推理延迟。
   - 真实控制闭环里建议异步更新，或者低频 background update。

5. **summary 注入占用 prompt token budget**
   - 当前 `long_memory_enabled=True` 时把 `max_token_len` 提升到 384。
   - 真实任务中长 instruction + long summary 仍可能截断，需要日志统计 truncation rate。

6. **训练-推理分布偏移尚未处理**
   - 训练时可能使用 GT summary。
   - 推理时使用模型生成 summary，错误会累积。
   - 后续需要 scheduled sampling 或 GT/generated summary mixture。

7. **memory reset 依赖调用方**
   - `Policy.reset_memory()` 已实现。
   - `Policy.infer()` 现在也会识别 `reset_memory` / `memory_reset` / `episode_start` / `is_first` 信号，并在注入 summary 前自动清空 memory。
   - 已新增 pytest `test_long_memory_resets_on_episode_boundary_signal`，用于验证 episode 边界信号不会继续携带旧 summary。

8. **当前 synthetic 实验还不是端到端 VLA 成功率**
   - Stage 6-10 是机制和流程 probe。
   - 真正有效性仍需迁移到 memory-intensive benchmark。
