# π0.7 MEM 相关文献调研

调研日期：2026-05-17

## 0. 结论先行

Physical Intelligence 的 MEM 可以放在 VLA 长时程能力的一条主线上理解：**把当前观测之外的历史信息变成可控、可压缩、可更新的条件**。它的特殊之处不是“看多帧”，而是把记忆拆成两个时间尺度：

- **short-term memory**：视频编码器用时空分离注意力压缩几十秒内的视觉历史，只向 VLA backbone 传递当前时刻的历史增强视觉 token。
- **long-term memory**：同一个 backbone 周期性生成自然语言 memory summary，把几分钟级任务进度、隐藏物体位置、已经完成的步骤、环境属性等写进文本上下文。

与你当前复现最相关的判断：

- 你现在的短期实现“在 SigLIP/ViT 层内插入 temporal attention，并只保留当前帧 token”是对 MEM 论文最关键设计的正确靠近，比简单 frame concat 更像原文。
- 长期记忆的代码闭环已经有了，但真正决定效果的是 **summary 监督数据、训练-推理分布偏移、memory-specific benchmark**，这三块目前比模型结构本身更关键。
- VLA 记忆评测还没有统一标准。主流做法是把记忆能力拆成 hidden-object recall、task-progress tracking、long-horizon multi-step completion、failed-attempt recovery、environment property inference、latency/token budget 等指标。

## 1. MEM 原论文：多尺度记忆的定位

来源：

- Physical Intelligence, `π0.5: a Vision-Language-Action Model with Open-World Generalization`, arXiv 2504.16054: https://arxiv.org/abs/2504.16054
- Physical Intelligence, `Memory: Scaling Robot Memory to 15 Minutes`, arXiv 2603.03596 / PI 页面: https://www.pi.website/research/memory
- PI MEM PDF: https://www.pi.website/download/Mem.pdf

MEM 的核心动机是：机器人执行长任务时，单帧或短窗口策略会忘记“刚才放了什么、做到了哪一步、某个不可见物体在哪里、某次尝试为什么失败”。论文把记忆分成：

- **短期视觉记忆**：覆盖 54 秒左右的稠密视觉历史。视频 encoder 做 space-time separable attention，压缩成包含历史信息的当前视觉表示。
- **长期语言记忆**：覆盖 15 分钟左右。模型周期性生成自然语言摘要，例如已经完成的动作、发现的物体位置、环境约束等。
- **关键工程点**：长期摘要不是额外模型生成，而是同一个 VLA backbone 自回归生成，因此不会引入单独的 memory LM 参数。
- **关键训练点**：训练演示往往是近最优的，但推理时会出现失败、重试、绕路，因此需要把成功轨迹和失败/恢复行为混合进训练分布。

PI 的评测不是单一 benchmark，而是几类真实长时程任务：

- 多步骤食谱/清洁/整理任务，测试是否记得任务进度。
- 隐藏或离开视野的物体，测试是否记得位置。
- 环境属性，例如冰箱铰链方向，测试是否记得早先观察到的状态。
- 需要调整动作参数的任务，例如筷子高度，测试短期视觉历史是否能修正执行。

论文报告的 ablation 方向对复现很重要：

- 关掉视频记忆后，模型更容易在中等长度任务中卡住或重复。
- 关掉文本摘要后，模型更容易丢失跨分钟级任务步骤。
- 短期和长期记忆不是互斥替代，而是互补。

## 2. 相邻工作谱系

### 2.1 历史帧 / 视频上下文型

这类方法把历史观测作为视觉序列输入，重点是解决短期遮挡、动作连续性和局部历史依赖。

代表：

- RT-1 / RT-2：早期机器人 transformer/VLA 通常使用短历史或 action chunk，但没有显式长期记忆。
  - RT-1: https://arxiv.org/abs/2212.06817
  - RT-2: https://arxiv.org/abs/2307.15818
- Octo：开放式机器人策略，支持多模态输入和历史窗口，但主要不是长期 memory summary。
  - https://arxiv.org/abs/2405.12213
- OpenVLA：开源 7B VLA，常见使用方式仍偏当前观测到动作，不以显式 memory 为核心。
  - https://arxiv.org/abs/2406.09246

和 MEM 的关系：

- 它们证明了历史窗口对控制有用，但大多没有把历史压缩成“当前 token”，也没有分钟级自然语言摘要。
- 对你的短期模块来说，这些工作更适合作 baseline：`current frame`、`N-frame concat`、`sliding window video encoder`、`MEM compressed current token`。

### 2.2 显式双记忆 / 多尺度记忆型

#### MemoryVLA

来源：

- Project: https://shihao1895.github.io/MemoryVLA/
- Paper 页面 / arXiv 聚合: https://arxiv.org/abs/2508.19236

MemoryVLA 明确提出 **perceptual memory + cognitive memory**：

- perceptual memory：保存关键视觉经验，辅助细粒度操作。
- cognitive memory：保存语义级经验、任务状态和环境知识。

它的思路和 PI MEM 很接近：短期/感知记忆负责“刚刚看见和做过什么”，长期/认知记忆负责“任务层面的事实和进展”。差异在于 MemoryVLA 更强调 memory bank / retrieval 式结构，而 PI MEM 更强调同 backbone 生成和消费自然语言 summary。

报告中值得关注的数字：

- LIBERO-Spatial / Object / Goal 分别约 71.9 / 72.7 / 96.5。
- MIKASA-Robo 约 41.2。
- 真实机器人 12 个任务约 84% 成功率，相比 OpenVLA-OFT 提升约 26 个百分点。

对复现启发：

- 如果你后续想增强长期记忆，可以把自然语言 summary 之外再加一个 retrieval memory bank，但 MVP 阶段不必做。
- 可以借鉴它的评测拆分：空间关系、物体状态、目标达成、真实长任务。

#### ReMem-VLA

来源：

- arXiv: https://arxiv.org/abs/2603.12942

ReMem-VLA 的关键词是 **hierarchical recurrent memory**。它把长期历史分成两个递归层次：

- frame-level memory：保留较密的局部历史。
- chunk-level memory：把一段时间压缩为更粗粒度记忆。

和 PI MEM 的差异：

- PI MEM 的长期通道是自然语言 summary，可解释性强，也容易和 prompt 融合。
- ReMem-VLA 更像 latent recurrent state，可能更连续、更紧凑，但可解释性弱。

对复现启发：

- 你当前 short-term 压缩 + long-term text summary 是 PI 路线；如果后续发现文本 summary 太难训，可以加一个 latent chunk memory 作为中间尺度。
- 评测时可加入“摘要可读性”和“反事实 summary 改动作”两类 probe，体现 text summary 的优势。

### 2.3 规划 trace / 高层状态记忆型

#### IVLR / Trace-Conditioned VLA Planning

来源：

- arXiv: https://arxiv.org/abs/2604.21924

这类方法不一定叫 memory，但功能上非常接近长期记忆：模型显式维护一个 execution trace / plan trace，记录任务分解、已完成步骤和下一步目标。

论文报告：

- 在 LIBERO 平均成功率约 95.5。
- 在 LIBERO-Long 约 92.4。
- 去掉 trace 后 LIBERO-Long 大幅下降到约 37.7。

和 MEM 的关系：

- PI MEM 的 summary 更自由，可以记录视觉事实、失败尝试、环境属性。
- Trace-conditioned planning 更结构化，尤其适合 multi-step instruction。

对复现启发：

- 长期 summary 不应只写“看到了什么”，也应写“已经完成了哪些子任务，下一步还缺什么”。
- 可以把 summary prompt 从自由格式改成半结构化字段：
  - `Completed: ...`
  - `Known locations: ...`
  - `Failed attempts: ...`
  - `Next: ...`

#### Long-VLA / LoHo-Manip / RAM 类系统

来源：

- Long-VLA, arXiv: https://arxiv.org/abs/2508.19958
- LoHo-Manip / Trace-Conditioned VLA Planning, arXiv: https://arxiv.org/abs/2604.21924

这类工作把长时程 VLA 拆成 reasoning、acting、memory 三个子系统。通常做法是：

- 高层 reasoner 生成子目标或计划。
- 低层 VLA 执行动作。
- memory 模块维护任务历史和环境状态。

和 PI MEM 的关系：

- PI MEM 更“端到端”：同 backbone 既行动又写 summary。
- Long-VLA 更关注长任务中的 phase-aware 输入选择和 skill chaining。
- LoHo-Manip 更像“任务管理器 + 执行器”：任务管理器维护 done / remaining 的语言记忆，并生成视觉 trace 引导低层 VLA。

对复现启发：

- 你第五步 High-Level Policy 可以和 MEM 接起来：长期 summary 给高层 policy 做 task-progress state，而短期 memory 给低层 action policy 做控制状态。

### 2.4 外部检索 / MAP / prompt memory 型

#### MAP-VLA / Memory-Augmented Prompting

来源：

- arXiv: https://arxiv.org/abs/2511.09516

MAP-VLA 的核心是把经验或历史事实作为 prompt augmentation 引入 VLA，而不是大改模型结构。它适合 low-cost adaptation。

和 PI MEM 的关系：

- MAP 更像“外部可检索经验库 + prompt 注入”。
- PI MEM 是 episode 内在线记忆，重点是当前任务内的时间连续性。

对复现启发：

- 你当前 `PrependMemorySummaryToPrompt` 已经是 MAP 风格接口。
- 后续可以把 long-term summary 的来源分为两类：episode 内 summary、跨 episode retrieval memory。

### 2.5 Belief state / partial observability 型

#### RB-VLA / Belief-aware VLA

来源：

- arXiv: https://arxiv.org/abs/2602.20659

这类方法把 memory 写成 **belief state**：模型维护对不可见世界状态的估计，例如隐藏物体位置、门是否开过、抽屉里有什么。它更接近 POMDP 视角。

论文报告中可借鉴的现象：

- 使用 belief 后的成功率显著高于无 belief baseline。
- 对“被遮挡/离开视野但仍影响动作”的任务尤其有效。

和 PI MEM 的关系：

- PI MEM 的语言 summary 可以看成一种可读的 belief state。
- Belief-aware VLA 更强调状态估计正确性，因此评测会更关注 recall / consistency，而不仅是最终成功率。

对复现启发：

- 你的长期记忆评测不应只看 action loss 或 task success，还应单独测 memory fact accuracy。
- 例如：红块最后被放在哪个抽屉、杯子是否已经倒空、冰箱门铰链在哪边。

### 2.6 其他可关注方向

- HELM：强调长时程执行失败不只是 context 不够，而是 memory gap、verification gap、recovery gap 的组合。它用 CLIP-indexed keyframe episodic memory、state verifier 和 harness controller 做检索、失败预测、回滚重规划，在 LIBERO-LONG 上把 OpenVLA 从 58.4% 提到 81.5%，并提出 LIBERO-Recovery 扰动协议。来源：https://arxiv.org/abs/2604.18791
- CoMe-VLA：强调 context / memory-aware VLA，在长时程任务中通过上下文压缩或记忆选择提升执行。
- Compressor-VLA / VLA-Cache：更多是推理加速和 token 压缩，不直接解决语义长期记忆，但对 MEM 的高 token 成本有工程价值。
  - VLA-Cache: https://arxiv.org/abs/2502.02175

## 3. VLA 中 MEM 能力如何评测

目前没有统一的“VLA memory benchmark”。不同论文通常用以下几类任务拼接评测。

### 3.1 最终任务成功率

最常见，也是最容易比较的指标。

常用 benchmark：

- LIBERO / LIBERO-Long：多任务机器人操作，Long 子集适合测任务进度记忆。
  - https://libero-project.github.io/main.html
- CALVIN / L-CALVIN：长时程语言条件操作，适合 multi-step instruction。
  - https://arxiv.org/abs/2112.03227
- SimplerEnv：评估真实到仿真、Bridge/Fractal 等设置下的机器人策略。
  - https://arxiv.org/abs/2405.05941
- MIKASA-Robo：偏多任务、多物体、多场景泛化。
  - https://arxiv.org/abs/2408.11852

局限：

- 成功率能说明“记忆有用”，但不能说明模型到底记住了什么。
- 对 memory summary 来说，必须配合 recall probe 或 ablation。

### 3.2 Memory fact recall / consistency

这是最该补的指标。做法是把任务中的关键事实显式标注出来：

- 隐藏物体位置：`red block in left drawer`
- 已完成步骤：`plate already placed in cabinet`
- 环境属性：`fridge hinge is on the right`
- 失败尝试：`top drawer blocked, use lower drawer`

评测指标：

- summary 中是否包含正确事实。
- 当前观测相同但 memory 不同，动作是否随 memory 改变。
- memory 是否在事实过期后更新，而不是一直保留旧事实。

你已有的 stage7 / stage8 / stage9 toy probe 正是这个方向，后续可以真实化。

### 3.3 Ablation

MEM 论文和后续工作普遍会做：

- no memory：单帧 baseline。
- short-term only：视频历史，不给 summary。
- long-term only：summary，不给视频历史。
- shuffled memory：给错误 summary，测试模型是否真的使用 memory。
- stale memory：不给更新，测试长期任务中是否崩。
- full frame concat：把所有帧直接丢给 backbone，对比 compressed short memory 的 token/latency收益。

这对你的复现尤其重要，因为没有 PI 私有数据时，ablation 的说服力比绝对成功率更重要。

### 3.4 时间尺度与延迟

Memory 模块的工程指标应包括：

- 支持的历史跨度：秒级、分钟级。
- 每步 action inference latency。
- 每次 summary update latency。
- prefix token 数、KV cache 大小、显存。
- 长 episode 中 memory update 是否稳定。

PI MEM 目标是 15 分钟上下文；你的复现可以先明确：

- short-term：6 帧 / 5 秒，后续扩到 16-18 帧 / 54 秒。
- long-term：每 30 step 更新一次，summary 64-96 token。

## 4. 当前领域水平的粗略判断

截至 2026-05，VLA memory 处在“快速出现论文，但标准尚未统一”的阶段：

- **短期记忆**已经比较确定：视频历史、时空分离注意力、recurrent/chunk memory 都能提升遮挡和局部时序控制。
- **长期记忆**还没有收敛：自然语言 summary、latent recurrent memory、belief state、retrieval memory、execution trace 都有人做。
- **评测还偏碎片化**：LIBERO-Long、CALVIN、MIKASA、真实厨房/桌面任务都在用，但很少有专门只测 memory 的统一 benchmark。
- **开源复现难点主要不是结构**：而是长时程数据、失败恢复数据、summary 监督，以及真实机器人闭环评测。

可以把当前水平理解成：

- 在仿真长任务里，显式 trace / memory 往往能把 LIBERO-Long 这类任务从不稳定提升到 80-90% 区间，但结果高度依赖训练数据和任务定义。
- 在真实机器人里，memory 类方法通常能带来 20-30 个百分点量级的成功率提升，但任务集较小，跨论文不可直接横比。
- PI MEM 的亮点是 15 分钟真实任务记忆和同 backbone 生成 summary，这比多数开源 VLA 的短历史上下文更进一步。

## 5. 对当前 π0.7 复现的建议

### 5.1 保留当前结构路线

你当前实现的关键选择是合理的：

- short-term memory 在视觉 encoder 内部融合，而不是把历史帧全部展开给 Gemma。
- temporal attention 只增强当前帧 token，控制 prefix token 数。
- long-term memory 走同 backbone generation + prompt 注入。
- policy 端维护 episode-level memory state。

这条路线和 PI MEM 的“短期视觉压缩 + 长期语言摘要”高度一致。

### 5.2 下一步最该补：真实 memory benchmark

建议新增一个 `tests/mem_benchmark/` 或 `benchmarks/mem/`，先做仿真/脚本化任务：

1. **Hidden Object Recall**
   - 前半段看到物体被放入左/右抽屉，后半段当前图像完全相同。
   - 指标：summary fact accuracy、action counterfactual accuracy、成功率。

2. **Task Progress Tracking**
   - 多步骤任务：拿杯子、放盘子、关门、擦桌子。
   - 指标：是否重复已完成步骤、是否跳过未完成步骤。

3. **Environment Property Memory**
   - 早期观察到门把手/铰链方向/容器位置，后面才需要使用。
   - 指标：第一次尝试是否选对动作方向。

4. **Failed Attempt Recovery**
   - 设置一个不可行动作，例如目标抽屉被挡住。
   - 指标：summary 是否写入失败原因，下一次是否换策略。

5. **Short-term Fine Control**
   - 可变高度、滑动物体、临时遮挡。
   - 指标：short-term only 是否已经显著提升。

### 5.3 长期 summary 训练数据优先级

建议从低成本到高质量分三阶段：

- **阶段 1：规则生成 / synthetic**  
  保持你当前 toy probe，用于验证代码路径。

- **阶段 2：仿真任务自动标注**  
  对 LIBERO/CALVIN episode 使用环境 state 自动生成 summary target。质量比外部 VLM 更稳定。

- **阶段 3：VLM 离线标注真实视频**  
  对 DROID/OXE 子集用 GPT-4o / Gemini / Qwen-VL 生成 episode summary，再人工抽检。

阶段 2 最适合你现在推进，因为它能产生可控 GT，也能直接做 memory fact accuracy。

### 5.4 需要补的对照实验

建议最少做这些：

- `baseline_pi05`: 单帧，无 MEM。
- `short_mem_only`: 只开视频短期记忆。
- `long_mem_only`: 只注入 memory summary。
- `full_mem`: 短期 + 长期。
- `wrong_summary`: 当前观测相同，summary 反事实错误。
- `stale_summary`: summary 不更新。
- `frame_concat`: 直接拼历史帧 token，不做压缩。

对外写论文/报告时，`wrong_summary` 和 `stale_summary` 很关键：它们能证明模型不是只是因为多了文本 token 或历史帧而提升，而是真的依赖 memory 内容。

## 6. 文献清单

优先精读：

- PI MEM: https://www.pi.website/research/memory
- PI MEM PDF: https://www.pi.website/download/Mem.pdf
- π0.5: https://arxiv.org/abs/2504.16054
- MemoryVLA: https://arxiv.org/abs/2508.19236
- ReMem-VLA: https://arxiv.org/abs/2603.12942
- Trace-Conditioned VLA Planning / IVLR: https://arxiv.org/abs/2604.21924
- Long-VLA: https://arxiv.org/abs/2508.19958
- HELM: https://arxiv.org/abs/2604.18791
- RB-VLA: https://arxiv.org/abs/2602.20659
- MAP-VLA: https://arxiv.org/abs/2511.09516

作为背景：

- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818
- Octo: https://arxiv.org/abs/2405.12213
- OpenVLA: https://arxiv.org/abs/2406.09246
- SimplerEnv: https://arxiv.org/abs/2405.05941
- CALVIN: https://arxiv.org/abs/2112.03227
- VLA-Cache: https://arxiv.org/abs/2502.02175
