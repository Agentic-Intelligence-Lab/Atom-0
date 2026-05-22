# π0.7 训练数据标注需求：Long-term Memory Summary

## 1. 目标

本标注需求面向 π0.7 复现中的 MEM long-term memory 模块。当前代码闭环已经支持：

```text
memory_summary + task / observation
→ 监督生成 target_memory_summary
→ policy 将 summary 注入 prompt
→ action policy 基于长期记忆执行
```

因此训练数据的核心新增字段不是模型结构，而是每个 episode 内若干“长期记忆状态不变”的时间段摘要。标注层只需要回答：

```text
在这一段时间内，policy 应该持有的长期记忆是什么？
```

代码训练时仍然使用 `memory_summary` 和 `target_memory_summary`，但这两个字段不要求人工直接标注，而是在采样训练数据时从不同帧对应的 summary 段自动派生：

- 如果两个采样帧落在同一 summary 段：`memory_summary == target_memory_summary`，表示记忆应保持不变。
- 如果两个采样帧跨过 summary 边界：`memory_summary != target_memory_summary`，表示模型应学习更新记忆。

标注目标是让模型学会保存“当前观测可能已经看不到、但后续决策仍需要”的信息，例如任务进度、隐藏物体位置、已完成步骤、失败尝试、环境属性和计数状态。

### 当前标注范围

当前阶段只在 **RMBench 数据集** 上做 long-term memory summary 标注，用作 MEM 长期记忆模块的试点数据和闭环验证数据。暂时不对 LIBERO / ALOHA / DROID / OXE 等数据集做大规模 summary 标注。

RMBench 阶段的目标是先验证三件事：

- `memory_segments → memory_summary / target_memory_summary` 的采样逻辑是否成立。
- summary 标注是否能覆盖 RMBench 中真正依赖长期记忆的能力点，例如隐藏物体回忆、计数、跨时间引用、任务进度跟踪和失败恢复。
- 标注 schema、切段规则、质检规则和训练 sampler 是否可以稳定复用。

后续大规模标注时，再沿用本文的 `memory_segments` 结构和规则，扩展到 LIBERO-Long、ALOHA、DROID 子集或其他真实机器人长任务数据。

## 2. 与现有代码接口对齐

当前实现中，`TokenizeMemorySummarySupervision` 会读取：

```text
prompt
memory_summary
target_memory_summary
```

并构造 teacher-forcing 文本：

```text
Task: <prompt>
Current memory: <memory_summary>
New memory: <target_memory_summary>
```

推理时，`PrependMemorySummaryToPrompt` 会把长期记忆注入主任务 prompt：

```text
Memory: <summary>
Task: <prompt>
```

因此要区分两层数据：

1. **人工/自动标注层**：标 `memory_segments`，即一段时间内稳定不变的 summary。
2. **训练样本层**：data sampler 根据帧对或窗口对，动态生成 `memory_summary` / `target_memory_summary`。

标注层至少要保证：

| 字段                                |           粒度 | 是否必需 | 说明                               |
| ----------------------------------- | -------------: | -------: | ---------------------------------- |
| `episode_index`                     |        episode |     必需 | 唯一 episode id                    |
| `segment_index`                     | memory segment |     必需 | episode 内 summary 段编号          |
| `start_frame` / `end_frame`         | memory segment |     必需 | summary 在该闭区间内有效           |
| `start_timestamp` / `end_timestamp` | memory segment |     推荐 | 便于人工 review                    |
| `task` / `prompt`                   |        episode |     必需 | 总任务指令                         |
| `summary`                           | memory segment |     必需 | 该时间段内 policy 应持有的长期记忆 |
| `change_event_type`                 | memory segment |     推荐 | 进入该 summary 状态的事件类型      |
|                                     |                |          |                                    |
|                                     |                |          |                                    |
|                                     |                |          |                                    |

训练样本层再临时展开为：

| 派生字段                | 来源                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `memory_summary`        | 采样起点帧所属 segment 的 `summary`，或 episode 开始前空字符串 |
| `target_memory_summary` | 采样目标帧/窗口结束帧所属 segment 的 `summary`               |

## 3. Summary 应该记什么

只写会影响后续动作或高层决策的信息。不要把每一帧都流水账化。

优先级从高到低：

| 类型          | 必标场景                                      | 示例                                                         |
| ------------- | --------------------------------------------- | ------------------------------------------------------------ |
| 任务进度      | RMBench 多步骤 / 长时程任务；后续可扩展到 LIBERO-Long、整理/烹饪/装配类任务 | `The drawer is open. The red block has been placed in the bowl.` |
| 隐藏物体位置  | 物体被遮挡、放入容器、离开当前视野            | `The red block is in the left drawer.`                       |
| 已知环境属性  | 门铰链、容器状态、可通行方向、工具位置        | `The cabinet door opens to the right.`                       |
| 计数/重复次数 | 需要做 N 次或收集多个物体                     | `Two blocks have been picked up; one remains.`               |
| 失败与恢复    | 抓取失败、碰撞、路径受阻、重试策略            | `The first grasp missed the cup; retry from the handle side.` |
| 当前子目标    | 低层策略下一段动作需要的局部目标              | `Next, move to the blue bowl and release the block.`         |

不建议写：

- 与后续无关的视觉细节，例如背景颜色、光照、无关物体。
- 模糊心理词，例如 `I think`, `maybe`, `probably`，除非事实确实不确定。
- 过长推理链。summary 是 memory state，不是 chain-of-thought。
- 只重复总任务，例如 `The task is to put the block in the bowl.`，除非没有任何历史进展。

## 4. 推荐格式（这一部分可以看一下RMBench的任务内容再优化）

为了训练稳定，并与后面的标注输出和 MVP 示例保持一致，推荐把 `summary` 统一写成**自然语言短句**，不要使用 `Progress:` / `Known facts:` / `Issues:` 这类显式字段标签。MVP 阶段优先统一成 1-4 句英文，长度控制在 10-60 words，最多不超过 `memory_summary_max_len=96` token。

推荐模板：

```text
<current persistent facts>. <completed or failed action if relevant>. Next, <immediate next subtask>.
```

其中：

- `<current persistent facts>`：当前 segment 内应该长期保留的事实，例如物体位置、抽屉/柜门状态、计数状态、环境属性。
- `<completed or failed action if relevant>`：只有当已完成步骤或失败尝试会影响后续动作时才写。
- `Next, ...`：建议保留，用一句话写下一步局部目标；如果当前 summary 只需要记住事实、没有明确下一步，可以省略。
- 没有对应信息时直接省略该句，不要写 `none` 或空字段标签。

示例：

```text
The left drawer is open. The red block is still on the table. Next, grasp the red block.
```

```text
Two blocks have been placed in the bowl. One block remains. Next, pick up the final block.
```

```text
The first grasp missed the cup. The cup is still on the table. Next, retry the grasp from the handle side.
```



## 5. Summary 分段与切点

标注时不需要写“上一条 summary → 下一条 summary”的更新对，只需要把 episode 切成若干 summary 不变的段。切段原则是：只要长期记忆状态发生语义变化，就开一个新段；否则保持同一个 summary。

| 新段触发点              | 是否需要 | 说明                                                         |
| ----------------------- | -------: | ------------------------------------------------------------ |
| episode 起点            |     必需 | 初始可见关键事实形成第一段；如果没有有效记忆，summary 可为空 |
| 物体位置/状态发生变化后 |     必需 | 放入抽屉、抓起、释放、打开/关闭容器                          |
| 子任务完成后            |     必需 | 可与 `subtask` segment 边界对齐                              |
| 失败/重试发生后         |     推荐 | 如果数据包含失败轨迹，必须进入 summary                       |
| 关键事实过期后          |     必需 | 例如物体从原位置被拿走，需要改写旧事实                       |
| 长时间无变化            |   不需要 | summary 没变就延长当前段，不额外标重复段                     |
| episode 结束            |     推荐 | 末段 summary 可用于 consistency 检查                         |

训练采样时可以灵活构造两类样本：

| 采样方式           | `memory_summary`  | `target_memory_summary`  | 学到什么                |
| ------------------ | ----------------- | ------------------------ | ----------------------- |
| 同段内采样         | segment A summary | segment A summary        | 保持记忆不变            |
| 跨边界采样         | segment A summary | segment B summary        | 根据新事件更新记忆      |
| episode 起点采样   | 空字符串          | first segment summary    | 从观察初始化记忆        |
| reset / 新 episode | 空字符串          | 新 episode first summary | 防止跨 episode 记忆泄漏 |

## 6. 标注输出

标注员输出样例：

```json
{
  "episode_index": 12,
  "segment_index": 2,
  "start_frame": 300,
  "end_frame": 520,
  "start_timestamp": 12.0,
  "end_timestamp": 20.8,
  "task": "put the red block into the left drawer",
  "summary": "The left drawer is open. The red block is still on the table. Next, grasp the red block.",
  "change_event_type": ["drawer_opened", "subtask_completed"],
  "evidence_start_frame": 300,
  "evidence_end_frame": 375,
  "annotation_status": "human_verified"
}
```



## 8. 质量标准

每条 summary 需要满足：

1. **事实正确**：不能写视频中没有发生或与任务矛盾的事实。
2. **时间一致**：`summary` 必须在 `[start_frame, end_frame]` 整段内成立，不能包含未来才发生的信息。
3. **可行动**：后续策略能据此改变动作，例如 left/right、done/remaining、retry/continue。
4. **不泄漏未来**：不能写当前 segment 之后才发生的信息。
5. **保留未过期事实**：被遮挡但仍有效的事实要保留，例如物体放进抽屉后当前看不到。
6. **删除或改写过期事实**：物体已被拿走后，不应继续写它还在原位置。
7. **长度受控**：优先 1-4 句，避免超过 tokenizer 长度。
8. **术语稳定**：同一对象和位置用同一套名称，例如始终用 `left drawer`，不要一会儿写 `left bin`。
9. **段边界清晰**：如果 summary 中任一关键事实变真或变假，必须在该帧附近切段。



## 11. MVP 示例

### Hidden Object Recall

```json
{
  "segment_index": 0,
  "start_frame": 0,
  "end_frame": 120,
  "task": "pick up the red block from the drawer",
  "summary": "The red block is in the left drawer. The drawer is closed. Next, open the left drawer.",
  "change_event_type": ["initial_observation", "object_location_seen"]
}
{
  "segment_index": 1,
  "start_frame": 121,
  "end_frame": 260,
  "task": "pick up the red block from the drawer",
  "summary": "The red block is in the left drawer. The drawer is open. Next, reach into the left drawer and grasp the red block.",
  "change_event_type": ["drawer_opened", "subtask_completed"]
}
```

### Counting

```json
{
  "segment_index": 2,
  "start_frame": 240,
  "end_frame": 360,
  "task": "pick up three blocks and place them in the bowl",
  "summary": "Two blocks have been placed in the bowl. One block remains. Next, pick up the final block.",
  "change_event_type": ["object_placed", "count_updated"]
}
```

### Failed Attempt

```json
{
  "segment_index": 1,
  "start_frame": 100,
  "end_frame": 170,
  "task": "put the cup on the plate",
  "summary": "The first grasp missed the cup. The cup is still on the table. Next, retry the grasp from the handle side.",
  "change_event_type": ["grasp_failed", "recovery_needed"]
}
```
