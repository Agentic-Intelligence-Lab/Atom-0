# Atom0 论文数据处理细节汇总

> 用途：为 `docs/Atom0_paper.txt` 中“数据构成和占比、RLDS 格式统一转换、norm 计算、数据质量筛选（3 种方法）”提供可核验的论文素材。
> 核验日期：2026-08-16（UTC）。
> 代码快照：Atom-0 `b1f6276`（工作区有未提交的数据配置改动），Atom-DataBackend `b4ae65d`。
> 重要原则：下文区分“原始转换规模”“实际 train 数据量”“代码采样概率”和“筛选后规模”；这四种口径不能混写。

## 1. 可直接引用的核心结论

Atom0 的数据管线由两级标准化和一级质量控制组成。首先，不同来源的机器人数据被转换为 InfiData 中间格式，以 episode 为组织单位保存逐帧状态、动作、时间戳、任务语义、视频帧索引和来源元数据。随后，各数据源由专用转换器构建为 RLDS/TFDS；每条 TFRecord Example 对应一个 episode，内部 `steps` 保存图像、状态、动作和文本，`episode_metadata` 保存机器人、控制模式、动作语义和来源信息。训练时，各 builder 再被映射到统一的三视角输入和 80 维物理动作空间，并使用逐数据集归一化统计进行数值标准化。

按当前 A-4 配置 `cotrain_full_all_atom_aligned_rl2`，训练集合由 44 个 RLDS builder、317,831 条 train episode 组成：保留 A-3 中除 `egoverse_scale` 外的 38 个 builder，加上 4 个自采 AtomAligned builder 和 2 个 EgoVerse-RL2 builder。实际 episode 构成中，公开机器人数据占 82.58%，EgoVerse-full 与 EgoVerse-RL2 占 15.18%，自采 Piper 与 AtomAligned 占 2.24%；训练采样概率则对齐 `dev/weizhongxing` 的历史分组权重，二者不是同一口径。

数据质量筛选包含三类互补检测：S1 检测状态/动作的瞬时突变；S2 检测 State–Action 的趋势、方向和时序一致性；S3 检测远离主体分布的极端值。44 个原始配置共 355,654 条 train episode，经 P0–P5 数据集特定策略判定 31,585 条（8.88%）应删除。训练侧同时保留五个既有主动排除项，因此最终 State–Action 筛选配置包含 36 个 builder、300,589 条 episode、243,049,603 帧，约 2,415.82 小时。

## 2. 数据构成、规模与占比

### 2.1 原始转换规模

`docs/预训练数据集总览_v2.txt` 记录的全部已转换数据约为 3,033.33 小时。该数字是转换产物总量，不是最终训练量，并且包含后来明确弃用的数据。

| 来源 | 原始 episode / frame | 标称时长 | 主要形态 | 备注 |
|---|---:|---:|---|---|
| DROID | 67,499 / 19,508,979 | 361.28 h | Franka，8D state/action，15 FPS，外部+双腕视角 | 训练 split 为 64,124 episode |
| realworld Piper | 419 + 5,586 + 950 / 3,133,755 | 29.65 h | 双臂 Piper，14D，15/30 FPS | 419 条 c3/no-EEF 数据质量差，全部弃用；训练使用 Piper30 与 Piper2 |
| AgiBot World Beta | 22,986 / 35,663,824 | 330.22 h | 移动双臂，20D，30 FPS，三视角 | train 21,837 episode |
| RoboCOIN | 22 个 builder | 874.09 h | 多种双臂/人形本体，14–54D 有效动作 | 训练前主动排除 3 个 builder，另有 1 个全局排除 |
| RoboMIND-full | 13 个 builder | 295.67 h | 真机与仿真、多种本体，7–38D | 训练前全局排除 Tiankung sim s38 |
| EgoVerse-full | 63,504 / 123,380,768 | 1,142.42 h | 人类/机器人双手 EEF，12D absolute pose，30 FPS | 5 个 builder |

注意：3,033.33 小时等于上述来源的原始标称时长之和，其中含 Piper c3 和训练配置主动排除的 builder。论文若描述“用于训练的数据量”，不应直接使用这个数字。

### 2.2 训练配置演化与推荐口径

| 配置 | builder | train episode | 用途/状态 |
|---|---:|---:|---|
| `cotrain_full_all_full_norm`（A-3） | 39 | 328,540 | 全量基础配置；排除 5 个已知问题 builder |
| `cotrain_full_all_atom_aligned_rl2`（A-4） | 44 | 317,831 | A-3 去除 `egoverse_scale`，再加 4 个 AtomAligned + 2 个 EgoVerse-RL2；数据侧与双头分支对齐 |
| `cotrain_full_all_sa_filtered` | 36 | 300,589 | A-3 的 State–Action 筛选版本，不含 AtomAligned/RL2 |

论文必须先确定最终 checkpoint 对应哪个配置。若使用 A-4 checkpoint，组成表应使用 A-4；若使用 State–Action 筛选训练，则不能把 A-4 的 317,831 条 episode 与筛选后的 300,589 条混为同一次训练。

### 2.3 A-4 的来源组成

下表的“实际 episode 占比”由 TFDS `dataset_info.json` 的 train split 求和；“代码采样概率”直接从当前 `config.py` 实例化得到。

| 来源组 | builder | train episode | 实际 episode 占比 | 当前代码采样概率 |
|---|---:|---:|---:|---:|
| 自采 Piper | 2 | 5,829 | 1.83400% | 1.75270% |
| AgiBot | 1 | 21,837 | 6.87063% | 6.56608% |
| DROID | 1 | 64,124 | 20.17550% | 19.28118% |
| EgoVerse-full | 4 | 44,023 | 13.85108% | 13.23709% |
| RoboCOIN | 18 | 93,352 | 29.37158% | 28.06963% |
| RoboMIND-full | 12 | 83,152 | 26.16233% | 29.43534% |
| 自采 AtomAligned | 4 | 1,296 | 0.40776% | 0.38969% |
| EgoVerse-RL2 | 2 | 4,218 | 1.32712% | 1.26829% |
| 合计 | 44 | 317,831 | 100% | 100% |

按照论文大纲可进一步归并为：

| 论文叙述类别 | 包含来源 | train episode | 实际 episode 占比 | 当前代码采样概率 |
|---|---|---:|---:|---:|
| 公开机器人数据 | AgiBot + DROID + RoboCOIN + RoboMIND | 262,465 | 82.58005% | 83.35223% |
| Ego/人类中心数据 | EgoVerse-full + EgoVerse-RL2 | 48,241 | 15.17819% | 14.50539% |
| 自采对齐数据 | Piper + AtomAligned | 7,125 | 2.24176% | 2.14239% |

这一归并把 AtomAligned 单列为“自采对齐数据”，因为其中同时有 human 与 robot builder。若论文希望将人类 AtomAligned 并入 Ego 类别，应按四个 builder 分开计算，不能把整个 AtomAligned 归入人类视频。

还应将“采集归属”和“数据模态”视为两个独立维度。仓库代码/训练文档能够明确支持：Piper 是 in-house robot 数据，AtomAligned 是自采且同时包含 human/robot，AgiBot、DROID、RoboCOIN、RoboMIND 是训练文档所称的公开 Robot 数据；EgoVerse-full/RL2 则是 Ego/EEF 模态组。仅凭当前代码无法完整证明 EgoVerse 各子集的采集团队、再分发许可和开源条款，因此论文定稿时应另行核对 dataset card 或许可证，不能把“Ego”直接等同于“开源”或“自采”。

### 2.4 A-3/A-4 权重口径

A-4 为控制单双头消融变量，完整对齐 `dev/weizhongxing` 的历史分组缩放与重新归一化
公式。该公式使用各来源的声明规模及组内权重，因此不严格等于当前 TFDS train split
episode 占比，尤其 RoboMIND 的采样概率较高。论文应报告上表“当前代码采样概率”。

### 2.5 A-3 的精确帧数与时长

以下数字来自每个数据集的 `assets/cotrain_full_all_full_norm/<dataset_id>/norm_stats_meta.json`，比原始总览更接近实际 train 数据。时长由 DROID 15 FPS、其他来源 30 FPS 换算。

| 来源 | train episode | train frame | frame 占比 | 换算时长 |
|---|---:|---:|---:|---:|
| Piper | 5,829 | 2,757,208 | 1.0344% | 25.53 h |
| AgiBot | 21,837 | 33,914,594 | 12.7241% | 314.02 h |
| DROID | 64,124 | 18,522,518 | 6.9493% | 343.01 h |
| EgoVerse-full | 60,246 | 116,584,930 | 43.7403% | 1,079.49 h |
| RoboCOIN | 93,352 | 70,815,556 | 26.5686% | 655.70 h |
| RoboMIND-full | 83,152 | 23,943,890 | 8.9833% | 221.70 h |
| 合计 | 328,540 | 266,538,696 | 100% | 2,639.46 h |

这也说明 episode 比例、frame 比例和小时比例差别很大。例如 EgoVerse-full 仅占 A-3 episode 的 18.34%，却占 43.74% 的 frame。当前采样按配置权重抽取 frame 流，权重本身由 episode 常量导出，因此长 episode 并不会自动按其帧数获得更大概率。

### 2.6 主动排除的数据

A-3/A-4 在进入混合前主动排除 5 个 builder，共 27,114 条 train episode：

| dataset_id | 原因/代码注释 |
|---|---|
| `robocoin_unitree_g1_dex3_s28_a28` | 全局禁用 |
| `robomind_tienkung_sim_s38_a38` | 全局禁用 |
| `robocoin_leju_robot_s54_a54` | 稀疏/异常尾部使分位数归一化不安全 |
| `robocoin_agilex_decoupled_magic_s14_a14_fps50` | 同上 |
| `robocoin_agilex_decoupled_magic_s26_a26` | 同上/破坏 state conditioning |

此外，Piper c3/no-EEF 的 419 条原始 episode 未进入训练配置。

## 3. 从源数据到 InfiData

### 3.1 两级转换设计

Atom-DataBackend 将处理链拆成两步：

```text
异构上游数据
  -> scripts/convert/：源格式转 InfiData
  -> scripts/validate/：schema 与结构校验
  -> scripts/convert2openpi/：InfiData 转 RLDS/TFDS
  -> scripts/quality/：质量分析、最终决策和筛选后物化
  -> Atom-0：运行时统一动作空间、归一化、混采和训练
```

InfiData 的作用是把不同数据源中不稳定的原始目录与字段，收敛到一个可审计的 episode/frame 中间层。大体组织为：每个 episode 一个 Parquet；视频以 MP4 保存，表中使用相对路径和 frame index 引用；`meta/episodes.jsonl`、`tasks.json`、`robots.json`、`stats.json`、`segments.jsonl` 保存数据集级和 episode 级信息。

### 3.2 必需字段与保留信息

机器人逐帧 schema 的必需字段包括：

- 标识与时序：`episode_index`、连续的 `frame_index`、`timestamp`；
- 数值信号：`observation.state`、`action`，可选 `observation.qvel`、`observation.effort`、`action_relative`；
- 语义：`task`、`subtask`；
- 本体和控制：`robot_type`、`control_mode`；
- 质量与结果：1–5 的 `quality`、`speed_bin`、`mistake`、`success`；
- 来源：`source_dataset`、可选 `domain`（real/sim）；
- 视觉索引：各相机的 `video.<camera>.path` 和 `frame_index`；
- 可选未来子目标：`subgoal.real_future.<camera>.path/frame_index`。

转换器还尽量保留 state/action layout、表示类型、`action_is_delta`、相机映射、原始 episode metadata、任务与机器人元数据，供后续兼容性判断和质量审计使用。

### 3.3 InfiData 校验

`validate_robot_dataset.py` 执行以下检查：

- 对每个 Parquet 的每一行执行 JSON Schema 校验；
- episode 不得为空；`frame_index` 必须从 0 连续递增；时间戳必须单调；一个文件只能有一个 episode id；
- 同一 episode 内 state/action 维数必须恒定；
- 表中引用的视频文件必须存在；
- `segments.jsonl` 必须满足 segment schema；
- `episodes.jsonl` 的记录数必须和 episode Parquet 文件数一致。

这些检查属于格式与引用完整性验证，不等同于运动学合理性或跨模态语义质量验证，后者由 S1–S3 或尚未实现的 C1–C3 负责。

## 4. InfiData 到 RLDS/TFDS

### 4.1 RLDS 结构

各数据源使用独立 `convert_infidata_*_to_rlds.py`，但产物遵循同一组织原则：

- 一个 TFDS Example 对应一个 episode，而不是一个 frame；
- Example 中的 `steps` 是逐帧序列，包含 float32 state/action、图像、任务文本、帧索引与可选 subgoal；
- `episode_metadata` 保存 episode id、帧数、FPS、任务、本体、控制模式、成功/错误标签、来源、相机集合、state/action schema 和原始元数据；
- 图像由 `tfds.features.Image(..., encoding_format="jpeg")` 编码；
- 每种稳定的 robot/schema/camera-set 组合生成独立 builder，避免异构 tensor shape 混在同一 TFDS 配置中；
- 转换前检查视频帧索引和图像 shape；失败 episode 写入 skip log，而不是生成半损坏样本。

当前覆盖的预训练来源为 AgiBot、DROID、EgoVerse、realworld Piper、RoboCOIN 和 RoboMIND-full。

### 4.2 train/seen/unseen 切分的真实语义

通用转换器的默认参数为 unseen 5%、seen 5%、seed 0，均按 episode 抽样：

1. 从全部 episode 中随机抽约 5% 到 `unseen_test`，这些 episode 从 train 移除；
2. 从剩余 train pool 中抽约 5% 到 `seen_test`；
3. 代码明确让这些 seen episode **继续保留在 train 中**。

因此，对于通用转换器生成的 builder：

- `unseen_test` 与 train 是 episode-disjoint；
- `seen_test` 是 train 的一个重复子集，不是 held-out trajectory；
- “seen validation”只能被描述为训练任务/训练轨迹上的确定性诊断，不能被描述为无泄漏的 held-out validation。

Piper30 v1.1.0 是例外。它使用专门的 repartition 脚本，将两个指定任务的所有 episode 独占分到 `unseen_test`，并从其余任务中按任务分别抽约 5% episode 独占分到 `seen_test`，剩余进入 train。因此 Piper30 的 seen/unseen 与 train 均不重叠，且 unseen 具备 task-disjoint 语义。

这一差异建议在论文评估协议中明确披露。

## 5. Atom-0 训练前处理

### 5.1 运行时统一 schema

每个数据集由 `restructure_name` 选择专用映射函数，最后统一输出：

```text
state
actions
image {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}
image_mask {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}
prompt
prompt_prefix（可选，标识 joint/eef action mode 与 EEF frame）
dataset_id
```

缺失视角使用空白图像占位，同时将对应 `image_mask` 设为 false。例如只有头戴相机的 EgoVerse/MECKA 数据不会伪装成存在腕部相机。

图像在混合前以保持宽高比的 `resize_with_pad` 统一到 224×224。编码后的 JPEG 字节先进入 shuffle buffer，混合和 shuffle 后才解码，避免将大量 raw uint8 图像放入内存。

### 5.2 统一 80D 物理动作空间

所有生产配置在混采前把 state/action scatter 到固定的 80 维物理槽位。未使用槽位置 0，`action_mask=0`，损失只监督实际映射槽位。

| 统一槽位（1-based） | 语义 | 最终时序表示 |
|---|---|---|
| U1–U7 | 左臂关节 1–7 | 相对当前 state |
| U8–U10 | 左 EEF xyz | absolute |
| U11–U13 | 左 EEF Euler yaw/pitch/roll | absolute |
| U14–U16 | 保留 | mask=0 |
| U17 | 左平行夹爪 | absolute |
| U18–U29 | 左灵巧手 1–12 | absolute |
| U30–U36 | 右臂关节 1–7 | 相对当前 state |
| U37–U39 | 右 EEF xyz | absolute |
| U40–U42 | 右 EEF Euler yaw/pitch/roll | absolute |
| U43–U45 | 保留 | mask=0 |
| U46 | 右平行夹爪 | absolute |
| U47–U58 | 右灵巧手 1–12 | absolute |
| U59–U64 | 左腿 1–6 | absolute |
| U65–U70 | 右腿 1–6 | absolute |
| U71–U72 | 头部 1–2 | absolute |
| U73–U74 | 腰部 1–2 | absolute |
| U75 | 其他单一躯干关节 | absolute |
| U76–U80 | 保留 | mask=0 |

单臂机器人统一映射到右臂/右夹爪槽位。对 absolute joint target，手臂动作转换为 `q_target - q_t`；夹爪、灵巧手、腿、头、腰保持 absolute。EgoVerse 和 AtomAligned 的 EEF pose 保持数据集坐标系下的 absolute `xyz + Euler`，不做 SE(3) 差分、不转 rotation-6D，也不额外变换到相机坐标系。

特殊处理包括：

- RoboCOIN 中语义不可靠的 EEF 字段不参与训练；
- Leju 的 14D arm velocity tail 被丢弃；
- Galaxea 18D 的无名尾部 2D、unknown 30D 的无名尾部 2D 被丢弃；
- Yinhe state 使用源维度 6–21（1-based）重排以对齐 16D action；
- 80D mapping 的 fingerprint 与 norm stats 一起保存，加载时强校验，避免用错动作语义。

### 5.3 动作时间窗与混采

当前 π0.5 统一配置使用 50-step action horizon。对 episode 的每个时刻 `t` 构造 `actions[t:t+50]`；靠近 episode 末尾时，越界位置重复最后一个 action。各数据集 train 流先 `repeat()`，再通过 `sample_from_datasets(..., weights)` 按配置概率混成无限流，随后执行全局 shuffle、图像解码和 batch。

因此配置权重是“每次从哪个数据流抽取一个 frame sample 的概率”，不是原始存储字节占比，也不是自动按 frame 总量得到的自然比例。

多机训练时，TFDS split 按 episode 用 `tfds.even_splits` 分给各 JAX process，避免不同主机读取同一 episode 切片。

## 6. 归一化统计（norm stats）

### 6.1 统计对象和顺序

归一化按 dataset 独立计算，而不是对混合数据计算一套全局统计。顺序为：

```text
读取 train split
-> 只提取 state/action（不解码图像）
-> 源维选择/重排
-> 映射到 Unified80
-> 构造 50-step action chunk
-> 对手臂 absolute target 做 delta
-> RunningStats
```

state 每个 frame 统计一次；action chunk 被展平后统计，即每个 frame 对应的 50 个未来 target 都进入 action 分布，episode 末尾重复的最后 action 也会重复计数。因此这里的 action norm 是“训练 action-chunk 目标分布”的统计，不是简单的原始逐帧 action 分布。

正式 full-norm 脚本设置 `repeat=False`、`shuffle=False`、`drop_remainder=False`，完整扫描每个 active dataset 的 train split。每个数据集输出：

```text
norm_stats.json
norm_stats_meta.json
unified_action_space.json
```

### 6.2 统计量与训练时公式

`RunningStats` 在线累计：

- mean；
- std（由 `E[x^2] - E[x]^2` 得到）；
- q01、q99（使用每维 5,000-bin 动态直方图近似）。

π0.5 配置使用分位数归一化。对 state 和 action 的每一维：

```text
x_norm = 2 * (x - q01) / (q99 - q01 + 1e-6) - 1
```

归一化由 `dataset_id` 在运行时分派，因此每种机器人本体保留自己的尺度。未使用的 Unified80 槽位被显式设为中性统计：mean=0、std=1、q01=-1、q99=1，保证零填充在归一化后仍接近 0。

筛选后的数据不能复用原始数据的 norm，因为 episode/帧分布已经变化；代码为 `cotrain_full_all_sa_filtered` 单独完整重算了 36 套统计。

### 6.3 当前资产完整性

- A-3：39/39 个 active dataset 有 `norm_stats_meta.json`；合计扫描 266,538,696 帧。
- SA-filtered：36/36 个 active dataset 有 `norm_stats_meta.json`；合计扫描 243,049,603 帧。
- A-4：44/44 个 active dataset 均有与当前映射一致的 norm 和映射元数据；合计扫描
  253,163,923 帧。其中 34 份非 Ego norm 与 A-3 字节级一致，另外 10 份数据侧资产
  与 `dev/weizhongxing` 字节级一致。

## 7. 数据质量筛选：S1、S2、S3

质量脚本只读取 RLDS 的 state、action 和 episode metadata，不读取相机张量，也不原地修改 TFRecord。它先做全 split 的阈值校准，再逐 episode 生成可审计证据。

### 7.1 S1：Sudden Change Detection

目标是检测传感、传输、碰撞或转换引起的瞬时跳变。

1. 每个 episode、每个 state/action 维先做 kernel=5 的 median filter，再做 window=9、polyorder=2 的 Savitzky–Golay 平滑。
2. 计算三类指标：原始值与平滑值的 absolute residual、中心二阶差分 acceleration、中心三阶差分 jerk。
3. 在整个 split 上逐维拟合鲁棒基线：center 为 median，scale 为 `1.4826 * MAD`。
4. scale 下限为 `max(1e-6, 0.002 * (q99-q01))`，避免分段常数信号的 MAD 退化为 0。
5. residual、acceleration、jerk 均使用单侧阈值 `center + 8 * scale`。
6. 仅当 `residual 异常 AND (acceleration 异常 OR jerk 异常)` 时命中突变。
7. acceleration 的首尾帧、jerk 的前后两帧因差分邻域不完整而不参与阈值拟合和检测。
8. gripper 跳过幅值突变规则；NaN/Inf 在任何维度都直接视为异常。

S1 原始输出是异常帧/人工复核候选，是否升级为 episode 删除由 P0–P5 决策。

### 7.2 S2：State–Action Trend Alignment

目标是检测 state/action 错配、时钟错位、动作丢包、方向不一致和反因果时序。

1. 先检查 state/action 维度、layout 和物理表示是否可比；不兼容则记录 `skipped`，不强行比较。
2. absolute action 直接平滑；明确标记的 delta action 先沿时间积分成近似绝对轨迹。缺失、`unknown` 或非法的 `action_is_delta` 按 absolute 处理，同时保留推断来源。
3. 对平滑后的 state/action 做一阶差分，在 `[-10, +10]` frame 搜索 cross-correlation 最大的 lag；正 lag 表示 action 领先 state。
4. 只在有效运动样本上比较方向。绝对差分不超过 `1e-5` 视为静止，否则取 sign；Directional Agreement（DA）为两侧方向值相同的比例。
5. 至少需要 10 个有效运动样本。DA < 0.6，或最佳 lag < 0（state 领先 action），则该维命中。
6. gripper 不参与连续轨迹的 correlation/DA。

S2 命中表示整条轨迹可能错配或反因果，默认建议删除 episode。若 action 本来就是由 next state 构造，DA 天然接近 1；此时 S2 只能验证转换后的时序关系，不是独立传感器交叉验证。

### 7.3 S3：Extreme Value Filtering

目标是排除会污染训练和 q01/q99 归一化的严重长尾值。

1. 在全 split 上逐维估计 state/action 的 q01、q99；非有限值先排除出阈值拟合。
2. 默认 `alpha=1.5`，允许范围为：

```text
[q01 - 1.5*(q99-q01), q99 + 1.5*(q99-q01)]
```

3. 任一非 gripper 维越界即标记该帧；任意维的 NaN/Inf 均标记。
4. gripper 因开/关双峰分布跳过数值极值检查，但不豁免 NaN/Inf。
5. 每类信号最多用 1,000,000 行校准；使用固定容量 priority reservoir 在整个 split 上等概率无放回抽样，默认内存上限 2,048 MiB。
6. 完全没有有限样本的维度记为 `invalid_threshold_dimension`，不伪造阈值。

### 7.4 数据集特定的 P0–P5 最终策略

不能直接把 `flagged_any` 当作删除清单。最终 episode 决策为：

| 策略 | 数据集 | episode 删除条件 |
|---|---|---|
| P0 | AgiBot、Piper2、Piper30 | S1、S2、S3 的去重并集 |
| P1 | DROID | S1 只看 state；加 S2、S3 |
| P2 | EgoVerse-full、RoboCOIN | 禁用 S1；只用 S2、S3 |
| P3 | RoboMIND master–puppet | S1 只看 state；禁用 S2；保留 S3 |
| P4 | RoboMIND Franka sim | S1 至少命中 3 个不同帧且命中率至少 1%；加 S2、S3 |
| P5 | RoboMIND Tiankung s38 | S1/S2 只看双臂维 0–6、19–25；S2 还要求 active samples >= 20；保留 S3 |

这些策略反映不同数据源的信号生成方式。例如 master–puppet action 与 state 并非普通跟踪目标，不适合直接使用通用 S2；EgoVerse/RoboCOIN 的 S1 则因源表示特性被禁用。

### 7.5 筛选规模与物化

全量质量决策口径：44 个配置、355,654 条 train episode；P0–P5 删除 31,585（8.88%），保留 324,069（91.12%）。训练配置还预先排除了第 2.6 节的 5 个 builder（27,114 条），两类删除存在 3,634 条重叠。因此最终相对于 355,654 条原始 episode 的去重删除量为 55,065（15.48%），保留 300,589（84.52%）。

| 阶段 | episode |
|---|---:|
| 44 个原始配置 | 355,654 |
| 既有 5 个主动排除项 | -27,114 |
| active 39 个 builder 内 S1–S3/P0–P5 额外删除 | -27,951 |
| 最终 36 个非空 builder | 300,589 |

筛选后的训练数据共 243,049,603 帧；按 DROID 15 FPS、其余 30 FPS 估算约 2,415.82 小时。相较 A-3 的 328,540 episode / 266,538,696 frame，episode 保留率 91.49%，frame 保留率 91.19%。

物化阶段只重写 train TFRecord：保留 episode 的 serialized Example 原样复制，不解码或重编码 JPEG；其他 split 与 metadata 逐字节复制；最后更新 `dataset_info.json` 的 train shard lengths 和字节数。原始 RLDS 保持只读，输出到独立目录。train 为空的 3 个 RoboCOIN builder 不进入训练清单：

- `robocoin_unitree_g1_s28_a28_high`；
- `robocoin_unitree_g1_s28_a28`；
- `robocoin_unknown_s30_a30_high`。

### 7.6 筛选后来源占比

| 来源 | 保留 episode | episode 权重/采样概率 | 保留 frame | frame 占比 | 换算时长 |
|---|---:|---:|---:|---:|---:|
| Piper | 5,650 | 1.87964% | 2,640,072 | 1.0862% | 24.45 h |
| AgiBot | 17,796 | 5.92038% | 27,595,338 | 11.3538% | 255.51 h |
| DROID | 62,649 | 20.84208% | 17,859,469 | 7.3481% | 330.73 h |
| EgoVerse-full | 59,088 | 19.65741% | 115,195,385 | 47.3958% | 1,066.62 h |
| RoboCOIN | 81,582 | 27.14071% | 58,374,792 | 24.0176% | 540.51 h |
| RoboMIND-full | 73,824 | 24.55978% | 21,384,547 | 8.7984% | 198.01 h |
| 合计 | 300,589 | 100% | 243,049,603 | 100% | 2,415.82 h |

与 A-3/A-4 不同，`cotrain_full_all_sa_filtered` 的代码权重明确等于 `kept_train_episodes / 300589`，并有总数、dataset id 与权重和的启动前强校验。

## 8. 不能在当前论文中声称已经实现的内容

`docs/Qwen数据筛选方案.txt` 还描述了 C1–C3 跨模态检查：

- C1：视频与语言指令一致性；
- C2：URDF/state 渲染与视频机器人 mask 一致性；
- C3：视频清晰度、遮挡、黑屏、冻结和静止冗余检测。

但当前仓库的正式质量代码只实现和物化了 S1 + S2 + S3。C1–C3 只有方案说明，没有与当前 44 个 RLDS 配置对应的可执行实现和验收结果。S4（Joint–EEF FK 一致性）与 S5（坐标系/朝向统一）也明确未复现。因此论文方法部分可以将 C1–C3 写为未来工作或候选方案，不能写成已经用于 Atom0 预训练数据筛选的步骤。

## 9. 建议用于论文的方法段落（初稿）

我们构建了一个面向异构机器人与第一视角操作数据的两阶段数据标准化流水线。首先，将各数据源转换为统一的 InfiData episode 表示，其中逐帧记录本体状态、控制目标、时间戳、语言任务以及相机视频索引，同时保留机器人类型、控制模式和动作语义等来源元数据。随后，将每个 episode 序列化为 RLDS/TFDS Example，并在训练加载阶段把不同本体的状态和动作映射至固定的 80 维物理槽位。未使用槽位以零填充并由二值动作掩码从训练损失中排除；手臂关节的绝对目标转为相对当前状态的增量，而夹爪、灵巧手、躯干关节和 EEF 位姿保留绝对表示。所有视觉输入被整理为一个基础视角和两个腕部视角，缺失视角由显式 mask 屏蔽。

为减小不同机器人量纲和控制范围的差异，我们在每个数据集的完整训练 split 上分别统计状态与动作的 1%/99% 分位数，并使用分位数线性变换将其映射到近似 [-1, 1]。在统计之前执行与训练一致的动作空间映射、动作时间窗构造和关节增量转换，从而保证归一化分布与实际监督目标一致。

我们进一步采用三类 State–Action 质量检查。S1 通过中值滤波、Savitzky–Golay 平滑以及 residual/acceleration/jerk 的联合鲁棒阈值识别瞬时跳变；S2 通过跨相关估计 state/action 时延，并以方向一致率检测错配和反因果轨迹；S3 基于每维 q01/q99 的扩展区间过滤严重长尾值。考虑不同来源的动作生成机制，我们使用数据集特定的 P0–P5 决策策略将帧级异常证据转为 episode 级删除清单，并将保留的 serialized RLDS Example 无损复制到独立数据根目录。

## 10. 定稿前必须确认的事项

1. 最终论文 checkpoint 对应 A-3、A-4 还是 SA-filtered；三个配置的组成不同。
2. A-4 与双头实验的数据侧已经对齐；比较时确认两者只改变单头/双头模型架构。
3. A-4 对齐双头分支的历史分组权重，不应描述成严格 episode-proportional 权重。
4. 评估部分不能把通用 builder 的 `seen_test` 写成 held-out，因为它与 train 重叠；只有 Piper30 v1.1 采用独占 seen/unseen 重划分。
5. “约 3,033.33 小时”是原始转换总量且包含弃用数据；A-3 实际 train 约 2,639.46 小时，SA-filtered 约 2,415.82 小时；A-4 的精确有效帧数为 253,163,923。
6. 不要声称 C1/C2/C3、S4/S5 已经用于正式筛选。

## 11. 主要代码证据索引

| 主题 | 文件 |
|---|---|
| 论文数据章节大纲 | `docs/Atom0_paper.txt` |
| A-3/A-4 数据集、episode 常量、权重和排除项 | `src/openpi/cotrain/config.py:275`、`:497`、`:881`、`:942`、`:979` |
| SA-filtered 逐数据集保留数与权重 | `src/openpi/cotrain/config.py:1012` |
| 加权混采、action chunk、图像 decode/resize | `src/openpi/cotrain/rlds_dataset.py:645` |
| 三相机标准 schema 与按 dataset 归一化 | `src/openpi/cotrain/transforms.py` |
| Unified80 槽位、mapping、mask 与 fingerprint | `src/openpi/cotrain/action_space.py` |
| full norm 统计流程 | `scripts/compute_cotrain_full_norm_stats_light.py`、`scripts/compute_cotrain_norm_stats_light.py` |
| mean/std/q01/q99 的在线统计 | `src/openpi/shared/normalize.py` |
| InfiData 行级 schema | `../../Atom-DataBackend/schemas/robot_episode.schema.json` |
| InfiData 结构校验 | `../../Atom-DataBackend/scripts/validate/validate_robot_dataset.py` |
| 各来源 RLDS 转换 | `../../Atom-DataBackend/scripts/convert2openpi/convert_infidata_*_to_rlds.py` |
| 通用 split 实现 | 例如 `../../Atom-DataBackend/scripts/convert2openpi/convert_infidata_droid_to_rlds.py:169` |
| Piper30 task-disjoint 重划分 | `../../Atom-DataBackend/scripts/convert2openpi/repartition_realworld_piper_rlds.py` |
| S1/S2/S3 实现 | `../../Atom-DataBackend/scripts/quality/filter_rlds_state_action.py` |
| P0–P5 最终决策与强校验 | `../../Atom-DataBackend/scripts/quality/build_final_rlds_decisions.py`、`select_robomind_quality_candidates.py` |
| 筛选后 TFRecord 物化 | `../../Atom-DataBackend/scripts/quality/materialize_filtered_rlds.py` |
| 原始规模汇总 | `docs/预训练数据集总览_v2.txt` |

## 附录 A：A-4 builder 与采样概率

A-4 的 44 个 builder 及权重以 `src/openpi/cotrain/config.py` 的
`_FULL_ALL_ATOM_ALIGNED_RL2_DATA` 为事实源；`_FULL_ALL_ATOM_ALIGNED_RL2_TRAIN_EPISODES_BY_ID`
只记录当前磁盘 train episode 数，不直接等于训练采样权重。来源级汇总见第 2.3 节。

## 附录 B：A-3 到 SA-filtered 的逐 builder 保留率

| 来源 | dataset_id | 筛选前 ep | 筛选后 ep | 保留率 |
|---|---|---:|---:|---:|
| Piper | `piper30` | 4,927 | 4,751 | 96.428% |
| Piper | `piper2` | 902 | 899 | 99.667% |
| AgiBot | `agibot` | 21,837 | 17,796 | 81.495% |
| DROID | `droid` | 64,124 | 62,649 | 97.700% |
| EgoVerse | `egoverse_aria` | 910 | 910 | 100.000% |
| EgoVerse | `egoverse_eva` | 2,813 | 2,593 | 92.179% |
| EgoVerse | `egoverse_human` | 770 | 769 | 99.870% |
| EgoVerse | `egoverse_mecka` | 39,530 | 39,482 | 99.879% |
| EgoVerse | `egoverse_scale` | 16,223 | 15,334 | 94.520% |
| RoboCOIN | `robocoin_agilex_cobot_magic_s26_a26` | 7,870 | 7,071 | 89.848% |
| RoboCOIN | `robocoin_airbot_mmk2_s36_a36` | 10,005 | 9,932 | 99.270% |
| RoboCOIN | `robocoin_galaxea_r1_lite_upper_s14_a14` | 1,337 | 1,293 | 96.709% |
| RoboCOIN | `robocoin_realman_rmc_aida_l_s28_a28` | 658 | 655 | 99.544% |
| RoboCOIN | `robocoin_agilex_decoupled_magic_s14_a14_fps30` | 7,389 | 7,353 | 99.513% |
| RoboCOIN | `robocoin_aloha_s26_a26` | 4,634 | 3,121 | 67.350% |
| RoboCOIN | `robocoin_alpha_bot_2_s28_a28` | 814 | 814 | 100.000% |
| RoboCOIN | `robocoin_discover_aitbot_mmk2_s36_a36` | 5,460 | 5,435 | 99.542% |
| RoboCOIN | `robocoin_galaxea_r1_lite_s14_a14` | 4,886 | 4,760 | 97.421% |
| RoboCOIN | `robocoin_galaxea_r1_lite_s16_a18` | 922 | 922 | 100.000% |
| RoboCOIN | `robocoin_leju_robot_s118_a54` | 17,002 | 10,454 | 61.487% |
| RoboCOIN | `robocoin_realman_rmc_aidal_s28_a28` | 17,481 | 17,471 | 99.943% |
| RoboCOIN | `robocoin_ruantong_a2d_s17_a17` | 1,633 | 1,475 | 90.325% |
| RoboCOIN | `robocoin_ruantong_a2d_s41_a34` | 6,136 | 6,126 | 99.837% |
| RoboCOIN | `robocoin_unitree_g1_s28_a28_high` | 216 | 0 | 0% |
| RoboCOIN | `robocoin_unitree_g1_s28_a28` | 884 | 0 | 0% |
| RoboCOIN | `robocoin_unknown_s30_a30_high` | 846 | 0 | 0% |
| RoboCOIN | `robocoin_yinhe_s49_a16` | 5,179 | 4,700 | 90.751% |
| RoboMIND | `robomind_agilex_cobot_magic_s14_a14` | 9,855 | 7,805 | 79.198% |
| RoboMIND | `robomind_franka_fr3_dual_s16_a16` | 1,685 | 1,585 | 94.065% |
| RoboMIND | `robomind_franka_panda_s8_a8` | 14,956 | 12,042 | 80.516% |
| RoboMIND | `robomind_franka_sim_franka_s8_a8` | 8,445 | 7,147 | 84.630% |
| RoboMIND | `robomind_franka_sim_simulation_s8_a8` | 8,662 | 6,230 | 71.923% |
| RoboMIND | `robomind_franka_sim_simulation_no_front_s8_a8` | 150 | 123 | 82.000% |
| RoboMIND | `robomind_franka_sim_none_s8_a8` | 211 | 182 | 86.256% |
| RoboMIND | `robomind_tienkung_gello_s16_a16` | 5,402 | 5,353 | 99.093% |
| RoboMIND | `robomind_tienkung_prod1_gello_s16_a16` | 2,811 | 2,678 | 95.269% |
| RoboMIND | `robomind_tienkung_xsens_s14_a14` | 5,775 | 5,526 | 95.688% |
| RoboMIND | `robomind_tienkung_real_s38_a38` | 139 | 113 | 81.295% |
| RoboMIND | `robomind_ur5e_s7_a7` | 25,061 | 25,040 | 99.916% |
