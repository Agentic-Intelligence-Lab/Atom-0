# 真机 + Ego 联合训练（Cotrain）框架技术文档

> 适用代码库：`pi07_reproduction`（openpi fork）
> 框架路径：`src/openpi/cotrain/`
> 最后更新：2026-06-20

本文档梳理 cotrain 框架新增功能的技术细节：多数据集训练、动作空间处理、ego 数据引入、以及一系列技术决定。供开发与交接参考。

---

## 0. 总体设计原则

**不修改 openpi 任何原文件，所有新代码自包含在 `src/openpi/cotrain/`**（fork-and-extend）。复用 openpi 的 `train_step`、sharding、transform、checkpoint 机制，只在新路径里加多数据集 + ego + 验证逻辑。

涉及文件：

```
src/openpi/cotrain/
  rlds_dataset.py    # 数据集定义 + TF 输入管线 + restructure 注册表
  transforms.py      # 标准化输入 + 按数据集分派的 delta/归一化
  data_loader.py     # loader 构建 + 多机 + val loaders
  eval.py            # flow loss / action MSE / 轨迹可视化
  config.py          # 配置工厂 + 配置注册表
  weight_loaders.py  # 本地 PaliGemma 权重加载
scripts/
  train_cotrain.py               # 训练入口（fork train.py + eval + 多机初始化）
  compute_cotrain_norm_stats.py  # 逐数据集归一化统计
  count_cotrain_frames.py        # 精确帧数统计
RLDS/inspect_cotrain_datasets.py # 数据集动作语义实测脚本
```

当前已接入 6 个数据集：`robomind`、`piper15`、`piper30`、`robocoin`、`egoverse_eva`、`egoverse_mecka`。

---

## 1. 多数据集加权混采

**核心机制**：复用 RLDS 的 `dl.DLataset.sample_from_datasets(datasets, weights)`——每个数据集独立无限 `repeat`，按权重作为采样概率混合成一个流。

**关键设计**：

- **采样量由权重决定，而非数据量**。某数据集被抽到的帧数 = `batch_size × num_train_steps × weight_i`。这让我们能精确控制每个集的"有效 epoch"，而非被大集自然主导。
- **权重规则（`cotrain_all_2ep`）**：`weight_i = 目标帧数_i / Σ目标帧数`，真机各 2 epoch、EgoVerse 等量化（见 config 的 `_TWO_EPOCH_WEIGHTS`）。
- **数据流水线顺序**（`CotrainRldsDataset.__init__`）：
  ```
  per-dataset: from_rlds → restructure → pad到40 → chunk → flatten
  → sample_from_datasets(加权混合) → shuffle → decode+resize → batch
  ```
- **避免 OOM 的关键决定**：图像在 shuffle buffer **之后**才 decode（buffer 里存的是编码的 JPEG 字节，不是 raw 像素），否则 25 万 buffer × 2.7MB ≈ 675GB 直接爆显存/内存。

---

## 2. 数据 schema 统一：运行时 restructure 注册表

6 个数据集原始 schema 各异，我们**不离线重存**，而是**运行时映射**。

- `STD_RESTRUCTURE_FNS` 注册表：`restructure_name → fn(traj, dataset_id)`，把各自原始字段映射到统一嵌套 schema：
  ```
  actions / state / image{base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}
  / image_mask{...} / prompt / dataset_id
  ```
- 已注册：`robomind`、`three_cam_task`(piper×2/robocoin)、`egoverse_eva`、`egoverse_mecka`、`standardized`、`droid`。
- **加新数据集 = 写一个 restructure + 配一条 `CotrainRLDSDataset`**，无需改核心代码。

**两个工程坑的解决**（`CotrainRLDSDataset` 新增字段）：

- `builder_dir`：直接指向 TFDS version 目录，用 `tfds.builder_from_directory` 加载——解决 6 个数据集**分布在不同父目录**的问题（绕开单一全局 data_dir）。
- `dataset_id` / `uid`：唯一逻辑 id——解决 eva/mecka 共用 tfds 名 `ego_verse_infidata`、piper15/30 共用 `realworld_piper_infidata` 的冲突。**全链路（归一化路径、delta、val/weight/action_dim dict、restructure 注入）都按 `uid` 索引**。

---

## 3. 动作空间处理：不对齐，而是 native + pad + per-dataset 归一化

这是最重要的技术决定，且经过纠错。

**决定：pi0/pi05 训练时不对齐不同数据集的动作空间。** 代码验证：DROID 8 维、Libero 7 维、Aloha 14 维各保留 native 维放在前面、零 pad 到统一宽度。模型靠**观测/本体条件 + per-dataset 归一化 + flow matching** 区分输出维度，padding ≠ alignment。

所以我们的三种动作空间（joint 14/36、cartesian EE 12）**各自保留**，不做 FK / 不做对齐：

- **统一宽度 `action_dim=40`**：因 RoboCOIN 是 36 维（> 默认 32）。pi05 里 action_dim 只约束 `action_in_proj` / `action_out_proj` 两个小层（state 走离散 token，不过 state_proj），从 PaliGemma 起训这两层本就随机初始化 → 加宽零代价。
- **per-dataset 归一化**（`DispatchNormalize`）：按 `dataset_id` 取该数据集自己的分位数统计。这是异构混合的**第一个硬需求**。

---

## 4. 异构 batching 的两层统一

`sample_from_datasets` 混合后 `.batch()` 要求样本同构，但各数据集动作维、图像分辨率不同。解决方案是在**混合 / batch 之前**在 TF 管线统一：

**(a) 动作 / 状态维度** → pad 到 40（`_pad_state_actions`，只在 train/val 传 `pad_action_dim`；norm-stats 路径不 pad、保持 native）。

**(b) 图像分辨率** → `tf.image.resize_with_pad` 到 224×224（mecka 360×640 vs 其余 480×640）。缩到的就是模型输入尺寸，之后模型的 `ResizeImages` 变恒等操作。

**配套：归一化统计的动态补宽**（`DispatchNormalize._pad_stats_to_data`）：norm stats 在 **native 维**上计算保存（避免 padding 维 std=0 导致除零），应用时按数据实际宽度（40）动态补到 40——padding 维填中性值（mean 0 / std 1 / q01 -1 / q99 1），归一化后为 ~0。**好处：已算的 native 统计永不需重算**。

---

## 5. Delta 转换：option A，按数据集分派

**经实测确认所有数据集存的都是绝对值**（`inspect_cotrain_datasets.py` 抽样拟合 `action[t]` vs `state[t]` / `state[t+1]` / delta）。结论：

- robocoin、piper15、EgoVerse：绝对当前态（`action[t] == state[t]`）
- piper30：next-step 绝对（`action[t] == state[t+1]`）
- robomind：绝对 joint（夹爪维 6/13 是命令值 vs 测量值，差异较大）

是否转 delta 是**建模选择**，与 pi05 无关（pi05 的 flow matching 对动作是绝对还是 delta 无假设）。

**决定 option A**（与 pi05_base 惯例一致）：

- joint 数据集：臂维做 delta、夹爪 / 灵巧手维绝对。
  - robomind / piper×2：mask `(6, -1, 6, -1)`
  - robocoin（36 = 臂6+6 / 手12+12）：mask `(6, 6, -12, -12)`
- EgoVerse（cartesian EE）：保持绝对（mask None）——绝对笛卡尔做 delta 不稳定，且规避欧拉角 ±π wrap 问题。

**实现**（`DispatchDeltaActions`）：按 `dataset_id` 取该数据集的 mask，调 openpi 的 `DeltaActions`。**关键性质**：`DeltaActions` 按 mask 长度切片（`actions[..., :len(mask)]`），所以 native 长度的 mask 作用在 pad 到 40 的数据上只改前 N 维、padding 维不动——无需为 padding 改 mask。

---

## 6. Ego 数据的引入

**两个 EgoVerse 数据集，性质不同，务必区分**：

- **eva**（`egoverse_eva`）：双臂机器人**遥操**数据，带左右腕相机、cmd_joints / cmd_ee_pose。
- **mecka**（`egoverse_mecka`）：**真正的人类第一视角 ego**，只有头戴 front_1 相机，带 hand keypoints / head_pose。最大（~22M 帧）。

**技术处理**：

- **动作表示 = 绝对笛卡尔双手 EEF 位姿（12 维）**。人手没有统一关节定义，cartesian 是 ego 的自然表示（不是为了对齐机器人）。layout：`左 xyz + 欧拉角(yaw/pitch/roll)，右 xyz + 欧拉角`。
- **相机缺失处理**（`_egoverse_mecka_restructure`）：mecka 只有 1 路相机，两个 wrist 槽填 blank JPEG 并设 `image_mask=False`，让模型不 attend 不存在的相机。
- **监督信号**：实测 `action[t] == actions_cartesian[t,0]`。生产 Stage 1 使用
  RLDS 已对齐的 `actions_cartesian[T,100,12]`，在完整物理时间窗上均匀重采样为
  `[T,50,12]`；不再用相邻 episode 帧 `action[t:t+H]` 拼接，因为 moving head frame
  下相邻帧 pose 可能不在同一坐标系。
- **Scale 子集**：当前 BOS 版本存在极端 pose tails，已从 `egoscale_stage1_ego` 暂时
  排除；保留 mapping 和旧混合配置，等待重处理数据后重新审计。
- **prompt 字段**：EgoVerse 用 `prompt`，真机用 `task`（restructure 里分别取）。

**设计定位（关键，尚未实现）**：EgoVerse 的真正杠杆是 **domain anchor**（ego 与真机共享任务 / 场景），不是动作空间对齐。原文显示增益（+30%）只在有锚点时出现。当前是把 ego 当通用数据混入，**anchor 对齐 + 有效性消融（real-only / +ego / +anchor）是下一阶段的核心工作，目前尚未开展**。

---

## 7. Token 预算：max_token_len 256

pi05 把 state **离散化后拼进 prompt 文本**（`"Task: <prompt>, State: n n n ...;\nAction: "`，见 `tokenizer.py`）。我们 pad 到 40 维后 state 串有 40 个数字 → 加任务文本超过默认 200 被截断（丢 state 尾维 + `Action:` 标记，对 robocoin 这种高维数据集会丢真实信息）。

**决定 `max_token_len=256`**（在 `cotrain_all` 模型 config 显式传值；pi05 的 `__post_init__` 只在该值为 `None` 时才强制设 200，显式传不会被覆盖）。

> 更干净的做法是只 tokenize native 维的 state（不含 padding），但需在 cotrain 里覆盖 openpi 的 `ModelTransformFactory`/tokenizer，工作量更大，暂未做。`max_token_len=256` 已消除信息丢失。

---

## 8. 验证设计：seen / unseen 双 val

- **划分在数据侧预 bake**：每个数据集自带 `train / seen_test / unseen_test` 三 split，我们不在训练时切。`val_splits={"seen":"seen_test","unseen":"unseen_test"}`。划分粒度是 episode（无帧级泄漏）。
- **seen = 已见任务的 held-out，unseen = 新任务 / 场景**，分开报以观察过拟合 vs 泛化。
- **指标**（`eval.py`）：
  - **flow loss**：fixed-seed（单次、低方差可比）+ multi-sample（K 次平均、低方差估计）两种模式。
  - **action MSE**：跑完整 `sample_actions` 采样器，与 GT chunk 比，按 native valid_dims 做 mask（只在有效维上算）。
  - **预测 vs GT 轨迹图**：matplotlib 逐维绘制，上传 wandb。
  - 按 `val/{label}/{dataset}/...` 上报，并按训练权重聚合 `val/{label}/agg/...`。
- val loaders 单数据集、有限、确定性（固定 seed、不 shuffle）→ 跨 checkpoint 可比。
- **注意**：eval 较重（12 个 loader × multi-sample × 采样 + 首次 JIT 编译）；验证期可用轻量配置（`--val-flow-loss-mode fixed_seed --no-viz-action-traj --eval-interval 5000`）。

---

## 9. 权重加载

`LocalPaliGemmaWeightLoader`：直接从本地 `pt_224.npz` 读 PaliGemma 主干（`open()` 跟随软链、不走 GCS），action expert 随机初始化。规避了 raw PaliGemma bucket 的匿名访问 401。

> 两种起训方式：从 PaliGemma 裸权重（动作头随机，cotrain 预训练的自然选择，action_dim=40 无痛）；或从 pi05_base checkpoint（动作头会因 32→40 shape mismatch 重初始化，主干保留——未测试）。

---

## 10. 多机训练（DLC 16 卡）

openpi 的数据 / checkpoint 本就多机就绪（`make_array_from_process_local_data`、orbax 协同、process 0 写 wandb），唯一障碍是 `RLDSDataLoader` 一句 `process_count > 1` 的 guard。改动：

- `train_cotrain._maybe_init_jax_distributed()`：从 DLC 环境变量（`WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT`）初始化 JAX 多机。
- `CotrainRLDSDataLoader`：复刻 openpi loader 去掉 guard（其 `__iter__` 本就用对了多机原语）。
- 每进程 `local_batch_size = batch_size / process_count` + `tfds.even_splits` 读不相交数据切片。
- 非 0 进程禁用 wandb（避免重复 run）；`save_state` 所有进程都调（orbax 多机协同）。
- 推荐 `fsdp_devices=8`（HYBRID_SHARD：节点内 8 卡分片 + 跨节点 2 路数据并行，规避 H800 弱 NVLink）。

**待验证**：DLC 注入的 `WORLD_SIZE/RANK` 必须是**节点级**（=2 和 0/1），不是 GPU 级（16 和 0..15）。

---

## 关键技术决定一览

| 决定 | 选择 | 理由 |
|---|---|---|
| 代码组织 | 不改 openpi，fork 到 cotrain/ | 用户硬约束 |
| 动作空间 | 不对齐，native + pad + per-dataset 归一化 | pi0/pi05 本就如此 |
| action_dim | 40 | 容纳 robocoin 36 维 |
| delta | option A（joint 臂 delta、ego 绝对） | 对齐 pi05_base 惯例 + 笛卡尔/欧拉角稳定性 |
| 异构 batch | TF 内 pad40 + resize224 | sample_from_datasets 后需同构 |
| norm stats | native 维算、应用时动态补宽 | 已算的不用重算、避免 padding std=0 |
| schema | 运行时 restructure 注册表 | 免离线重存 |
| 数据路由 | builder_dir + uid | 多父目录 + 同名 tfds |
| ego 表示 | 绝对笛卡尔 EEF + 相机 mask | 人手无关节 |
| max_token_len | 256 | state 拼进 prompt 撑长 |
| 多机 | JAX distributed + even_splits + 去 guard | DLC 16 卡 |

---

## 常用命令

```bash
# 精确帧数统计（校准权重）
uv run --group rlds python scripts/count_cotrain_frames.py --config-name cotrain_all

# 逐数据集归一化统计（已存在自动跳过，--overwrite 强制重算）
uv run --group rlds python scripts/compute_cotrain_norm_stats.py --config-name cotrain_all

# 数据集动作语义实测
python RLDS/inspect_cotrain_datasets.py --dataset all --num_episodes 8

# 单机 8 卡训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds python scripts/train_cotrain.py \
    cotrain_all_2ep --exp_name=cotrain_all_2ep --fsdp_devices 4

# 多机 16 卡（DLC，每 worker 同命令）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds python scripts/train_cotrain.py \
    cotrain_all_2ep --exp_name=cotrain_all_2ep --fsdp_devices 8 \
    --batch-size 256 --num-train-steps 72000
```

> RLDS 路径依赖 dlimp：先 `uv sync --group rlds`。数据在 `/mnt/data/RLDS/`（各数据集 `builder_dir` 已写死在 config）。

---

## 未完成 / 后续工作

- **Stage 2 cotrain 有效性消融**（项目核心，未开始）：real-only / real+ego / real+anchor 等对比。
- **Domain anchor 搭建**：ego 与真机共享任务/场景的对齐。
- **评估链 L2/L3**：当前只有 L1（离线 action MSE），RMBench 仿真、实机 rollout 未接。
- **权重精确化**：当前基于估算帧数，需用 `count_cotrain_frames.py` 校准。
- **扩数据**：接入 droid/agibot 等，向 ~10000h 规模推进（当前 ~285h ≈ 目标 3%）。
- **EgoVerse 旋转表示**：欧拉角 ±π wrap 风险（低优先，必要时换 6D/sin-cos）。
