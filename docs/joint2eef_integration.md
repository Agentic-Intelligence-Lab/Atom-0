# Joint → EEF 集成说明（迁移到其他训练代码）

本文说明 Atom-0 中 **joint 转 EEF（FK 填充）** 的设计，以及若要把同样能力接到**别的训练仓库 / 模型**上，对方需要增加哪些模块与约定。

> **核心事实**：当前实现 **不修改 RLDS 原数据**。EEF 在 **读 batch 后、delta / normalize 之前** 用 URDF 正运动学在线写入 unified 向量中的 reserved EEF slot。Norm stats 也是在同样逻辑下 **离线统计** 得到，而不是写回数据集。

---

## 1. 本仓库里的数据流（先对齐概念）

```
RLDS 原始帧 (native joint/gripper/...)
    │
    ▼  [TF load] map_trajectory_tensorflow
    │           native → unified 80D（关节/hand 等已映射；EEF slot 仍为 0）
    │
    ▼  [numpy] StandardizedInputs
    │
    ▼  DispatchFillEefFromFk          ← URDF FK，填 state + actions 的 EEF slot
    │
    ▼  DispatchDeltaActions           ← 仅关节等 configured slot 变 delta
    │
    ▼  DispatchNormalize              ← 用 per-dataset norm_stats.json
    │
    ▼  Model (pi0 / FastWAM / …)
```

**顺序不能乱**：FK 必须在 **delta 之前**（需要 absolute joint）；normalize 必须在 **FK + delta 之后**（stats 按此空间统计）。

对应源码：

| 步骤 | 文件 |
|------|------|
| FK 核心 | `src/openpi/cotrain/fk_eef.py` |
| 80D slot 定义 + dataset 映射 | `src/openpi/cotrain/action_space.py` |
| 训练管线挂接 | `src/openpi/cotrain/transforms.py` → `DispatchFillEefFromFk` |
| Config 挂 FK slot / supervision | `src/openpi/cotrain/config.py` → `_attach_fk_eef_slots` |
| RLDS → 80D | `src/openpi/cotrain/rlds_dataset.py` → `map_trajectory_tensorflow` |
| Norm 统计（含 FK） | `scripts/compute_cotrain_norm_stats_light.py` |
| 批量重算 norm | `scripts/compute_fk_eef_norm_stats.py` |
| FK 校验 | `scripts/validate_joint2eef_fk.py` |
| URDF 清单 | `docs/joint2eef.md` + `assets/urdf/` |

---

## 2. Unified 80D 里 EEF 放哪（必须统一）

若对方也要混训 / 共用 norm，**slot 索引必须一致**（0-based）：

| Slot | 索引 | 含义 |
|------|------|------|
| `LEFT_ARM` | 0–6 | 左臂关节（最多 7 DOF） |
| `LEFT_EEF_POSITION` | **7–9** | 左 EE 位置 xyz（米） |
| `LEFT_EEF_EULER` | **10–12** | 左 EE 姿态 yaw/pitch/roll（弧度） |
| `LEFT_GRIPPER` | 16 | 左夹爪 |
| `RIGHT_ARM` | 29–35 | 右臂关节 |
| `RIGHT_EEF_POSITION` | **36–38** | 右 EE 位置 |
| `RIGHT_EEF_EULER` | **39–41** | 右 EE 姿态 |
| `RIGHT_GRIPPER` | 45 | 右夹爪 |

FK 输出格式：**absolute** `xyz + yaw/pitch/roll`（与 EgoVerse native EEF 一致，见 `fk_eef.py` 模块 docstring）。

常量定义：`src/openpi/cotrain/action_space.py` 中 `LEFT_EEF_POSITION = 7` 等。

---

## 3. 迁移到其他代码：需要增加的 7 块

下面按 **最小可运行** 列出。不要求对方 fork 全仓库，但 **语义要对齐**。

### 3.1 URDF 资源

- 拷贝或引用 `assets/urdf/<robot>.urdf`
- 每个 robot / dataset 的对应关系见 `docs/joint2eef.md`

### 3.2 FK 注册表（dataset_id → URDF / 关节名 / EE link）

等价于本仓库的 `FK_EEF_SPECS`（`fk_eef.py`）：

```text
dataset_id  →  urdf_file, left/right arm joint names, ee_link
```

每条记录还要能对应到 unified 里 **哪一段 slot 读关节角**（`LEFT_ARM` / `RIGHT_ARM` 起始索引）。

**校验**（强烈建议移植 `validate_fk_spec` 逻辑）：

- URDF 文件存在
- 关节名存在且为 revolute/continuous/prismatic
- URDF 臂 DOF == unified mapping 里该臂映射的 DOF
- 零位 FK 有限且非 NaN

仅校验通过的 dataset 才启用 FK（本仓库 `enabled_fk_dataset_ids()`）。

### 3.3 Native → Unified 映射

每个 dataset 需要一张 **source index → unified slot** 表（本仓库 `UNIFIED_ACTION_SPECS`）。

要求：

- 臂关节映射到 `LEFT_ARM` / `RIGHT_ARM` 连续 slot
- **不要**把 native EEF 映射进 7–12 / 36–41（joint-only 数据集由 FK 填）
- 指定哪些 slot 做 **absolute → delta**（通常仅 arm joint，EEF 保持 absolute）

若对方 action 维度不是 80，可以只用子集，但 **EEF 相对 arm 的 slot 约定** 建议保持一致，便于共用 norm assets。

### 3.4 运行时 FK 填充（训练 + 推理）

在对方 DataLoader / transform pipeline 中插入一步，等价于：

```python
# 伪代码 — 对应 fill_batch_dict / fill_eef_vectors_batch
if fk_enabled(dataset_id):
    state  = fk_fill(state,  fk_spec)   # [..., 80]
    action = fk_fill(action, fk_spec)   # [..., H, 80]
```

**输入**：已是 unified 布局、关节为 **absolute** 的向量  
**输出**：同一向量，EEF slot 被写入 FK 结果  
**不改**：RLDS 文件、关节 slot、hand/gripper slot

挂接位置：**delta 之前、normalize 之前**（与 `config.py` 里 `data_transforms` 顺序一致）。

可独立移植的文件：`fk_eef.py`（依赖 `numpy`, `scipy`；URDF 解析为自包含 XML，不依赖 ROS）。

### 3.5 Supervision / Loss mask

若模型只对部分维度算 loss，需要告诉模型 **EEF 维也要监督**：

- 本仓库：`UnifiedActionSpec.fk_eef_slots` + `action_supervision_mode`
- `action_mask = action_target_slots ∪ fk_eef_slots`（`JOINT_AND_EEF` 模式）

对方若没有 80D mask，至少要做到：

- **Joint-only 训练**：可以不填 EEF、不监督 EEF
- **Joint + EEF 训练**：loss mask 必须包含 FK 填写的 12 维（左 6 + 右 6）
- **EEF-only 训练**：只监督 `fk_eef_slots`（见 `supervision.py`）

FastWAM / pi0 通过 `observation.action_mask` 传入；对方模型需有等价机制，否则 EEF 维 loss 为 0。

### 3.6 Normalization（norm stats）

FK 开启后 **必须重算 norm**，不能复用 joint-only 的 stats（否则 actions 的 EEF 维 mean/std 仍是占位 0/1）。

本仓库流程：

```bash
scripts/validate_joint2eef_fk.py
scripts/compute_fk_eef_norm_stats.py --variant anchor --dataset-id <id> --overwrite
```

统计时的变换顺序与训练一致：**map → FK → delta → accumulate stats**。

注意 finalize 约定（`compute_cotrain_norm_stats_light.py`）：

| | state norm | actions norm |
|--|------------|--------------|
| 关节 / hand（mapped slots） | 真实 q01/q99 | 真实 q01/q99 |
| FK EEF slots | **占位** mean=0, q01=-1, q99=1（identity norm） | **真实**统计 |
| 未用 reserved slot | 占位 | 占位 |

因此：**训练时 state EEF 在线 FK 后几乎不缩放；actions EEF 会 normalize 到 ~[-1,1]**。对方若 proprio 也吃 EEF，需知晓这一不对称，或自行改为 state EEF 也统计真实 quantile。

输出目录结构（per dataset）：

```text
assets/<config_name>/<dataset_id>/
  norm_stats.json
  norm_stats_meta.json
  unified_action_space.json   # fingerprint，防 mapping 漂移
```

### 3.7 Config / 训练入口

本仓库侧需切换：

- `assets_name` → 含 FK 的 assets（如 `cotrain_fk_eef_plus_piper_ego`）
- `action_supervision_mode` → `JOINT_AND_EEF` 或 `EEF`
- 数据集 mixture 包含已注册 FK 的 `dataset_id`
- 训练机部署 `assets/urdf/`

---

## 4. 两种集成模式对比

### 模式 A：在线 FK（与本仓库相同，推荐）

| 项 | 说明 |
|----|------|
| 原数据 | 不改 RLDS，只存 joint |
| EEF 何时出现 | 每个 batch 读入后 FK 计算 |
| 对方要加 | §3.1–3.7 全部 |
| 优点 | 不重复存储；改 URDF/EE link 可重算 norm 而不重刷数据 |
| 缺点 | 训练时多一步 FK CPU；需带 URDF |

### 模式 B：离线预写 EEF 进数据

若对方希望 **数据集里直接带 EEF**（仍可不覆盖原 joint 字段）：

| 项 | 说明 |
|----|------|
| 离线脚本 | 对每帧用同一套 `fill_eef_vectors_batch` 写 unified 80D 或直接写 native 扩展列 |
| RLDS | 新增字段或替换 pipeline 中 map 后的 tensor |
| 训练时 | **可跳过** `DispatchFillEefFromFk`，但 **slot 布局、姿态约定、delta/norm 顺序** 仍须一致 |
| Norm | 仍按「已有 EEF 的 unified 向量」重算；不能混用 joint-only stats |
| 风险 | URDF 变更需全量重刷数据；左右臂 / 坐标系错误更难排查 |

**无论 A 还是 B**，模型侧看到的都应是：**normalize 后的 unified 向量 + 正确的 action_mask**。

---

## 5. 最小移植包（文件级）

若对方只想要「能算 EEF」的最小集合，建议打包：

```text
src/openpi/cotrain/fk_eef.py          # FK 核心（可改名为独立 package）
src/openpi/cotrain/action_space.py    # 至少 slot 常量 + map_array + apply_delta
assets/urdf/                          # 对应机器人
scripts/validate_joint2eef_fk.py      # 校验
```

并在对方仓库自行实现：

1. **Dataset map**：native → unified（等价 `UNIFIED_ACTION_SPECS[dataset_id]`）
2. **One transform**：调用 `fill_batch_dict`
3. **Delta + Normalize**：顺序与本仓库一致
4. **Loss mask**：包含 EEF slot
5. **Norm 脚本**：复制 `_state_actions_from_light_batch` 逻辑

不必移植整个 cotrain RLDS 栈，但 **80D 语义** 和 **变换顺序** 必须对齐，否则 norm stats 与训练输入不一致。

---

## 6. 新机器人接入检查清单

- [ ] URDF 放入 `assets/urdf/`，关节名与数据集 joint 顺序可对应
- [ ] 在 `FK_EEF_SPECS` 增加条目（dataset_id, urdf, joint names, ee_link）
- [ ] 在 `UNIFIED_ACTION_SPECS` 增加 native→80D 映射，arm DOF 与 URDF 一致
- [ ] `validate_joint2eef_fk.py` 对该 dataset 为 OK
- [ ] 重算 norm（`compute_fk_eef_norm_stats.py --dataset-id ... --overwrite`）
- [ ] 检查 `norm_stats.json`：**actions** 段 EEF 索引 7–12 / 36–41 非零
- [ ] 训练 config 使用 FK assets + `JOINT_AND_EEF`（或 `EEF`）
- [ ] 推理 pipeline 同样插入 FK（或数据已含 EEF），并使用同一 norm stats

---

## 7. 常见问题

**Q: 为什么 norm 里 state 的 EEF 是 0，actions 的 EEF 有值？**  
A: finalize 时 state 只对 `state_target_slots`（RLDS 里有的 proprio）保留统计；EEF 来自 FK，state 侧用 identity norm。actions 的 EEF 是监督目标，必须真实 quantile。

**Q: 只移植 FK 函数不够吗？**  
A: 不够。没有 unified mapping、delta 规则、action_mask 和匹配 norm，模型读到的 EEF 维仍是 0 或未归一化，loss 也不会监督这些维。

**Q: 对方不是 80 维 action 怎么办？**  
A: 可以只映射 arm + EEF + gripper 到对方维度，但需在文档中固定 **EEF 6 维语义**（xyz + ypr）和 **absolute vs delta** 规则；norm 与 mask 按对方维度重算，不能直接用 80D stats 的子切片（除非索引完全一致）。

**Q: piper / agibot 等特殊 case？**  
A: 注册表示例见 `fk_eef.py` 中 `piper30`/`piper2`（`piper_gripper.urdf`）、`agibot`（`agibot_G2.urdf`，joint 名 `idx21_arm_l_joint1` 等）。

---

## 8. 参考命令（本仓库内）

```bash
export RLDS_DATA_DIR=/path/to/RLDS
export PYTHONPATH=src

# 校验
uv run --group rlds python scripts/validate_joint2eef_fk.py --dataset-id piper30

# 单 dataset 重算 norm
uv run --group rlds python scripts/compute_fk_eef_norm_stats.py \
  --variant anchor \
  --output-assets-dir assets/cotrain_fk_eef_plus_piper_ego \
  --rlds-data-dir "$RLDS_DATA_DIR" \
  --dataset-id agibot \
  --overwrite

# 训练前检查
uv run --group rlds python scripts/preflight_cotrain_baige.py \
  fastwam_cotrain_fk_eef_plus_piper_ego
```

---

## 9. 与对方对齐的接口摘要（给对方负责人）

| 接口 | 类型 | 说明 |
|------|------|------|
| `fill_eef_vectors_batch(vectors, fk_spec)` | `[..., 80] → [..., 80]` | 核心 FK API |
| `FK_EEF_SPECS[dataset_id]` | 配置 | URDF + 臂定义 |
| `UnifiedActionSpec.action_mask` | `bool[80]` | 模型 loss 用 |
| `UnifiedActionSpec.delta_mask` | `bool[80]` | 哪些 action 维变 delta |
| `norm_stats.json` | 文件 | per-dataset state/actions 分 quantile |
| `assets/urdf/*.urdf` | 文件 | 训练/推理机器必须可访问 |

按上表在对方代码中找到 **等价挂接点**（dataloader transform、collate、model forward 前的 mask），即可在不改 RLDS 的前提下使用与本仓库一致的「计算好的 EEF」。
