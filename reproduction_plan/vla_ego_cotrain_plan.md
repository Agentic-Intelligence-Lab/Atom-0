# VLA 基础模型预训练计划：真机 + Ego 数据 Co-train

> 目标：参考 pi05/pi07，自研预训练一个 VLA 基础模型。**核心亮点 = 在大规模开源真机数据中引入 ego（人类第一视角）数据一起 co-train。**
> 数据集：robocoin、robomind、droid、agibot、egoverse。
> Codebase：`pi07_reproduction`（openpi），**暂不使用新增的 ki / mem / dcc 模块**（保持 vanilla pi0/pi05 配方）。
> 当前不直接上大规模训练，先做基础实验验证：(A) 真机 + ego cotrain 有效；(B) 大规模多数据集训练框架正确。

---

## 核心技术决策（已讨论确定）

| 项 | 决策 | 依据 |
|---|---|---|
| 动作空间（真机数据） | **不对齐！各数据集保留原生 state/action**，放前部 + zero-pad 到 `action_dim=32` + per-dataset 归一化；模型靠观测/本体条件化消歧 | 已核实 pi0/pi05 代码：DROID 8 维、Libero 7 维、Aloha 14 维各保留原生（`droid_policy.py:45`、`transforms.py:745`） |
| Ego 数据使用 | EgoVerse **cartesian mode**，双手末端 6-DoF 位姿 + 抓握（EgoVerse 自带 pose 标注）。笛卡尔是 ego 被迫的表示（人手无关节），**不是为对齐机器人** | EgoVerse Sec.IV-B |
| Ego↔机器人是否共享动作槽 | **降级为消融假设**，非硬要求。EgoVerse 真正的杠杆是 domain anchor（任务/场景重叠），不是动作空间对齐 | — |
| 绝对 vs 相对 | per-dataset 决定：绝对关节数据转 **delta**（关节 delta、夹爪绝对）；ego 用 camera-relative。**per-dataset 处理，混合前完成** | pi0 DROID/Aloha 做法 |
| 归一化 | **per-dataset 归一化（硬 must-have）**，pi05 用 quantile（1%/99%→[-1,1]） | EgoVerse + OXE 标准做法 |
| 多数据集框架 | 已实现独立 `src/openpi/cotrain/` 包（见下方实现进展），复用 RLDS `sample_from_datasets`；**未改动 openpi 原文件** | — |
| 模型架构 | vanilla pi05 共享 action expert + pad-to-32（暂不用 ki/mem/dcc/metadata/subgoal） | — |
| 评测 | L1 离线 action MSE + val flow loss（快代理，已实现，按 per-dataset + seen/unseen 输出）→ L2 RMBench 仿真 → L3 真机 rollout；需先标定 L1↔L3 | EgoVerse 也用 offline MSE 作代理 |

### ⚠️ 两个必须记住的风险
1. **Anchor 依赖**：EgoVerse 证明，cotrain 涨点（+30%）**只在有"domain-aligned 人类数据"（同任务/同场景的 ego）做锚定时才出现**。纯堆 diverse ego 数据无 anchor → 不涨甚至掉点。**消融必须包含 anchor 轴**，否则可能误判成框架 bug。
2. **关掉 metadata/subgoal = 退回 vanilla 配方**：pi07 解决"异构数据 averaging 变差"靠的正是 metadata/subgoal（我们暂不用）。靠观测条件化 + per-dataset 归一化来处理异构混合，而非 pi07 的 prompt steering。

---

## 实现进展（2026-06-19）

### 已实现：cotrain 框架（独立包，未改动 openpi 原文件）
`src/openpi/cotrain/`：
- **`rlds_dataset.py`**：多数据集加权混合 + 每数据集 train/val split（支持多 val label，如 seen/unseen）。`restructure` 注册表（`droid` / `standardized` / `robomind`）。**图像在 shuffle 之后再解码**（buffer 只装编码字节，避免 OOM）。
- **`transforms.py`**：`StandardizedInputs`（通用 inputs，可写 actions 副本）+ 按 `dataset_id` 分派的 `DispatchDeltaActions`（绝对→delta）和 `DispatchNormalize`（per-dataset 归一化）。
- **`data_loader.py`** / **`eval.py`**：训练/验证 loader；验证指标 = val flow loss（fixed-seed + 多采样平均）+ action MSE，**按 per-dataset + 聚合、seen/unseen 分别**输出（`val/{label}/{dataset}/...`）。
- **`config.py`**：`CotrainTrainConfig`(子类，含 eval 参数) + `CotrainDataConfig` + 配置注册表。
- `scripts/train_cotrain.py`（fork train.py + eval pass）、`scripts/compute_cotrain_norm_stats.py`（per-dataset 归一化统计）、`scripts/inspect_robomind.py`（数据检查）。

**hybrid 数据流水线**：schema 统一可离线烤，也可像 RoboMIND 这样在运行时用轻量 restructure 映射；per-dataset 的归一化/delta 在混合前/分派时完成（混合后只有一套通用 transform）。

### 已接入并跑通：RoboMIND（`robomind_infidata` v1.1.0）
- 经 `inspect_robomind.py` 核实：图像**编码存储**（保留 decode）；动作 = **绝对关节**；双臂 14 维 = [6 关节+夹爪]×2（夹爪 idx 6/13）；3 相机齐全；自带 `train`/`seen_test`/`unseen_test`。
- 运行时 `robomind` restructure 直接映射（无需离线重生成）；绝对→delta mask `(6,-1,6,-1)`；`action_dim=14`。
- 配置：`cotrain_robomind`（pi05_base 公开权重，正式）、`cotrain_robomind_smoke`（NoOp 随机初始化，快速冒烟）。
- **状态：2026-06-19 在 4×H800（`--fsdp_devices 4`）上 smoke test 成功跑起**，数据/标准化/归一化+delta/训练步/wandb/seen-unseen 验证全链路通过。

### 排障要点（已解决）
RLDS 依赖 dlimp（`uv sync --group rlds`）；shuffle-buffer OOM→解码移到 shuffle 后；只读数组→`np.array` 拷贝；tyro 裸 tuple→`tuple[int,...]`、`repo_id` 必填→给默认；PaliGemma bucket 401→改用公开 `pi05_base`；单卡 OOM→FSDP 4 卡分片 + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`。

### 运行
```bash
# 冒烟（随机初始化，验证管道）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds python scripts/train_cotrain.py \
    cotrain_robomind_smoke --exp_name=smoke --fsdp_devices 4
# 正式（pi05 预训练权重）：先算 norm stats，再训练
uv run --group rlds python scripts/compute_cotrain_norm_stats.py --config-name cotrain_robomind
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds python scripts/train_cotrain.py \
    cotrain_robomind --exp_name=robomind_pi05 --fsdp_devices 4
```

### 下一步
- 跑完整 smoke，确认 loss 下降 + 验证指标合理（per-dataset / seen-unseen）。
- 接入第二个数据集（验证真·多数据集混合 + 异构）。
- 接入 ego（EgoVerse），落地阶段 0 的 ego 表示与 anchor 消融。

---

## 阶段 0：统一数据 schema + ego/真机动作对齐（地基）

**目标**：定义统一中间格式，把异构数据收敛到同一套 state/action 表示。不跑训练。

任务：
- [ ] 定义统一样本 schema：`state[32]` / `action[chunk,32]`（布局 `[左手: xyz(3)+rot6d(6)+grip(1)=10, 右手: 10]`，剩余 pad）、`mask`（标记缺失臂/缺失值）、`images`（多视角 + padding/mask）、`prompt`、`dataset_id`、`embodiment_id`。
- [ ] 实现 ego（EgoVerse cartesian）→ schema 的转换：手部 pose → camera-relative delta 未来轨迹（6D 旋转）+ 抓握标量。
- [ ] 实现真机（先 droid）→ schema：FK 求末端位姿 → 同样 camera-relative delta + 单臂 mask 另一侧。
- [ ] **可视化验收**：把转换后的 action 还原叠回视频/轨迹，确认 ego 和真机在同一约定下数值语义一致（坐标系、单位、旋转方向）。
- [ ] 先只接 **2 个数据集**（droid 单臂 + egoverse）打通端到端。

**Exit criteria**：droid 与 egoverse 样本经转换后 state/action 维度、布局、单位、坐标系约定完全一致，可视化回贴正确。

---

## 阶段 1：多数据集训练框架正确性验证（工程，不谈效果）

**目标**：排除框架 bug。全程小模型、小步数。

任务：
- [ ] 把 `droid_rlds_dataset.py` 的 `restructure()` 从 DROID 专用改为**每数据集可插拔**（每个数据集一个 restructure，输出阶段 0 的统一 schema）；`chunk_actions` 改为相对当前帧（现实现是绝对关节，`droid_rlds_dataset.py:117`，不能照搬）。
- [ ] 接入 per-dataset quantile 归一化统计的计算与加载（扩展 `data_loader.py:192/220`）。
- [ ] **混合比例测试**：抽大量 batch 统计各 `dataset_id` 实际占比 == 设定 weight；shape/dtype/mask 正确；多 worker / 分布式下无重复或丢样本。
- [ ] **单 batch 过拟合测试**：固定一个 batch 反复训，loss 应压到接近 0（验证训练 loop + flow matching loss + 反传）。
- [ ] **单数据集 parity**：用新混合 loader 只放 1 个数据集，复现已知单数据集结果，确认无回退。

**Exit criteria**：占比/shape/mask 测试通过；单 batch 能过拟合；单数据集 parity 不掉点。

---

## 阶段 2：cotrain 有效性消融（科学验证）

**目标**：证明"真机 + ego cotrain 有效"，并定位 anchor 的作用。小模型、小数据、快速迭代。

先做：
- [ ] **L1↔L3 相关性标定**：少数 ckpt 上同时测离线 action MSE 与真机 rollout 成功率，确认 L1 可作为快速代理（这本身也是评测框架正确性的一部分）。

受控消融（唯一变量锁死：total step / 真机数据量 / 模型 / 超参全固定）：

| 实验 | 训练数据 | 目的 |
|---|---|---|
| A | real-only | baseline |
| B | real + **对齐 ego**（同任务/场景） | 核心假设：B > A |
| B' | real + **diverse ego（无 anchor）** | 验证 anchor 必要性（预期 B' 不如 B，甚至不涨——EgoVerse 已报告） |
| C | real + **对齐 + diverse ego** | 最优组合 |
| D | C 基础上扫 2–3 个 ego 混合比例 | 找 ego 边际贡献曲线 |

注意：优先选**真机 low-data** 的下游任务放大 ego 增益；结论必须落在 L2/L3 成功率，L1 只用于排除明显更差方案 + 画趋势。

**Exit criteria**：在 L2/L3 上 B/C 显著优于 A；anchor 轴（B vs B'）结论清晰。

---

## 阶段 3：逐数据集 scale-up → 全量预训练

**目标**：确认框架在全部 5 个数据集上稳定，再上规模。

任务：
- [ ] 逐个加数据集（robomind / agibot / robocoin），每加一个监控：各 dataset 的 loss、grad norm、是否拖垮整体、下游成功率。
- [ ] 确认 per-dataset 归一化 / mask / 混合权重在 5 数据集规模下仍正确。
- [ ] 调混合权重（参考 anchor 结论与各数据集质量）。
- [ ] （可选）若 vanilla 共享头出现明显 averaging，再切 EgoVerse 式 per-embodiment decoder 头（Plan B）。
- [ ] 启动全量预训练。

**Exit criteria**：5 数据集混合训练稳定收敛，下游指标不低于阶段 2 最优组合的趋势外推。

---

## 暂不做（明确排除）
- ki / mem / dcc 模块
- pi07 的 metadata / subgoal image / world model（亮点不在 steering，而在 ego cotrain）
- 欧拉角旋转表示、关节空间作为统一动作空间、ego 用绝对世界系位姿
