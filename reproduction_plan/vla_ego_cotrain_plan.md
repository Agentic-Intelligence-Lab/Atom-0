# VLA 基础模型预训练计划：真机 + Ego 数据 Co-train

> 目标：参考 pi05/pi07，自研预训练一个 VLA 基础模型。**核心亮点 = 在大规模开源真机数据中引入 ego（人类第一视角）数据一起 co-train。**
> 数据集：robocoin、robomind、droid、agibot、egoverse。
> Codebase：`pi07_reproduction`（openpi），**暂不使用新增的 ki / mem / dcc 模块**（保持 vanilla pi0/pi05 配方）。
> 当前不直接上大规模训练，先做基础实验验证：(A) 真机 + ego cotrain 有效；(B) 大规模多数据集训练框架正确。

---

## 核心技术决策（已讨论确定）

| 项 | 决策 | 依据 |
|---|---|---|
| Ego 数据使用 | EgoVerse **cartesian mode**，双手末端 6-DoF 位姿 + 抓握作 state/action（EgoVerse 已自带手部/相机 pose 标注，无需自己抽 pose） | EgoVerse Sec.IV-B |
| 统一动作空间 | **双手末端笛卡尔**，固定布局 `[左手, 右手]`，单臂只填一侧 + mask 缺失侧；pad 到 openpi 的 `action_dim=32`（state/action 共用同一维度，`models/pi0.py:100/105/108`） | 人手与双臂机器人唯一天然共享空间 |
| 绝对 vs 相对 | **camera-centered 相对 delta 轨迹**（人手和真机统一约定）；绕开 ego 无稳定 base 系的问题，跨本体迁移更好 | EgoVerse `a^H=(T_t)⁻¹T_{t+i}p_{t+i}` |
| 旋转表示 | **6D 旋转表示**（不用欧拉角，避免 gimbal lock / ±π 不连续，利于 flow matching 回归） | — |
| 抓握 | 统一到 `[0,1]` 标量（人手连续抓握 vs 夹爪） | EgoVerse |
| 归一化 | **per-dataset quantile normalization**（1%/99% 分位 → [-1,1]），+ 随机 crop / color jitter | EgoVerse |
| 多数据集框架 | 复用 RLDS **`sample_from_datasets(datasets, weights)`**（`droid_rlds_dataset.py:232`，已支持加权混合、无限 repeat、shuffle buffer、权重和=1 校验）；把 `restructure()` 改成**每数据集可插拔** | 这是 codebase 内唯一已验证的大规模混合路径；LeRobot 路径是单数据集（`data_loader.py:134`） |
| 模型架构 | 主线：vanilla pi0 共享 action expert + pad-to-32；Plan B：EgoVerse 式 per-embodiment decoder 头；可保留 pi07 的 control-mode 文本 token | — |
| 评测 | L1 离线 action MSE（快代理）→ L2 RMBench 仿真成功率 → L3 实验室真机 rollout（金标准）；需先标定 L1↔L3 相关性 | EgoVerse 也用 offline MSE 作代理并承认其局限 |

### ⚠️ 两个必须记住的风险
1. **Anchor 依赖**：EgoVerse 证明，cotrain 涨点（+30%）**只在有"domain-aligned 人类数据"（同任务/同场景的 ego）做锚定时才出现**。纯堆 diverse ego 数据无 anchor → 不涨甚至掉点。**消融必须包含 anchor 轴**，否则可能误判成框架 bug。
2. **关掉 metadata/subgoal = 退回 vanilla 配方**：pi07 解决"异构数据 averaging 变差"靠的正是 metadata/subgoal（我们暂不用）。所以异构混合的拉平问题会原样存在，主线靠 EgoVerse 的"统一笛卡尔动作对齐 + per-dataset 归一化"来缓解，而非 pi07 的 prompt steering。

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
