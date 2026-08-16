# A-4 单头消融：对齐 AtomAligned + EgoVerse RL2 数据配置

## 配置结论

训练配置名为 `cotrain_full_all_atom_aligned_rl2`。它以 A-3
`cotrain_full_all_full_norm` 为基础，排除 `egoverse_scale`，再增加 4 个 AtomAligned
builder 和 2 个 EgoVerse RL2 builder，共 44 个 builder、317,831 条 train episode。
逐数据集 norm 全量扫描后的 source frames 合计为 253,163,923。

原有 A-3 的五个主动排除项保持不变，并额外对齐 `dev/weizhongxing` 对
`egoverse_scale` 的排除。采样权重完整复用该分支的历史分组缩放与归一化公式；
数据处理、norm 和权重均对齐双头分支，但本配置仍使用单 action head。

完整配置按来源分组如下：

| 来源 | builder 数 | 说明 |
| --- | ---: | --- |
| 自采 Piper | 2 | Piper30 新 task-split 版本 + Piper2 |
| AgiBot | 1 | 与 A-3 相同 |
| DROID | 1 | 与 A-3 相同 |
| EgoVerse_full | 4 | A-3 中排除 `egoverse_scale` |
| RoboCOIN | 18 | 保持 A-3 的主动排除策略 |
| RoboMIND_full | 12 | 保持 A-3 的主动排除策略 |
| AtomAligned_full | 4 | 本配置新增 |
| EgoVerse_rl2 | 2 | 本配置新增 |
| 合计 | 44 | A-3 的 38 个保留项 + 新增 6 个 |

## 新增数据

| dataset id | RLDS builder 目录 | train episodes | 原生 state/action |
| --- | --- | ---: | --- |
| `aligned_hangzhou_human_right` | `AtomAligned_full/aligned_hangzhou_human_right/1.0.0` | 387 | 7D absolute，右臂 EEF pose + gripper |
| `aligned_hangzhou_robot_right` | `AtomAligned_full/aligned_hangzhou_robot_right/1.0.0` | 90 | 7D absolute，右臂 EEF pose + gripper |
| `aligned_shenzhen_human_bimanual` | `AtomAligned_full/aligned_shenzhen_human_bimanual/1.0.0` | 656 | 14D absolute，左右臂 EEF pose + gripper |
| `aligned_shenzhen_robot_bimanual` | `AtomAligned_full/aligned_shenzhen_robot_bimanual/1.0.0` | 163 | 14D absolute，左右臂 EEF pose + gripper |
| `egoverse_rl2_eva` | `EgoVerse_rl2/eva_bimanual_front_1_left_wrist_right_wrist/ego_verse_infidata/1.0.0` | 2,831 | 14D absolute，左右臂 EEF pose + gripper |
| `egoverse_rl2_human` | `EgoVerse_rl2/human_bimanual_front_1/ego_verse_infidata/1.0.0` | 1,387 | 12D absolute，左右臂 EEF pose |

宿主机上的 `/data/wudi/RLDS` 是 `/mnt/bos/bo23lu` 的软链接，因此配置使用的实际
根目录为 `/mnt/bos/bo23lu`；百舸任务按原有方式挂载同一个 BOS 路径即可。

## Unified80 映射

- AtomAligned 单臂 7D：`xyz + rpy + gripper` 映射到右臂 EEF 与右 gripper 槽；
- AtomAligned 双臂 14D：左右各 `xyz + rpy + gripper` 映射到对应 EEF 与 gripper 槽；
- EgoVerse Human 12D：左右各 `xyz + yaw/pitch/roll` 映射到对应 EEF position/euler 槽；
- EgoVerse EVA 14D：上述 12D pose 加左右 gripper，gripper 映射到 Unified80 槽 16/45；
- 六个 builder 都是 absolute action，不启用 delta mask；
- 图像统一整理到 base、left wrist、right wrist 三个逻辑槽，不存在的视角由 image mask 屏蔽。
- AtomAligned/EgoVerse 读取 RLDS 内预生成的 100 点 chunk，再均匀采样为 50 点。

## norm 计算与断点续跑

A-3 中 34 个非 Ego builder 的 norm 可以直接复用；4 个保留的 EgoVerse-full 和新增
6 个 builder 因 action chunk/EVA gripper 语义改变，需要按对齐后的输入重新计算。
脚本按 dataset 逐个落盘，中断后重跑会跳过已经完成的目录。

```bash
cd /data/wudi/Atom-0
./scripts/compute_cotrain_full_all_atom_aligned_rl2_norm_stats.sh
```

norm 输出目录：

```text
/data/wudi/Atom-0/assets/cotrain_full_all_atom_aligned_rl2
```

## 训练前预检

```bash
cd /data/wudi/Atom-0
source scripts/atom0_env.sh
PARAMS_PATH=/data/models/paligemma/pt_224.npz \
.venv/bin/python scripts/preflight_cotrain_baige.py \
  cotrain_full_all_atom_aligned_rl2
```

预检会检查 44 个 builder 的路径、split、episode 总数、全局采样权重、逐数据集 norm、
Unified80 指纹和 PaliGemma 初始化文件。

正式训练命令见 [百度云训练指南](./百度云训练指南.md) 的“正式训练 A-4”。

## 自采数据版本确认

新增配置继承 A-3 的原始数据集合，因此自采数据没有切换到旧版本：

- Piper30：`/mnt/bos/bo23lu/realworld_piper_task_split/piper_s14_a14_fps30_c4_ee_pose_cam_front_cam_high_cam_left_wrist_cam_right_wrist/realworld_piper_infidata/1.1.0`，train 4,927；
- Piper2：`/mnt/bos/bo23lu/realworld_piper_2/realworld_piper_infidata/1.0.0`，train 902。
