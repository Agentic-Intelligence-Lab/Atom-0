# EgoScale-inspired staged training（平行夹爪版）

本文档对应以下配置：

| 阶段 | 配置 | 数据 | 初始化 |
|---|---|---|---|
| Stage 1 | `egoscale_stage1_ego` | EgoVerse 4 个干净 builder + EgoVerse-RL2 2 个 builder（暂不含 Scale） | 服务器 PaliGemma/Gemma NPZ 初始化视觉语言骨干，80D action stack 随机初始化 |
| Stage 2 baseline | `egoscale_stage2_robot` | full-all 去掉全部 EgoVerse | Stage 1 严格 checkpoint |
| Stage 2 aligned | `egoscale_stage2_aligned` | 新采 human/robot EEF+gripper | Stage 1 严格 checkpoint |
| Stage 3 | `egoscale_stage3_robot` | robot-only | aligned Stage 2 严格 checkpoint |

后续阶段必须显式传 `--weight-loader.params-path`。加载器要求 checkpoint 与当前 80D
模型完全同构，任何 shape 或缺失参数都会在训练前失败，避免静默随机初始化。

## NAS 路径

所有现有 builder 路径由一个环境变量控制：

```bash
export ATOM_RLDS_ROOT=/mnt/workspace/RLDS
```

目录结构保持现有约定：

```text
/mnt/workspace/RLDS/
  AgiBot/
  DROID/
  EgoVerse_full/
  realworld_piper/
  RoboCOIN/
  RoboMIND_full/
  aligned_parallel_gripper/       # 尚未采集时可不存在
```

aligned 数据也可单独设置：

```bash
export ATOM_ALIGNED_RLDS_ROOT=/mnt/workspace/RLDS/aligned_parallel_gripper
export ATOM_ALIGNED_HUMAN_BUILDER_DIR=/path/to/human/aligned_parallel_gripper/1.0.0
export ATOM_ALIGNED_ROBOT_BUILDER_DIR=/path/to/robot/aligned_parallel_gripper/1.0.0
```

## Aligned RLDS 14D contract

human 和 robot 分别保存为独立 TFDS builder，便于独立归一化和评估。每帧必须包含：

```text
state[T,14]   = L xyz, L yaw/pitch/roll, L gripper,
                R xyz, R yaw/pitch/roll, R gripper
actions[T,14] = 同一顺序的 absolute target
image_base[T], image_left_wrist[T], image_right_wrist[T] = encoded JPEG
image_mask_base[T], image_mask_left_wrist[T], image_mask_right_wrist[T] = bool
prompt[T] = string
eef_frame = scalar episode string
```

EEF 保持项目最新版统一动作空间约定：absolute `xyz + yaw/pitch/roll`，不做 SE(3) delta；
夹爪为连续 absolute target，统一 `0=open, 1=closed`。映射槽位为
`U8-U13/U17` 和 `U37-U42/U46`。

默认 human:robot 采样权重为 `0.8:0.2`，正式实验前应依据采集规模做消融。

## Norm stats

Stage 1 使用专用的
`assets/egoscale_stage1_ego_cartesian_clean_rl2`：aria、eva、human、mecka 以及
EgoVerse-RL2 的 EVA/human 两个 builder 的统计量来自官方
`actions_cartesian`，并带有 `action_chunk_metadata.json`。当前 BOS 中的 Scale 子集具有异常
pose tails，已从生产 Stage 1 暂时排除，但 mapping 和旧配置仍保留，待数据重处理后重新审计。
旧的 `assets/egoscale_stage1_ego_cartesian_clean` 不含 RL2，
`assets/cotrain_full_all_full_norm` 则按相邻帧 `action` 计算；二者都不能用于新的 Stage 1。

robot 阶段继续复用经过数据审计的 `assets/cotrain_real_robot_fix`。这些统计量都已随 Git
仓库提供，`ASSETS_BASE_DIR` 默认就是仓库内的 `assets`。未来新增 aligned builder 时，才需要
先计算它自己的 smoke stats：

```bash
uv run --group rlds python scripts/compute_cotrain_norm_stats_light.py \
  --config-name egoscale_stage2_aligned \
  --exp-name norm_probe \
  --assets-base-dir ./assets \
  --max-frames 10000
```

robot/aligned 阶段替换 config name 即可。正式训练前必须运行
`compute_cotrain_full_norm_stats_light.py` 得到全量统计，不能把 probe stats 用于论文实验。

## Stage 1 动作语义

Stage 1 的 state 和 action 仍是 absolute 双手 EEF
`xyz + yaw/pitch/roll`，不做 delta。区别在时间维：RLDS 的
`actions_cartesian[t]` 已提供与当前帧对齐的 100 步未来轨迹，且第 0 步等于
`action[t]`。loader 将完整 100 步时间窗均匀重采样为模型的 50 步 horizon，不再从相邻
episode 帧重新拼接。这样避免移动 head frame 下相邻帧 pose 坐标系不一致。

动作表示、数据 mixture 和 norm 均已改变，因此以前使用 5 个 builder 训练得到的 Stage 1
checkpoint **不能 resume**。新训练必须使用新的 `EXP_NAME`、`RESUME=0`，从
`pi05_base` 重新初始化。

## DSW smoke

```bash
export ATOM_RLDS_ROOT=/mnt/workspace/RLDS
export ATOM_PI05_BASE_PARAMS=/mnt/workspace/cache/openpi/openpi-assets/checkpoints/pi05_base/params
export ASSETS_BASE_DIR=$PWD/assets
export CHECKPOINT_BASE_DIR=/mnt/workspace/Atom-0-checkpoints

STAGE=stage1_ego \
EXP_NAME=stage1_dsw_smoke \
FSDP_DEVICES=2 BATCH_SIZE=2 NUM_TRAIN_STEPS=20 \
bash scripts/run_egoscale_stage.sh
```

Stage 2：

```bash
export STAGE1_EXP=stage1_dsw_smoke
export STAGE1_STEP=19  # 以 NAS 中真实生成的 checkpoint 步数为准
export PARAMS_PATH=/mnt/workspace/Atom-0-checkpoints/egoscale_stage1_ego/${STAGE1_EXP}/${STAGE1_STEP}/params
export ASSETS_BASE_DIR=$PWD/assets
export CHECKPOINT_BASE_DIR=/mnt/workspace/Atom-0-checkpoints
STAGE=stage2_robot \
EXP_NAME=stage2_dsw_smoke \
FSDP_DEVICES=2 BATCH_SIZE=2 NUM_TRAIN_STEPS=20 \
bash scripts/run_egoscale_stage.sh
```

首次 smoke 默认关闭 W&B、action-MSE 和轨迹图，减少 JIT 时间。数据、loss、保存恢复通过后再设置
`WANDB_ENABLED=1 RUN_ACTION_MSE=1`。

## 阿里云 DSW / DLC 环境

推荐使用阿里云 PAI 官方镜像：

```text
modelscope:1.31.0-pytorch2.8.0-gpu-py311-cu124-ubuntu22.04
```

项目会在实例本地 `${HOME}/.cache/atom0/venvs/Atom-0-py311` 中按 `uv.lock` 安装
JAX 0.5.3、Torch 2.7.1 和 TensorFlow CPU 2.15，并在仓库创建 `.venv` 软链接；这样可避免
向 NAS 写入大量 Python 小文件。安装默认使用阿里云 PyPI 且只加入训练所需的 `rlds` group；
需要开发工具时设置 `INSTALL_DEV=1`。不要在 DSW 内升级 NVIDIA driver。

上传代码后执行：

```bash
bash scripts/setup_aliyun_dsw_env.sh
```

当前 DSW 已实测为 2×L20Y 80GB、128GB RAM、128GB `/dev/shm`，适合用
`FSDP_DEVICES=2 BATCH_SIZE=2` 做计算和数据链路 smoke。该配置保存完整 Adam 训练状态时会在
Orbax 的 GPU-to-host 回传阶段超过 128GB 主机内存，因此 DSW smoke 应设置
`CHECKPOINT_PARAMS_ONLY=1`。此模式仍生成可用于推理和下一阶段初始化的 `<step>/params` 以及
norm assets，但不包含优化器状态，不能用于 `RESUME=1`。全参数大规模训练仍建议使用 8×80GB
GPU、至少 512GB 主机内存，并保持默认的完整 checkpoint。当前 NAS 挂载点是
`/mnt/workspace`；DLC 若使用不同挂载点，只需同步修改 `ATOM_RLDS_ROOT` 和 checkpoint 环境变量。

DLC 使用与 DSW 相同的镜像或把验证后的 DSW 环境制作成同地域 ACR 自定义镜像。DLC 的
`WORLD_SIZE/RANK` 是节点级变量，当前 JAX 入口每个节点只启动一个 Python 进程：

```bash
cd /mnt/workspace/junhe/Atom-0
ATOM_RLDS_ROOT=/mnt/workspace/RLDS \
ATOM_PI05_BASE_PARAMS=/mnt/workspace/cache/openpi/openpi-assets/checkpoints/pi05_base/params \
ASSETS_BASE_DIR=$PWD/assets \
CHECKPOINT_BASE_DIR=/mnt/workspace/Atom-0-checkpoints \
STAGE=stage1_ego FSDP_DEVICES=8 BATCH_SIZE=512 NUM_TRAIN_STEPS=100000 \
WANDB_ENABLED=1 RUN_ACTION_MSE=1 bash scripts/run_egoscale_stage.sh
```

不要用 `torchrun --nproc_per_node=8` 包裹该命令；否则会在每个节点启动 8 个 JAX 进程，
与当前 node-level JAX distributed 和 `fsdp_devices=8` 冲突。

大规模训练前先在 DLC 做 2 节点 × 8 卡、100 steps 测试，确认：两个节点均加入、各节点读取
不同 split、只有 rank 0 创建 W&B run、checkpoint 能保存并恢复。

首次启动默认使用 `OVERWRITE=1`。中断后从同一个实验目录恢复时设置 `RESUME=1`（脚本会自动
关闭 overwrite）；不要同时设置 `RESUME=1 OVERWRITE=1`。后续阶段的 `PARAMS_PATH` 必须填写
checkpoint 目录中真实存在的 `<step>/params`，不要按总步数猜目录名。
