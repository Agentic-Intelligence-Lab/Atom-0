# EgoMimic Stage 2 接入与训练

## 结论

EgoMimic 不能复用项目中预留的双臂 14D
`EEF xyz+ypr+gripper` aligned contract。公开文件的真实监督是：

| 域 | observation | 100-step future action |
|---|---|---|
| human，单臂任务 | 当前相机坐标系中的手部 XYZ，3D | `actions_xyz_act[100,3]` |
| robot，单臂任务 | 6 arm joints + 1 gripper + EEF XYZ，10D | `actions_joints_act[100,7] + actions_xyz_act[100,3]` |
| human，双臂任务 | 左右手 XYZ，6D | `actions_xyz_act[100,6]` |
| robot，双臂任务 | 双臂 joint/gripper + 左右 EEF XYZ，20D | `actions_joints_act[100,14] + actions_xyz_act[100,6]` |

因此 Atom 的适配方式是：

- human 和 robot 的 XYZ 写入相同的 EEF position slots；
- robot 样本额外监督 arm joint 和 gripper slots；
- robot arm absolute joint target 在 normalization 前转成 delta；
- gripper 和 EEF XYZ 保持 absolute；
- Euler、human gripper 等不存在的标签保持 `mask=0`；
- human/robot 使用独立 norm stats；
- 官方 100-point future trajectory 均匀重采样到 Pi0.5 的 50-step horizon。

第一轮流程使用最小的 groceries 人机配对数据。配置名为
`egoscale_stage2_egomimic`，启动 stage 名为 `stage2_egomimic`。
真实文件审计显示 groceries 是双臂数据：human source width 为 6，
robot source width 为 20（14D joint/gripper + 6D EEF XYZ）。

> 数据授权注意：截至接入时，Hugging Face 数据集页面没有声明 dataset
> license。EgoMimic 代码仓库的 MIT license 不自动等价于数据授权。公开
> 数据可先用于内部研究验证，进一步发布模型或再分发数据前应向作者确认。

## 一、下载 groceries 人机数据

百度环境无法直接访问 Hugging Face 时使用国内镜像：

```bash
mkdir -p /data/junhe/datasets/EgoMimic

HF_ENDPOINT=https://hf-mirror.com \
/data/junhe/Atom-0/.venv/bin/huggingface-cli download \
  gatech/EgoMimic \
  --repo-type dataset \
  --include groceries_human.hdf5 groceries_robot.hdf5 \
  --local-dir /data/junhe/datasets/EgoMimic
```

下载中断时重复同一命令即可从 `.incomplete` 文件续传。

完整性检查：

```bash
ls -lh \
  /data/junhe/datasets/EgoMimic/groceries_human.hdf5 \
  /data/junhe/datasets/EgoMimic/groceries_robot.hdf5
```

## 二、转换 smoke RLDS

先只转换少量 episode，验证 schema、图像、split、mapping 和训练链路：

```bash
cd /data/junhe/Atom-0

.venv/bin/python scripts/convert_egomimic_hdf5_to_rlds.py \
  --source-hdf5 /data/junhe/datasets/EgoMimic/groceries_human.hdf5 \
  --output-data-dir /data/junhe/RLDS/EgoMimic_smoke_v2 \
  --max-train-episodes 2 \
  --max-validation-episodes 1

.venv/bin/python scripts/convert_egomimic_hdf5_to_rlds.py \
  --source-hdf5 /data/junhe/datasets/EgoMimic/groceries_robot.hdf5 \
  --output-data-dir /data/junhe/RLDS/EgoMimic_smoke_v2 \
  --max-train-episodes 2 \
  --max-validation-episodes 1
```

生成目录：

```text
/data/junhe/RLDS/EgoMimic_smoke_v2/
└── ego_mimic_rlds/
    ├── groceries_human/1.0.0/
    └── groceries_robot/1.0.0/
```

转换器使用官方 `mask/train` 和 `mask/valid`。公开 groceries 文件只有一个
5000-frame demo，两个官方 mask 都指向该 demo，因此 validation 也不是独立
episode。为避免把同一份大图像数据物理写入三次，groceries builder 只保存
一次 `train` split，配置中的 `seen/unseen` 都读取该 split。这里的验证指标
只能作为流程健康检查，不能作为 held-out 或 unseen-task 结果汇报。

转换器还会把公开文件中的 5000-frame 长 demo 切成最多 256 帧的 RLDS
episodes。每帧的 `actions_*_act[100]` 已经预先对齐，因此切 episode 不会
截断 future-action 监督；这样也避免单个 TFRecord example 达到 1GB，并让
两个 JAX host 能按 episode 读取互斥数据。

## 三、计算 smoke norm stats

smoke stats 与正式 stats 分开，避免少量样本统计污染正式训练：

```bash
cd /data/junhe/Atom-0

export ATOM_EGOMIMIC_RLDS_ROOT=/data/junhe/RLDS/EgoMimic_smoke_v2
export ASSETS_BASE_DIR=/data/junhe/smoke-assets-v2

.venv/bin/python scripts/compute_cotrain_norm_stats_light.py \
  --config-name egoscale_stage2_egomimic \
  --exp-name egomimic_groceries_smoke_norm \
  --assets-base-dir "$ASSETS_BASE_DIR" \
  --max-frames 4096
```

应生成：

```text
/data/junhe/smoke-assets-v2/egoscale_stage2_egomimic_groceries/
├── action_chunk_metadata.json
├── egomimic_groceries_human/
│   ├── norm_stats.json
│   └── unified_action_space.json
└── egomimic_groceries_robot/
    ├── norm_stats.json
    └── unified_action_space.json
```

## 四、Stage 2 smoke

从一个已经完成写入的 Stage 1 checkpoint 初始化，例如：

```text
/data/junhe/checkpoints/egoscale_stage1_ego/
  stage1_ego_cartesian_clean_baidu_v2/5000/params
```

不要读取 `.orbax-checkpoint-tmp-*`，也不要读取正在写入的 step。

下发器环境：

```bash
cd /data/junhe/baige-cluster
source /data/junhe/.venvs/baige-py311/bin/activate

set -a
source .env
set +a

export PFS_NAME=pfs-AndjNd

export MODE=smoke
export STAGE=stage2_egomimic
export EXP_NAME=stage2_egomimic_groceries_smoke_v1

export INSTANCES=1
export GPU_PER_NODE=8
export FSDP_DEVICES=8
export BATCH_SIZE=16
export NUM_TRAIN_STEPS=100

export PARAMS_PATH=/data/junhe/checkpoints/egoscale_stage1_ego/stage1_ego_cartesian_clean_baidu_v2/5000/params
export ATOM_EGOMIMIC_RLDS_ROOT=/data/junhe/RLDS/EgoMimic_smoke_v2
export ASSETS_BASE_DIR=/data/junhe/smoke-assets-v2
export CHECKPOINT_BASE_DIR=/data/junhe/checkpoints

export DATA_NUM_PARALLEL_READS=1
export DATA_NUM_PARALLEL_CALLS=2
export SHUFFLE_BUFFER_SIZE=128

export SAVE_INTERVAL=50
export EVAL_INTERVAL=50
export NUM_VAL_BATCHES=1
export RUN_ACTION_MSE=0

export WANDB_ENABLED=0
export OVERWRITE=1
export RESUME=0

python atom0_jax_job.py
```

验收项：

- strict loader 完整加载 Stage 1 80D params；
- human batch 只激活左右 EEF XYZ 六个 action slots；
- robot batch 激活左右臂各 6 joints、两个 gripper、左右 EEF XYZ；
- robot joint 走 absolute-to-delta，gripper/XYZ 保持 absolute；
- 100-step source chunk 被均匀重采样为 50 steps；
- human/robot loss、aggregate validation 均为有限值；
- step 50 和最终完整 checkpoint 成功；
- 使用保存的 checkpoint 恢复 5–10 steps。

## 五、正式转换与 Stage 2 训练

smoke 通过后使用新的输出根目录转换全部 episode，不覆盖 smoke builder：

```bash
cd /data/junhe/Atom-0

.venv/bin/python scripts/convert_egomimic_hdf5_to_rlds.py \
  --source-hdf5 /data/junhe/datasets/EgoMimic/groceries_human.hdf5 \
  --output-data-dir /data/junhe/RLDS/EgoMimic_full

.venv/bin/python scripts/convert_egomimic_hdf5_to_rlds.py \
  --source-hdf5 /data/junhe/datasets/EgoMimic/groceries_robot.hdf5 \
  --output-data-dir /data/junhe/RLDS/EgoMimic_full
```

正式统计放入独立 assets 根目录：

```bash
export ATOM_EGOMIMIC_RLDS_ROOT=/data/junhe/RLDS/EgoMimic_full
export ASSETS_BASE_DIR=/data/junhe/assets

.venv/bin/python scripts/compute_cotrain_norm_stats_light.py \
  --config-name egoscale_stage2_egomimic \
  --exp-name egomimic_groceries_full_norm \
  --assets-base-dir "$ASSETS_BASE_DIR" \
  --max-frames 1000000
```

正式训练建议先使用最新稳定的 Stage 1 checkpoint，而不是固定使用早期
5000 step。第一轮配置：

```bash
export MODE=full
export STAGE=stage2_egomimic
export EXP_NAME=stage2_egomimic_groceries_full_v1

export INSTANCES=2
export GPU_PER_NODE=8
export FSDP_DEVICES=8
export BATCH_SIZE=512
export NUM_TRAIN_STEPS=50000

export PARAMS_PATH=/data/junhe/checkpoints/egoscale_stage1_ego/<stage1-exp>/<completed-step>/params
export ATOM_EGOMIMIC_RLDS_ROOT=/data/junhe/RLDS/EgoMimic_full
export ASSETS_BASE_DIR=/data/junhe/assets
export CHECKPOINT_BASE_DIR=/data/junhe/checkpoints

export DATA_NUM_PARALLEL_READS=4
export DATA_NUM_PARALLEL_CALLS=8
export SHUFFLE_BUFFER_SIZE=50000

export SAVE_INTERVAL=5000
export EVAL_INTERVAL=5000
export NUM_VAL_BATCHES=10
export RUN_ACTION_MSE=1

export WANDB_ENABLED=1
export WANDB_API_KEY_FILE=/data/junhe/.secrets/wandb_api_key
export OVERWRITE=0
export RESUME=0

python atom0_jax_job.py
```

本 Stage 2 默认冻结 PaliGemma language transformer，继续训练 vision encoder、
action expert 和动作投影。正式效果对照至少需要：

1. `pi05_base → robot-only`
2. `pi05_base → Stage 1 EgoVerse → robot-only`
3. `pi05_base → Stage 1 EgoVerse → EgoMimic aligned → robot-only`

EgoMimic groceries 虽是双臂 ALOHA 数据，但 human 侧仍没有 orientation 或
gripper 标签，且公开 valid 与 train 共享同一 demo。它不能替代项目未来计划
采集的双臂 EEF+平行夹爪 aligned 数据，也不能直接证明对目标机器人任务有效。
