# State-Action 筛选数据预训练配置

## 1. 配置结论

新增生产配置：

```text
cotrain_full_all_sa_filtered
```

数据根目录默认为：

```text
/data/wudi/RLDS_SA_Filtered
```

可通过 `SA_FILTERED_RLDS_DATA_DIR` 覆盖。该配置不修改现有
`cotrain_full_all_full_norm`，两者的数据路径、采样权重和 norm assets 完全独立。

## 2. 数据组成

配置从 44 个 RLDS 配置出发，保留原有 5 个主动排除：

- `robocoin_unitree_g1_dex3_s28_a28`
- `robomind_tienkung_sim_s38_a38`
- `robocoin_leju_robot_s54_a54`
- `robocoin_agilex_decoupled_magic_s14_a14_fps50`
- `robocoin_agilex_decoupled_magic_s26_a26`

此外不加载 State-Action 筛选后 train 为空的 3 个 RoboCOIN 配置：

- `robocoin_unitree_g1_s28_a28_high`
- `robocoin_unitree_g1_s28_a28`
- `robocoin_unknown_s30_a30_high`

最终包含 36 个 active dataset：

| 口径 | 数量 |
|---|---:|
| 原始 train episode | 355,654 |
| 原有 5 个主动排除 | 27,114 |
| State-Action 额外删除 | 27,951 |
| 最终 train episode | 300,589 |
| 最终 train frame | 243,049,603 |

采样权重按每个 dataset 筛选后的 train episode 数计算：

```text
weight(dataset) = kept_train_episodes(dataset) / 300589
```

36 个期望数量固化在 `SA_FILTERED_TRAIN_EPISODES`。启动前预检会将这些数量、
`dataset_info.json` 与权重逐项对账，不允许静默使用空 split 或过期权重。

## 3. 计算 norm stats

不能复用原始 BOS RLDS 的 norm stats。运行以下专用脚本：

```bash
cd /data/wudi/Atom-0
./scripts/compute_cotrain_full_all_sa_filtered_norm_stats.sh
```

默认输出：

```text
/data/wudi/Atom-0/assets/cotrain_full_all_sa_filtered/<dataset_id>/
```

每个 dataset 必须生成：

```text
norm_stats.json
norm_stats_meta.json
unified_action_space.json
```

脚本会显示逐 dataset 进度和 ETA。中断后直接重跑同一命令，已完成且具有
full-run metadata 的 dataset 会跳过。强制重算时：

```bash
./scripts/compute_cotrain_full_all_sa_filtered_norm_stats.sh --overwrite
```

可以只重算一个 dataset：

```bash
./scripts/compute_cotrain_full_all_sa_filtered_norm_stats.sh \
  --dataset-id droid \
  --overwrite
```

若需替换数据或 assets 路径：

```bash
SA_FILTERED_RLDS_DATA_DIR=/path/to/RLDS_SA_Filtered \
OUTPUT_ASSETS_DIR=/path/to/assets/cotrain_full_all_sa_filtered \
./scripts/compute_cotrain_full_all_sa_filtered_norm_stats.sh
```

## 4. 训练前预检

norm 完成后运行：

```bash
cd /data/wudi/Atom-0

SA_FILTERED_RLDS_DATA_DIR=/data/wudi/RLDS_SA_Filtered \
PARAMS_PATH=/data/models/openpi \
.venv/bin/python scripts/preflight_cotrain_baige.py \
  cotrain_full_all_sa_filtered \
  --assets-base /data/wudi/Atom-0/assets \
  --params-path /data/models/openpi
```

预期输出包含：

```text
PASS cotrain_full_all_sa_filtered: datasets=36, source_frames=243,049,603
```

预检会验证：

- 36 个 dataset id 与 8 个排除项；
- 每个 train split 的 episode 数；
- 所有权重与权重和；
- builder 目录与 norm metadata 的路径一致性；
- 80D norm 的 shape、finite value 和 inactive-slot 默认值；
- 初始化 checkpoint 格式。

## 5. 启动预训练

`train_cotrain_baige.sh` 已支持新配置，并会在启动训练前自动执行上述预检。

先做 20 step smoke test：

```bash
cd /data/wudi/Atom-0

CONFIG_NAME=cotrain_full_all_sa_filtered \
EXP_NAME=cotrain_full_all_sa_filtered_smoke \
SA_FILTERED_RLDS_DATA_DIR=/data/wudi/RLDS_SA_Filtered \
PARAMS_PATH=/data/models/openpi \
MODE=smoke \
bash scripts/train_cotrain_baige.sh
```

正式预训练：

```bash
cd /data/wudi/Atom-0

export WANDB_API_KEY='...'

CONFIG_NAME=cotrain_full_all_sa_filtered \
EXP_NAME=cotrain_full_all_sa_filtered_pretrain_v1 \
SA_FILTERED_RLDS_DATA_DIR=/data/wudi/RLDS_SA_Filtered \
PARAMS_PATH=/data/models/openpi \
MODE=train \
bash scripts/train_cotrain_baige.sh
```

默认训练步数按 243,049,603 个保留 frame 和实际 global batch size 计算为一个 aggregate
pass。可通过 `NUM_TRAIN_STEPS`、`BATCH_SIZE`、`WARMUP_STEPS`、`DECAY_STEPS` 显式覆盖。

## 6. 完整性约束

- 新配置只使用 `/data/wudi/RLDS_SA_Filtered` 的 builder，不回退到 BOS 原始数据；
- 原有配置与 assets 不变，可继续复现旧训练；
- 若 RLDS 内容再次变化，必须重新生成 episode 数、权重和 norm stats；
- 正式训练不应使用部分完成的 norm assets，自动 preflight 会阻止这种情况。
