# 真机与 Robot 数据训练配置

本文说明两套新增的 80D 统一动作空间训练配置：

| 配置名 | 数据组成 |
| --- | --- |
| `cotrain_real_only` | 原真机 `piper30` + 新真机 `piper2` |
| `cotrain_real_robot` | 上述两份真机数据 + AgiBot + DROID + RoboCOIN + RoboMIND_full；不包含 EgoVerse |

`cotrain_real_robot` 延续现有 `cotrain_full_all` 的数据质量选择，仍排除
`robocoin_unitree_g1_dex3_s28_a28` 和 `robomind_tienkung_sim_s38_a38`。两套配置均按 train
episode 数量设置采样权重。`piper2` 默认按 902 个 train episode 计算；若训练机上的 builder
数量不同，必须通过环境变量覆盖。

## 1. 数据路径

训练进程启动前设置以下环境变量。`REALWORLD_PIPER_2_BUILDER_DIR` 必须指向直接包含
`dataset_info.json` 和 `features.json` 的 TFDS version 目录。

```bash
cd /data/wudi/Atom-0

export RLDS_DATA_DIR=/mnt/bos/bo23lu
export REALWORLD_PIPER_2_BUILDER_DIR=/mnt/bos/bo23lu/realworld_piper_2/realworld_piper_infidata/1.0.0
export REALWORLD_PIPER_2_TRAIN_EPISODES=902

test -f "${REALWORLD_PIPER_2_BUILDER_DIR}/dataset_info.json"
test -f "${REALWORLD_PIPER_2_BUILDER_DIR}/features.json"
```

当前训练机已经按上述结构整理数据。若其他训练机的 RLDS 根目录不同，只需让
`REALWORLD_PIPER_2_BUILDER_DIR` 指向实际的 `.../<builder-name>/<version>` 目录，不需要修改代码。

新数据使用独立 id `piper2`，其 14D state/action 布局为：

```text
left_joint_1..6, left_gripper, right_joint_1..6, right_gripper
```

它映射到统一空间的 `U1-U6, U17, U30-U35, U46`。手臂 target 转成相对当前 state 的
delta，两个 gripper 保持 absolute。

该映射经过实际数据检查：builder metadata 和 train/seen/unseen 抽样均给出相同的左右臂字段
顺序；在一个 949 帧真实 train episode 中，全部 948 个相邻 transition、全部 14 个维度都满足
`action[t] == state[t+1]`，最大误差为 0。因此这里的 action 是 next-step absolute target，不能
当作源数据已经提供的 delta 再使用。

## 2. 计算 norm stats

每套训练配置使用独立的 assets 目录。首次训练前必须先计算全部 active dataset 的 80D norm
stats。下面的命令会跳过目标目录中已有且带 full-run metadata 的数据集；需要强制重算时增加
`--overwrite`。

### 2.1 仅真机数据

```bash
RLDS_DATA_DIR="${RLDS_DATA_DIR}" \
REALWORLD_PIPER_2_BUILDER_DIR="${REALWORLD_PIPER_2_BUILDER_DIR}" \
REALWORLD_PIPER_2_TRAIN_EPISODES="${REALWORLD_PIPER_2_TRAIN_EPISODES}" \
UV_CACHE_DIR=/data/wudi/.cache/uv \
uv run --group rlds python scripts/compute_cotrain_full_norm_stats_light.py \
  --config-name cotrain_real_only \
  --output-assets-dir /data/wudi/Atom-0/assets/cotrain_real_only \
  --num-parallel-reads 1 \
  --num-parallel-calls 2 \
  --overwrite
```

### 2.2 真机 + Robot 数据，不含 EgoVerse

```bash
RLDS_DATA_DIR="${RLDS_DATA_DIR}" \
REALWORLD_PIPER_2_BUILDER_DIR="${REALWORLD_PIPER_2_BUILDER_DIR}" \
REALWORLD_PIPER_2_TRAIN_EPISODES="${REALWORLD_PIPER_2_TRAIN_EPISODES}" \
UV_CACHE_DIR=/data/wudi/.cache/uv \
uv run --group rlds python scripts/compute_cotrain_full_norm_stats_light.py \
  --config-name cotrain_real_robot \
  --output-assets-dir /data/wudi/Atom-0/assets/cotrain_real_robot \
  --num-parallel-reads 1 \
  --num-parallel-calls 2 \
  --overwrite
```

如果两套配置都要训练，推荐先运行 2.2。norm stats 是逐 dataset 计算的，与 mixture 采样权重无关，
因此 `piper30` 和 `piper2` 可以无损复用到 `cotrain_real_only`，不必再次扫描两份真机数据：

```bash
mkdir -p \
  /data/wudi/Atom-0/assets/cotrain_real_only/piper30 \
  /data/wudi/Atom-0/assets/cotrain_real_only/piper2

cp -a /data/wudi/Atom-0/assets/cotrain_real_robot/piper30/. \
  /data/wudi/Atom-0/assets/cotrain_real_only/piper30/
cp -a /data/wudi/Atom-0/assets/cotrain_real_robot/piper2/. \
  /data/wudi/Atom-0/assets/cotrain_real_only/piper2/
```

计算完成后，每个 dataset 子目录都应同时包含：

```text
norm_stats.json
norm_stats_meta.json
unified_action_space.json
```

## 3. 启动训练

下面以本地 `pi05_base` 参数为例。`--batch-size` 是全局 batch size；多机任务中的所有进程必须
使用相同的环境变量和命令参数。

```bash
export PARAMS_PATH=/data/models/openpi/openpi-assets/checkpoints/pi05_base/params
export WANDB_API_KEY='...'
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

### 3.1 仅真机数据

```bash
UV_CACHE_DIR=/data/wudi/.cache/uv \
uv run --group rlds python -u scripts/train_cotrain.py cotrain_real_only \
  --exp-name=cotrain_real_only_run1 \
  --fsdp-devices=8 \
  --batch-size=256 \
  --num-train-steps=3000000 \
  --data-num-parallel-reads=1 \
  --data-num-parallel-calls=2 \
  --weight-loader.params-path="${PARAMS_PATH}"
```

### 3.2 真机 + Robot 数据，不含 EgoVerse

```bash
UV_CACHE_DIR=/data/wudi/.cache/uv \
uv run --group rlds python -u scripts/train_cotrain.py cotrain_real_robot \
  --exp-name=cotrain_real_robot_run1 \
  --fsdp-devices=8 \
  --batch-size=256 \
  --num-train-steps=3000000 \
  --data-num-parallel-reads=1 \
  --data-num-parallel-calls=2 \
  --weight-loader.params-path="${PARAMS_PATH}"
```

单机 GPU 数量不同时相应修改 `--fsdp-devices` 和全局 `--batch-size`。多机启动继续使用项目已有的
`RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 环境变量；训练入口会据此初始化 JAX distributed。

## 4. 启动前检查配置组成

```bash
UV_CACHE_DIR=/data/wudi/.cache/uv uv run python - <<'PY'
from openpi.cotrain import config

for name in ("cotrain_real_only", "cotrain_real_robot"):
    cfg = config.get_config(name)
    ids = [dataset.uid for dataset in cfg.data.datasets]
    print(name, len(ids), ids)
    assert "piper2" in ids
    assert not any(dataset_id.startswith("egoverse_") for dataset_id in ids)
PY
```

预期 `cotrain_real_only` 有 2 个 dataset，`cotrain_real_robot` 有 37 个 active dataset。

## 5. 检查 norm 与统一动作空间

完成计算或复制后运行：

```bash
UV_CACHE_DIR=/data/wudi/.cache/uv uv run python - <<'PY'
from pathlib import Path

import numpy as np

from openpi.cotrain import action_space, config
from openpi.shared import normalize


def check(config_name: str) -> None:
    cfg = config.get_config(config_name)
    root = Path("assets") / config_name
    for dataset in cfg.data.datasets:
        directory = root / dataset.uid
        spec = action_space.UNIFIED_ACTION_SPECS[dataset.uid]
        stats = normalize.load(directory)
        action_space.validate_metadata(directory, spec)
        assert (directory / "norm_stats_meta.json").is_file(), directory

        state_mask = np.zeros(action_space.UNIFIED_ACTION_DIM, dtype=bool)
        state_mask[list(spec.state_target_slots)] = True
        for key, mask in (("state", state_mask), ("actions", np.asarray(spec.action_mask))):
            value = stats[key]
            for field in ("mean", "std", "q01", "q99"):
                array = np.asarray(getattr(value, field))
                assert array.shape == (action_space.UNIFIED_ACTION_DIM,), (dataset.uid, key, field)
                assert np.isfinite(array).all(), (dataset.uid, key, field)
            assert np.allclose(np.asarray(value.mean)[~mask], 0)
            assert np.allclose(np.asarray(value.std)[~mask], 1)
            assert np.allclose(np.asarray(value.q01)[~mask], -1)
            assert np.allclose(np.asarray(value.q99)[~mask], 1)
    print(f"PASS {config_name}: {len(cfg.data.datasets)} datasets")


check("cotrain_real_robot")
check("cotrain_real_only")
PY
```

预期输出：

```text
PASS cotrain_real_robot: 37 datasets
PASS cotrain_real_only: 2 datasets
```
