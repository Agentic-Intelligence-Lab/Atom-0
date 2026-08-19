# FastWAM wam-cross configs

Three registered FastWAM training configs (see `openpi.cotrain.config`):

| CONFIG_NAME | Data | Purpose |
|---|---|---|
| **`wam-cross-robot`** | 15 real-robot datasets | Pretrain Video+Action MoT (robot-only loss) |
| **`wam-cross-fix`** | 47 datasets (real robot + EgoVerse + AtomAligned) | Pretrain full mixture (ego+robot four-way loss) |
| **`wam-cross-piper-ft`** | piper2 + piper30 | Fine-tune from `wam-cross-robot` ckpt; **freeze Video DiT (Wan)**, train Action DiT + MoT |

Shared model layout (all three):

- **80D** unified action space, `action_horizon=32`, `video_num_frames=9`
- **3 cameras** + `concat_multi_camera=robot_wrist`
- **288×256** compose (head 192×256 + wrists 96×128 each)
- Norm assets: `assets/cotrain_real_robot_ego_fix/<dataset_uid>/`
- RLDS: `RLDS_DATA_DIR` (default `/mnt/bos/bo23lu`)
- DDP: `wam-cross-fix` 使用 `rlds_partition_builders_by_rank=True`（每 rank 只打开部分 builder，避免 47 路 TFDS CPU OOM）；`wam-cross-robot` / `wam-cross-piper-ft` 仍为 `False`

## ActionDiT init (pretrain only)

`wam-cross-robot` / `wam-cross-fix` load Wan2.2 TI2V-5B + ActionDiT linear-interp backbone  
(`checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`).

`wam-cross-piper-ft` skips Wan pretrain download; loads full MoT from `pytorch_weight_path` / `INIT_CHECKPOINT`.

## Baige submit examples

```bash
# 1) Pretrain 15 real-robot datasets
CONFIG_NAME=wam-cross-robot EXP_NAME=fw-wam-cross-robot-v1 \
  RLDS_DATA_DIR=/mnt/bos/bo23lu BATCH_SIZE=160 \
  pixi run python atom0_fastwam_train_job.py

# 2) Pretrain full mixture (47 datasets)
CONFIG_NAME=wam-cross-fix EXP_NAME=fw-wam-cross-fix-v1 \
  RLDS_DATA_DIR=/mnt/bos/bo23lu BATCH_SIZE=160 \
  pixi run python atom0_fastwam_train_job.py

# 3) Piper fine-tune (freeze Wan Video DiT)
CONFIG_NAME=wam-cross-piper-ft EXP_NAME=fw-wam-cross-piper-ft-v1 \
  INIT_CHECKPOINT=checkpoints/wam-cross-robot/<exp>/20000 \
  BATCH_SIZE=208 PEAK_LR=1e-5 NUM_TRAIN_STEPS=20000 \
  pixi run python atom0_fastwam_train_job.py
```

Local smoke:

```bash
cd Atom-0
MODE=smoke CONFIG_NAME=wam-cross-robot EXP_NAME=smoke bash scripts/train_fastwam_baige.sh
```

## Inference / eval

```bash
source scripts/atom0_env.sh
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints/fastwam"

PYTHONPATH=src .venv/bin/python scripts/run_fastwam_joint_episode.py \
  --checkpoint checkpoints/wam-cross-piper/<exp>/70000 \
  --config-name wam-cross-piper-ft \
  --dataset piper2 --split seen_test --episode-index 117 \
  --image-resolution 288,256
```

Use `--config-name wam-cross-robot` or `wam-cross-fix` when evaluating ckpts from those runs (same architecture).
