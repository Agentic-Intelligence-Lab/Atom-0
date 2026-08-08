# wam-cross-robot

Cross-embodiment FastWAM training on 15 real-robot datasets (no EgoVerse).

See also **`wam-cross-robot-ego`** (same robot mix + aria/eva/human EgoVerse).

## CONFIG_NAME

```bash
CONFIG_NAME=wam-cross-robot
```

## Datasets (15)

- agibot
- robocoin_agilex_cobot_magic_s26_a26
- robocoin_airbot_mmk2_s36_a36
- robocoin_realman_rmc_aida_l_s28_a28
- robocoin_agilex_decoupled_magic_s14_a14_fps50
- robocoin_agilex_decoupled_magic_s26_a26
- robocoin_aloha_s26_a26
- robocoin_alpha_bot_2_s28_a28
- robocoin_discover_aitbot_mmk2_s36_a36
- robocoin_realman_rmc_aidal_s28_a28
- robocoin_ruantong_a2d_s17_a17
- robocoin_unitree_g1_s28_a28 (3 cameras)
- robomind_agilex_cobot_magic_s14_a14
- piper2
- piper30

## DDP data loading

`rlds_partition_builders_by_rank=False`（默认）：**每个 rank 打开全部 builders**，数据经 `tfds.even_splits` 分片。不再把 builder 切分到不同 rank。

## ActionDiT init

Matches upstream FastWAM: load Video-DiT→ActionDiT linear-interp backbone from
`checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
(`action_encoder` / `head` stay random). Generate once if missing:

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints/fastwam"
uv run python scripts/preprocess_action_dit_backbone.py
```

Fine-tune configs with `skip_dit_load_from_pretrain=True` skip this file and load the full MoT from `pytorch_weight_path` instead.

## Video-loss diagnostics / overfit

Training logs now include float32 MSE diagnostics:

- `loss_video_raw` — mean MSE (no FM timestep weight)
- `loss_video_weighted` — mean(`MSE * w(t)`)
- `video_sigma` / `video_fm_weight` — batch-mean σ and `w(t)`

Overfit probe (`wam-cross-piper-overfit`): fixed first batch + fixed noise + σ=0.5, 500 steps, global batch 8.

```bash
cd /data/zjyang/baige-cluster
CONFIG_NAME=wam-cross-piper-overfit EXP_NAME=fw-piper-overfit-s05 \
  RLDS_DATA_DIR=/mnt/bos/bo23lu pixi run python atom0_fastwam_train_job.py
```

Piper train default global batch is **224** on 8 GPUs (`wam-cross-piper`).

## wam-cross-piper

Piper-only ablation: **piper2 + piper30**, same 80D mapping / norm assets as `wam-cross-robot`.

| 项 | 值 |
|---|---|
| data | piper2 + piper30 |
| `rlds_partition_builders_by_rank` | **False** — 每个 rank 打开两个 builders，episode 经 `tfds.even_splits` 分片（**不做 builder 平均分配**） |
| steps / warmup | 30k / 1.8k |
| global batch | 224（8 GPU） |
| in-train eval | 每 1000 step，rank 0 |

### W&B validation curves（与 π0.5 cotrain 同 key）

训练时 rank 0 跑 val，写入 wandb（`fastwam_eval.run_eval`）：

| Key | 含义 |
|---|---|
| `val/seen/piper2/flow_loss_fixed` | piper2 seen_test flow loss |
| `val/unseen/piper2/flow_loss_fixed` | piper2 unseen_test |
| `val/seen/piper30/action_mse` | piper30 seen action MSE（masked 80D） |
| `val/unseen/piper30/loss_video` | FastWAM 额外 video 分支 loss |
| `val/seen/agg/flow_loss_fixed` | seen 上 mixture 加权 aggregate |

每个数据集、seen/unseen 各一条曲线；仅 2 个 builder 时 `val_max_datasets=None`（全量 eval，不 subsample）。

```bash
cd /data/zjyang/baige-cluster
CONFIG_NAME=wam-cross-piper EXP_NAME=fw-wam-cross-piper-v2 \
  MODE=train INSTANCES=1 GPU_PER_NODE=8 BATCH_SIZE=224 \
  EVAL_INTERVAL=1000 RUN_ACTION_MSE=1 \
  ASSET_CONFIG_NAME=cotrain_real_robot_ego_fix \
  RLDS_DATA_DIR=/mnt/bos/bo23lu \
  pixi run python atom0_fastwam_train_job.py
```

本地 smoke（关 wandb / 少 step）：

```bash
cd /data/zjyang/Atom-0
MODE=smoke EVAL_INTERVAL=0 bash scripts/train_fastwam_baige.sh  # 需 CONFIG_NAME/EXP_NAME
```

---

## Norm stats

Reuse `assets/cotrain_real_robot_ego_fix/<dataset_uid>/` (same 80D mapping as ego_fix). All 15 datasets already have norm assets.

## Camera layout (`robot_wrist`)

Three cameras: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`.

Native ~480×640 JPEG stays **encoded** in the shuffle buffer. After shuffle, TF
`decode_std_images` JPEG-decodes then `resize_with_pad`s each camera to compose
targets **before batch** (cuts post-decode CPU RAM vs holding native frames).

Composed size is `model.image_resolution`:

| Config | Composed | Head | Each wrist |
|--------|----------|------|------------|
| `wam-cross-robot` / piper / ego | **288×256** (half of 576×512) | 192×256 | 96×128 |

```text
head (2H/3 × W)
─────────────────
left (H/3 × W/2) | right (H/3 × W/2)
```

Per-slot sizes from `fastwam_config.robot_wrist_slot_hw`; apply in
`resolve_train_image_resize_hw_by_slot` → RLDS decode, and
`fastwam_pytorch._compose_robot_wrist_video`.

### VAE input debug (offline)

Check composed frames without touching training:

```bash
PYTHONPATH=src uv run --group rlds scripts/debug_fastwam_vae_input.py wam-cross-robot \\
    --num-batches 8 --samples-per-batch 2 --output-dir tmp/fastwam_vae_input
```

PNG + metadata land under `tmp/fastwam_vae_input/<config>/<timestamp>/`.

## MoT cross-modal attention

Default FastWAMConfig (all wam-cross configs inherit):

| Direction | Default |
|-----------|---------|
| action→video | first frame only |
| video→action | **on**, `group_diagonal` (first latent frame free; later frames ↔ action groups) |

Disable / change via model fields: `mot_video_attends_to_action`, `mot_action_attends_to_video`, `mot_video_to_action_mode`. Keep `video_dit_config.action_conditioned=False`.

## Learning rate

- peak_lr = **1e-4**
- decay_lr = **1e-6**
- warmup_steps = 18k, decay_steps = 300k

## Baige submit example

```bash
CONFIG_NAME=wam-cross-robot \
EXP_NAME=fw-wam-cross-v1 \
MODE=train \
INSTANCES=2 GPU_PER_NODE=8 \
BATCH_SIZE=160 \
ASSET_CONFIG_NAME=cotrain_real_robot_ego_fix \
...
```

---

# wam-cross-robot-ego

Same 15 robot datasets as `wam-cross-robot`, plus 5 EgoVerse subsets (aria/eva/human + rl2 eva/human; 20 datasets total). `rlds_partition_builders_by_rank=True`（每 rank ~1–2 builders，省 host RAM）。

## CONFIG_NAME

```bash
CONFIG_NAME=wam-cross-robot-ego
```

## Added EgoVerse datasets

| dataset_id | source | native cameras | action | mask B/L/R |
|---|---|---|---|---|
| `egoverse_aria` | EgoVerse_full | `front_1` | 12D EE | T/F/F |
| `egoverse_eva` | EgoVerse_full | `front_1` + wrists | **14D** EE+gripper | T/T/T |
| `egoverse_human` | EgoVerse_full | `front_1` | 12D EE | T/F/F |
| `egoverse_rl2_eva` | EgoVerse_rl2 | `front_1` + wrists | **14D** EE+gripper | T/T/T |
| `egoverse_rl2_human` | EgoVerse_rl2 | `front_1` | 12D EE | T/F/F |

`egoverse_full` restructure maps `front_1` → `base_0_rgb`; missing wrists get a blank JPEG + `image_mask=False`. Eva (full/rl2) uses `egoverse_eva` restructure: concat `left/right_*_gripper` onto 12D EE → 14D. Same `robot_wrist` compose (576×512) as real robots.

Loss heads: ego video/action enabled (`lambda_ego_*` > 0); robot heads unchanged.

## Baige submit example

```bash
CONFIG_NAME=wam-cross-robot-ego \
EXP_NAME=fw-wam-cross-ego-v1 \
MODE=train \
INSTANCES=2 GPU_PER_NODE=8 \
BATCH_SIZE=160 \
ASSET_CONFIG_NAME=cotrain_real_robot_ego_fix \
...
```

---

# wam-cross-piper-ft

Fine-tune **piper2 + piper30** from an existing FastWAM checkpoint (model weights only; fresh optimizer / step=0).

## CONFIG_NAME

```bash
CONFIG_NAME=wam-cross-piper-ft
```

## Defaults

| 项 | 值 |
|---|---|
| data | piper2 + piper30 |
| `rlds_partition_builders_by_rank` | **False**（2 builders / 8 GPU） |
| peak_lr | `1e-5` |
| steps / warmup | 20k / 1k |
| init ckpt | `checkpoints/wam-cross-robot/fw-wam-cross-v3-8gpu-b208/20000` |

Override init with `PYTORCH_WEIGHT_PATH` / `INIT_CHECKPOINT`（step 目录、exp 目录或 `model.safetensors`）。

Same-exp resume（续训同一 `EXP_NAME`）用 `RESUME=1`，不要和 init path 同时用。

## Baige submit example

```bash
cd /data/zjyang/baige-cluster
CONFIG_NAME=wam-cross-piper-ft \
EXP_NAME=fw-wam-cross-piper-ft-v1-from-v3-20k \
MODE=train \
INSTANCES=1 GPU_PER_NODE=8 \
BATCH_SIZE=208 \
PEAK_LR=1e-5 \
NUM_TRAIN_STEPS=20000 WARMUP_STEPS=1000 \
INIT_CHECKPOINT=/data/zjyang/Atom-0/checkpoints/wam-cross-robot/fw-wam-cross-v3-8gpu-b208/20000 \
ASSET_CONFIG_NAME=cotrain_real_robot_ego_fix \
MEMORY_PER_NODE=1536 SHM_GIB=1280 CPU_PER_NODE=128 \
WANDB_ENABLED=1 \
.venv/bin/python atom0_fastwam_train_job.py
```
