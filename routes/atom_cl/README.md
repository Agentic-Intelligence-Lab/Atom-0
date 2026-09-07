# Atom-CL: progressive embodiment alignment

Run this route from the **repository root**, using its own `.venv`. The code is
in [`src/openpi`](../../src/openpi/) and [`scripts`](../../scripts/).

## Paper recipe

| Stage | Configuration | Initialization | Updates |
| --- | --- | --- | ---: |
| Ego pre-training | `egoscale_stage1_ego` | Base PaliGemma `.npz` | 100,000 |
| Human–robot alignment | `egoscale_stage2_aligned` | Stage 1 `<step>/params` | 50,000 |
| Robot pre-training | `egoscale_stage3_real_robot_fix` | Stage 2 `<step>/params` | 97,728 |
| Piper post-training | `egoscale_stage4_piper_finetune` | Stage 3 `<step>/params` | 20,000 |

Stage 2 trains the vision encoder and action expert while freezing the VLM
language component. The remaining paper stages train the full model. Aligned
human and robot demonstrations use matching task-space motion supervision;
subsequent robot pre-training returns to dataset-specific control targets.

The historical `egoscale_stage3_robot` name means **Piper-only adaptation**.
Use `egoscale_stage3_real_robot_fix` for the paper's broad robot pre-training.
The newer `egoscale_stage3_ego` configuration is a swapped-order experiment and
is retained separately. All three configuration names remain available.

## Prepare data and weights

```bash
# From the Atom-0 repository root, after installing the root environment.
export ATOM_RLDS_ROOT=/path/to/rlds
export ATOM_SELF_COLLECTED_ALIGNED_RLDS_ROOT=/path/to/aligned-rlds
export ASSETS_BASE_DIR="$PWD/assets"
export CHECKPOINT_BASE_DIR=/path/to/output/checkpoints
export PARAMS_PATH=/path/to/paligemma-pt-224.npz
```

Inspect the builder paths in [`config.py`](../../src/openpi/cotrain/config.py).
Use [`convert_self_collected_aligned_to_rlds.py`](../../scripts/convert_self_collected_aligned_to_rlds.py)
for the aligned data contract and [`compute_cotrain_full_norm_stats_light.py`](../../scripts/compute_cotrain_full_norm_stats_light.py)
for full normalization. Existing statistics are valid only for their original
builder, split, action mapping, and chunk semantics.

## Preflight and training

```bash
STAGE=stage1_ego PREFLIGHT_ONLY=1 bash scripts/run_egoscale_stage.sh

# A short data/model smoke run; this is not a paper reproduction run.
STAGE=stage1_ego EXP_NAME=atom_cl_smoke \
FSDP_DEVICES=1 BATCH_SIZE=2 NUM_TRAIN_STEPS=20 OVERWRITE=0 \
bash scripts/run_egoscale_stage.sh
```

For a full stage, set the stage, batch size, device count, update count, and
evaluation/checkpoint intervals explicitly. The general wrapper defaults to
a 20-update smoke run. For each subsequent stage, point `PARAMS_PATH` at the
actual preceding checkpoint's `params` directory, and use a new experiment name.

```bash
# Example: the paper's robot pre-training stage on the required hardware.
export PARAMS_PATH=/path/to/stage2/checkpoint/params
STAGE=stage3_real_robot_fix EXP_NAME=atom_cl_robot_pretrain \
FSDP_DEVICES=4 BATCH_SIZE=1536 NUM_TRAIN_STEPS=97728 \
SAVE_INTERVAL=10000 EVAL_INTERVAL=1000 OVERWRITE=0 \
bash scripts/run_egoscale_stage.sh

export PARAMS_PATH=/path/to/stage3/checkpoint/params
STAGE=stage4_piper_finetune EXP_NAME=atom_cl_piper_ft \
FSDP_DEVICES=4 BATCH_SIZE=512 NUM_TRAIN_STEPS=20000 \
SAVE_INTERVAL=5000 EVAL_INTERVAL=1000 OVERWRITE=0 \
bash scripts/run_egoscale_stage.sh
```

The reported pre-training hardware is 3 nodes × 8 B200 GPUs. Multi-node JAX
requires the corresponding distributed launch environment; setting the batch
size alone does not configure the cluster. The imported `*_dlc_*` launchers
retain platform-specific setup and must be configured for your infrastructure.

## Inference integration

The model's `sample_actions` implementation and policy/client infrastructure
are included. The upstream `scripts/serve_policy.py` resolves the upstream
training registry, which does not register the `egoscale_*` co-training names.
A Piper deployment must explicitly load the co-training configuration and apply
its per-dataset normalization and native/Unified80 mapping. A turnkey co-training
Piper server is not supplied by this release. The [remote inference guide](../../docs/remote_inference.md)
documents the underlying transport.
