# Atom-WAM: joint world–action modeling

This independent OpenPI/PyTorch project contains the NAS Atom-0 FastWAM
integration, modified to follow the manuscript's explicit interface. It is not
the native-14D RealWorld or RoboMemArena experiment from Atom-0.5.

## Paper-aligned implementation

- Normalized **50 × 80** output actions.
- Current three-camera images and language condition the model; **no learned
  proprioception input**. State remains outside the network for relative
  training targets and recovery of absolute robot commands.
- Video DiT / Action DiT joint flow matching. Action tokens see only the clean
  current-frame anchor, not noisy future-video tokens.
- Domain loss contributions use the whole-batch denominator, preserving the
  effective ego sampling probability. The retained video coefficient is 0.6.
- Inference denoises actions with cached current visual context, without
  future-video generation or decoding.

Nine training video frames are sampled across the 50-step horizon at offsets
`[0, 6, 12, 19, 25, 31, 38, 44, 50]`. Wan compression produces three latent
frames; each of the two future groups aligns with 25 action tokens. The frame
count and sampling choice are implementation details, not manuscript numbers.

**These are code changes, not retrained results.** The manuscript's result table
does not validate this version. The original integration used 32 actions and
an 80D proprio encoder; its checkpoints and statistics must not be silently
treated as compatible with this modified model.

## Setup and weights

Use a separate Linux/CUDA environment, not the Atom-CL or Atom-DH environment.

```bash
cd routes/atom_wam
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 --group rlds
export RLDS_DATA_DIR=/path/to/rlds
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/wan-models
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export DIFFSYNTH_SKIP_DOWNLOAD=true
```

Supply Wan2.2-TI2V-5B Video DiT/VAE and Wan UMT5 encoder/tokenizer weights under
the layout in [`helpers/loader.py`](src/openpi/models_pytorch/fastwam/wan22/helpers/loader.py).
No weights are included. Generate ActionDiT's interpolation initialization:

```bash
uv run python scripts/preprocess_action_dit_backbone.py \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cpu --dtype bfloat16
```

## Data and normalization

The retained mixture registry has **47 entries**; the manuscript states
**45 builders** without listing their exact IDs. No two datasets were removed
by guesswork. Resolve that selection before claiming exact corpus reproduction.
The source robot-only preset has 15 datasets and is likewise not certified as
the final manuscript baseline corpus.

Paper-aligned presets use a new `assets/atom_wam_paper50` namespace. Old metadata
in `assets/cotrain_real_robot_ego_fix` is preserved only for provenance, not used
as the new preset. Recompute from the selected training splits:

```bash
uv run --group rlds python scripts/compute_cotrain_full_norm_stats_light.py \
  --config-name wam-cross-fix --rlds-data-dir "$RLDS_DATA_DIR" \
  --output-assets-dir "$PWD/assets/atom_wam_paper50"
```

This is a full split scan, not a small preflight. Resolve corpus selection first.
The regular normalization script's default frame cap is not a substitute.

## Training

| Config | Role |
| --- | --- |
| `wam-cross-fix` | Ego–robot joint pre-training |
| `wam-cross-robot` | Robot-only WAM baseline pre-training |
| `wam-cross-piper-ft` | Piper post-training, with frozen Video DiT |

```bash
# Included recipe example, not certified paper run hyperparameters.
CONFIG_NAME=wam-cross-fix EXP_NAME=atom_wam_paper50 \
NPROC_PER_NODE=8 WORLD_SIZE=1 RANK=0 BATCH_SIZE=160 \
WANDB_ENABLED=0 bash scripts/train_fastwam_baige.sh

# Use a compatible checkpoint produced by THIS implementation.
CONFIG_NAME=wam-cross-piper-ft EXP_NAME=atom_wam_paper50_piper \
INIT_CHECKPOINT=/path/to/paper50/checkpoint \
NPROC_PER_NODE=8 WORLD_SIZE=1 RANK=0 BATCH_SIZE=208 \
WANDB_ENABLED=0 bash scripts/train_fastwam_baige.sh
```

`BATCH_SIZE` is global; `WORLD_SIZE` is the number of nodes in this launcher.
Post-training requires an explicit matching initialization checkpoint (or a
resume checkpoint); the historical experiment path is no longer a default.
Preset defaults are 300k pre-training and 20k post-training updates. The paper
does not specify the full WAM run schedule, so these defaults are not a claim
about the checkpoint behind its result table.

`sample_actions`, offline evaluation and co-training-aware policy serving are
included. Robot clients still need the correct dataset ID, mapping, statistics
and hardware safety checks. This release starts no robot or server.

## Tests and attribution

```bash
PYTHONPATH=src python -m pytest tests/test_paper_contract.py \
  tests/cotrain/test_fastwam_action_mask.py \
  tests/cotrain/test_fastwam_robot_wrist.py \
  tests/cotrain/test_fastwam_vae_input_debug.py -q -o addopts=''
```

Tests use tiny real DiTs and a test-only latent encoder: they check 50-step
forward/backward, sampling-weighted loss, future-information isolation,
action-only inference, state independence and matching image resolution.
They do not validate full pretrained models, the real VAE or robot success.

Based on [Fast-WAM](https://github.com/yuantianyuan01/FastWAM), with its
[MIT notice](third_party/fastwam/LICENSE) retained. The OpenPI integration uses
[Apache-2.0](LICENSE), subject to upstream notices.
