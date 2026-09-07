# Atom-0 new-host environment

The repository environment is installed in `.venv` with Python 3.11. The full
RLDS dependency group is included.

Load the host-local paths before running commands:

```bash
cd /path/to/Atom-0
source scripts/atom0_env.sh
```

The defaults are:

- RLDS root: `/path/to/rlds`
- OpenPI runtime cache: `/path/to/cache/openpi`
- Shared OpenPI model root: `/data/models/openpi`
- pi05 parameters on this host: `/data/models/openpi`
- Hugging Face cache: `/path/to/cache/huggingface`
- XDG/JAX cache: `/path/to/cache`
- Logs: `/path/to/Atom-0/logs`

Override any of `ATOM0_STATE_ROOT`, `RLDS_DATA_DIR`, `OPENPI_DATA_HOME`,
`OPENPI_MODEL_HOME`, `PARAMS_PATH`, `HF_HOME`, or `LOG_DIR` before sourcing the
script if storage is mounted elsewhere.

Verify the environment:

```bash
UV_CACHE_DIR=/path/to/cache/uv \
UV_PYTHON_INSTALL_DIR=/path/to/local/share/uv/python \
UV_LINK_MODE=copy \
uv sync --offline --group rlds

.venv/bin/python -c 'import jax, torch, tensorflow; print(jax.devices()); print(torch.cuda.device_count())'
.venv/bin/pytest -q tests/cotrain
```

Before training, the RLDS tree and the pi05 checkpoint must exist at the paths
printed by `source scripts/atom0_env.sh`. Set a fresh W&B key in the shell; do
not store it in this repository.

Download the pi05 base parameters into the path expected by the training
wrapper:

```bash
source scripts/atom0_env.sh
OPENPI_DATA_HOME="${OPENPI_MODEL_HOME}" .venv/bin/python - <<'PY'
from openpi.shared.download import maybe_download

path = maybe_download("gs://openpi-assets/checkpoints/pi05_base/params")
print(f"Downloaded pi05 parameters to: {path}")
PY

test -f "${PARAMS_PATH}/_CHECKPOINT_METADATA"
test -f "${PARAMS_PATH}/manifest.ocdbt"
```

The training container must also expose the NVIDIA device nodes (at minimum
`/dev/nvidiactl`, the assigned `/dev/nvidiaN`, and `/dev/nvidia-uvm`). If
`nvidia-smi` cannot communicate with the driver or `jax.devices()` only lists a
CPU, fix the container/Kubernetes GPU allocation before launching training; a
Python package reinstall cannot create that device assignment.

Eight-GPU training (the wrapper validates the checkpoint before starting):

```bash
source scripts/atom0_env.sh
export WANDB_API_KEY='...'
FSDP_DEVICES=8 BATCH_SIZE=256 NUM_TRAIN_STEPS=36000 \
EXP_NAME=cotrain_full_all_full_norm_8gpus \
bash scripts/train_cotrain_full_all_full_norm_local_weights.sh
```

For a 16-GPU, two-node job, keep `FSDP_DEVICES=8`, use a global batch size of
512, and provide the launcher variables (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`,
and `MASTER_PORT`). `scripts/train_cotrain.py` initializes JAX distributed mode
from those variables.
