from openpi.training import config as _config
from openpi.policies import policy_config
from pathlib import Path


config = _config.get_config("pi05_droid")


checkpoint_dir = Path(
    "/mnt/data/weizhongxing/openpi_data/openpi-assets/checkpoints/pi05_base"
)


print("Config loaded")

policy = policy_config.create_trained_policy(
    config,
    checkpoint_dir
)

print("Policy loaded successfully")