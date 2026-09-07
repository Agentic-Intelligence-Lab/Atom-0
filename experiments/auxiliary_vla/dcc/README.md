# Diverse Context Conditioning

Main recipe: `pi05_dcc_libero` (KI + six-frame MEM + DCC).

Ablation configs:

- `pi05_dcc_libero_no_subgoal`
- `pi05_dcc_libero_no_metadata`
- `pi05_dcc_libero_no_dropout`

Core implementation:

- `src/openpi/models/model.py`
- `src/openpi/models/pi0.py`
- `src/openpi/models/pi0_config.py`
- `src/openpi/transforms.py`
- `tests/dcc/`
