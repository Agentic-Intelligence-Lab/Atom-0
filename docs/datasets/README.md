# Data sources and contracts

The manuscript's expanded recipe contains 334,054 training episodes and
approximately 2,659 hours after filtering an initial 3,033-hour pool.

| Source | Training episodes | Approximate hours |
| --- | ---: | ---: |
| In-house Piper | 5,829 | 25.5 |
| AgiBot World Beta | 21,837 | 314.0 |
| DROID | 64,124 | 343.0 |
| RoboCOIN | 93,352 | 655.7 |
| RoboMIND | 83,152 | 221.7 |
| EgoVerse | 64,464 | 1,079.5 |
| In-house aligned human–robot data | 1,296 | 20.0 |

These are corpus-level manuscript figures. An individual stage uses its own
subset and sampling recipe; inspect the chosen route's config for the actual
builders. Imported builder/statistics snapshots should not be assumed to have
the same full-corpus coverage merely because they share the recipe name.

## Input contract

Filtered episodes are represented in RLDS/TFDS. The manuscript describes 45
builders; source-level registries also retain optional and historical builders.
The common VLA interface uses:

- An 80-dimensional semantic state/action mapping, rather than native-vector padding.
- A 50-step target action chunk.
- One base and two wrist image slots, with masks for missing images.
- Action masks for unavailable dimensions and dataset-specific relative/absolute semantics.
- Per-dataset 1st/99th-percentile normalization computed after the same action mapping and chunk conversion used in training.

See the root [`action_space.py`](../../src/openpi/cotrain/action_space.py) and
the [Atom-DH counterpart](../../routes/atom_dh/src/openpi/cotrain/action_space.py).
Single-arm samples map to the right-arm slots. Atom-CL's aligned subset uses
the relative end-effector convention implemented by its converter and registry.
Its converter, builder version, and normalization must be used together.

## Assets and preparation

`assets/` contains small JSON statistics, mapping specifications, and chunk
metadata. For new or changed data, recompute statistics rather than reusing a
file with the same dataset name. Use the selected project's
`scripts/compute_cotrain_full_norm_stats_light.py --help` and config registry
to select the correct dataset mixture.

Raw datasets and pretrained weights are obtained separately under their
respective provider licenses. In-house data are not distributed here. The
quality-control description in the paper does not imply that every upstream
raw-data processing tool is included in this repository.
