# Paper alignment audit — 2026-09-07

The current manuscript is the specification for the release. Original NAS
worktrees were left unchanged. A complete code-only pre-edit copy was archived
on NAS before the following implementation changes. No data or model weights
were moved, deleted, loaded or trained during these changes.

| Paper contract | Source code | Release change |
| --- | --- | --- |
| WAM outputs B×50×80 | B×32×80 | Unified WAM presets use horizon 50 |
| WAM takes current images and language | Adds an 80D proprio token | Disable proprio encoder and bypass state in training/inference |
| Joint loss is expectation under the sampled ego/robot mixture | Adds separate domain means | Average each masked contribution over the full batch |
| No future information reaches action branch | First-frame attention mask already present | Retained; tested for direct and multi-layer leakage |
| No test-time future-video generation | Action-only cached inference already present | Retained; test forbids VAE decode |
| Same visual preprocessing at training and deployment | Wrapper inference omits configured resolution | Pass the configured resolution during inference |
| Post-training starts from a compatible pre-trained model | Default path points to a historical run | Require explicit initialization or resume when pretrained loading is skipped |
| Normalize matching 50-step relative targets | Historical stats have incomplete horizon provenance | New, initially empty `atom_wam_paper50` stats namespace |

Atom-CL's retained paper stages are 100k ego, 50k alignment, 97,728 robot
pre-training and 20k Piper post-training. Its language-freeze and aligned
relative-SE(3) paths are preserved. Atom-DH's separate domain projections and
OT implementation are also preserved. No unrelated model refactor was made.

## Unresolved specification details

- The manuscript says 45 builders, but the WAM source mixture has 47 entries.
  The exact final builder IDs/weights are not given; no datasets were dropped
  solely to make the count match. The robot-only source preset has 15 datasets.
- The WAM appendix does not provide a complete run schedule or checkpoint
  provenance. Source defaults are not silently presented as paper settings.
- Newly selected 50-step normalization must be computed from the final train
  corpus. Historical JSON files are retained as provenance, not certified
  replacement statistics.
- Existing 32-step/proprio-conditioned WAM checkpoints need their original
  implementation. A model trained with this modified interface is a new run,
  not the checkpoint behind the paper's current result table.

## Verification boundary

The CPU contract tests exercise tiny real Video/Action DiTs, a test-only latent
encoder, actual forward/backward and action denoising, and wrapper boundaries.
They are not full 5B/VAE acceptance, dataset validation or robot evaluation.
Full training and real-robot success were not rerun. The Overleaf manuscript
was read, not edited. GitHub visibility was not changed.
