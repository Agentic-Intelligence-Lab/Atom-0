# Atom-0

### Exploring Ego–Robot Integration for Embodied Foundation Model Pretraining

Atom-0 studies how egocentric human experience can improve robot policies during pre-training. We compare three ways to bridge the human–robot embodiment gap: domain-specific action heads, progressive embodiment alignment, and joint world–action modeling.

This code release includes **Atom-DH**, **Atom-CL**, and **Atom-WAM**, with robot-only baseline recipes. The WAM code has been adapted to the manuscript's 50×80, image-and-language-only interface; it has not been retrained. See the [paper alignment audit](docs/paper_alignment.md) for remaining corpus and reproducibility gaps.

## Research routes

| Route | Method | Code in this release |
| --- | --- | --- |
| **Atom-DH** | Joint ego–robot co-training with a shared backbone and separate ego/robot action projections; optional trajectory-aware OT alignment | [Code and training guide](routes/atom_dh/README.md) |
| **Atom-CL** | Ego pre-training → human–robot embodiment alignment → robot pre-training → Piper post-training | [Training guide](routes/atom_cl/README.md); implementation in the repository root |
| **Atom-WAM** | Fast-WAM-based joint future-video and action modeling | [Code and training guide](routes/atom_wam/README.md) |
| **Robot-only baselines** | Robot-only pre-training followed by downstream adaptation | [VLA and WAM baseline guide](baselines/README.md) |

The VLA routes use the π₀.₅ architecture with base **PaliGemma VLM initialization** for the paper experiments. Loading a pretrained π₀.₅ policy is supported by the underlying code, but changes the initialization protocol.

## Main findings

The current manuscript reports that progressive alignment gives the strongest real-robot results. Both VLA routes improve OOD performance over the VLA robot-only baseline. Direct ego–robot WAM co-training underperforms the WAM robot-only baseline in this evaluation.

Seven tasks are evaluated with **10 ID and 10 OOD trials per task**, giving 70 ID, 70 OOD, and 140 combined trials per checkpoint. Each OOD setting preserves the task goal and changes a spatial, appearance, or physical factor.

| Family | Method | ID success | OOD success | Combined success |
| --- | --- | ---: | ---: | ---: |
| VLA | Robot-only baseline | 42.86% (30/70) | 22.86% (16/70) | 32.86% (46/140) |
| VLA | Atom-DH | 44.29% (31/70) | **42.86% (30/70)** | 43.57% (61/140) |
| VLA | **Atom-CL** | **60.00% (42/70)** | **42.86% (30/70)** | **51.43% (72/140)** |
| WAM | Robot-only baseline | 28.57% (20/70) | 28.57% (20/70) | 28.57% (40/140) |
| WAM | Atom-WAM | 20.00% (14/70) | 21.43% (15/70) | 20.71% (29/140) |

These are manuscript-reported results for the best measured checkpoint from each route, synchronized on **2026-09-07**. They are not results of the release validation checks or the newly modified WAM code. See [evaluation protocol and representation analysis](docs/evaluation.md).

## Repository layout

```text
Atom-0/
├── routes/
│   ├── atom_dh/          # Independent OpenPI project: dual heads and OT
│   ├── atom_cl/          # Guide to the root progressive-training project
│   └── atom_wam/         # Independent PyTorch/OpenPI world–action project
├── baselines/           # Robot-only baseline recipes and release status
├── evaluation/          # VLA and robot-only baseline offline evaluators
├── src/openpi/          # Atom-CL models, data loading, training and policies
├── scripts/             # Atom-CL conversion, normalization, training and serving
├── packages/            # Policy client
├── assets/              # Small normalization statistics and action-space metadata
├── tests/               # Root-project tests
├── docs/                # Data contracts, evaluation and release notes
├── experiments/         # Atom-CL stages and auxiliary VLA experiments
└── third_party/         # Upstream integrations
```

Atom-DH, Atom-CL and Atom-WAM contain different versions of the `openpi` package. **Use separate Python environments and run commands from the route's project directory.** The root layout is retained so existing Atom-CL scripts continue to work. Supporting KI/MEM/DCC experiments are grouped under [`experiments/auxiliary_vla`](experiments/auxiliary_vla/README.md); they are not additional paper routes. Public navigation uses method names, while internal model/config identifiers remain compatible with existing checkpoints.

## Installation

Use Linux, Python 3.11, and an NVIDIA GPU for model training. The projects pin JAX/Flax and other dependencies in their respective `pyproject.toml` and `uv.lock` files.

```bash
git clone --recurse-submodules https://github.com/Agentic-Intelligence-Lab/Atom-0.git
cd Atom-0

# Atom-CL / root project
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 --group rlds
```

For Atom-DH, create its environment from its own directory:

```bash
cd routes/atom_dh
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 --group rlds
```

Choose one route guide before launching training. It specifies dataset roots, normalization assets, initialization weights, and the correct configuration names. Dataset conversion and full normalization must match the selected recipe.

## Data and control interface

The manuscript's curated recipe contains **334,054 episodes, approximately 2,659 hours**, from robot demonstrations, EgoVerse, and task-matched human–robot alignment demonstrations. The initial pool is approximately 3,033 hours before filtering. The [data guide](docs/datasets/README.md) describes the sources and input contract.

Both VLA routes use a semantic **80-dimensional state/action space**, a **50-step action horizon**, three canonical image slots, per-dataset normalization, and masks for unavailable action dimensions and views. Alignment-stage motion representations require the matching converter and statistics; they must not be interchanged with ordinary joint-space targets.

Raw datasets, videos, model weights, optimizer state, training logs, and credentials are not distributed in this repository. The JSON files under `assets/` contain small normalization and schema metadata. Obtain pretrained weights from their original providers under the applicable terms.

## Training and inference

- [Atom-CL: four-stage execution of the progressive route](routes/atom_cl/README.md)
- [Atom-DH: dual heads, OT ablation, and Piper fine-tuning](routes/atom_dh/README.md)
- [Atom-WAM: paper-aligned world–action modeling](routes/atom_wam/README.md)
- [Robot-only VLA and WAM baselines](baselines/README.md)
- [Remote policy serving](docs/remote_inference.md)
- [Release provenance and validation scope](docs/release.md)

The paper describes **three pre-training stages** for Atom-CL. The repository additionally names downstream Piper fine-tuning **Stage 4**. Checkpoint directories are zero-indexed: for example, a completed 20,000-update run can end at `19999`.

## Attribution and citation

Atom-0 is developed at the **Agentic Intelligence Lab** and builds on [Physical Intelligence's OpenPI](https://github.com/Physical-Intelligence/openpi). See [upstream provenance](UPSTREAM.md), [contributors](CONTRIBUTORS.md), and [CITATION.cff](CITATION.cff).

Code is distributed under [Apache-2.0](LICENSE), subject to retained third-party notices. Gemma-related terms are retained in [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt); model and dataset licenses remain separate from the code license. A public paper link will be added when available.
