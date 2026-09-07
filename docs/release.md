# Release notes — 2026-09-07

## Canonical repository and route names

The publication repository is
[Agentic-Intelligence-Lab/Atom-0](https://github.com/Agentic-Intelligence-Lab/Atom-0).
Its previous main commit `ddb80a4944c2bf7e18c1f01668fb3917dfd7067d` is retained on
`codex/backup-main-before-paper-release-20260907` and in main's ancestry.
The organization-only `src/openpi/cotrain/config-fps30.py` is retained as a
historical configuration, not substituted for the paper presets.

Directories use technical roles: `experiments/atom_cl`,
`experiments/auxiliary_vla`, `evaluation/vla`, and `evaluation/baseline_vla`.
Auxiliary method plans live in `docs/reproduction/auxiliary_vla`. Internal
model/config identifiers and source-paper attribution remain unchanged.

## Source provenance

The release was assembled from three L20Y working trees and the existing GitHub
repository. Source commit identifiers identify each tree's Git base; imported
working files may also include uncommitted changes.

| Component | Source base |
| --- | --- |
| Atom-CL root project | `ebaf24fdac58171f382c86ae276a8daba8d7e925` |
| Atom-DH project, including OT | `041236fb8702f935010776f45820a0ea1e147c90` |
| Atom-WAM integration before paper-alignment edits | `98179217e9fe1950334069bfe6a1b44768fb0f67` |
| Existing repository before synchronization | `7bb13762a54e6371f3107ce2943ad7c51f744823` |

The Atom-CL import retains the existing paper configuration
`egoscale_stage3_real_robot_fix` alongside the newer `egoscale_stage3_ego`
swapped-order experiment. Training code, conversion scripts, tests, client
package, and small normalization metadata are included. Private host paths and
personal workspace labels were replaced in the distributed working files.

Atom-DH is isolated under `routes/atom_dh` because its `openpi` model/data code
differs from the root project. Installing both into the same environment would
make imports depend on installation order.

The root and route guides were rewritten against the current Atom-0 manuscript.
Its title and route labels are used in the main README. Corpus size uses the
filtered data table (approximately 2,659 hours), rather than the rounded
3,000-hour statement elsewhere in the draft. Result tables retain the draft's
Success/Trials denominators. No public paper URL or checkpoint release is
claimed.

## Validation scope

Release checks cover Python syntax, shell syntax, JSON metadata parsing,
documentation links, action-space contracts, and screening for accidental
weights, data, credentials, and personal workspace names. The provided
training scripts still require Linux/CUDA, route-specific dependencies, datasets,
normalization assets, and weights. This synchronization does not rerun model
training or real-robot evaluation.

The root action-space suite passed 80 tests (2 TensorFlow-dependent tests
skipped), and the Atom-DH CPU suite passed 3 tests covering action masks,
trajectory matching, padded OT supports, and finite gradients. These checks
do not validate a full VLA forward/backward pass or distributed GPU training.

Atom-WAM passed 16 CPU tests using PyTorch 2.7.1: the five new paper-interface
tests plus existing action-mask, camera-composition and VAE-input diagnostics.
The new tests include tiny real DiT forward/backward and action denoising with
a test-only latent encoder. TensorFlow-dependent WAM configuration/data tests
were not run in this CPU environment; no full model weights were loaded.

After directory renaming, the offline VLA and baseline-VLA suites passed 4 and
7 tests respectively in separate processes. The relocated baseline test's
repository-root lookup was updated to match the new directory depth.

## Remaining release work

- Atom-WAM now includes the paper-interface changes described in [the alignment audit](paper_alignment.md); new normalization, exact corpus selection and full-model validation remain outstanding.
- Co-training Piper policy serving requires an explicit dataset/normalization adapter; the upstream server does not accept co-training config names directly.
- The original cloud launchers contain infrastructure assumptions; use the route guides and set paths and hardware parameters for your environment.
- Some exact real-robot protocol parameters and public paper/weight links are not yet provided.

Prior Git history is preserved. Cleaning author names out of historical Git
objects would require a separate history rewrite; the current working files
and new commit metadata use project-level naming.
