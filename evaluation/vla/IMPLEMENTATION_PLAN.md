# Piper JAX Validation Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable offline validation evaluator for Piper JAX checkpoints 20000/25000/30000 without modifying the training repository.

**Architecture:** Keep all new files under `/path/to/Atom-0/evaluation/vla`. The Python evaluator imports the read-only OpenPI training repo, confirms config/data/checkpoint metadata, reads TFDS trajectories on CPU, uses OpenPI native transforms/policy/model restore, and writes CSV/JSON/Markdown/HTML/figures per checkpoint step.

**Tech Stack:** Python 3.11, TensorFlow/TFDS for CPU data reading, JAX/OpenPI/Orbax for model restore and native inference/loss, pandas/numpy/matplotlib for metrics and reports, bash launcher for reusable step selection.

---

### Task 1: Deterministic Anchor And Diagnostic Helpers

**Files:**
- Create: `tests/test_anchor_utils.py`
- Create/modify: `scripts/evaluate_validation.py`

- [x] Write failing tests for linspace anchors, no tail padding, exclusive target end, and normal/swapped arm MAE.
- [ ] Run pytest and observe failure because implementation is missing.
- [ ] Implement helper functions in `evaluate_validation.py`.
- [ ] Re-run pytest and observe pass.

### Task 2: Metadata And Validation CLI

**Files:**
- Create/modify: `scripts/evaluate_validation.py`
- Create: `config/eval_defaults.env`
- Output: `config/training_metadata_0629.json`

- [ ] Add all required CLI flags.
- [ ] Collect environment, git, wandb, config, checkpoint, norm stats, and TFDS metadata.
- [ ] Implement `--metadata-only` and `--validate-only`.

### Task 3: Evaluation Pipeline

**Files:**
- Modify: `scripts/evaluate_validation.py`
- Create: `scripts/generate_anchor_manifest.py`
- Create: `scripts/generate_report.py`
- Create: `scripts/run_validation.sh`

- [ ] Read or generate reusable anchor manifests for open-loop h8 and flow full horizon.
- [ ] Restore OpenPI native policy with `create_trained_policy`.
- [ ] Restore native JAX model for `compute_loss`.
- [ ] Evaluate per-anchor, per-episode, per-task metrics and bootstrap episode-level 95% CI.
- [ ] Save required CSV/JSON/figures/report artifacts.

### Task 4: Verification Sequence

**Files:**
- Outputs under `results/step_020000`

- [ ] Run `py_compile`.
- [ ] Run `--help`.
- [ ] Run `--metadata-only`.
- [ ] Run `--validate-only`.
- [ ] Run 1 episode x 1 anchor smoke test.
- [ ] Run 5 episode x 20 anchor benchmark.
- [ ] If smoke and benchmark pass, run full `seen_test` for checkpoint 20000 and generate reports.
