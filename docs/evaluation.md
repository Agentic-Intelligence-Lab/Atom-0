# Evaluation

The following results and protocol are transcribed from the Atom-0 manuscript
snapshot read on 2026-09-07. Release checks do not rerun these experiments.

## Real-robot protocol

Each checkpoint is evaluated on Cook Bread, Hang Cup, Place Fruits, Place Plate,
Place Pot, Press Button, and Push Box. Every task has ten nominal ID trials and
ten task-matched OOD trials. Success requires the complete specified task end
state. The [main results table](../README.md#main-findings) reports counts as
well as percentages: 70 trials per condition and 140 when pooled.

| Task | Primary OOD change |
| --- | --- |
| Cook Bread | Pot rotated 45 degrees counterclockwise |
| Hang Cup | Increased cup mass |
| Place Fruits | Added visually similar non-target fruit |
| Place Plate | Translated rack |
| Place Pot | Pot rotated 60 degrees |
| Press Button | Reduced illumination |
| Push Box | Box rotated 90 degrees |

The manuscript still expresses some mass, illumination, displacement, timing,
and stability settings symbolically. Those numerical protocol settings must be
fixed before an independent reproduction. The reported comparison uses the
best measured checkpoint from each route, not an average over training seeds.

## Language-conditioned representation analysis

A separate seven-task alignment-video set contains 62 ego and 62 robot clips.
Each clip contributes 32 sampled front-view frames and its task instruction;
features are pooled using temporal-change weights. A 7×7 task matrix averages
cross-domain clip cosine similarities. The raw diagonal measures matched-task
consistency, while the row-wise min–max-normalized off-diagonal mean measures
relative different-task response.

| Method | Raw matched-task mean ↑ | Normalized different-task mean ↓ |
| --- | ---: | ---: |
| Atom-DH | 0.999625 | 0.592719 |
| Atom-CL | 0.994481 | 0.387240 |
| Atom-WAM | 0.997357 | 0.480727 |

Same-task clips receive the same instruction. These are language-conditioned
diagnostics, and the VLA and WAM feature extractors differ. Row normalization
also removes the absolute similarity scale. These quantities therefore have
a different interpretation from real-robot Success/Trials.

The end-to-end representation evaluation package and raw trial records are
not part of this training-code synchronization. Existing `evaluation/vla`
directories retain earlier offline action-evaluation tools.
