"""MEM Stage 7: memory-affects-action counterfactual probe.

The current observation and task are identical across samples; only the memory
summary changes. A memory-conditioned linear policy should fit opposite actions,
while an observation-only baseline should fail.
"""

import numpy as np


def _train_linear(features, targets, steps=160, lr=0.3):
    weights = np.zeros((features.shape[1], targets.shape[1]), dtype=np.float64)
    losses = []
    for _ in range(steps):
        pred = features @ weights
        err = pred - targets
        losses.append(float(np.mean(err**2)))
        weights -= lr * (features.T @ err) / len(features)
    return weights, losses


def main():
    # Same current observation; two counterfactual memory summaries.
    obs = np.zeros((2, 1), dtype=np.float64)
    memory = np.array([[-1.0], [1.0]], dtype=np.float64)  # left vs right
    targets = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    full_features = np.concatenate([obs, memory], axis=-1)
    blank_features = obs + 1.0  # same feature for both samples

    weights, losses = _train_linear(full_features, targets)
    blank_weights, blank_losses = _train_linear(blank_features, targets)

    pred = full_features @ weights
    same_obs_diff = float(np.linalg.norm(pred[0] - pred[1]))
    repeated_diff = float(np.linalg.norm(pred[0] - pred[0]))
    blank_pred = blank_features @ blank_weights

    assert losses[-1] < 0.05, f"memory-conditioned policy did not overfit: {losses[-1]:.4f}"
    assert blank_losses[-1] > losses[-1] * 5, "blank-memory baseline did not degrade enough"
    assert same_obs_diff > repeated_diff + 1.0, "counterfactual memories did not produce distinct actions"
    assert float(np.mean((blank_pred - targets) ** 2)) > 0.2
    print(
        "PASS: memory changes action "
        f"loss={losses[0]:.4f}->{losses[-1]:.4f}, blank_loss={blank_losses[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
