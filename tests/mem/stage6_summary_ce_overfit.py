"""MEM Stage 6: long-term summary CE overfit probe.

This is a lightweight synthetic probe for the long-term memory objective. It checks
that a summary generator can learn a controlled mapping from hidden episode facts
to target memory summaries, while a blank/no-fact baseline cannot.
"""

import numpy as np


SUMMARIES = [
    "red block in left drawer",
    "red block in right drawer",
    "blue block in left drawer",
    "blue block in right drawer",
]


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def train_classifier(features, labels, steps=120, lr=0.8):
    weights = np.zeros((features.shape[1], len(SUMMARIES)), dtype=np.float64)
    losses = []
    for _ in range(steps):
        logits = features @ weights
        probs = _softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12))
        grad_logits = probs
        grad_logits[np.arange(len(labels)), labels] -= 1.0
        grad_logits /= len(labels)
        weights -= lr * (features.T @ grad_logits)
        losses.append(loss)
    pred = np.argmax(features @ weights, axis=-1)
    return losses, pred


def main():
    labels = np.arange(len(SUMMARIES), dtype=np.int64)
    fact_features = np.eye(len(SUMMARIES), dtype=np.float64)
    blank_features = np.ones((len(SUMMARIES), 1), dtype=np.float64)

    losses, pred = train_classifier(fact_features, labels)
    blank_losses, blank_pred = train_classifier(blank_features, labels)

    start = np.mean(losses[:10])
    end = np.mean(losses[-10:])
    acc = np.mean(pred == labels)
    blank_acc = np.mean(blank_pred == labels)

    assert end < start * 0.5, f"summary CE did not fall enough: {start:.4f} -> {end:.4f}"
    assert acc >= 0.9, f"summary key-field match too low: {acc:.2f}"
    assert blank_acc < 0.5, f"blank baseline recovered summaries too well: {blank_acc:.2f}"
    print(f"PASS: summary CE overfit loss {start:.4f} -> {end:.4f}, acc={acc:.2f}, blank_acc={blank_acc:.2f}")


if __name__ == "__main__":
    main()
