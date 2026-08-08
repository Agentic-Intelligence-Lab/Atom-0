import numpy as np
import pytest
import torch

from openpi.models_pytorch.fastwam.wan22 import fastwam as fw


def test_action_mask_broadcasts_across_horizon() -> None:
    mask = torch.tensor([[True, False, True], [False, True, False]])
    broadcast = fw._broadcast_action_mask(mask, (2, 4, 3))
    assert broadcast.shape == (2, 4, 3)
    np.testing.assert_array_equal(broadcast[:, 0].cpu().numpy(), mask.cpu().numpy())
    np.testing.assert_array_equal(broadcast[:, -1].cpu().numpy(), mask.cpu().numpy())


def test_missing_action_mask_returns_none() -> None:
    assert fw._broadcast_action_mask(None, (2, 4, 3)) is None


def test_action_mask_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="`action_mask` shape must be"):
        fw._broadcast_action_mask(torch.ones((2, 2), dtype=torch.bool), (2, 4, 3))


def test_masked_action_loss_ignores_inactive_dims() -> None:
    """Inactive dims with huge target error should not change the per-token loss."""
    pred = torch.zeros(1, 2, 5)
    target_clean = torch.zeros(1, 2, 5)
    target_dirty = target_clean.clone()
    target_dirty[..., 1] = 1000.0
    target_dirty[..., 3:] = -1000.0

    mask = torch.tensor([[True, False, True, False, False]])
    dim_mask = fw._broadcast_action_mask(mask, tuple(target_clean.shape))

    def masked_mean_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        sq = (a - b) ** 2
        sq = sq * dim_mask
        denom = dim_mask.sum(dim=-1).clamp(min=1.0)
        return sq.sum(dim=-1) / denom

    clean = masked_mean_sq(pred, target_clean)
    dirty = masked_mean_sq(pred, target_dirty)
    torch.testing.assert_close(clean, dirty)
