"""Frozen vision / language encoders used by HPT stems."""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("openpi")


def _from_pretrained_kwargs() -> dict:
    """Respect Baige offline HF cache (PFS hub/ dir)."""
    kwargs: dict = {}
    cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not cache and os.environ.get("HF_HOME"):
        cache = os.path.join(os.environ["HF_HOME"], "hub")
    if cache and os.path.isdir(cache):
        kwargs["cache_dir"] = cache
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        kwargs["local_files_only"] = True
    return kwargs


class FrozenDINOv2(nn.Module):
    """DINOv2 patch tokens (+ CLS) for image stems and world targets."""

    def __init__(self, model_id: str = "facebook/dinov2-base", device: str = "cuda"):
        super().__init__()
        from transformers import AutoImageProcessor, Dinov2Model

        hf_kwargs = _from_pretrained_kwargs()
        self.processor = AutoImageProcessor.from_pretrained(model_id, **hf_kwargs)
        self.model = Dinov2Model.from_pretrained(model_id, **hf_kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_size = int(self.model.config.hidden_size)
        self._device = device

    def train(self, mode: bool = True):
        # Keep encoder in eval forever.
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: float tensor [B, H, W, 3] in [-1, 1] or [0, 1] or [0, 255]
        Returns:
            tokens: [B, 1+N, D]
        """
        x = images.detach()
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 2.0:
            x = x / 255.0
        elif x.min() < -0.1:
            x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        # DINOv2 expects [B, 3, H, W] ImageNet-normalized.
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        out = self.model(pixel_values=x)
        return out.last_hidden_state


class FrozenT5Encoder(nn.Module):
    """T5 encoder → per-token language features for the language stem."""

    def __init__(self, model_id: str = "t5-base", max_length: int = 32, device: str = "cuda"):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer

        hf_kwargs = _from_pretrained_kwargs()
        self.tokenizer = T5Tokenizer.from_pretrained(model_id, **hf_kwargs)
        self.model = T5EncoderModel.from_pretrained(model_id, **hf_kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.max_length = max_length
        self.hidden_size = int(self.model.config.d_model)
        self._device = device

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        # Zero out padded positions for a cleaner stem input.
        mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
        return out.last_hidden_state * mask
