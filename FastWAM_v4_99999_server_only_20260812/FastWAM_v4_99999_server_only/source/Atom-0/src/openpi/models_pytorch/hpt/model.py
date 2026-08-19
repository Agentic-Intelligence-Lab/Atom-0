"""HPT model: multi-stem → shared trunk → action (flow) + world (DINO) heads."""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from openpi.models.hpt_config import HPTConfig
from openpi.models_pytorch.hpt.encoders import FrozenDINOv2, FrozenT5Encoder
from openpi.models_pytorch.hpt.modules import (
    FlowMatchingActionHead,
    Stem,
    TransformerTrunk,
    WorldHead,
)

logger = logging.getLogger("openpi")


def _sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor:
    dist = torch.distributions.Beta(
        torch.tensor(alpha, device=device, dtype=torch.float32),
        torch.tensor(beta, device=device, dtype=torch.float32),
    )
    return dist.sample((bsize,))


class HPTModel(nn.Module):
    """Heterogeneous Pre-trained Transformer for Atom-0 co-training."""

    def __init__(self, config: HPTConfig, *, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device_str = device
        d = config.embed_dim

        self.dino: FrozenDINOv2 | None = None
        self.t5: FrozenT5Encoder | None = None
        dino_dim = config.dino_dim
        t5_dim = config.t5_dim
        if config.load_encoders:
            self.dino = FrozenDINOv2(config.image_encoder, device=device)
            self.t5 = FrozenT5Encoder(config.language_encoder, max_length=config.max_token_len, device=device)
            dino_dim = self.dino.hidden_size
            t5_dim = self.t5.hidden_size

        # Domain / modality stems (HPT-style).
        self.ego_stem = Stem(dino_dim, d, config.ego_tokens)
        self.robot_stem = Stem(dino_dim, d, config.robot_tokens)
        self.wrist_stem = Stem(dino_dim, d, config.wrist_tokens)
        self.state_stem = Stem(config.proprio_dim, d, config.state_tokens, widths=[128, 128])
        self.language_stem = Stem(t5_dim, d, config.language_tokens)

        self.action_tokens = nn.Parameter(torch.randn(1, config.num_action_tokens, d) * 0.02)
        self.future_tokens = nn.Parameter(torch.randn(1, config.num_future_tokens, d) * 0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.trunk = TransformerTrunk(
            embed_dim=d,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            drop_path=config.drop_path,
        )

        self.action_head = FlowMatchingActionHead(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            cond_dim=d,
        )
        self.world_head = WorldHead(cond_dim=d, dino_dim=dino_dim)

        if not config.load_encoders:
            # Deterministic smoke projection when DINOv2 is not loaded.
            self.register_buffer("_smoke_img_proj", torch.randn(3, dino_dim) * 0.02, persistent=False)

        if config.pretrained_trunk_path:
            self.load_pretrained_trunk(config.pretrained_trunk_path)

        if config.train_mode == "finetune":
            self.freeze_trunk()

    # ------------------------------------------------------------------ utils
    def freeze_encoders(self) -> None:
        for enc in (self.dino, self.t5):
            if enc is None:
                continue
            for p in enc.parameters():
                p.requires_grad = False
            enc.eval()

    def freeze_trunk(self) -> None:
        for p in self.trunk.parameters():
            p.requires_grad = False
        self.action_tokens.requires_grad = False
        self.future_tokens.requires_grad = False
        self.pos_embed.requires_grad = False
        logger.info("HPT finetune mode: trunk + query tokens frozen")

    def unfreeze_trunk(self) -> None:
        for p in self.trunk.parameters():
            p.requires_grad = True
        self.action_tokens.requires_grad = True
        self.future_tokens.requires_grad = True
        self.pos_embed.requires_grad = True

    def load_pretrained_trunk(self, path: str) -> None:
        """Load a liruiw/HPT-style ``trunk.pth`` if shapes match."""
        import os

        ckpt = path
        if os.path.isdir(path):
            ckpt = os.path.join(path, "trunk.pth")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.trunk.load_state_dict(state, strict=False)
        logger.info(
            "Loaded HPT trunk from %s (missing=%s unexpected=%s)",
            ckpt,
            len(missing),
            len(unexpected),
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # ---------------------------------------------------------------- encode
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.dino is None:
            # Smoke / offline path without downloading DINOv2: random-projection patches.
            x = F.interpolate(image.permute(0, 3, 1, 2).float(), size=(16, 16), mode="bilinear")
            tokens = x.flatten(2).transpose(1, 2)  # [B, 256, 3]
            return tokens @ self._smoke_img_proj.to(device=tokens.device, dtype=tokens.dtype)
        return self.dino(image)

    def _encode_language(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        if self.t5 is None:
            return torch.zeros(len(prompts), self.config.max_token_len, self.config.t5_dim, device=device)
        return self.t5(prompts, device=device)

    def _maybe_gate(self, tokens: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Zero inactive rows; optionally stop grad into inactive domain stems."""
        # active: [B] bool
        mask = active.to(dtype=tokens.dtype).view(-1, 1, 1)
        if self.config.domain_stem_grad_gate:
            # Inactive samples: detach so stem grads don't flow from the other domain.
            tokens = tokens * mask + tokens.detach() * (1.0 - mask)
            return tokens * mask
        return tokens * mask

    def encode_obs(
        self,
        *,
        base_img: torch.Tensor,
        left_wrist: torch.Tensor | None,
        right_wrist: torch.Tensor | None,
        state: torch.Tensor,
        prompts: list[str],
        is_ego: torch.Tensor,
        image_masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Build observation tokens for a mixed ego/robot batch."""
        b = base_img.shape[0]
        device = base_img.device
        is_ego = is_ego.to(device=device, dtype=torch.bool).reshape(-1)
        is_robot = ~is_ego

        base_feat = self._encode_image(base_img)
        ego_tok = self._maybe_gate(self.ego_stem(base_feat), is_ego)
        robot_tok = self._maybe_gate(self.robot_stem(base_feat), is_robot)

        # Wrist cameras (robot only). Missing masks → zeros.
        if left_wrist is None:
            left_wrist = torch.zeros_like(base_img)
        if right_wrist is None:
            right_wrist = torch.zeros_like(base_img)
        left_feat = self._encode_image(left_wrist)
        right_feat = self._encode_image(right_wrist)
        wrist_feat = torch.cat([left_feat, right_feat], dim=1)
        wrist_tok = self._maybe_gate(self.wrist_stem(wrist_feat), is_robot)

        if image_masks is not None:
            left_m = image_masks.get("left_wrist_0_rgb")
            right_m = image_masks.get("right_wrist_0_rgb")
            if left_m is not None and right_m is not None:
                wrist_valid = (left_m.to(device).reshape(-1) | right_m.to(device).reshape(-1)) & is_robot
                wrist_tok = wrist_tok * wrist_valid.to(wrist_tok.dtype).view(-1, 1, 1)

        state_tok = self.state_stem(state.float())
        lang_feat = self._encode_language(prompts, device)
        lang_tok = self.language_stem(lang_feat)

        obs_tokens = torch.cat([ego_tok, robot_tok, wrist_tok, state_tok, lang_tok], dim=1)
        aux = {
            "is_ego": is_ego,
            "is_robot": is_robot,
            "base_feat": base_feat,
        }
        return obs_tokens, aux

    def forward_trunk(self, obs_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = obs_tokens.shape[0]
        action_q = self.action_tokens.expand(b, -1, -1)
        future_q = self.future_tokens.expand(b, -1, -1)
        tokens = torch.cat([obs_tokens, action_q, future_q], dim=1)
        # Add positional embedding (truncate / tile as needed).
        n = tokens.shape[1]
        if n <= self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed[:, :n]
        else:
            pe = self.pos_embed.repeat(1, math.ceil(n / self.pos_embed.shape[1]), 1)[:, :n]
            tokens = tokens + pe
        tokens = self.trunk(tokens)
        n_obs = obs_tokens.shape[1]
        n_act = self.config.num_action_tokens
        action_features = tokens[:, n_obs : n_obs + n_act]
        future_features = tokens[:, n_obs + n_act :]
        return tokens, action_features, future_features

    # ---------------------------------------------------------------- losses
    def training_loss(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.config
        loss_w = cfg.loss
        device = sample["base_0_rgb"].device

        base = sample["base_0_rgb"]
        # Support [B,T,H,W,3] video window: index 0 current, -1 future.
        if base.ndim == 5:
            base_cur = base[:, 0]
            base_fut = base[:, -1]
        else:
            base_cur = base
            base_fut = sample.get("future_base_0_rgb", base)

        def _cam(key: str) -> torch.Tensor | None:
            x = sample.get(key)
            if x is None:
                return None
            return x[:, 0] if x.ndim == 5 else x

        left = _cam("left_wrist_0_rgb")
        right = _cam("right_wrist_0_rgb")
        state = sample["state"]
        prompts = sample.get("prompts") or [""] * base_cur.shape[0]
        is_ego = sample.get("is_ego")
        if is_ego is None:
            is_ego = torch.zeros(base_cur.shape[0], device=device, dtype=torch.bool)
        else:
            is_ego = torch.as_tensor(is_ego, device=device, dtype=torch.bool).reshape(-1)

        image_masks = {}
        for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            m = sample.get(f"{k}_mask")
            if m is not None:
                image_masks[k] = m[:, 0] if m.ndim == 2 else m

        obs_tokens, aux = self.encode_obs(
            base_img=base_cur,
            left_wrist=left,
            right_wrist=right,
            state=state,
            prompts=prompts,
            is_ego=is_ego,
            image_masks=image_masks or None,
        )
        _, action_features, future_features = self.forward_trunk(obs_tokens)

        actions = sample["action"].float()
        action_mask = sample.get("action_mask")
        if action_mask is not None:
            action_mask = action_mask.to(device=device, dtype=torch.float32)
            if action_mask.ndim == 2:
                action_mask_h = action_mask[:, None, :].expand_as(actions)
            else:
                action_mask_h = action_mask.float()
            actions = actions * action_mask_h
        else:
            action_mask_h = torch.ones_like(actions)

        # ---- flow matching action loss (pi05-style) ----
        noise = torch.randn_like(actions)
        time = _sample_beta(1.5, 1.0, actions.shape[0], device) * 0.999 + 0.001
        t = time[:, None, None]
        x_t = t * noise + (1.0 - t) * actions
        u_t = noise - actions
        if action_mask is not None:
            x_t = x_t * action_mask_h
            u_t = u_t * action_mask_h
            noise = noise * action_mask_h
        v_t = self.action_head(x_t, time, action_features)
        action_sq = ((v_t - u_t) ** 2) * action_mask_h
        denom = action_mask_h.sum(dim=(1, 2)).clamp_min(1.0)
        action_per = action_sq.sum(dim=(1, 2)) / denom

        # ---- world DINO loss ----
        with torch.no_grad():
            fut_feat = self._encode_image(base_fut)
            # CLS token as compact world target.
            world_tgt = fut_feat[:, 0]
        world_pred = self.world_head(future_features)
        world_per = F.mse_loss(world_pred.float(), world_tgt.float(), reduction="none").mean(dim=-1)

        is_ego = aux["is_ego"]
        is_robot = aux["is_robot"]

        def _domain_mean(per: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            if mask.any():
                return per[mask].mean()
            # Keep a grad edge when a domain is absent in this batch (DDP / small batch).
            return per.sum() * 0.0

        loss_ego_action = _domain_mean(action_per, is_ego)
        loss_robot_action = _domain_mean(action_per, is_robot)
        loss_ego_world = _domain_mean(world_per, is_ego)
        loss_robot_world = _domain_mean(world_per, is_robot)

        loss = (
            float(loss_w.get("lambda_ego_action", 0.5)) * loss_ego_action
            + float(loss_w.get("lambda_robot_action", 1.0)) * loss_robot_action
            + float(loss_w.get("lambda_ego_world", 1.0)) * loss_ego_world
            + float(loss_w.get("lambda_robot_world", 0.5)) * loss_robot_world
        )
        # Keep a non-zero graph if one domain is absent in the batch.
        if not torch.isfinite(loss):
            loss = action_per.mean() + world_per.mean()

        stats = {
            "loss_ego_action": loss_ego_action.detach(),
            "loss_robot_action": loss_robot_action.detach(),
            "loss_ego_world": loss_ego_world.detach(),
            "loss_robot_world": loss_robot_world.detach(),
            "loss_action": action_per.mean().detach(),
            "loss_world": world_per.mean().detach(),
        }
        return loss, stats

    @torch.no_grad()
    def sample_actions(
        self,
        sample: dict[str, Any],
        *,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Euler integrate the action flow from noise → action."""
        cfg = self.config
        num_steps = num_steps or cfg.num_inference_steps
        device = sample["base_0_rgb"].device
        base = sample["base_0_rgb"]
        base_cur = base[:, 0] if base.ndim == 5 else base

        def _cam(key: str) -> torch.Tensor | None:
            x = sample.get(key)
            if x is None:
                return None
            return x[:, 0] if x.ndim == 5 else x

        prompts = sample.get("prompts") or [""] * base_cur.shape[0]
        is_ego = sample.get("is_ego")
        if is_ego is None:
            is_ego = torch.zeros(base_cur.shape[0], device=device, dtype=torch.bool)
        obs_tokens, _ = self.encode_obs(
            base_img=base_cur,
            left_wrist=_cam("left_wrist_0_rgb"),
            right_wrist=_cam("right_wrist_0_rgb"),
            state=sample["state"],
            prompts=prompts,
            is_ego=is_ego,
        )
        _, action_features, _ = self.forward_trunk(obs_tokens)

        b = base_cur.shape[0]
        x_t = torch.randn(b, cfg.action_horizon, cfg.action_dim, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((b,), 1.0 - i * dt, device=device)
            v = self.action_head(x_t, t, action_features)
            x_t = x_t - dt * v
        return x_t
