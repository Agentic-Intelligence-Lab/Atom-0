"""HPT model: multi-stem → shared trunk → tdec action + DINO world DiT."""

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
    CrossTransformerActionHead,
    FlowMatchingActionDiTHead,
    FlowMatchingActionHead,
    Stem,
    TransformerTrunk,
    WorldDiTHead,
    get_sinusoid_encoding_table,
)
from openpi.models_pytorch.hpt.official_action_heads import DiffusionActionHead, TransformerDecoderActionHead

logger = logging.getLogger("openpi")


def _remap_official_hpt_trunk_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map ``liruiw/HPT`` ``trunk.pth`` keys to Atom-0 ``TransformerTrunk``.

    Official SimpleTransformer blocks use ``norm_1`` / ``norm_2`` and ``mlp.fc1`` / ``mlp.fc2``.
    Atom-0 uses ``norm1`` / ``norm2`` and ``nn.Sequential`` indices ``mlp.0`` / ``mlp.2``.
    Official MHA may include ``bias_k`` / ``bias_v`` (``add_bias_kv=True``); we skip those.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        k = key
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("trunk."):
            k = k[len("trunk.") :]
        if not k.startswith("blocks."):
            continue
        k = k.replace(".norm_1.", ".norm1.").replace(".norm_2.", ".norm2.")
        k = k.replace(".mlp.fc1.", ".mlp.0.").replace(".mlp.fc2.", ".mlp.2.")
        if ".bias_k" in k or ".bias_v" in k:
            continue
        remapped[k] = value
    return remapped


def _sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor:
    dist = torch.distributions.Beta(
        torch.tensor(alpha, device=device, dtype=torch.float32),
        torch.tensor(beta, device=device, dtype=torch.float32),
    )
    return dist.sample((bsize,))


def count_action_head_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def load_hpt_weights(model: nn.Module, weight_path: str, *, device: str | None = None) -> None:
    """Load safetensors with ``strict=False`` and log action-head key mismatches."""
    import safetensors.torch

    dev = device or getattr(model, "device_str", "cpu")
    missing, unexpected = safetensors.torch.load_model(model, weight_path, strict=False, device=str(dev))
    head = getattr(getattr(model, "hpt", model), "action_head", None)
    head_type = type(head).__name__ if head is not None else "unknown"
    if missing:
        head_missing = [k for k in missing if k.startswith("hpt.action_head.") or k.startswith("action_head.")]
        other_missing = [k for k in missing if k not in head_missing]
        if head_missing:
            logger.warning(
                "Action head mismatch (%s): %d missing keys (expected when switching head type). First: %s",
                head_type,
                len(head_missing),
                head_missing[:5],
            )
        if other_missing:
            logger.warning("Missing non-action-head keys (%d): %s", len(other_missing), other_missing[:8])
    if unexpected:
        head_unexp = [k for k in unexpected if "action_head" in k]
        if head_unexp:
            logger.warning(
                "Unexpected action-head keys in checkpoint (%d): %s",
                len(head_unexp),
                head_unexp[:5],
            )
        other_unexp = [k for k in unexpected if k not in head_unexp]
        if other_unexp:
            logger.warning("Unexpected keys (%d): %s", len(other_unexp), other_unexp[:8])
    logger.info("Loaded HPT weights from %s (head=%s)", weight_path, head_type)


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

        # Modality stems: ego (base/head, shared human+robot), wrist (robot-only), proprio, language.
        self.ego_stem = Stem(dino_dim, d, config.ego_tokens)
        self.wrist_stem = Stem(dino_dim, d, config.wrist_tokens)
        self.state_stem = Stem(config.proprio_dim, d, config.state_tokens, widths=[128, 128])
        self.language_stem = Stem(t5_dim, d, config.language_tokens)

        self.action_only = config.head_mode == "action_only"
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.trunk = TransformerTrunk(
            embed_dim=d,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            drop_path=config.drop_path,
        )

        if config.action_head_type == "dit":
            self.action_head = FlowMatchingActionDiTHead(
                action_dim=config.action_dim,
                action_horizon=config.action_horizon,
                cond_dim=d,
                hidden_dim=config.action_head_dim,
                num_blocks=config.action_head_blocks,
                num_heads=config.action_head_heads,
                mlp_ratio=config.action_head_mlp_ratio,
                dropout=config.action_head_dropout,
                drop_path=config.action_head_drop_path,
            )
        elif config.action_head_type == "cross_transformer":
            self.action_head = CrossTransformerActionHead(
                action_dim=config.action_dim,
                action_horizon=config.action_horizon,
                cond_dim=d,
                hidden_dim=config.action_head_dim,
                num_layers=config.action_head_blocks,
                num_heads=config.action_head_heads,
                mlp_ratio=config.action_head_mlp_ratio,
                dropout=config.action_head_dropout,
                drop_path=config.action_head_drop_path,
            )
        elif config.action_head_type == "mlp":
            self.action_head = FlowMatchingActionHead(
                action_dim=config.action_dim,
                action_horizon=config.action_horizon,
                cond_dim=d,
                hidden_dim=config.action_head_dim,
            )
        elif config.action_head_type == "diffusion":
            self.action_head = DiffusionActionHead(
                action_dim=config.action_dim,
                action_horizon=config.action_horizon,
                cond_dim=d,
                train_timesteps=config.diffusion_train_timesteps,
                num_inference_steps=config.num_inference_steps,
                down_dims=config.diffusion_down_dims,
            )
        elif config.action_head_type == "transformer_decoder":
            self.action_head = TransformerDecoderActionHead(
                action_dim=config.action_dim,
                action_horizon=config.action_horizon,
                cond_dim=d,
                hidden_dim=config.action_head_dim,
                num_heads=config.action_head_heads,
                dropout=config.action_head_dropout,
                huber_delta=config.transformer_decoder_huber_delta,
            )
        else:
            raise ValueError(f"Unknown action_head_type={config.action_head_type!r}")
        self.world_head = (
            None
            if self.action_only
            else WorldDiTHead(
                dino_dim=dino_dim,
                cond_dim=d,
                num_patches=config.world_num_patches,
                dit_dim=config.world_dit_dim,
                dit_blocks=config.world_dit_blocks,
                dit_heads=config.world_dit_heads,
                dit_mlp_ratio=config.world_dit_mlp_ratio,
                wide_dim=config.world_wide_dim,
                wide_blocks=config.world_wide_blocks,
                wide_heads=config.world_wide_heads,
                wide_mlp_ratio=config.world_wide_mlp_ratio,
                time_dim=config.world_time_dim,
                dropout=config.world_dropout,
                drop_path=config.world_drop_path,
            )
        )

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
        self.pos_embed.requires_grad = False
        logger.info("HPT finetune mode: trunk + pos_embed frozen")

    def unfreeze_trunk(self) -> None:
        for p in self.trunk.parameters():
            p.requires_grad = True
        self.pos_embed.requires_grad = True

    def load_pretrained_trunk(self, path: str) -> None:
        """Load a liruiw/HPT-style ``trunk.pth`` (key remap + shape check)."""
        import os

        ckpt = path
        if os.path.isdir(path):
            ckpt = os.path.join(path, "trunk.pth")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        remapped = _remap_official_hpt_trunk_state(state)
        if not remapped:
            logger.warning("No mappable trunk keys in %s; is this an official HPT trunk.pth?", ckpt)
        missing, unexpected = self.trunk.load_state_dict(remapped, strict=False)
        loaded = len(remapped) - len(unexpected)
        logger.info(
            "Loaded HPT trunk from %s (mapped=%s loaded=%s missing=%s unexpected=%s)",
            ckpt,
            len(remapped),
            loaded,
            len(missing),
            len(unexpected),
        )
        if loaded == 0:
            logger.warning(
                "Trunk warm-start loaded 0 tensors — check embed_dim=%s matches checkpoint (hpt-base-lang=256).",
                self.config.embed_dim,
            )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # ---------------------------------------------------------------- encode
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self._ensure_bhwc(image)
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

    def _maybe_crop_horizon(self, x: torch.Tensor) -> torch.Tensor:
        """Official random_horizon_masking: keep a random suffix of length 1..T (train only)."""
        t = x.shape[1]
        if not (self.training and self.config.random_horizon_masking and t > 1):
            return x
        horizon = int(torch.randint(1, t + 1, (1,), device=x.device).item())
        return x[:, t - horizon :]

    def _add_horizon_sinusoid(self, feat: torch.Tensor) -> torch.Tensor:
        """feat [B, T, N, D] → flatten T*N and add official sinusoid PE."""
        b, t, n, d = feat.shape
        table = get_sinusoid_encoding_table(t * n, d, device=feat.device, dtype=feat.dtype)
        return feat.reshape(b, t * n, d) + table

    def _encode_image_history(self, image: torch.Tensor) -> torch.Tensor:
        """image [B, T, H, W, 3] → DINO tokens [B, T, N, D]."""
        b, t, h, w, c = image.shape
        feat = self._encode_image(image.reshape(b * t, h, w, c))
        return feat.reshape(b, t, feat.shape[1], feat.shape[2])

    def _as_image_history(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 4:
            return image.unsqueeze(1)
        if image.ndim != 5:
            raise ValueError(f"Expected image [B,H,W,3] or [B,T,H,W,3], got {tuple(image.shape)}")
        return image

    def _maybe_gate(self, tokens: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Zero inactive rows; optionally stop grad into inactive domain stems."""
        # active: [B] bool
        mask = active.to(dtype=tokens.dtype).view(-1, 1, 1)
        if self.config.domain_stem_grad_gate:
            # Inactive samples: detach so stem grads don't flow from the other domain.
            tokens = tokens * mask + tokens.detach() * (1.0 - mask)
            return tokens * mask
        return tokens * mask

    def _ensure_bhwc(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image layout to ``[B, H, W, 3]`` or ``[B, T, H, W, 3]``."""
        if image.ndim == 4:
            if image.shape[1] == 3 and image.shape[-1] != 3:
                return image.permute(0, 2, 3, 1).contiguous()
            return image
        if image.ndim == 5:
            if image.shape[2] == 3 and image.shape[-1] != 3:
                return image.permute(0, 1, 3, 4, 2).contiguous()
            return image
        raise ValueError(f"Expected image [B,H,W,3] or [B,T,H,W,3], got {tuple(image.shape)}")

    def _current_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Take the latest frame from an image tensor ``[B,H,W,C]`` or ``[B,T,H,W,C]``."""
        x = self._ensure_bhwc(x)
        x = self._maybe_crop_horizon(self._as_image_history(x))
        return x[:, -1]

    def _encode_current_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a single-frame image to DINO tokens ``[B, 1, N, D]``."""
        frame = self._current_frame(image)
        feat = self._encode_image(frame)
        return feat.unsqueeze(1)

    def _dino_patch_features(self, image: torch.Tensor) -> torch.Tensor:
        """Frozen DINOv2 patch tokens, dropping CLS. Shape [B, world_num_patches, dino_dim]."""
        feat = self._encode_image(self._current_frame(image))
        n = int(self.config.world_num_patches)
        if feat.shape[1] == n + 1:
            feat = feat[:, 1:]
        elif feat.shape[1] > n:
            feat = feat[:, -n:]
        elif feat.shape[1] < n:
            raise ValueError(f"DINO tokens {feat.shape[1]} < world_num_patches={n}")
        return feat

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
        """Build current-frame observation tokens for a mixed ego/robot batch.

        Token layout: ego(32) + wrist(16) + proprio(16) + language(8) = 72.
        Human and robot both use ``ego_stem`` for the base/head camera. Wrist
        tokens are zeroed for human samples.
        """
        device = base_img.device
        is_ego = is_ego.to(device=device, dtype=torch.bool).reshape(-1)
        is_robot = ~is_ego

        base_hist = self._encode_current_image(base_img)
        base_ctx = self._add_horizon_sinusoid(base_hist)
        ego_tok = self.ego_stem(base_ctx)

        if left_wrist is None:
            left_wrist = torch.zeros_like(base_img)
        if right_wrist is None:
            right_wrist = torch.zeros_like(base_img)
        left_feat = self._encode_current_image(left_wrist)
        right_feat = self._encode_current_image(right_wrist)
        wrist_feat = torch.cat([left_feat, right_feat], dim=2)
        wrist_tok = self._maybe_gate(self.wrist_stem(self._add_horizon_sinusoid(wrist_feat)), is_robot)

        if image_masks is not None:
            left_m = image_masks.get("left_wrist_0_rgb")
            right_m = image_masks.get("right_wrist_0_rgb")
            if left_m is not None and right_m is not None:
                def _mask_valid(m: torch.Tensor) -> torch.Tensor:
                    m = m.to(device)
                    if m.ndim > 1:
                        m = m.reshape(m.shape[0], -1).any(dim=-1)
                    return m.reshape(-1)

                wrist_valid = (_mask_valid(left_m) | _mask_valid(right_m)) & is_robot
                wrist_tok = wrist_tok * wrist_valid.to(wrist_tok.dtype).view(-1, 1, 1)

        if state.ndim == 2:
            state = state.unsqueeze(1)
        state = self._maybe_crop_horizon(state.float())[:, -1:]
        state_feat = state.unsqueeze(2)  # [B, T, 1, D]
        state_tok = self.state_stem(self._add_horizon_sinusoid(state_feat))
        lang_feat = self._encode_language(prompts, device)
        lang_tok = self.language_stem(lang_feat)

        obs_tokens = torch.cat([ego_tok, wrist_tok, state_tok, lang_tok], dim=1)
        aux = {
            "is_ego": is_ego,
            "is_robot": is_robot,
            "base_feat": base_hist[:, 0],
        }
        return obs_tokens, aux

    def _apply_pos_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[1]
        if n <= self.pos_embed.shape[1]:
            return tokens + self.pos_embed[:, :n]
        pe = self.pos_embed.repeat(1, math.ceil(n / self.pos_embed.shape[1]), 1)[:, :n]
        return tokens + pe

    def _run_trunk(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.trunk(self._apply_pos_embed(tokens))

    def forward_trunk_obs(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        """Shared trunk over current-frame obs tokens only (no action/future queries)."""
        return self._run_trunk(obs_tokens)

    def forward_trunk(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_trunk_obs(obs_tokens)

    # ---------------------------------------------------------------- losses
    def training_loss(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.config
        loss_w = cfg.loss
        device = sample["base_0_rgb"].device

        base = sample["base_0_rgb"]
        obs_h = int(self.config.observation_horizon)
        # Support [B,T,H,W,3]: first obs_h frames = history (oldest→current), last = future world.
        if base.ndim == 5:
            t_all = base.shape[1]
            if t_all > obs_h:
                base_obs = base[:, :obs_h]
                base_fut = base[:, -1]
            else:
                base_obs = base
                base_fut = base[:, -1]
        else:
            base_obs = base
            base_fut = sample.get("future_base_0_rgb", base)

        def _cam_obs(key: str) -> torch.Tensor | None:
            x = sample.get(key)
            if x is None:
                return None
            if x.ndim == 5:
                return x[:, :obs_h] if x.shape[1] > obs_h else x
            return x

        left = _cam_obs("left_wrist_0_rgb")
        right = _cam_obs("right_wrist_0_rgb")
        state_hist = sample.get("state_history")
        if state_hist is None:
            state_hist = sample["state"]
        prompts = sample.get("prompts") or [""] * base_obs.shape[0]
        is_ego = sample.get("is_ego")
        if is_ego is None:
            is_ego = torch.zeros(base_obs.shape[0], device=device, dtype=torch.bool)
        else:
            is_ego = torch.as_tensor(is_ego, device=device, dtype=torch.bool).reshape(-1)

        image_masks = {}
        for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            m = sample.get(f"{k}_mask")
            if m is not None:
                if m.ndim == 2:
                    image_masks[k] = m[:, :obs_h] if m.shape[1] > obs_h else m
                else:
                    image_masks[k] = m

        obs_tokens, aux = self.encode_obs(
            base_img=base_obs,
            left_wrist=left,
            right_wrist=right,
            state=state_hist,
            prompts=prompts,
            is_ego=is_ego,
            image_masks=image_masks or None,
        )

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

        head_type = self.config.action_head_type
        trunk_obs = self.forward_trunk_obs(obs_tokens)
        if head_type == "diffusion":
            global_cond = trunk_obs.mean(dim=1)
            action_per = self.action_head.compute_loss(global_cond, actions, action_mask_h)
            smooth_per = action_per * 0.0
        elif head_type == "transformer_decoder":
            action_per = self.action_head.compute_loss(trunk_obs, actions, action_mask_h)
            smooth_per = action_per * 0.0
        else:
            noise = torch.randn_like(actions)
            time = _sample_beta(1.5, 1.0, actions.shape[0], device) * 0.999 + 0.001
            x_t = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
            u_t = noise - actions
            if action_mask is not None:
                x_t = x_t * action_mask_h
                u_t = u_t * action_mask_h
                noise = noise * action_mask_h
            v_t = self.action_head(x_t, time, trunk_obs)
            action_sq = ((v_t - u_t) ** 2) * action_mask_h
            denom = action_mask_h.sum(dim=(1, 2)).clamp_min(1.0)
            action_per = action_sq.sum(dim=(1, 2)) / denom

            lam_smooth = float(loss_w.get("lambda_action_smooth", 0.0))
            if lam_smooth > 0.0:
                a_hat = (noise - v_t) * action_mask_h
                d_hat = a_hat[:, 1:] - a_hat[:, :-1]
                d_gt = actions[:, 1:] - actions[:, :-1]
                pair_mask = action_mask_h[:, 1:] * action_mask_h[:, :-1]
                smooth_sq = ((d_hat - d_gt) ** 2) * pair_mask
                smooth_denom = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
                smooth_per = smooth_sq.sum(dim=(1, 2)) / smooth_denom
            else:
                smooth_per = action_per * 0.0

        lam_smooth = float(loss_w.get("lambda_action_smooth", 0.0))
        if self.action_only or self.world_head is None:
            world_per = action_per.new_zeros(action_per.shape)
        else:
            with torch.no_grad():
                z = self._dino_patch_features(base_fut)
            noise_w = torch.randn_like(z)
            time_w = _sample_beta(1.5, 1.0, z.shape[0], device) * 0.999 + 0.001
            x_w = (1.0 - time_w[:, None, None]) * z + time_w[:, None, None] * noise_w
            u_w = noise_w - z
            v_w = self.world_head(x_w, time_w, trunk_obs.mean(dim=1))
            world_per = ((v_w.float() - u_w.float()) ** 2).mean(dim=(1, 2))

        is_ego = aux["is_ego"]
        is_robot = aux["is_robot"]

        def _domain_mean(per: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            if mask.any():
                return per[mask].mean()
            # Keep a grad edge when a domain is absent in this batch (DDP / small batch).
            return per.sum() * 0.0

        loss_ego_action = _domain_mean(action_per, is_ego)
        loss_robot_action = _domain_mean(action_per, is_robot)
        loss_ego_smooth = _domain_mean(smooth_per, is_ego)
        loss_robot_smooth = _domain_mean(smooth_per, is_robot)

        loss = (
            float(loss_w.get("lambda_ego_action", 0.5)) * loss_ego_action
            + float(loss_w.get("lambda_robot_action", 1.0)) * loss_robot_action
            + lam_smooth * (loss_ego_smooth + loss_robot_smooth)
        )
        if not self.action_only:
            loss_ego_world = _domain_mean(world_per, is_ego)
            loss_robot_world = _domain_mean(world_per, is_robot)
            loss = (
                loss
                + float(loss_w.get("lambda_ego_world", 1.0)) * loss_ego_world
                + float(loss_w.get("lambda_robot_world", 0.5)) * loss_robot_world
            )
        else:
            loss_ego_world = world_per.mean() * 0.0
            loss_robot_world = world_per.mean() * 0.0
        # Keep a non-zero graph if one domain is absent in the batch.
        if not torch.isfinite(loss):
            loss = action_per.mean() if self.action_only else action_per.mean() + world_per.mean()

        stats = {
            "loss_ego_action": loss_ego_action.detach(),
            "loss_robot_action": loss_robot_action.detach(),
            "loss_ego_world": loss_ego_world.detach(),
            "loss_robot_world": loss_robot_world.detach(),
            "loss_action": action_per.mean().detach(),
            "loss_world": world_per.mean().detach(),
            "loss_action_smooth": smooth_per.mean().detach(),
            "loss_ego_smooth": loss_ego_smooth.detach(),
            "loss_robot_smooth": loss_robot_smooth.detach(),
        }
        return loss, stats

    @torch.no_grad()
    def sample_actions(
        self,
        sample: dict[str, Any],
        *,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Sample actions: FM Euler for dit/cross/mlp; DDIM or direct decode for official heads.

        World head is never used at inference.
        """
        cfg = self.config
        device = sample["base_0_rgb"].device
        base = sample["base_0_rgb"]
        obs_h = int(self.config.observation_horizon)
        base_obs = base[:, :obs_h] if base.ndim == 5 else base

        def _cam_obs(key: str) -> torch.Tensor | None:
            x = sample.get(key)
            if x is None:
                return None
            if x.ndim == 5:
                return x[:, :obs_h] if x.shape[1] > obs_h else x
            return x

        prompts = sample.get("prompts") or [""] * base_obs.shape[0]
        is_ego = sample.get("is_ego")
        if is_ego is None:
            is_ego = torch.zeros(base_obs.shape[0], device=device, dtype=torch.bool)
        state_hist = sample.get("state_history")
        if state_hist is None:
            state_hist = sample["state"]
        obs_tokens, _ = self.encode_obs(
            base_img=base_obs,
            left_wrist=_cam_obs("left_wrist_0_rgb"),
            right_wrist=_cam_obs("right_wrist_0_rgb"),
            state=state_hist,
            prompts=prompts,
            is_ego=is_ego,
        )

        head_type = cfg.action_head_type
        trunk_obs = self.forward_trunk_obs(obs_tokens)
        if head_type == "diffusion":
            return self.action_head.sample(trunk_obs.mean(dim=1), num_steps=num_steps)
        if head_type == "transformer_decoder":
            return self.action_head(trunk_obs)

        num_steps = num_steps or cfg.num_inference_steps
        b = base_obs.shape[0]
        x_t = torch.randn(b, cfg.action_horizon, cfg.action_dim, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((b,), 1.0 - i * dt, device=device)
            v = self.action_head(x_t, t, trunk_obs)
            x_t = x_t - dt * v
        return x_t
