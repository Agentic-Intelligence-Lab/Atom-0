# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def posemb_sincos_1d_zero_current(times, width, temperature=10_000.0, dtype=jnp.float32):
    """Fixed temporal sin/cos embedding with exactly zero embedding at the current timestep.

    MEM uses fixed temporal position encodings and requires the current frame (t=0) to add no
    perturbation, so K=1 exactly matches the original single-image ViT initialization.
    """
    assert width % 2 == 0, "Width must be even for sincos temporal posemb"
    times = jnp.asarray(times, dtype=jnp.float32)
    omega = jnp.arange(width // 2, dtype=jnp.float32) / (width // 2 - 1)
    omega = 1.0 / (temperature**omega)

    def encode(t):
        phase = jnp.einsum("...,d->...d", t, omega)
        return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)

    pe = encode(times) - encode(jnp.asarray(0.0, dtype=jnp.float32))
    return jnp.asarray(pe, dtype)


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        is_video = x.ndim == 4
        x = sharding.activation_sharding_constraint(x)

        pre_attn_norm = nn.LayerNorm(dtype=self.dtype_mm)
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )

        if is_video:
            b, t, p, d = x.shape
            y = jnp.reshape(x, (b * t, p, d))
            y = pre_attn_norm(y)
            y = out["sa"] = attention(y, y)
            y = jnp.reshape(y, (b, t, p, d))
            y = sharding.activation_sharding_constraint(y)
            y = nn.Dropout(rate=self.dropout)(y, deterministic)
            x = out["+sa"] = x + y
        else:
            y = pre_attn_norm(x)
            y = out["sa"] = attention(y, y)
            y = sharding.activation_sharding_constraint(y)
            y = nn.Dropout(rate=self.dropout)(y, deterministic)
            x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        if is_video:
            y = jnp.reshape(y, (b * t, p, d))
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        if is_video:
            y = jnp.reshape(y, (b, t, p, d))
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


class VideoEncoder1DBlock(nn.Module):
    """ViT block with MEM-style space-time separable attention.

    The same attention parameters are reused for spatial and temporal attention, so video memory does not
    introduce additional learnable attention weights compared with the single-image ViT block.
    """

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    temporal_attention: bool = False

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        b, t, p, d = x.shape
        x = sharding.activation_sharding_constraint(x)

        pre_attn_norm = nn.LayerNorm(dtype=self.dtype_mm)
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )

        spatial = jnp.reshape(x, (b * t, p, d))
        y = pre_attn_norm(spatial)
        y = out["sa"] = attention(y, y)
        y = jnp.reshape(y, (b, t, p, d))
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        if self.temporal_attention and t > 1:
            temporal = jnp.reshape(jnp.swapaxes(x, 1, 2), (b * p, t, d))
            y = pre_attn_norm(temporal)
            causal_mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))[None, None, :, :]
            y = out["ta"] = attention(y, y, mask=causal_mask)
            y = jnp.swapaxes(jnp.reshape(y, (b, p, t, d)), 1, 2)
            y = sharding.activation_sharding_constraint(y)
            y = nn.Dropout(rate=self.dropout)(y, deterministic)
            x = out["+ta"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = jnp.reshape(y, (b * t, p, d))
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = jnp.reshape(y, (b, t, p, d))
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    temporal_attention_every_n_layers: int = 4

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        is_video = x.ndim == 4

        if is_video and self.scan:
            raise ValueError("Video memory encoder requires scan=False so temporal attention can be inserted by layer.")

        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            # Input Encoder
            for lyr in range(self.depth):
                if is_video:
                    block_cur = VideoEncoder1DBlock(
                        name=f"encoderblock_{lyr}",
                        dtype_mm=self.dtype_mm,
                        mlp_dim=self.mlp_dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                        temporal_attention=(lyr + 1) % self.temporal_attention_every_n_layers == 0,
                    )
                else:
                    block_cur = Encoder1DBlock(
                        name=f"encoderblock_{lyr}",
                        dtype_mm=self.dtype_mm,
                        mlp_dim=self.mlp_dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                    )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x  # Alias for last block, but without the number in it.

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


class _Module(nn.Module):
    """ViT model."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"
    head_zeroinit: bool = True
    scan: bool = False
    # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    history_length: int = 1
    temporal_attention_every_n_layers: int = 4

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}
        has_time_dim = image.ndim == 5
        if has_time_dim:
            b, t = image.shape[:2]
            image = jnp.reshape(image, (b * t, *image.shape[-3:]))
        else:
            b = image.shape[0]
            t = 1

        # Kevin edit: do patch extraction and posemb in float32,
        # because I feel like it's a bit safer.
        image = jnp.asarray(image, jnp.float32)

        # Patch extraction
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        if has_time_dim:
            p = h * w
            x = jnp.reshape(x, (b, t, p, c))
            # Frames are ordered from past to current; current frame has temporal index 0.
            temporal_indices = jnp.arange(t, dtype=jnp.float32) - (t - 1)
            temporal_posemb = posemb_sincos_1d_zero_current(temporal_indices, c, dtype=jnp.float32)
            x = out["with_temporal_posemb"] = x + temporal_posemb[None, :, None, :]

        if self.pool_type == "tok":
            if has_time_dim:
                raise ValueError("MEM video encoder only supports patch-token outputs, not cls-token pooling.")
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        x = nn.Dropout(rate=self.dropout)(x, not train)

        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            temporal_attention_every_n_layers=self.temporal_attention_every_n_layers,
            name="Transformer",
        )(x, deterministic=not train)

        if has_time_dim:
            # MEM compresses the visual history into the current timestep representation before
            # passing tokens to the VLA backbone.
            x = x[:, -1]
        encoded = out["encoded"] = x
        n = x.shape[0]

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
