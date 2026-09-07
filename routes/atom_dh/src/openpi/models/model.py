import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"
    # High-level policy π_HL (text-only: jointly generates subtask + long-term memory).
    PI0_HL = "pi0_hl"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3] or [*b, T, h, w, 3],
#             # RGB image(s) in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b] or bool[*b, T],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "subgoal_image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # Optional future-goal image(s)
#         ...
#     },
#     "subgoal_image_mask": {
#         "base_0_rgb": bool[*b],  # True if the subgoal image is valid
#         ...
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "state_history": float32[*b, T, s],  # Optional MEM proprioceptive history
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "dcc_metadata_tokens": int32[*b, d],  # Optional DCC metadata token segment
#     "dcc_metadata_mask": bool[*b, d],
#     "dcc_control_tokens": int32[*b, ctrl],  # Optional DCC control-mode token segment
#     "dcc_control_mask": bool[*b, ctrl],
#     "dcc_subtask_tokens": int32[*b, u],  # Optional DCC subtask token segment
#     "dcc_subtask_mask": bool[*b, u],
#     "token_ar_mask": int32[*b, n],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, n],  # Optional, loss mask for FAST model
#     "memory_summary_tokens": int32[*b, m],  # Optional MEM long-term summary generation tokens
#     "memory_summary_mask": bool[*b, m],
#     "memory_summary_ar_mask": bool[*b, m],
#     "memory_summary_loss_mask": bool[*b, m],
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32. MEM models may include a named time dimension before h/w/c.
    images: dict[str, at.Float[ArrayT, "b h w c"] | at.Float[ArrayT, "b t h w c"]]
    # Image masks, with same keys as images. MEM models may include the same named time dimension as images.
    image_masks: dict[str, at.Bool[ArrayT, "b"] | at.Bool[ArrayT, "b t"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "b s"]
    # Valid action dimensions. Missing means every action dimension is valid.
    action_mask: at.Bool[ArrayT, "b ad"] | None = None

    # Optional π0.7 subgoal images, encoded as future-state visual context.
    subgoal_images: dict[str, at.Float[ArrayT, "b h w c"]] | None = None
    subgoal_image_masks: dict[str, at.Bool[ArrayT, "b"]] | None = None
    # Optional low-dimensional state history for MEM short-term observation memory.
    state_history: at.Float[ArrayT, "b t s"] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "b l"] | None = None

    # π0.7 Diverse Context Conditioning text segments. The main task remains in tokenized_prompt;
    # metadata/control/subtask are tokenized separately so each component has its own prefix segment.
    dcc_metadata_tokens: at.Int[ArrayT, "b d"] | None = None
    dcc_metadata_mask: at.Bool[ArrayT, "b d"] | None = None
    dcc_control_tokens: at.Int[ArrayT, "b ctrl"] | None = None
    dcc_control_mask: at.Bool[ArrayT, "b ctrl"] | None = None
    dcc_subtask_tokens: at.Int[ArrayT, "b u"] | None = None
    dcc_subtask_mask: at.Bool[ArrayT, "b u"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "b n"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "b n"] | None = None

    # KI (Knowledge Insulation) specific fields.
    # FAST-tokenized action tokens used as teacher-forcing input to the VLM prefix.
    ki_fast_tokens: at.Int[ArrayT, "b n"] | None = None
    # Validity mask for ki_fast_tokens (True = valid token, False = padding).
    ki_fast_mask: at.Bool[ArrayT, "b n"] | None = None

    # MEM long-term language memory supervision fields.
    # These tokens are used only by the summary-generation CE branch and are kept separate
    # from tokenized_prompt so target summaries cannot leak into the action flow loss.
    memory_summary_tokens: at.Int[ArrayT, "b m"] | None = None
    memory_summary_mask: at.Bool[ArrayT, "b m"] | None = None
    memory_summary_ar_mask: at.Bool[ArrayT, "b m"] | None = None
    memory_summary_loss_mask: at.Bool[ArrayT, "b m"] | None = None

    # ========================================= 新增：双域训练掩码字段 ========================================================
    domain_mask: jax.Array | None = None        # [B] bool，True=Ego，False=Robot
    # OT bridge tags: 0=none, 1=hz_human, 2=hz_robot, 3=sz_human, 4=sz_robot
    ot_group: jax.Array | None = None
    # =======================================================================================================================

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for image_group in ("image", "subgoal_image"):
            if image_group not in data or data[image_group] is None:
                continue
            for key in data[image_group]:
                if data[image_group][key].dtype == np.uint8:
                    data[image_group][key] = data[image_group][key].astype(np.float32) / 255.0 * 2.0 - 1.0
                elif hasattr(data[image_group][key], "dtype") and data[image_group][key].dtype == torch.uint8:
                    data[image_group][key] = (
                        data[image_group][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                    )
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            subgoal_images=data.get("subgoal_image"),
            subgoal_image_masks=data.get("subgoal_image_mask"),
            state=data["state"],
            action_mask=data.get("action_mask"),
            state_history=data.get("state_history"),
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            dcc_metadata_tokens=data.get("dcc_metadata_tokens"),
            dcc_metadata_mask=data.get("dcc_metadata_mask"),
            dcc_control_tokens=data.get("dcc_control_tokens"),
            dcc_control_mask=data.get("dcc_control_mask"),
            dcc_subtask_tokens=data.get("dcc_subtask_tokens"),
            dcc_subtask_mask=data.get("dcc_subtask_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            ki_fast_tokens=data.get("ki_fast_tokens"),
            ki_fast_mask=data.get("ki_fast_mask"),
            memory_summary_tokens=data.get("memory_summary_tokens"),
            memory_summary_mask=data.get("memory_summary_mask"),
            memory_summary_ar_mask=data.get("memory_summary_ar_mask"),
            memory_summary_loss_mask=data.get("memory_summary_loss_mask"),

            #============================================================ 新增字段 ==========================================================
            domain_mask=data.get("domain_mask"),
            ot_group=data.get("ot_group"),
            #===============================================================================================================================
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        result["subgoal_image"] = result.pop("subgoal_images")
        result["subgoal_image_mask"] = result.pop("subgoal_image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        has_time_dim = image.ndim == len(batch_shape) + 4
        if has_time_dim:
            time_len = image.shape[-4]
            image_batch_shape = image.shape[:-4]
            image = jnp.reshape(image, (-1, *image.shape[-3:]))
        else:
            time_len = None
            image_batch_shape = None

        if image.shape[-3:-1] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[-3:-1]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            if has_time_dim:
                sequence_count = int(np.prod(image_batch_shape))
                sub_rngs = jax.random.split(rng, sequence_count)
                sub_rngs = jnp.repeat(sub_rngs, time_len, axis=0)
            else:
                sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        if has_time_dim:
            image = jnp.reshape(image, (*image_batch_shape, time_len, *image.shape[-3:]))

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(out_images[key].shape[:-3], dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    out_subgoal_images = None
    out_subgoal_masks = None
    if observation.subgoal_images is not None:
        out_subgoal_images = {}
        for key, image in observation.subgoal_images.items():
            if image.shape[-3:-1] != image_resolution:
                logger.info(f"Resizing subgoal image {key} from {image.shape[-3:-1]} to {image_resolution}")
                image = image_tools.resize_with_pad(image, *image_resolution)
            out_subgoal_images[key] = image

        out_subgoal_masks = {}
        source_masks = observation.subgoal_image_masks or {}
        for key in out_subgoal_images:
            if key not in source_masks:
                out_subgoal_masks[key] = jnp.ones(out_subgoal_images[key].shape[:-3], dtype=jnp.bool)
            else:
                out_subgoal_masks[key] = jnp.asarray(source_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        subgoal_images=out_subgoal_images,
        subgoal_image_masks=out_subgoal_masks,
        state=observation.state,
        action_mask=observation.action_mask,
        state_history=observation.state_history,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        dcc_metadata_tokens=observation.dcc_metadata_tokens,
        dcc_metadata_mask=observation.dcc_metadata_mask,
        dcc_control_tokens=observation.dcc_control_tokens,
        dcc_control_mask=observation.dcc_control_mask,
        dcc_subtask_tokens=observation.dcc_subtask_tokens,
        dcc_subtask_mask=observation.dcc_subtask_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        ki_fast_tokens=observation.ki_fast_tokens,
        ki_fast_mask=observation.ki_fast_mask,
        memory_summary_tokens=observation.memory_summary_tokens,
        memory_summary_mask=observation.memory_summary_mask,
        memory_summary_ar_mask=observation.memory_summary_ar_mask,
        memory_summary_loss_mask=observation.memory_summary_loss_mask,

        # =========================================== 新增：原样透传两个掩码字段 =============================================
        domain_mask=observation.domain_mask,
        ot_group=observation.ot_group,
         # =================================================================================================================
    )

@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
