from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        memory_summary_tokenizer: Any | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._memory_summary_tokenizer = memory_summary_tokenizer
        self._long_memory_enabled = bool(getattr(model, "long_memory_enabled", False))
        self._memory_update_interval_steps = int(getattr(model, "memory_update_interval_steps", 30))
        self._memory_generation_max_new_tokens = int(getattr(model, "memory_generation_max_new_tokens", 64))
        self.memory_summary = ""
        self.memory_step = 0
        self._memory_log: list[dict[str, Any]] = []

        if self._is_pytorch_model:
            if self._long_memory_enabled:
                raise NotImplementedError("Long-term MEM policy state is only implemented for the JAX model path.")
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        if self._long_memory_enabled and self._obs_requests_memory_reset(inputs):
            self.reset_memory()
        memory_summary_before = self.memory_summary
        memory_event = None
        if self._long_memory_enabled:
            inputs["memory_summary"] = self.memory_summary
        inputs = self._input_transform(inputs)
        # `memory_summary` is a policy-side text control field. In normal trained policies,
        # PrependMemorySummaryToPrompt consumes it before tokenization; keep direct/unit-test
        # policy construction robust by dropping it if no transform consumed it.
        inputs.pop("memory_summary", None)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._long_memory_enabled and self._should_update_memory():
            memory_event = self._update_memory_summary(observation, summary_before=memory_summary_before)
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if self._long_memory_enabled:
            outputs["memory_summary"] = self.memory_summary
            outputs["memory_step"] = self.memory_step
            outputs["memory_event"] = memory_event or {
                "step": self.memory_step,
                "updated": False,
                "summary_before": memory_summary_before,
                "summary_after": self.memory_summary,
                "generated_summary": "",
            }
        self.memory_step += 1
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def reset_memory(self) -> None:
        self.memory_summary = ""
        self.memory_step = 0
        self._memory_log = []

    def get_memory_summary(self) -> str:
        return self.memory_summary

    def get_memory_log(self) -> list[dict[str, Any]]:
        return list(self._memory_log)

    def clear_memory_log(self) -> None:
        self._memory_log = []

    def _obs_requests_memory_reset(self, obs: dict) -> bool:
        reset = False
        for key in ("reset_memory", "memory_reset", "episode_start", "is_first"):
            if key not in obs:
                continue
            value = obs.pop(key)
            if isinstance(value, np.ndarray):
                reset = reset or bool(np.asarray(value).item())
            else:
                reset = reset or bool(value)
        return reset

    def _should_update_memory(self) -> bool:
        return self.memory_step % self._memory_update_interval_steps == 0

    def _update_memory_summary(
        self, observation: _model.Observation, *, summary_before: str | None = None
    ) -> dict[str, Any]:
        summary_before = self.memory_summary if summary_before is None else summary_before
        event = {
            "step": self.memory_step,
            "updated": False,
            "summary_before": summary_before,
            "summary_after": self.memory_summary,
            "generated_summary": "",
        }
        if self._memory_summary_tokenizer is None:
            self._memory_log.append(event)
            return event
        tokens = self._model.generate_memory_summary_tokens(
            observation,
            max_new_tokens=self._memory_generation_max_new_tokens,
        )
        token_list = np.asarray(tokens[0]).astype(np.int32)
        eos_positions = np.where(token_list == 1)[0]
        if eos_positions.size:
            token_list = token_list[: eos_positions[0]]
        summary = self._memory_summary_tokenizer.decode(token_list)
        if summary:
            self.memory_summary = summary.strip()
            event.update(
                {
                    "updated": True,
                    "summary_after": self.memory_summary,
                    "generated_summary": self.memory_summary,
                }
            )
        self._memory_log.append(event)
        return event


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
