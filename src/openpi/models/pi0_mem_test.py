import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import siglip


def _make_model(history_length: int = 6, *, ki_enabled: bool = False, long_memory_enabled: bool = False):
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        discrete_state_input=False,
        history_length=history_length,
        ki_enabled=ki_enabled,
        long_memory_enabled=long_memory_enabled,
        memory_summary_max_len=8,
        memory_generation_max_new_tokens=2,
    )
    return config, config.create(jax.random.key(0))


def _make_obs(config: pi0_config.Pi0Config, batch_size: int = 2):
    if config.history_length == 1:
        image = jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32)
        image_mask = jnp.ones((batch_size,), dtype=jnp.bool_)
        state_history = None
    else:
        image = jnp.zeros((batch_size, config.history_length, 224, 224, 3), dtype=jnp.float32)
        image_mask = jnp.ones((batch_size, config.history_length), dtype=jnp.bool_)
        state_history = jnp.zeros((batch_size, config.history_length, config.action_dim), dtype=jnp.float32)

    with _model.at.disable_typechecking():
        return _model.Observation(
            images={key: image for key in _model.IMAGE_KEYS},
            image_masks={key: image_mask for key in _model.IMAGE_KEYS},
            state=jnp.zeros((batch_size, config.action_dim), dtype=jnp.float32),
            state_history=state_history,
            tokenized_prompt=jnp.zeros((batch_size, config.max_token_len), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((batch_size, config.max_token_len), dtype=jnp.bool_),
            ki_fast_tokens=jnp.zeros((batch_size, config.ki_fast_max_len), dtype=jnp.int32)
            if config.ki_enabled
            else None,
            ki_fast_mask=jnp.ones((batch_size, config.ki_fast_max_len), dtype=jnp.bool_)
            if config.ki_enabled
            else None,
            token_loss_mask=jnp.ones((batch_size, config.ki_fast_max_len), dtype=jnp.bool_)
            if config.ki_enabled
            else None,
            memory_summary_tokens=jnp.zeros((batch_size, config.memory_summary_max_len), dtype=jnp.int32)
            if config.long_memory_enabled
            else None,
            memory_summary_mask=jnp.ones((batch_size, config.memory_summary_max_len), dtype=jnp.bool_)
            if config.long_memory_enabled
            else None,
            memory_summary_ar_mask=jnp.ones((batch_size, config.memory_summary_max_len), dtype=jnp.bool_)
            if config.long_memory_enabled
            else None,
            memory_summary_loss_mask=jnp.array([[False, True, True, False, False, False, False, False]] * batch_size)
            if config.long_memory_enabled
            else None,
        )


def test_history_length_1_matches_original_pi05_path():
    """MEM defaults must be exactly identical to the pre-MEM pi05 graph and loss."""
    base_config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        discrete_state_input=False,
    )
    mem_disabled_config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        discrete_state_input=False,
        history_length=1,
    )
    base_model = base_config.create(jax.random.key(123))
    mem_disabled_model = mem_disabled_config.create(jax.random.key(123))

    obs = _make_obs(base_config)
    actions = jax.random.normal(jax.random.key(5), (2, base_config.action_horizon, base_config.action_dim)) * 0.1
    rng = jax.random.key(9)

    base_loss = base_model.compute_loss(rng, obs, actions)
    mem_disabled_loss = mem_disabled_model.compute_loss(rng, obs, actions)

    np.testing.assert_allclose(base_loss, mem_disabled_loss, rtol=1e-5, atol=1e-5)


def test_temporal_posemb_current_frame_is_zero():
    pe = siglip.posemb_sincos_1d_zero_current(jnp.array([-5, -4, -3, -2, -1, 0]), 64)
    assert jnp.allclose(pe[-1], 0.0)


def test_mem_video_encoder_keeps_current_frame_token_count():
    config, model = _make_model(history_length=6)
    obs = _make_obs(config)
    image_tokens, _ = model.PaliGemma.img(obs.images["base_0_rgb"], train=False)

    single_config, single_model = _make_model(history_length=1)
    single_obs = _make_obs(single_config)
    single_image_tokens, _ = single_model.PaliGemma.img(single_obs.images["base_0_rgb"], train=False)

    assert image_tokens.shape[:2] == single_image_tokens.shape[:2]


def test_current_frame_loss_backprops_to_history_frames():
    """The current-frame video representation must depend on past visual frames."""
    config, model = _make_model(history_length=6)
    images = jax.random.normal(jax.random.key(123), (1, config.history_length, 224, 224, 3)) * 0.01

    def current_frame_representation_loss(video):
        _, intermediates = model.PaliGemma.img(video, train=False)
        # The public image tokens pass through a zero-initialized projection head, so a loss on those
        # tokens has zero gradient at initialization. The encoded representation is the current-frame
        # video representation before that head and should still depend on history.
        encoded = intermediates["encoded"]
        return jnp.mean(jnp.square(encoded.astype(jnp.float32)))

    grads = jax.grad(current_frame_representation_loss)(images)
    per_frame_grad_norm = jnp.sqrt(jnp.sum(jnp.square(grads), axis=(0, 2, 3, 4)))

    assert jnp.all(jnp.isfinite(per_frame_grad_norm))
    assert float(jnp.max(per_frame_grad_norm[:-1])) > 1e-8, per_frame_grad_norm
    assert float(per_frame_grad_norm[-1]) > 1e-8, per_frame_grad_norm


def test_mem_prefix_includes_one_state_token_per_history_step():
    config, model = _make_model(history_length=6)
    obs = _make_obs(config)
    prefix_tokens, prefix_mask, _ = model.embed_prefix(obs)

    single_config, single_model = _make_model(history_length=1)
    single_obs = _make_obs(single_config)
    single_prefix_tokens, _, _ = single_model.embed_prefix(single_obs)

    assert prefix_tokens.shape[1] == single_prefix_tokens.shape[1] + config.history_length
    assert prefix_mask.shape[:2] == prefix_tokens.shape[:2]


def test_mem_compute_loss_finite():
    config, model = _make_model(history_length=6)
    obs = _make_obs(config)
    actions = jnp.zeros((2, config.action_horizon, config.action_dim), dtype=jnp.float32)
    loss = model.compute_loss(jax.random.key(1), obs, actions)
    assert loss.shape == (2, config.action_horizon)
    assert jnp.all(jnp.isfinite(loss))


def test_mem_and_ki_compute_loss_finite():
    config, model = _make_model(history_length=6, ki_enabled=True)
    obs = _make_obs(config)
    actions = jnp.zeros((2, config.action_horizon, config.action_dim), dtype=jnp.float32)
    loss = model.compute_loss(jax.random.key(1), obs, actions)
    assert set(loss) == {"flow", "ki_fast"}
    assert jnp.all(jnp.isfinite(loss["flow"]))
    assert jnp.all(jnp.isfinite(loss["ki_fast"]))


def test_long_memory_summary_loss_finite_and_separate():
    config, model = _make_model(history_length=1, long_memory_enabled=True)
    obs = _make_obs(config, batch_size=1)
    actions = jnp.zeros((1, config.action_horizon, config.action_dim), dtype=jnp.float32)

    loss = model.compute_loss(jax.random.key(1), obs, actions)
    assert set(loss) == {"flow", "mem_summary"}
    assert loss["mem_summary"].shape == (1,)
    assert jnp.all(jnp.isfinite(loss["flow"]))
    assert jnp.all(jnp.isfinite(loss["mem_summary"]))


def test_long_memory_summary_loss_mask_can_disable_ce():
    config, model = _make_model(history_length=1, long_memory_enabled=True)
    obs = _make_obs(config, batch_size=1)
    obs = obs.replace(memory_summary_loss_mask=jnp.zeros_like(obs.memory_summary_loss_mask))

    loss = model.compute_memory_summary_loss(obs)
    assert loss.shape == (1,)
    np.testing.assert_allclose(loss, jnp.zeros((1,)), rtol=1e-6, atol=1e-6)


def test_long_memory_summary_loss_does_not_call_action_suffix_path():
    config, model = _make_model(history_length=1, long_memory_enabled=True)
    obs = _make_obs(config, batch_size=1)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("summary CE must not call the action suffix path")

    model.embed_suffix = fail_if_called
    loss = model.compute_memory_summary_loss(obs)
    assert loss.shape == (1,)
    assert jnp.all(jnp.isfinite(loss))


def test_long_memory_generate_summary_tokens_shape():
    config, model = _make_model(history_length=1, long_memory_enabled=True)
    obs = _make_obs(config, batch_size=1)

    tokens = model.generate_memory_summary_tokens(obs, max_new_tokens=2)
    assert tokens.shape == (1, 2)
    assert tokens.dtype == jnp.int32


def test_long_memory_prompt_injection_happens_before_tokenization():
    from openpi import transforms

    transform = transforms.PrependMemorySummaryToPrompt()
    out = transform(
        {
            "prompt": "put the block away",
            "memory_summary": "The red block is in the left drawer.",
        }
    )
    assert out["prompt"] == "Memory: The red block is in the left drawer.\nTask: put the block away"
