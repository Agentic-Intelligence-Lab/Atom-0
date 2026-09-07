"""Audit unified action mappings against the real RLDS builders.

Run from the repository root with the training environment:
    PYTHONPATH=src python scripts/audit_unified_action_space.py
"""

import argparse
import dataclasses
import gc
import json
from pathlib import Path
import tempfile

import numpy as np

from openpi.cotrain import action_space


def _configured_datasets():
    from openpi.cotrain import config  # noqa: PLC0415

    groups = (
        config._AGIBOT_DATA.datasets,
        config._DROID_DATA.datasets,
        config._EGOVERSE_FULL_DATA.datasets,
        config._PIPER30_DATA.datasets,
        config._PIPER2_DATA.datasets,
        config._ROBOCOIN_DATA.datasets,
        config._ROBOMIND_FULL_DATA.datasets,
    )
    datasets = tuple(dataset for group in groups for dataset in group)
    by_id = {dataset.uid: dataset for dataset in datasets}
    if set(by_id) != set(action_space.UNIFIED_ACTION_SPECS):
        missing = sorted(set(action_space.UNIFIED_ACTION_SPECS) - set(by_id))
        extra = sorted(set(by_id) - set(action_space.UNIFIED_ACTION_SPECS))
        raise ValueError(f"Config/spec registry mismatch: missing={missing}, extra={extra}")
    active_ids = {dataset.uid for dataset in config._FULL_ALL_DATA.datasets}
    # cotrain_full_all predates the second in-house Piper drop; the two new mixtures cover it.
    expected_active = set(by_id) - config._FULL_ALL_EXCLUDED_DATASET_IDS - {"piper2"}
    if active_ids != expected_active or len(active_ids) != 41:
        raise ValueError(
            f"full-all active dataset mismatch: expected={sorted(expected_active)}, got={sorted(active_ids)}"
        )
    return tuple(
        dataclasses.replace(dataset, unified_action_spec=action_space.UNIFIED_ACTION_SPECS[dataset.uid])
        for dataset in datasets
    )


def _first_raw_step(dataset):
    import tensorflow_datasets as tfds  # noqa: PLC0415

    builder = tfds.builder_from_directory(dataset.builder_dir)
    episode = next(iter(builder.as_dataset(split=f"{dataset.train_split}[:1]", shuffle_files=False)))
    return next(iter(episode["steps"]))


def _source_arrays(step) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(step["action"])
    state = np.asarray(step["observation"]["state"])
    if action.ndim != 1 or state.ndim != 1:
        raise ValueError(f"Expected rank-1 step state/action, got state={state.shape}, action={action.shape}")
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(state)):
        raise ValueError("First step contains non-finite state/action values")
    return state, action


def audit_raw_builders(datasets) -> list[dict]:
    results = []
    for index, dataset in enumerate(datasets, start=1):
        spec = dataset.unified_action_spec
        step = _first_raw_step(dataset)
        state, actions = _source_arrays(step)
        spec.validate_source_dims(state.shape[-1], actions.shape[-1])

        mapped_state = action_space.map_array(state, spec.state_mapping)
        mapped_actions = action_space.map_array(actions, spec.action_mapping)
        action_mask = np.asarray(spec.action_mask)
        if np.any(mapped_actions[~action_mask] != 0):
            raise ValueError(f"{dataset.uid}: nonzero action outside mask")

        chunk = np.broadcast_to(mapped_actions, (2, action_space.UNIFIED_ACTION_DIM)).copy()
        converted = action_space.apply_delta(mapped_state, chunk, spec.delta_mask)
        delta_mask = np.asarray(spec.delta_mask)
        np.testing.assert_array_equal(converted[:, ~delta_mask], chunk[:, ~delta_mask])

        result = {
            "dataset_id": dataset.uid,
            "state_dim": int(state.shape[-1]),
            "action_dim": int(actions.shape[-1]),
            "active_slots": int(action_mask.sum()),
            "delta_slots": int(delta_mask.sum()),
            "builder_dir": dataset.builder_dir,
        }
        results.append(result)
        print(
            f"[{index:02d}/{len(datasets)}] {dataset.uid}: "
            f"state={state.shape[-1]} action={actions.shape[-1]} "
            f"active={action_mask.sum()} delta={delta_mask.sum()}"
        )
        del step
        gc.collect()
    return results


def audit_mixed_pipeline(datasets) -> dict:
    from openpi.cotrain.rlds_dataset import CotrainRldsDataset  # noqa: PLC0415

    selected_ids = {
        "egoverse_eva",
        "piper30",
        "robocoin_aloha_s26_a26",
        "robocoin_yinhe_s49_a16",
    }
    selected = tuple(
        dataclasses.replace(dataset, weight=1 / len(selected_ids))
        for dataset in datasets
        if dataset.uid in selected_ids
    )
    mixed = CotrainRldsDataset(
        data_dir="/mnt/data/RLDS",
        batch_size=4,
        datasets=selected,
        split_label="train",
        shuffle=False,
        repeat=True,
        action_chunk_size=4,
        pad_action_dim=action_space.UNIFIED_ACTION_DIM,
        image_resize_hw=(32, 32),
        shuffle_buffer_size=1,
        num_parallel_reads=1,
        num_parallel_calls=1,
    )

    seen = set()
    mixed_batch_seen = False
    for batch_index, batch in enumerate(mixed, start=1):
        if batch["state"].shape != (4, action_space.UNIFIED_ACTION_DIM):
            raise ValueError(f"Unexpected state shape: {batch['state'].shape}")
        if batch["actions"].shape != (4, 4, action_space.UNIFIED_ACTION_DIM):
            raise ValueError(f"Unexpected actions shape: {batch['actions'].shape}")
        if batch["action_mask"].shape != (4, action_space.UNIFIED_ACTION_DIM):
            raise ValueError(f"Unexpected action_mask shape: {batch['action_mask'].shape}")

        ids = [value.decode() if isinstance(value, bytes) else str(value) for value in batch["dataset_id"]]
        mixed_batch_seen |= len(set(ids)) > 1
        for row, dataset_id in enumerate(ids):
            spec = action_space.UNIFIED_ACTION_SPECS[dataset_id]
            np.testing.assert_array_equal(batch["action_mask"][row], spec.action_mask)
            assert np.all(batch["actions"][row, :, ~np.asarray(spec.action_mask)] == 0)
            prompt_prefix = batch["prompt_prefix"][row]
            if isinstance(prompt_prefix, bytes):
                prompt_prefix = prompt_prefix.decode()
            expected_mode = "eef" if dataset_id.startswith("egoverse_") else "joint"
            if not (
                prompt_prefix.startswith(f"Embodiment: robot. Action Mode: {expected_mode}.")
                or prompt_prefix.startswith(f"Embodiment: human. Action Mode: {expected_mode}.")
            ):
                raise ValueError(f"{dataset_id}: unexpected prompt prefix {prompt_prefix!r}")
            if expected_mode == "eef" and "EEF Frame:" not in prompt_prefix:
                raise ValueError(f"{dataset_id}: EEF prompt is missing its coordinate frame")
            seen.add(dataset_id)
        if seen == selected_ids and mixed_batch_seen:
            break
        if batch_index >= 40:
            raise RuntimeError(f"Did not observe all selected datasets in mixed batches: seen={sorted(seen)}")

    return {"datasets": sorted(seen), "mixed_batch_seen": mixed_batch_seen, "batches": batch_index}


def audit_model_smoke(datasets) -> dict:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from openpi.cotrain import config  # noqa: PLC0415
    from openpi.cotrain import data_loader  # noqa: PLC0415
    from openpi.models import pi0_config  # noqa: PLC0415
    from openpi.shared import normalize  # noqa: PLC0415

    selected_ids = {"egoverse_eva", "piper30", "robocoin_aloha_s26_a26", "robocoin_yinhe_s49_a16"}
    selected = tuple(
        dataclasses.replace(dataset, weight=1 / len(selected_ids), unified_action_spec=None)
        for dataset in datasets
        if dataset.uid in selected_ids
    )
    model_config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        action_dim=action_space.UNIFIED_ACTION_DIM,
        action_horizon=4,
        max_token_len=384,
    )

    with tempfile.TemporaryDirectory(prefix="unified-action-smoke-") as temp_dir:
        assets_root = Path(temp_dir) / "unified_action_smoke"
        neutral = normalize.NormStats(
            mean=np.zeros(80),
            std=np.ones(80),
            q01=-np.ones(80),
            q99=np.ones(80),
        )
        for dataset in selected:
            directory = assets_root / dataset.uid
            normalize.save(directory, {"state": neutral, "actions": neutral})
            action_space.write_metadata(directory, action_space.UNIFIED_ACTION_SPECS[dataset.uid])

        data_factory = config.CotrainDataConfig(rlds_data_dir="/mnt/data/RLDS", datasets=selected)
        train_config = config.CotrainTrainConfig(
            name="unified_action_smoke",
            model=model_config,
            data=data_factory,
            assets_base_dir=temp_dir,
            batch_size=4,
            exp_name="smoke",
            data_num_parallel_reads=1,
            data_num_parallel_calls=1,
        )
        loader = data_loader.create_cotrain_data_loader(
            train_config,
            split_label="train",
            shuffle=False,
            num_batches=1,
            shuffle_buffer_size=1,
        )
        observation, actions = next(iter(loader))
        if observation.action_mask.shape != (4, 80) or actions.shape != (4, 4, 80):
            raise ValueError(
                f"Unexpected transformed shapes: mask={observation.action_mask.shape}, actions={actions.shape}"
            )
        if observation.tokenized_prompt is None:
            raise ValueError("Transformed batch is missing tokenized prompts")

        model = model_config.create(jax.random.key(0))
        loss = model.compute_loss(jax.random.key(1), observation, actions)
        pred = model.sample_actions(jax.random.key(2), observation, num_steps=2)
        if loss.shape != (4, 4) or not bool(jnp.all(jnp.isfinite(loss))):
            raise ValueError(f"Invalid flow loss: shape={loss.shape}")
        broadcast_mask = jnp.broadcast_to(observation.action_mask[:, None, :], pred.shape)
        if not bool(jnp.all(jnp.where(broadcast_mask, True, pred == 0))):
            raise ValueError("Model sampling produced nonzero values outside the action mask")

    return {
        "action_shape": list(actions.shape),
        "mask_shape": list(observation.action_mask.shape),
        "loss_shape": list(loss.shape),
        "sample_shape": list(pred.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--model-smoke", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    datasets = _configured_datasets()
    raw = [] if args.skip_raw else audit_raw_builders(datasets)
    pipeline = None if args.skip_pipeline else audit_mixed_pipeline(datasets)
    model_smoke = audit_model_smoke(datasets) if args.model_smoke else None
    report = {
        "configured_builder_count": len(datasets),
        "audited_builder_count": len(raw),
        "raw": raw,
        "pipeline": pipeline,
        "model_smoke": model_smoke,
    }
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: raw={len(raw)}/{len(datasets)} builders; pipeline={pipeline}; model={model_smoke}")


if __name__ == "__main__":
    main()
