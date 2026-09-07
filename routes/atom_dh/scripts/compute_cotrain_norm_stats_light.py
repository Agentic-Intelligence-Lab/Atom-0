"""Compute per-dataset cotrain norm stats without decoding images.

This script intentionally does not replace compute_cotrain_norm_stats.py. It keeps the
same state/action semantics while avoiding image and prompt materialization:

  RLDS -> lightweight state/action restructure -> state/action index selection
       -> action chunking -> per-dataset delta actions -> RunningStats

Use --verify-against-old on a small frame count to compare against the original full
pipeline before computing production stats.
"""

import dataclasses
from itertools import islice
import json
from pathlib import Path
import tempfile

import numpy as np
import tqdm
import tyro

from openpi.cotrain import action_space as cotrain_action_space
import openpi.cotrain.config as cotrain_config
import openpi.cotrain.rlds_dataset as cotrain_rlds_dataset
import openpi.shared.download as download
import openpi.shared.normalize as normalize
from openpi.training.data_loader import IterableTransformedDataset
import openpi.transforms as _transforms


def _light_restructure(traj, dataset_id: str, restructure_name: str):
    """Return only state/actions/dataset_id, matching cotrain standardized restructures."""
    import tensorflow as tf
    if restructure_name == "standardized":
        n = tf.shape(traj["actions"])[0]
        return {
            "actions": traj["actions"],
            "state": traj["state"],
            "dataset_id": tf.fill([n], dataset_id),
        }
    # eva: full 14D EE+gripper (not raw 12D action).
    if restructure_name in ("egoverse_eva", "egoverse_rl2_eva"):
        full = cotrain_rlds_dataset._egoverse_eva_restructure(traj, dataset_id)
        return {
            "actions": full["actions"],
            "state": full["state"],
            "dataset_id": full["dataset_id"],
        }

    if restructure_name == "aligned_parallel_gripper":
        n = tf.shape(traj["actions"])[0]
        return {
            "actions": traj["actions"],
            "state": traj["state"],
            "dataset_id": tf.fill([n], dataset_id),
        }

    if restructure_name in {
        "agibot",
        "robomind",
        "three_cam_task",
        "piper2",
        "egoverse_mecka",
        "egoverse_full",
        "robocoin",
        "robomind_full",
    }:
        n = tf.shape(traj["action"])[0]
        return {
            "actions": traj["action"],
            "state": traj["observation"]["state"],
            "dataset_id": tf.fill([n], dataset_id),
        }
    raise ValueError(f"Unsupported lightweight restructure_name: {restructure_name!r}")


def _make_filter_table(filter_dict_path: str | None):
    """Build the same DROID frame filter table as the full loader."""
    import tensorflow as tf

    if filter_dict_path is None:
        return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True)

    cached_filter_dict_path = download.maybe_download(filter_dict_path)
    with Path(cached_filter_dict_path).open("r") as f:
        filter_dict = json.load(f)

    keys_tensor = []
    values_tensor = []
    for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
        for start, end in ranges:
            for t in range(start, end):
                keys_tensor.append(f"{episode_key}--{t}")
                values_tensor.append(True)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
    )


def _create_light_dataset(
    data_config,
    dataset_cfg: cotrain_rlds_dataset.CotrainRLDSDataset,
    action_horizon: int,
    batch_size: int,
    *,
    split_label: str = "train",
    shuffle: bool = False,
    repeat: bool | None = None,
    drop_remainder: bool = True,
    num_parallel_reads: int = -1,
    num_parallel_calls: int = -1,
):
    """Create a batched RLDS iterator containing only state/actions/dataset_id."""
    import dlimp as dl
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    if repeat is None:
        repeat = split_label == "train"

    split_name = dataset_cfg.resolve_split(split_label)
    if dataset_cfg.builder_dir is not None:
        builder = tfds.builder_from_directory(dataset_cfg.builder_dir)
    else:
        builder = tfds.builder(dataset_cfg.name, data_dir=data_config.rlds_data_dir, version=dataset_cfg.version)

    dataset = dl.DLataset.from_rlds(builder, split=split_name, shuffle=shuffle, num_parallel_reads=num_parallel_reads)

    def chunk_actions(traj):
        traj_len = tf.shape(traj["actions"])[0]
        action_chunk_indices = tf.broadcast_to(
            tf.range(action_horizon)[None],
            [traj_len, action_horizon],
        ) + tf.broadcast_to(
            tf.range(traj_len)[:, None],
            [traj_len, action_horizon],
        )
        action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
        traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
        return traj

    def select_state_actions(traj):
        if dataset_cfg.state_indices is not None:
            traj["state"] = tf.gather(traj["state"], tf.constant(dataset_cfg.state_indices, tf.int32), axis=-1)
        if dataset_cfg.action_indices is not None:
            traj["actions"] = tf.gather(traj["actions"], tf.constant(dataset_cfg.action_indices, tf.int32), axis=-1)
        return traj

    if dataset_cfg.restructure_name in cotrain_rlds_dataset.STD_RESTRUCTURE_FNS:
        if dataset_cfg.restructure_name in cotrain_rlds_dataset._EGO_EVA_GRIPPER_FILTER_NAMES:
            dataset = dataset.filter(cotrain_rlds_dataset._egoverse_eva_gripper_fields_finite)
        if dataset_cfg.episode_task_name_regex is not None:
            _pat = dataset_cfg.episode_task_name_regex
            dataset = dataset.filter(
                lambda traj, p=_pat: tf.strings.regex_full_match(
                    traj["traj_metadata"]["episode_metadata"]["task_name"][0], p
                )
            )
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.traj_map(
            lambda traj: _light_restructure(traj, dataset_cfg.uid, dataset_cfg.restructure_name),
            num_parallel_calls,
        )
        if dataset_cfg.unified_action_spec is not None:
            dataset = dataset.traj_map(
                lambda traj: cotrain_action_space.map_trajectory_tensorflow(traj, dataset_cfg.unified_action_spec),
                num_parallel_calls,
            )
        elif dataset_cfg.state_indices is not None or dataset_cfg.action_indices is not None:
            dataset = dataset.traj_map(select_state_actions, num_parallel_calls)

        # EgoVerse eva: already [T, 100, D]; resample like full train loader.
        if dataset_cfg.restructure_name in ("egoverse_eva", "egoverse_rl2_eva","aligned_parallel_gripper"):
            def _resample_ego(traj):
                actions = traj["actions"]  # [T, 100, D]
                src_len = tf.shape(actions)[1]
                idx = tf.cast(
                    tf.round(
                        tf.linspace(0.0, tf.cast(src_len - 1, tf.float32), action_horizon)
                    ),
                    tf.int32,
                )
                traj["actions"] = tf.gather(actions, idx, axis=1)  # [T, H, D]
                return traj

            dataset = dataset.traj_map(_resample_ego, num_parallel_calls)
        else:
            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
    else:
        # Legacy DROID path, kept for compatibility with older cotrain configs.
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )
        if repeat:
            dataset = dataset.repeat()
        filter_table = _make_filter_table(dataset_cfg.filter_dict_path)
        restructure_fn = cotrain_rlds_dataset.RESTRUCTURE_FNS[dataset_cfg.restructure_name]

        def legacy_restructure(traj):
            full = restructure_fn(traj, data_config.action_space, filter_table)
            n = tf.shape(full["actions"])[0]
            return {
                "actions": full["actions"],
                "state": tf.concat(
                    [full["observation"]["joint_position"], full["observation"]["gripper_position"]],
                    axis=-1,
                ),
                "passes_filter": full["passes_filter"],
                "dataset_id": tf.fill([n], dataset_cfg.uid),
            }

        dataset = dataset.traj_map(legacy_restructure, num_parallel_calls)
        if dataset_cfg.state_indices is not None or dataset_cfg.action_indices is not None:
            dataset = dataset.traj_map(select_state_actions, num_parallel_calls)
        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(lambda frame: frame["passes_filter"])

        def remove_filter(frame):
            frame.pop("passes_filter")
            return frame

        dataset = dataset.map(remove_filter)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.with_ram_budget(1)
    return dataset


def _resolve_light_data_config(config):
    """Resolve unified mappings without loading tokenizer or any existing norm stats."""
    datasets = cotrain_config._resolve_unified_datasets(config.data.datasets, config.model)
    return dataclasses.replace(config.data, datasets=datasets)


def _state_actions_from_light_batch(batch: dict, dataset_cfg: cotrain_rlds_dataset.CotrainRLDSDataset):
    state = np.asarray(batch["state"])
    state = state[:, -1] if state.ndim == 3 else state
    actions = np.array(batch["actions"])

    if dataset_cfg.unified_action_spec is not None:
        actions = cotrain_action_space.apply_delta(state, actions, dataset_cfg.unified_action_spec.delta_mask)
    elif dataset_cfg.delta_action_mask_dims is not None:
        mask = np.asarray(_transforms.make_bool_mask(*dataset_cfg.delta_action_mask_dims))
        actions = cotrain_action_space.apply_delta(state, actions, mask)
    return state, actions


def _empty_stats():
    return {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}


def _update_stats(stats: dict, state: np.ndarray, actions: np.ndarray):
    stats["state"].update(state.reshape(-1, state.shape[-1]))
    stats["actions"].update(actions.reshape(-1, actions.shape[-1]))


def _neutralize_inactive_stats(stats, mask: tuple[bool, ...]):
    mask = np.asarray(mask, dtype=bool)
    mean = np.asarray(stats.mean).copy()
    std = np.asarray(stats.std).copy()
    q01 = None if stats.q01 is None else np.asarray(stats.q01).copy()
    q99 = None if stats.q99 is None else np.asarray(stats.q99).copy()
    mean[~mask] = 0
    std[~mask] = 1
    if q01 is not None:
        q01[~mask] = -1
    if q99 is not None:
        q99[~mask] = 1
    return normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)


def _finalize_stats(stats: dict, dataset_cfg):
    finalized = {key: value.get_statistics() for key, value in stats.items()}
    spec = dataset_cfg.unified_action_spec
    if spec is None:
        return finalized
    state_targets = set(spec.state_target_slots)
    state_mask = tuple(index in state_targets for index in range(cotrain_action_space.UNIFIED_ACTION_DIM))
    return {
        "state": _neutralize_inactive_stats(finalized["state"], state_mask),
        "actions": _neutralize_inactive_stats(finalized["actions"], spec.action_mask),
    }


def _compute_light_stats(config, data_config, dataset_cfg, max_frames: int, *, show_progress: bool = True):
    batch_size = config.batch_size
    num_batches = max(1, max_frames // batch_size)
    dataset = _create_light_dataset(data_config, dataset_cfg, config.model.action_horizon, batch_size)

    stats = _empty_stats()
    n_frames = 0
    iterator = islice(iter(dataset.as_numpy_iterator()), num_batches)
    if show_progress:
        iterator = tqdm.tqdm(iterator, total=num_batches, desc=dataset_cfg.name)
    for batch in iterator:
        state, actions = _state_actions_from_light_batch(batch, dataset_cfg)
        _update_stats(stats, state, actions)
        n_frames += int(state.shape[0])
    return _finalize_stats(stats, dataset_cfg), n_frames


def _compute_light_stats_deterministic(config, data_config, dataset_cfg, max_frames: int):
    batch_size = config.batch_size
    num_batches = max(1, max_frames // batch_size)
    dataset = _create_light_dataset(
        data_config,
        dataset_cfg,
        config.model.action_horizon,
        batch_size,
        num_parallel_reads=1,
        num_parallel_calls=1,
    )

    stats = _empty_stats()
    n_frames = 0
    for batch in tqdm.tqdm(
        islice(iter(dataset.as_numpy_iterator()), num_batches),
        total=num_batches,
        desc=f"{dataset_cfg.name} light",
    ):
        state, actions = _state_actions_from_light_batch(batch, dataset_cfg)
        _update_stats(stats, state, actions)
        n_frames += int(state.shape[0])
    return _finalize_stats(stats, dataset_cfg), n_frames


def _compute_old_stats(config, data_config, dataset_cfg, max_frames: int):
    batch_size = config.batch_size
    num_batches = max(1, max_frames // batch_size)
    single_dc = dataclasses.replace(data_config, datasets=(dataclasses.replace(dataset_cfg, weight=1.0),))
    dataset = cotrain_rlds_dataset.CotrainRldsDataset(
        data_dir=single_dc.rlds_data_dir,
        batch_size=batch_size,
        datasets=single_dc.datasets,
        split_label="train",
        shuffle=False,
        action_chunk_size=config.model.action_horizon,
        action_space=single_dc.action_space,
        num_parallel_reads=1,
        num_parallel_calls=1,
    )
    dataset = IterableTransformedDataset(
        dataset,
        [*single_dc.repack_transforms.inputs, *single_dc.data_transforms.inputs],
        is_batched=True,
    )

    stats = _empty_stats()
    n_frames = 0
    for batch in tqdm.tqdm(islice(iter(dataset), num_batches), total=num_batches, desc=f"{dataset_cfg.name} old"):
        state = np.asarray(batch["state"])
        actions = np.asarray(batch["actions"])
        _update_stats(stats, state, actions)
        n_frames += int(state.shape[0])
    return _finalize_stats(stats, dataset_cfg), n_frames


def _max_abs_diff(a, b) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _verify_against_old(config, data_config, dataset_cfg, verify_frames: int, tolerance: float):
    print(f"\n=== Verifying lightweight stats against old pipeline for '{dataset_cfg.uid}' ===")
    old_stats, old_frames = _compute_old_stats(config, data_config, dataset_cfg, verify_frames)
    light_stats, light_frames = _compute_light_stats_deterministic(config, data_config, dataset_cfg, verify_frames)
    print(f"  old frames:   {old_frames}")
    print(f"  light frames: {light_frames}")

    ok = old_frames == light_frames
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            diff = _max_abs_diff(getattr(old_stats[key], field), getattr(light_stats[key], field))
            print(f"  max_abs_diff[{key}.{field}] = {diff:.8g}")
            ok = ok and diff <= tolerance
    if not ok:
        raise RuntimeError(f"Lightweight stats verification failed for '{dataset_cfg.uid}' (tolerance={tolerance}).")
    print("  verification passed")


def main(
    config_name: str,
    exp_name: str,
    max_frames: int = 1_000_000,
    rlds_data_dir: str | None = None,
    overwrite: bool = False,
    dataset_id: str | None = None,
    verify_against_old: bool = False,
    verify_frames: int = 1024,
    verify_tolerance: float = 1e-5,
) -> None:
    config = cotrain_config.get_config(config_name)
    config = dataclasses.replace(config, exp_name=exp_name)
    if rlds_data_dir is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, rlds_data_dir=rlds_data_dir))
    if verify_against_old:
        # The old pipeline needs the full transforms, but verification must not normalize
        # with stale/existing stats. Build those transforms against a guaranteed-empty root.
        with tempfile.TemporaryDirectory(prefix="cotrain-norm-verify-assets-") as empty_assets:
            data_config = config.data.create(Path(empty_assets), config.model)
    else:
        data_config = _resolve_light_data_config(config)

    selected = [ds for ds in data_config.datasets if dataset_id is None or ds.uid == dataset_id]
    if not selected:
        raise ValueError(f"No dataset matched dataset_id={dataset_id!r}")

    if verify_against_old:
        for ds in selected:
            _verify_against_old(config, data_config, ds, verify_frames, verify_tolerance)
        return

    for ds in selected:
        out_dir = config.assets_dirs / ds.uid
        if not overwrite:
            try:
                normalize.load(out_dir)
                print(f"\n=== Skipping '{ds.uid}': norm stats already exist at {out_dir} (use --overwrite to redo) ===")
                continue
            except FileNotFoundError:
                pass

        print(f"\n=== Computing LIGHT norm stats for dataset '{ds.uid}' (split='{ds.train_split}') ===")
        norm_stats, n_frames = _compute_light_stats(config, data_config, ds, max_frames)
        if n_frames == 0:
            raise RuntimeError(f"No frames read for dataset '{ds.uid}' (split '{ds.train_split}').")
        print(f"  accumulated {n_frames} frames")
        normalize.save(out_dir, norm_stats)
        if ds.unified_action_spec is not None:
            cotrain_action_space.write_metadata(out_dir, ds.unified_action_spec)
        print(f"Saved norm stats for '{ds.uid}' to {out_dir}")


if __name__ == "__main__":
    tyro.cli(main)
