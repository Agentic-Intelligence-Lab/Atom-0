"""Convert LIBERO RLDS data to a LeRobot dataset excluding LIBERO-10.

This is the KI-V3 hold-out dataset builder. It creates a LeRobot dataset from:

    libero_goal_no_noops
    libero_object_no_noops
    libero_spatial_no_noops

and intentionally excludes:

    libero_10_no_noops

The resulting dataset can be used to train KI/no-KI on spatial/object/goal and
evaluate held-out harder-suite performance on libero_10.

Example:
    HF_LEROBOT_HOME=/mnt/data/lerobot \
    uv run scripts/convert_libero_no10_to_lerobot.py \
      --data-dir /mnt/data/libero_rlds \
      --repo-name xule/libero_no10
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro


RAW_DATASET_NAMES = [
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]


def main(
    data_dir: str,
    *,
    repo_name: str = "xule/libero_no10",
    push_to_hub: bool = False,
    overwrite: bool = False,
) -> None:
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for raw_dataset_name in RAW_DATASET_NAMES:
        print(f"[INFO] Loading {raw_dataset_name} from {data_dir}")
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        episode_count = 0
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()
            episode_count += 1
        print(f"[INFO] Saved {episode_count} episodes from {raw_dataset_name}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "ki-v3", "exclude-libero-10"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"[INFO] Done. LeRobot dataset written to: {output_path}")
    print(f"[INFO] repo_id: {repo_name}")


if __name__ == "__main__":
    tyro.cli(main)
