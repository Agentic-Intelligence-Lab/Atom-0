"""Pre-cache FAST tokens for a LeRobot dataset.

FAST tokenizer init (AutoProcessor.from_pretrained) takes ~30s per worker process
and the per-sample neural tokenization is also non-trivial for large datasets.
This script pre-tokenizes all samples once and saves the result as a sidecar
directory of .npz files, one per episode.

The cached files can be loaded by a custom transform (CachedKITokenize) to avoid
re-running the tokenizer in every training worker.

Usage:
    cd Atom-0
    uv run scripts/cache_fast_tokens.py \\
        --repo-id physical-intelligence/libero \\
        --action-horizon 10 \\
        --out-dir ./cache/libero_fast_tokens \\
        --num-workers 4

Output layout:
    <out-dir>/
        episode_000000.npz   # ki_fast_tokens, ki_fast_mask, token_ar_mask, token_loss_mask
        episode_000001.npz
        ...
        meta.json            # {repo_id, action_horizon, ki_fast_max_len, n_episodes}
"""

import argparse
import json
import logging
import os
import pathlib
import sys

import numpy as np
import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from openpi.models import tokenizer as _tokenizer


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _get_episode_data(dataset, episode_idx: int):
    """Return all frames for one episode as a list of dicts."""
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx   = dataset.episode_data_index["to"][episode_idx].item()
    frames = []
    for i in range(from_idx, to_idx):
        frames.append(dataset[i])
    return frames


def _tokenize_episode(frames: list[dict], fast_tok: _tokenizer.FASTTokenizer, action_horizon: int):
    """Run FASTTokenizer on each frame of one episode.

    For frame i we tokenize:
        prompt  = frames[i]["prompt"]   (task string)
        state   = frames[i]["state"]    (continuous robot state)
        actions = frames[i]["actions"]  (action chunk of shape [action_horizon, action_dim])

    Returns per-frame arrays stacked into episode arrays.
    """
    all_tokens   = []
    all_masks    = []
    all_ar_masks = []
    all_loss_masks = []

    for frame in frames:
        prompt  = frame.get("prompt", "")
        if isinstance(prompt, (bytes, np.bytes_)):
            prompt = prompt.decode()

        state   = np.array(frame["observation.state"]) if "observation.state" in frame else np.array(frame.get("state", np.zeros(1)))
        actions = np.array(frame.get("action", frame.get("actions", np.zeros((action_horizon, state.shape[-1])))))

        # Ensure action chunk has correct horizon
        if actions.ndim == 1:
            # single-step action; repeat to fill horizon
            actions = np.stack([actions] * action_horizon, axis=0)
        actions = actions[:action_horizon]

        tokens, mask, ar_mask, loss_mask = fast_tok.tokenize(
            prompt=prompt,
            state=state,
            actions=actions,
        )
        all_tokens.append(tokens)
        all_masks.append(mask)
        all_ar_masks.append(ar_mask)
        all_loss_masks.append(loss_mask)

    return {
        "ki_fast_tokens":  np.stack(all_tokens,    axis=0).astype(np.int32),
        "ki_fast_mask":    np.stack(all_masks,     axis=0).astype(bool),
        "token_ar_mask":   np.stack(all_ar_masks,  axis=0).astype(np.int32),
        "token_loss_mask": np.stack(all_loss_masks, axis=0).astype(bool),
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-cache FAST tokens for a LeRobot dataset.")
    parser.add_argument("--repo-id",         default="physical-intelligence/libero")
    parser.add_argument("--action-horizon",  type=int, default=10)
    parser.add_argument("--ki-fast-max-len", type=int, default=256,
                        help="Max token sequence length passed to FASTTokenizer.")
    parser.add_argument("--out-dir",         default="./cache/libero_fast_tokens")
    parser.add_argument("--fast-tokenizer-path", default="physical-intelligence/fast",
                        help="HuggingFace repo or local path for the FAST AutoProcessor.")
    parser.add_argument("--num-workers",     type=int, default=1,
                        help="Number of parallel worker processes (each initialises the tokenizer once).")
    parser.add_argument("--overwrite",       action="store_true",
                        help="Re-compute even if the output directory already exists.")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if out_dir.exists() and not args.overwrite:
        log.info(f"Output directory {out_dir} already exists. Use --overwrite to recompute.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {out_dir}")

    # -- load dataset --
    log.info(f"Loading LeRobot dataset '{args.repo_id}' ...")
    try:
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        ds = lerobot_dataset.LeRobotDataset(repo_id=args.repo_id)
    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    n_episodes = ds.num_episodes
    log.info(f"Dataset loaded: {len(ds)} frames, {n_episodes} episodes.")

    # -- init tokenizer (once in main process; workers inherit via fork) --
    log.info(f"Initialising FASTTokenizer (path='{args.fast_tokenizer_path}') ...")
    fast_tok = _tokenizer.FASTTokenizer(
        max_len=args.ki_fast_max_len,
        fast_tokenizer_path=args.fast_tokenizer_path,
    )
    log.info("FASTTokenizer ready.")

    # -- process episodes --
    def process_episode(ep_idx: int):
        out_path = out_dir / f"episode_{ep_idx:06d}.npz"
        if out_path.exists() and not args.overwrite:
            return

        frames = _get_episode_data(ds, ep_idx)
        arrays = _tokenize_episode(frames, fast_tok, args.action_horizon)
        np.savez_compressed(out_path, **arrays)

    if args.num_workers > 1:
        from multiprocessing.pool import Pool
        with Pool(processes=args.num_workers) as pool:
            list(tqdm.tqdm(
                pool.imap(process_episode, range(n_episodes)),
                total=n_episodes,
                desc="Caching episodes",
            ))
    else:
        for ep_idx in tqdm.tqdm(range(n_episodes), desc="Caching episodes"):
            process_episode(ep_idx)

    # -- write metadata --
    meta = {
        "repo_id":        args.repo_id,
        "action_horizon": args.action_horizon,
        "ki_fast_max_len": args.ki_fast_max_len,
        "n_episodes":     n_episodes,
        "n_frames":       len(ds),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info(f"Done. Cached {n_episodes} episodes to {out_dir}")
    log.info("To use the cache, pass --fast-tokens-cache-dir to the KITokenize transform.")


if __name__ == "__main__":
    main()
