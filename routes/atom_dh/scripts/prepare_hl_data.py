"""Phase 3: prepare high-level policy (Pi0HL) training labels for RMBench.

The high-level policy learns π_HL(l_{t+1}, m_{t+1} | o_t, m_t, g): given the current memory m_t,
predict the next subtask l and the updated memory m_{t+1}. This script turns *per-frame*
annotations (one row per timestep, as produced by the long-term-memory annotation pipeline)
into a *per-frame* training table that supplies, for every frame:

    memory_summary          = m_t           (the memory the policy holds entering the frame)
    target_subtask          = l             (subtask to generate)
    target_memory_summary   = m_{t+1}        (memory to generate)

The trick for getting both "keep" and "update" supervision from dense per-frame labels:
within each episode we group consecutive frames with the same memory_summary into segments
(a segment = a stretch where the long-term memory is stable). For the first `update_window`
frames of a segment we feed the *previous* segment's summary as m_t (so the model learns to
UPDATE), and elsewhere we feed the same segment's summary (so it learns to KEEP). The first
segment's update window uses an empty m_t (learns to INITIALIZE from observation).

Input (per-frame), one of:
  - JSONL: one JSON object per line, OR
  - Parquet
with at least these columns:
    episode_index : int
    frame_index   : int
    task          : str   (top-level instruction g; may repeat per frame)
    memory_summary: str   (the stable long-term memory at this frame; "" if none yet)
    subtask       : str   (optional; the subtask being executed at this frame)

Output: a Parquet table with columns
    episode_index, frame_index, task, memory_summary, target_subtask, target_memory_summary, sample_kind
to be consumed at train time by transforms.AttachHLTextFromTable (wired in LeRobotHLDataConfig).

Usage:
    python scripts/prepare_hl_data.py --input annotations.jsonl --output data/hl_text.parquet
    python scripts/prepare_hl_data.py --selftest      # offline logic check, no real data
"""

import argparse
import json

import pandas as pd


def _norm(s) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.strip()


def build_hl_table(frames: pd.DataFrame, *, update_window: int = 10) -> pd.DataFrame:
    """frames: per-frame rows with episode_index, frame_index, task, memory_summary[, subtask]."""
    required = {"episode_index", "frame_index", "task", "memory_summary"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"input is missing required columns: {sorted(missing)}")
    if "subtask" not in frames.columns:
        frames = frames.assign(subtask="")

    rows = []
    for ep, ep_df in frames.groupby("episode_index"):
        ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)
        summaries = [_norm(s) for s in ep_df["memory_summary"].tolist()]

        # Segment = run of consecutive equal summaries. seg_id[i] and seg_start_summary per segment.
        seg_ids = [0] * len(summaries)
        seg_start_frame = [0] * len(summaries)
        prev_seg_summary = {0: ""}  # segment id -> previous segment's summary ("" before first)
        cur_seg = 0
        seg_ids[0] = 0
        seg_start_frame[0] = 0
        for i in range(1, len(summaries)):
            if summaries[i] != summaries[i - 1]:
                cur_seg += 1
                prev_seg_summary[cur_seg] = summaries[i - 1]
                seg_start_frame_val = i
            else:
                seg_start_frame_val = seg_start_frame[i - 1]
            seg_ids[i] = cur_seg
            seg_start_frame[i] = seg_start_frame_val

        for i in range(len(summaries)):
            s = seg_ids[i]
            within = i - seg_start_frame[i]
            target_mem = summaries[i]
            if within < update_window:
                if s == 0:
                    m_t = ""
                    kind = "init"
                else:
                    m_t = prev_seg_summary.get(s, "")
                    kind = "update"
            else:
                m_t = summaries[i]
                kind = "keep"
            rows.append(
                {
                    "episode_index": int(ep),
                    "frame_index": int(ep_df.loc[i, "frame_index"]),
                    "task": _norm(ep_df.loc[i, "task"]),
                    "memory_summary": m_t,
                    "target_subtask": _norm(ep_df.loc[i, "subtask"]),
                    "target_memory_summary": target_mem,
                    "sample_kind": kind,
                }
            )
    return pd.DataFrame(rows)


def _read_input(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl"):
        with open(path) as f:
            return pd.DataFrame([json.loads(line) for line in f if line.strip()])
    if path.endswith(".json"):
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"unsupported input extension: {path}")


def _read_lerobot(root: str) -> pd.DataFrame:
    """Read a LeRobot dataset dir that already has per-frame `subtask` and `memory` columns.

    Pulls episode_index, frame_index, subtask, memory from the data parquets and the top-level
    `task` string from meta/tasks.jsonl (via task_index). Matches the RMBench conversion schema.
    """
    import glob
    import os

    # task_index -> task text
    task_by_index = {}
    tasks_path = os.path.join(root, "meta", "tasks.jsonl")
    with open(tasks_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                task_by_index[int(rec["task_index"])] = rec["task"]

    parts = []
    files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "episode_*.parquet")))
    if not files:
        raise ValueError(f"no episode parquets found under {root}/data/chunk-*/")
    for fp in files:
        df = pd.read_parquet(fp, columns=["episode_index", "frame_index", "subtask", "memory", "task_index"])
        df = df.rename(columns={"memory": "memory_summary"})
        df["task"] = df["task_index"].map(lambda i: task_by_index.get(int(i), ""))
        parts.append(df[["episode_index", "frame_index", "task", "memory_summary", "subtask"]])
    return pd.concat(parts, ignore_index=True)


def _selftest():
    frames = pd.DataFrame(
        [
            # episode 0: seg A (frames 0-2), seg B (3-5)
            {"episode_index": 0, "frame_index": 0, "task": "T", "memory_summary": "A", "subtask": "open"},
            {"episode_index": 0, "frame_index": 1, "task": "T", "memory_summary": "A", "subtask": "open"},
            {"episode_index": 0, "frame_index": 2, "task": "T", "memory_summary": "A", "subtask": "open"},
            {"episode_index": 0, "frame_index": 3, "task": "T", "memory_summary": "B", "subtask": "grab"},
            {"episode_index": 0, "frame_index": 4, "task": "T", "memory_summary": "B", "subtask": "grab"},
            {"episode_index": 0, "frame_index": 5, "task": "T", "memory_summary": "B", "subtask": "grab"},
        ]
    )
    out = build_hl_table(frames, update_window=2)
    by_frame = {int(r.frame_index): r for r in out.itertuples()}
    # frame 0,1: init (first segment update window) -> m_t == ""
    assert by_frame[0].memory_summary == "" and by_frame[0].sample_kind == "init", by_frame[0]
    assert by_frame[1].memory_summary == "" and by_frame[1].sample_kind == "init"
    # frame 2: keep within seg A -> m_t == "A"
    assert by_frame[2].memory_summary == "A" and by_frame[2].sample_kind == "keep"
    # frame 3,4: update window of seg B -> m_t == "A" (prev), target == "B"
    assert by_frame[3].memory_summary == "A" and by_frame[3].target_memory_summary == "B"
    assert by_frame[3].sample_kind == "update"
    # frame 5: keep within seg B -> m_t == "B"
    assert by_frame[5].memory_summary == "B" and by_frame[5].sample_kind == "keep"
    # targets carry subtask
    assert by_frame[3].target_subtask == "grab"
    print("PASS: prepare_hl_data selftest (init/update/keep derivation correct)")
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="per-frame annotations (.jsonl/.json/.parquet)")
    ap.add_argument("--lerobot_root", help="LeRobot dataset dir with per-frame subtask/memory columns")
    ap.add_argument("--output", help="output parquet path")
    ap.add_argument("--update_window", type=int, default=10, help="# frames at a segment start labeled as UPDATE")
    ap.add_argument("--selftest", action="store_true", help="run offline logic check and exit")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.output or not (args.input or args.lerobot_root):
        ap.error("provide --output and one of --input / --lerobot_root (or use --selftest)")

    frames = _read_lerobot(args.lerobot_root) if args.lerobot_root else _read_input(args.input)
    table = build_hl_table(frames, update_window=args.update_window)
    table.to_parquet(args.output, index=False)
    counts = table["sample_kind"].value_counts().to_dict()
    print(f"wrote {len(table)} rows -> {args.output}; sample_kind counts: {counts}")


if __name__ == "__main__":
    main()
