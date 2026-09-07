"""Fixed-quota mixture weights for Atom-0-v3 ablation (does not change episode mode)."""

from __future__ import annotations

import dataclasses
from typing import Iterable

from openpi.cotrain.rlds_dataset import CotrainRLDSDataset

LAB_HUMAN = frozenset({
    "aligned_hangzhou_human_right",
    "aligned_shenzhen_human_bimanual",
})
LAB_ROBOT = frozenset({
    "aligned_hangzhou_robot_right",
    "aligned_shenzhen_robot_bimanual",
})
RL2_ROBOT = frozenset({"egoverse_rl2_eva", "egoverse_rl2_robot"})
RL2_ID = frozenset({"egoverse_rl2_indomain"})
RL2_DIV = frozenset({"egoverse_rl2_diverse"})
# fallback if ID/diverse not split yet:
RL2_HUMAN_FALLBACK = frozenset({"egoverse_rl2_human"})

EV_HUMAN = frozenset({"egoverse_aria", "egoverse_human", "egoverse_mecka"})
EV_ROBOT = frozenset({"egoverse_eva"})
PIPER = frozenset({"piper30", "piper2"})


def _uid(ds: CotrainRLDSDataset) -> str:
    return ds.uid


def _renorm(weights: dict[str, float]) -> dict[str, float]:
    s = sum(weights.values())
    if s <= 0:
        raise ValueError("mixture weights sum to 0")
    return {k: v / s for k, v in weights.items()}


def _split_equal(uids: list[str], budget: float) -> dict[str, float]:
    if not uids or budget <= 0:
        return {}
    w = budget / len(uids)
    return {u: w for u in uids}


def _split_by_hours(uids: list[str], hours: dict[str, float], budget: float) -> dict[str, float]:
    if not uids or budget <= 0:
        return {}
    tot = sum(max(hours.get(u, 0.0), 0.0) for u in uids)
    if tot <= 0:
        return _split_equal(uids, budget)
    return {u: budget * (hours.get(u, 0.0) / tot) for u in uids}


def apply_fixed_quota(
    datasets: tuple[CotrainRLDSDataset, ...],
    *,
    p_other: float,
    lab_in_align: float,
    o_ev_human: float = 0.02,
    # optional hour priors for Other-rest / Align-by-duration in round1
    hours_by_uid: dict[str, float] | None = None,
    align_by_duration: bool = False,
) -> tuple[CotrainRLDSDataset, ...]:
    """Rewrite dataset weights to sum to 1 under the ablation recipe."""
    if not (0.0 < p_other < 1.0):
        raise ValueError("p_other must be in (0,1)")
    if not (0.0 <= lab_in_align <= 1.0):
        raise ValueError("lab_in_align must be in [0,1]")
    if o_ev_human < 0 or o_ev_human >= p_other:
        raise ValueError("o_ev_human must satisfy 0 <= o_ev_human < p_other")

    uids = [_uid(ds) for ds in datasets]
    present = set(uids)
    hours = hours_by_uid or {}

    p_align = 1.0 - p_other
    if align_by_duration:
        lab_uids = [u for u in uids if u in (LAB_HUMAN | LAB_ROBOT)]
        rl2_uids = [u for u in uids if u in (RL2_ROBOT | RL2_ID | RL2_DIV | RL2_HUMAN_FALLBACK)]
        lab_h = sum(max(hours.get(u, 0.0), 0.0) for u in lab_uids)
        rl2_h = sum(max(hours.get(u, 0.0), 0.0) for u in rl2_uids)
        if lab_h + rl2_h <= 0:
            raise ValueError("align_by_duration: lab/rl2 hour proxies sum to 0")
        r_lab = lab_h / (lab_h + rl2_h)
    else:
        r_lab = lab_in_align
    p_lab = p_align * r_lab
    p_rl2 = p_align * (1.0 - r_lab)

    out: dict[str, float] = {u: 0.0 for u in uids}

    # ---- Lab: human:robot = 1:1 ----
    lab_h = [u for u in uids if u in LAB_HUMAN]
    lab_r = [u for u in uids if u in LAB_ROBOT]
    out.update(_split_equal(lab_h, 0.5 * p_lab))
    out.update(_split_equal(lab_r, 0.5 * p_lab))

    # ---- RL2: robot:human = 1:1; human ID:diverse = 2:8 -> 5:1:4 ----
    rl2_r = [u for u in uids if u in RL2_ROBOT]
    rl2_id = [u for u in uids if u in RL2_ID]
    rl2_div = [u for u in uids if u in RL2_DIV]
    rl2_h_fb = [u for u in uids if u in RL2_HUMAN_FALLBACK]

    out.update(_split_equal(rl2_r, 0.5 * p_rl2))
    if rl2_id or rl2_div:
        out.update(_split_equal(rl2_id, 0.1 * p_rl2))
        out.update(_split_equal(rl2_div, 0.4 * p_rl2))
    else:
        # fallback: single human blob gets all human half
        out.update(_split_equal(rl2_h_fb, 0.5 * p_rl2))

    # ---- Other ----
    out.update(_split_by_hours([u for u in uids if u in EV_HUMAN], hours, o_ev_human))
    rest = p_other - o_ev_human
    other_rest = [
        u for u in uids
        if u not in (LAB_HUMAN | LAB_ROBOT | RL2_ROBOT | RL2_ID | RL2_DIV | RL2_HUMAN_FALLBACK | EV_HUMAN)
    ]
    # prefer hour split; else equal
    out.update(_split_by_hours(other_rest, hours, rest))

    out = _renorm(out)
    return tuple(dataclasses.replace(ds, weight=out[_uid(ds)]) for ds in datasets)