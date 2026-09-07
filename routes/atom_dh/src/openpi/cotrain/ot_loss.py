"""EgoBridge-style Soft-DTW matching + Sinkhorn OT (JAX).

Faithful to egomimic/algo/hpt.py::compute_ot with supervised+dtw:
  Soft-DTW on GT xyz -> W (lambda on best match) -> Sinkhorn on flattened tokens
  with cost = 0.5 ||x-y||^2 * W, blur=0.05, 18 iters.
Padded KxK buffer with validity masks (rectangular OT on real human/robot supports).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Unified 80D slot indices (action_space.py)
RIGHT_EEF_XYZ = (36, 37, 38)
LEFT_EEF_XYZ = (7, 8, 9)


def soft_dtw_distance(x, y, gamma: float = 0.1):
    """x,y: [T, D] -> scalar Soft-DTW."""
    diff = x[:, None, :] - y[None, :, :]
    c = jnp.sum(diff * diff, axis=-1)
    t = c.shape[0]

    def first_row(carry, j):
        prev = carry
        val = jnp.where(j == 0, c[0, 0], c[0, j] + prev)
        return val, val

    _, r0 = jax.lax.scan(first_row, jnp.asarray(0.0, dtype=c.dtype), jnp.arange(t))

    def body(i, r_prev):
        def scan_j(j_carry, j):
            r_i_jm1 = j_carry
            cand = jnp.stack(
                [r_prev[j], r_i_jm1, jnp.where(j > 0, r_prev[j - 1], jnp.inf)]
            )
            cand = jnp.where(j == 0, jnp.stack([r_prev[0], jnp.inf, jnp.inf]), cand)
            rij = c[i, j] + (-gamma * jax.nn.logsumexp(-cand / gamma, axis=0))
            return rij, rij

        _, row = jax.lax.scan(scan_j, jnp.asarray(jnp.inf, dtype=c.dtype), jnp.arange(t))
        return row

    r_last = jax.lax.fori_loop(1, t, body, r0)
    return r_last[t - 1]


def pairwise_soft_dtw(actions_h, actions_r, gamma: float = 0.1):
    # [Kh,T,D] x [Kr,T,D] -> [Kh,Kr]
    return jax.vmap(
        lambda ah: jax.vmap(lambda ar: soft_dtw_distance(ah, ar, gamma))(actions_r)
    )(actions_h)


def matching_cost_scale(dtw_matrix, lambd: float = 0.5):
    """EgoBridge: for each robot j, discount the best human i*(j)."""
    dtw_matrix = jax.lax.stop_gradient(dtw_matrix)
    i_star = jnp.argmin(dtw_matrix, axis=0)  # [Kr]
    kr = dtw_matrix.shape[1]
    w = jnp.ones_like(dtw_matrix).at[i_star, jnp.arange(kr)].set(lambd)
    return jax.lax.stop_gradient(w)


def sinkhorn_ot_loss(x, y, cost_scale, blur: float = 0.05, num_iters: int = 18,
                     valid_h=None, valid_r=None):
    """x,y: [K, F]; cost = 0.5||x-y||^2 * W. Uniform mass on *valid* supports."""
    diff = x[:, None, :] - y[None, :, :]
    c = 0.5 * jnp.mean(diff * diff, axis=-1) * cost_scale
    k = c.shape[0]
    if valid_h is None:
        valid_h = jnp.ones((k,), dtype=bool)
    if valid_r is None:
        valid_r = jnp.ones((k,), dtype=bool)
    pair_ok = valid_h[:, None] & valid_r[None, :]
    c = jnp.where(pair_ok, c, jnp.asarray(1e6, dtype=c.dtype))
    nh = jnp.maximum(jnp.sum(valid_h.astype(c.dtype)), jnp.asarray(1.0, c.dtype))
    nr = jnp.maximum(jnp.sum(valid_r.astype(c.dtype)), jnp.asarray(1.0, c.dtype))
    neg_inf = jnp.asarray(-1e9, dtype=c.dtype)
    log_a = jnp.where(valid_h, -jnp.log(nh), neg_inf)
    log_b = jnp.where(valid_r, -jnp.log(nr), neg_inf)
    eps = blur
    log_k = -c / eps

    def step(f_g, _):
        f, g = f_g
        f = log_a - jax.nn.logsumexp(log_k + g[None, :], axis=1)
        g = log_b - jax.nn.logsumexp(log_k + f[:, None], axis=0)
        return (f, g), None

    (f, g), _ = jax.lax.scan(
        step,
        (jnp.zeros((k,), c.dtype), jnp.zeros((k,), c.dtype)),
        None,
        length=num_iters,
    )
    p_plan = jnp.exp(f[:, None] + g[None, :] + log_k)
    p_plan = jnp.where(pair_ok, p_plan, jnp.asarray(0.0, dtype=p_plan.dtype))
    return jnp.sum(p_plan * c)


def gather_xyz_for_bridge(actions_80, bridge: str):
    """actions_80: [B,T,80] (or [K,T,80]). EgoBridge-style xyz only."""
    if bridge == "hz":
        # hangzhou single right: RIGHT_EEF xyz
        idx = jnp.array(RIGHT_EEF_XYZ, dtype=jnp.int32)
    elif bridge == "sz":
        # shenzhen bimanual: L_xyz + R_xyz (6D; still position-only like EgoBridge)
        idx = jnp.array(LEFT_EEF_XYZ + RIGHT_EEF_XYZ, dtype=jnp.int32)
    else:
        raise ValueError(bridge)
    return actions_80[..., idx]


def bridge_ot_square(
    tokens_h,  # [K,T,D]
    tokens_r,
    actions_xyz_h,  # [K,T,3or6]
    actions_xyz_r,
    *,
    lambd: float = 0.5,
    dtw_gamma: float = 0.1,
    blur: float = 0.05,
    sinkhorn_iters: int = 18,
    valid_h=None,
    valid_r=None,
):
    """KxK buffer OT; valid_h/valid_r mask padding so real supports may be rectangular."""
    dtw = pairwise_soft_dtw(actions_xyz_h, actions_xyz_r, gamma=dtw_gamma)
    if valid_h is not None and valid_r is not None:
        dtw = jnp.where(
            valid_h[:, None] & valid_r[None, :],
            dtw,
            jnp.asarray(jnp.inf, dtype=dtw.dtype),
        )
    w = matching_cost_scale(dtw, lambd=lambd)
    kh = tokens_h.shape[0]
    loss = sinkhorn_ot_loss(
        tokens_h.reshape(kh, -1),
        tokens_r.reshape(kh, -1),
        w,
        blur=blur,
        num_iters=sinkhorn_iters,
        valid_h=valid_h,
        valid_r=valid_r,
    )
    return loss, {"ot_dtw_mean": jnp.mean(dtw)}
