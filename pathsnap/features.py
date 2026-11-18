"""Candidate-level feature engineering.

Every GPS point gets up to N_CANDIDATES candidate arcs. Each (point, candidate)
pair becomes one row; the classifier scores how likely that candidate is the
arc the vehicle was on. Only GPS data and road geometry are used here.
"""
import numpy as np
import pandas as pd

from .geo import point_segment

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio",
    "win2_mean_dist", "win2_max_dist", "win5_mean_dist",
    "head_cos", "head_cos_wide", "head_abs_deg", "heading_valid",
    "speed", "dt", "gap", "proj_t", "arc_len",
    "to_out_deg", "from_in_deg", "n_cand_25m",
    "prev_near_conn", "next_near_conn", "is_cand_prev", "is_cand_next",
]


def _heading(pxy, w):
    n = len(pxy)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    d = pxy[i1] - pxy[i0]
    return np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1])


def build_features(net, pxy, times, cand, cd, ct):
    n, k = cand.shape
    valid = cand >= 0
    c = np.where(valid, cand, 0)

    dt = np.r_[1.0, np.diff(times)]
    step = np.r_[0.0, np.hypot(*np.diff(pxy, axis=0).T)]
    speed = step / np.maximum(dt, 1e-3)

    dmin = np.where(valid, cd, np.inf).min(axis=1, keepdims=True)
    rank = np.argsort(np.argsort(np.where(valid, cd, np.inf), axis=1), axis=1)

    # distances of neighbouring GPS points to the same candidate arc
    def neigh_dist(offsets):
        out = []
        for o in offsets:
            j = np.clip(np.arange(n) + o, 0, n - 1)
            d, _ = point_segment(pxy[j, 0:1], pxy[j, 1:2], net.ax[c], net.ay[c], net.bx[c], net.by[c])
            out.append(d)
        return np.stack(out)

    w2 = neigh_dist(range(-2, 3))
    w5 = neigh_dist(range(-5, 6, 2))

    h1, m1 = _heading(pxy, 2)
    h2, _ = _heading(pxy, 6)
    hvalid = (m1 > 3.0).astype(float)
    diff = h1[:, None] - net.bearing[c]
    head_cos = np.cos(diff) * hvalid[:, None]
    head_cos_wide = np.cos(h2[:, None] - net.bearing[c]) * hvalid[:, None]
    head_abs = np.degrees(np.abs(np.angle(np.exp(1j * diff)))) * hvalid[:, None] + 90 * (1 - hvalid[:, None])

    near = cand[:, 0]
    prev_near = np.r_[near[:1], near[:-1]]
    next_near = np.r_[near[1:], near[-1:]]
    prev_conn = net.connected(np.repeat(prev_near[:, None], k, 1), cand)
    next_conn = net.connected(cand, np.repeat(next_near[:, None], k, 1))
    prev_set = np.r_[cand[:1], cand[:-1]]
    next_set = np.r_[cand[1:], cand[-1:]]
    is_prev = (cand[:, :, None] == prev_set[:, None, :]).any(axis=2)
    is_next = (cand[:, :, None] == next_set[:, None, :]).any(axis=2)
    n25 = (np.where(valid, cd, np.inf) <= 25).sum(axis=1)

    rep = lambda v: np.repeat(v[:, None], k, 1)
    cols = {
        "point_idx": rep(np.arange(n)),
        "cand_slot": np.tile(np.arange(k), (n, 1)),
        "arc_local": cand,
        "dist": cd,
        "dist_rank": rank,
        "dist_minus_min": cd - dmin,
        "dist_ratio": (cd + 1) / (dmin + 1),
        "win2_mean_dist": w2.mean(0),
        "win2_max_dist": w2.max(0),
        "win5_mean_dist": w5.mean(0),
        "head_cos": head_cos,
        "head_cos_wide": head_cos_wide,
        "head_abs_deg": head_abs,
        "heading_valid": rep(hvalid),
        "speed": rep(speed),
        "dt": rep(dt),
        "gap": rep((dt > 5).astype(float)),
        "proj_t": ct,
        "arc_len": net.length[c],
        "to_out_deg": net.out_deg[net.to[c]],
        "from_in_deg": net.in_deg[net.frm[c]],
        "n_cand_25m": rep(n25),
        "prev_near_conn": prev_conn.astype(float),
        "next_near_conn": next_conn.astype(float),
        "is_cand_prev": is_prev.astype(float),
        "is_cand_next": is_next.astype(float),
    }
    m = valid.ravel()
    df = pd.DataFrame({key: np.asarray(v).ravel()[m] for key, v in cols.items()})
    df["arc_orig"] = net.arc_orig[df["arc_local"].to_numpy()]
    return df
