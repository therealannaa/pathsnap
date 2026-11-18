"""HMM map matching decoded with Viterbi.

Hidden state at GPS point i: one of its candidate arcs (local arc id).
Emission: the classifier probability for that candidate, renormalised over the
point's candidates, raised to `emission_weight`.
Transition a -> b between points i and i+1: compares the driving distance
from the projection on a to the projection on b (through the road graph) with
the straight-line distance between the two GPS points:
    log P = -|route_dist - gps_dist| / beta_eff
beta_eff grows with the square root of the time gap, so long GPS dropouts get
more freedom. Pairs with no path inside the Dijkstra search radius get
probability 0. If every transition into point i+1 is impossible, the chain is
broken: decoding restarts from the emissions at i+1 and the break is recorded.
"""
import numpy as np
from scipy.sparse.csgraph import dijkstra

from . import config


def _route_distances(net, a_arcs, a_t, b_arcs, b_t, limit):
    """Driving distance matrix (len(a), len(b)) between projections on arcs."""
    src_nodes, ri = np.unique(net.to[a_arcs], return_inverse=True)
    dist = dijkstra(net.graph, directed=True, indices=src_nodes, limit=limit)
    sp = dist[ri][:, net.frm[b_arcs]]
    rd = (1 - a_t)[:, None] * net.length[a_arcs][:, None] + sp + (b_t * net.length[b_arcs])[None, :]
    same = a_arcs[:, None] == b_arcs[None, :]
    same_d = (b_t[None, :] - a_t[:, None]) * net.length[a_arcs][:, None]
    # moving backwards on the same arc is treated as a distance mismatch
    return np.where(same, same_d, rd)


def viterbi(net, pxy, times, cand, ct, probs, params=None):
    """Returns (state slot per point, list of break indices)."""
    p = dict(config.HMM_DEFAULTS)
    if params:
        p.update(params)
    n, k = cand.shape
    valid = cand >= 0
    pr = np.where(valid, np.maximum(probs, p["prob_floor"]), 0.0)
    pr = pr / np.maximum(pr.sum(1, keepdims=True), 1e-12)
    log_em = np.where(valid, p["emission_weight"] * np.log(np.maximum(pr, 1e-300)), -np.inf)

    delta = log_em[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    seg_end = {}        # break index i -> best state at i-1
    for i in range(1, n):
        va, vb = valid[i - 1], valid[i]
        a, b = cand[i - 1, va], cand[i, vb]
        gps_d = float(np.hypot(*(pxy[i] - pxy[i - 1])))
        dt = max(float(times[i] - times[i - 1]), 1.0)
        rd = _route_distances(net, a, ct[i - 1, va], b, ct[i, vb], 2 * gps_d + p["max_extra_m"])
        beta_eff = p["beta"] * np.sqrt(dt)
        log_tr = np.where(np.isfinite(rd), -np.abs(rd - gps_d) / beta_eff, -np.inf)
        full = np.full((k, k), -np.inf)
        full[np.ix_(np.nonzero(va)[0], np.nonzero(vb)[0])] = log_tr
        score = delta[:, None] + full
        back[i] = np.argmax(score, axis=0)
        new = score[back[i], np.arange(k)] + log_em[i]
        if not np.isfinite(new).any():
            seg_end[i] = int(np.argmax(delta))
            back[i] = -1
            new = log_em[i].copy()
        # keep numbers bounded
        delta = new - np.max(new)
    states = np.empty(n, dtype=np.int64)
    states[-1] = int(np.argmax(delta))
    for i in range(n - 1, 0, -1):
        states[i - 1] = seg_end[i] if i in seg_end else back[i, states[i]]
    return states, sorted(seg_end)


def reconstruct_route(net, arc_seq, max_gap_m=3000.0):
    """Fill the gaps between consecutive distinct predicted arcs with shortest paths.

    Returns a list of local arc ids forming the reconstructed route.
    """
    seq = [int(a) for i, a in enumerate(arc_seq) if i == 0 or a != arc_seq[i - 1]]
    if not seq:
        return []
    route = [seq[0]]
    arc_of = {}
    for i, (f, t) in enumerate(zip(net.frm, net.to)):
        key = (int(f), int(t))
        if key not in arc_of or net.length[i] < net.length[arc_of[key]]:
            arc_of[key] = i
    for a, b in zip(seq[:-1], seq[1:]):
        s, g = int(net.to[a]), int(net.frm[b])
        if s != g:
            d, pred = dijkstra(net.graph, directed=True, indices=s, limit=max_gap_m, return_predecessors=True)
            if np.isfinite(d[g]):
                path = [g]
                while path[-1] != s:
                    path.append(int(pred[path[-1]]))
                path = path[::-1]
                route.extend(arc_of[(u, v)] for u, v in zip(path[:-1], path[1:]))
        route.append(b)
    return route
