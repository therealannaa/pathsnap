"""Turning the .route file into one reference arc per GPS point.

The .route file lists the arcs the vehicle drove, in order, but it does not say
which GPS point belongs to which arc. We assign points to route positions with a
monotone dynamic programme: point i gets route index j_i, with j_i never
decreasing. The cost is the point-to-arc distance plus a penalty when the
assignment jumps over route arcs whose total length is much longer than the
distance the GPS actually moved. This keeps loops and repeated streets in the
correct order.

This alignment only uses the reference route, so it is used for training labels
and for server-side scoring. It is never called by the prediction pipeline.
"""
import numpy as np

from .geo import point_segment

MAX_JUMP = 40       # how many route arcs one GPS step may advance
SKIP_SLACK_M = 30.0
SKIP_WEIGHT = 0.5   # cost per metre of skipped route length beyond the GPS step
CHANGE_COST = 1.0


def align_route_to_track(pxy, times, route, nodes_xy, arcs):
    route = np.asarray(route)
    a = arcs[route]
    ax, ay = nodes_xy[a[:, 0], 0], nodes_xy[a[:, 0], 1]
    bx, by = nodes_xy[a[:, 1], 0], nodes_xy[a[:, 1], 1]
    seg_len = np.hypot(bx - ax, by - ay)
    start = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])   # route distance at start of arc j
    end = start + seg_len

    n, m = len(pxy), len(route)
    D, _ = point_segment(pxy[:, :1], pxy[:, 1:2], ax[None], ay[None], bx[None], by[None])  # (n, m)
    step = np.r_[0.0, np.hypot(*np.diff(pxy, axis=0).T)]

    cost = D[0] + SKIP_WEIGHT * np.maximum(0, start - SKIP_SLACK_M)
    back = np.zeros((n, m), dtype=np.int16)
    for i in range(1, n):
        best = cost.copy()
        arg = np.zeros(m, dtype=np.int16)
        for w in range(1, min(MAX_JUMP, m - 1) + 1):
            prev = np.full(m, np.inf)
            prev[w:] = cost[:-w]
            # route length strictly between arc j-w and arc j
            skipped = start[w:] - end[:-w]
            pen = np.full(m, np.inf)
            pen[w:] = CHANGE_COST + SKIP_WEIGHT * np.maximum(0, skipped - step[i] - SKIP_SLACK_M)
            cand = prev + pen
            better = cand < best
            best[better] = cand[better]
            arg[better] = w
        cost = best + D[i]
        back[i] = arg
    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        j[i - 1] = j[i] - back[i, j[i]]
    return {
        "route_index": j,
        "label_arc": route[j],            # original arc id per point
        "label_dist": D[np.arange(n), j],
    }
