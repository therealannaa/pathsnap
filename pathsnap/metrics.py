"""Evaluation metrics shared by experiments and the API's server-side scoring."""
import numpy as np


def top1_accuracy(pred_arc_orig, label_arc_orig):
    return float(np.mean(np.asarray(pred_arc_orig) == np.asarray(label_arc_orig)))


def route_continuity(net, pred_local):
    """Share of consecutive GPS points whose predicted arcs are equal or directly connected."""
    p = np.asarray(pred_local)
    if len(p) < 2:
        return 1.0
    return float(np.mean(net.connected(p[:-1], p[1:])))


def route_overlap(pred_route_orig, ref_route_orig, arc_length_by_orig):
    """Length-weighted precision / recall / F1 between predicted and reference arc sets."""
    P, R = set(map(int, pred_route_orig)), set(map(int, ref_route_orig))
    L = lambda s: sum(arc_length_by_orig.get(a, 0.0) for a in s)
    inter = L(P & R)
    prec = inter / max(L(P), 1e-9)
    rec = inter / max(L(R), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
