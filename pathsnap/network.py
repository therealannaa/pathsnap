"""Road network for one trip, projected to metres and cropped around the track."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from . import config
from .geo import LocalProjection, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat, arcs, proj: LocalProjection, bbox_xy=None, margin=config.NETWORK_MARGIN_M):
        self.proj = proj
        xy = proj.forward(nodes_lonlat)
        arcs = np.asarray(arcs, dtype=np.int64)
        valid = (arcs >= 0).all(axis=1) & (arcs < len(xy)).all(axis=1) & (arcs[:, 0] != arcs[:, 1])
        arc_ids = np.nonzero(valid)[0]
        if bbox_xy is not None:
            x0, y0, x1, y1 = bbox_xy
            x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
            inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
            a = arcs[arc_ids]
            arc_ids = arc_ids[inside[a[:, 0]] | inside[a[:, 1]]]
        a = arcs[arc_ids]
        used, inv = np.unique(a.ravel(), return_inverse=True)
        self.node_orig = used                      # local node -> original node id
        self.xy = xy[used]
        self.arc_orig = arc_ids                    # local arc -> original arc id
        self.frm = inv[0::2]
        self.to = inv[1::2]
        self.n_nodes = len(used)
        self.n_arcs = len(arc_ids)
        self.ax, self.ay = self.xy[self.frm, 0], self.xy[self.frm, 1]
        self.bx, self.by = self.xy[self.to, 0], self.xy[self.to, 1]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = np.arctan2(self.by - self.ay, self.bx - self.ax)
        self.out_deg = np.bincount(self.frm, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.to, minlength=self.n_nodes)
        self.orig_to_local = {int(o): i for i, o in enumerate(self.arc_orig)}
        # Graph over nodes; parallel arcs keep the shortest length.
        w = np.maximum(self.length, 0.01)
        order = np.lexsort((w, self.to, self.frm))
        f, t, w = self.frm[order], self.to[order], w[order]
        first = np.ones(len(f), bool)
        first[1:] = (f[1:] != f[:-1]) | (t[1:] != t[:-1])
        self.graph = csr_matrix((w[first], (f[first], t[first])), shape=(self.n_nodes, self.n_nodes))
        self._build_tree()

    def _build_tree(self):
        n = np.maximum(np.ceil(self.length / config.SAMPLE_SPACING_M).astype(int), 1) + 1
        arc_idx = np.repeat(np.arange(self.n_arcs), n)
        start = np.repeat(np.cumsum(n) - n, n)
        frac = (np.arange(n.sum()) - start) / np.repeat(n - 1, n)
        sx = self.ax[arc_idx] + frac * (self.bx - self.ax)[arc_idx]
        sy = self.ay[arc_idx] + frac * (self.by - self.ay)[arc_idx]
        self.sample_arc = arc_idx
        self.tree = cKDTree(np.column_stack([sx, sy]))

    @classmethod
    def from_trip(cls, trip, proj=None):
        proj = proj or LocalProjection.for_track(trip["track"])
        pxy = proj.forward(trip["track"][:, :2])
        bbox = (pxy[:, 0].min(), pxy[:, 1].min(), pxy[:, 0].max(), pxy[:, 1].max())
        return cls(trip["nodes"], trip["arcs"], proj, bbox), pxy

    def dist_to_arcs(self, px, py, arcs):
        return point_segment(px, py, self.ax[arcs], self.ay[arcs], self.bx[arcs], self.by[arcs])

    def candidates(self, pxy, k=config.N_CANDIDATES, n_query=128):
        """For each point return the k nearest arcs (local ids), their distances and projection fractions.

        Missing slots are filled with -1 / inf.
        """
        n = len(pxy)
        nq = min(n_query, len(self.sample_arc))
        _, idx = self.tree.query(pxy, k=nq)
        idx = idx.reshape(n, nq)
        cand = np.full((n, k), -1, dtype=np.int64)
        cd = np.full((n, k), np.inf)
        ct = np.zeros((n, k))
        for i in range(n):
            arcs = np.unique(self.sample_arc[idx[i]])
            d, t = self.dist_to_arcs(pxy[i, 0], pxy[i, 1], arcs)
            o = np.lexsort((arcs, d))[:k]
            m = len(o)
            cand[i, :m], cd[i, :m], ct[i, :m] = arcs[o], d[o], t[o]
        return cand, cd, ct

    def connected(self, a, b):
        """True where arc b can directly follow arc a (or is the same arc)."""
        a = np.asarray(a)
        b = np.asarray(b)
        ok = (a >= 0) & (b >= 0)
        res = np.zeros(a.shape, bool)
        res[ok] = (a[ok] == b[ok]) | (self.to[a[ok]] == self.frm[b[ok]])
        return res
