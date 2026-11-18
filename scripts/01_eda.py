"""Exploratory data analysis of the raw train and student-test partitions.

Outputs (results/):
  eda_inventory.csv          one row per trip: files, sizes, counts, gaps, extent, network stats
  eda_summary.csv            partition-level summary
  data_quality_notes.md      issues found and decisions taken
  plots/eda_*.png            figures
  maps/eda_trip_<id>.html    Folium maps of raw GPS + road network + reference route
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap import config, data_io, maps  # noqa: E402
from pathsnap.geo import LocalProjection, haversine_m  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402

P = config.PLOTS


def raw_stats(folder, tid, part):
    d = Path(folder) / tid
    files = {ext: d / f"{tid}.{ext}" for ext in ("track", "nodes", "arcs", "route")}
    raw_track = pd.read_csv(files["track"], sep=r"\s+", header=None)
    trip = data_io.load_trip(folder, tid)
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    dt = np.diff(tr[:, 2])
    step = haversine_m(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    speed = step / np.maximum(dt, 1e-9)
    n = len(nodes)
    g = csr_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(arcs.ravel(), minlength=n)
    route = trip.get("route")
    row = {
        "partition": part, "trip_id": tid,
        **{f"has_{e}": f.exists() for e, f in files.items()},
        **{f"kb_{e}": round(f.stat().st_size / 1024, 1) if f.exists() else np.nan for e, f in files.items()},
        "raw_points": len(raw_track), "raw_track_cols": raw_track.shape[1],
        "missing_values": int(raw_track.isna().sum().sum()),
        **trip["clean_info"],
        "duration_s": tr[-1, 2] - tr[0, 2],
        "dt_median": np.median(dt), "share_dt_eq_1": np.mean(dt == 1), "dt_max": dt.max(),
        "gaps_gt_5s": int((dt > 5).sum()), "gaps_gt_30s": int((dt > 30).sum()),
        "track_km": step.sum() / 1000, "speed_p95_kmh": np.percentile(speed, 95) * 3.6,
        "max_jump_m": step.max(), "jumps_gt_100m": int((step > 100).sum()),
        "lon_min": tr[:, 0].min(), "lon_max": tr[:, 0].max(), "lat_min": tr[:, 1].min(), "lat_max": tr[:, 1].max(),
        "extent_km_x": haversine_m(tr[:, 0].min(), tr[:, 1].mean(), tr[:, 0].max(), tr[:, 1].mean()) / 1000,
        "extent_km_y": haversine_m(tr[:, 0].mean(), tr[:, 1].min(), tr[:, 0].mean(), tr[:, 1].max()) / 1000,
        "nodes": n, "arcs": len(arcs), "isolated_nodes": int((deg == 0).sum()),
        "weak_components": n_weak, "largest_weak_share": np.bincount(lab_w).max() / n,
        "strong_components": n_strong, "largest_strong_share": np.bincount(lab_s).max() / n,
        "self_loops": int((arcs[:, 0] == arcs[:, 1]).sum()),
        "duplicate_arcs": int(len(arcs) - len(np.unique(arcs, axis=0))),
    }
    if route is not None:
        ra = arcs[route]
        row.update({"route_arcs": len(route), "route_unique_arcs": len(np.unique(route)),
                    "route_continuity": float(np.mean(ra[:-1, 1] == ra[1:, 0])) if len(route) > 1 else 1.0})
    return row, trip


def main():
    rows, trips = [], {}
    for part, folder in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        for tid in data_io.list_trips(folder):
            r, trip = raw_stats(folder, tid, part)
            rows.append(r)
            trips[tid] = (folder, trip)
    inv = pd.DataFrame(rows)
    pts = pd.concat([pd.read_csv(config.FEATURES_DIR / f"{p}_points.csv", dtype={"trip_id": str})
                     for p in ("train", "student_test")])
    lab = pts.groupby("trip_id").agg(label_dist_median=("label_dist", "median"),
                                     label_dist_p95=("label_dist", lambda s: s.quantile(.95)),
                                     candidate_miss_rate=("label_in_cand", lambda s: 1 - s.mean()),
                                     nearest_arc_acc=("nearest_correct", "mean")).reset_index()
    inv = inv.merge(lab, on="trip_id", how="left")
    inv.to_csv(config.RESULTS / "eda_inventory.csv", index=False)

    summ = inv.groupby("partition").agg(
        trips=("trip_id", "size"), points=("clean_points", "sum"), points_min=("clean_points", "min"),
        points_median=("clean_points", "median"), points_max=("clean_points", "max"),
        missing_values=("missing_values", "sum"), invalid_removed=("invalid_removed", "sum"),
        duplicate_time_removed=("duplicate_time_removed", "sum"), out_of_order=("out_of_order", "sum"),
        trips_with_gap_gt_30s=("gaps_gt_30s", lambda s: int((s > 0).sum())),
        nodes_median=("nodes", "median"), arcs_median=("arcs", "median"),
        isolated_nodes=("isolated_nodes", "sum"), weak_components_median=("weak_components", "median"),
        route_arcs_median=("route_arcs", "median"), route_continuity_min=("route_continuity", "min"),
        candidate_miss_rate_mean=("candidate_miss_rate", "mean"))
    summ.T.to_csv(config.RESULTS / "eda_summary.csv")
    print(summ.T)

    # ---- plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, c in [("train", "#1f77b4"), ("student_test", "#ff7f0e")]:
        s = inv[inv.partition == part]
        ax[0].hist(s.clean_points, bins=30, alpha=.7, label=part, color=c)
        ax[1].hist(s.track_km, bins=30, alpha=.7, label=part, color=c)
        ax[2].hist(s.label_dist_median, bins=30, alpha=.7, label=part, color=c)
    ax[0].set(title="GPS points per trip", xlabel="points", ylabel="trips")
    ax[1].set(title="Trip length", xlabel="km")
    ax[2].set(title="Median GPS-to-reference-arc distance", xlabel="metres")
    for a in ax:
        a.legend()
    fig.tight_layout(); fig.savefig(P / "eda_trip_distributions.png", dpi=130); plt.close(fig)

    dts = np.concatenate([np.diff(t[1]["track"][:, 2]) for t in trips.values()])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vals, cnts = np.unique(np.clip(dts, 0, 11), return_counts=True)
    ax[0].bar([str(int(v)) if v < 11 else ">10" for v in vals], cnts / cnts.sum())
    ax[0].set(title="Sampling interval (all trips)", xlabel="seconds between fixes", ylabel="share", yscale="log")
    ax[1].hist(pts.label_dist.clip(upper=200), bins=80)
    ax[1].set(title="GPS point distance to its reference arc", xlabel="metres (clipped at 200)", ylabel="points", yscale="log")
    fig.tight_layout(); fig.savefig(P / "eda_sampling_and_noise.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for part, c in [("train", "#1f77b4"), ("student_test", "#ff7f0e")]:
        s = inv[inv.partition == part]
        ax.scatter((s.lon_min + s.lon_max) / 2, (s.lat_min + s.lat_max) / 2, s=20, label=part, c=c)
    ax.set(title="Trip locations (centre of each track)", xlabel="longitude", ylabel="latitude")
    ax.legend(); fig.tight_layout(); fig.savefig(P / "eda_trip_locations.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(inv.nodes, inv.arcs, c=(inv.partition == "train").map({True: "#1f77b4", False: "#ff7f0e"}), s=15)
    ax[0].set(xscale="log", yscale="log", title="Road network size per trip", xlabel="nodes", ylabel="arcs")
    ax[1].hist(inv.largest_strong_share, bins=30)
    ax[1].set(title="Share of nodes in largest strongly connected component", xlabel="share", ylabel="trips")
    fig.tight_layout(); fig.savefig(P / "eda_network.png", dpi=130); plt.close(fig)

    # trajectory + network static plot for two trips, Folium maps for three
    for tid in ["00000001", "00000033", "00000000"]:
        folder, trip = trips[tid]
        proj = LocalProjection.for_track(trip["track"])
        net, pxy = RoadNetwork.from_trip(trip, proj)
        near = maps.nearby_arcs(net, pxy, 300)
        ref = [net.orig_to_local[a] for a in trip["route"] if a in net.orig_to_local]
        fig, ax = plt.subplots(figsize=(7, 7))
        for a in near:
            ax.plot([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], c="#bbb", lw=.6)
        for a in ref:
            ax.plot([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], c="#2e7d32", lw=2.5)
        ax.scatter(pxy[:, 0], pxy[:, 1], s=2, c="#1565c0", zorder=3)
        ax.set(title=f"Trip {tid}: GPS (blue), reference route (green), network (grey)", xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(P / f"eda_trajectory_{tid}.png", dpi=130); plt.close(fig)
        m = maps.trip_map(net, trip["track"], pxy, ref_route_local=ref, title=f"Trip {tid} (EDA)")
        m.save(str(config.MAPS / f"eda_trip_{tid}.html"))

    bad = inv[inv.label_dist_median > 50]
    notes = [
        "# Data-quality notes (generated by scripts/01_eda.py)", "",
        f"- Trips: {', '.join(f'{k} {v}' for k, v in inv.partition.value_counts().items())}. Missing values in .track files: {int(inv.missing_values.sum())}. "
        f"Invalid coordinates removed: {int(inv.invalid_removed.sum())}. Duplicate timestamps removed: {int(inv.duplicate_time_removed.sum())}. "
        f"Out-of-order timestamps: {int(inv.out_of_order.sum())}.",
        f"- Sampling is nominally 1 Hz: {np.mean(dts == 1):.1%} of steps are exactly 1 s. {int((inv.gaps_gt_30s > 0).sum())} trips contain at least one gap over 30 s (longest {dts.max():.0f} s).",
        f"- Each trip ships its own road-network extract (median {inv.nodes.median():.0f} nodes, {inv.arcs.median():.0f} arcs). "
        f"Trips are spread across many countries, so a per-trip local projection is used.",
        f"- Reference routes are continuous chains of arcs (minimum continuity {inv.route_continuity.min():.3f}). They list arcs, not GPS points, "
        "so a monotone alignment (pathsnap/labeling.py) assigns each GPS point a reference arc.",
        f"- Trips whose median GPS-to-reference distance is over 50 m: {', '.join(bad.trip_id)} "
        "(see plots/eda_trajectory_00000033.png: the reference route covers only part of the drive, so GPS points on the "
        "uncovered part are aligned to a distant arc).",
        f"- The longest single time gap is {dts.max():.0f} s; such gaps are kept and the HMM widens its transition model with the gap length.",
        f"- Decision: training rows from GPS points more than 60 m from their aligned reference arc are dropped as unreliable labels "
        f"({(pts[pts.trip_id.isin(inv[inv.partition == 'train'].trip_id)].label_dist > 60).mean():.1%} of training points). "
        "Evaluation tables report accuracy on all points.",
    ]
    (config.RESULTS / "data_quality_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
