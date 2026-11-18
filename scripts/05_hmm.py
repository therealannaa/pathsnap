"""HMM / Viterbi route reconstruction: decoder tuning and evaluation.

1. Decoder tuning (training data only): fit the final LightGBM configuration on
   64 training trips, score 16 held-out training trips, grid-search beta and the
   emission weight on those 16 trips. Saves artifacts/hmm_config.json.
2. Student-test evaluation: classifier-only vs classifier + HMM for the final
   model and for every model from 03 (shows how each model's probabilities work
   with the decoder). Saves per-trip tables, plots and Folium maps.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap import config, data_io, maps, metrics, pipeline  # noqa: E402
from pathsnap.datasets import LABEL_MAX_DIST, load_partition  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.hmm import reconstruct_route, viterbi  # noqa: E402

BETAS = [5, 10, 20, 40, 80]
EMISSION_WEIGHTS = [0.5, 1.0, 2.0]


def load_prep(folder, tid):
    trip = data_io.load_trip(folder, tid)
    prep = pipeline.prepare_trip(trip)
    pipeline.attach_labels(prep, trip)
    return trip, prep


def evaluate(trip, prep, P, params):
    net, cand = prep["net"], prep["cand"]
    L = prep["label_arc"]
    reliable = prep["label_dist"] <= LABEL_MAX_DIST
    clf = cand[np.arange(len(cand)), np.argmax(np.where(cand >= 0, P, -1), axis=1)]
    t0 = time.time()
    slots, breaks = viterbi(net, prep["pxy"], prep["times"], cand, prep["ct"], P, params)
    dec_s = time.time() - t0
    hmm = cand[np.arange(len(cand)), slots]
    lengths = {**prep["ref_len"], **dict(zip(map(int, net.arc_orig), net.length))}
    r = {"trip_id": trip["trip_id"], "points": len(L), "breaks": len(breaks), "decode_s": dec_s}
    for name, pred in (("clf", clf), ("hmm", hmm)):
        po = net.arc_orig[pred]
        r[f"{name}_top1"] = metrics.top1_accuracy(po, L)
        r[f"{name}_top1_reliable"] = metrics.top1_accuracy(po[reliable], L[reliable]) if reliable.any() else np.nan
        r[f"{name}_continuity"] = metrics.route_continuity(net, pred)
        route = reconstruct_route(net, pred)
        r.update({f"{name}_{k}": v for k, v in metrics.route_overlap(net.arc_orig[route], trip["route"], lengths).items()})
    return r, clf, hmm


def tune_decoder(params_model):
    train = load_partition("train", train_filter=True, point_stride=3)
    inv = pd.read_csv(config.RESULTS / "eda_inventory.csv", dtype={"trip_id": str})
    short = set(inv[(inv.partition == "train") & (inv.clean_points <= 4000)].trip_id)
    trips = np.array(sorted(train.trip_id.unique()))
    gss = GroupShuffleSplit(n_splits=1, test_size=16 / 80, random_state=config.SEED)
    fit_idx, hold_idx = next(gss.split(trips, groups=trips))
    hold = [t for t in trips[hold_idx] if t in short]
    fit = train[train.trip_id.isin(set(trips[fit_idx]))]
    m = LGBMClassifier(subsample_freq=1, random_state=config.SEED, verbose=-1, **params_model)
    m.fit(fit[FEATURES].to_numpy(np.float32), fit.y.to_numpy())
    preps = []
    for tid in hold:
        trip, prep = load_prep(config.TRAIN_DIR, tid)
        p = m.predict_proba(prep["df"][FEATURES].to_numpy(np.float32))[:, 1]
        preps.append((trip, prep, pipeline.probs_matrix(prep, p)))
    rows = []
    for beta, ew in itertools.product(BETAS, EMISSION_WEIGHTS):
        params = {"beta": beta, "emission_weight": ew}
        res = pd.DataFrame([evaluate(t, pr, P, params)[0] for t, pr, P in preps])
        w = res.points
        rows.append({"beta": beta, "emission_weight": ew,
                     "clf_top1": np.average(res.clf_top1_reliable, weights=w),
                     "hmm_top1": np.average(res.hmm_top1_reliable, weights=w),
                     "hmm_continuity": np.average(res.hmm_continuity, weights=w),
                     "hmm_route_f1": res.hmm_route_f1.mean()})
        print(rows[-1])
    grid = pd.DataFrame(rows)
    grid.to_csv(config.RESULTS / "hmm_tuning_grid.csv", index=False)
    best = grid.sort_values(["hmm_top1", "hmm_route_f1"], ascending=False).iloc[0]
    cfg = {**config.HMM_DEFAULTS, "beta": float(best.beta), "emission_weight": float(best.emission_weight),
           "tuned_on": f"{len(hold)} held-out training trips: {', '.join(hold)}"}
    (config.ARTIFACTS / "hmm_config.json").write_text(json.dumps(cfg, indent=2))
    fig, ax = plt.subplots(figsize=(6, 4))
    for ew, g in grid.groupby("emission_weight"):
        ax.plot(g.beta, g.hmm_top1, "o-", label=f"emission weight {ew}")
    ax.axhline(grid.clf_top1.iloc[0], color="grey", ls="--", label="classifier only")
    ax.set(xscale="log", xlabel="beta (m)", ylabel="Top-1 (held-out training trips)", title="HMM decoder tuning")
    ax.legend(); fig.tight_layout(); fig.savefig(config.PLOTS / "hmm_tuning.png", dpi=130); plt.close(fig)
    return cfg


def main():
    final_cfg = json.loads((config.ARTIFACTS / "final_model_config.json").read_text())
    hmm_cfg = tune_decoder(final_cfg["params"])
    print("HMM config", hmm_cfg)

    test_rows = load_partition("student_test")[["trip_id", "point_idx", "cand_slot"]]
    probs = pd.read_parquet(config.RESULTS / "student_test_probs.parquet")
    model_cols = [c for c in probs.columns if c not in ("trip_id", "point_idx", "cand_slot", "y")]
    per_trip, per_model = [], []
    for tid in data_io.list_trips(config.STUDENT_TEST_DIR):
        trip, prep = load_prep(config.STUDENT_TEST_DIR, tid)
        sel = (probs.trip_id.astype(str) == tid).to_numpy()
        assert (test_rows[sel].point_idx.to_numpy() == prep["df"].point_idx.to_numpy()).all()
        for mc in model_cols:
            P = pipeline.probs_matrix(prep, probs.loc[sel, mc].to_numpy())
            r, clf, hmm = evaluate(trip, prep, P, hmm_cfg)
            per_model.append({"model": mc, **r})
            if mc == "LightGBM_tuned":
                per_trip.append(r)
                net = prep["net"]
                ref = [net.orig_to_local[a] for a in trip["route"] if a in net.orig_to_local]
                wrong_clf = [a for a in np.unique(clf) if net.arc_orig[a] not in set(trip["route"])]
                m = maps.trip_map(net, trip["track"], prep["pxy"], ref_route_local=ref, title=f"Student-test trip {tid}",
                                  layers={"Classifier-only arcs not on reference": (wrong_clf, "#e65100"),
                                          "Predicted route (classifier + HMM)": (reconstruct_route(net, hmm), "#c62828")})
                m.save(str(config.MAPS / f"student_test_{tid}.html"))
    pt = pd.DataFrame(per_trip)
    pt.to_csv(config.RESULTS / "hmm_student_test_per_trip.csv", index=False)
    pm = pd.DataFrame(per_model)
    agg = pm.groupby("model").apply(lambda g: pd.Series({
        "clf_top1": np.average(g.clf_top1, weights=g.points), "hmm_top1": np.average(g.hmm_top1, weights=g.points),
        "clf_continuity": np.average(g.clf_continuity, weights=g.points), "hmm_continuity": np.average(g.hmm_continuity, weights=g.points),
        "clf_route_f1": g.clf_route_f1.mean(), "hmm_route_f1": g.hmm_route_f1.mean(),
        "breaks": g.breaks.sum(), "decode_s_total": g.decode_s.sum()}), include_groups=False).reset_index()
    agg.to_csv(config.RESULTS / "hmm_by_model.csv", index=False)
    print(pt.round(4).to_string())
    print(agg.round(4).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(pt))
    for a, (col, title) in zip(ax, [("top1", "Top-1 accuracy"), ("continuity", "Route continuity"), ("route_f1", "Route F1 (length-weighted)")]):
        a.bar(x - 0.2, pt[f"clf_{col}"], 0.4, label="classifier only")
        a.bar(x + 0.2, pt[f"hmm_{col}"], 0.4, label="classifier + HMM")
        a.set_xticks(x, pt.trip_id.str[-2:], rotation=0)
        a.set(title=f"{title} per student-test trip", xlabel="trip (last 2 digits)",
              ylim=(min(pt[f"clf_{col}"].min(), pt[f"hmm_{col}"].min()) - 0.05, 1.0))
        a.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(config.PLOTS / "hmm_vs_classifier_per_trip.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    agg = agg.sort_values("hmm_top1")
    x = np.arange(len(agg))
    ax.bar(x - 0.2, agg.clf_top1, 0.4, label="classifier only")
    ax.bar(x + 0.2, agg.hmm_top1, 0.4, label="classifier + HMM")
    ax.set_xticks(x, agg.model, rotation=30, ha="right")
    ax.set(ylim=(0.5, 1), ylabel="student-test Top-1", title="Effect of HMM decoding for each model")
    ax.legend(); fig.tight_layout(); fig.savefig(config.PLOTS / "hmm_by_model.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
