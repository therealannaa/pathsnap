"""Trains and compares all candidate classifiers.

Protocol
- Training data: the 80 training trips, GPS points with reliable labels, every
  3rd GPS point (all of its candidates) to keep CPU time manageable.
- 5-fold GroupKFold by trip: no trip appears in both the fit and validation folds.
- Student test: model refit on the whole training sample, scored on every GPS
  point of the 10 student-test trips.
Outputs: results/model_comparison.csv, results/cv_folds.csv, plots, artifacts/models/*.joblib,
results/student_test_probs.parquet (row probabilities used later by the HMM step).
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap import config, data_io, pipeline  # noqa: E402
from pathsnap.datasets import load_partition  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import route_continuity  # noqa: E402
from pathsnap.models import MODELS, classification_metrics, subsample_rows  # noqa: E402

STRIDE = 3
N_FOLDS = 5
MODEL_DIR = config.ARTIFACTS / "models"
MODEL_DIR.mkdir(exist_ok=True)


def student_networks():
    nets = {}
    for tid in data_io.list_trips(config.STUDENT_TEST_DIR):
        trip = data_io.load_trip(config.STUDENT_TEST_DIR, tid, with_route=False)
        prep = pipeline.prepare_trip(trip)
        nets[tid] = prep
    return nets


def continuity_from_rows(df, p, preps):
    vals, weights = [], []
    for tid, g in df.assign(p=p).groupby("trip_id"):
        best = g.loc[g.groupby("point_idx")["p"].idxmax()].sort_values("point_idx")
        vals.append(route_continuity(preps[tid]["net"], best["arc_local"].to_numpy()))
        weights.append(len(best))
    return float(np.average(vals, weights=weights))


def main(only=None):
    train = load_partition("train", train_filter=True, point_stride=STRIDE)
    test = load_partition("student_test")
    print(f"train rows {len(train)} ({train.trip_id.nunique()} trips), student-test rows {len(test)}")
    Xte = test[FEATURES].to_numpy(np.float32)
    preps = student_networks()
    groups = train.trip_id.to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(train, groups=groups))

    rows, fold_rows = [], []
    test_probs = pd.DataFrame({"trip_id": test.trip_id, "point_idx": test.point_idx, "cand_slot": test.cand_slot, "y": test.y})
    names = only or list(MODELS)
    for name in names:
        family, ctor, cfg, cap = MODELS[name]
        print(f"\n== {name}")
        cv = []
        for f, (tr_i, va_i) in enumerate(folds):
            tr = subsample_rows(train.iloc[tr_i], cap)
            va = train.iloc[va_i]
            m = ctor()
            t0 = time.time()
            m.fit(tr[FEATURES].to_numpy(np.float32), tr.y.to_numpy())
            fit_s = time.time() - t0
            p_tr = m.predict_proba(tr[FEATURES].to_numpy(np.float32))[:, 1]
            p_va = m.predict_proba(va[FEATURES].to_numpy(np.float32))[:, 1]
            mt, mv = classification_metrics(tr, p_tr), classification_metrics(va, p_va)
            fold_rows.append({"model": name, "fold": f, "fit_s": fit_s,
                              **{f"train_{k}": v for k, v in mt.items()}, **{f"val_{k}": v for k, v in mv.items()}})
            cv.append(mv)
            print(f"  fold {f}: val top1 {mv['top1']:.4f} macroF1 {mv['macro_f1']:.4f} logloss {mv['log_loss']:.4f} "
                  f"(train top1 {mt['top1']:.4f}) fit {fit_s:.0f}s")
        cvdf = pd.DataFrame(cv)

        full = subsample_rows(train, cap)
        m = ctor()
        t0 = time.time()
        m.fit(full[FEATURES].to_numpy(np.float32), full.y.to_numpy())
        fit_s = time.time() - t0
        t0 = time.time()
        p = m.predict_proba(Xte)[:, 1]
        inf_s = time.time() - t0
        st = classification_metrics(test, p)
        cont = continuity_from_rows(test, p, preps)
        test_probs[name] = p
        joblib.dump(m, MODEL_DIR / f"{name}.joblib", compress=3)
        cm = confusion_matrix(test.y, (p >= 0.5).astype(int))
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(cm, display_labels=["other arc", "reference arc"]).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"{name}\nstudent test, threshold 0.5")
        fig.tight_layout(); fig.savefig(config.PLOTS / f"cm_{name}.png", dpi=120); plt.close(fig)

        row = {"model": name, "family": family, "config": cfg,
               "cv_macro_f1_mean": cvdf.macro_f1.mean(), "cv_macro_f1_std": cvdf.macro_f1.std(),
               "cv_top1_mean": cvdf.top1.mean(), "cv_top1_std": cvdf.top1.std(),
               "cv_log_loss_mean": cvdf.log_loss.mean(),
               "test_top1": st["top1"], "test_macro_f1": st["macro_f1"], "test_macro_precision": st["macro_precision"],
               "test_macro_recall": st["macro_recall"], "test_log_loss": st["log_loss"],
               "test_route_continuity": cont, "fit_time_s": fit_s, "train_rows": len(full),
               "inference_ms_per_1k_rows": 1000 * inf_s / len(test) * 1000,
               "train_top1_mean": np.mean([r["train_top1"] for r in fold_rows if r["model"] == name])}
        rows.append(row)
        print(f"  student test: top1 {st['top1']:.4f} macroF1 {st['macro_f1']:.4f} logloss {st['log_loss']:.4f} continuity {cont:.4f}")

        if name == "MLP":
            mlp = m[-1]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(mlp.loss_curve_, label="training loss (log loss)")
            ax2 = ax.twinx()
            ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation accuracy (10% hold-out)")
            ax.set(xlabel="epoch", ylabel="training loss", title="MLP training history")
            ax2.set_ylabel("validation accuracy")
            fig.legend(loc="upper right"); fig.tight_layout(); fig.savefig(config.PLOTS / "mlp_training_history.png", dpi=130); plt.close(fig)
        if hasattr(m, "feature_importances_"):
            imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values()
            fig, ax = plt.subplots(figsize=(6, 6))
            imp.plot.barh(ax=ax)
            ax.set_title(f"{name} feature importance")
            fig.tight_layout(); fig.savefig(config.PLOTS / f"importance_{name}.png", dpi=120); plt.close(fig)

    # merge with any earlier results so a single model can be re-run
    out = config.RESULTS / "model_comparison.csv"
    res = pd.DataFrame(rows)
    folds_df = pd.DataFrame(fold_rows)
    if only and out.exists():
        res = pd.concat([pd.read_csv(out).query("model not in @names"), res])
        folds_df = pd.concat([pd.read_csv(config.RESULTS / "cv_folds.csv").query("model not in @names"), folds_df])
        old = pd.read_parquet(config.RESULTS / "student_test_probs.parquet")
        for c in old.columns:
            if c not in test_probs:
                test_probs[c] = old[c].to_numpy()
    res.to_csv(out, index=False)
    folds_df.to_csv(config.RESULTS / "cv_folds.csv", index=False)
    test_probs.to_parquet(config.RESULTS / "student_test_probs.parquet", index=False)
    (config.RESULTS / "training_setup.json").write_text(json.dumps({
        "point_stride": STRIDE, "folds": N_FOLDS, "cv": "GroupKFold by trip_id", "seed": config.SEED,
        "train_rows": len(train), "train_trips": int(train.trip_id.nunique()), "features": FEATURES}, indent=2))
    plot_comparison(res)
    print(res[["model", "cv_macro_f1_mean", "cv_top1_mean", "test_top1", "test_macro_f1", "test_log_loss",
               "test_route_continuity", "inference_ms_per_1k_rows"]].round(4).to_string())


def plot_comparison(res):
    res = res.sort_values("test_top1")
    folds = pd.read_csv(config.RESULTS / "cv_folds.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = list(res.model)
    ax.boxplot([folds[folds.model == m].val_macro_f1 for m in order], tick_labels=order, vert=True)
    ax.set(title="Cross-validation macro F1 (5 grouped folds)", ylabel="macro F1")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(config.PLOTS / "cmp_cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(res))
    ax.bar(x - 0.2, res.test_top1, 0.4, label="Top-1 accuracy")
    ax.bar(x + 0.2, res.test_macro_f1, 0.4, label="Macro F1")
    ax.set_xticks(x, res.model, rotation=30, ha="right")
    ax.set(ylim=(0, 1), title="Student-test Top-1 accuracy and macro F1", ylabel="score")
    ax.legend(); fig.tight_layout(); fig.savefig(config.PLOTS / "cmp_test_top1_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(res.model, res.test_log_loss, color="tab:red")
    ax.set(title="Student-test log loss (lower is better)", ylabel="log loss", yscale="log")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(config.PLOTS / "cmp_test_log_loss.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(res.inference_ms_per_1k_rows, res.test_top1)
    for _, r in res.iterrows():
        ax.annotate(r.model, (r.inference_ms_per_1k_rows, r.test_top1), fontsize=8)
    ax.set(xscale="log", xlabel="inference time (ms per 1,000 candidate rows)", ylabel="student-test Top-1",
           title="Accuracy vs inference cost")
    fig.tight_layout(); fig.savefig(config.PLOTS / "cmp_speed_vs_accuracy.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
