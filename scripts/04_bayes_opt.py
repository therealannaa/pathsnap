"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model.

Search: grouped 3-fold CV on the training trips (every 6th GPS point, to fit
the time budget). Objective: mean observation-level Top-1 accuracy, which is
the metric the route decoder depends on (the HMM consumes the ranking of
candidates per point, not the 0.5-threshold decision).

After the search the tuned model is evaluated with exactly the protocol of
03_train_models.py (5 grouped folds, every 3rd point, same student-test rows),
so tuned and untuned numbers are directly comparable. The tuned model is saved
as artifacts/final_model.joblib.
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
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap import config  # noqa: E402
from pathsnap.datasets import load_partition  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.models import classification_metrics  # noqa: E402

N_TRIALS = 40
SEARCH_STRIDE = 6


def make_model(params):
    return LGBMClassifier(subsample_freq=1, random_state=config.SEED, verbose=-1, n_jobs=12, **params)


def space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def cv_eval(df, params, n_splits, record=None):
    out = []
    for f, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=df.trip_id)):
        m = make_model(params)
        m.fit(df.iloc[tr][FEATURES].to_numpy(np.float32), df.iloc[tr].y.to_numpy())
        p = m.predict_proba(df.iloc[va][FEATURES].to_numpy(np.float32))[:, 1]
        r = classification_metrics(df.iloc[va], p)
        if record is not None:
            p_tr = m.predict_proba(df.iloc[tr][FEATURES].to_numpy(np.float32))[:, 1]
            record.append({"fold": f, **{f"val_{k}": v for k, v in r.items()},
                           "train_top1": classification_metrics(df.iloc[tr], p_tr)["top1"]})
        out.append(r)
    return pd.DataFrame(out)


def main():
    search = load_partition("train", train_filter=True, point_stride=SEARCH_STRIDE)
    print(f"search rows {len(search)}")
    t_start = time.time()

    def objective(trial):
        res = cv_eval(search, space(trial), 3)
        trial.set_user_attr("cv_macro_f1", float(res.macro_f1.mean()))
        trial.set_user_attr("cv_log_loss", float(res.log_loss.mean()))
        return float(res.top1.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=10),
                                study_name="lightgbm_top1")
    # trial 0 = the untuned configuration from 03_train_models.py, as a reference point
    study.enqueue_trial({"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
                         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1e-3, "reg_alpha": 1e-3})
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    search_s = time.time() - t_start
    trials = study.trials_dataframe()
    trials.to_csv(config.RESULTS / "bayes_opt_trials.csv", index=False)
    best = study.best_params
    print("best", best, study.best_value)

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = trials.value.to_numpy()
    ax.plot(vals, "o", ms=4, label="trial")
    ax.plot(np.maximum.accumulate(vals), "-", label="best so far")
    ax.set(xlabel="trial", ylabel="3-fold grouped CV Top-1", title="Bayesian optimisation history (TPE)")
    ax.legend(); fig.tight_layout(); fig.savefig(config.PLOTS / "bayes_opt_history.png", dpi=130); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 4))
        pd.Series(imp).sort_values().plot.barh(ax=ax)
        ax.set_title("Hyperparameter importance (fANOVA)")
        fig.tight_layout(); fig.savefig(config.PLOTS / "bayes_opt_param_importance.png", dpi=130); plt.close(fig)
    except Exception as e:  # importance needs enough completed trials
        print("importance skipped:", e)

    # ---- tuned vs untuned with the 03 protocol
    train = load_partition("train", train_filter=True, point_stride=3)
    test = load_partition("student_test")
    folds = []
    cv = cv_eval(train, best, 5, record=folds)
    m = make_model(best)
    t0 = time.time()
    m.fit(train[FEATURES].to_numpy(np.float32), train.y.to_numpy())
    fit_s = time.time() - t0
    t0 = time.time()
    p = m.predict_proba(test[FEATURES].to_numpy(np.float32))[:, 1]
    inf_s = time.time() - t0
    st = classification_metrics(test, p)
    joblib.dump(m, config.ARTIFACTS / "final_model.joblib", compress=3)
    probs = pd.read_parquet(config.RESULTS / "student_test_probs.parquet")
    probs["LightGBM_tuned"] = p
    probs.to_parquet(config.RESULTS / "student_test_probs.parquet", index=False)

    untuned = pd.read_csv(config.RESULTS / "model_comparison.csv").set_index("model").loc["LightGBM"]
    cmp_rows = [
        {"model": "LightGBM untuned", "cv_macro_f1": untuned.cv_macro_f1_mean, "cv_top1": untuned.cv_top1_mean,
         "test_top1": untuned.test_top1, "test_macro_f1": untuned.test_macro_f1, "test_log_loss": untuned.test_log_loss,
         "inference_ms_per_1k_rows": untuned.inference_ms_per_1k_rows, "fit_time_s": untuned.fit_time_s},
        {"model": "LightGBM tuned", "cv_macro_f1": cv.macro_f1.mean(), "cv_top1": cv.top1.mean(),
         "test_top1": st["top1"], "test_macro_f1": st["macro_f1"], "test_log_loss": st["log_loss"],
         "inference_ms_per_1k_rows": 1000 * inf_s / len(test) * 1000, "fit_time_s": fit_s},
    ]
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(config.RESULTS / "tuned_vs_untuned.csv", index=False)
    pd.DataFrame(folds).to_csv(config.RESULTS / "tuned_cv_folds.csv", index=False)
    (config.ARTIFACTS / "final_model_config.json").write_text(json.dumps({
        "algorithm": "LightGBM LGBMClassifier", "params": best, "features": FEATURES,
        "trained_on": "80 training trips, points with label distance <= 60 m, every 3rd GPS point",
        "bayes_trials": N_TRIALS, "search_time_s": round(search_s, 1), "seed": config.SEED}, indent=2))
    print(cmp.round(4).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    metrics_ = ["cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1"]
    x = np.arange(len(metrics_))
    ax[0].bar(x - 0.2, cmp.iloc[0][metrics_].astype(float), 0.4, label="untuned")
    ax[0].bar(x + 0.2, cmp.iloc[1][metrics_].astype(float), 0.4, label="tuned")
    ax[0].set_xticks(x, metrics_, rotation=20)
    lo = float(cmp[metrics_].min().min())
    ax[0].set(ylim=(lo - 0.03, 1), title="Untuned vs tuned LightGBM")
    ax[0].legend()
    ax[1].bar(["untuned", "tuned"], cmp.test_log_loss, color=["tab:blue", "tab:orange"])
    ax[1].set(title="Student-test log loss")
    fig.tight_layout(); fig.savefig(config.PLOTS / "tuned_vs_untuned.png", dpi=130); plt.close(fig)
    print(f"search {search_s:.0f}s")


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
