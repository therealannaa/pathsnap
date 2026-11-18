"""Training-history plots for the models that train iteratively, plus a CV Top-1 box plot.

- MLP: loss curve and validation accuracy stored in the saved model (03).
- LightGBM (tuned params): binary log loss per boosting round on the training
  trips and on a held-out group of training trips (16 trips).
Only models with a real per-iteration history get a curve.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap import config  # noqa: E402
from pathsnap.datasets import load_partition  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402


def main():
    mlp = joblib.load(config.ARTIFACTS / "models" / "MLP.joblib")[-1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_)
    ax[0].set(title="MLP training loss (log loss)", xlabel="epoch", ylabel="loss")
    ax[1].plot(np.arange(1, len(mlp.validation_scores_) + 1), mlp.validation_scores_, color="tab:orange")
    ax[1].set(title="MLP validation accuracy (10% early-stopping split)", xlabel="epoch", ylabel="accuracy")
    fig.tight_layout(); fig.savefig(config.PLOTS / "mlp_training_history.png", dpi=130); plt.close(fig)

    params = json.loads((config.ARTIFACTS / "final_model_config.json").read_text())["params"]
    train = load_partition("train", train_filter=True, point_stride=3)
    trips = np.array(sorted(train.trip_id.unique()))
    fit_i, val_i = next(GroupShuffleSplit(1, test_size=16 / 80, random_state=config.SEED).split(trips, groups=trips))
    tr = train[train.trip_id.isin(set(trips[fit_i]))]
    va = train[train.trip_id.isin(set(trips[val_i]))]
    m = LGBMClassifier(subsample_freq=1, random_state=config.SEED, verbose=-1, **params)
    m.fit(tr[FEATURES].to_numpy(np.float32), tr.y, eval_set=[(tr[FEATURES].to_numpy(np.float32), tr.y),
                                                          (va[FEATURES].to_numpy(np.float32), va.y)],
          eval_names=["training trips", "held-out trips"], eval_metric="binary_logloss")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, h in m.evals_result_.items():
        ax.plot(h["binary_logloss"], label=name)
    ax.set(title="Tuned LightGBM: log loss per boosting round", xlabel="boosting round", ylabel="binary log loss", yscale="log")
    ax.legend(); fig.tight_layout(); fig.savefig(config.PLOTS / "lightgbm_training_history.png", dpi=130); plt.close(fig)
    pd.DataFrame({k: v["binary_logloss"] for k, v in m.evals_result_.items()}).to_csv(
        config.RESULTS / "lightgbm_training_history.csv", index_label="round")

    folds = pd.read_csv(config.RESULTS / "cv_folds.csv")
    order = folds.groupby("model").val_top1.median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([folds[folds.model == mname].val_top1 for mname in order], tick_labels=list(order))
    ax.set(title="Cross-validation observation-level Top-1 (5 grouped folds)", ylabel="Top-1 accuracy")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(config.PLOTS / "cmp_cv_top1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    g = folds.groupby("model")[["train_top1", "val_top1"]].mean().loc[order]
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.train_top1, 0.4, label="training folds")
    ax.bar(x + 0.2, g.val_top1, 0.4, label="validation folds")
    ax.set_xticks(x, g.index, rotation=30, ha="right")
    ax.set(ylim=(0.75, 1), title="Train vs validation Top-1 (overfitting check)", ylabel="Top-1")
    ax.legend(); fig.tight_layout(); fig.savefig(config.PLOTS / "cmp_train_vs_val.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
