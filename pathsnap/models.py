"""Model registry and evaluation helpers for the candidate classifier."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import SEED

# name -> (family, constructor, key configuration text, max training rows or None)
MODELS = {
    "LogisticRegression": ("Linear", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
                           "StandardScaler, L2, C=1.0", None),
    "GaussianNB": ("Probabilistic", lambda: GaussianNB(var_smoothing=1e-9), "var_smoothing=1e-9", None),
    "KNN": ("Instance-based", lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1)),
            "k=50, distance-weighted, scaled; 150k training rows", 150_000),
    "DecisionTree": ("Tree", lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED),
                     "max_depth=12, min_samples_leaf=50", None),
    "RandomForest": ("Ensemble (bagging)", lambda: RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=5,
                                                                          n_jobs=-1, random_state=SEED),
                     "150 trees, max_depth=18, min_samples_leaf=5", None),
    "ExtraTrees": ("Ensemble (bagging)", lambda: ExtraTreesClassifier(n_estimators=150, max_depth=20, min_samples_leaf=5,
                                                                      n_jobs=-1, random_state=SEED),
                   "150 trees, max_depth=20, min_samples_leaf=5", None),
    "HistGradientBoosting": ("Boosting", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                                                              early_stopping=False, random_state=SEED),
                             "300 iterations, lr=0.1, 31 leaves", None),
    "LightGBM": ("Boosting", lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                                    subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1),
                 "400 trees, lr=0.05, 63 leaves, 0.8 row/col sampling", None),
    "MLP": ("Neural network", lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                                                   batch_size=1024, learning_rate_init=1e-3,
                                                                                   max_iter=60, early_stopping=True,
                                                                                   n_iter_no_change=6, random_state=SEED)),
            "(64,32) ReLU, Adam lr=1e-3, early stopping; 300k training rows", 300_000),
}


def top1_from_rows(df, p):
    """Observation-level Top-1 accuracy: the highest-scoring candidate per GPS point must be the reference arc.
    Points whose reference arc is missing from the candidate set count as errors."""
    tmp = pd.DataFrame({"g": df["trip_id"].astype(str) + "_" + df["point_idx"].astype(str), "p": p, "y": df["y"].to_numpy()})
    best = tmp.loc[tmp.groupby("g", sort=False)["p"].idxmax()]
    return float(best["y"].mean())


def classification_metrics(df, p, threshold=0.5):
    y = df["y"].to_numpy()
    pred = (p >= threshold).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, pc, labels=[0, 1]),
        "top1": top1_from_rows(df, p),
    }


def subsample_rows(df, max_rows, seed=SEED):
    """Keep whole GPS points (all their candidates) so ranking is still meaningful."""
    if max_rows is None or len(df) <= max_rows:
        return df
    keys = df["trip_id"].astype(str) + "_" + df["point_idx"].astype(str)
    uk = keys.unique()
    rng = np.random.default_rng(seed)
    keep = rng.choice(uk, size=int(len(uk) * max_rows / len(df)), replace=False)
    return df[keys.isin(set(keep))]
