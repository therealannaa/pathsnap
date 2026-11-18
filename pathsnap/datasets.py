"""Loading the feature tables written by scripts/02_build_dataset.py."""
import pandas as pd

from . import config

LABEL_MAX_DIST = 60.0


def load_partition(part, train_filter=False, point_stride=1):
    df = pd.read_parquet(config.FEATURES_DIR / f"{part}.parquet")
    df["trip_id"] = df["trip_id"].astype(str)
    if train_filter:
        pts = pd.read_csv(config.FEATURES_DIR / f"{part}_points.csv", dtype={"trip_id": str})
        good = pts[pts.label_dist <= LABEL_MAX_DIST][["trip_id", "point_idx"]]
        df = df.merge(good, on=["trip_id", "point_idx"], how="inner")
    if point_stride > 1:
        df = df[df.point_idx % point_stride == 0]
    return df.reset_index(drop=True)
