"""Builds the candidate-level feature tables for the train and student-test partitions.

Output:
  results/features/{train,student_test}.parquet   one row per (GPS point, candidate arc)
  results/features/{partition}_points.csv         per-point label diagnostics
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap import config, data_io, pipeline  # noqa: E402


def work(args):
    folder, tid = args
    t0 = time.time()
    trip = data_io.load_trip(folder, tid)
    prep = pipeline.prepare_trip(trip)
    pipeline.attach_labels(prep, trip)
    n = len(prep["pxy"])
    pts = pd.DataFrame({
        "trip_id": tid, "point_idx": np.arange(n),
        "label_arc": prep["label_arc"], "label_dist": prep["label_dist"],
        "label_in_cand": prep["label_in_cand"],
        "nearest_correct": prep["net"].arc_orig[prep["cand"][:, 0]] == prep["label_arc"],
    })
    return prep["df"], pts, time.time() - t0


def main():
    for part, folder in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        trips = data_io.list_trips(folder)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=12) as ex:
            res = list(ex.map(work, [(folder, t) for t in trips]))
        df = pd.concat([r[0] for r in res], ignore_index=True)
        pts = pd.concat([r[1] for r in res], ignore_index=True)
        df.to_parquet(config.FEATURES_DIR / f"{part}.parquet", index=False)
        pts.to_csv(config.FEATURES_DIR / f"{part}_points.csv", index=False)
        print(f"{part}: {len(trips)} trips, {len(pts)} points, {len(df)} candidate rows, "
              f"positive rate {df.y.mean():.4f}, candidate miss rate {1 - pts.label_in_cand.mean():.4f}, "
              f"nearest-arc accuracy {pts.nearest_correct.mean():.4f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
