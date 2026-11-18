"""Scores the evaluator-test trips once, through the API code, after all modelling choices were frozen.

Nothing here trains, tunes or selects anything: it calls the same /trips and /predict
endpoints the evaluator UI uses and writes what they return.

    python scripts/09_evaluator_results.py [path\\to\\evaluator_test]

Default folder: ../evaluator_test next to the repository.
Output: results/evaluator_test_results.csv
"""
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
eval_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "evaluator_test"
os.environ["PATHSNAP_EVAL_DIR"] = str(eval_dir.resolve())
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from pathsnap import config  # noqa: E402


def main():
    client = TestClient(app)
    rows = []
    for tid in client.get("/trips").json()["trips"]:
        r = client.post("/predict", json={"trip_id": tid}).json()
        e = r["evaluation"]
        rows.append({"trip_id": tid, "points": r["n_points"],
                     "clf_top1": e["point_accuracy_classifier_only"], "hmm_top1": e["point_accuracy_hmm"],
                     "hmm_continuity": r["route_continuity"], "route_precision": e["route_precision"],
                     "route_recall": e["route_recall"], "route_f1": e["route_f1"], "breaks": r["hmm_breaks"],
                     "api_time_s": round(sum(r["timing_s"].values()), 2)})
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS / "evaluator_test_results.csv", index=False)
    w = df.points / df.points.sum()
    print(df.round(4).to_string(index=False))
    print(f"ALL points {df.points.sum()}: classifier {(df.clf_top1 * w).sum():.4f}, HMM {(df.hmm_top1 * w).sum():.4f}, "
          f"continuity {(df.hmm_continuity * w).sum():.4f}, mean route F1 {df.route_f1.mean():.4f}")


if __name__ == "__main__":
    main()
