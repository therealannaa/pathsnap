"""Breakdown of the remaining HMM errors and route discontinuities on the student-test trips.

Error types for points where the decoded arc differs from the reference arc:
  reverse_direction   the decoded arc is the same street in the opposite direction
  on_reference_route  the decoded arc is on the reference route, but the alignment assigns
                      this point to a different (usually neighbouring) arc
  off_route           the decoded arc is not on the reference route at all
Discontinuities: consecutive decoded arcs that are neither equal nor directly connected;
we check whether a road path of at most 100 m joins them.
Output: results/hmm_error_analysis.csv
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap import config, data_io, pipeline  # noqa: E402


def main():
    model = joblib.load(config.ARTIFACTS / "final_model.joblib")
    cfg = json.loads((config.ARTIFACTS / "hmm_config.json").read_text())
    rows = []
    for tid in data_io.list_trips(config.STUDENT_TEST_DIR):
        trip = data_io.load_trip(config.STUDENT_TEST_DIR, tid)
        prep = pipeline.prepare_trip(trip)
        pipeline.attach_labels(prep, trip)
        out = pipeline.predict_trip(prep, model, cfg)
        net, L = prep["net"], prep["label_arc"]
        pred = out["hmm_arc"]
        on_route = set(map(int, trip["route"]))
        r = {"trip_id": tid, "points": len(L), "errors": 0, "reverse_direction": 0, "on_reference_route": 0, "off_route": 0,
             "discontinuities": 0, "reconnect_within_100m": 0}
        for i in np.nonzero(net.arc_orig[pred] != L)[0]:
            r["errors"] += 1
            a, lab = pred[i], net.orig_to_local.get(int(L[i]))
            if lab is not None and net.frm[a] == net.to[lab] and net.to[a] == net.frm[lab]:
                r["reverse_direction"] += 1
            elif int(net.arc_orig[a]) in on_route:
                r["on_reference_route"] += 1
            else:
                r["off_route"] += 1
        for a, b in zip(pred[:-1], pred[1:]):
            if a != b and net.to[a] != net.frm[b]:
                r["discontinuities"] += 1
                d = dijkstra(net.graph, directed=True, indices=int(net.to[a]), limit=100)[net.frm[b]]
                r["reconnect_within_100m"] += int(np.isfinite(d))
        rows.append(r)
    df = pd.DataFrame(rows)
    total = df.drop(columns="trip_id").sum()
    df = pd.concat([df, pd.DataFrame([{"trip_id": "ALL", **total.to_dict()}])], ignore_index=True)
    df.to_csv(config.RESULTS / "hmm_error_analysis.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
