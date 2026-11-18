"""Zoomed Folium maps for the report: the stretch of a student-test trip where the
classifier-only prediction disagrees most with the HMM, with every layer shown.

    python scripts/07_detail_maps.py 00000077 00000009
"""
import json
import sys
from pathlib import Path

import folium
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap import config, data_io, maps, pipeline  # noqa: E402

WINDOW = 120  # GPS points


def main(trip_ids):
    model = joblib.load(config.ARTIFACTS / "final_model.joblib")
    hmm_cfg = json.loads((config.ARTIFACTS / "hmm_config.json").read_text())
    for tid in trip_ids:
        trip = data_io.load_trip(config.STUDENT_TEST_DIR, tid)
        prep = pipeline.prepare_trip(trip)
        pipeline.attach_labels(prep, trip)
        out = pipeline.predict_trip(prep, model, hmm_cfg)
        net, L = prep["net"], prep["label_arc"]
        clf_wrong = net.arc_orig[out["clf_arc"]] != L
        hmm_right = net.arc_orig[out["hmm_arc"]] == L
        gain = np.convolve((clf_wrong & hmm_right).astype(float), np.ones(WINDOW), "valid")
        s = int(np.argmax(gain))
        sl = slice(s, s + WINDOW)
        ref = [net.orig_to_local[a] for a in trip["route"] if a in net.orig_to_local]
        track = trip["track"][sl]
        m = maps.trip_map(net, track, prep["pxy"][sl], ref_route_local=ref,
                          title=f"Trip {tid}, GPS points {s}-{s + WINDOW - 1}",
                          layers={"Classifier-only prediction": (np.unique(out["clf_arc"][sl]).tolist(), "#ef6c00"),
                                  "Classifier + HMM prediction": (np.unique(out["hmm_arc"][sl]).tolist(), "#c62828")})
        m.save(str(config.MAPS / f"detail_{tid}.html"))
        print(tid, "window", s, "clf acc", (~clf_wrong[sl]).mean().round(3), "hmm acc", hmm_right[sl].mean().round(3))


if __name__ == "__main__":
    main(sys.argv[1:] or ["00000077", "00000009"])
