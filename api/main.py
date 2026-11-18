"""PathSnap API (runs on Laptop 1).

Start:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:    PATHSNAP_EVAL_DIR  folder with the evaluation trips (one sub-folder per trip id).
                           Defaults to the student-test folder.

Endpoints
  GET  /health            model and data status
  GET  /trips             trip ids available for evaluation
  POST /predict           {"trip_id": "..."} -> predicted route + server-side accuracy

The prediction pipeline only reads .track, .nodes and .arcs. The .route file is
read afterwards by `score_prediction` to compute accuracy, and the reference
route itself is never sent to the client.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap import config, data_io, metrics, pipeline  # noqa: E402
from pathsnap.labeling import align_route_to_track  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", config.STUDENT_TEST_DIR))
MODEL = joblib.load(config.ARTIFACTS / "final_model.joblib")
HMM_CFG = json.loads((config.ARTIFACTS / "hmm_config.json").read_text())
MODEL_CFG = json.loads((config.ARTIFACTS / "final_model_config.json").read_text())
log.info("Loaded model %s, HMM beta=%.1f emission_weight=%.2f, evaluation folder %s",
         MODEL_CFG["algorithm"], HMM_CFG["beta"], HMM_CFG["emission_weight"], EVAL_DIR)

app = FastAPI(title="PathSnap map-matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def _trip_ids():
    return data_io.list_trips(EVAL_DIR) if EVAL_DIR.exists() else []


def _polylines(net, arcs):
    """Chain consecutive arcs into polylines of [lat, lon]."""
    lines, cur = [], []
    for a in arcs:
        p0 = net.proj.inverse(np.array([[net.ax[a], net.ay[a]]]))[0]
        p1 = net.proj.inverse(np.array([[net.bx[a], net.by[a]]]))[0]
        if cur and (abs(cur[-1][0] - p0[1]) > 1e-9 or abs(cur[-1][1] - p0[0]) > 1e-9):
            lines.append(cur)
            cur = []
        if not cur:
            cur.append([p0[1], p0[0]])
        cur.append([p1[1], p1[0]])
    if cur:
        lines.append(cur)
    return lines


def score_prediction(trip_id, prep, out):
    """Server-side scoring against the reference route (never returned to the client)."""
    route_file = EVAL_DIR / trip_id / f"{trip_id}.route"
    if not route_file.exists():
        return {"available": False}
    route = np.loadtxt(route_file, dtype=np.int64, ndmin=1)
    trip = data_io.load_trip(EVAL_DIR, trip_id, with_route=False)
    full_xy = prep["proj"].forward(trip["nodes"])
    labels = align_route_to_track(prep["pxy"], prep["times"], route, full_xy, trip["arcs"])["label_arc"]
    net = prep["net"]
    a = trip["arcs"][route]
    lengths = dict(zip(map(int, route), np.hypot(*(full_xy[a[:, 1]] - full_xy[a[:, 0]]).T)))
    lengths.update(dict(zip(map(int, net.arc_orig), net.length)))
    overlap = metrics.route_overlap(net.arc_orig[out["route_local"]], route, lengths)
    return {
        "available": True,
        "point_accuracy_hmm": metrics.top1_accuracy(net.arc_orig[out["hmm_arc"]], labels),
        "point_accuracy_classifier_only": metrics.top1_accuracy(net.arc_orig[out["clf_arc"]], labels),
        **overlap,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_CFG["algorithm"], "hmm": {k: HMM_CFG[k] for k in ("beta", "emission_weight")},
            "eval_dir": str(EVAL_DIR), "trips": len(_trip_ids())}


@app.get("/trips")
def trips():
    ids = _trip_ids()
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in _trip_ids():
        raise HTTPException(404, f"unknown trip {tid}")
    log.info("POST /predict trip=%s: request received", tid)
    t0 = time.time()
    trip = data_io.load_trip(EVAL_DIR, tid, with_route=False)   # prediction never sees the route file
    prep = pipeline.prepare_trip(trip)
    t1 = time.time()
    out = pipeline.predict_trip(prep, MODEL, HMM_CFG)
    t2 = time.time()
    net = prep["net"]
    score = score_prediction(tid, prep, out)
    t3 = time.time()
    log.info("trip=%s points=%d candidates=%d | features %.2fs, classifier+HMM %.2fs, scoring %.2fs | "
             "HMM accuracy=%s classifier-only=%s route F1=%s breaks=%d",
             tid, len(prep["pxy"]), len(prep["df"]), t1 - t0, t2 - t1, t3 - t2,
             f"{score.get('point_accuracy_hmm', float('nan')):.4f}",
             f"{score.get('point_accuracy_classifier_only', float('nan')):.4f}",
             f"{score.get('route_f1', float('nan')):.4f}", len(out["breaks"]))
    track = trip["track"]
    return {
        "trip_id": tid,
        "n_points": int(len(track)),
        "gps": track[:, [1, 0]].round(7).tolist(),
        "predicted_route": _polylines(net, out["route_local"]),
        "predicted_arc_ids": net.arc_orig[out["hmm_arc"]].tolist(),
        "route_continuity": metrics.route_continuity(net, out["hmm_arc"]),
        "hmm_breaks": len(out["breaks"]),
        "evaluation": score,
        "timing_s": {"features": round(t1 - t0, 3), "inference": round(t2 - t1, 3), "scoring": round(t3 - t2, 3)},
    }
