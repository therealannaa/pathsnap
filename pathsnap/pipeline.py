"""End-to-end processing of one trip: projection, network, candidates, features,
classifier scoring and HMM decoding. Reference labels are handled separately in
`attach_labels`, which the prediction path never calls."""
import numpy as np

from . import config
from .features import FEATURES, build_features
from .geo import LocalProjection
from .hmm import reconstruct_route, viterbi
from .labeling import align_route_to_track
from .network import RoadNetwork


def prepare_trip(trip, k=config.N_CANDIDATES):
    proj = LocalProjection.for_track(trip["track"])
    net, pxy = RoadNetwork.from_trip(trip, proj)
    times = trip["track"][:, 2]
    cand, cd, ct = net.candidates(pxy, k=k)
    df = build_features(net, pxy, times, cand, cd, ct)
    df.insert(0, "trip_id", trip["trip_id"])
    return {"trip_id": trip["trip_id"], "proj": proj, "net": net, "pxy": pxy, "times": times,
            "cand": cand, "cd": cd, "ct": ct, "df": df}


def attach_labels(prep, trip):
    """Server-side / training only: per-point reference arc from the .route file."""
    full_xy = prep["proj"].forward(trip["nodes"])
    al = align_route_to_track(prep["pxy"], prep["times"], trip["route"], full_xy, trip["arcs"])
    prep["label_arc"] = al["label_arc"]
    prep["label_dist"] = al["label_dist"]
    df = prep["df"]
    df["y"] = (df["arc_orig"].to_numpy() == al["label_arc"][df["point_idx"].to_numpy()]).astype(np.int8)
    in_cand = (prep["net"].arc_orig[np.where(prep["cand"] >= 0, prep["cand"], 0)] == al["label_arc"][:, None]) & (prep["cand"] >= 0)
    prep["label_in_cand"] = in_cand.any(axis=1)
    a = trip["arcs"][trip["route"]]
    prep["ref_len"] = dict(zip(map(int, trip["route"]),
                               np.hypot(*(full_xy[a[:, 1]] - full_xy[a[:, 0]]).T)))
    return prep


def probs_matrix(prep, row_probs):
    n, k = prep["cand"].shape
    P = np.zeros((n, k))
    df = prep["df"]
    P[df["point_idx"].to_numpy(), df["cand_slot"].to_numpy()] = row_probs
    return P


def predict_trip(prep, model, hmm_params=None, use_hmm=True):
    """Runs classifier (+ optional HMM). Returns per-point local arcs and the reconstructed route."""
    X = prep["df"][FEATURES].to_numpy(dtype=np.float32)
    row_p = model.predict_proba(X)[:, 1]
    P = probs_matrix(prep, row_p)
    cand = prep["cand"]
    clf_slot = np.argmax(np.where(cand >= 0, P, -1), axis=1)
    clf_arc = cand[np.arange(len(cand)), clf_slot]
    out = {"row_probs": row_p, "P": P, "clf_arc": clf_arc}
    if use_hmm:
        slots, breaks = viterbi(prep["net"], prep["pxy"], prep["times"], cand, prep["ct"], P, hmm_params)
        out["hmm_arc"] = cand[np.arange(len(cand)), slots]
        out["breaks"] = breaks
        out["route_local"] = reconstruct_route(prep["net"], out["hmm_arc"])
    return out
