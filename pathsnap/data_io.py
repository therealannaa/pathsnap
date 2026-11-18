"""Reading and cleaning the raw trip files.

File formats (no headers, whitespace separated):
  .track  lon lat t      GPS observations, t in seconds
  .nodes  lon lat        road-network nodes, row index = node id
  .arcs   from to        directed road segments, row index = arc id
  .route  arc_id         reference arcs in travel order (not per GPS point)
"""
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path, ncols, dtype=float):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    if df.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {df.shape[1]}")
    return df.to_numpy(dtype=dtype)


def list_trips(folder):
    folder = Path(folder)
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


def clean_track(track):
    """Drop invalid rows, sort by time and remove duplicate timestamps.

    Returns the cleaned array and a dict describing what was removed.
    """
    info = {"raw_points": len(track)}
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    ok &= ~((track[:, 0] == 0) & (track[:, 1] == 0))
    info["invalid_removed"] = int((~ok).sum())
    track = track[ok]
    order = np.argsort(track[:, 2], kind="stable")
    info["out_of_order"] = int((order != np.arange(len(order))).sum())
    track = track[order]
    _, first = np.unique(track[:, 2], return_index=True)
    info["duplicate_time_removed"] = int(len(track) - len(first))
    track = track[np.sort(first)]
    info["clean_points"] = len(track)
    return track, info


def load_trip(folder, trip_id, with_route=True):
    d = Path(folder) / trip_id
    track, clean_info = clean_track(_read(d / f"{trip_id}.track", 3))
    trip = {
        "trip_id": trip_id,
        "track": track,
        "nodes": _read(d / f"{trip_id}.nodes", 2),
        "arcs": _read(d / f"{trip_id}.arcs", 2, dtype=np.int64),
        "clean_info": clean_info,
    }
    route_path = d / f"{trip_id}.route"
    if with_route and route_path.exists():
        trip["route"] = _read(route_path, 1, dtype=np.int64).ravel()
    return trip
