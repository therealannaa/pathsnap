"""Project-wide paths and default parameters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw data lives outside the repository (it is not committed). Override with the
# PATHSNAP_DATA environment variable if the folder is somewhere else.
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", ROOT.parent / "Student folder"))
TRAIN_DIR = DATA_ROOT / "train"
STUDENT_TEST_DIR = DATA_ROOT / "student_test"

ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
MAPS = RESULTS / "maps"
FEATURES_DIR = RESULTS / "features"

for _d in (ARTIFACTS, RESULTS, PLOTS, MAPS, FEATURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Candidate generation
N_CANDIDATES = 16          # arcs kept per GPS point (both directions of a street count separately)
NETWORK_MARGIN_M = 600.0   # road network is cropped to the track bounding box plus this margin
SAMPLE_SPACING_M = 10.0    # spacing of points sampled along arcs for the KD-tree

# Default HMM parameters (overwritten by artifacts/hmm_config.json after tuning)
HMM_DEFAULTS = {
    "beta": 20.0,          # metres; scale of |route distance - GPS distance|
    "emission_weight": 1.0,
    "prob_floor": 1e-4,
    "max_extra_m": 300.0,  # Dijkstra search radius beyond the GPS step distance
}
