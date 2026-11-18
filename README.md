# PathSnap: map matching with a candidate classifier and HMM/Viterbi decoding

PathSnap takes a noisy GPS track and the road network around it and returns the road segment (arc) the vehicle was on at every GPS point, plus a continuous reconstructed route.

The pipeline:

1. Project each trip to a local metric plane (equirectangular, centred on the trip).
2. For every GPS point, take the 16 nearest directed arcs as candidates.
3. Score every (point, candidate) pair with a trained classifier (23 geometric, heading, speed and connectivity features).
4. Decode the most likely arc sequence with an HMM: the classifier score is the emission, and the transition compares driving distance on the road graph with the straight-line GPS distance.
5. Fill gaps between consecutive arcs with shortest paths to get one connected route.

## Results on the 10 student-test trips

| | Point accuracy (Top-1) | Route continuity | Route F1 |
|---|---|---|---|
| Nearest arc (baseline) | 51.8% | | |
| Tuned LightGBM, classifier only | 88.4% | 95.4% | 85.5% |
| Tuned LightGBM + HMM/Viterbi | 92.6% | 98.3% | 98.9% |

The full analysis is in [report/PathSnap_Technical_Report.pdf](report/PathSnap_Technical_Report.pdf).

## Team contributions

| Member | Responsibility | Main files |
|---|---|---|
| Anna | HMM and Viterbi decoding, route reconstruction, continuity and route metrics, decoder tuning, end-to-end pipeline, FastAPI backend and deployment | `pathsnap/hmm.py`, `pathsnap/metrics.py`, `pathsnap/pipeline.py`, `scripts/05_hmm.py`, `scripts/08_error_analysis.py`, `api/` |
| Annanya | Feature engineering, the nine models, grouped cross-validation, model selection, Bayesian optimisation | `pathsnap/features.py`, `pathsnap/models.py`, `pathsnap/datasets.py`, `scripts/03_train_models.py`, `scripts/04_bayes_opt.py`, `scripts/06_extra_plots.py` |
| Bhoomika | Data loading and cleaning, projection, reference-route alignment, EDA, Folium maps | `pathsnap/data_io.py`, `pathsnap/geo.py`, `pathsnap/labeling.py`, `pathsnap/maps.py`, `scripts/01_eda.py`, `scripts/07_detail_maps.py` |
| Bhuvan | Road-network cropping, candidate generation, dataset build, evaluator UI | `pathsnap/network.py`, `scripts/02_build_dataset.py`, `ui/` |

## Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | Library code: data loading and cleaning (`data_io.py`), projection (`geo.py`), road network and candidate search (`network.py`), label alignment (`labeling.py`), features (`features.py`), model registry (`models.py`), HMM/Viterbi (`hmm.py`), metrics, Folium maps, end-to-end pipeline |
| `scripts/` | Numbered experiment scripts, run in order (see below) |
| `api/` | FastAPI backend for Laptop 1 |
| `ui/` | Evaluator web page for Laptop 2 and a small static server |
| `artifacts/` | Final model (`final_model.joblib`), its configuration, the HMM configuration, and every compared model under `artifacts/models/` |
| `results/` | CSV results, experiment logs, data-quality notes, `plots/` and Folium `maps/` |
| `report/` | Technical report (PDF and HTML) and the script that builds it from `results/` |
The raw data is not in the repository. By default the code looks for it in `../Student folder` (with `train/` and `student_test/` inside). Set `PATHSNAP_DATA` to use another location.

## Installation

Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the experiments

Run from the repository root:

```powershell
python scripts/02_build_dataset.py   # candidates, features and labels for all trips (about 1 min)
python scripts/01_eda.py             # EDA tables, plots, Folium maps, data-quality notes
python scripts/03_train_models.py    # 9 models, 5-fold grouped CV + student-test evaluation
python scripts/04_bayes_opt.py       # Optuna TPE search for LightGBM, tuned vs untuned
python scripts/05_hmm.py             # HMM decoder tuning, classifier-only vs HMM, maps
python scripts/06_extra_plots.py     # MLP and LightGBM training histories, CV box plots
python scripts/07_detail_maps.py     # zoomed Folium maps for the report
python scripts/08_error_analysis.py  # breakdown of the remaining HMM errors
python scripts/09_evaluator_results.py  # scores the evaluator trips once through the API (no tuning)
python report/build_report.py        # report HTML + PDF from results/
```

`02` must run before `01`, because the EDA reads the label diagnostics that `02` writes. All randomness uses seed 42. Approximate run times on a 16-thread laptop: `03` 30 min, `04` 15 min, `05` 15 min, the rest a few minutes each.

`artifacts/final_model.joblib` and `artifacts/hmm_config.json` are already committed, so the API runs without retraining.

## Running the two-laptop demo

Both laptops must be on the same network.

### Laptop 1: API

Install the requirements (see Installation). Put the evaluation trips in one folder with a sub-folder per trip id, in the same layout as `student_test/`. Then:

```powershell
$env:PATHSNAP_EVAL_DIR = "C:\path\to\evaluation_trips"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find the laptop's IP address with `ipconfig`. Windows Firewall may ask to allow Python on private networks; allow it. The console prints one log line per request with the trip id, timings and accuracy. Show this console to the evaluator.

### Laptop 2: UI

Laptop 2 only needs Python 3 (standard library) and a browser with internet access for the map tiles. No `pip install` is needed.

```powershell
git clone https://github.com/therealannaa/pathsnap.git
cd pathsnap
python ui/serve.py
```

Open http://localhost:8501, type Laptop 1's address (for example `http://192.168.1.20:8000`), press **Load trips**, then click a trip.
The UI sends only the trip id. The API reads `.track`, `.nodes` and `.arcs` for prediction. It then reads `.route` separately to compute accuracy, and it never returns the reference route to the UI.

### API reference

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | model name, HMM settings, evaluation folder, trip count |
| GET | `/trips` | none | `{"trips": [ids]}` |
| POST | `/predict` | `{"trip_id": "00000000"}` | GPS points, predicted route polylines, predicted arc id per point, continuity, server-side accuracy, timings |

Interactive docs are at `http://<laptop-1>:8000/docs`.
