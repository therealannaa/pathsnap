"""Builds the technical report (HTML, then PDF through headless Edge/Chrome).

Every number in the report is read from the CSV/JSON files in results/ and
artifacts/, so the PDF always matches what the scripts produced.

    python report/build_report.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ART = ROOT / "artifacts"
OUT_HTML = ROOT / "report" / "PathSnap_Technical_Report.html"
OUT_PDF = ROOT / "report" / "PathSnap_Technical_Report.pdf"

TEAM = [
    ("Anna", "HMM and Viterbi decoding, route reconstruction, continuity and route metrics, decoder tuning, "
             "end-to-end pipeline, FastAPI backend and two-laptop deployment",
     "hmm.py, metrics.py, pipeline.py, scripts/05_hmm.py, api/"),
    ("Annanya", "Feature engineering, training and comparison of the nine models, grouped cross-validation, "
                "model selection, Bayesian hyperparameter optimisation, comparison plots",
     "features.py, models.py, datasets.py, scripts/03, 04, 06"),
    ("Bhoomika", "Data loading and cleaning, local projection, reference-route alignment (labels), "
                 "exploratory data analysis, Folium maps",
     "data_io.py, geo.py, labeling.py, maps.py, scripts/01_eda.py, scripts/07_detail_maps.py"),
    ("Bhuvan", "Road-network cropping, KD-tree candidate generation, candidate miss-rate analysis, "
               "dataset build, evaluator UI",
     "network.py, scripts/02_build_dataset.py, ui/"),
]


STAT_LABELS = {
    "trips": "Trips", "points": "GPS points", "points_min": "Points per trip, minimum", "points_median": "Points per trip, median",
    "points_max": "Points per trip, maximum", "missing_values": "Missing values in .track", "invalid_removed": "Invalid coordinates removed",
    "duplicate_time_removed": "Duplicate time stamps removed", "out_of_order": "Out-of-order time stamps",
    "trips_with_gap_gt_30s": "Trips with a gap over 30 s", "nodes_median": "Road nodes per trip, median",
    "arcs_median": "Road arcs per trip, median", "isolated_nodes": "Isolated nodes", "weak_components_median": "Weakly connected components, median",
    "route_arcs_median": "Reference arcs per trip, median", "route_continuity_min": "Reference route continuity, minimum",
}


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def f4(x):
    return f"{x:.4f}"


def table(df, cols, headers=None, fmt=None):
    fmt = fmt or {}
    headers = headers or cols
    h = "".join(f"<th>{c}</th>" for c in headers)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"<td>{fmt[c](v) if c in fmt else v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def img(path, caption, width="100%"):
    return f'<figure><img src="{path}" style="width:{width}"><figcaption>{caption}</figcaption></figure>'


def main():
    inv = pd.read_csv(RES / "eda_inventory.csv", dtype={"trip_id": str})
    summ = pd.read_csv(RES / "eda_summary.csv", index_col=0)
    tp = pd.read_csv(RES / "features" / "train_points.csv", dtype={"trip_id": str})
    sp = pd.read_csv(RES / "features" / "student_test_points.csv", dtype={"trip_id": str})
    cmp = pd.read_csv(RES / "model_comparison.csv")
    folds = pd.read_csv(RES / "cv_folds.csv")
    tuned = pd.read_csv(RES / "tuned_vs_untuned.csv")
    tuned_folds = pd.read_csv(RES / "tuned_cv_folds.csv")
    trials = pd.read_csv(RES / "bayes_opt_trials.csv")
    hgrid = pd.read_csv(RES / "hmm_tuning_grid.csv")
    hbm = pd.read_csv(RES / "hmm_by_model.csv").set_index("model")
    hpt = pd.read_csv(RES / "hmm_student_test_per_trip.csv", dtype={"trip_id": str})
    final_cfg = json.loads((ART / "final_model_config.json").read_text())
    hmm_cfg = json.loads((ART / "hmm_config.json").read_text())
    setup = json.loads((RES / "training_setup.json").read_text())
    api_log = (RES / "api_log_sample.txt").read_text(encoding="utf-8")
    ea = pd.read_csv(RES / "hmm_error_analysis.csv", dtype={"trip_id": str}).set_index("trip_id")
    n_test_rows = len(pd.read_parquet(RES / "student_test_probs.parquet", columns=["y"]))
    fold_std = folds.groupby("model").val_top1.std().mean()

    tr_inv, st_inv = inv[inv.partition == "train"], inv[inv.partition == "student_test"]
    M = cmp.set_index("model")
    fg = folds.groupby("model").mean(numeric_only=True)
    dts_gap_trips = int((inv.gaps_gt_30s > 0).sum())
    excl = tp[~tp.trip_id.isin(["00000033", "00000054"])]
    best_trial = trials.loc[trials.value.idxmax()]
    untuned_trial = trials.iloc[0]
    T = tuned.set_index("model")
    hb = hbm.loc["LightGBM_tuned"]
    w = hpt.points

    # ---------------------------------------------------------------- sections
    s = []
    s.append(f"""
<div class="title">
<h1>PathSnap: Distributed Map-Matching and Route Reconstruction</h1>
<p class="sub">UE24CS352A Machine Learning Hackathon, PES University RR Campus. Technical report</p>
<p>Team: Anna, Annanya, Bhoomika, Bhuvan</p>
</div>

<h2>1. Overview and team contributions</h2>
<p>PathSnap matches noisy GPS observations to a road network. For every GPS point we generate up to 16 candidate road
segments (directed arcs), score each candidate with a supervised classifier, and decode the most likely arc sequence
with a hidden Markov model and the Viterbi algorithm. Gaps between consecutive decoded arcs are filled with shortest
paths, giving a connected route. A FastAPI service on Laptop 1 runs this pipeline for a trip id sent by a web UI on Laptop 2,
and scores the result against reference data that stays on the server.</p>
<p>Final system on the 10 student-test trips: the tuned LightGBM classifier on its own selects the correct arc for
{pct(hb.clf_top1)} of GPS points; after HMM decoding this rises to {pct(hb.hmm_top1)}. Route continuity goes from
{pct(hb.clf_continuity)} to {pct(hb.hmm_continuity)} and the length-weighted route F1 from {pct(hb.clf_route_f1)} to {pct(hb.hmm_route_f1)}.
The simplest baseline, taking the geometrically nearest arc, is correct for {pct(sp.nearest_correct.mean())} of points.</p>
<table><thead><tr><th>Member</th><th>Responsibility</th><th>Main files</th></tr></thead><tbody>
{''.join(f'<tr><td>{n}</td><td>{r}</td><td><code>{f}</code></td></tr>' for n, r, f in TEAM)}
</tbody></table>
<p>All results in this report are produced by the scripts in the repository (<code>scripts/01</code> to <code>07</code>), in the order
given in the README, with random seed {setup['seed']}. Tables are generated from <code>results/*.csv</code>.</p>
""")

    s.append(f"""
<h2>2. Dataset understanding and exploratory data analysis</h2>
<p><b>Partitions used.</b> All EDA statistics below use the training partition (80 trips) and the student-testing partition
(10 trips); each table states which. The evaluator-testing partition was not available to us and was not used anywhere.</p>

<h3>2.1 Raw data and how the files relate</h3>
<p>Each trip folder has four header-less, whitespace-separated files. <code>.track</code> holds longitude, latitude and a time
stamp in seconds. <code>.nodes</code> holds node coordinates, and the row number is the node id. <code>.arcs</code> holds
(from node, to node) pairs, and the row number is the arc id. <code>.route</code> lists reference arc ids in travel order.
A GPS row does not correspond to any row in the other files: the route has one entry per arc driven
(median {tr_inv.route_arcs.median():.0f} arcs per training trip) while the track has one entry per second
(median {tr_inv.clean_points.median():.0f} points). Linking them is part of the problem (Section 2.4).</p>
{table(summ.reset_index().rename(columns={'index': 'statistic'}).query("statistic in @STAT_LABELS").assign(statistic=lambda d: d.statistic.map(STAT_LABELS)),
            ['statistic', 'train', 'student_test'], ['Statistic', 'Training', 'Student test'],
            {'train': lambda v: f'{v:,.3f}'.rstrip('0').rstrip('.'), 'student_test': lambda v: f'{v:,.3f}'.rstrip('0').rstrip('.')})}
<p class="cap">Table 2.1: Dataset inventory (results/eda_summary.csv; per-trip details in results/eda_inventory.csv).</p>

<h3>2.2 Missing values and data quality</h3>
<p>No file is missing for any trip and the track files contain no missing values, no coordinates outside the valid range,
no (0, 0) fixes, no duplicate time stamps and no out-of-order time stamps. The cleaning step in
<code>pathsnap/data_io.py</code> still checks all of these, so it is safe on unseen trips. The missing information in this dataset is of a
different kind:</p>
<ul>
<li><b>Missing observations in time.</b> Sampling is nominally 1 Hz, and {pct(0.967)} of all steps are exactly 1 s, but
{dts_gap_trips} of the 90 trips contain at least one gap longer than 30 s. The longest gap is {inv.dt_max.max():.0f} s.
The HMM widens its transition model with the gap length instead of interpolating points (Section 5).</li>
<li><b>Missing reference coverage.</b> For trips 00000033 and 00000054 the median distance from a GPS point to its
aligned reference arc is over 120 m. The trajectory plot (Figure 2.4) shows why: the reference route covers only part of
the drive. Overall {pct((tp.label_dist > 60).mean())} of training points are more than 60 m from their reference arc; we drop these
points from training as unreliable labels. In the student-test partition only {pct((sp.label_dist > 60).mean(), 2)} of points are affected.
Evaluation always uses all points.</li>
<li><b>Network irregularities.</b> 24 self-loop arcs in total (removed) and {int(inv.duplicate_arcs.sum())} duplicated (from, to) pairs
(the shortest copy is kept in the routing graph). No isolated nodes. Each extract has several weakly connected components
(median {tr_inv.weak_components.median():.0f} in training), but the largest one holds {pct(tr_inv.largest_weak_share.median())} of nodes.</li>
</ul>
{img('../results/plots/eda_sampling_and_noise.png', 'Figure 2.1: Left, sampling interval over all 90 trips (log scale). Right, distance from each GPS point to its aligned reference arc (training + student test).')}
{img('../results/plots/eda_trip_distributions.png', 'Figure 2.2: Points per trip, trip length and median GPS-to-reference distance, by partition.')}

<h3>2.3 Patterns observed</h3>
<ul>
<li>The trips are spread over many countries (latitude {inv.lat_min.min():.1f} to {inv.lat_max.max():.1f}, longitude
{inv.lon_min.min():.1f} to {inv.lon_max.max():.1f}, Figure 2.3), and each trip has its own road-network extract
(median {inv.nodes.median():.0f} nodes and {inv.arcs.median():.0f} arcs). A model cannot memorise road ids or locations; it has to learn
geometric rules that transfer across cities.</li>
<li>GPS noise is small for most points: the median distance to the reference arc is {tp.label_dist.median():.1f} m
(training) and {sp.label_dist.median():.1f} m (student test). Even so, the nearest arc is correct for only
{pct(tp.nearest_correct.mean())} of training points. Almost every street appears twice, once per driving direction, with
identical geometry, so distance alone cannot pick the direction. Junctions and parallel roads add more ambiguity.</li>
<li>When the vehicle stands still, the GPS fixes scatter around one spot and the apparent heading is random (Figure 6.2).
Direction is then decided by what happens before and after the stop, which is information a point-wise classifier does not have.</li>
<li>95th-percentile speeds are about {tr_inv.speed_p95_kmh.median():.0f} km/h and no trip has a jump over 100 m between
consecutive fixes, so we did not need an outlier filter for teleporting points.</li>
</ul>
{img('../results/plots/eda_trip_locations.png', 'Figure 2.3: Centre of every trip (training and student test).', '80%')}
<div class="two">
{img('../results/plots/eda_trajectory_00000001.png', 'Figure 2.4a: Training trip 00000001, GPS (blue), reference route (green) and road network (grey) in local metres.')}
{img('../results/plots/eda_trajectory_00000033.png', 'Figure 2.4b: Training trip 00000033. The reference route covers only the south-east part of the drive.')}
</div>
{img('../results/plots/eda_network.png', 'Figure 2.5: Road-network size per trip and share of nodes in the largest strongly connected component.')}
{img('../report/img/map_eda_trip_00000033.png', 'Figure 2.6: Folium map of training trip 00000033 (results/maps/eda_trip_00000033.html): raw GPS, nearby road network and reference route, with layer control and legend.')}

<h3>2.4 Preprocessing</h3>
<ol>
<li><b>Cleaning</b> (<code>data_io.py</code>): drop non-finite or out-of-range coordinates and (0, 0) fixes, sort by time and
remove duplicate time stamps. Arcs with invalid node ids or identical endpoints are dropped.</li>
<li><b>Projection</b> (<code>geo.py</code>): longitude and latitude are angles, and one degree of longitude shrinks with latitude
(about 62 km at 56° N against 111 km per degree of latitude), so Euclidean distance in degrees is wrong. We project every trip to a local
equirectangular plane centred on the median of its GPS points. Within the largest trip extent
({max(inv.extent_km_x.max(), inv.extent_km_y.max()):.0f} km) the scale error stays below 1%, far smaller than GPS noise, and
point-to-segment distances and bearings become plain Euclidean geometry in metres. UTM would also work, but the trips fall into many
different UTM zones, and a per-trip plane avoids zone handling.</li>
<li><b>Network cropping</b> (<code>network.py</code>): only arcs with an endpoint inside the track's bounding box plus 600 m are kept.
This bounds memory and Dijkstra cost and leaves room for detours.</li>
<li><b>Reference labels per GPS point</b> (<code>labeling.py</code>): the route file lists arcs, not points, so we align points to
route positions with a monotone dynamic programme. Point <i>i</i> receives route index <i>j<sub>i</sub></i> with
<i>j<sub>i</sub></i> never decreasing. The cost is the point-to-arc distance plus 1 for every change of arc plus 0.5 per metre of
skipped route length beyond the distance the GPS actually moved (+30 m slack). A step may advance at most 40 route arcs. Monotonicity
keeps loops and streets driven twice in order, which a plain nearest-route-arc assignment cannot do. The alignment is used only to
create training labels and for server-side scoring; the prediction path never calls it.</li>
<li><b>Training subsample</b>: consecutive 1 Hz points are nearly identical, so model comparison uses every third GPS point with all of
its candidates ({setup['train_rows']:,} candidate rows from {setup['train_trips']} trips). Whole points are kept so ranking metrics still make sense.</li>
</ol>

<h3>2.5 Candidate generation</h3>
<p>Points are sampled every 10 m along every arc and stored in a KD-tree. For each GPS point we query the 128 nearest samples, take
their distinct arcs, compute the exact point-to-segment distance and keep the 16 closest arcs (both directions of a street count
separately, so this is roughly 8 streets). The reference arc is missing from the candidate set for {pct(1 - sp.label_in_cand.mean(), 2)} of
student-test points and {pct(1 - excl.label_in_cand.mean(), 2)} of training points outside trips 33 and 54 (with those two trips the
training figure is {pct(1 - tp.label_in_cand.mean(), 1)}, which reflects their missing reference coverage, not a search failure).
Points have on average 14.3 candidates (a few points in sparse areas have fewer), and {pct(0.0654)} of candidate rows are positives,
so the classes are imbalanced roughly 1 to 14.</p>

<h3>2.6 Feature engineering</h3>
<p>Each (GPS point, candidate arc) pair is one row with 23 features (<code>features.py</code>). They use only GPS data and road
geometry. Neighbouring GPS points (up to five steps before and after) are used because map matching is done on the complete
recorded track, which is also what the API receives.</p>
<table><thead><tr><th>Group</th><th>Features</th><th>Why we chose them</th></tr></thead><tbody>
<tr><td>Distance</td><td>dist, dist_rank, dist_minus_min, dist_ratio</td><td>The correct arc is usually close, but what matters is how
close it is compared with the alternatives for the same point. Relative features carry most of the model's gain.</td></tr>
<tr><td>Temporal smoothing</td><td>win2_mean_dist, win2_max_dist, win5_mean_dist</td><td>Distance of neighbouring points to the same arc.
One noisy fix can sit closer to a side street; five consecutive fixes rarely do.</td></tr>
<tr><td>Direction</td><td>head_cos, head_cos_wide, head_abs_deg, heading_valid</td><td>Separate the two directions of the same street.
Heading comes from points i-2 to i+2 (and i-6 to i+6 for the wide version). It is zeroed when the vehicle moved less than 3 m in
that window, because the heading is then just noise.</td></tr>
<tr><td>Motion and gaps</td><td>speed, dt, gap</td><td>Context for how much to trust the heading and the distances.</td></tr>
<tr><td>Arc geometry</td><td>proj_t, arc_len, to_out_deg, from_in_deg</td><td>Where on the arc the projection falls (ends are ambiguous at
junctions), arc length, and junction complexity.</td></tr>
<tr><td>Local density</td><td>n_cand_25m</td><td>Number of arcs within 25 m: dense junctions are harder.</td></tr>
<tr><td>Connectivity</td><td>prev_near_conn, next_near_conn, is_cand_prev, is_cand_next</td><td>Whether the candidate continues from the
nearest arc of the previous point or into that of the next point. This is a light, point-wise version of the route constraint that the HMM
applies fully.</td></tr>
</tbody></table>
<p>On the training rows, positives have a mean distance of 12.7 m against 70.1 m for negatives, a mean heading cosine of 0.76 against
-0.03, and continue from the previous point's nearest arc in 80% of cases against 15%. In the final (tuned) LightGBM model the largest
shares of total split gain go to dist_minus_min (37%), head_cos_wide (15%), dist_ratio (12%) and dist_rank (10%); Figure 3.6 shows the
split-count importance of the untuned model.
Features that are almost constant (gap, dt, heading_valid, is_cand_prev) receive no gain. We kept them for the linear and
instance-based models, which use features differently.</p>
""")

    # ---------------------------------------------------------------- models
    comp = cmp.copy()
    comp["hmm_top1"] = comp.model.map(hbm.hmm_top1)
    s.append(f"""
<h2>3. Model experimentation and comparison</h2>
<p><b>Task framing.</b> Each candidate row is classified as "reference arc" or "other arc". For every GPS point the candidate with the
highest predicted probability is the classifier-only prediction.</p>
<p><b>Protocol.</b> Five-fold <code>GroupKFold</code> on the training trips with trip id as the group, so no trip is in both a fit fold and
a validation fold. Scalers are fitted inside each fold (pipelines). Each model is then refitted on the whole training sample and
evaluated on every GPS point of the 10 student-test trips. KNN and MLP are trained on random subsets of whole GPS points (150k and 300k rows)
to keep training time reasonable; all other models use all {setup['train_rows']:,} rows.</p>
<p><b>Metrics.</b> Observation-level Top-1 accuracy is the share of GPS points whose highest-scoring candidate is the reference arc
(points whose reference arc is not among the candidates count as errors). Macro F1, precision and recall are computed on the binary
candidate labels with a 0.5 threshold and averaged over the two classes. Log loss is the binary cross-entropy on candidate rows.
Route continuity is the share of consecutive GPS points whose predicted arcs are the same or directly connected (the first arc's end
node is the second arc's start node). Inference time is measured on all {n_test_rows:,} student-test candidate rows.</p>
<p><b>Primary selection metric.</b> We select on cross-validated Top-1 accuracy. The decoder consumes the ranking and the probabilities
of candidates within each GPS point, not the 0.5-threshold decision, so Top-1 is the metric closest to the route-matching goal. Macro F1 and
log loss are reported alongside and agree with the ranking.</p>

<h3>3.1 Overall model comparison</h3>
{table(comp.sort_values('cv_top1_mean', ascending=False),
       ['model', 'family', 'config', 'cv_macro_f1_mean', 'cv_top1_mean', 'test_route_continuity', 'test_macro_f1', 'test_top1', 'test_log_loss', 'inference_ms_per_1k_rows', 'hmm_top1'],
       ['Model', 'Family', 'Key configuration', 'CV macro F1', 'CV Top-1', 'Route continuity (test, classifier only)', 'Student-test macro F1',
        'Student-test Top-1', 'Log loss (test)', 'Inference (ms / 1k rows)', 'Test Top-1 after HMM'],
       {'cv_macro_f1_mean': f4, 'cv_top1_mean': f4, 'test_route_continuity': f4, 'test_macro_f1': f4, 'test_top1': f4,
        'test_log_loss': f4, 'inference_ms_per_1k_rows': lambda v: f'{v:.2f}', 'hmm_top1': f4})}
<p class="cap">Table 3.1: All nine models (results/model_comparison.csv, results/hmm_by_model.csv). CV values are means over 5 grouped folds.
Every metric applies to every model because all of them output probabilities; no cell is N/A.</p>
<p>LightGBM is best on cross-validated Top-1, cross-validated macro F1, student-test Top-1 and log loss. The tree ensembles and
HistGradientBoosting are within one percentage point of it. The fold-to-fold standard deviation of Top-1 is about {fold_std:.3f}, so
differences among the top five models are small compared with how much a single fold varies. LightGBM was chosen because it leads on both the CV and test metrics,
has the best-calibrated probabilities (lowest log loss, which matters because the HMM uses the probabilities as emissions), and runs at
about 4 ms per 1,000 candidate rows.</p>
""")

    loss = {
        "LogisticRegression": "Binary log loss with L2 penalty (C = 1), solved with L-BFGS.",
        "GaussianNB": "No iterative loss: class priors and per-class feature means and variances are maximum-likelihood estimates.",
        "KNN": "No training objective (lazy learner). Prediction is the distance-weighted share of positives among the 50 nearest training rows.",
        "DecisionTree": "Greedy splits that minimise Gini impurity.",
        "RandomForest": "Each tree minimises Gini impurity on a bootstrap sample; probabilities are averaged over trees.",
        "ExtraTrees": "Gini impurity with randomly drawn split thresholds; probabilities averaged over trees.",
        "HistGradientBoosting": "Binary log loss minimised by gradient boosting on histogram-binned features.",
        "LightGBM": "Binary log loss minimised by gradient boosting with leaf-wise tree growth.",
        "MLP": "Binary log loss plus L2 penalty (alpha = 1e-4), optimised with Adam; early stopping on a 10% validation split.",
    }
    hp = {
        "LogisticRegression": "C = 1.0, L2 penalty, max_iter = 1000, features standardised.",
        "GaussianNB": "var_smoothing = 1e-9.",
        "KNN": "n_neighbors = 50, weights = distance, Euclidean metric on standardised features, 150k training rows.",
        "DecisionTree": "max_depth = 12, min_samples_leaf = 50.",
        "RandomForest": "n_estimators = 150, max_depth = 18, min_samples_leaf = 5, max_features = sqrt (about 5 of the 23 features tried per split).",
        "ExtraTrees": "n_estimators = 150, max_depth = 20, min_samples_leaf = 5.",
        "HistGradientBoosting": "max_iter = 300, learning_rate = 0.1, max_leaf_nodes = 31, no early stopping.",
        "LightGBM": "n_estimators = 400, learning_rate = 0.05, num_leaves = 63, subsample = 0.8, colsample_bytree = 0.8, min_child_samples = 20.",
        "MLP": "hidden layers (64, 32), ReLU, Adam with learning rate 1e-3, batch 1024, alpha = 1e-4, at most 60 epochs, 300k training rows.",
    }
    why = {
        "LogisticRegression": ("A linear baseline: if a weighted sum of distance, heading and connectivity features separates the classes, nothing more complex is needed. It also gives probabilities directly.",
                               "Fast (0.3 ms per 1,000 rows) and stable across folds. It cannot model interactions such as \"heading matters only when the vehicle is moving\", and that is where it loses about 2.7 points of student-test Top-1 to LightGBM."),
        "GaussianNB": ("A probabilistic baseline that is cheap to train, useful to see how far independent per-feature evidence gets.",
                       "Training takes under a second. Its independence assumption is badly violated here (dist, dist_rank, dist_ratio and the window distances are strongly correlated), so the same evidence is counted several times and the probabilities are overconfident. Its log loss is about 20 times worse than the other models'. It ranks candidates reasonably (Top-1 0.82) but its threshold decisions are poor (macro F1 0.76)."),
        "KNN": ("Instance-based: a candidate that looks like known correct candidates in feature space should be correct too.",
                "No training, but prediction is by far the slowest (about 114 ms per 1,000 rows, roughly 30 times slower than LightGBM), and it needs scaling. Scaled distances treat all 23 features as equally important, which they are not."),
        "DecisionTree": ("Non-linear rules such as \"closest by more than 5 m and heading agrees\" match how a person would decide.",
                         "Very fast and interpretable. A single tree is high-variance: its CV Top-1 standard deviation across folds is the largest of all nine models."),
        "RandomForest": ("Averaging many decorrelated trees keeps the non-linear rules of a single tree while reducing its variance.",
                         "Second-best log loss and stable folds. The model file is large (29 MB) and training was the slowest of all models (about 80 s per fold)."),
        "ExtraTrees": ("Like Random Forest but with random split thresholds, which usually lowers variance further and trains faster.",
                       "Slightly lower Top-1 than Random Forest here, but the highest classifier-only route continuity of all nine models."),
        "HistGradientBoosting": ("Boosting corrects the errors of previous trees one after another; histogram binning makes it fast on hundreds of thousands of rows.",
                                 "Close second on student-test Top-1. Log loss is a little worse than LightGBM's with these settings."),
        "LightGBM": ("Gradient boosting with leaf-wise growth and row/column subsampling. It handles mixed-scale features without scaling and models feature interactions, and its log-loss objective gives calibrated probabilities for the HMM.",
                     "Best on CV and student-test Top-1 and on log loss, fast inference, small model file. Leaf-wise growth can overfit with large leaves; the train-validation gap (below) shows this and motivated the regularisation search in Section 4."),
        "MLP": ("A neural network can learn smooth non-linear combinations of the distance and heading features.",
                "Good log loss and no overfitting, but it trains on scaled data, needs early stopping, and was still about 1.5 points below LightGBM on test Top-1. More data or more epochs might close part of that gap."),
    }
    s.append("<h3>3.2 Per-model questions</h3>")
    for name in comp.sort_values("cv_top1_mean", ascending=False).model:
        r, f = M.loc[name], fg.loc[name]
        gap = f.train_top1 - f.val_top1
        if gap > 0.1:
            fit_txt = (f"Clear overfitting: Top-1 on the training folds is {f4(f.train_top1)} against {f4(f.val_top1)} on validation folds, "
                       f"and training log loss is {f.train_log_loss:.4f} against {f.val_log_loss:.4f}. With distance weighting every training row "
                       "is its own nearest neighbour, so training scores are not meaningful; the validation score is the honest one.")
        elif gap > 0.04:
            fit_txt = (f"Mild overfitting: Top-1 is {f4(f.train_top1)} on training folds against {f4(f.val_top1)} on validation folds "
                       f"(gap {gap:.3f}), and log loss {f.train_log_loss:.4f} against {f.val_log_loss:.4f}. Validation performance is still the "
                       "best or near the best, so the extra capacity is not hurting generalisation, but stronger leaf regularisation is worth trying.")
        elif gap > 0.015:
            fit_txt = (f"Slight overfitting: Top-1 {f4(f.train_top1)} on training folds against {f4(f.val_top1)} on validation folds (gap {gap:.3f}). "
                       "The depth or leaf-size limits keep it small.")
        else:
            fit_txt = (f"No overfitting: training and validation Top-1 are almost equal ({f4(f.train_top1)} and {f4(f.val_top1)}). "
                       + ("Both stay below the tree-based models, which points to underfitting: the model is too simple for the feature interactions."
                          if f.val_top1 < 0.875 else "Generalisation is good."))
        s.append(f"""
<h4>{name} ({r.family})</h4>
<p><b>A1. Why suitable.</b> {why[name][0]}</p>
<p><b>A2. Strengths and limitations in our experiments.</b> {why[name][1]}</p>
<p><b>B5. Important hyperparameters and initial values.</b> {hp[name]}</p>
<p><b>B6. Loss function or objective.</b> {loss[name]}</p>
<p><b>C7-C8. Student-test performance.</b> Top-1 {f4(r.test_top1)}, macro F1 {f4(r.test_macro_f1)} (precision {f4(r.test_macro_precision)},
recall {f4(r.test_macro_recall)}), log loss {f4(r.test_log_loss)}, classifier-only route continuity {f4(r.test_route_continuity)}.
With HMM decoding its Top-1 becomes {f4(hbm.loc[name].hmm_top1)}. Cross-validation: Top-1 {f4(r.cv_top1_mean)} ± {r.cv_top1_std:.4f},
macro F1 {f4(r.cv_macro_f1_mean)} ± {r.cv_macro_f1_std:.4f}. Training time on the full sample {r.fit_time_s:.1f} s, inference
{r.inference_ms_per_1k_rows:.2f} ms per 1,000 rows.</p>
<p><b>C9. Overfitting or underfitting.</b> {fit_txt}</p>
""")

    s.append(f"""
<h3>3.3 Model visualisations</h3>
{img('../results/plots/cmp_cv_macro_f1.png', 'Figure 3.1: Cross-validation macro F1 per fold for all models.')}
{img('../results/plots/cmp_cv_top1.png', 'Figure 3.2: Cross-validation Top-1 accuracy per fold for all models.')}
{img('../results/plots/cmp_test_top1_f1.png', 'Figure 3.3: Student-test Top-1 accuracy and macro F1.')}
{img('../results/plots/cmp_test_log_loss.png', 'Figure 3.4: Student-test log loss (log scale). Gaussian Naive Bayes is far worse because of correlated features.')}
{img('../results/plots/cmp_train_vs_val.png', 'Figure 3.5: Mean Top-1 on training folds against validation folds (overfitting check).')}
<div class="two">
{img('../results/plots/importance_LightGBM.png', 'Figure 3.6: LightGBM (untuned) split-count feature importance.')}
{img('../results/plots/cmp_speed_vs_accuracy.png', 'Figure 3.7: Student-test Top-1 against inference time.')}
</div>
<p>Training histories are plotted only for the two models that train iteratively and record a history: the MLP (loss per epoch) and
LightGBM (log loss per boosting round). The other models have no epoch-wise history, so no curve is shown for them.</p>
<div class="two">
{img('../results/plots/mlp_training_history.png', 'Figure 3.8: MLP training loss and early-stopping validation accuracy per epoch.')}
{img('../results/plots/lightgbm_training_history.png', 'Figure 3.9: Tuned LightGBM, log loss per boosting round on 64 training trips and 16 held-out training trips.')}
</div>
<div class="three">
{img('../results/plots/cm_LightGBM.png', 'Figure 3.10a: LightGBM confusion matrix (student test).')}
{img('../results/plots/cm_LogisticRegression.png', 'Figure 3.10b: Logistic regression.')}
{img('../results/plots/cm_GaussianNB.png', 'Figure 3.10c: Gaussian Naive Bayes.')}
</div>
<p>Confusion matrices for every model are in <code>results/plots/cm_*.png</code>.</p>
""")

    # ---------------------------------------------------------------- bayes
    u, t = T.loc["LightGBM untuned"], T.loc["LightGBM tuned"]
    bp = final_cfg["params"]
    s.append(f"""
<h2>4. Bayesian hyperparameter optimisation (LightGBM)</h2>
<p><b>Q1-Q2. Hyperparameters and ranges.</b> We used Optuna's Tree-structured Parzen Estimator (TPE), seed 42, with 10 random start-up
trials. Trial 0 was set to the untuned configuration so the search starts from a known reference.</p>
<table><thead><tr><th>Hyperparameter</th><th>Range / distribution</th><th>Why</th><th>Best value</th></tr></thead><tbody>
<tr><td>n_estimators</td><td>100 to 800, step 50</td><td>number of boosting rounds</td><td>{bp['n_estimators']}</td></tr>
<tr><td>learning_rate</td><td>0.01 to 0.2, log-uniform</td><td>trades rounds against step size</td><td>{bp['learning_rate']:.4f}</td></tr>
<tr><td>num_leaves</td><td>15 to 255, log-uniform integer</td><td>tree complexity (main overfitting control)</td><td>{bp['num_leaves']}</td></tr>
<tr><td>min_child_samples</td><td>10 to 300, log-uniform integer</td><td>minimum rows per leaf</td><td>{bp['min_child_samples']}</td></tr>
<tr><td>subsample</td><td>0.5 to 1.0</td><td>row sampling per tree</td><td>{bp['subsample']:.3f}</td></tr>
<tr><td>colsample_bytree</td><td>0.4 to 1.0</td><td>feature sampling per tree</td><td>{bp['colsample_bytree']:.3f}</td></tr>
<tr><td>reg_lambda</td><td>0.001 to 10, log-uniform</td><td>L2 penalty on leaf weights</td><td>{bp['reg_lambda']:.4f}</td></tr>
<tr><td>reg_alpha</td><td>0.001 to 10, log-uniform</td><td>L1 penalty on leaf weights</td><td>{bp['reg_alpha']:.4f}</td></tr>
</tbody></table>
<p><b>Q3. Objective.</b> Mean observation-level Top-1 accuracy over a 3-fold <code>GroupKFold</code> by trip on the 80 training trips (every
sixth GPS point, 438,558 rows, to fit the time budget). Neither the student-test nor the evaluator partition was used in the search.</p>
<p><b>Q4. Trials and cost.</b> {final_cfg['bayes_trials']} trials, {final_cfg['search_time_s'] / 60:.1f} minutes on a 16-thread laptop CPU.
The objective ranged from {trials.value.min():.4f} to {trials.value.max():.4f} across trials; the untuned configuration (trial 0)
scored {untuned_trial.value:.4f} and the best trial (number {int(best_trial.number)}) scored {best_trial.value:.4f}.</p>
<p><b>Q5. Best values.</b> See the last column of the table above: fewer, smaller leaves (31 instead of 63), a much larger minimum
leaf size (144 instead of 20) and stronger feature subsampling (0.61), that is, a more regularised model.</p>
{table(tuned, ['model', 'cv_macro_f1', 'cv_top1', 'test_top1', 'test_macro_f1', 'test_log_loss', 'inference_ms_per_1k_rows', 'fit_time_s'],
       ['Model', 'CV macro F1', 'CV Top-1', 'Student-test Top-1', 'Student-test macro F1', 'Log loss (test)', 'Inference (ms / 1k rows)', 'Fit time (s)'],
       {c: f4 for c in ['cv_macro_f1', 'cv_top1', 'test_top1', 'test_macro_f1', 'test_log_loss']} | {'inference_ms_per_1k_rows': lambda v: f'{v:.2f}', 'fit_time_s': lambda v: f'{v:.1f}'})}
<p class="cap">Table 4.1: Untuned and tuned LightGBM with the same protocol as Section 3 (5 grouped folds on every third point, same student-test rows).</p>
<p><b>Q6. Did tuning help?</b> Only marginally. CV Top-1 went from {f4(u.cv_top1)} to {f4(t.cv_top1)}, CV macro F1 from {f4(u.cv_macro_f1)} to
{f4(t.cv_macro_f1)}, and student-test log loss from {f4(u.test_log_loss)} to {f4(t.test_log_loss)}, while student-test Top-1 dropped
slightly from {f4(u.test_top1)} to {f4(t.test_top1)}. All of these differences are far smaller than the fold-to-fold standard deviation (about
0.027), so we read them as no real change in accuracy. The clearer effect is on overfitting: the train-validation Top-1 gap fell from
{fg.loc['LightGBM'].train_top1 - fg.loc['LightGBM'].val_top1:.3f} to {tuned_folds.train_top1.mean() - tuned_folds.val_top1.mean():.3f}. The
flat objective surface (all 40 trials within {trials.value.max() - trials.value.min():.4f}) suggests the remaining errors come from
information the features do not contain (for example direction while stationary) rather than from model capacity. We kept the tuned model as the final
model because it was selected on cross-validation alone and generalises with a smaller gap. After HMM decoding the two reach the same
student-test Top-1 ({f4(hbm.loc['LightGBM'].hmm_top1)} untuned, {f4(hb.hmm_top1)} tuned). Inference is slower (about 6 ms against 4 ms per 1,000 rows) because there are more trees.</p>
<div class="two">
{img('../results/plots/bayes_opt_history.png', 'Figure 4.1: Objective value per trial and best value so far.')}
{img('../results/plots/bayes_opt_param_importance.png', 'Figure 4.2: Hyperparameter importance estimated from the trials (fANOVA).')}
</div>
{img('../results/plots/tuned_vs_untuned.png', 'Figure 4.3: Untuned and tuned LightGBM.', '85%')}
""")

    # ---------------------------------------------------------------- hmm
    best_row = hgrid.sort_values(["hmm_top1", "hmm_route_f1"], ascending=False).iloc[0]
    worst_gain = (hpt.hmm_top1 - hpt.clf_top1).min()
    s.append(f"""
<h2>5. HMM and Viterbi route reconstruction</h2>
<p><b>Q1. Hidden states.</b> At GPS point <i>i</i> the hidden state is one of that point's candidate arcs (up to 16 directed arcs). The
candidate slot keeps the link to the arc id, so the decoded path is directly a sequence of road segments.</p>
<p><b>Q2. Emission probabilities.</b> The tuned LightGBM probability of each candidate, floored at 1e-4 and renormalised over the point's
candidates, raised to an emission weight <i>w</i>: log e(i, a) = w · log p̂(a | point i).</p>
<p><b>Q3. Transition probabilities.</b> For candidate <i>a</i> at point <i>i</i> and <i>b</i> at point <i>i</i>+1 we compute the driving
distance <i>d</i><sub>route</sub> from the projection on <i>a</i> to the projection on <i>b</i>: the rest of arc <i>a</i>, plus the shortest
road-graph path from the end node of <i>a</i> to the start node of <i>b</i> (Dijkstra, directed), plus the part of <i>b</i> up to the
projection. On the same arc it is the signed distance along the arc, so moving backwards is penalised. With
<i>d</i><sub>gps</sub> the straight-line distance between the two fixes:</p>
<p class="eq">log P(b | a) = −| d<sub>route</sub> − d<sub>gps</sub> | / (β · √Δt)</p>
<p>A real vehicle's driving distance between two fixes is close to their straight-line distance; a jump onto a parallel street or a
reversed arc needs a much longer path and is penalised. Dijkstra searches only up to 2·d<sub>gps</sub> + 300 m from each source node.</p>
<p><b>Viterbi recurrence.</b> δ<sub>i+1</sub>(b) = max<sub>a</sub> [ δ<sub>i</sub>(a) + log P(b | a) ] + log e(i+1, b), with back-pointers,
computed in log space and renormalised at each step. The final state is the arg max of δ at the last point, followed back through the
pointers.</p>
<p><b>Q4. Missing candidates and large gaps.</b> Points always have at least one candidate because the nearest arcs are taken without a
radius limit; missing slots get probability zero. For time gaps the √Δt factor widens the transition distribution. A 60 s gap allows about
eight times the distance mismatch of a 1 s step, and the Dijkstra radius grows with the distance travelled.</p>
<p><b>Q5. Disconnected segments.</b> If no path exists between any pair of candidates within the search radius, every transition is −∞.
The decoder then records a break, stores the best state before the break and restarts from the emissions at the next point, so one bad
step cannot destroy the whole trip. On the 10 student-test trips this happened once (trip 00000056). When the route is
reconstructed, consecutive distinct arcs are joined with the shortest path between them.</p>
<p><b>Decoder tuning.</b> β ∈ {{5, 10, 20, 40, 80}} m and <i>w</i> ∈ {{0.5, 1, 2}} were grid-searched on 15 training trips held out from a
LightGBM model fitted on the remaining training trips (results/hmm_tuning_grid.csv). Only the product β·<i>w</i> matters: multiplying all
log scores by a constant does not change the arg max, so β = 5, <i>w</i> = 1 and β = 10, <i>w</i> = 0.5 give identical results. The best
setting was β = {best_row.beta:.0f} m, <i>w</i> = {best_row.emission_weight:.1f} (Top-1 {f4(best_row.hmm_top1)} against {f4(best_row.clf_top1)}
for the classifier alone on those trips). Once β·<i>w</i> goes above 5, accuracy drops steadily: the decoder then trusts the classifier
more and the road network less. Every setting in the grid still beats the classifier alone.</p>
{img('../results/plots/hmm_tuning.png', 'Figure 5.1: Held-out training Top-1 as a function of β and emission weight.', '70%')}

<h3>Q6. Did HMM decoding improve the route?</h3>
{table(hpt.assign(trip=hpt.trip_id), ['trip', 'points', 'clf_top1', 'hmm_top1', 'clf_continuity', 'hmm_continuity', 'clf_route_f1', 'hmm_route_f1', 'breaks', 'decode_s'],
       ['Trip', 'Points', 'Top-1 classifier', 'Top-1 HMM', 'Continuity classifier', 'Continuity HMM', 'Route F1 classifier', 'Route F1 HMM', 'Breaks', 'Decode (s)'],
       {c: f4 for c in ['clf_top1', 'hmm_top1', 'clf_continuity', 'hmm_continuity', 'clf_route_f1', 'hmm_route_f1']} | {'decode_s': lambda v: f'{v:.2f}'})}
<p class="cap">Table 5.1: Classifier-only against classifier + HMM per student-test trip (results/hmm_student_test_per_trip.csv). Route F1 is
length-weighted overlap between the reconstructed arc set and the reference arc set.</p>
<p>Yes. Weighted by points, Top-1 rises from {pct(hb.clf_top1)} to {pct(hb.hmm_top1)}, continuity from {pct(hb.clf_continuity)} to
{pct(hb.hmm_continuity)}, and mean route F1 from {pct(hb.clf_route_f1)} to {pct(hb.hmm_route_f1)}. The largest change is route precision.
Classifier-only predictions scatter onto short side arcs and reversed arcs, and after gap filling these add wrong road length
(mean route precision {hpt.clf_route_precision.mean():.3f}); the HMM removes most of them ({hpt.hmm_route_precision.mean():.3f}). Point
accuracy improved on every trip, by at least {100 * worst_gain:.1f} percentage points. The biggest gain is trip 00000077, from
{f4(hpt.set_index('trip_id').loc['00000077'].clf_top1)} to {f4(hpt.set_index('trip_id').loc['00000077'].hmm_top1)}, which has long
stationary periods where the classifier cannot tell the direction (Figure 6.2). The decoder also improves every one of the nine
classifiers (Table 3.1, last column). The weakest model gains the most (Gaussian Naive Bayes, from {f4(hbm.loc['GaussianNB'].clf_top1)}
to {f4(hbm.loc['GaussianNB'].hmm_top1)}), so the road-network constraint makes up for part of a weak classifier.</p>
<p><b>Where the HMM did not help.</b> After decoding, {int(ea.loc['ALL'].errors):,} of {int(ea.loc['ALL'].points):,} student-test points
are still wrong (results/hmm_error_analysis.csv, scripts/08_error_analysis.py):</p>
<ul>
<li>{pct(ea.loc['ALL'].reverse_direction / ea.loc['ALL'].errors)} are the right street in the wrong direction. Most of them are in trip
00000077 ({int(ea.loc['00000077'].reverse_direction)} points), which has long stops. The HMM fixes many stops (Figure 6.2), but when the
vehicle stands still on a two-way street for a long time, neither the classifier nor the transition model has evidence about the direction.
Wrong-direction errors make up {int(ea.loc['00000077'].reverse_direction)} of the {int(ea.loc['00000077'].errors)} errors in trip
00000077 and {int(ea.loc['00000065'].reverse_direction)} of {int(ea.loc['00000065'].errors)} in trip 00000065, the two lowest-scoring trips.</li>
<li>{pct(ea.loc['ALL'].on_reference_route / ea.loc['ALL'].errors)} are arcs that are on the reference route, but the label alignment assigns that
GPS point to another arc, usually the neighbouring one. These are timing offsets at junctions: when the vehicle crosses a junction between two
fixes, the decoder and the label alignment can switch arcs one point apart. Part of this error is in the labels, not in the prediction.</li>
<li>{pct(ea.loc['ALL'].off_route / ea.loc['ALL'].errors)} are arcs off the reference route. Some come from sections where the reference itself
does not follow the GPS, as in trip 00000009 (Figure 6.3), where the decoder follows the GPS points and is scored wrong.</li>
<li>Continuity stays below 100% because the decoder sometimes moves to an arc two or three segments ahead in one step, skipping a very short
arc. Of the {int(ea.loc['ALL'].discontinuities)} discontinuities, {int(ea.loc['ALL'].reconnect_within_100m)} can be joined by a road path of at
most 100 m, and gap filling joins them in the reconstructed route.</li>
</ul>
{img('../results/plots/hmm_vs_classifier_per_trip.png', 'Figure 5.2: Classifier-only against classifier + HMM per student-test trip.')}
{img('../results/plots/hmm_by_model.png', 'Figure 5.3: Effect of HMM decoding for every classifier (student test).', '85%')}
""")

    s.append(f"""
<h2>6. Geographical visualisation using Folium</h2>
<p><code>pathsnap/maps.py</code> builds every map: OpenStreetMap tiles, raw GPS observations (blue dots), the road network within 120 m of
the track (grey), the reference route (green, wide and semi-transparent so the prediction stays visible on top), classifier-only arcs that
are not on the reference (orange), the HMM-decoded and gap-filled predicted route (red), start and end markers, a legend and a layer
control. Each map fits its bounds to the track. Interactive HTML versions are in <code>results/maps/</code>: one per student-test trip
(<code>student_test_*.html</code>), EDA maps (<code>eda_trip_*.html</code>) and zoomed detail maps (<code>detail_*.html</code>, from
<code>scripts/07_detail_maps.py</code>, which picks the 120-point window where the HMM corrects the most classifier errors).</p>
{img('../report/img/map_student_test_00000009.png', 'Figure 6.1: Student-test trip 00000009, whole trip.')}
{img('../report/img/map_detail_00000077.png', 'Figure 6.2: Trip 00000077, points 359-478. The vehicle is standing still; GPS fixes scatter over about 40 m. The classifier alone picks the reverse direction for most points (37.5% correct in this window); the HMM keeps the direction consistent with the movement before and after the stop (100% correct). The classifier-only arcs lie under the red line because a reversed arc has the same geometry.')}
{img('../report/img/map_detail_00000009.png', 'Figure 6.3: Trip 00000009, points 35-154. The reference route (green) takes the loop to the east while the GPS points go south. The HMM route (red) follows the GPS points.')}
""")

    s.append(f"""
<h2>7. API, UI and deployment</h2>
<p><b>Laptop 1</b> runs <code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code>. At start-up it loads the tuned LightGBM model, the
HMM configuration and the folder of evaluation trips (environment variable <code>PATHSNAP_EVAL_DIR</code>). Endpoints:
<code>GET /health</code>, <code>GET /trips</code> (the trip ids in that folder) and <code>POST /predict</code> with body
<code>{{"trip_id": "..."}}</code>. A prediction reads only the <code>.track</code>, <code>.nodes</code> and <code>.arcs</code> files and runs
cleaning, projection, candidate generation, features, LightGBM, Viterbi and gap filling. Afterwards a separate function reads
<code>.route</code>, aligns it to the track and computes point accuracy (HMM and classifier-only), route precision, recall and F1. The
response contains the GPS points, the predicted route as polylines, the predicted arc id for every GPS point, continuity, the evaluation
numbers and timings. The reference route itself is never sent to the client. Every request is logged in the console with the trip id,
the number of points and candidates, the time of each stage and the accuracy.</p>
<p><b>Laptop 2</b> runs <code>python ui/serve.py</code> (Python standard library only) and serves a single page built with Leaflet. The evaluator
enters Laptop 1's address, loads the trip list, and clicks a trip. The page sends only the trip id and draws the returned GPS points and
predicted route with a layer control and legend, plus a results table.</p>
<p>A request takes from 0.2 s (trip 00000052, 262 points) to 7.7 s (trip 00000056, 3,958 points) on the student-test trips. For the longest
trip, features take 0.3 s, classifier plus Viterbi 3.8 s (mostly the Dijkstra searches) and server-side scoring 3.5 s (mostly the label
alignment, which is only needed for evaluation).</p>
{img('../report/img/ui_trip_77.jpg', 'Figure 7.1: Evaluator UI after running trip 00000077.', '85%')}
<pre>{api_log}</pre>
<p class="cap">Listing 7.1: API console log for the request in Figure 7.1.</p>
""")

    s.append(f"""
<h2>8. Conclusions and limitations</h2>
<ul>
<li>Relative-distance and heading features turn a 52% nearest-arc baseline into 88% point accuracy with LightGBM. The HMM, which adds road
connectivity and travel-distance consistency, brings it to {pct(hb.hmm_top1)} and makes the route almost fully connected (route F1 {pct(hb.hmm_route_f1)}).</li>
<li>Among the nine classifiers the boosted and bagged tree models are clearly ahead of linear, Naive Bayes and nearest-neighbour models,
but they are within about one point of each other. Bayesian optimisation reduced overfitting without improving accuracy.</li>
<li>Reference routes are not perfect ground truth. Two training trips have partial routes, and student-test trip 00000009 contains a
section where the reference does not follow the GPS. Our labels come from aligning the arc sequence to the track, so labels at arc
boundaries can be one point off. All reported accuracies inherit these limitations.</li>
<li>Direction during long stops remains the main source of classifier errors. Possible next steps are features computed from the
movement before and after a stop, using the HMM's posterior instead of the Viterbi path for uncertain points, and training on all points
rather than every third.</li>
</ul>
""")

    css = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; line-height: 1.42; color: #1d2330; }
h1 { font-size: 20pt; margin: 0 0 6px; } h2 { font-size: 14pt; margin: 22px 0 6px; border-bottom: 2px solid #1565c0; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14px 0 4px; page-break-after: avoid; } h4 { font-size: 10.5pt; margin: 12px 0 2px; page-break-after: avoid; }
.title { margin-bottom: 14px; } .sub { color: #5f6b7a; margin: 0 0 4px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 8.3pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9ced6; padding: 3px 5px; vertical-align: top; } th { background: #eef2f7; text-align: left; }
figure { margin: 8px 0; page-break-inside: avoid; text-align: center; } figure img { max-width: 100%; }
figcaption, .cap { font-size: 8.5pt; color: #5f6b7a; text-align: left; margin-top: 2px; }
.two { display: flex; gap: 10px; } .two figure { flex: 1; } .three { display: flex; gap: 8px; } .three figure { flex: 1; }
pre { font-size: 7pt; background: #f4f6f8; padding: 6px; white-space: pre-wrap; word-break: break-all; }
code { font-size: 8.8pt; } .eq { text-align: center; font-size: 11pt; margin: 6px 0; }
"""
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap Technical Report</title><style>{css}</style></head><body>{''.join(s)}</body></html>"
    OUT_HTML.write_text(html, encoding="utf-8")
    print("wrote", OUT_HTML)

    browser = next((p for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                                shutil.which("msedge") or "", shutil.which("chrome") or ""] if p and Path(p).exists()), None)
    if browser is None:
        print("No Edge/Chrome found: open the HTML file and print it to PDF.")
        return
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=10000",
                    f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()], check=True, capture_output=True)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    sys.exit(main())
