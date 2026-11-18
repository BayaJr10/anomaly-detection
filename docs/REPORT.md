# Final Report

## An Explainable, Ensemble-Based ML System for Detecting Anomalous Air Pollution Events in Nairobi's Low-Cost Sensor Network

**Baya Isaac** (222327) · MSc Data Science & Analytics · DSA 8401 Applied Machine Learning · Strathmore University
Lecturers: Prof. Javier Serrano & Emmanuel Bundi

> All numbers in this report come from `reports/metrics.json` (written by `python -m airq.pipeline train`) and from the executed `notebooks/02_anomaly_detection.ipynb`. Section 18 of the notebook asserts that the two produce identical features and probabilities.

---

## 1. Problem framing

Fine particulate matter (PM2.5) is linked to asthma, heart disease and stroke. Reference-grade monitors are too costly for most African cities, so Nairobi's long-term record comes mainly from low-cost optical sensors in the sensors.AFRICA network. These sensors drift, over-read in high humidity, and fail in ways that look like pollution: a stuck channel, a firmware sentinel, a dead fan. Analysts currently review readings by hand or with fixed thresholds. That does not scale, and it gives officials flags with no explanation.

**Task.** For every hourly sensor reading, decide whether it is anomalous *for that place and time*. For each anomaly, say how severe it is, whether it is more likely a genuine pollution event or a sensor malfunction, why the model flagged it, and who should act on it.

**Why it is hard.**

* There are **no ground-truth labels**, so both training and evaluation must be built and validated indirectly.
* Anomalies are **rare**: about 0.7% of training readings under the consensus definition.
* The data are **messy**:
  * impossible values (humidity −85% to 65,535, a firmware sentinel);
  * structural missingness (SDS011 sensors have no PM1 channel);
  * months with no data;
  * strong sensor attrition after 2022 (47 sensors in training, 7 in the test period).

**Relevance.** The system supports SDG 3 (health) and SDG 11 (sustainable cities) and the Kenya National AI Strategy 2025–2030. It routes genuine events to environment and public-health officers and probable faults to the network's maintenance team.

## 2. Data

### 2.1 Source and pipeline

The source is the sensors.AFRICA air-quality archive for Nairobi on openAFRICA. It holds one CSV per month of raw readings in long format (`sensor_id; sensor_type; location; lat; lon; timestamp; value_type; value`).

The original data-pipeline notebook was lost, so it was rebuilt as a scripted, tested module (`src/airq/data/`, `notebooks/01_data_pipeline.ipynb`):

1. Download or locate the monthly files.
2. Keep the five value types (P0→PM1, P2→PM2.5, P1→PM10, humidity, temperature) as float32, drop exact duplicate rows, and floor timestamps to the hour.
3. Average to hourly values per *location* (a site that may host a DHT22 and an SDS011/PMS5003) using running sums and counts, so files are processed one at a time.
4. Record the sorted list of sensor types that reported in each hour (`SENSOR_TYPE`).
5. Take each location's most frequent coordinates.
6. Reverse-geocode to a neighbourhood with OpenStreetMap Nominatim, through a frozen cache so names are reproducible.
7. Add UTC calendar columns.

**Verification.** Two archive-format sample months were regenerated from the published file and pushed through the pipeline. Every row was reproduced, in the same order, with identical sensor-type strings, neighbourhood names and values (to float32 precision). The float32 artefacts of the published file, such as `99.900002`, also reappear, which confirms the original parsed values as single precision.

### 2.2 Dataset

292,059 hourly rows · 53 sites · 27 neighbourhoods · January 2019 – June 2026 · 16 columns.

| Quality issue (raw hourly data) | Count | Handling (notebook 02, Section 5) |
|---|---|---|
| Missing PM2.5 / PM10 (DHT22-only hours) | 9,901 rows | dropped (target unavailable) |
| Humidity outside 0–100 % (incl. 65,535 sentinel) | 336 | set to NaN, then imputed |
| Temperature outside −10 to 45 °C | 617 | set to NaN, then imputed |
| PM above the sensors' measuring range (> 1000 µg/m³) | 439 (410 from sensor 3626) | **set aside as certain faults**; labelled by rule at serving time |
| PM ordering violations (PM10 < PM2.5 or PM2.5 < PM1) | 444 | PM set to NaN → row dropped |
| Site without coordinates | 26 | dropped |
| PM1 missing (structural, SDS011) | 42% of kept rows | kept as NaN + indicator |
| Months with no readings at all | 4 (local time) | treated as no evidence, never interpolated |

**A finding about time.** The archive and the published file are in **UTC**. Median PM2.5 peaks at 04:00 and 16:00–17:00 in the stored hour, which are 07:00 and 19:00–20:00 in Nairobi (EAT, UTC+3): the morning rush and the evening traffic and cooking peak. All calendar features, the baseline and the split boundaries therefore use local time.

## 3. Method

### 3.1 Pipeline

```
raw archive ─► hourly dataset ─► clean (UTC→local, bounds, range faults aside, indicators,
short-gap interpolation, train-only conditional imputation) ─► past-only rolling and neighbourhood
features ─► temporal split ─► seasonal baseline + residual ─► 5 detectors ─► consensus label
─► feature pruning ─► SMOTE + XGBoost meta-model ─► threshold on validation ─► SHAP, severity,
cause ─► API
```

| Split | Period (local) | Rows | Sensors | Pseudo-positive rate |
|---|---|---|---|---|
| Train | 2019-01 → 2023-12 | 245,741 | 47 | 0.68% |
| Validation | 2024 | 21,128 | 12 | 1.09% |
| Test (locked) | 2025-01 → 2026-06 | 14,824 | 7 | 0.55% |

### 3.2 Key components

* **Seasonal baseline.** The median PM2.5 by neighbourhood × local hour × weekday, fitted on training data, with fallbacks to neighbourhood × hour and then the global median. The residual (`PM2.5 − baseline`) turns "high" into "unusual for this place and time".
* **Five detectors, five notions of normal** (all fitted on the full training period):
  * **Statistical:** MAD robust z of the residual, |z| > 3.5.
  * **Density:** Local Outlier Factor, novelty mode.
  * **Kernel:** Nystroem-RBF + SGD One-Class SVM.
  * **Clustering:** an exact DBSCAN noise rule via KD-trees.
  * **Time series:** per-sensor STL residual on unbroken segments, with training MAD thresholds.

  The consensus pseudo-label is set when at least 3 of the 5 flag a reading.
* **Meta-model.** Candidate features exclude the detector flags (no label leakage) and are pruned on training data only, by exact redundancy, |r| > 0.95, and < 1% gain. That leaves 9: residual, PM2.5, PM10, humidity, temperature, PM2.5/PM10 ratio, deviation from neighbourhood mean, 72-reading past mean, and the PM1-missing indicator. The model is a median-impute → RobustScaler → SMOTE → XGBoost pipeline (500 trees, depth 6), with SMOTE applied only to training folds. The threshold is the max-F1 point on validation (0.970) and is then frozen.
* **Explainability.** TreeSHAP for every flag. The API turns the top-3 drivers into a sentence, e.g. *"PM2.5 is +530 µg/m³ from the usual 29 µg/m³ for Nairobi City at 12:00 on a Wednesday"*.
* **Severity.** Critical when robust z ≥ 10 or PM2.5 ≥ 125.5 µg/m³; Alert when z ≥ 6 or PM2.5 ≥ 55.5 µg/m³ (US EPA 2024 breakpoints); Watch otherwise.
* **Likely cause.** Mean residual of the *other* sensors in the same neighbourhood and hour.
  * **genuine:** the neighbours share at least half of the reading's excess.
  * **likely_sensor_fault:** neighbours reported, but do not share the excess.
  * **unverifiable:** no neighbour reported. For these, a random forest trained on neighbour-free features gives an advisory guess.
  * **Humidity flag:** readings at ≥ 85% relative humidity with a positive residual get a `humidity_artefact_risk` flag.

## 4. Technique justification

### 4.1 Why these techniques

| Technique | Reason |
|---|---|
| Baseline residual | "Anomalous" is relative to place and time; a fixed threshold flags every rush hour. A median lookup table is robust, transparent and free to serve. |
| Five detectors + consensus | No labels exist, and each paradigm has blind spots. Agreement across paradigms is the most defensible unsupervised definition of "anomalous". |
| Scalable detector variants | Exact One-Class SVM and DBSCAN are quadratic. The Nystroem/SGD and KD-tree variants fit all 245k training rows, so nothing depends on a subsample. |
| Stacked XGBoost | Collapses five detectors (STL needs whole series; DBSCAN needs the training cloud) into one scorer that needs only a reading and its recent history. It handles missing values natively and supports exact TreeSHAP. |
| SMOTE inside the pipeline | 0.7% positives. Resampling only training folds prevents synthetic leakage. Its distortion of probabilities is handled by tuning the threshold on validation and not using probability for severity. |
| Temporal split + StratifiedGroupKFold | Monitoring is always applied to the future. Grouping by sensor stops CV memorising devices; stratification keeps the proposal's stratified k-fold. |
| Neighbour evidence for cause | Physically meaningful (real plumes reach neighbours) and needs no labels. |
| Corroboration, flag concentration, Moran's I, documented events | Evidence that does not come from the detectors themselves, which limits the circularity of pseudo-label metrics. |

### 4.2 Why not the alternatives

| Alternative | Reason for exclusion |
|---|---|
| Isolation Forest | Overlaps heavily with LOF and with the tree meta-model. |
| Autoencoders, LSTM and Transformer forecasters | Short, gappy, attrition-hit series and no labels to tune their thresholds; harder to explain to officials. Listed as future work. |
| Supervised training on real labels | No labels exist. |
| Fixed thresholds | Used only where the answer is certain (sensor range) and as health bands. |
| Global median imputation of PM1 | Its missingness is structural. |
| Random splits | Leak the future through autocorrelation and repeated sensors. |
| Probability-based severity | The SMOTE-trained probability saturates (every flag has p > 0.97). |
| Streaming infrastructure | The archive is published monthly, so a batch job plus an API is proportionate. |

## 5. Results

### 5.1 Meta-model against the consensus label

| | CV on train (5 folds) | Validation 2024 | Test 2025–26 |
|---|---|---|---|
| ROC-AUC | 0.991 ± 0.008 | 0.994 | 0.959 |
| PR-AUC | 0.821 ± 0.098 | 0.852 | 0.448 |
| Precision / Recall | | 0.841 / 0.783 | 0.600 / 0.366 |
| F1 | 0.668 ± 0.152 | 0.811 | 0.455 |
| Confusion [[TN, FP], [FN, TP]] | | [[20864, 34], [50, 180]] | [[14722, 20], [52, 30]] |

Validation numbers use the threshold chosen on validation, so they are optimistic. The test set was scored once with the frozen threshold.

### 5.2 Effect of the corrections

| | Previous notebook run | This run |
|---|---|---|
| Validation PR-AUC | 0.975 | 0.852 |
| Test PR-AUC | 0.415 | 0.448 |
| Test F1 | 0.377 | 0.455 |
| Severity among validation flags | 579 Critical, 2 Alert, 0 Watch | 143 Critical, 71 Alert |

The earlier validation score was inflated by one sensor: 410 hours from sensor 3626 above the measuring limit made up 368 of the 393 "fault" flags. Setting certain faults aside removes those easy positives. The gap between validation and test narrowed, and test performance improved.

### 5.3 What the model adds (contribution check, validation PR-AUC)

| Scorer | PR-AUC |
|---|---|
| Chance (anomaly rate) | 0.011 |
| \|residual\| alone, no model | 0.785 |
| Full stacked model | 0.852 |
| Model without PM-level features | 0.045 |
| … and without rolling means | 0.010 |

**Interpretation.** Most of the consensus is captured by the size of the baseline residual. The meta-model adds a modest, real gain by combining it with PM10, PM1 availability and context. Without PM-level information it collapses to chance, so it does not discover anomaly types the detectors miss: it is a fast, explainable *reproduction* of the detector consensus. SHAP agrees: PM10, the residual and PM2.5 dominate.

### 5.4 Detectors and independent evidence (validation)

| Method | Flag rate | F1 vs label | Corroboration* |
|---|---|---|---|
| MAD (statistical) | 13.6% | 0.15 | 0.40 (n = 1,089) |
| LOF | 2.1% | 0.15 | 0.00 (n = 128) |
| One-Class SVM | 1.1% | 0.79 | 0.07 (n = 60) |
| DBSCAN | 0.8% | 0.78 | 0.08 (n = 24) |
| STL | 14.4% | 0.11 | 0.15 (n = 1,022) |
| Stacked model | 1.0% | 0.81 | 0.08 (n = 24) |

\*Share of flags, in neighbourhood-hours where at least two sensors reported, that another sensor in the same neighbourhood also flagged.

**Consensus ablation.** Dropping the MAD detector changes the label most (F1 vs full label 0.63); dropping LOF changes it least (0.91).

**Interpretation.** Corroboration is low for every selective method. Most stacked flags are isolated to one device, which is the signature of local sources or faults rather than city-wide episodes. Only 24 stacked flags in 2024 even had a reporting neighbour.

**Flag concentration.** 68% of validation flags come from one sensor (3576, Nairobi City, including the 846 µg/m³ spike on 15 May 2024). In the test period, 56% come from sensor 3966. Metrics in these years mostly describe one or two devices.

### 5.5 Triage outputs

| | Validation | Test |
|---|---|---|
| Severity of flags | 143 Critical · 71 Alert | 29 Critical · 17 Alert · 4 Watch |
| Cause of flags | 190 unverifiable · 18 likely fault · 6 genuine | 37 unverifiable · 12 likely fault · 1 genuine |

The neighbour-free cause model reaches macro-F1 0.52 ± 0.16 in sensor-grouped CV (balanced accuracy 0.56) on 339 labelled flags. That is close to chance, so the API labels its output as an advisory guess.

### 5.6 Spatial and external validation

* **Moran's I** (10 sensors with ≥ 100 readings in validation + test; k = 5; E[I] = −0.11):
  * Mean PM2.5 is spatially clustered (I = 0.16, p = 0.017).
  * The anomaly rate shows weak evidence of clustering (I ≈ 0.00, p = 0.04).
  * With ten units, the test has little power.
* **Documented events** (New Year fireworks windows):
  * 2024: flag rate 6.7% inside the window vs 0.1% outside (one flag in 15 readings).
  * 2025 and 2026: no flags, although mean residuals were positive (+25 and +31 µg/m³).
  * Each window has fewer than 20 readings, so this is anecdotal until more sourced events are added.

## 6. Deployment

| Piece | Implementation |
|---|---|
| Reproducible pipeline | `python -m airq.pipeline ingest \| train \| predict`. Training takes about 2 minutes on one CPU core and writes `models/` (meta-model, preprocessing tables, cause model, `model_config.json` with data SHA-256 and package versions) and `reports/metrics.json`. |
| API | FastAPI (`api/main.py`): `/health`, `/model-info`, `/predict` (JSON), `/predict/file` (CSV/XLSX, JSON or CSV out); per-request threshold and severity overrides; optional SHAP explanations. |
| Containers | `Dockerfile` (API, health check, non-root user), `Dockerfile.pipeline` (retraining with bind mounts), `docker-compose.yml` (API :8000, Streamlit dashboard :8501, pipeline service). |
| Quality | 13 pytest tests (pipeline round trip, leakage guards, API, Streamlit app); GitHub Actions runs them, a quick retrain and a Docker build on every push. The notebook asserts feature/probability/threshold parity with the deployed package. |

## 7. Limitations

1. **No ground truth.** Label metrics measure agreement with the detector consensus, not correctness; the test PR-AUC of 0.45 means the consensus is only partly reproducible on new sensors and years.
2. **Few, concentrated sensors after 2023.** Validation and test results are driven by one or two devices each, and test has only 82 pseudo-positives.
3. **Cause attribution is weak where it matters most.** Most flags have no reporting neighbour, and the neighbour-free model is near chance; genuinely local sources (roadside burning) can look like faults.
4. **Humidity bias is flagged, not corrected.** There is no co-located reference monitor for calibration.
5. **External validation is thin.** Documented events still need citations and more coverage.
6. **Docker** images were specified and the services tested locally, but the image build must be confirmed on a machine with Docker (CI does this on push).

## 8. Future work

* Co-location with reference monitors for humidity calibration, and an expert-labelled event set to audit the pseudo-labels and calibrate probabilities.
* Monthly scheduled ingestion and retraining with drift gates (Evidently) and experiment tracking (MLflow).
* Sequence models once longer, cleaner series exist.
* Integration with Nairobi County alert routing.

## 9. Real-world use

An analyst uploads the latest month of archive data, or a scheduled job posts it to the API, and receives a short, ranked list instead of thousands of readings to inspect:

* **Critical flags** with a one-line reason.
* **Out-of-range and likely-fault readings** routed to the maintenance team with the sensor ID.
* **Neighbour-supported events** routed to public-health officers.
* **Unverifiable flags** queued for manual review, with the model's advisory guess.
