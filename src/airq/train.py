"""End-to-end training: processed dataset -> fitted artifacts in models/ and metrics in reports/.

    python -m airq.pipeline train
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from importlib import metadata as md

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix

from airq import __version__
from airq.config import load_config, resolve
from airq.data.ingest import load_dataset
from airq.detectors import BASE_FLAGS, DetectorSuite
from airq.evaluate import (consensus_ablation, corroboration, detector_comparison, documented_event_check,
                           label_metrics, morans_i_by_sensor, per_sensor_concentration)
from airq.model import choose_threshold, cross_validate_meta, make_meta_model, prune_features
from airq.preprocess import (ConditionalImputer, SeasonalBaseline, add_ratio_and_neighbourhood_features,
                             add_rolling_features, clean_rows, interpolate_short_gaps, parse_times)
from airq.triage import CAUSE_FEATURES, assign_cause, fit_cause_model, severity

log = logging.getLogger(__name__)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.round(4).to_dict(orient="index"))
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self, msg):
        log.info("[%6.0fs] %s", time.time() - self.t0, msg)


def prepare_splits(df_raw: pd.DataFrame, cfg: dict):
    """Sections 2, 5, 8, 9 and 10: clean, impute (train-fitted), engineer, split, add baseline."""
    df = parse_times(df_raw, cfg["local_utc_offset_hours"])
    df, faults, rep = clean_rows(df, cfg)
    df = interpolate_short_gaps(df, cfg["cleaning"]["interpolate_limit"])
    train_end, val_end = pd.Timestamp(cfg["split"]["train_end"]), pd.Timestamp(cfg["split"]["val_end"])
    imputer = ConditionalImputer().fit(df[df["DATETIME"] < train_end])
    df = imputer.transform(df)
    df = add_rolling_features(df, cfg["features"]["rolling_windows"])
    df = add_ratio_and_neighbourhood_features(df)
    train = df[df["DATETIME"] < train_end].reset_index(drop=True)
    val = df[(df["DATETIME"] >= train_end) & (df["DATETIME"] < val_end)].reset_index(drop=True)
    test = df[df["DATETIME"] >= val_end].reset_index(drop=True)
    baseline = SeasonalBaseline().fit(train)
    train, val, test = (baseline.transform(d) for d in (train, val, test))
    return train, val, test, faults, rep, imputer, baseline


def run_training(cfg: dict | None = None, data_path: str | None = None, quick: bool = False) -> dict:
    cfg = cfg or load_config()
    rs = cfg["random_state"]
    tick = Timer()
    models_dir, reports_dir = resolve(cfg["paths"]["models_dir"]), resolve(cfg["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    path = resolve(data_path or cfg["data_path"])
    if not path.exists():
        alt = resolve(cfg["ingest"]["output_xlsx"])
        if not alt.exists():
            raise FileNotFoundError(f"No dataset at {path} or {alt}. Run `python -m airq.pipeline ingest` first.")
        path = alt
    raw = load_dataset(path)
    tick(f"Loaded {len(raw):,} rows from {path.name}")

    train, val, test, faults, clean_rep, imputer, baseline = prepare_splits(raw, cfg)
    tick(f"Prepared splits train={len(train):,} val={len(val):,} test={len(test):,}; "
         f"{len(faults)} readings above sensor range set aside")

    det = DetectorSuite(cfg, rs).fit(train)
    tick("Fitted five detectors")
    flags = {"train": det.flag_train(train), "val": det.flag(val), "test": det.flag(test)}
    for name, d in (("train", train), ("val", val), ("test", test)):
        for c in flags[name].columns:
            d[c] = flags[name][c].values
    tick("Detector flags and pseudo-labels computed")
    assert train["pseudo_label"].nunique() == 2, "Pseudo-label has one class in train."

    meta_features, dropped, gain = prune_features(train, rs)
    tick(f"Pruned features -> {meta_features}")

    n_est = 150 if quick else None
    model = make_meta_model(meta_features, cfg, n_estimators=n_est, random_state=rs)
    cv = cross_validate_meta(model, train, meta_features, cfg, rs) if not quick else {}
    tick(f"Cross-validation: {cv}")
    model.fit(train[meta_features], train["pseudo_label"])
    tick("Fitted final meta-model")

    val_p = model.predict_proba(val[meta_features])[:, 1]
    thr = choose_threshold(val["pseudo_label"], val_p)
    test_p = model.predict_proba(test[meta_features])[:, 1]
    val["anomaly_probability"], val["is_anomaly"] = val_p, (val_p >= thr).astype(int)
    test["anomaly_probability"], test["is_anomaly"] = test_p, (test_p >= thr).astype(int)

    metrics = {"val": label_metrics(val["pseudo_label"], val_p, val["is_anomaly"]),
               "val_at_0.5": label_metrics(val["pseudo_label"], val_p, (val_p >= 0.5).astype(int)),
               "test": label_metrics(test["pseudo_label"], test_p, test["is_anomaly"])}
    for name, d in (("val", val), ("test", test)):
        r, n = corroboration(d, d["is_anomaly"])
        metrics[name]["corroboration"], metrics[name]["corroboration_n"] = r, n
        metrics[name]["confusion_matrix"] = confusion_matrix(d["pseudo_label"], d["is_anomaly"]).tolist()
        metrics[name]["flag_concentration"] = per_sensor_concentration(d, d["is_anomaly"])
    comparison = detector_comparison(val, val_p, val["is_anomaly"])
    ablation = consensus_ablation(val, cfg["detectors"]["consensus_votes"])
    tick(f"Validation: {metrics['val']}")

    # Contribution check (Section 13.4)
    contrib = {"chance (anomaly rate)": float(val["pseudo_label"].mean()),
               "|residual| alone": float(average_precision_score(val["pseudo_label"], val["residual"].abs())),
               "full stacked model": metrics["val"].get("pr_auc")}
    pm_level = [f for f in meta_features if f in ["PM25", "PM10", "residual", "pm25_pm10_ratio",
                                                   "hood_hour_mean", "hood_deviation"]]
    roll = [f for f in meta_features if f.startswith("pm25_roll_mean")]
    for label, feats in (("without PM-level features", [f for f in meta_features if f not in pm_level]),
                         ("without PM-level + rolling means",
                          [f for f in meta_features if f not in pm_level + roll])):
        if feats and not quick:
            m = make_meta_model(feats, cfg, n_estimators=300, random_state=rs)
            m.fit(train[feats], train["pseudo_label"])
            contrib[label] = float(average_precision_score(val["pseudo_label"], m.predict_proba(val[feats])[:, 1]))
    tick(f"Contribution check: {contrib}")

    # Severity and cause (Section 15)
    for d in (val, test):
        d["severity"] = severity(d["is_anomaly"], d["residual"], d["PM25"], det.res_med_, det.res_mad_, cfg)
        d["cause_label"], d["nbr_resid"], _ = assign_cause(d, d["is_anomaly"], cfg["cause"]["nbr_support"])
    pool = []
    for d in (train, val):
        c, n, _ = assign_cause(d, d["pseudo_label"], cfg["cause"]["nbr_support"])
        x = d.assign(cause_label=c, nbr_resid=n)
        pool.append(x[x["cause_label"].isin(["genuine", "likely_sensor_fault"])])
    cause_bundle = fit_cause_model(pd.concat(pool, ignore_index=True), cfg["cause"]["min_per_class"], rs)
    ct = test[test["cause_label"].isin(["genuine", "likely_sensor_fault"])]
    if cause_bundle["model"] is not None and ct["cause_label"].nunique() == 2:
        cause_bundle["info"]["test_report"] = classification_report(
            ct["cause_label"], cause_bundle["model"].predict(ct[CAUSE_FEATURES]), output_dict=True, zero_division=0)
    else:
        cause_bundle["info"]["test_report"] = f"skipped; test flagged labels: {ct['cause_label'].value_counts().to_dict()}"
    tick("Severity, cause labels and cause classifier done")

    scored = pd.concat([val, test], ignore_index=True)
    ev_path = resolve(cfg["paths"]["events_file"])
    events = documented_event_check(scored, pd.read_csv(ev_path)) if ev_path.exists() else pd.DataFrame()
    moran = morans_i_by_sensor(scored, seed=rs)
    tick("External checks done")

    # ----------------------------------------------------------------- save
    prep = {"imputer": imputer, "baseline": baseline, "res_med": det.res_med_, "res_mad": det.res_mad_}
    joblib.dump(model, models_dir / "meta_model.joblib")
    joblib.dump(prep, models_dir / "preprocess.joblib")
    joblib.dump({"model": cause_bundle["model"], "features": CAUSE_FEATURES, "info": cause_bundle["info"]},
                models_dir / "cause_model.joblib")
    versions = {}
    for p in ["pandas", "numpy", "scikit-learn", "imbalanced-learn", "xgboost", "shap", "statsmodels"]:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    model_config = {
        "airq_version": __version__, "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "meta_features": meta_features, "threshold": thr, "nbr_support": cfg["cause"]["nbr_support"],
        "severity": cfg["severity"], "pm_sensor_max": cfg["cleaning"]["pm_sensor_max"],
        "local_utc_offset_hours": cfg["local_utc_offset_hours"], "split": cfg["split"],
        "data_file": path.name, "data_sha256": _sha256(path), "quick_mode": quick,
        "python": platform.python_version(), "packages": versions,
    }
    with open(models_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(model_config), f, indent=2)

    report = {
        "cleaning": clean_rep,
        "splits": {n: {"rows": int(len(d)), "sensors": int(d["LOCATION"].nunique()),
                       "start": str(d["DATETIME"].min()), "end": str(d["DATETIME"].max()),
                       "pseudo_positive_rate": float(d["pseudo_label"].mean())}
                   for n, d in (("train", train), ("val", val), ("test", test))},
        "detector_flag_rates": {n: {c: float(d[c].mean()) for c in BASE_FLAGS}
                                for n, d in (("train", train), ("val", val), ("test", test))},
        "base_votes_train": train["base_votes"].value_counts().sort_index().to_dict(),
        "feature_pruning": {"kept": meta_features, "dropped": dropped, "probe_gain": gain.round(4).to_dict()},
        "cv": cv, "threshold": thr, "metrics": metrics, "detector_comparison": comparison,
        "consensus_ablation": ablation, "contribution": contrib,
        "severity_counts": {n: pd.Series(d["severity"]).value_counts().to_dict() for n, d in (("val", val), ("test", test))},
        "cause_counts": {n: d.loc[d["is_anomaly"] == 1, "cause_label"].value_counts().to_dict()
                         for n, d in (("val", val), ("test", test))},
        "cause_model": cause_bundle["info"],
        "documented_events": events, "morans_i": moran,
        "range_fault_readings": {"n": int(len(faults)),
                                 "by_sensor": faults["LOCATION"].value_counts().to_dict() if len(faults) else {}},
    }
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=2, default=str)
    cols = ["DATETIME", "DATETIME_UTC", "LOCATION", "NEIGHBORHOOD", "PM25", "PM10", "HUMIDITY", "baseline",
            "residual", "anomaly_probability", "is_anomaly", "severity", "cause_label", "nbr_resid",
            "pseudo_label", "base_votes"] + BASE_FLAGS
    scored[cols].to_csv(reports_dir / "scored_val_test.csv.gz", index=False, compression="gzip")
    tick(f"Saved artifacts to {models_dir} and reports to {reports_dir}")
    return report
