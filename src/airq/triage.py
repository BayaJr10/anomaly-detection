"""Severity tiers and likely-cause labels for flagged readings (notebook Section 15)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from airq.preprocess import robust_z

SEVERITY_ORD = {"Normal": 0, "Watch": 1, "Alert": 2, "Critical": 3}
CAUSE_FEATURES = ["PM25", "PM10", "HUMIDITY", "TEMPERATURE", "pm25_roll_mean_24", "pm25_roll_std_24",
                  "pm25_pm10_ratio", "residual", "PM1_missing", "HUMIDITY_missing", "TEMPERATURE_missing"]
STAKEHOLDER = {  # who a flag would be routed to
    "genuine": "Environment / public-health officer",
    "likely_sensor_fault": "Sensor network maintenance team",
    "sensor_fault_out_of_range": "Sensor network maintenance team",
    "unverifiable": "Data analyst (manual review)",
    "not_flagged": "",
}


def severity(flag, residual, pm25, res_med: float, res_mad: float, cfg: dict) -> np.ndarray:
    """Watch / Alert / Critical for flagged readings, from how unusual AND how unhealthy they are.

    The meta-model's probability is not used: after SMOTE it is uncalibrated and every reading
    above the frozen threshold already has p > 0.99, so it cannot separate tiers.
      Critical: robust z >= critical_z  OR  PM2.5 >= critical_pm25
      Alert:    robust z >= alert_z     OR  PM2.5 >= alert_pm25
      Watch:    any other flagged reading
    """
    s = cfg["severity"]
    z = np.abs(robust_z(residual, res_med, res_mad))
    pm = np.asarray(pm25, dtype=float)
    f = np.asarray(flag).astype(int) == 1
    out = np.full(len(z), "Normal", dtype=object)
    out[f] = "Watch"
    out[f & ((z >= s["alert_z"]) | (pm >= s["alert_pm25"]))] = "Alert"
    out[f & ((z >= s["critical_z"]) | (pm >= s["critical_pm25"]))] = "Critical"
    return out


def neighbour_evidence(d: pd.DataFrame):
    """(sensors reporting in the neighbourhood-hour, mean residual of the OTHER sensors)."""
    grp = d.groupby(["NEIGHBORHOOD", "DATETIME"])
    n_sens = grp["LOCATION"].transform("nunique")
    n_rows = grp["residual"].transform("count")
    nbr = (grp["residual"].transform("sum") - d["residual"]) / (n_rows - 1).where(n_rows > 1)
    return n_sens, nbr


def assign_cause(d: pd.DataFrame, flag, support: float = 0.5):
    """genuine / likely_sensor_fault / unverifiable from how much of the excess neighbours share."""
    n_sens, nbr = neighbour_evidence(d)
    own = d["residual"]
    supported = (np.sign(own) * nbr) >= support * own.abs().clip(lower=1)
    f = pd.Series(np.asarray(flag).astype(int), index=d.index) == 1
    cause = pd.Series("not_flagged", index=d.index, dtype=object)
    cause[f & (n_sens < 2)] = "unverifiable"
    cause[f & (n_sens >= 2) & supported] = "genuine"
    cause[f & (n_sens >= 2) & ~supported] = "likely_sensor_fault"
    return cause, nbr, n_sens


def humidity_artefact_risk(d: pd.DataFrame, rh: float = 85.0) -> pd.Series:
    """Positive excess while relative humidity is high: optical sensors over-read (hygroscopic growth)."""
    return ((d["HUMIDITY"] >= rh) & (d["HUMIDITY_missing"] == 0) & (d["residual"] > 0)).astype(int)


def make_cause_model(random_state: int = 42):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=3,
                                       random_state=random_state, n_jobs=-1)),
    ])


def fit_cause_model(cause_df: pd.DataFrame, min_per_class: int = 30, random_state: int = 42) -> dict:
    """Neighbour-free approximation of the cause rule, for readings with no reporting neighbours."""
    vc = cause_df["cause_label"].value_counts()
    info = {"label_counts": vc.to_dict(), "n_sensors": int(cause_df["LOCATION"].nunique())}
    if len(vc) < 2 or vc.min() < min_per_class:
        info["status"] = "skipped: too few examples of one class"
        return {"model": None, "info": info}
    model = make_cause_model(random_state)
    n_s = cause_df["LOCATION"].nunique()
    if n_s >= 3:
        cv, kw = GroupKFold(n_splits=min(5, n_s)), {"groups": cause_df["LOCATION"]}
        info["cv"] = f"GroupKFold by sensor ({min(5, n_s)} folds)"
    else:
        cv, kw = StratifiedKFold(5, shuffle=True, random_state=random_state), {}
        info["cv"] = "StratifiedKFold (fewer than 3 sensors; optimistic)"
    s = cross_validate(model, cause_df[CAUSE_FEATURES], cause_df["cause_label"], cv=cv,
                       scoring=["f1_macro", "balanced_accuracy"], **kw)
    info["cv_f1_macro"] = (float(s["test_f1_macro"].mean()), float(s["test_f1_macro"].std()))
    info["cv_balanced_accuracy"] = float(s["test_balanced_accuracy"].mean())
    model.fit(cause_df[CAUSE_FEATURES], cause_df["cause_label"])
    info["feature_importance"] = dict(zip(CAUSE_FEATURES, model.named_steps["clf"].feature_importances_.round(4)))
    info["status"] = "trained"
    return {"model": model, "info": info}
