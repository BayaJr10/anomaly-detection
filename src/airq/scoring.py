"""Load the trained artifacts and score raw hourly readings.

One code path serves the batch CLI (`python -m airq.pipeline predict`), the FastAPI service
and the notebook parity check, so features at serving time are built exactly as in training.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from airq.config import load_config, resolve
from airq.preprocess import (add_ratio_and_neighbourhood_features, add_rolling_features, check_columns,
                             clean_rows, interpolate_short_gaps, parse_times, robust_z)
from airq.triage import (CAUSE_FEATURES, SEVERITY_ORD, STAKEHOLDER, assign_cause, humidity_artefact_risk,
                         severity)

OUTPUT_COLUMNS = [
    "LOCATION", "NEIGHBORHOOD", "DATETIME_UTC", "DATETIME", "PM25", "PM10", "HUMIDITY", "TEMPERATURE",
    "baseline", "residual", "robust_z", "anomaly_probability", "is_anomaly", "rule", "severity",
    "cause_label", "cause_model_guess", "cause_model_confidence", "nbr_resid", "sensors_in_neighbourhood",
    "humidity_artefact_risk", "route_to",
]


def build_features(raw: pd.DataFrame, prep: dict, cfg: dict):
    """Raw hourly rows -> (model-ready features, range-fault rows, cleaning report)."""
    df = parse_times(raw, cfg["local_utc_offset_hours"])
    df, faults, rep = clean_rows(df, cfg)
    df = interpolate_short_gaps(df, cfg["cleaning"]["interpolate_limit"])
    df = prep["imputer"].transform(df)
    df = add_rolling_features(df, cfg["features"]["rolling_windows"])
    df = add_ratio_and_neighbourhood_features(df)
    df = prep["baseline"].transform(df)
    return df, faults, rep


FEATURE_TEXT = {
    "residual": lambda r: (f"PM2.5 is {r['residual']:+.0f} µg/m³ from the usual {r['baseline']:.0f} µg/m³ "
                           f"for {r['NEIGHBORHOOD']} at {int(r['HOUR']):02d}:00 on a {r['WEEKDAY']}"),
    "PM25": lambda r: f"PM2.5 reading of {r['PM25']:.0f} µg/m³",
    "PM10": lambda r: f"PM10 reading of {r['PM10']:.0f} µg/m³",
    "pm25_pm10_ratio": lambda r: (f"PM2.5/PM10 ratio of {r['pm25_pm10_ratio']:.2f} ("
                                  + ("fine, combustion-like" if r["pm25_pm10_ratio"] >= 0.7 else
                                     "coarse, dust-like" if r["pm25_pm10_ratio"] < 0.4 else "mixed")
                                  + " particles)"),
    "hood_deviation": lambda r: f"{r['hood_deviation']:+.0f} µg/m³ versus other sensors in the neighbourhood this hour",
    "hood_hour_std": lambda r: f"spread of {r['hood_hour_std']:.0f} µg/m³ between sensors in the neighbourhood",
    "HUMIDITY": lambda r: f"relative humidity of {r['HUMIDITY']:.0f}%",
    "TEMPERATURE": lambda r: f"temperature of {r['TEMPERATURE']:.0f} °C",
    "PM1_missing": lambda r: "sensor has no PM1 channel" if r["PM1_missing"] else "sensor reports PM1",
    "HUMIDITY_missing": lambda r: "no humidity reading this hour" if r["HUMIDITY_missing"] else "humidity reported",
}


def _feature_text(name: str, row: pd.Series) -> str:
    if name in FEATURE_TEXT:
        try:
            return FEATURE_TEXT[name](row)
        except (KeyError, TypeError, ValueError):
            pass
    if name.startswith("pm25_roll_mean_"):
        return f"average PM2.5 over the previous {name.rsplit('_', 1)[-1]} hours of {row[name]:.0f} µg/m³"
    if name.startswith("pm25_roll_std_"):
        return f"PM2.5 variability over the previous {name.rsplit('_', 1)[-1]} hours of {row[name]:.1f} µg/m³"
    return f"{name} = {row.get(name)}"


class AnomalyScorer:
    def __init__(self, models_dir: str | Path | None = None, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        d = resolve(models_dir or self.cfg["paths"]["models_dir"])
        self.models_dir = d
        self.meta_model = joblib.load(d / "meta_model.joblib")
        self.prep = joblib.load(d / "preprocess.joblib")
        self.cause = joblib.load(d / "cause_model.joblib")
        with open(d / "model_config.json", encoding="utf-8") as f:
            self.model_config = json.load(f)
        self.meta_features = self.model_config["meta_features"]
        self.threshold = float(self.model_config["threshold"])
        self._explainer = None

    # ------------------------------------------------------------------ SHAP
    @property
    def explainer(self):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.meta_model.named_steps["clf"])
        return self._explainer

    def shap_values(self, feats: pd.DataFrame) -> np.ndarray:
        Xp = self.meta_model.named_steps["prep"].transform(feats[self.meta_features])
        sv = np.asarray(self.explainer.shap_values(Xp))
        return sv[..., -1] if sv.ndim == 3 else sv

    def explain(self, feats: pd.DataFrame, top_k: int = 3) -> list[dict]:
        if feats.empty:
            return []
        sv = self.shap_values(feats)
        out = []
        for i, (_, row) in enumerate(feats.iterrows()):
            order = np.argsort(-np.abs(sv[i]))[:top_k]
            drivers = [{"feature": self.meta_features[j], "value": float(row[self.meta_features[j]]),
                        "shap": float(sv[i, j]),
                        "direction": "raises" if sv[i, j] > 0 else "lowers",
                        "text": _feature_text(self.meta_features[j], row)} for j in order]
            ups = [d["text"] for d in drivers if d["direction"] == "raises"]
            sentence = ("Flagged mainly because of " + "; ".join(ups) + "." if ups else
                        "No single feature pushed the score up strongly.")
            out.append({"drivers": drivers, "explanation": sentence})
        return out

    # --------------------------------------------------------------- scoring
    def score(self, raw: pd.DataFrame, threshold: float | None = None, explain: bool = False,
              top_k: int = 3, severity_cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
        missing = check_columns(raw)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        cfg = dict(self.cfg)
        if severity_cfg:
            cfg["severity"] = {**self.cfg["severity"], **severity_cfg}
        thr = self.threshold if threshold is None else float(threshold)
        feats, faults, rep = build_features(raw, self.prep, cfg)

        proba = self.meta_model.predict_proba(feats[self.meta_features])[:, 1] if len(feats) else np.array([])
        feats["anomaly_probability"] = proba
        feats["is_anomaly"] = (proba >= thr).astype(int)
        feats["rule"] = np.where(feats["is_anomaly"] == 1, "meta_model", "")
        med, mad = self.prep["res_med"], self.prep["res_mad"]
        feats["robust_z"] = robust_z(feats["residual"], med, mad)
        feats["severity"] = severity(feats["is_anomaly"], feats["residual"], feats["PM25"], med, mad, cfg)
        cause, nbr, n_s = assign_cause(feats, feats["is_anomaly"], self.model_config["nbr_support"])
        feats["cause_label"], feats["nbr_resid"], feats["sensors_in_neighbourhood"] = cause, nbr, n_s
        feats["humidity_artefact_risk"] = humidity_artefact_risk(feats)
        feats["cause_model_guess"], feats["cause_model_confidence"] = "", np.nan
        unv = feats["cause_label"] == "unverifiable"
        if self.cause.get("model") is not None and unv.any():
            m = self.cause["model"]
            p = m.predict_proba(feats.loc[unv, CAUSE_FEATURES])
            feats.loc[unv, "cause_model_guess"] = m.classes_[p.argmax(1)]
            feats.loc[unv, "cause_model_confidence"] = p.max(1)

        if len(faults):     # readings beyond the PM sensors' measuring range: certain malfunctions
            f = faults.copy()
            for c in ["baseline", "residual", "robust_z", "nbr_resid", "sensors_in_neighbourhood",
                      "cause_model_confidence"]:
                f[c] = np.nan
            f["anomaly_probability"], f["is_anomaly"], f["rule"] = 1.0, 1, "pm_above_sensor_range"
            f["severity"], f["cause_label"] = "Critical", "sensor_fault_out_of_range"
            f["cause_model_guess"], f["humidity_artefact_risk"] = "", 0
            feats = pd.concat([feats, f[[c for c in OUTPUT_COLUMNS if c in f.columns]]], ignore_index=True)

        feats["route_to"] = feats["cause_label"].map(STAKEHOLDER).fillna("")
        feats["severity_ord"] = feats["severity"].map(SEVERITY_ORD)
        out = feats.sort_values(["DATETIME", "LOCATION"]).reset_index(drop=True)
        if explain:
            mask = (out["is_anomaly"] == 1) & (out["rule"] == "meta_model")
            ex = self.explain(out[mask], top_k)
            out["explanation"] = ""
            out["top_drivers"] = None
            out.loc[mask, "explanation"] = [e["explanation"] for e in ex]
            out.loc[mask, "top_drivers"] = pd.Series([e["drivers"] for e in ex], index=out.index[mask])
        info = {"threshold": thr, "rows_received": int(len(raw)), "rows_scored": int(len(out)),
                "flagged": int(out["is_anomaly"].sum()), "cleaning": rep}
        return out, info
