"""Evaluation: label metrics, cross-sensor corroboration, ablation, external checks (Sections 13, 16)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

from airq.detectors import BASE_FLAGS


def label_metrics(y, proba, pred) -> dict:
    y = np.asarray(y)
    out = {"n": int(len(y)), "positives": int(y.sum()), "positive_rate": float(y.mean()),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)), "flag_rate": float(np.mean(pred))}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["pr_auc"] = float(average_precision_score(y, proba))
    return out


def corroboration(d: pd.DataFrame, flag) -> tuple[float, int]:
    """Of flags in neighbourhood-hours with >= 2 sensors, share where another sensor is also flagged."""
    f = pd.Series(np.asarray(flag).astype(int), index=d.index)
    hood_n = d.groupby(["NEIGHBORHOOD", "DATETIME"])["LOCATION"].transform("nunique")
    hood_f = f.groupby([d["NEIGHBORHOOD"], d["DATETIME"]]).transform("sum")
    elig = (f == 1) & (hood_n >= 2)
    n = int(elig.sum())
    return (float(((hood_f >= 2) & elig).sum() / n) if n else float("nan")), n


def detector_comparison(d: pd.DataFrame, proba, pred) -> pd.DataFrame:
    y = d["pseudo_label"]
    rows = {}
    for col in BASE_FLAGS:
        r, n = corroboration(d, d[col])
        rows[col.replace("_flag", "")] = {"flag_rate": d[col].mean(),
                                          "PR-AUC vs label": average_precision_score(y, d[col]),
                                          "F1 vs label": f1_score(y, d[col], zero_division=0),
                                          "corroboration": r, "n_eligible": n}
    r, n = corroboration(d, y)
    rows["consensus (label)"] = {"flag_rate": y.mean(), "PR-AUC vs label": np.nan, "F1 vs label": np.nan,
                                 "corroboration": r, "n_eligible": n}
    r, n = corroboration(d, pred)
    rows["stacked"] = {"flag_rate": np.mean(pred), "PR-AUC vs label": average_precision_score(y, proba),
                       "F1 vs label": f1_score(y, pred, zero_division=0), "corroboration": r, "n_eligible": n}
    return pd.DataFrame(rows).T


def consensus_ablation(d: pd.DataFrame, votes_needed: int = 3) -> pd.DataFrame:
    y = d["pseudo_label"]
    rows = {}
    for drop in BASE_FLAGS:
        rest = [c for c in BASE_FLAGS if c != drop]
        v = d[rest].sum(axis=1)
        lab = (v >= votes_needed).astype(int)
        rows[drop] = {"label_flip_rate": float((lab != y).mean()),
                      "F1 vs full label": f1_score(y, lab, zero_division=0),
                      "PR-AUC vs full label": average_precision_score(y, v / len(rest))}
    return pd.DataFrame(rows).T


def per_sensor_concentration(d: pd.DataFrame, flag) -> dict:
    """How much of the flagging is driven by a single sensor (a fault episode inflates metrics)."""
    f = pd.Series(np.asarray(flag).astype(int), index=d.index)
    by = f.groupby(d["LOCATION"]).sum().sort_values(ascending=False)
    tot = int(by.sum())
    return {"total_flags": tot, "top_sensor": int(by.index[0]) if tot else None,
            "top_sensor_share": float(by.iloc[0] / tot) if tot else float("nan"),
            "top3": {int(k): int(v) for k, v in by.head(3).items()}}


def documented_event_check(scored: pd.DataFrame, events: pd.DataFrame, flag_col: str = "is_anomaly") -> pd.DataFrame:
    """Flag-rate lift inside documented pollution-event windows vs the same sensors outside them.

    events columns: start_local, end_local, neighbourhood ('' or 'ALL'), description, source.
    """
    rows = []
    for _, e in events.iterrows():
        start, end = pd.Timestamp(e["start_local"]), pd.Timestamp(e["end_local"])
        hood = str(e.get("neighbourhood", "") or "").strip()
        scope = scored if hood in ("", "ALL", "nan") else scored[scored["NEIGHBORHOOD"] == hood]
        inside = scope["DATETIME"].between(start, end)
        if inside.sum() == 0:
            continue
        sensors = scope.loc[inside, "LOCATION"].unique()
        outside = scope["LOCATION"].isin(sensors) & ~inside
        r_in = float(scope.loc[inside, flag_col].mean())
        r_out = float(scope.loc[outside, flag_col].mean()) if outside.any() else float("nan")
        rows.append({"event": e["description"], "start": start, "end": end, "scope": hood or "ALL",
                     "readings_in_window": int(inside.sum()), "sensors": len(sensors),
                     "flag_rate_in": r_in, "flag_rate_out": r_out,
                     "lift": r_in / r_out if r_out and r_out > 0 else float("nan"),
                     "mean_residual_in": float(scope.loc[inside, "residual"].mean()),
                     "source": e.get("source", "")})
    return pd.DataFrame(rows)


def morans_i_by_sensor(scored: pd.DataFrame, flag_col: str = "is_anomaly", min_n: int = 100,
                       k: int = 5, permutations: int = 999, seed: int = 42) -> dict:
    """Global Moran's I of the per-sensor anomaly rate (sensor-level coordinates, KNN weights)."""
    try:
        from esda.moran import Moran
        from libpysal.weights import KNN
    except ImportError:
        return {"status": "esda/libpysal not installed"}
    st = (scored.groupby("LOCATION")
          .agg(rate=(flag_col, "mean"), pm25=("PM25", "mean"), n=(flag_col, "size"),
               lat=("NEIGHBORHOOD_LAT", "first"), lon=("NEIGHBORHOOD_LON", "first")).dropna())
    st = st[st["n"] >= min_n]
    if len(st) < 6:
        return {"status": f"too few sensors ({len(st)}) for Moran's I", "n_sensors": int(len(st))}
    np.random.seed(seed)
    w = KNN.from_array(st[["lon", "lat"]].values, k=min(k, len(st) - 1))
    w.transform = "r"
    out = {"status": "ok", "n_sensors": int(len(st))}
    for var in ["pm25", "rate"]:
        mi = Moran(st[var].values, w, permutations=permutations)
        out[var] = {"I": float(mi.I), "p_sim": float(mi.p_sim)}
    return out
