"""Cleaning, imputation, feature engineering and the seasonal baseline.

Mirrors notebook Sections 2, 5, 8 and 10. Anything learned from data has a fit()
on the training period and a transform() used unchanged at scoring time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MEASURES = ["PM1", "PM25", "PM10", "HUMIDITY", "TEMPERATURE"]
REQUIRED_COLUMNS = ["LOCATION", "NEIGHBORHOOD", "PM25", "PM10"]   # plus DATETIME_UTC / DATETIME / DATE+TIME


# ---------------------------------------------------------------------------
# Time handling (Section 2)
# ---------------------------------------------------------------------------
def parse_times(df: pd.DataFrame, utc_offset_hours: int = 3) -> pd.DataFrame:
    """Add DATETIME_UTC and DATETIME (local Nairobi time) and local calendar columns.

    The archive and the hourly dataset are in UTC. Diurnal structure (rush hours, cooking,
    weekdays) is local, so HOUR / WEEKDAY / MONTH / YEAR are recomputed from local time.
    """
    df = df.copy()
    if "DATETIME_UTC" in df.columns:
        utc = pd.to_datetime(df["DATETIME_UTC"])
    elif "DATETIME" in df.columns:           # API / CSV clients: DATETIME is UTC, like the archive
        utc = pd.to_datetime(df["DATETIME"])
    else:
        utc = pd.to_datetime(pd.to_datetime(df["DATE"]).dt.strftime("%Y-%m-%d") + " " + df["TIME"].astype(str))
    if getattr(utc.dt, "tz", None) is not None:
        utc = utc.dt.tz_convert("UTC").dt.tz_localize(None)
    df["DATETIME_UTC"] = utc.dt.floor("h")
    df["DATETIME"] = df["DATETIME_UTC"] + pd.Timedelta(hours=utc_offset_hours)
    df["HOUR"] = df["DATETIME"].dt.hour
    df["WEEKDAY"] = df["DATETIME"].dt.day_name()
    df["MONTH"] = df["DATETIME"].dt.month
    df["YEAR"] = df["DATETIME"].dt.year
    return df


def check_columns(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if not ({"DATETIME_UTC"} <= set(df.columns) or {"DATETIME"} <= set(df.columns)
            or {"DATE", "TIME"} <= set(df.columns)):
        missing.append("DATETIME (UTC) or DATE + TIME")
    return missing


# ---------------------------------------------------------------------------
# Row-level cleaning (Section 5.1 - 5.4)
# ---------------------------------------------------------------------------
def clean_rows(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Physical bounds -> NaN, sensor-range faults set aside, unusable rows dropped.

    Returns (clean, range_faults, report). range_faults are readings above the PM sensors'
    measuring limit: they are certain malfunctions, so they are labelled by rule and kept
    out of every fitted quantity (baseline, scalers, detectors, rolling windows).
    """
    c = cfg["cleaning"]
    df = df.copy()
    for col in MEASURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    rep = {"rows_in": int(len(df))}

    lo, hi = c["temperature_range"]
    temp_bad = df["TEMPERATURE"].notna() & ~df["TEMPERATURE"].between(lo, hi)
    lo, hi = c["humidity_range"]
    hum_bad = df["HUMIDITY"].notna() & ~df["HUMIDITY"].between(lo, hi)
    df.loc[temp_bad, "TEMPERATURE"] = np.nan
    df.loc[hum_bad, "HUMIDITY"] = np.nan
    rep["invalid_temperature"] = int(temp_bad.sum())
    rep["invalid_humidity"] = int(hum_bad.sum())

    pm_max = c["pm_sensor_max"]
    range_bad = ((df["PM25"] > pm_max) | (df["PM10"] > pm_max)).fillna(False)
    range_faults = df[range_bad].copy()
    df = df[~range_bad].copy()
    rep["pm_above_sensor_range"] = int(range_bad.sum())

    pm_order_bad = ((df["PM10"] < df["PM25"]) | (df["PM25"] < df["PM1"])).fillna(False)
    df.loc[pm_order_bad, ["PM1", "PM25", "PM10"]] = np.nan
    rep["pm_ordering_violations"] = int(pm_order_bad.sum())

    before = len(df)
    df = df.dropna(subset=["PM25", "PM10"])
    rep["dropped_missing_pm"] = int(before - len(df))
    before = len(df)
    df = df[df["NEIGHBORHOOD"] != "No coordinates"].copy()
    rep["dropped_no_coordinates"] = int(before - len(df))

    for col in ["PM1", "HUMIDITY", "TEMPERATURE"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    df = df.sort_values(["LOCATION", "DATETIME"]).reset_index(drop=True)
    rep["rows_out"] = int(len(df))
    return df, range_faults, rep


def interpolate_short_gaps(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Section 5.6: per-sensor linear interpolation of humidity/temperature, at most `limit` rows."""
    df = df.copy()
    for col in ["HUMIDITY", "TEMPERATURE"]:
        df[col] = df.groupby("LOCATION")[col].transform(lambda s: s.interpolate(limit=limit))
    return df


class ConditionalImputer:
    """Section 5.7: neighbourhood x hour medians from the training period, then a global median."""

    cols = ["HUMIDITY", "TEMPERATURE"]

    def fit(self, train: pd.DataFrame) -> "ConditionalImputer":
        self.tables_ = {c: train.groupby(["NEIGHBORHOOD", "HOUR"])[c].median() for c in self.cols}
        self.global_ = {c: float(train[c].median()) for c in self.cols}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        keys = pd.MultiIndex.from_frame(df[["NEIGHBORHOOD", "HOUR"]])
        for c in self.cols:
            fill = pd.Series(self.tables_[c].reindex(keys).values, index=df.index)
            df[c] = df[c].fillna(fill).fillna(self.global_[c])
        return df


# ---------------------------------------------------------------------------
# Features (Section 8)
# ---------------------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame, windows=(6, 24, 72)) -> pd.DataFrame:
    """Past-only rolling mean/std of PM2.5 per sensor (shift(1) excludes the current reading)."""
    df = df.sort_values(["LOCATION", "DATETIME"]).copy()
    past = df.groupby("LOCATION")["PM25"].shift(1).groupby(df["LOCATION"])
    for w in windows:
        df[f"pm25_roll_mean_{w}"] = past.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"pm25_roll_std_{w}"] = past.transform(lambda x: x.rolling(w, min_periods=2).std())
    return df


def add_ratio_and_neighbourhood_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pm25_pm10_ratio"] = df["PM25"] / (df["PM10"] + 1e-6)
    g = df.groupby(["NEIGHBORHOOD", "DATETIME"])["PM25"]
    df["hood_hour_mean"] = g.transform("mean")
    df["hood_hour_std"] = g.transform("std")
    df["hood_deviation"] = df["PM25"] - df["hood_hour_mean"]
    return df


# ---------------------------------------------------------------------------
# Seasonal baseline (Section 10)
# ---------------------------------------------------------------------------
class SeasonalBaseline:
    """Median PM2.5 by neighbourhood x hour x weekday (train only) with a fallback chain."""

    keys3 = ["NEIGHBORHOOD", "HOUR", "WEEKDAY"]
    keys2 = ["NEIGHBORHOOD", "HOUR"]

    def fit(self, train: pd.DataFrame) -> "SeasonalBaseline":
        self.tbl3_ = train.groupby(self.keys3)["PM25"].median()
        self.tbl2_ = train.groupby(self.keys2)["PM25"].median()
        self.global_ = float(train["PM25"].median())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        b3 = self.tbl3_.reindex(pd.MultiIndex.from_frame(df[self.keys3])).values
        b2 = self.tbl2_.reindex(pd.MultiIndex.from_frame(df[self.keys2])).values
        b = np.where(np.isnan(b3), b2, b3)
        df["baseline"] = np.where(np.isnan(b), self.global_, b)
        df["residual"] = df["PM25"] - df["baseline"]
        return df


def robust_z(residual, med: float, mad: float):
    """Modified z-score 0.6745 (x - median) / MAD."""
    return 0.6745 * (np.asarray(residual, dtype=float) - med) / mad if mad else np.zeros(len(residual))
