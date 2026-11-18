"""Leakage and cleaning guarantees of the feature pipeline."""
import numpy as np
import pandas as pd

from airq.config import load_config
from airq.preprocess import (SeasonalBaseline, add_rolling_features, clean_rows, parse_times)

CFG = load_config()


def _frame(n=10, loc=1, hood="A"):
    t = pd.date_range("2023-06-01 00:00", periods=n, freq="h")
    return pd.DataFrame({"LOCATION": loc, "NEIGHBORHOOD": hood, "DATE": t.normalize(),
                         "TIME": t.strftime("%H:%M:%S"), "PM25": np.arange(n, dtype=float) + 5,
                         "PM10": np.arange(n, dtype=float) + 10, "HUMIDITY": 60.0, "TEMPERATURE": 20.0})


def test_utc_to_local_time():
    d = parse_times(_frame(2), 3)
    assert d["DATETIME_UTC"].iloc[0] == pd.Timestamp("2023-06-01 00:00")
    assert d["DATETIME"].iloc[0] == pd.Timestamp("2023-06-01 03:00")
    assert d["HOUR"].iloc[0] == 3 and d["WEEKDAY"].iloc[0] == "Thursday"


def test_range_faults_are_set_aside_and_bounds_applied():
    d = parse_times(_frame(4), 3)
    d.loc[0, "PM25"], d.loc[0, "PM10"] = 3300.0, 3346.0
    d.loc[1, "HUMIDITY"] = 65535.0
    clean, faults, rep = clean_rows(d, CFG)
    assert len(faults) == 1 and rep["pm_above_sensor_range"] == 1
    assert clean["HUMIDITY"].isna().sum() == 1 and clean["HUMIDITY_missing"].sum() == 1
    assert (clean["PM25"] <= CFG["cleaning"]["pm_sensor_max"]).all()


def test_rolling_features_are_past_only():
    d = parse_times(_frame(10), 3)
    a = add_rolling_features(d)
    d2 = d.copy()
    d2.loc[9, "PM25"] = 999.0                 # change the current reading only
    b = add_rolling_features(d2)
    last_a = a[a["DATETIME"] == a["DATETIME"].max()]
    last_b = b[b["DATETIME"] == b["DATETIME"].max()]
    for c in ["pm25_roll_mean_6", "pm25_roll_std_6", "pm25_roll_mean_72"]:
        assert last_a[c].iloc[0] == last_b[c].iloc[0]


def test_baseline_fallback_chain():
    tr = parse_times(_frame(48), 3)
    bl = SeasonalBaseline().fit(tr)
    new = parse_times(_frame(3, hood="UNSEEN"), 3)
    out = bl.transform(new)
    assert np.allclose(out["baseline"], tr["PM25"].median())
