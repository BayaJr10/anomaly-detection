"""Compare a rebuilt hourly dataset with the published reference file."""
from __future__ import annotations

import numpy as np
import pandas as pd

from airq.data.ingest import MEASURES, OUTPUT_COLUMNS

KEY = ["DATE", "TIME", "LOCATION"]


def schema_report(df: pd.DataFrame) -> dict:
    return {
        "columns_match": list(df.columns) == OUTPUT_COLUMNS,
        "missing_columns": [c for c in OUTPUT_COLUMNS if c not in df.columns],
        "rows": int(len(df)),
        "duplicate_keys": int(df.duplicated(KEY).sum()),
        "all_measures_missing_rows": int(df[MEASURES].isna().all(axis=1).sum()),
    }


def compare(rebuilt: pd.DataFrame, reference: pd.DataFrame, rtol: float = 1e-4, atol: float = 1e-3) -> dict:
    """Row coverage and value agreement on the shared (DATE, TIME, LOCATION) keys."""
    a, b = rebuilt.copy(), reference.copy()
    for d in (a, b):
        d["DATE"] = pd.to_datetime(d["DATE"]).dt.normalize()
        d["TIME"] = d["TIME"].astype(str)
    m = a.merge(b, on=KEY, how="outer", suffixes=("_new", "_ref"), indicator=True)
    out = {
        "rows_rebuilt": int(len(a)), "rows_reference": int(len(b)),
        "rows_matched": int((m["_merge"] == "both").sum()),
        "only_in_rebuilt": int((m["_merge"] == "left_only").sum()),
        "only_in_reference": int((m["_merge"] == "right_only").sum()),
        "columns": {},
    }
    both = m[m["_merge"] == "both"]
    for c in MEASURES + ["NEIGHBORHOOD_LAT", "NEIGHBORHOOD_LON"]:
        x, y = both[f"{c}_new"].astype(float), both[f"{c}_ref"].astype(float)
        nan_mismatch = int((x.isna() != y.isna()).sum())
        ok = x.notna() & y.notna()
        close = np.isclose(x[ok], y[ok], rtol=rtol, atol=atol)
        out["columns"][c] = {
            "nan_mismatches": nan_mismatch,
            "share_equal": float(close.mean()) if ok.any() else 1.0,
            "max_abs_diff": float(np.abs(x[ok] - y[ok]).max()) if ok.any() else 0.0,
        }
    for c in ["NEIGHBORHOOD", "SENSOR_TYPE", "WEEKDAY", "HOUR", "YEAR", "MONTH"]:
        out["columns"][c] = {"share_equal": float((both[f"{c}_new"].astype(str) == both[f"{c}_ref"].astype(str)).mean())
                             if len(both) else 1.0}
    return out


def print_report(rep: dict) -> None:
    print(f"rows rebuilt={rep['rows_rebuilt']:,}  reference={rep['rows_reference']:,}  "
          f"matched={rep['rows_matched']:,}  only-new={rep['only_in_rebuilt']:,}  "
          f"only-ref={rep['only_in_reference']:,}")
    for c, r in rep["columns"].items():
        print(f"  {c:18s} " + "  ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in r.items()))
