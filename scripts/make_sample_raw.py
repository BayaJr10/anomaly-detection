"""Write archive-format raw CSVs that aggregate back to known hourly values.

Used for (a) an offline demo of the data pipeline and (b) the round-trip test in
tests/test_ingest.py. For each hourly row of the reference dataset in the chosen months,
it emits several raw readings per variable whose mean equals the hourly value, spread
over the sensor types listed in SENSOR_TYPE, in the sensors.AFRICA long format.

    python scripts/make_sample_raw.py --months 2019-01 --out data/sample_raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from airq.config import load_config, resolve  # noqa: E402
from airq.data.ingest import load_dataset  # noqa: E402

CAPABLE = {  # which sensor type normally reports which variable
    "PM1": ["pms5003"], "PM25": ["pms5003", "SDS011"], "PM10": ["pms5003", "SDS011"],
    "HUMIDITY": ["DHT22", "pms5003"], "TEMPERATURE": ["DHT22", "pms5003"],
}
CODE = {"PM1": "P0", "PM25": "P2", "PM10": "P1", "HUMIDITY": "humidity", "TEMPERATURE": "temperature"}
OFFSETS = np.array([-1.0, 0.0, 1.0])     # zero-sum spread -> mean of the raw readings == hourly value


def emit(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in df.itertuples(index=False):
        types = [t.strip() for t in str(r.SENSOR_TYPE).split(",")]
        hour = pd.Timestamp(f"{pd.Timestamp(r.DATE):%Y-%m-%d} {r.TIME}")
        assigned = {t: [] for t in types}
        for var in CODE:
            val = getattr(r, var)
            if pd.isna(val):
                continue
            who = [t for t in CAPABLE[var] if t in types][:1] or types
            for t in who:
                assigned[t].append((var, float(val)))
        for t in types:                      # every listed type must report something in that hour
            if not assigned[t]:
                assigned[t] = [(v, float(getattr(r, v))) for v in CODE if not pd.isna(getattr(r, v))][:1]
        for k, t in enumerate(types):
            sensor_id = int(r.LOCATION) * 10 + k
            for var, val in assigned[t]:
                spread = min(abs(val) * 0.25, 2.0)
                minutes = np.sort(rng.choice(np.arange(0, 60, 2.5), size=3, replace=False))
                for off, mnt in zip(OFFSETS, minutes):
                    ts = hour + pd.Timedelta(minutes=float(mnt))
                    rows.append((sensor_id, t, int(r.LOCATION),
                                 "" if pd.isna(r.NEIGHBORHOOD_LAT) else f"{r.NEIGHBORHOOD_LAT:.3f}",
                                 f"{r.NEIGHBORHOOD_LON:.3f}",
                                 ts.strftime("%Y-%m-%dT%H:%M:%S.000000+00:00"),
                                 CODE[var], f"{val + off * spread:.6f}"))
    return pd.DataFrame(rows, columns=["sensor_id", "sensor_type", "location", "lat", "lon",
                                       "timestamp", "value_type", "value"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=["2019-01"])
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", default="data/sample_raw")
    a = ap.parse_args()
    ref = load_dataset(a.reference or load_config()["data_path"])
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in a.months:
        part = ref[ref["DATE"].dt.strftime("%Y-%m") == m]
        raw = emit(part)
        path = out / f"sensorsafrica_nairobi_{m}.csv.gz"
        raw.to_csv(path, sep=";", index=False, compression="gzip")
        print(f"{m}: {len(part):,} hourly rows -> {len(raw):,} raw readings -> {path}")


if __name__ == "__main__":
    main()
