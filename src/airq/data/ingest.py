"""Rebuild the hourly Nairobi dataset from the raw sensors.AFRICA archive.

Raw format (one CSV per month on openAFRICA, ';'-separated, long format):

    sensor_id;sensor_type;location;lat;lon;timestamp;value_type;value
    2582;pms5003;3573;-1.270;36.810;2023-01-01T00:00:43.437733+00:00;P2;19.00

Steps
1. list / download the monthly CSVs (CKAN API) or use files already on disk
2. read each file in chunks, keep the five value types we use, coerce values to float32
3. aggregate to (LOCATION, hour) means using running sums and counts, so files can be
   processed one at a time and hours that straddle two files are still averaged correctly
4. record which sensor types reported at that location in that hour (SENSOR_TYPE)
5. take each location's coordinates (most frequent lat / lon in the archive)
6. reverse-geocode each location to a neighbourhood name (cached; see geocode.py)
7. add calendar columns and write the dataset in the original column order

The archive timestamps are UTC and the published dataset keeps UTC; local time is
added later in the modelling pipeline.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from airq.config import load_config, resolve
from airq.data import geocode

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "DATE", "TIME", "PM1", "PM25", "PM10", "HUMIDITY", "TEMPERATURE", "LOCATION",
    "NEIGHBORHOOD", "SENSOR_TYPE", "YEAR", "MONTH", "WEEKDAY", "HOUR",
    "NEIGHBORHOOD_LAT", "NEIGHBORHOOD_LON",
]
MEASURES = ["PM1", "PM25", "PM10", "HUMIDITY", "TEMPERATURE"]
RAW_LONG_COLUMNS = ["sensor_id", "sensor_type", "location", "lat", "lon", "timestamp", "value_type", "value"]

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


# ---------------------------------------------------------------------------
# 1. Discover and download the monthly archive files
# ---------------------------------------------------------------------------
def month_from_name(name: str) -> str | None:
    """Extract 'YYYY-MM' from a resource name or URL such as '..._2019_01.csv' or 'January 2019'."""
    s = name.lower()
    m = re.search(r"(20\d{2})[-_ ]?(0[1-9]|1[0-2])(?!\d)", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-_ ]*(20\d{2})", s)
    if m:
        return f"{m.group(2)}-{_MONTHS[m.group(1)]:02d}"
    return None


def month_range(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def list_archive_resources(cfg: dict | None = None, timeout: int = 60) -> pd.DataFrame:
    """Ask the openAFRICA CKAN API for the archive's CSV resources."""
    cfg = cfg or load_config()
    ing = cfg["ingest"]
    r = requests.get(ing["ckan_api"], params={"id": ing["ckan_package_id"]}, timeout=timeout)
    if r.status_code != 200:   # fall back to a search if the package id has changed
        search = ing["ckan_api"].replace("package_show", "package_search")
        r = requests.get(search, params={"q": "sensorsafrica nairobi archive", "rows": 5}, timeout=timeout)
        r.raise_for_status()
        results = r.json()["result"]["results"]
        if not results:
            raise RuntimeError("No sensors.AFRICA Nairobi archive found on openAFRICA")
        package = results[0]
    else:
        package = r.json()["result"]
    rows = []
    for res in package.get("resources", []):
        url = res.get("url", "")
        if not (str(res.get("format", "")).lower() == "csv" or url.lower().endswith((".csv", ".csv.gz"))):
            continue
        rows.append({"name": res.get("name", ""), "url": url,
                     "month": month_from_name(res.get("name", "")) or month_from_name(url)})
    out = pd.DataFrame(rows, columns=["name", "url", "month"])
    wanted = set(month_range(ing["start_month"], ing["end_month"]))
    return out[out["month"].isin(wanted)].sort_values("month").reset_index(drop=True)


def download_archive(cfg: dict | None = None, raw_dir: str | Path | None = None,
                     overwrite: bool = False, retries: int = 3) -> list[Path]:
    """Download every monthly CSV in the configured window into raw_dir (skips existing files)."""
    cfg = cfg or load_config()
    raw_dir = resolve(raw_dir or cfg["ingest"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    resources = list_archive_resources(cfg)
    log.info("Archive lists %d monthly files in the configured window", len(resources))
    paths = []
    for _, res in resources.iterrows():
        dest = raw_dir / f"sensorsafrica_nairobi_{res['month']}.csv"
        if dest.exists() and not overwrite:
            paths.append(dest)
            continue
        for attempt in range(1, retries + 1):
            try:
                with requests.get(res["url"], stream=True, timeout=300) as r:
                    r.raise_for_status()
                    tmp = dest.with_suffix(".part")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                    tmp.rename(dest)
                paths.append(dest)
                log.info("Downloaded %s", dest.name)
                break
            except requests.RequestException as e:
                log.warning("Attempt %d for %s failed: %s", attempt, res["url"], e)
                time.sleep(5 * attempt)
        else:
            log.error("Giving up on %s", res["url"])
    return paths


def find_raw_files(raw_dir: str | Path) -> list[Path]:
    raw_dir = resolve(raw_dir)
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file()
                   and p.name.lower().endswith((".csv", ".csv.gz", ".txt")))
    return files


# ---------------------------------------------------------------------------
# 2. Read one raw file into a tidy long table
# ---------------------------------------------------------------------------
def _sniff_sep(path: Path) -> str:
    opener = __import__("gzip").open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        head = f.readline()
    return ";" if head.count(";") >= head.count(",") else ","


def _to_long(chunk: pd.DataFrame, vmap: dict) -> pd.DataFrame:
    """Normalise a raw chunk (long or wide layout) to the tidy long layout."""
    chunk.columns = [c.strip().lower() for c in chunk.columns]
    if "value_type" not in chunk.columns:          # wide (sensor.community style): one column per value type
        keep = [k for k in vmap if k.lower() in chunk.columns]
        id_cols = [c for c in ["sensor_id", "sensor_type", "location", "lat", "lon", "timestamp"] if c in chunk]
        chunk = chunk.melt(id_vars=id_cols, value_vars=[k.lower() for k in keep],
                           var_name="value_type", value_name="value")
        lower = {k.lower(): k for k in vmap}
        chunk["value_type"] = chunk["value_type"].map(lower)
    return chunk


def read_raw_file(path: str | Path, cfg: dict | None = None, chunksize: int = 1_000_000) -> Iterable[pd.DataFrame]:
    """Yield tidy chunks: LOCATION, SENSOR_ID, SENSOR_TYPE, LAT, LON, HOUR_UTC, VARIABLE, VALUE."""
    cfg = cfg or load_config()
    vmap = cfg["ingest"]["value_type_map"]
    path = Path(path)
    sep = _sniff_sep(path)
    reader = pd.read_csv(path, sep=sep, dtype=str, chunksize=chunksize, on_bad_lines="skip",
                         encoding_errors="replace")
    for chunk in reader:
        chunk = _to_long(chunk, vmap)
        chunk = chunk[chunk["value_type"].isin(list(vmap))]
        if chunk.empty:
            continue
        chunk = chunk.drop_duplicates(subset=["sensor_id", "timestamp", "value_type", "value"])
        ts = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce", format="mixed")
        out = pd.DataFrame({
            "LOCATION": pd.to_numeric(chunk["location"], errors="coerce"),
            "SENSOR_ID": chunk["sensor_id"].astype(str),
            "SENSOR_TYPE": chunk["sensor_type"].astype(str).str.strip(),
            "LAT": pd.to_numeric(chunk["lat"], errors="coerce").astype("float32"),
            "LON": pd.to_numeric(chunk["lon"], errors="coerce").astype("float32"),
            "HOUR_UTC": ts.dt.floor("h").dt.tz_localize(None),
            "VARIABLE": chunk["value_type"].map(vmap),
            # sensor firmware reports single precision; float32 reproduces the published values
            "VALUE": pd.to_numeric(chunk["value"], errors="coerce").astype("float32"),
        })
        out = out.dropna(subset=["LOCATION", "HOUR_UTC", "VALUE"])
        out["LOCATION"] = out["LOCATION"].astype("int64")
        yield out


# ---------------------------------------------------------------------------
# 3-5. Aggregate to hourly means, sensor-type combos and coordinates
# ---------------------------------------------------------------------------
class HourlyAccumulator:
    """Running sums/counts so the archive can be processed one file at a time."""

    def __init__(self):
        self._sums: list[pd.DataFrame] = []
        self._types: list[pd.DataFrame] = []
        self._coords: list[pd.DataFrame] = []

    def add(self, tidy: pd.DataFrame) -> None:
        v = tidy["VALUE"].astype("float64")
        g = tidy.assign(VALUE=v).groupby(["LOCATION", "HOUR_UTC", "VARIABLE"], observed=True)["VALUE"]
        self._sums.append(g.agg(["sum", "count"]).reset_index())
        self._types.append(tidy[["LOCATION", "HOUR_UTC", "SENSOR_TYPE"]].drop_duplicates())
        self._coords.append(tidy.groupby(["LOCATION", "LAT", "LON"], dropna=False).size()
                            .rename("n").reset_index())
        self._compact()

    def _compact(self, every: int = 20) -> None:
        if len(self._sums) >= every:
            self._sums = [pd.concat(self._sums).groupby(["LOCATION", "HOUR_UTC", "VARIABLE"],
                                                        observed=True)[["sum", "count"]].sum().reset_index()]
            self._types = [pd.concat(self._types).drop_duplicates()]
            self._coords = [pd.concat(self._coords).groupby(["LOCATION", "LAT", "LON"], dropna=False)["n"]
                            .sum().reset_index()]

    def result(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self._sums:
            raise RuntimeError("No raw readings were read; check the raw directory and file format.")
        s = pd.concat(self._sums).groupby(["LOCATION", "HOUR_UTC", "VARIABLE"], observed=True)[["sum", "count"]].sum()
        means = (s["sum"] / s["count"]).unstack("VARIABLE")
        means = means.reindex(columns=MEASURES)
        types = (pd.concat(self._types).drop_duplicates()
                 .groupby(["LOCATION", "HOUR_UTC"])["SENSOR_TYPE"]
                 .agg(lambda x: ", ".join(sorted(set(x)))))
        hourly = means.join(types, how="left").reset_index()
        hourly = hourly.dropna(subset=MEASURES, how="all")
        coords = location_coordinates(pd.concat(self._coords))
        return hourly, coords


def location_coordinates(counts: pd.DataFrame) -> pd.DataFrame:
    """Most frequent latitude and longitude per location (each taken separately, NaN if never reported)."""
    out = {}
    for axis in ["LAT", "LON"]:
        c = counts.dropna(subset=[axis]).groupby(["LOCATION", axis])["n"].sum().reset_index()
        c = c.sort_values(["LOCATION", "n", axis], ascending=[True, False, True]).drop_duplicates("LOCATION")
        out[axis] = c.set_index("LOCATION")[axis]
    locs = pd.Index(counts["LOCATION"].unique(), name="LOCATION")
    return pd.DataFrame({"LAT": out["LAT"].reindex(locs), "LON": out["LON"].reindex(locs)}).reset_index()


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["HOUR_UTC"]
    df["DATE"] = dt.dt.normalize()
    df["TIME"] = dt.dt.strftime("%H:%M:%S")
    df["YEAR"] = dt.dt.year.astype("int64")
    df["MONTH"] = dt.dt.month.astype("int64")
    df["WEEKDAY"] = dt.dt.day_name()
    df["HOUR"] = dt.dt.hour.astype("int64")
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_dataset(raw_files: Iterable[str | Path], cfg: dict | None = None,
                  allow_network_geocoding: bool = True) -> pd.DataFrame:
    """Raw archive files -> hourly, geocoded dataset in the published column order."""
    cfg = cfg or load_config()
    acc = HourlyAccumulator()
    n_files = 0
    for path in raw_files:
        n_files += 1
        for tidy in read_raw_file(path, cfg):
            acc.add(tidy)
        log.info("Read %s", Path(path).name)
    log.info("Processed %d raw files", n_files)
    hourly, coords = acc.result()

    names = geocode.geocode_locations(coords, cfg, allow_network=allow_network_geocoding)
    hourly = hourly.merge(names, on="LOCATION", how="left")
    hourly = add_calendar(hourly)
    hourly = hourly.rename(columns={"LAT": "NEIGHBORHOOD_LAT", "LON": "NEIGHBORHOOD_LON"})
    for c in ["NEIGHBORHOOD_LAT", "NEIGHBORHOOD_LON"]:
        hourly[c] = hourly[c].astype("float64")
    hourly = hourly.sort_values(["HOUR_UTC", "LOCATION"], kind="stable").reset_index(drop=True)
    return hourly[OUTPUT_COLUMNS]


def save_dataset(df: pd.DataFrame, cfg: dict | None = None, xlsx: bool = True) -> dict:
    cfg = cfg or load_config()
    out = {}
    csv_path = resolve(cfg["ingest"]["output_csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, compression="gzip")
    out["csv"] = csv_path
    if xlsx:
        xlsx_path = resolve(cfg["ingest"]["output_xlsx"])
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        out["xlsx"] = xlsx_path
    return out


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the hourly dataset from .xlsx, .csv or .csv.gz with consistent dtypes."""
    path = resolve(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, parse_dates=["DATE"], float_precision="round_trip")
    df["TIME"] = df["TIME"].astype(str)
    return df
