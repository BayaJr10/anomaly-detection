"""Data pipeline: raw archive format -> hourly dataset."""
import gzip
from pathlib import Path

import pandas as pd
import pytest

from airq.config import load_config, resolve
from airq.data import ingest, validate

SAMPLE = resolve("data/sample_raw/sensorsafrica_nairobi_2025-11.csv.gz")
REFERENCE = resolve(load_config()["data_path"])


def test_month_from_name():
    assert ingest.month_from_name("sensorsAFRICA_nairobi_2019_01.csv") == "2019-01"
    assert ingest.month_from_name("https://x/archive-2024-12.csv") == "2024-12"
    assert ingest.month_from_name("Nairobi air quality January 2021") == "2021-01"
    assert ingest.month_from_name("readme.txt") is None


def test_long_format_hourly_mean_and_sensor_types(tmp_path):
    raw = tmp_path / "m.csv"
    raw.write_text(
        "sensor_id;sensor_type;location;lat;lon;timestamp;value_type;value\n"
        "1;SDS011;7;-1.298;36.791;2020-03-01T10:05:00+00:00;P2;10\n"
        "1;SDS011;7;-1.298;36.791;2020-03-01T10:35:00+00:00;P2;20\n"
        "1;SDS011;7;-1.298;36.791;2020-03-01T10:35:00+00:00;P2;20\n"      # exact duplicate -> dropped
        "1;SDS011;7;-1.298;36.791;2020-03-01T10:35:00+00:00;P1;30\n"
        "2;DHT22;7;-1.298;36.791;2020-03-01T10:40:00+00:00;humidity;70\n"
        "2;DHT22;7;-1.298;36.791;2020-03-01T10:40:00+00:00;noise_LAeq;55\n"  # unused value type -> ignored
        "2;DHT22;7;-1.298;36.791;2020-03-01T11:01:00+00:00;temperature;21\n")
    df = ingest.build_dataset([raw], allow_network_geocoding=False)
    assert list(df.columns) == ingest.OUTPUT_COLUMNS
    assert len(df) == 2
    r = df.iloc[0]
    assert r["PM25"] == pytest.approx(15.0) and r["PM10"] == pytest.approx(30.0)
    assert r["HUMIDITY"] == pytest.approx(70.0) and pd.isna(r["PM1"])
    assert r["SENSOR_TYPE"] == "DHT22, SDS011" and r["TIME"] == "10:00:00"
    assert df.iloc[1]["SENSOR_TYPE"] == "DHT22"
    assert r["NEIGHBORHOOD"] == "Kilimani sublocation"       # from the frozen geocode cache


def test_wide_format_is_supported(tmp_path):
    raw = tmp_path / "w.csv"
    raw.write_text("sensor_id;sensor_type;location;lat;lon;timestamp;P1;P2\n"
                   "5;SDS011;7;-1.298;36.791;2020-03-01T10:05:00+00:00;30;12\n")
    df = ingest.build_dataset([raw], allow_network_geocoding=False)
    assert df.iloc[0]["PM25"] == pytest.approx(12.0) and df.iloc[0]["PM10"] == pytest.approx(30.0)


@pytest.mark.skipif(not (SAMPLE.exists() and REFERENCE.exists()), reason="sample raw or reference missing")
def test_round_trip_reproduces_published_rows():
    new = ingest.build_dataset([SAMPLE], allow_network_geocoding=False)
    ref = ingest.load_dataset(REFERENCE)
    ref = ref[ref["DATE"].dt.strftime("%Y-%m") == "2025-11"]
    rep = validate.compare(new, ref)
    assert rep["rows_matched"] == len(ref) == len(new)
    for c in ["PM1", "PM25", "PM10", "HUMIDITY", "TEMPERATURE"]:
        assert rep["columns"][c]["nan_mismatches"] == 0 and rep["columns"][c]["share_equal"] == 1.0
    for c in ["NEIGHBORHOOD", "SENSOR_TYPE", "WEEKDAY", "HOUR"]:
        assert rep["columns"][c]["share_equal"] == 1.0
    assert (new["NEIGHBORHOOD"] == "No coordinates").any()        # location without latitude
