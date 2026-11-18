"""API smoke tests (need trained artifacts in models/)."""
import json

import pytest
from fastapi.testclient import TestClient

from airq.config import resolve

pytestmark = pytest.mark.skipif(not resolve("models/meta_model.joblib").exists(),
                                reason="train the model first: python -m airq.pipeline train")


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_json_with_explanations(client):
    body = json.loads(resolve("data/sample/sample_request.json").read_text())
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert len(preds) == len(body["readings"])
    flagged = [p for p in preds if p["is_anomaly"] == 1]
    assert flagged, "the sample week contains a known pollution spike"
    assert all(p["severity"] in {"Watch", "Alert", "Critical"} for p in flagged)
    assert all(p["explanation"] and len(p["top_drivers"]) == 3 for p in flagged if p["rule"] == "meta_model")


def test_predict_file_handles_out_of_range_faults(client):
    path = resolve("data/sample/sample_request.csv")
    with open(path, "rb") as f:
        r = client.post("/predict/file", files={"file": ("sample.csv", f, "text/csv")},
                        params={"only_flagged": True})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    faults = [p for p in preds if p["rule"] == "pm_above_sensor_range"]
    assert faults and all(p["cause_label"] == "sensor_fault_out_of_range" and p["severity"] == "Critical"
                          for p in faults)


def test_missing_columns_rejected(client):
    r = client.post("/predict/file", files={"file": ("bad.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422
