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


# ------------------------------------------------------------------ website
def test_landing_and_analyzer_pages(client):
    for path, marker in [("/", "Is that spike"), ("/app", "Analyzer")]:
        r = client.get(path)
        assert r.status_code == 200 and marker in r.text
    for asset in ["/static/js/app.js", "/static/js/landing.js", "/static/css/site.css",
                  "/static/vendor/gsap.min.js", "/static/vendor/plotly.min.js", "/static/data/hero_replay.json",
                  "/static/fonts/public-sans-latin-400-normal.woff2"]:
        assert client.get(asset).status_code == 200, asset


def test_browser_rules_match_server(client):
    """The analyzer re-applies threshold, severity and cause in JavaScript; the same rules
    in Python must reproduce the server's own labels at the tuned threshold."""
    from api.main import STATE
    p = client.get("/api/samples/jul-aug-2024/score").json()
    rows = [dict(zip(p["fields"], r)) for r in p["rows"]]
    rules, s = p["rules"], p["rules"]["severity"]
    thr = rules["threshold"]

    def label(r):
        if r["rule"] == "pm_above_sensor_range":
            return True, "Critical", "sensor_fault_out_of_range"
        if r["anomaly_probability"] is None or r["anomaly_probability"] < thr:
            return False, "Normal", "not_flagged"
        z, pm = abs(r["robust_z"] or 0), r["PM25"] or 0
        sev = ("Critical" if z >= s["critical_z"] or pm >= s["critical_pm25"]
               else "Alert" if z >= s["alert_z"] or pm >= s["alert_pm25"] else "Watch")
        n = r["sensors_in_neighbourhood"]
        if n is None or n < 2 or r["nbr_resid"] is None:
            return True, sev, "unverifiable"
        own = r["residual"] or 0
        sign = (own > 0) - (own < 0)
        supported = sign * r["nbr_resid"] >= rules["nbr_support"] * max(abs(own), 1)
        return True, sev, "genuine" if supported else "likely_sensor_fault"

    import pandas as pd
    from airq.config import resolve
    raw = pd.read_csv(resolve("data/sample/demo_jul_aug_2024.csv"))
    raw["TIME"] = raw["TIME"].astype(str)
    out, _ = STATE["scorer"].score(raw)
    out = out.sort_values(["LOCATION", "DATETIME"]).reset_index(drop=True)
    assert len(out) == len(rows)
    mism = 0
    for r, (_, o) in zip(rows, out.iterrows()):
        f, sev, cause = label(r)
        # rounding of probabilities to 4 dp can only matter exactly at the threshold
        if abs((r["anomaly_probability"] or 0) - thr) < 1e-3:
            continue
        mism += (int(f) != int(o["is_anomaly"])) or (sev != o["severity"]) or (cause != o["cause_label"])
    assert mism == 0
