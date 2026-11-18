"""Web front end: landing page, interactive analyzer, and the JSON endpoints they use.

The analyzer scores data once on the server (at a probability floor of 0.5, with SHAP
explanations for every reading above that floor). The browser then re-applies the
threshold, severity and cause rules itself, so sliders respond instantly. Those rules
mirror airq.triage exactly.
"""
from __future__ import annotations

import io
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from airq.config import load_config, resolve

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
PROB_FLOOR = 0.5

SAMPLES = {
    "jul-aug-2024": {
        "file": "data/sample/demo_jul_aug_2024.csv",
        "title": "July–August 2024",
        "summary": "7 sensors over a month with genuine events, faults and unverifiable spikes",
    },
    "sep-2024": {
        "file": "data/sample/sample_request.csv",
        "title": "1–14 September 2024",
        "summary": "Includes sensor 3626 reporting beyond its measuring range for days",
    },
}

ROW_FIELDS = ["LOCATION", "NEIGHBORHOOD", "DATETIME", "PM25", "PM10", "HUMIDITY", "TEMPERATURE",
              "baseline", "residual", "robust_z", "anomaly_probability", "rule", "nbr_resid",
              "sensors_in_neighbourhood", "humidity_artefact_risk", "cause_model_guess",
              "cause_model_confidence", "NEIGHBORHOOD_LAT", "NEIGHBORHOOD_LON"]

router = APIRouter()


def _num(v, nd=3):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def _payload(scorer, raw: pd.DataFrame) -> dict:
    """Score once at the probability floor and return a compact table for the browser."""
    out, info = scorer.score(raw, threshold=PROB_FLOOR, explain=True, top_k=len(scorer.meta_features))
    out = out.sort_values(["LOCATION", "DATETIME"]).reset_index(drop=True)
    for c in ["NEIGHBORHOOD_LAT", "NEIGHBORHOOD_LON"]:
        if c not in out.columns:
            hood = raw.groupby("LOCATION")[c].first() if c in raw.columns else None
            out[c] = out["LOCATION"].map(hood) if hood is not None else np.nan
    rows, explanations = [], {}
    for i, r in enumerate(out.to_dict(orient="records")):
        row = []
        for f in ROW_FIELDS:
            v = r.get(f)
            if f == "DATETIME":
                row.append(pd.Timestamp(v).strftime("%Y-%m-%dT%H:%M"))
            elif f in ("LOCATION", "humidity_artefact_risk"):
                row.append(int(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0)
            elif f in ("NEIGHBORHOOD", "rule", "cause_model_guess"):
                row.append(v if isinstance(v, str) else "")
            elif f == "sensors_in_neighbourhood":
                row.append(None if v is None or (isinstance(v, float) and math.isnan(v)) else int(v))
            elif f in ("NEIGHBORHOOD_LAT", "NEIGHBORHOOD_LON"):
                row.append(_num(v, 4))
            else:
                row.append(_num(v, 4 if f == "anomaly_probability" else 2))
        rows.append(row)
        drivers = r.get("top_drivers")
        if isinstance(drivers, list) and drivers:
            explanations[i] = {"text": r.get("explanation", ""),
                               "drivers": [{"feature": d["feature"], "shap": round(d["shap"], 3),
                                            "text": d["text"]} for d in drivers]}
    mc = scorer.model_config
    return {
        "fields": ROW_FIELDS,
        "rows": rows,
        "explanations": explanations,
        "rules": {"threshold": mc["threshold"], "prob_floor": PROB_FLOOR,
                  "nbr_support": mc["nbr_support"], "severity": scorer.cfg["severity"],
                  "pm_sensor_max": mc["pm_sensor_max"]},
        "info": {"rows_received": info["rows_received"], "rows_scored": info["rows_scored"],
                 "cleaning": info["cleaning"]},
    }


def mount(app, get_scorer):
    """Attach the web routes to the FastAPI app."""

    @router.get("/", include_in_schema=False)
    def landing():
        return FileResponse(WEB_DIR / "index.html")

    @router.get("/app", include_in_schema=False)
    def analyzer():
        return FileResponse(WEB_DIR / "app.html")

    @router.get("/api/overview", tags=["web"])
    def overview():
        """Headline numbers for the landing page, taken from the training report."""
        mc = get_scorer().model_config
        out = {"threshold": mc["threshold"], "features": mc["meta_features"], "trained_at": mc["trained_at"]}
        rep = resolve(load_config()["paths"]["reports_dir"]) / "metrics.json"
        if rep.exists():
            m = json.loads(rep.read_text())
            out["splits"] = m["splits"]
            out["metrics"] = {k: {kk: m["metrics"][k].get(kk) for kk in
                                  ["n", "positives", "precision", "recall", "f1", "roc_auc", "pr_auc"]}
                              for k in ["val", "test"]}
            out["cv"] = m.get("cv")
            out["cleaning"] = m.get("cleaning")
        return out

    @router.get("/api/samples", tags=["web"])
    def samples():
        return [{"id": k, "title": v["title"], "summary": v["summary"]} for k, v in SAMPLES.items()]

    @lru_cache(maxsize=4)
    def _sample_payload(sid: str) -> str:
        raw = pd.read_csv(resolve(SAMPLES[sid]["file"]))
        raw["TIME"] = raw["TIME"].astype(str)
        return json.dumps(_payload(get_scorer(), raw))

    @router.get("/api/samples/{sid}/score", tags=["web"])
    def score_sample(sid: str):
        """Score a bundled sample (cached after the first call)."""
        if sid not in SAMPLES:
            raise HTTPException(404, f"Unknown sample '{sid}'. Choose one of {list(SAMPLES)}.")
        return json.loads(_sample_payload(sid))

    @router.post("/api/score", tags=["web"])
    async def score_upload(file: UploadFile = File(..., description="CSV, CSV.GZ or XLSX in the hourly dataset format")):
        """Score an uploaded file for the analyzer."""
        content = await file.read()
        name = (file.filename or "").lower()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(413, "File is larger than 25 MB. Split it into shorter periods and upload again.")
        try:
            if name.endswith((".xlsx", ".xls")):
                raw = pd.read_excel(io.BytesIO(content))
            else:
                raw = pd.read_csv(io.BytesIO(content), compression="gzip" if name.endswith(".gz") else None)
        except Exception as e:
            raise HTTPException(400, f"Could not read the file as CSV or Excel: {e}")
        if "TIME" in raw.columns:
            raw["TIME"] = raw["TIME"].astype(str)
        max_rows = int(os.environ.get("AIRQ_MAX_UPLOAD_ROWS", "20000"))
        if len(raw) > max_rows:
            raise HTTPException(413, f"The file has {len(raw):,} rows; this server scores up to {max_rows:,} at once. "
                                     "Upload about a month at a time.")
        try:
            return _payload(get_scorer(), raw)
        except ValueError as e:
            raise HTTPException(422, str(e))

    app.include_router(router)
