"""FastAPI service: send hourly sensor readings, get anomaly flags, severity, likely cause and SHAP reasons.

Run locally:   uvicorn api.main:app --reload --port 8000
Docs:          http://localhost:8000/docs
"""
from __future__ import annotations

import io
import json
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from airq import __version__
from airq.config import load_config, resolve
from airq.scoring import OUTPUT_COLUMNS, AnomalyScorer

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["scorer"] = AnomalyScorer(models_dir=os.environ.get("AIRQ_MODELS_DIR"))
    yield
    STATE.clear()


app = FastAPI(
    title="Nairobi Air-Quality Anomaly API",
    version=__version__,
    description=("Scores hourly low-cost sensor readings from Nairobi. Each reading gets an anomaly "
                 "probability, a flag, a severity tier (Watch / Alert / Critical), a likely cause "
                 "(genuine event / likely sensor fault / unverifiable) and, on request, a plain-language "
                 "SHAP explanation. Send at least the previous 72 hours per sensor so rolling features "
                 "are complete, and all sensors for the same hours so neighbour evidence is available."),
    lifespan=lifespan,
)


# ----------------------------------------------------------------- schemas
class Reading(BaseModel):
    LOCATION: int = Field(..., description="sensors.AFRICA location id", examples=[3573])
    NEIGHBORHOOD: str = Field(..., examples=["Syokimau-Mulolongo ward"])
    DATETIME: datetime = Field(..., description="Start of the hour, UTC (as in the archive)",
                               examples=["2025-01-15T05:00:00"])
    PM25: Optional[float] = Field(None, examples=[34.2])
    PM10: Optional[float] = Field(None, examples=[51.0])
    PM1: Optional[float] = None
    HUMIDITY: Optional[float] = Field(None, examples=[71.5])
    TEMPERATURE: Optional[float] = Field(None, examples=[19.4])
    NEIGHBORHOOD_LAT: Optional[float] = None
    NEIGHBORHOOD_LON: Optional[float] = None


class SeverityOverrides(BaseModel):
    critical_z: Optional[float] = None
    alert_z: Optional[float] = None
    critical_pm25: Optional[float] = None
    alert_pm25: Optional[float] = None


class PredictRequest(BaseModel):
    readings: list[Reading] = Field(..., min_length=1)
    threshold: Optional[float] = Field(None, ge=0, le=1, description="Override the frozen validation threshold")
    explain: bool = Field(False, description="Add SHAP top drivers and a plain-language reason to flags")
    only_flagged: bool = False
    severity: Optional[SeverityOverrides] = None


class Driver(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str
    text: str


class Prediction(BaseModel):
    LOCATION: int
    NEIGHBORHOOD: str
    DATETIME_UTC: datetime
    DATETIME_LOCAL: datetime
    PM25: Optional[float]
    PM10: Optional[float]
    baseline: Optional[float]
    residual: Optional[float]
    robust_z: Optional[float]
    anomaly_probability: Optional[float]
    is_anomaly: int
    rule: str
    severity: Literal["Normal", "Watch", "Alert", "Critical"]
    cause_label: str
    cause_model_guess: str
    cause_model_confidence: Optional[float]
    sensors_in_neighbourhood: Optional[int]
    humidity_artefact_risk: int
    route_to: str
    explanation: Optional[str] = None
    top_drivers: Optional[list[Driver]] = None


class PredictResponse(BaseModel):
    info: dict
    predictions: list[Prediction]


# ----------------------------------------------------------------- helpers
def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def _records(out: pd.DataFrame) -> list[dict]:
    cols = [c for c in OUTPUT_COLUMNS + ["explanation", "top_drivers"] if c in out.columns]
    recs = []
    for r in out[cols].to_dict(orient="records"):
        rec = {k: _clean(v) for k, v in r.items()}
        rec["DATETIME_LOCAL"] = rec.get("DATETIME")
        if rec.get("sensors_in_neighbourhood") is not None:
            rec["sensors_in_neighbourhood"] = int(rec["sensors_in_neighbourhood"])
        rec["cause_model_guess"] = rec.get("cause_model_guess") or ""
        rec["rule"] = rec.get("rule") or ""
        rec["route_to"] = rec.get("route_to") or ""
        rec["humidity_artefact_risk"] = int(rec.get("humidity_artefact_risk") or 0)
        if not isinstance(rec.get("top_drivers"), list):
            rec["top_drivers"] = None
        if not rec.get("explanation"):
            rec["explanation"] = None
        recs.append(rec)
    return recs


def _score(df: pd.DataFrame, threshold=None, explain=False, only_flagged=False, severity=None):
    try:
        out, info = STATE["scorer"].score(df, threshold=threshold, explain=explain, severity_cfg=severity)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if only_flagged:
        out = out[out["is_anomaly"] == 1]
    return out, info


# ----------------------------------------------------------------- routes
@app.get("/")
def root():
    return {"service": "nairobi-airq-anomaly", "version": __version__, "docs": "/docs"}


@app.get("/health")
def health():
    s = STATE.get("scorer")
    return {"status": "ok" if s else "loading", "model_loaded": s is not None,
            "threshold": s.threshold if s else None}


@app.get("/model-info")
def model_info():
    s = STATE["scorer"]
    info = {"model_config": s.model_config, "cause_model": s.cause.get("info", {}).get("status")}
    rep = resolve(load_config()["paths"]["reports_dir"]) / "metrics.json"
    if rep.exists():
        m = json.loads(rep.read_text())
        info["metrics"] = {"val": m["metrics"]["val"], "test": m["metrics"]["test"], "cv": m.get("cv")}
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([r.model_dump() for r in req.readings])
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], utc=True).dt.tz_localize(None)
    sev = req.severity.model_dump(exclude_none=True) if req.severity else None
    out, info = _score(df, req.threshold, req.explain, req.only_flagged, sev)
    return {"info": info, "predictions": _records(out)}


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(..., description="CSV or XLSX in the hourly dataset format"),
                       explain: bool = Query(False), only_flagged: bool = Query(False),
                       threshold: Optional[float] = Query(None, ge=0, le=1),
                       output: Literal["json", "csv"] = Query("json")):
    content = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content), compression="infer" if not name.endswith(".gz") else "gzip")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")
    if "TIME" in df.columns:
        df["TIME"] = df["TIME"].astype(str)
    out, info = _score(df, threshold, explain, only_flagged)
    if output == "csv":
        buf = io.StringIO()
        out.drop(columns=[c for c in ["top_drivers"] if c in out.columns]).to_csv(buf, index=False)
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=anomaly_scores.csv"})
    return {"info": info, "predictions": _records(out)}
