"""Standalone Streamlit app for Nairobi air-quality anomaly detection.

Loads the trained models from models/ and scores uploaded hourly readings directly
(no separate API needed), using the same AnomalyScorer as the FastAPI service and the
batch CLI. Deploy on Streamlit Community Cloud with main file: streamlit_app/app.py

    streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AIRQ_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from airq.config import load_config, resolve  # noqa: E402
from airq.scoring import AnomalyScorer  # noqa: E402

SAMPLES = {
    "Jul–Aug 2024 · mixed events and faults (7 sensors)": ROOT / "data" / "sample" / "demo_jul_aug_2024.csv",
    "1–14 Sep 2024 · sensor 3626 out-of-range fault episode": ROOT / "data" / "sample" / "sample_request.csv",
}
SEV_ORDER = ["Critical", "Alert", "Watch"]
SEV_COLORS = {"Critical": "#b2182b", "Alert": "#ef8a62", "Watch": "#f4c542", "Normal": "#9ecae1"}
CAUSE_TEXT = {
    "genuine": "Genuine event (neighbours saw it too)",
    "likely_sensor_fault": "Likely sensor fault (neighbours did not see it)",
    "sensor_fault_out_of_range": "Sensor fault (reading beyond the sensor's measuring range)",
    "unverifiable": "Unverifiable (no neighbouring sensor reported)",
}

st.set_page_config(page_title="Nairobi Air-Quality Anomaly Detection", page_icon="🌫️", layout="wide")


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading trained models...")
def get_scorer() -> AnomalyScorer:
    return AnomalyScorer(cfg=load_config())


@st.cache_data(show_spinner=False)
def read_upload(content: bytes, name: str) -> pd.DataFrame:
    name = name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content), compression="gzip" if name.endswith(".gz") else None)
    if "TIME" in df.columns:
        df["TIME"] = df["TIME"].astype(str)
    return df


@st.cache_data(show_spinner="Scoring readings (cleaning, features, baseline, model)...")
def score(content: bytes, name: str, threshold: float, sev: tuple) -> tuple[pd.DataFrame, dict]:
    raw = read_upload(content, name)
    sev_cfg = dict(zip(["critical_z", "alert_z", "critical_pm25", "alert_pm25"], sev))
    out, info = get_scorer().score(raw, threshold=threshold, explain=False, severity_cfg=sev_cfg)
    out["DATETIME"] = pd.to_datetime(out["DATETIME"])
    return out, info


@st.cache_data(show_spinner="Computing SHAP explanations...")
def explain_rows(content: bytes, name: str, threshold: float, sev: tuple, keys: tuple) -> list[dict]:
    out, _ = score(content, name, threshold, sev)
    rows = out.set_index(["LOCATION", "DATETIME"]).loc[list(keys)].reset_index()
    return get_scorer().explain(rows, top_k=len(get_scorer().meta_features))


try:
    scorer = get_scorer()
except FileNotFoundError:
    st.error("No trained models in models/. Run `python -m airq.pipeline train` and commit models/.")
    st.stop()
cfg = scorer.cfg

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Data")
    up = st.file_uploader("Hourly readings (CSV, CSV.GZ or XLSX)", type=["csv", "gz", "xlsx"],
                          help="Same columns as the hourly dataset. Times are UTC, as in the sensors.AFRICA archive.")
    use_sample = st.toggle("Use a bundled sample", value=up is None)
    sample_pick = st.selectbox("Sample", list(SAMPLES), disabled=not use_sample or up is not None)
    st.header("2 · Detection")
    thr = st.slider("Anomaly probability cutoff", 0.50, 0.999, float(scorer.threshold), 0.001,
                    help=f"Default {scorer.threshold:.3f} was chosen on the 2024 validation set (max F1).")
    st.header("3 · Severity tiers")
    s = cfg["severity"]
    crit_z = st.slider("Critical: robust z ≥", 3.0, 30.0, float(s["critical_z"]), 0.5)
    alert_z = st.slider("Alert: robust z ≥", 2.0, 20.0, float(s["alert_z"]), 0.5)
    crit_pm = st.number_input("Critical: PM2.5 ≥ (µg/m³)", 0.0, 1000.0, float(s["critical_pm25"]), 5.0)
    alert_pm = st.number_input("Alert: PM2.5 ≥ (µg/m³)", 0.0, 1000.0, float(s["alert_pm25"]), 5.0)
    st.caption("A flag is Critical if EITHER test passes, Alert likewise; otherwise Watch. "
               "PM2.5 bands follow the US EPA 2024 AQI breakpoints.")
    sev = (crit_z, alert_z, crit_pm, alert_pm)

st.title("Nairobi air-quality anomaly detection")
st.caption("Explainable ensemble anomaly detection for the sensors.AFRICA low-cost PM2.5 network · "
           "Baya Isaac · DSA 8401 Capstone, Strathmore University")

if up is not None:
    content, name = up.getvalue(), up.name
elif use_sample and SAMPLES[sample_pick].exists():
    content, name = SAMPLES[sample_pick].read_bytes(), SAMPLES[sample_pick].name
else:
    st.info("Upload hourly readings or switch on the bundled sample in the sidebar.")
    with st.expander("Expected input format"):
        st.markdown("Required: `LOCATION`, `NEIGHBORHOOD`, `PM25`, `PM10`, and `DATETIME` (UTC) or `DATE` + `TIME` (UTC). "
                    "Optional: `PM1`, `HUMIDITY`, `TEMPERATURE`, `NEIGHBORHOOD_LAT`, `NEIGHBORHOOD_LON`. "
                    "Include the previous 72 hours per sensor for complete rolling features, and all sensors "
                    "for the same hours so neighbour evidence is available.")
    st.stop()

try:
    df, info = score(content, name, thr, sev)
except ValueError as e:
    st.error(str(e))
    st.stop()

flags = df[df["is_anomaly"] == 1].copy()
faults = flags["cause_label"].isin(["likely_sensor_fault", "sensor_fault_out_of_range"])

# ----------------------------------------------------------------------------- KPIs
k = st.columns(5)
k[0].metric("Readings scored", f"{len(df):,}", f"{df['LOCATION'].nunique()} sensors", delta_color="off")
k[1].metric("Flagged", f"{len(flags):,}", f"{len(flags) / max(len(df), 1):.2%} of readings", delta_color="off")
k[2].metric("Critical", int((flags["severity"] == "Critical").sum()))
k[3].metric("Sensor faults", int(faults.sum()))
k[4].metric("Genuine events", int((flags["cause_label"] == "genuine").sum()))
if info["cleaning"].get("pm_above_sensor_range"):
    n_rng = info['cleaning']['pm_above_sensor_range']
    st.warning(f"{n_rng} reading{'s' if n_rng != 1 else ''} exceed{'s' if n_rng == 1 else ''} the PM sensors' 1000 µg/m³ "
               "measuring limit. They are certain malfunctions, flagged Critical by rule and routed to maintenance.")

tabs = st.tabs(["📈 Time series", "🚩 Flagged readings", "🔍 Explain a flag", "🛠️ Sensor health", "📋 Model card"])

# ----------------------------------------------------------------------------- time series
with tabs[0]:
    sensors = sorted(int(x) for x in df["LOCATION"].unique())
    default = int(flags["LOCATION"].mode().iloc[0]) if len(flags) else int(sensors[0])
    c1, c2 = st.columns([1, 3])
    hood_of = df.groupby("LOCATION")["NEIGHBORHOOD"].first()
    sensor_labels = {f"{s_} · {hood_of[s_]}": s_ for s_ in sensors}
    sensor = sensor_labels[c1.selectbox("Sensor", list(sensor_labels), index=sensors.index(default))]
    log_y = c1.toggle("Log scale", value=bool((df.loc[df["LOCATION"] == sensor, "PM25"] > 300).any()))
    d = df[df["LOCATION"] == sensor].sort_values("DATETIME")
    fig = go.Figure()
    fig.add_scatter(x=d["DATETIME"], y=d["PM25"], mode="lines", name="PM2.5", line=dict(color="#4c72b0", width=1.4))
    fig.add_scatter(x=d["DATETIME"], y=d["baseline"], mode="lines", name="usual for this place & hour",
                    line=dict(color="grey", dash="dash", width=1))
    for sv in SEV_ORDER:
        f = d[(d["is_anomaly"] == 1) & (d["severity"] == sv)]
        if len(f):
            fig.add_scatter(x=f["DATETIME"], y=f["PM25"], mode="markers", name=sv,
                            marker=dict(color=SEV_COLORS[sv], size=9, line=dict(color="black", width=0.5)),
                            text=f["cause_label"].map(CAUSE_TEXT), hovertemplate="%{x}<br>PM2.5 %{y:.0f} µg/m³<br>%{text}<extra></extra>")
    fig.update_layout(height=430, hovermode="x unified", yaxis_title="PM2.5 (µg/m³)", xaxis_title="Nairobi local time",
                      yaxis_type="log" if log_y else "linear", legend=dict(orientation="h", y=1.08))
    c2.plotly_chart(fig, width="stretch")
    if len(flags):
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(flags, x="severity", color="cause_label", barmode="stack",
                                    category_orders={"severity": SEV_ORDER}, title="Flags by severity and likely cause"),
                       width="stretch")
        hm = df.pivot_table(index=df["DATETIME"].dt.day_name(), columns=df["DATETIME"].dt.hour,
                            values="is_anomaly", aggfunc="mean").reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        b.plotly_chart(px.imshow(hm, color_continuous_scale="Reds", aspect="auto",
                                 labels=dict(x="Local hour", y="", color="Flag rate"),
                                 title="Flag rate by weekday × hour"), width="stretch")

# ----------------------------------------------------------------------------- flagged table
with tabs[1]:
    if flags.empty:
        st.success("No readings flagged at this cutoff.")
    else:
        c1, c2, c3 = st.columns(3)
        sv_pick = c1.multiselect("Severity", SEV_ORDER, default=SEV_ORDER)
        causes = sorted(flags["cause_label"].unique())
        cause_pick = c2.multiselect("Likely cause", causes, default=causes)
        hood_pick = c3.multiselect("Neighbourhood", sorted(flags["NEIGHBORHOOD"].unique()))
        view = flags[flags["severity"].isin(sv_pick) & flags["cause_label"].isin(cause_pick)]
        if hood_pick:
            view = view[view["NEIGHBORHOOD"].isin(hood_pick)]
        view = view.assign(_o=view["severity"].map({s_: i for i, s_ in enumerate(SEV_ORDER)})) \
                   .sort_values(["_o", "anomaly_probability", "residual"], ascending=[True, False, False])
        cols = ["DATETIME", "LOCATION", "NEIGHBORHOOD", "PM25", "baseline", "residual", "robust_z",
                "anomaly_probability", "severity", "cause_label", "cause_model_guess", "humidity_artefact_risk", "route_to"]
        st.dataframe(view[cols], width="stretch", hide_index=True,
                     column_config={"DATETIME": st.column_config.DatetimeColumn("Local time", format="YYYY-MM-DD HH:mm"),
                                    "PM25": st.column_config.NumberColumn("PM2.5", format="%.1f"),
                                    "baseline": st.column_config.NumberColumn("Usual", format="%.1f"),
                                    "residual": st.column_config.NumberColumn("Excess", format="%+.1f"),
                                    "robust_z": st.column_config.NumberColumn("Robust z", format="%.1f"),
                                    "anomaly_probability": st.column_config.ProgressColumn("Probability", min_value=0, max_value=1, format="%.3f"),
                                    "humidity_artefact_risk": st.column_config.CheckboxColumn("High-RH risk")})
        st.caption(f"{len(view):,} of {len(flags):,} flags shown. Pick one in the 'Explain a flag' tab for its SHAP reasons.")
        st.download_button("⬇️ Download flagged readings (CSV)", view[cols].to_csv(index=False),
                           "flagged_readings.csv", "text/csv")
    st.download_button("⬇️ Download all scored readings (CSV)", df.drop(columns=["severity_ord"], errors="ignore")
                       .to_csv(index=False), "all_scores.csv", "text/csv")

# ----------------------------------------------------------------------------- explain
with tabs[2]:
    model_flags = flags[flags["rule"] == "meta_model"].sort_values("anomaly_probability", ascending=False)
    if model_flags.empty:
        st.info("No model-flagged readings to explain (out-of-range faults are flagged by rule, not by the model).")
    else:
        labels = {i: f"{r.DATETIME:%Y-%m-%d %H:%M} · sensor {r.LOCATION} · {r.NEIGHBORHOOD} · PM2.5 {r.PM25:.0f} · {r.severity}"
                  for i, r in model_flags.iterrows()}
        by_label = {v: i for i, v in labels.items()}
        pick = by_label[st.selectbox("Flagged reading", list(by_label))]
        r = model_flags.loc[pick]
        ex = explain_rows(content, name, thr, sev, ((r["LOCATION"], r["DATETIME"]),))[0]
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown(f"#### {r['severity']} · {CAUSE_TEXT.get(r['cause_label'], r['cause_label'])}")
            st.markdown(f"**Why flagged:** {ex['explanation']}")
            st.markdown(f"""
| | |
|---|---|
| Reading | **{r['PM25']:.1f}** µg/m³ PM2.5 |
| Usual for this place, hour & weekday | {r['baseline']:.1f} µg/m³ |
| Excess (residual) | {r['residual']:+.1f} µg/m³ · robust z {r['robust_z']:.1f} |
| Anomaly probability | {r['anomaly_probability']:.4f} |
| Sensors reporting in neighbourhood | {int(r['sensors_in_neighbourhood']) if pd.notna(r['sensors_in_neighbourhood']) else '–'} |
| Neighbours' mean excess | {'–' if pd.isna(r['nbr_resid']) else f"{r['nbr_resid']:+.1f} µg/m³"} |
| Model's cause guess | {r['cause_model_guess'] or '–'}{'' if pd.isna(r['cause_model_confidence']) else f" ({r['cause_model_confidence']:.0%}, advisory)"} |
| High-humidity artefact risk | {'yes' if r['humidity_artefact_risk'] else 'no'} |
| Route to | **{r['route_to']}** |
""")
        with c2:
            drv = pd.DataFrame(ex["drivers"]).iloc[::-1]
            import textwrap
            wrapped = ["<br>".join(textwrap.wrap(t_, 38)) for t_ in drv["text"]]
            fig = go.Figure(go.Bar(x=drv["shap"], y=wrapped, orientation="h",
                                   marker_color=np.where(drv["shap"] > 0, "#b2182b", "#2166ac"),
                                   text=[f"{v:+.2f}" for v in drv["shap"]], textposition="outside", cliponaxis=False))
            span = float(drv["shap"].abs().max()) or 1.0
            fig.update_layout(title="SHAP contribution to the anomaly score (log-odds)",
                              height=max(380, 62 * len(drv) + 110),
                              xaxis=dict(title="red pushes towards anomaly · blue towards normal",
                                         range=[min(0, drv["shap"].min()) - 0.25 * span, drv["shap"].max() + 0.3 * span],
                                         zeroline=True, zerolinecolor="#888"),
                              yaxis=dict(automargin=True, tickfont=dict(size=11)),
                              margin=dict(l=10, r=30, t=50, b=50), showlegend=False)
            st.plotly_chart(fig, width="stretch")
        win = df[(df["NEIGHBORHOOD"] == r["NEIGHBORHOOD"]) & ((df["DATETIME"] - r["DATETIME"]).abs() <= pd.Timedelta("48h"))]
        fig = px.line(win.sort_values("DATETIME"), x="DATETIME", y="PM25", color=win["LOCATION"].astype(str),
                      title=f"±48 h in {r['NEIGHBORHOOD']}: did neighbouring sensors see it?",
                      labels={"color": "sensor", "PM25": "PM2.5 (µg/m³)", "DATETIME": "local time"})
        fig.add_vline(x=r["DATETIME"], line_dash="dash", line_color="black")
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- sensor health
with tabs[3]:
    hood_of = df.groupby("LOCATION")["NEIGHBORHOOD"].first()
    health = (df.groupby("LOCATION")
                .agg(neighbourhood=("NEIGHBORHOOD", "first"), readings=("is_anomaly", "size"),
                     flags=("is_anomaly", "sum"), flag_rate=("is_anomaly", "mean"),
                     faults=("cause_label", lambda c: c.isin(["likely_sensor_fault", "sensor_fault_out_of_range"]).sum()),
                     median_pm25=("PM25", "median"), first_seen=("DATETIME", "min"), last_seen=("DATETIME", "max"))
                .reset_index().sort_values(["faults", "flag_rate"], ascending=False))
    st.dataframe(health, width="stretch", hide_index=True,
                 column_config={"flag_rate": st.column_config.ProgressColumn("Flag rate", min_value=0,
                                                                             max_value=max(health["flag_rate"].max(), 1e-9), format="percent"),
                                "median_pm25": st.column_config.NumberColumn("Median PM2.5", format="%.1f")})
    # PM2.5 distribution per sensor: log axis shows the whole distribution; labels sit on their own boxes
    order_s = df.groupby("LOCATION")["PM25"].median().sort_values(ascending=False).index
    lab = {s_: f"{s_} · {hood_of[s_]}" for s_ in order_s}
    dist = df.assign(sensor=df["LOCATION"].map(lab), PM25_plot=df["PM25"].clip(lower=0.5))
    fig = px.box(dist, x="PM25_plot", y="sensor", color=dist["is_anomaly"].map({0: "normal", 1: "flagged"}),
                 category_orders={"sensor": [lab[s_] for s_ in order_s]}, points="outliers", log_x=True,
                 color_discrete_map={"normal": "#4c72b0", "flagged": "#b2182b"},
                 labels={"PM25_plot": "PM2.5 (µg/m³, log scale)", "sensor": "", "color": ""},
                 title="PM2.5 distribution per sensor, normal vs flagged readings")
    fig.update_traces(width=0.55, selector=dict(name="normal"))
    fig.update_traces(width=0.3, selector=dict(name="flagged"))
    fig.update_layout(height=max(320, 64 * len(order_s) + 120), boxmode="overlay",
                      yaxis=dict(automargin=True), legend=dict(orientation="h", y=1.06), margin=dict(l=10, r=10, t=70, b=40))
    st.plotly_chart(fig, width="stretch")

    # Where the sensors are: a plain lon/lat scatter renders everywhere (no map tiles or WebGL needed)
    if {"NEIGHBORHOOD_LAT", "NEIGHBORHOOD_LON"} <= set(df.columns):
        geo = df.groupby("LOCATION").agg(lat=("NEIGHBORHOOD_LAT", "first"), lon=("NEIGHBORHOOD_LON", "first"),
                                         rate=("is_anomaly", "mean"), n=("is_anomaly", "size"),
                                         hood=("NEIGHBORHOOD", "first")).dropna().reset_index()
        if len(geo):
            geo["label"] = geo["LOCATION"].astype(str) + " · " + geo["hood"]
            fig = px.scatter(geo, x="lon", y="lat", size="n", color="rate", text="label", size_max=38,
                             color_continuous_scale="Reds", range_color=[0, max(geo["rate"].max(), 1e-9)],
                             hover_data={"rate": ":.2%", "n": ":,", "lon": ":.3f", "lat": ":.3f", "label": False},
                             labels={"lon": "longitude", "lat": "latitude", "rate": "flag rate", "n": "readings"},
                             title="Sensor locations (size = readings, colour = flag rate)")
            fig.update_traces(textposition="top center", textfont=dict(size=10),
                              marker=dict(line=dict(color="black", width=0.6)))
            pad_x = max(geo["lon"].max() - geo["lon"].min(), 0.02) * 0.25
            pad_y = max(geo["lat"].max() - geo["lat"].min(), 0.02) * 0.25
            fig.update_layout(height=520, xaxis=dict(range=[geo["lon"].min() - pad_x, geo["lon"].max() + pad_x]),
                              yaxis=dict(range=[geo["lat"].min() - pad_y, geo["lat"].max() + pad_y],
                                         scaleanchor="x", scaleratio=1), coloraxis_colorbar=dict(tickformat=".0%"))
            st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- model card
with tabs[4]:
    mc = scorer.model_config
    st.markdown(f"""
**Model:** SMOTE-balanced XGBoost meta-model trained to reproduce the consensus (≥ 3 of 5) of five unsupervised
detectors (MAD robust z, Local Outlier Factor, One-Class SVM, DBSCAN, STL) on 2019–2023 readings.

**Features ({len(mc['meta_features'])}):** {', '.join(f'`{f}`' for f in mc['meta_features'])}
**Frozen threshold:** {mc['threshold']:.4f} · **Trained:** {mc['trained_at'][:10]} · **Data SHA-256:** `{mc['data_sha256'][:16]}…`
""")
    rep = resolve(cfg["paths"]["reports_dir"]) / "metrics.json"
    if rep.exists():
        m = json.loads(rep.read_text())
        tbl = pd.DataFrame({sp: {k_: m["metrics"][sp].get(k_) for k_ in ["n", "positives", "precision", "recall", "f1", "roc_auc", "pr_auc"]}
                            for sp in ["val", "test"]}).rename(columns={"val": "Validation 2024", "test": "Test 2025–26"})
        st.markdown("**Anomaly-class metrics against the consensus pseudo-label**")
        st.dataframe(tbl.T.style.format({"n": "{:,.0f}", "positives": "{:,.0f}", "precision": "{:.3f}", "recall": "{:.3f}",
                                        "f1": "{:.3f}", "roc_auc": "{:.3f}", "pr_auc": "{:.3f}"}), width="stretch")
    st.info("**Read with care.** There are no ground-truth anomaly labels, so these numbers measure agreement with the "
            "detector consensus, not correctness. The neighbour-free cause model is close to chance in sensor-grouped "
            "cross-validation, so its guess for single-sensor flags is advisory only. Optical sensors over-read at "
            "high humidity; such flags carry a high-RH warning.")
