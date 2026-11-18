"""Streamlit dashboard (proposal stretch goal): a thin client of the FastAPI service.

    AIRQ_API_URL=http://localhost:8000 streamlit run dashboard/app.py
"""
import io
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.environ.get("AIRQ_API_URL", "http://localhost:8000")
SAMPLE = next((p for p in [Path("sample/sample_request.csv"), Path("data/sample/sample_request.csv")] if p.exists()), None)
SEV_COLORS = {"Critical": "#b2182b", "Alert": "#ef8a62", "Watch": "#fddbc7", "Normal": "#d1e5f0"}

st.set_page_config(page_title="Nairobi air-quality anomalies", layout="wide")
st.title("Nairobi air-quality anomaly monitor")
st.caption(f"Scores hourly low-cost sensor readings via the API at {API}. "
           "Times are shown in Nairobi local time (EAT, UTC+3).")

with st.sidebar:
    try:
        h = requests.get(f"{API}/health", timeout=5).json()
        st.success(f"API ok · threshold {h['threshold']:.3f}")
    except Exception as e:
        st.error(f"API not reachable: {e}")
        st.stop()
    up = st.file_uploader("Hourly readings (CSV / XLSX, dataset format, UTC)", type=["csv", "xlsx", "gz"])
    use_sample = st.checkbox("Use bundled sample", value=up is None and SAMPLE is not None)
    explain = st.checkbox("Explain flags with SHAP", value=True)
    thr = st.slider("Probability cutoff", 0.05, 0.999, float(h["threshold"]), 0.001)


@st.cache_data(show_spinner="Scoring readings...")
def score(content: bytes, name: str, explain: bool, thr: float) -> pd.DataFrame:
    r = requests.post(f"{API}/predict/file", files={"file": (name, content)},
                      params={"explain": explain, "threshold": thr}, timeout=600)
    r.raise_for_status()
    return pd.DataFrame(r.json()["predictions"])


if up is not None:
    content, name = up.getvalue(), up.name
elif use_sample and SAMPLE:
    content, name = SAMPLE.read_bytes(), SAMPLE.name
else:
    st.info("Upload a file or tick 'Use bundled sample'.")
    st.stop()

df = score(content, name, explain, thr)
df["DATETIME_LOCAL"] = pd.to_datetime(df["DATETIME_LOCAL"])
flags = df[df["is_anomaly"] == 1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Readings scored", f"{len(df):,}")
c2.metric("Flagged", f"{len(flags):,}", f"{len(flags) / max(len(df), 1):.2%}")
c3.metric("Critical", int((flags["severity"] == "Critical").sum()))
c4.metric("Likely sensor faults", int(flags["cause_label"].isin(["likely_sensor_fault", "sensor_fault_out_of_range"]).sum()))

tab1, tab2, tab3 = st.tabs(["Time series", "Flagged readings", "Sensor health"])
with tab1:
    sensors = sorted(df["LOCATION"].unique())
    default = flags["LOCATION"].mode().iloc[0] if len(flags) else sensors[0]
    s = st.selectbox("Sensor", sensors, index=sensors.index(default))
    d = df[df["LOCATION"] == s].sort_values("DATETIME_LOCAL")
    fig = px.line(d, x="DATETIME_LOCAL", y=["PM25", "baseline"], labels={"value": "µg/m³"})
    f = d[d["is_anomaly"] == 1]
    fig.add_scatter(x=f["DATETIME_LOCAL"], y=f["PM25"], mode="markers", name="flagged",
                    marker=dict(color="crimson", size=8), text=f["severity"] + " · " + f["cause_label"])
    st.plotly_chart(fig, width='stretch')
with tab2:
    show = ["DATETIME_LOCAL", "LOCATION", "NEIGHBORHOOD", "PM25", "baseline", "residual", "severity",
            "cause_label", "cause_model_guess", "humidity_artefact_risk", "route_to"]
    if "explanation" in flags:
        show.append("explanation")
    st.dataframe(flags.sort_values(["severity", "DATETIME_LOCAL"])[show], width='stretch')
    st.plotly_chart(px.histogram(flags, x="severity", color="cause_label", barmode="stack",
                                 category_orders={"severity": ["Watch", "Alert", "Critical"]}),
                    width='stretch')
    st.download_button("Download flagged readings (CSV)", flags.drop(columns=["top_drivers"], errors="ignore")
                       .to_csv(index=False), "flagged_readings.csv")
with tab3:
    health = (df.groupby("LOCATION")
                .agg(neighbourhood=("NEIGHBORHOOD", "first"), readings=("is_anomaly", "size"),
                     flag_rate=("is_anomaly", "mean"), last_seen=("DATETIME_LOCAL", "max"),
                     faults=("cause_label", lambda c: c.isin(["likely_sensor_fault", "sensor_fault_out_of_range"]).sum()))
                .sort_values("flag_rate", ascending=False))
    st.dataframe(health.style.format({"flag_rate": "{:.2%}"}), width='stretch')
