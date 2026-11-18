/* Hewa analyzer. The server scores once (probability floor 0.5); thresholds, severity and
   cause are re-applied here so every control responds instantly. The rules mirror
   src/airq/triage.py exactly. */
(() => {
  "use strict";
  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const SEV_ORDER = ["Watch", "Alert", "Critical"];
  const SEV_COLOR = { Watch: "#E3A73B", Alert: "#DD7034", Critical: "#B23A3F" };
  const CAUSE = {
    genuine: { label: "Genuine event", color: "#2F8F8B", route: "Environment / public-health officer",
      text: "Neighbouring sensors saw the same rise" },
    likely_sensor_fault: { label: "Likely sensor fault", color: "#6B4E9B", route: "Sensor maintenance team",
      text: "Neighbours reported, but did not see this rise" },
    sensor_fault_out_of_range: { label: "Beyond sensor range", color: "#3B3F6B", route: "Sensor maintenance team",
      text: "Reading is above the 1,000 µg/m³ the sensor can measure" },
    unverifiable: { label: "Can’t be checked", color: "#8C9AA1", route: "Data analyst for review",
      text: "No other sensor in the neighbourhood was reporting" },
  };
  const PLOT_FONT = { family: "Public Sans, system-ui, sans-serif", size: 13, color: "#23313B" };
  const PLOT_CFG = { responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"] };
  const fitHeight = (id, h) => { document.getElementById(id).style.height = `${h}px`; return h; };
  const baseLayout = extra => Object.assign({
    font: PLOT_FONT, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 56, r: 16, t: 16, b: 48 }, hoverlabel: { font: { family: PLOT_FONT.family } },
    xaxis: { gridcolor: "#E5EBED", zeroline: false }, yaxis: { gridcolor: "#E5EBED", zeroline: false },
  }, extra);

  const state = {
    payload: null, rows: [], rules: null, thr: 0.97,
    sev: null, sensor: null, log: false,
    sevFilter: new Set(SEV_ORDER), causeFilter: new Set(Object.keys(CAUSE)), hood: "",
    sort: { key: "DATETIME", dir: -1 }, explainIdx: null, activeTab: "timeline", drawn: new Set(),
    sourceLabel: "",
  };

  /* ---------------------------------------------------------------- helpers */
  const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toLocaleString("en-GB", { maximumFractionDigits: d, minimumFractionDigits: d }));
  const fmtInt = v => (v == null ? "–" : Math.round(v).toLocaleString("en-GB"));
  const fmtTime = t => new Date(t + ":00").toLocaleString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
  const status = (msg, isError = false) => { const s = $("#status"); s.textContent = msg; s.classList.toggle("is-error", isError); };
  const loading = (on, text = "Scoring readings…") => { $("#loading").hidden = !on; $("#loading-text").textContent = text; };
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const escapeHtml = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ---------------------------------------------------------------- rules (mirror airq.triage) */
  function recompute() {
    const { nbr_support } = state.rules, s = state.sev, thr = state.thr;
    for (const r of state.rows) {
      const isRule = r.rule === "pm_above_sensor_range";
      r.flag = isRule || (r.anomaly_probability != null && r.anomaly_probability >= thr);
      if (!r.flag) { r.severity = "Normal"; r.cause = ""; r.sevRank = 0; continue; }
      if (isRule) { r.severity = "Critical"; r.cause = "sensor_fault_out_of_range"; }
      else {
        const z = Math.abs(r.robust_z ?? 0), pm = r.PM25 ?? 0;
        r.severity = (z >= s.critical_z || pm >= s.critical_pm25) ? "Critical"
                   : (z >= s.alert_z || pm >= s.alert_pm25) ? "Alert" : "Watch";
        const n = r.sensors_in_neighbourhood;
        if (n == null || n < 2 || r.nbr_resid == null) r.cause = "unverifiable";
        else {
          const own = r.residual ?? 0;
          const supported = Math.sign(own) * r.nbr_resid >= nbr_support * Math.max(Math.abs(own), 1);
          r.cause = supported ? "genuine" : "likely_sensor_fault";
        }
      }
      r.sevRank = SEV_ORDER.indexOf(r.severity) + 1;
    }
  }
  const flagged = () => state.rows.filter(r => r.flag);

  /* ---------------------------------------------------------------- data loading */
  async function loadSamples() {
    const list = await (await fetch("/api/samples")).json();
    const box = $("#samples");
    box.replaceChildren(...list.map((s, k) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "sample"; b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", k === 0 ? "true" : "false"); b.dataset.id = s.id;
      b.innerHTML = `<strong>${escapeHtml(s.title)}</strong><span>${escapeHtml(s.summary)}</span>`;
      b.addEventListener("click", () => selectSample(s.id, s.title));
      return b;
    }));
    if (list.length) selectSample(list[0].id, list[0].title);
  }

  async function selectSample(id, title) {
    $$(".sample").forEach(b => b.setAttribute("aria-checked", String(b.dataset.id === id)));
    loading(true, "Scoring the sample. The first visit can take up to a minute while the server wakes up…");
    try {
      const res = await fetch(`/api/samples/${id}/score`);
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      ingest(await res.json(), `Sample: ${title}`);
    } catch (e) { status(`Could not score the sample: ${e.message}`, true); }
    finally { loading(false); closeControls(); }
  }

  async function uploadFile(file) {
    if (!file) return;
    $$(".sample").forEach(b => b.setAttribute("aria-checked", "false"));
    loading(true, `Scoring ${file.name}…`);
    const fd = new FormData(); fd.append("file", file);
    try {
      const res = await fetch("/api/score", { method: "POST", body: fd });
      const body = await res.json();
      if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
      ingest(body, `Your file: ${file.name}`);
    } catch (e) { status(`Could not score ${file.name}. ${e.message}`, true); }
    finally { loading(false); $("#file-input").value = ""; closeControls(); }
  }

  function ingest(payload, label) {
    state.payload = payload; state.rules = payload.rules; state.sourceLabel = label;
    state.rows = payload.rows.map((arr, i) => {
      const o = { idx: i }; payload.fields.forEach((f, k) => { o[f] = arr[k]; }); return o;
    });
    if (!state.sev) {
      state.sev = { ...payload.rules.severity };
      state.thr = payload.rules.threshold;
      syncControls();
    }
    const sensors = [...new Set(state.rows.map(r => r.LOCATION))];
    recompute();
    const counts = {}; flagged().forEach(r => { counts[r.LOCATION] = (counts[r.LOCATION] || 0) + 1; });
    state.sensor = sensors.sort((a, b) => (counts[b] || 0) - (counts[a] || 0))[0];
    state.explainIdx = null;
    const cl = payload.info.cleaning || {};
    const notes = [];
    if (cl.pm_above_sensor_range) { const n = cl.pm_above_sensor_range; notes.push(`${n} reading${n === 1 ? "" : "s"} above the sensor’s measuring range counted as ${n === 1 ? "a fault" : "faults"}`); }
    if (cl.dropped_missing_pm) notes.push(`${fmtInt(cl.dropped_missing_pm)} without PM2.5 or PM10 were skipped`);
    status(`${label}. ${fmtInt(state.rows.length)} readings from ${sensors.length} sensors.${notes.length ? " " + notes.join("; ") + "." : ""}`);
    buildSensorSelect(sensors, counts);
    refresh(true);
  }

  /* ---------------------------------------------------------------- controls */
  function syncControls() {
    $("#thr").value = state.thr; $("#thr-out").textContent = state.thr.toFixed(3);
    $("#crit-z").value = state.sev.critical_z; $("#crit-pm").value = state.sev.critical_pm25;
    $("#alert-z").value = state.sev.alert_z; $("#alert-pm").value = state.sev.alert_pm25;
  }
  const rerun = debounce(() => { recompute(); refresh(false); }, 60);
  $("#thr").addEventListener("input", e => { state.thr = +e.target.value; $("#thr-out").textContent = state.thr.toFixed(3); rerun(); });
  $("#thr-reset").addEventListener("click", () => { if (!state.rules) return; state.thr = state.rules.threshold; syncControls(); rerun(); });
  [["#crit-z", "critical_z"], ["#crit-pm", "critical_pm25"], ["#alert-z", "alert_z"], ["#alert-pm", "alert_pm25"]].forEach(([sel, key]) =>
    $(sel).addEventListener("input", e => { const v = parseFloat(e.target.value); if (!Number.isNaN(v) && state.sev) { state.sev[key] = v; rerun(); } }));

  const dz = $("#dropzone");
  $("#file-input").addEventListener("change", e => uploadFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("is-over"); }));
  dz.addEventListener("drop", e => uploadFile(e.dataTransfer.files[0]));

  const toggleBtn = $("#controls-toggle"), controls = $("#controls");
  function closeControls() { controls.classList.remove("is-open"); toggleBtn.setAttribute("aria-expanded", "false"); }
  toggleBtn.addEventListener("click", () => {
    const open = !controls.classList.contains("is-open");
    controls.classList.toggle("is-open", open); toggleBtn.setAttribute("aria-expanded", String(open));
  });

  /* ---------------------------------------------------------------- tabs */
  const tabs = $$('[role="tab"]');
  function showTab(name) {
    state.activeTab = name;
    tabs.forEach(t => {
      const on = t.id === `tab-${name}`;
      t.setAttribute("aria-selected", String(on)); t.tabIndex = on ? 0 : -1;
      $(`#${t.getAttribute("aria-controls")}`).hidden = !on;
    });
    renderActive();
  }
  tabs.forEach((t, k) => {
    t.addEventListener("click", () => showTab(t.id.replace("tab-", "")));
    t.addEventListener("keydown", e => {
      if (!["ArrowRight", "ArrowLeft"].includes(e.key)) return;
      const next = tabs[(k + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      next.focus(); showTab(next.id.replace("tab-", ""));
    });
  });

  /* ---------------------------------------------------------------- refresh */
  function refresh(newData) {
    updateKpis();
    state.drawn.clear();
    if (newData) buildFilters();
    renderActive();
  }
  function renderActive() {
    if (!state.rows.length) return;
    const t = state.activeTab;
    if (state.drawn.has(t)) { $$(`#view-${t} .js-plotly-plot`).forEach(el => Plotly.Plots.resize(el)); return; }
    ({ timeline: renderTimeline, flags: renderFlags, explain: renderExplain, sensors: renderSensors, model: renderModel })[t]();
    state.drawn.add(t);
  }

  const kpiShown = {};
  function updateKpis() {
    const f = flagged();
    const vals = { rows: state.rows.length, flags: f.length,
      critical: f.filter(r => r.severity === "Critical").length,
      faults: f.filter(r => r.cause === "likely_sensor_fault" || r.cause === "sensor_fault_out_of_range").length,
      genuine: f.filter(r => r.cause === "genuine").length };
    for (const [k, v] of Object.entries(vals)) {
      const el = document.querySelector(`[data-kpi="${k}"]`);
      const from = kpiShown[k] ?? 0; kpiShown[k] = v;
      if (reduceMotion || typeof gsap === "undefined") { el.textContent = fmtInt(v); continue; }
      const o = { v: from };
      gsap.to(o, { v, duration: 0.5, ease: "power2.out", onUpdate: () => { el.textContent = fmtInt(o.v); } });
    }
  }

  /* ---------------------------------------------------------------- timeline */
  function buildSensorSelect(sensors, counts) {
    const hoodOf = {}; state.rows.forEach(r => { hoodOf[r.LOCATION] = r.NEIGHBORHOOD; });
    const sel = $("#sensor-select");
    sel.replaceChildren(...[...sensors].sort((a, b) => a - b).map(s => {
      const o = document.createElement("option"); o.value = s;
      o.textContent = `${s}, ${hoodOf[s]}${counts[s] ? ` (${counts[s]} flagged)` : ""}`; return o;
    }));
    sel.value = state.sensor;
  }
  $("#sensor-select").addEventListener("change", e => { state.sensor = +e.target.value; renderTimeline(); });
  $("#log-toggle").addEventListener("change", e => { e.target.dataset.touched = "1"; state.log = e.target.checked; renderTimeline(); });

  function renderTimeline() {
    const d = state.rows.filter(r => r.LOCATION === state.sensor);
    const traces = [
      { x: d.map(r => r.DATETIME), y: d.map(r => r.PM25), name: "PM2.5", mode: "lines", type: "scatter",
        line: { color: "#3E7CA6", width: 1.4 }, hovertemplate: "%{x}<br>PM2.5 %{y:.1f} µg/m³<extra></extra>" },
      { x: d.map(r => r.DATETIME), y: d.map(r => r.baseline), name: "Usual level", mode: "lines", type: "scatter",
        line: { color: "#8C9AA1", width: 1, dash: "dash" }, hovertemplate: "Usual %{y:.1f} µg/m³<extra></extra>" },
      ...SEV_ORDER.map(sv => {
        const f = d.filter(r => r.flag && r.severity === sv);
        return { x: f.map(r => r.DATETIME), y: f.map(r => r.PM25), name: sv, mode: "markers", type: "scatter",
          marker: { color: SEV_COLOR[sv], size: 9, line: { color: "#23313B", width: 0.6 } },
          customdata: f.map(r => [CAUSE[r.cause].label, r.idx]),
          hovertemplate: `${sv}: %{customdata[0]}<br>PM2.5 %{y:.1f} µg/m³<br>Click for the reasons<extra></extra>` };
      }),
    ];
    const tall = d.some(r => r.PM25 > 300);
    if (tall && !state.log && !$("#log-toggle").dataset.touched) { state.log = true; $("#log-toggle").checked = true; }
    Plotly.react("chart-timeline", traces, baseLayout({
      margin: { l: 60, r: 16, t: 8, b: 40 }, hovermode: "closest",
      legend: { orientation: "h", y: 1.08, x: 0 },
      yaxis: { title: { text: `PM2.5 (µg/m³${state.log ? ", log scale" : ""})` }, type: state.log ? "log" : "linear", gridcolor: "#E5EBED" },
      xaxis: { gridcolor: "#E5EBED", rangeslider: { visible: true, thickness: 0.07 }, type: "date" },
    }), PLOT_CFG);
    const el = $("#chart-timeline");
    el.removeAllListeners?.("plotly_click");
    el.on("plotly_click", ev => { const p = ev.points[0]; if (p && p.customdata) { state.explainIdx = p.customdata[1]; showTab("explain"); } });

    const f = flagged();
    const causes = Object.keys(CAUSE).filter(c => f.some(r => r.cause === c));
    Plotly.react("chart-sevcause", causes.map(c => ({
      type: "bar", name: CAUSE[c].label, x: SEV_ORDER, marker: { color: CAUSE[c].color },
      y: SEV_ORDER.map(sv => f.filter(r => r.severity === sv && r.cause === c).length),
      hovertemplate: `${CAUSE[c].label}<br>%{x}: %{y}<extra></extra>`,
    })), baseLayout({ barmode: "stack", legend: { orientation: "h", y: -0.18 }, margin: { l: 48, r: 8, t: 8, b: 70 },
      yaxis: { title: { text: "Flagged readings" }, gridcolor: "#E5EBED", rangemode: "tozero" } }), PLOT_CFG);

    const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
    const tot = days.map(() => Array(24).fill(0)), hit = days.map(() => Array(24).fill(0));
    state.rows.forEach(r => { const dt = new Date(r.DATETIME + ":00"); const w = (dt.getDay() + 6) % 7, h = dt.getHours(); tot[w][h]++; if (r.flag) hit[w][h]++; });
    Plotly.react("chart-heat", [{ type: "heatmap", x: [...Array(24).keys()], y: days,
      z: tot.map((row, w) => row.map((n, h) => (n ? hit[w][h] / n : null))),
      colorscale: [[0, "#F7F2EA"], [0.4, "#EFB06E"], [1, "#9E2F35"]], hoverongaps: false,
      hovertemplate: "%{y} %{x}:00<br>flag rate %{z:.1%}<extra></extra>", colorbar: { tickformat: ".0%", thickness: 12 } }],
      baseLayout({ margin: { l: 90, r: 8, t: 8, b: 44 }, xaxis: { title: { text: "Local hour" }, dtick: 3 },
        yaxis: { autorange: "reversed" } }), PLOT_CFG);
  }

  /* ---------------------------------------------------------------- flags table */
  function buildFilters() {
    const sev = $("#sev-filter"), cause = $("#cause-filter");
    sev.replaceChildren(...SEV_ORDER.slice().reverse().map(sv => chip(sv, SEV_COLOR[sv], state.sevFilter, sv)));
    cause.replaceChildren(...Object.keys(CAUSE).map(c => chip(CAUSE[c].label, CAUSE[c].color, state.causeFilter, c)));
  }
  function chip(label, color, set, key) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.setAttribute("aria-pressed", String(set.has(key)));
    b.innerHTML = `<span class="swatch" style="background:${color}"></span>${escapeHtml(label)}`;
    b.addEventListener("click", () => {
      set.has(key) ? set.delete(key) : set.add(key);
      b.setAttribute("aria-pressed", String(set.has(key))); renderFlags();
    });
    return b;
  }
  $("#hood-search").addEventListener("input", debounce(e => { state.hood = e.target.value.trim().toLowerCase(); renderFlags(); }, 120));
  $$("#flag-table th[data-sort]").forEach(th => th.querySelector("button").addEventListener("click", () => {
    const key = th.dataset.sort;
    state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : -1 };
    $$("#flag-table th[data-sort]").forEach(h => h.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", state.sort.dir === 1 ? "ascending" : "descending");
    renderFlags();
  }));

  function visibleFlags() {
    const { key, dir } = state.sort;
    return flagged().filter(r => state.sevFilter.has(r.severity) && state.causeFilter.has(r.cause)
        && (!state.hood || r.NEIGHBORHOOD.toLowerCase().includes(state.hood)))
      .sort((a, b) => { const x = a[key], y = b[key]; return (x > y ? 1 : x < y ? -1 : 0) * dir; });
  }
  function renderFlags() {
    const rows = visibleFlags(), total = flagged().length;
    $("#flag-count").textContent = total ? `Showing ${rows.length} of ${total} flagged readings. Select a row to see why it was flagged.`
      : "No readings are flagged at this threshold. Lower the alert threshold to see more.";
    const tb = $("#flag-table tbody");
    tb.replaceChildren(...rows.slice(0, 2000).map(r => {
      const tr = document.createElement("tr"); tr.tabIndex = 0;
      tr.innerHTML = `<td>${escapeHtml(r.DATETIME.replace("T", " "))}</td><td>${r.LOCATION}</td><td>${escapeHtml(r.NEIGHBORHOOD)}</td>
        <td class="num">${fmt(r.PM25)}</td><td class="num">${fmt(r.baseline)}</td><td class="num">${r.anomaly_probability == null ? "–" : r.anomaly_probability.toFixed(3)}</td>
        <td><span class="sev sev-${r.severity}">${r.severity}</span></td><td>${CAUSE[r.cause].label}</td><td>${CAUSE[r.cause].route}</td>`;
      const open = () => { state.explainIdx = r.idx; showTab("explain"); };
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", e => { if (e.key === "Enter") open(); });
      return tr;
    }));
  }
  $("#download").addEventListener("click", () => {
    const cols = ["DATETIME", "LOCATION", "NEIGHBORHOOD", "PM25", "baseline", "residual", "robust_z", "anomaly_probability", "severity", "cause", "humidity_artefact_risk"];
    const csv = [cols.join(",") + ",route_to", ...visibleFlags().map(r => cols.map(c => `"${String(r[c] ?? "").replace(/"/g, '""')}"`).join(",") + `,"${CAUSE[r.cause].route}"`)].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "hewa_flagged_readings.csv"; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });

  /* ---------------------------------------------------------------- explain */
  function explainList() {
    return flagged().slice().sort((a, b) => b.sevRank - a.sevRank || (b.anomaly_probability ?? 0) - (a.anomaly_probability ?? 0) || (a.DATETIME < b.DATETIME ? 1 : -1));
  }
  function renderExplain() {
    const list = explainList(), sel = $("#explain-select");
    if (!list.length) {
      sel.replaceChildren(); $("#explain-card").innerHTML = `<p class="empty">No flagged readings at this threshold. Lower the alert threshold in Data and rules.</p>`;
      Plotly.purge("chart-shap"); Plotly.purge("chart-nbr"); return;
    }
    if (state.explainIdx == null || !list.some(r => r.idx === state.explainIdx)) state.explainIdx = list[0].idx;
    sel.replaceChildren(...list.map(r => {
      const o = document.createElement("option"); o.value = r.idx;
      o.textContent = `${r.DATETIME.replace("T", " ")}, sensor ${r.LOCATION}, ${r.NEIGHBORHOOD}, PM2.5 ${fmt(r.PM25, 0)} (${r.severity})`;
      return o;
    }));
    sel.value = state.explainIdx;
    const r = state.rows[state.explainIdx], ex = state.payload.explanations[r.idx];
    const c = CAUSE[r.cause];
    const guess = r.cause === "unverifiable" && r.cause_model_guess
      ? `${CAUSE[r.cause_model_guess] ? CAUSE[r.cause_model_guess].label : r.cause_model_guess} (${Math.round((r.cause_model_confidence || 0) * 100)}% confident, advisory only)` : "–";
    $("#explain-card").innerHTML = `
      <h3><span class="sev sev-${r.severity}">${r.severity}</span> ${escapeHtml(c.label)}</h3>
      <p class="explain-why">${r.rule === "pm_above_sensor_range"
        ? "This reading is above the 1,000 µg/m³ that the sensor can physically measure, so it is treated as a sensor fault without needing the model."
        : escapeHtml(ex ? ex.text : "Explanation not available for this reading.")}</p>
      <dl class="facts">
        <dt>When</dt><dd>${fmtTime(r.DATETIME)} (Nairobi time)</dd>
        <dt>Sensor</dt><dd>${r.LOCATION}, ${escapeHtml(r.NEIGHBORHOOD)}</dd>
        <dt>Reading</dt><dd>${fmt(r.PM25)} µg/m³ PM2.5, ${fmt(r.PM10)} µg/m³ PM10</dd>
        <dt>Usual level</dt><dd>${fmt(r.baseline)} µg/m³ (${r.residual == null ? "–" : (r.residual >= 0 ? "+" : "") + fmt(r.residual)} above)</dd>
        <dt>Unusualness</dt><dd>${r.robust_z == null ? "–" : fmt(Math.abs(r.robust_z))} (robust z)</dd>
        <dt>Probability</dt><dd>${r.anomaly_probability == null ? "–" : r.anomaly_probability.toFixed(4)}</dd>
        <dt>Neighbours</dt><dd>${r.sensors_in_neighbourhood == null ? "–" : r.sensors_in_neighbourhood} sensor(s) reporting${r.nbr_resid == null ? "" : `, their mean excess ${fmt(r.nbr_resid)} µg/m³`}</dd>
        <dt>Why this cause</dt><dd>${c.text}</dd>
        <dt>Model’s guess</dt><dd>${guess}</dd>
        <dt>Humidity</dt><dd>${r.HUMIDITY == null ? "–" : fmt(r.HUMIDITY, 0) + "%"}${r.humidity_artefact_risk ? ". High humidity can make optical sensors over-read" : ""}</dd>
        <dt>Goes to</dt><dd>${c.route}</dd>
      </dl>`;

    if (ex && ex.drivers && ex.drivers.length) {
      const dr = ex.drivers.slice().reverse();
      const wrap = t => t.replace(/(.{1,42})(\s|$)/g, "$1<br>").replace(/<br>$/, "");
      Plotly.react("chart-shap", [{ type: "bar", orientation: "h", x: dr.map(d => d.shap), y: dr.map(d => wrap(d.text)),
        marker: { color: dr.map(d => (d.shap > 0 ? "#B23A3F" : "#3E7CA6")) },
        text: dr.map(d => (d.shap > 0 ? "+" : "") + d.shap.toFixed(2)), textposition: "outside", cliponaxis: false,
        hovertemplate: "%{y}<br>contribution %{x:.2f}<extra></extra>" }],
        baseLayout({ margin: { l: 10, r: 24, t: 8, b: 48 }, height: fitHeight("chart-shap", Math.max(320, dr.length * 52 + 70)),
          xaxis: { title: { text: "Contribution (log-odds)" }, zeroline: true, zerolinecolor: "#8C9AA1", gridcolor: "#E5EBED",
            range: (() => { const v = dr.map(d => d.shap), lo = Math.min(0, ...v), hi = Math.max(0, ...v), sp = (hi - lo) || 1;
              return [lo - 0.4 * sp, hi + 0.18 * sp]; })() },
          yaxis: { automargin: true, tickfont: { size: 12 } } }), PLOT_CFG);
    } else {
      Plotly.purge("chart-shap");
      $("#chart-shap").innerHTML = `<p class="empty">${r.rule === "pm_above_sensor_range" ? "Flagged by the sensor-range rule, so there are no model reasons." : "No explanation stored for this reading."}</p>`;
    }

    const t0 = new Date(r.DATETIME + ":00").getTime();
    const near = state.rows.filter(x => x.NEIGHBORHOOD === r.NEIGHBORHOOD && Math.abs(new Date(x.DATETIME + ":00").getTime() - t0) <= 48 * 3600e3);
    const sensors = [...new Set(near.map(x => x.LOCATION))];
    $("#nbr-title").textContent = `Did neighbouring sensors in ${r.NEIGHBORHOOD} see it? (48 hours either side)`;
    const palette = ["#3E7CA6", "#2F8F8B", "#6B4E9B", "#E3A73B", "#8C9AA1", "#DD7034"];
    Plotly.react("chart-nbr", sensors.map((s, k) => {
      const d = near.filter(x => x.LOCATION === s);
      return { type: "scatter", mode: "lines", name: `Sensor ${s}${s === r.LOCATION ? " (flagged)" : ""}`,
        x: d.map(x => x.DATETIME), y: d.map(x => x.PM25), line: { width: s === r.LOCATION ? 2.4 : 1.4, color: palette[k % palette.length] },
        hovertemplate: `Sensor ${s}<br>%{x}<br>PM2.5 %{y:.1f}<extra></extra>` };
    }), baseLayout({ legend: { orientation: "h", y: 1.12 }, margin: { l: 60, r: 16, t: 24, b: 44 },
      yaxis: { title: { text: "PM2.5 (µg/m³)" }, gridcolor: "#E5EBED", type: near.some(x => x.PM25 > 300) ? "log" : "linear" },
      xaxis: { type: "date", gridcolor: "#E5EBED" },
      shapes: [{ type: "line", x0: r.DATETIME, x1: r.DATETIME, yref: "paper", y0: 0, y1: 1, line: { color: "#23313B", dash: "dot", width: 1.5 } }] }), PLOT_CFG);
  }
  $("#explain-select").addEventListener("change", e => { state.explainIdx = +e.target.value; renderExplain(); });
  const stepExplain = d => { const list = explainList(); const k = list.findIndex(r => r.idx === state.explainIdx);
    if (k < 0) return; state.explainIdx = list[(k + d + list.length) % list.length].idx; renderExplain(); };
  $("#explain-prev").addEventListener("click", () => stepExplain(-1));
  $("#explain-next").addEventListener("click", () => stepExplain(1));

  /* ---------------------------------------------------------------- sensors */
  function renderSensors() {
    const by = new Map();
    for (const r of state.rows) {
      if (!by.has(r.LOCATION)) by.set(r.LOCATION, { s: r.LOCATION, hood: r.NEIGHBORHOOD, n: 0, f: 0, faults: 0, pm: [], last: r.DATETIME, lat: r.NEIGHBORHOOD_LAT, lon: r.NEIGHBORHOOD_LON });
      const g = by.get(r.LOCATION); g.n++; if (r.PM25 != null) g.pm.push(r.PM25);
      if (r.flag) { g.f++; if (r.cause === "likely_sensor_fault" || r.cause === "sensor_fault_out_of_range") g.faults++; }
      if (r.DATETIME > g.last) g.last = r.DATETIME;
    }
    const med = a => { const s = a.slice().sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
    const groups = [...by.values()].map(g => ({ ...g, rate: g.f / g.n, median: med(g.pm) })).sort((a, b) => b.faults - a.faults || b.rate - a.rate);
    const maxRate = Math.max(...groups.map(g => g.rate), 1e-9);
    $("#sensor-table tbody").replaceChildren(...groups.map(g => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${g.s}</td><td>${escapeHtml(g.hood)}</td><td class="num">${fmtInt(g.n)}</td><td class="num">${g.f}</td>
        <td><span class="bar" style="width:${Math.round(70 * g.rate / maxRate)}px"></span>${(g.rate * 100).toFixed(1)}%</td>
        <td class="num">${g.faults}</td><td class="num">${fmt(g.median)}</td><td>${escapeHtml(g.last.replace("T", " "))}</td>`;
      tr.addEventListener("click", () => { state.sensor = g.s; $("#sensor-select").value = g.s; showTab("timeline"); });
      return tr;
    }));

    const order = groups.slice().sort((a, b) => (b.median ?? 0) - (a.median ?? 0));
    const label = g => `${g.s} · ${g.hood}`;
    const cats = order.map(label);
    const mk = (name, filt, color, width) => {
      const d = state.rows.filter(filt);
      return { type: "box", orientation: "h", name, x: d.map(r => Math.max(r.PM25, 0.5)), y: d.map(r => label(by.get(r.LOCATION))),
        marker: { color, size: 3, opacity: 0.6 }, line: { color, width: 1.4 }, fillcolor: color + "55", width, boxpoints: "outliers",
        hovertemplate: `${name}<br>%{y}<br>PM2.5 %{x:.1f}<extra></extra>` };
    };
    Plotly.react("chart-dist", [mk("Normal", r => !r.flag, "#3E7CA6", 0.55), mk("Flagged", r => r.flag, "#B23A3F", 0.3)],
      baseLayout({ boxmode: "overlay", height: fitHeight("chart-dist", Math.max(300, cats.length * 58 + 110)), legend: { orientation: "h", y: 1.06 },
        margin: { l: 10, r: 16, t: 30, b: 50 },
        xaxis: { type: "log", title: { text: "PM2.5 (µg/m³, log scale)" }, gridcolor: "#E5EBED" },
        yaxis: { categoryorder: "array", categoryarray: cats.slice().reverse(), automargin: true } }), PLOT_CFG);

    const geo = groups.filter(g => g.lat != null && g.lon != null);
    if (!geo.length) { Plotly.purge("chart-map"); $("#chart-map").innerHTML = `<p class="empty">No coordinates in this data, so locations can’t be shown.</p>`; return; }
    const span = (k, f) => { const v = geo.map(f); return [Math.min(...v), Math.max(...v)]; };
    const [x0, x1] = span(0, g => g.lon), [y0, y1] = span(0, g => g.lat);
    const px = Math.max(x1 - x0, 0.03) * 0.25, py = Math.max(y1 - y0, 0.03) * 0.25;
    const nmax = Math.max(...geo.map(g => g.n));
    Plotly.react("chart-map", [{ type: "scatter", mode: "markers+text", x: geo.map(g => g.lon), y: geo.map(g => g.lat),
      text: geo.map(label), textposition: "top center", textfont: { size: 11 },
      marker: { size: geo.map(g => 12 + 30 * Math.sqrt(g.n / nmax)), color: geo.map(g => g.rate), cmin: 0, cmax: maxRate,
        colorscale: [[0, "#F7F2EA"], [0.4, "#EFB06E"], [1, "#9E2F35"]], line: { color: "#23313B", width: 0.8 },
        colorbar: { title: { text: "flag rate" }, tickformat: ".0%", thickness: 12 } },
      customdata: geo.map(g => [g.n, g.f]), hovertemplate: "%{text}<br>%{customdata[0]} readings, %{customdata[1]} flagged<br>flag rate %{marker.color:.1%}<extra></extra>" }],
      baseLayout({ margin: { l: 60, r: 16, t: 8, b: 48 },
        xaxis: { title: { text: "Longitude" }, range: [x0 - px, x1 + px], gridcolor: "#E5EBED" },
        yaxis: { title: { text: "Latitude" }, range: [y0 - py, y1 + py], scaleanchor: "x", scaleratio: 1, gridcolor: "#E5EBED" } }), PLOT_CFG);
  }

  /* ---------------------------------------------------------------- model */
  async function renderModel() {
    const box = $("#model-card");
    try {
      const o = await (await fetch("/api/overview")).json();
      const m = o.metrics || {}, s = o.splits || {};
      const f2 = x => (x == null ? "–" : x.toFixed(3));
      const row = (label, k) => `<tr><th scope="row">${label}</th><td class="num">${f2(m.val && m.val[k])}</td><td class="num">${f2(m.test && m.test[k])}</td></tr>`;
      box.innerHTML = `
        <h3>How the model works</h3>
        <p>Five unsupervised detectors look at each reading from different angles: a robust statistical test, local density (LOF), a learned boundary (one-class SVM), cluster membership (DBSCAN) and the sensor’s own daily rhythm (STL). When at least three agree, the reading is labelled anomalous. An XGBoost model trained on 2019–2023 learns that vote from ${o.features.length} features: <code>${o.features.join("</code>, <code>")}</code>. The tuned alert threshold is ${o.threshold.toFixed(3)}, chosen on the 2024 data.</p>
        <h3>How well it agrees with the detectors</h3>
        <div class="table-wrap"><table class="data-table"><thead><tr><th></th><th class="num">Validation, 2024</th><th class="num">Test, 2025–26</th></tr></thead>
          <tbody>${row("Precision (anomalies)", "precision")}${row("Recall (anomalies)", "recall")}${row("F1 (anomalies)", "f1")}${row("ROC-AUC", "roc_auc")}${row("Precision-recall area", "pr_auc")}
          <tr><th scope="row">Readings</th><td class="num">${fmtInt(m.val && m.val.n)}</td><td class="num">${fmtInt(m.test && m.test.n)}</td></tr>
          <tr><th scope="row">Active sensors</th><td class="num">${s.val ? s.val.sensors : "–"}</td><td class="num">${s.test ? s.test.sensors : "–"}</td></tr></tbody></table></div>
        <h3>Read these numbers with care</h3>
        <ul>
          <li>There are no confirmed anomaly labels for Nairobi, so the scores measure agreement with the five-detector vote, not correctness.</li>
          <li>The test years had only seven active sensors, and one or two devices produced most flags.</li>
          <li>When no neighbouring sensor was reporting, the cause can’t be checked. The model’s guess in that case is close to chance, so treat it as a hint.</li>
          <li>Optical sensors over-read in very humid air. Flags with a positive excess at 85% humidity or more carry a warning.</li>
        </ul>`;
    } catch { box.innerHTML = `<p class="empty">Model details are unavailable right now.</p>`; }
  }

  /* ---------------------------------------------------------------- start */
  loadSamples().catch(e => status(`The server did not respond: ${e.message}. Refresh in a minute.`, true));
})();
