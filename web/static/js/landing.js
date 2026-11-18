/* Hewa landing page: particle haze driven by a replay of real readings, and the pinned pipeline. */
(() => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hasGsap = typeof window.gsap !== "undefined";
  if (hasGsap && window.ScrollTrigger) gsap.registerPlugin(ScrollTrigger);

  /* ------------------------------------------------------------- haze canvas */
  const canvas = document.getElementById("haze");
  const ctx = canvas.getContext("2d");
  let W = 0, H = 0, dpr = 1, particles = [];
  const haze = { density: 0.25, warmth: 0 };   // 0..1, eased towards target by GSAP

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(Math.min(900, (W * H) / 1800));
    particles = Array.from({ length: n }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      r: 0.6 + Math.random() * 2.2, vx: 0.08 + Math.random() * 0.35, vy: (Math.random() - 0.5) * 0.12,
      a: Math.random(), rank: Math.random(),
    }));
  }

  function drawHaze() {
    ctx.clearRect(0, 0, W, H);
    // a veil that thickens with PM2.5 and warms from grey to brown
    const r = Math.round(150 + 60 * haze.warmth), g = Math.round(160 + 10 * haze.warmth), b = Math.round(165 - 75 * haze.warmth);
    const veil = ctx.createLinearGradient(0, 0, 0, H);
    veil.addColorStop(0, `rgba(${r},${g},${b},${0.04 + 0.32 * haze.density})`);
    veil.addColorStop(0.6, `rgba(${r},${g},${b},${0.02 + 0.22 * haze.density})`);
    veil.addColorStop(1, "rgba(31,43,51,0)");
    ctx.fillStyle = veil; ctx.fillRect(0, 0, W, H);
    // particles: the share that is visible follows the density
    for (const p of particles) {
      if (p.rank > 0.18 + 0.82 * haze.density) continue;
      ctx.beginPath();
      ctx.fillStyle = `rgba(${r + 30},${g + 25},${b + 20},${0.18 + 0.35 * p.a})`;
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
      if (!reduceMotion) {
        p.x += p.vx * (0.6 + haze.density); p.y += p.vy;
        if (p.x > W + 4) { p.x = -4; p.y = Math.random() * H; }
        if (p.y < -4 || p.y > H + 4) p.y = Math.random() * H;
      }
    }
  }

  let hazeVisible = true;
  new IntersectionObserver(([e]) => { hazeVisible = e.isIntersecting; }).observe(canvas);
  function loop() { if (hazeVisible) drawHaze(); if (!reduceMotion) requestAnimationFrame(loop); }
  resize(); window.addEventListener("resize", () => { resize(); if (reduceMotion) drawHaze(); });

  /* ------------------------------------------------------------- telemetry replay */
  const el = id => document.getElementById(id);
  const svg = el("t-spark"), statusEl = el("t-status"), valueEl = el("t-value");
  const NS = "http://www.w3.org/2000/svg";
  const fmtTime = t => new Date(t + ":00").toLocaleString("en-GB",
    { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

  function densityFor(pm) { return Math.max(0.12, Math.min(1, Math.log10(Math.max(pm, 1)) / Math.log10(400))); }

  function buildSpark(data) {
    const vmax = Math.log10(Math.max(...data.map(d => d.pm25)) * 1.1);
    const x = i => (i / (data.length - 1)) * 300;
    const y = v => 82 - (Math.log10(Math.max(v, 1)) / vmax) * 78;      // log scale so ordinary hours stay readable
    const base = document.createElementNS(NS, "polyline");
    base.setAttribute("points", data.map((d, i) => `${x(i)},${y(d.baseline)}`).join(" "));
    base.setAttribute("fill", "none"); base.setAttribute("stroke", "rgba(255,255,255,.45)");
    base.setAttribute("stroke-dasharray", "3 3"); base.setAttribute("stroke-width", "1.2");
    base.setAttribute("vector-effect", "non-scaling-stroke");
    const line = document.createElementNS(NS, "polyline");
    line.setAttribute("fill", "none"); line.setAttribute("stroke", "#8FD3CF"); line.setAttribute("stroke-width", "2");
    line.setAttribute("stroke-linejoin", "round"); line.setAttribute("vector-effect", "non-scaling-stroke");
    const dots = document.createElementNS(NS, "g");
    svg.append(base, line, dots);
    return { x, y, line, dots };
  }

  function render(data, spark, i) {
    const d = data[i];
    el("t-time").textContent = fmtTime(d.t);
    el("t-usual").textContent = `Usual for this hour: ${Math.round(d.baseline)} µg/m³` +
      (d.humidity != null ? `, humidity ${d.humidity}%` : "");
    spark.line.setAttribute("points", data.slice(0, i + 1).map((p, k) => `${spark.x(k)},${spark.y(p.pm25)}`).join(" "));
    spark.dots.replaceChildren(...data.slice(0, i + 1).map((p, k) => {
      if (!p.flag) return null;
      const c = document.createElementNS(NS, "circle");
      c.setAttribute("cx", spark.x(k)); c.setAttribute("cy", spark.y(p.pm25)); c.setAttribute("r", 3.4);
      c.setAttribute("fill", p.severity === "Critical" ? "#E05A60" : "#F0955F");
      c.setAttribute("stroke", "#1F2B33"); c.setAttribute("stroke-width", "1");
      return c;
    }).filter(Boolean));
    const level = d.flag ? (d.severity || "Alert") : "Normal";
    statusEl.dataset.level = level;
    statusEl.textContent = d.flag ? `Flagged: ${level}` : "Normal for this hour";
  }

  fetch("/static/data/hero_replay.json").then(r => r.json()).then(rep => {
    const data = rep.readings;
    el("t-site").textContent = `Sensor ${rep.sensor}, ${rep.neighbourhood}`;
    const spark = buildSpark(data);
    const shown = { v: data[0].pm25 };
    if (reduceMotion || !hasGsap) {
      const peak = data.reduce((m, d, k) => (d.pm25 > data[m].pm25 ? k : m), 0);
      render(data, spark, peak); valueEl.textContent = Math.round(data[peak].pm25);
      haze.density = densityFor(data[peak].pm25); haze.warmth = haze.density; drawHaze();
      el("t-toggle").hidden = true; return;
    }
    let i = 0, paused = false;
    const step = () => {
      if (paused) return;
      const d = data[i]; render(data, spark, i);
      gsap.to(shown, { v: d.pm25, duration: 0.6, ease: "power2.out",
        onUpdate: () => { valueEl.textContent = Math.round(shown.v); } });
      const dens = densityFor(d.pm25);
      gsap.to(haze, { density: dens, warmth: Math.max(0, (dens - 0.45) / 0.55), duration: 1.4, ease: "sine.inOut" });
      if (d.flag) gsap.fromTo(".telemetry", { boxShadow: "0 0 0 0 rgba(224,90,96,.7)" },
        { boxShadow: "0 0 0 14px rgba(224,90,96,0)", duration: 1.1, ease: "power2.out" });
      i = (i + 1) % data.length;
    };
    step();
    const timer = setInterval(step, 950);
    el("t-toggle").addEventListener("click", e => {
      paused = !paused; e.currentTarget.textContent = paused ? "Play" : "Pause";
      e.currentTarget.setAttribute("aria-pressed", String(paused));
    });
    document.addEventListener("visibilitychange", () => { if (document.hidden) paused = true; });
    window.addEventListener("pagehide", () => clearInterval(timer));
  }).catch(() => { el("t-status").textContent = "Replay unavailable"; });

  if (!reduceMotion) { requestAnimationFrame(loop); }

  /* ------------------------------------------------------------- pinned pipeline */
  const steps = [...document.querySelectorAll(".step")];
  const fill = el("meter-fill");
  function setStep(k, progress) {
    steps.forEach((s, j) => { s.classList.toggle("is-active", j === k); s.classList.toggle("is-done", j < k); });
    if (fill) fill.style.transform = `scaleX(${progress})`;
  }
  if (hasGsap && window.ScrollTrigger && !reduceMotion) {
    ScrollTrigger.matchMedia({
      "(min-width: 861px)": () => {
        setStep(0, 0.05);
        ScrollTrigger.create({
          trigger: ".pipeline", start: "top top", end: () => `+=${window.innerHeight * 2.2}`,
          pin: ".pipeline-stage", scrub: true,
          onUpdate: self => setStep(Math.min(steps.length - 1, Math.floor(self.progress * steps.length)),
                                    Math.max(0.05, self.progress)),
        });
      },
      "(max-width: 860px)": () => { setStep(steps.length - 1, 1); steps.forEach(s => s.classList.add("is-done")); },
    });
  } else {
    steps.forEach(s => s.classList.add("is-done")); if (fill) fill.style.transform = "scaleX(1)";
  }

  /* ------------------------------------------------------------- live numbers */
  fetch("/api/overview").then(r => r.ok ? r.json() : null).then(o => {
    if (!o || !o.metrics) return;
    const v = o.metrics.val, t = o.metrics.test, s = o.splits || {};
    const f2 = x => (x == null ? "–" : x.toFixed(2));
    const n = x => (x == null ? "–" : x.toLocaleString("en-GB"));
    el("score-body").innerHTML = `
      <tr><th scope="row">Precision-recall area</th><td><strong>${f2(v.pr_auc)}</strong></td><td><strong>${f2(t.pr_auc)}</strong></td></tr>
      <tr><th scope="row">F1 on anomalies</th><td>${f2(v.f1)}</td><td>${f2(t.f1)}</td></tr>
      <tr><th scope="row">Readings scored</th><td>${n(v.n)}</td><td>${n(t.n)}</td></tr>
      <tr><th scope="row">Active sensors</th><td>${n(s.val && s.val.sensors)}</td><td>${n(s.test && s.test.sensors)}</td></tr>`;
  }).catch(() => {});
})();
