/* ---------------------------------------------------------------------------
   Credit-risk case study — page logic.

   Every number on the page comes from docs/data.json, which is generated from
   reports/results/summary.json by `python -m src.pipeline figures`. This file
   contains no metric literal: text placeholders are filled from data.json's
   preformatted `display` map, and the tables/charts are drawn from the data.json
   arrays. The Monte-Carlo simulation already ran in Python — the page only
   displays precomputed results and never simulates.
   --------------------------------------------------------------------------- */
"use strict";

(function () {
  const DATA_URL = "data.json";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const num = (x) => (typeof x === "number" && isFinite(x) ? x : null);
  const fmtNum = (x, nd = 4) => (num(x) === null ? "—" : num(x).toFixed(nd));
  const fmtPct = (x, nd = 1) => (num(x) === null ? "—" : (num(x) * 100).toFixed(nd) + "%");
  const fmtInt = (x) => (num(x) === null ? "—" : Math.round(num(x)).toLocaleString("en-US"));

  function getPath(obj, path) {
    return path.split(".").reduce((acc, key) => (acc == null ? acc : acc[key]), obj);
  }

  // -------------------------------------------------- fill display placeholders
  function renderDisplay(d) {
    const disp = d.display || {};
    $$('[data-field^="display."]').forEach((el) => {
      const key = el.getAttribute("data-field").slice("display.".length);
      if (Object.prototype.hasOwnProperty.call(disp, key)) {
        el.textContent = disp[key];
      }
    });
  }

  // --------------------------------------------------------- Plotly theming
  function themed() {
    const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const fg = dark ? "#e8eef5" : "#14202e";
    const muted = dark ? "#9db0c4" : "#5a6b7d";
    const grid = dark ? "#26384b" : "#dde3ea";
    return {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', size: 12, color: fg },
      margin: { l: 60, r: 20, t: 40, b: 50 },
      xaxis: { gridcolor: grid, zerolinecolor: grid, color: muted },
      yaxis: { gridcolor: grid, zerolinecolor: grid, color: muted },
      legend: { orientation: "h", y: 1.12, x: 0, font: { color: fg } },
      colorway: ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"],
    };
  }

  const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

  function plot(id, traces, layout) {
    const el = document.getElementById(id);
    if (!el) return;
    if (window.__plotlyLoadFailed ||
        typeof window.Plotly === "undefined" ||
        typeof window.Plotly.newPlot !== "function") {
      showStaticFallback(el);
      return;
    }
    try {
      const drawn = window.Plotly.newPlot(el, traces, Object.assign(themed(), layout), PLOTLY_CONFIG);
      if (drawn && typeof drawn.catch === "function") {
        drawn.catch((err) => {
          console.error("Plotly failed for #" + id + ":", err);
          showStaticFallback(el);
        });
      }
    } catch (err) {
      console.error("Plotly threw for #" + id + ":", err);
      showStaticFallback(el);
    }
  }

  function showStaticFallback(el) {
    if (el.dataset.fallbackShown === "1") return;
    const src = el.getAttribute("data-fallback-src");
    if (!src) return;
    el.dataset.fallbackShown = "1";
    el.innerHTML = "";
    const img = document.createElement("img");
    img.src = src;
    img.alt = el.getAttribute("aria-label") || "static chart";
    img.loading = "lazy";
    el.appendChild(img);
  }

  // ------------------------------------------------------------------ charts
  function chartCensoring(d) {
    const rows = getPath(d, "pd.split_design.resolved_share_by_year_term") || [];
    const byYear = {};
    rows.forEach((r) => {
      const t = String(r.term_months);
      (byYear[t] = byYear[t] || {})[r.issue_year] = num(r.resolved_share);
    });
    const terms = ["36", "60"].filter((t) => byYear[t]);
    const years = Array.from(new Set(rows.map((r) => r.issue_year))).sort((a, b) => a - b);
    const traces = terms.map((t, i) => ({
      x: years,
      y: years.map((y) => (byYear[t] && byYear[t][y] != null ? byYear[t][y] * 100 : null)),
      type: "bar",
      name: t + "-month term",
      marker: { color: i === 0 ? "#0072B2" : "#E69F00" },
    }));
    traces.push({
      x: [years[0] - 0.5, years[years.length - 1] + 0.5],
      y: [90, 90],
      type: "scatter",
      mode: "lines",
      name: "90% threshold (primary split)",
      line: { color: "#CC79A7", width: 2, dash: "dash" },
      hoverinfo: "skip",
    });
    plot("chart-censoring", traces, {
      barmode: "group",
      xaxis: { title: "Issue year", dtick: 1 },
      yaxis: { title: "Resolved share (%)", range: [0, 105] },
    });
  }

  function chanceMarker(x0, x1, label) {
    return {
      type: "line", xref: "x", yref: "paper", x0: 0.5, x1: 0.5, y0: 0, y1: 1,
      line: { color: "#14202e", width: 1.3, dash: "dot" },
      name: label,
    };
  }

  function chartAucWindow(d) {
    const wc = d.pd.window_comparison || [];
    const aucs = wc.map((r) => num(r.auc));
    plot("chart-auc-window", [{
      x: aucs,
      y: wc.map((r) => (r.window || "window") + " (" + (r.test_years || "") + ")"),
      type: "scatter",
      mode: "markers+text",
      marker: { size: 12, color: ["#0072B2", "#D55E00"] },
      text: aucs.map((v) => fmtNum(v)), textposition: "middle right",
      showlegend: false,
    }], {
      xaxis: { title: "Out-of-time AUC (axis 0.50–0.78; chance = 0.50)", range: [0.5, 0.78] },
      yaxis: { automargin: true },
      shapes: [chanceMarker(0.5, 0.5, "Chance")],
      margin: { l: 150, r: 60, t: 20, b: 55 },
    });
  }

  function chartModels(d) {
    const models = (d.pd.models || []).filter((m) => (m.sample || "oot") === "oot");
    const aucs = models.map((m) => num(m.auc));
    const hi = Math.max(0.7, Math.max(...aucs) + 0.05);
    plot("chart-models", [{
      x: aucs, y: models.map((m) => m.model),
      type: "scatter", mode: "markers+text",
      marker: { size: 11, color: models.map((m) => (m.model === d.pd.champion ? "#009E73" : "#999999")) },
      text: aucs.map((v) => fmtNum(v)), textposition: "middle right",
      showlegend: false,
    }], {
      xaxis: { title: "Out-of-time AUC (axis 0.50–" + hi.toFixed(2) + "; chance = 0.50)",
        range: [0.5, hi] },
      yaxis: { automargin: true },
      shapes: [chanceMarker(0.5, hi, "Chance")],
      margin: { l: 130, r: 60, t: 20, b: 50 },
    });
  }

  function chartCalibration(d) {
    const raw = (d.calibration_curve && d.calibration_curve.raw) || [];
    const cal = (d.calibration_curve && d.calibration_curve.calibrated) || [];
    const pts = (rows) => rows.map((r) => ({ x: num(r.predicted) * 100, y: num(r.observed) * 100 }));
    let hi = 1;
    [...raw, ...cal].forEach((r) => { hi = Math.max(hi, num(r.predicted), num(r.observed)); });
    hi = hi * 100 * 1.05;
    plot("chart-calibration", [
      { x: [0, hi], y: [0, hi], mode: "lines", name: "Perfect calibration",
        line: { color: "#14202e", dash: "dash", width: 1.3 } },
      Object.assign({ mode: "lines+markers", name: "Raw scores",
        line: { color: "#D55E00", width: 1.6 }, marker: { size: 7 } }, pts(raw)),
      Object.assign({ mode: "lines+markers", name: "Isotonic-calibrated",
        line: { color: "#0072B2", width: 1.6 }, marker: { size: 7 } }, pts(cal)),
    ], {
      xaxis: { title: "Mean predicted PD (%)", range: [0, hi] },
      yaxis: { title: "Observed default rate (%)", range: [0, hi] },
    });
  }

  function chartElGrade(d) {
    const grades = d.lgd.el_by_grade || [];
    const vals = grades.map((g) => num(g.el_rate) * 100);
    plot("chart-el-grade", [{
      x: grades.map((g) => g.grade), y: vals, type: "bar",
      marker: { color: "#0072B2" },
      text: grades.map((g) => fmtPct(g.el_rate, 2)), textposition: "outside",
    }], {
      xaxis: { title: "Lender grade" },
      yaxis: { title: "Expected loss (% of EAD)", range: [0, Math.max(...vals) * 1.2] },
      showlegend: false,
    });
  }

  function drawHist(name, centers, counts, marks) {
    const shapes = [];
    const annotations = [];
    [[marks.el, "EL", "#14202e", "solid"],
     [marks.var99, "VaR 99%", "#E69F00", "dash"],
     [marks.var999, "VaR 99.9%", "#D55E00", "dash"],
     [marks.es99, "ES 99%", "#009E73", "dot"]].forEach(([p, label, color, dash]) => {
      if (num(p) == null) return;
      const x = p * 100;
      shapes.push({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1,
        line: { color: color, width: 1.6, dash: dash } });
      annotations.push({ x: x, yref: "paper", y: 1.02, text: label, showarrow: false,
        font: { size: 11, color: color }, textangle: -90, yanchor: "bottom" });
    });
    plot("chart-loss", [{
      x: centers, y: counts, type: "bar", marker: { color: "#0072B2" },
      hovertemplate: "%{x:.1f}% of EAD<br>%{y} paths<extra></extra>",
    }], {
      bargap: 0.02,
      xaxis: { title: "Portfolio loss (% of EAD)", range: [0, Math.max(...centers) * 1.02] },
      yaxis: { title: "Simulated paths" },
      showlegend: false, shapes: shapes, annotations: annotations,
      margin: { l: 70, r: 20, t: 40, b: 55 },
    });
    void name;
  }

  function histToCenters(h) {
    const e = h.bin_edges_pct_ead;
    return e.slice(0, -1).map((v, i) => (v + e[i + 1]) / 2);
  }

  function chartSimulation(d) {
    const st = d.simulation.statistics || {};
    const rh = d.rho_histograms || {};
    const sel = $("#rho-select");
    const labels = rh.labels || [];
    const readout = $("#rho-readout");

    function updateReadout(label) {
      if (!readout || !rh.rho) return;
      const j = labels.indexOf(label);
      if (j >= 0) readout.textContent = "ρ = " + num(rh.rho[j]).toFixed(3);
    }

    if (sel && labels.length) {
      sel.innerHTML = labels.map((l) => '<option value="' + l + '">' + l + "</option>").join("");
      sel.value = rh.base_label || labels[Math.floor(labels.length / 2)];
      const draw = () => {
        const h = (rh.histograms || {})[sel.value];
        if (!h) return;
        drawHist(sel.value, histToCenters(h), h.counts, h.exact || {});
        updateReadout(sel.value);
      };
      sel.addEventListener("change", draw);
      draw();
    } else if (d.loss_histogram) {
      const h = d.loss_histogram;
      drawHist("base case", histToCenters(h), h.counts, {
        el: st["expected_loss_pct_ead"],
        var99: st["var_0.99_pct_ead"],
        var999: st["var_0.999_pct_ead"],
        es99: st["es_0.99_pct_ead"],
      });
    }

    const sens = d.simulation.rho_sensitivity || [];
    plot("chart-rho", [
      { x: sens.map((r) => r.label), y: sens.map((r) => num(r["expected_loss_pct_ead"]) * 100),
        type: "bar", name: "Expected loss", marker: { color: "#0072B2" } },
      { x: sens.map((r) => r.label), y: sens.map((r) => num(r["var_0.999_pct_ead"]) * 100),
        type: "bar", name: "VaR 99.9%", marker: { color: "#56B4E9" } },
      { x: sens.map((r) => r.label), y: sens.map((r) => num(r["es_0.99_pct_ead"]) * 100),
        type: "bar", name: "ES 99%", marker: { color: "#E69F00" } },
      { x: sens.map((r) => r.label), y: sens.map((r) => num(r["es_0.999_pct_ead"]) * 100),
        type: "bar", name: "ES 99.9%", marker: { color: "#D55E00" } },
    ], {
      barmode: "group",
      xaxis: { title: "Asset correlation scenario", tickangle: -20 },
      yaxis: { title: "Loss (% of EAD)", range: [0, null] },
      margin: { l: 60, r: 20, t: 40, b: 90 },
    });
  }

  // ------------------------------------------------------------------ tables
  function renderTables(d) {
    const models = (d.pd.models || []).filter((m) => (m.sample || "oot") === "oot");
    const mt = $("#model-table tbody");
    if (mt) {
      mt.innerHTML = models.map((m) =>
        "<tr><td>" + m.model + (m.model === d.pd.champion ? " ★" : "") + "</td>" +
        "<td>" + fmtNum(m.auc) + "</td><td>" + fmtNum(m.gini) + "</td>" +
        "<td>" + fmtNum(m.ks) + "</td><td>" + (m.n_features != null ? m.n_features : "—") + "</td></tr>"
      ).join("");
    }

    const lt = $("#lgd-table tbody");
    if (lt) {
      lt.innerHTML = (d.lgd.model_comparison || []).map((m) =>
        "<tr><td>" + m.model + (m.model === d.lgd.best_model ? " ★" : "") + "</td>" +
        "<td>" + fmtNum(m.mae) + "</td><td>" + fmtNum(m.rmse) + "</td>" +
        "<td>" + fmtNum(m.r2) + "</td></tr>"
      ).join("");
    }

    const ct = $("#cluster-table tbody");
    if (ct) {
      ct.innerHTML = (d.clustering.kmeans_profile || []).map((s) =>
        "<tr><td>k-means " + s.segment + "</td><td>" + fmtInt(s.n) + "</td>" +
        "<td>" + fmtPct(s.share) + "</td><td>" + fmtPct(s.default_rate) + "</td></tr>"
      ).join("");
    }

    const ft = $("#forecast-table tbody");
    if (ft) {
      ft.innerHTML = ((d.forecast && d.forecast.benchmark) || []).map((r) =>
        "<tr><td>" + String(r.method).replace(/_/g, " ") + "</td>" +
        "<td>" + fmtNum(r.rmse) + "</td><td>" + fmtNum(r.mae) + "</td>" +
        "<td>" + fmtNum(r.mase) + "</td></tr>"
      ).join("");
    }

    const ul = $("#limitations-list");
    if (ul) {
      ul.innerHTML = (d.limitations || []).map((t) => "<li>" + t + "</li>").join("");
    }
  }

  // ---------------------------------------------------------------- interactive widgets
  function hasWebGL() {
    try {
      const canvas = document.createElement('canvas');
      return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
    } catch(e) { return false; }
  }

  function chartRiskSurface(d) {
    const el = document.getElementById('chart-risk-surface');
    if (!el) return;
    const surf = (d.interactive && d.interactive.risk_surface) || null;
    if (!surf) { showStaticFallback(el); return; }
    if (window.__plotlyLoadFailed || typeof window.Plotly === 'undefined') { showStaticFallback(el); return; }
    const metricSel = document.getElementById('risk-surface-metric');
    const baseInfo = document.getElementById('risk-surface-base');
    if (baseInfo) baseInfo.textContent = `Base: ρ=${surf.base_case.rho}, PD mult=${surf.base_case.pd_mult} (simulation grid, ${surf.n_paths_per_point.toLocaleString()} paths/point)`;
    
    function draw() {
      const metric = metricSel ? metricSel.value : 'var_0.999';
      const rhos = surf.rhos;
      const pds = surf.pd_multipliers;
      const z = Array(pds.length).fill().map(() => Array(rhos.length).fill(0));
      surf.grid.rho.forEach((rho, idx) => {
        const ri = rhos.indexOf(rho);
        const pj = pds.indexOf(surf.grid.pd_mult[idx]);
        if (ri >=0 && pj >=0) z[pj][ri] = num(surf.grid[metric]) / (d.simulation.total_ead || 1) * 100;
      });
      // Fix the mapping
      const z2 = Array(pds.length).fill().map((_,pj) => Array(rhos.length).fill().map((_,ri) => {
        // find matching
        for (let k=0; k<surf.grid.rho.length; k++) {
          if (Math.abs(surf.grid.rho[k]-rhos[ri]) < 1e-9 && Math.abs(surf.grid.pd_mult[k]-pds[pj]) < 1e-9) {
            return num(surf.grid[metric]) / (d.simulation.total_ead || 1) * 100;
          }
        }
        return 0;
      }));
      if (!hasWebGL()) {
        // 2D heatmap fallback
        try {
          Plotly.newPlot(el, [{
            z: z2, x: rhos, y: pds, type: 'heatmap', colorscale: 'YlOrRd'
          }], Object.assign(themed(), {title: `${metric} (% of EAD) - 2D fallback`, xaxis:{title:'ρ'}, yaxis:{title:'PD mult'}}), PLOTLY_CONFIG);
        } catch(e) { showStaticFallback(el); }
        return;
      }
      try {
        Plotly.newPlot(el, [{
          type: 'surface',
          z: z2,
          x: rhos,
          y: pds,
          colorscale: 'YlOrRd',
          showscale: true
        }], Object.assign(themed(), {
          title: `${metric.replace('_',' ').toUpperCase()} (% of EAD)`,
          scene: {xaxis:{title:'ρ'}, yaxis:{title:'PD mult'}, zaxis:{title:'% of EAD'}},
          margin: {l:0,r:0,t:40,b:0}
        }), PLOTLY_CONFIG);
        // Mark base case
      } catch(e) { showStaticFallback(el); }
    }
    if (metricSel) metricSel.addEventListener('change', draw);
    draw();
  }

  function chartScenarioExplorer(d) {
    const el = document.getElementById('chart-scenario');
    if (!el) return;
    const scen = (d.interactive && d.interactive.scenario_explorer) || null;
    if (!scen) { showStaticFallback(el); return; }
    if (window.__plotlyLoadFailed || typeof window.Plotly === 'undefined') { showStaticFallback(el); return; }
    
    const rsel = document.getElementById('scen-rho');
    const psel = document.getElementById('scen-pd');
    const lsel = document.getElementById('scen-lgd');
    const rhos = [...new Set(scen.scenarios.map(s=>s.rho))].sort((a,b)=>a-b);
    const pds = [...new Set(scen.scenarios.map(s=>s.pd_mult))].sort((a,b)=>a-b);
    const lgds = [...new Set(scen.scenarios.map(s=>s.lgd_mult))].sort((a,b)=>a-b);
    if (rsel) rsel.innerHTML = rhos.map(r=>`<option value="${r}">${r.toFixed(3)}</option>`).join('');
    if (psel) psel.innerHTML = pds.map(p=>`<option value="${p}">${p.toFixed(2)}x</option>`).join('');
    if (lsel) lsel.innerHTML = lgds.map(l=>`<option value="${l}">${l.toFixed(2)}x</option>`).join('');
    
    function update() {
      const rk = parseFloat(rsel.value), pk = parseFloat(psel.value), lk = parseFloat(lsel.value);
      const key = `rho${rk.toFixed(3)}_pd${pk.toFixed(2)}_lgd${lk.toFixed(2)}`;
      const h = scen.histograms[key] || Object.values(scen.histograms)[0];
      const edges = h.bin_edges_pct_ead;
      const centers = edges.slice(0,-1).map((v,i)=>(v+edges[i+1])/2);
      const exact = h.exact;
      document.getElementById('scen-el').textContent = ((exact['expected_loss_pct_ead']||0)*100).toFixed(2);
      document.getElementById('scen-ul').textContent = ((exact['unexpected_loss_pct_ead']||0)*100).toFixed(2);
      document.getElementById('scen-var99').textContent = ((exact['var_0.99_pct_ead']||0)*100).toFixed(2);
      document.getElementById('scen-var999').textContent = ((exact['var_0.999_pct_ead']||0)*100).toFixed(2);
      document.getElementById('scen-es99').textContent = ((exact['es_0.99_pct_ead']||0)*100).toFixed(2);
      document.getElementById('scen-es999').textContent = ((exact['es_0.999_pct_ead']||0)*100).toFixed(2);
      try {
        Plotly.react ? Plotly.react(el, [{
          x: centers, y: h.counts, type: 'bar', marker:{color:'#0072B2'}
        }], Object.assign(themed(), {bargap:0.02, xaxis:{title:'Loss (% of EAD)'}, yaxis:{title:'Paths'}, margin:{l:60,r:20,t:20,b:50}}), PLOTLY_CONFIG) :
        Plotly.newPlot(el, [{
          x: centers, y: h.counts, type: 'bar', marker:{color:'#0072B2'}
        }], Object.assign(themed(), {bargap:0.02, xaxis:{title:'Loss (% of EAD)'}, yaxis:{title:'Paths'}, margin:{l:60,r:20,t:20,b:50}}), PLOTLY_CONFIG);
      } catch(e) { showStaticFallback(el); }
    }
    if (rsel) rsel.addEventListener('change', update);
    if (psel) psel.addEventListener('change', update);
    if (lsel) lsel.addEventListener('change', update);
    update();
  }

  function chartApplicantSimulator(d) {
    const ctrl = document.getElementById('app-sim-controls');
    if (!ctrl) return;
    const simd = (d.interactive && d.interactive.applicant_simulator) || d.pd;
    const points = simd.points_table || d.pd.scorecard_points || [];
    const scaler = simd.scaler || d.pd.scorecard_scaler || {base_score:600, base_odds:50, pdo:20};
    if (!points.length) return;
    const vars = [...new Set(points.map(p=>p.variable))];
    const state = {};
    vars.forEach(v => {
      const bins = [...new Set(points.filter(p=>p.variable===v).map(p=>p.bin))];
      state[v] = bins[0];
      const div = document.createElement('div');
      div.className = 'sim-ctrl';
      div.innerHTML = `<label for="app-${v}">${v.replace(/_/g,' ')}</label><select id="app-${v}">${bins.map(b=>`<option value="${b.replace(/"/g,'&quot;')}">${b}</option>`).join('')}</select>`;
      ctrl.appendChild(div);
      div.querySelector('select').addEventListener('change', (e) => { state[v] = e.target.value; compute(); });
    });
    function compute() {
      let totalPoints = scaler.offset !== undefined ? (scaler.base_score - scaler.factor * Math.log(scaler.base_odds)) : 0;
      // Use formula: score = base_score - PDO/ln(2)*ln(odds) roughly; easier use points table sum
      let pts = 0;
      vars.forEach(v => {
        const row = points.find(p => p.variable===v && p.bin === state[v]);
        if (row) pts += num(row.points) || 0;
      });
      const factor = scaler.factor || (scaler.pdo / Math.log(2));
      const offset = scaler.offset !== undefined ? scaler.offset : (scaler.base_score - factor * Math.log(scaler.base_odds));
      // points table gives per-variable points; total score
      const score = offset + pts;
      document.getElementById('app-score').textContent = Math.round(score);
      const lnOddsGood = (score - offset) / factor;
      const pd = 1.0 / (1.0 + Math.exp(lnOddsGood));
      document.getElementById('app-pd').textContent = (pd*100).toFixed(2) + '%';
      document.getElementById('app-pct').textContent = '—'; // descriptive
    }
    compute();
    // Percentile chart from real test distribution
    const pel = document.getElementById('chart-app-percentile');
    if (pel && typeof Plotly !== 'undefined' && !window.__plotlyLoadFailed && simd.test_score_hist) {
      try {
        const bins = simd.test_score_bins || [];
        const hist = simd.test_score_hist || [];
        const x = [];
        for (let i=0; i<hist.length && i<bins.length-1; i++) {
          x.push((bins[i]+bins[i+1])/2.0);
        }
        Plotly.newPlot(pel, [{x:x, y:hist, type:'bar', marker:{color:'#2f5d62'}}], Object.assign(themed(), {xaxis:{title:'Score'}, yaxis:{title:'Count'}}), {displayModeBar:false});
      } catch(e) { if (typeof showStaticFallback==='function') showStaticFallback(pel); }
    } else if (pel && typeof showStaticFallback==='function') { showStaticFallback(pel); }
    // Redraw charts with the right palette if the OS theme flips.
    if (window.matchMedia) {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      const onChange = () => render(d);
      if (mq.addEventListener) mq.addEventListener("change", onChange);
    }
  }

  fetch(DATA_URL, { cache: "no-cache" })
    .then((r) => {
      if (!r.ok) throw new Error("data.json " + r.status);
      return r.json();
    })
    .then(render)
    .catch((err) => {
      console.error("Could not load data.json:", err);
      const banner = $("#data-error");
      if (banner) banner.hidden = false;
      // Static fallbacks for every chart host.
      $$(".chart-host").forEach(showStaticFallback);
    });
})();
