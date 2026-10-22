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
    if (typeof window.Plotly === "undefined") {
      showStaticFallback(el);
      return;
    }
    window.Plotly.newPlot(el, traces, Object.assign(themed(), layout), PLOTLY_CONFIG);
  }

  function showStaticFallback(el) {
    if (el.dataset.fallbackShown === "1") return;
    const src = el.getAttribute("data-fallback-src");
    if (!src) return;
    el.dataset.fallbackShown = "1";
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
    plot("chart-censoring", traces, {
      barmode: "group",
      xaxis: { title: "Issue year", dtick: 1 },
      yaxis: { title: "Resolved share (%)", range: [0, 105] },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 90, y1: 90,
        line: { color: "#D55E00", width: 1.5, dash: "dash" } }],
      annotations: [{ xref: "paper", x: 1, y: 92, xanchor: "right", yanchor: "bottom",
        text: "90% resolved bar for the primary split", showarrow: false,
        font: { size: 11, color: "#D55E00" } }],
    });
  }

  function chartAucWindow(d) {
    const wc = d.pd.window_comparison || [];
    const ly = wc.map((r) => r.test_years);
    const aucs = wc.map((r) => num(r.auc));
    plot("chart-auc-window", [{
      x: wc.map((r, i) => (r.window || "window " + i) + " (" + (r.test_years || "") + ")"),
      y: aucs, type: "bar",
      marker: { color: ["#0072B2", "#D55E00"] },
      text: aucs.map((v) => fmtNum(v)), textposition: "outside",
    }], {
      yaxis: { title: "Out-of-time AUC", range: [0.5, 0.78] },
      showlegend: false,
      margin: { l: 60, r: 20, t: 30, b: 60 },
    });
    void ly;
  }

  function chartModels(d) {
    const models = (d.pd.models || []).filter((m) => (m.sample || "oot") === "oot");
    plot("chart-models", [{
      x: models.map((m) => m.auc), y: models.map((m) => m.model),
      type: "bar", orientation: "h",
      marker: { color: models.map((m) => (m.model === d.pd.champion ? "#009E73" : "#999999")) },
      text: models.map((m) => fmtNum(m.auc)), textposition: "outside",
    }], {
      xaxis: { title: "Out-of-time AUC (mature window)",
        range: [0.5, Math.max(0.62, ...models.map((m) => m.auc)) + 0.02] },
      yaxis: { automargin: true },
      showlegend: false,
      margin: { l: 130, r: 50, t: 20, b: 50 },
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
      { x: sens.map((r) => r.label), y: sens.map((r) => num(r["es_0.99_pct_ead"]) * 100),
        type: "bar", name: "ES 99%", marker: { color: "#E69F00" } },
      { x: sens.map((r) => r.label), y: sens.map((r) => num(r["es_0.999_pct_ead"]) * 100),
        type: "bar", name: "ES 99.9%", marker: { color: "#D55E00" } },
    ], {
      barmode: "group",
      xaxis: { title: "Asset correlation scenario", tickangle: -20 },
      yaxis: { title: "Loss (% of EAD)" },
      margin: { l: 60, r: 20, t: 40, b: 80 },
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

  // -------------------------------------------------------------------- boot
  function render(d) {
    renderDisplay(d);
    renderTables(d);
    chartCensoring(d);
    chartAucWindow(d);
    chartModels(d);
    chartCalibration(d);
    chartElGrade(d);
    chartSimulation(d);
    document.documentElement.setAttribute("data-loaded", "true");

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
