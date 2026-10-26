with open('docs/assets/site.js', 'r', encoding='utf-8', errors='ignore') as f:
    s = f.read()

old = """  function renderDisplay(d) {
    const disp = d.display || {};
    $$('[data-field^="display."]').forEach((el) => {
      const key = el.getAttribute("data-field").slice("display.".length);
      if (Object.prototype.hasOwnProperty.call(disp, key)) {
        el.textContent = disp[key];
      }
    });
  }

  // --------------------------------------------------------- Plotly theming"""

new = """  function renderDisplay(d) {
    const disp = d.display || {};
    $$('[data-field^="display."]').forEach((el) => {
      const key = el.getAttribute("data-field").slice("display.".length);
      if (Object.prototype.hasOwnProperty.call(disp, key)) {
        el.textContent = disp[key];
      }
    });
  }

  function render(d) {
    try { renderDisplay(d); } catch(e) { console.error(e); }
    try { renderTables(d); } catch(e) { console.error(e); }
    try { chartCensoring(d); } catch(e) { console.error(e); }
    try { chartAucWindow(d); } catch(e) { console.error(e); }
    try { chartModels(d); } catch(e) { console.error(e); }
    try { chartCalibration(d); } catch(e) { console.error(e); }
    try { chartElGrade(d); } catch(e) { console.error(e); }
    try { chartSimulation(d); } catch(e) { console.error(e); }
    try { chartRiskSurface(d); } catch(e) { console.error(e); }
    try { chartScenarioExplorer(d); } catch(e) { console.error(e); }
    try { chartApplicantSimulator(d); } catch(e) { console.error(e); }
    try { chartCutoffExplorer(d); } catch(e) { console.error(e); }
    try { chartLoanMap(d); } catch(e) { console.error(e); }
    try { chartValidationTimeline(d); } catch(e) { console.error(e); }
    document.documentElement.setAttribute("data-loaded", "true");
    if (window.matchMedia) {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      const onChange = () => { try { render(d); } catch(e) { console.error(e); } };
      if (mq.addEventListener) mq.addEventListener("change", onChange);
    }
  }

  // --------------------------------------------------------- Plotly theming"""

if old not in s:
    print('not found')
else:
    s2 = s.replace(old, new)
    with open('docs/assets/site.js', 'w', encoding='utf-8') as f:
        f.write(s2)
    print('ok')
