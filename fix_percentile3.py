with open('docs/assets/site.js','r',encoding='latin-1') as f:
    s=f.read()
old = """    // Percentile chart
    const pel = document.getElementById('chart-app-percentile');
    if (pel && typeof Plotly !== 'undefined' && !window.__plotlyLoadFailed) {
      try {
        Plotly.newPlot(pel, [{x:[400,500,600,700,800], y:[0.3,0.2,0.15,0.1,0.05], type:'scatter', mode:'lines'}], Object.assign(themed(), {xaxis:{title:'Score'}, yaxis:{title:'Density'}}), {displayModeBar:false});
      } catch(e) {}
    }"""
new = """    // Percentile chart from real test distribution
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
    } else if (pel && typeof showStaticFallback==='function') { showStaticFallback(pel); }"""
if old in s:
    s2 = s.replace(old,new)
    with open('docs/assets/site.js','w',encoding='latin-1') as f:
        f.write(s2)
    print('ok')
else:
    print('no')
