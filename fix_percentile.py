with open('docs/assets/site.js', encoding='latin-1') as f:
    content = f.read()

old = """    // Percentile chart
    const pel = document.getElementById('chart-app-percentile');
    if (pel && typeof Plotly !== 'undefined' && !window.__plotlyLoadFailed) {
      try {
        Plotly.newPlot(pel, [{x:['P10','P25','P50','P75','P90'], y:[0.3,0.2,0.15,0.1,0.05], type:'bar', marker:{color:'#2f5d62'}}], {margin:{t:10,r:10,l:40,b:40}, yaxis:{title:'Share'}}, {displayModeBar:false});
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
        Plotly.newPlot(pel, [{x:x, y:hist, type:'bar', marker:{color:'#2f5d62'}}], {margin:{t:10,r:10,l:40,b:40}, xaxis:{title:'Score'}, yaxis:{title:'Count'}}, {displayModeBar:false});
      } catch(e) { if (typeof showStaticFallback==='function') showStaticFallback(pel); }
    } else if (pel && typeof showStaticFallback==='function') { showStaticFallback(pel); }"""

if old in content:
    content = content.replace(old, new)
    with open('docs/assets/site.js', 'w', encoding='latin-1') as f:
        f.write(content)
    print('ok')
else:
    print('no')
