with open('docs/assets/site.js','r',encoding='latin-1') as f:
    s=f.read()
start = s.find('// Percentile chart')
if start == -1:
    print('no')
    raise SystemExit
end = s.find('\n    //', start+1)
if end == -1:
    end = s.find('\n  function', start+1)
if end == -1:
    end = start + 500
block = s[start:end]
new_block = """// Percentile chart from real test distribution
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
s2 = s.replace(block, new_block)
with open('docs/assets/site.js','w',encoding='latin-1') as f:
    f.write(s2)
print('ok')
