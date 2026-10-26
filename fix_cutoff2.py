with open('docs/assets/site.js','r',encoding='latin-1') as f:
    s=f.read()
old = """    const cel = document.getElementById('chart-cutoff');
    if (cel && typeof Plotly !== 'undefined' && !window.__plotlyLoadFailed) {
      try {
        Plotly.newPlot(cel, [
          {x:[0,1], y:[0,1], type:'scatter', mode:'lines', name:'Goods'},
          {x:[0,1], y:[0,0.8], type:'scatter', mode:'lines', name:'Bads'}
        ], Object.assign(themed(), {xaxis:{title:'Score'}, yaxis:{title:'Density'}, margin:{l:50,r:20,t:10,b:50}}), PLOTLY_CONFIG);
      } catch(e) {}
    }
  }"""
new = """    const cel = document.getElementById('chart-cutoff');
    if (cel) {
      const cdata = (d.interactive && d.interactive.cutoff_explorer) || {};
      const bins = cdata.bins || [];
      const goods = cdata.goods_counts || [];
      const bads = cdata.bads_counts || [];
      if (goods.length === 0 || bads.length === 0 || bins.length < 2) {
        showStaticFallback(cel);
      } else if (typeof Plotly !== 'undefined' && !window.__plotlyLoadFailed) {
        try {
          const x = [];
          for (let i=0; i<goods.length; i++) x.push((bins[i]+bins[i+1])/2.0);
          Plotly.newPlot(cel, [
            {x:x, y:goods, type:'bar', name:'Goods', marker:{color:'#009E73'}},
            {x:x, y:bads, type:'bar', name:'Bads', marker:{color:'#D55E00'}}
          ], Object.assign(themed(), {barmode:'group', xaxis:{title:'Score'}, yaxis:{title:'Count'}, margin:{l:50,r:20,t:10,b:50}}), PLOTLY_CONFIG);
        } catch(e) { showStaticFallback(cel); }
      } else {
        showStaticFallback(cel);
      }
    }
  }"""
if old in s:
    s2 = s.replace(old,new)
    with open('docs/assets/site.js','w',encoding='latin-1') as f:
        f.write(s2)
    print('ok')
else:
    print('no')
