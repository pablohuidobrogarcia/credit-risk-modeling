with open('docs/assets/site.js','r',encoding='latin-1') as f:
    s=f.read()
old = """  function chartCutoffExplorer(d) {
    const slider = document.getElementById('cutoff-slider');
    if (!slider) return;
    const ksCutoff = num(d.pd.cutoff) || 0.5;
    const readout = document.getElementById('cutoff-readout');
    const ksMarker = document.getElementById('cutoff-ks-marker');
    if (ksMarker) ksMarker.textContent = `KS cutoff (train): ${ksCutoff.toFixed(4)}`;
    const confusion = d.pd.confusion_oot || {};
    function update() {
      const cut = parseFloat(slider.value);
      if (readout) readout.textContent = `Cutoff score: ${cut}`;
      // Approximate from confusion if using PD cutoff
      const tp = confusion.tp||0, fp = confusion.fp||0, fn = confusion.fn||0, tn = confusion.tn||0;
      const total = tp+fp+fn+tn || 1;
      const approved = tp+fp;
      document.getElementById('cut-approval').textContent = (approved/total*100).toFixed(1)+'%';
      document.getElementById('cut-bad-approved').textContent = (approved>0?(fp/(tp+fp)*100).toFixed(1):'0')+'%';
      document.getElementById('cut-cm').textContent = `TP:${tp} FP:${fp} FN:${fn} TN:${tn}`;
      document.getElementById('cut-default').textContent = ((tp+fn)/total*100).toFixed(1)+'%';
    }
    slider.addEventListener('input', update);
    update();
    const cel = document.getElementById('chart-cutoff');
    if (cel && typeof Plotly !== 'undefined' && !window.__plotlyLoadFailed) {
      try {
        Plotly.newPlot(cel, [
          {x:[0,1], y:[0,1], type:'scatter', mode:'lines', name:'Goods'},
          {x:[0,1], y:[0,0.8], type:'scatter', mode:'lines', name:'Bads'}
        ], Object.assign(themed(), {xaxis:{title:'Score'}, yaxis:{title:'Density'}, margin:{l:50,r:20,t:10,b:50}}), PLOTLY_CONFIG);
      } catch(e) {}
    }
  }"""
new = """  function chartCutoffExplorer(d) {
    const slider = document.getElementById('cutoff-slider');
    if (!slider) return;
    const cdata = (d.interactive && d.interactive.cutoff_explorer) || d.pd.cutoff_explorer || {};
    const ksCutoff = num(cdata.ks_cutoff || d.pd.cutoff) || 0.5;
    const readout = document.getElementById('cutoff-readout');
    const ksMarker = document.getElementById('cutoff-ks-marker');
    if (ksMarker) ksMarker.textContent = `KS cutoff (train): ${ksCutoff.toFixed(4)}`;
    const confusion = cdata.confusion_oot || d.pd.confusion_oot || {};
    const bins = cdata.bins || [];
    const goods = cdata.goods_counts || [];
    const bads = cdata.bads_counts || [];
    function update() {
      const cut = parseFloat(slider.value);
      if (readout) readout.textContent = `Cutoff score: ${cut}`;
      const tp = confusion.tp||0, fp = confusion.fp||0, fn = confusion.fn||0, tn = confusion.tn||0;
      const total = tp+fp+fn+tn || 1;
      const approved = tp+fp;
      const elApp = document.getElementById('cut-approval');
      if (elApp) elApp.textContent = (approved/total*100).toFixed(1)+'%';
      const elBad = document.getElementById('cut-bad-approved');
      if (elBad) elBad.textContent = (approved>0?(fp/(tp+fp)*100).toFixed(1):'0')+'%';
      const elCm = document.getElementById('cut-cm');
      if (elCm) elCm.textContent = `TP:${tp} FP:${fp} FN:${fn} TN:${tn}`;
      const elDef = document.getElementById('cut-default');
      if (elDef) elDef.textContent = ((tp+fn)/total*100).toFixed(1)+'%';
    }
    slider.addEventListener('input', update);
    update();
    const cel = document.getElementById('chart-cutoff');
    if (cel) {
      if (goods.length === 0 || bads.length === 0 || bins.length < 2) {
        showStaticFallback(cel);
        return;
      }
      if (typeof Plotly !== 'undefined' && !window.__plotlyLoadFailed) {
        try {
          const x = [];
          for (let i=0; i<goods.length; i++) {
            x.push((bins[i] + bins[i+1])/2.0);
          }
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
