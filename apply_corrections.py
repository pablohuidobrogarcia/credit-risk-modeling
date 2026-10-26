with open('docs/assets/site.js','r',encoding='latin-1') as f:
    s=f.read()

# Applicant formula
old = """      const score = Math.round(offset + pts); // approximation
      document.getElementById('app-score').textContent = Math.round(score);
      // PD approx from scorecard
      const odds = Math.exp((scaler.base_score - score)/factor);
      const pd = 1/(1+odds);
      document.getElementById('app-pd').textContent = (pd*100).toFixed(2) + '%';"""
new = """      const score = offset + pts;
      document.getElementById('app-score').textContent = Math.round(score);
      const lnOddsGood = (score - offset) / factor;
      const pd = 1.0 / (1.0 + Math.exp(lnOddsGood));
      document.getElementById('app-pd').textContent = (pd*100).toFixed(2) + '%';"""
if old in s:
    s = s.replace(old, new)
    print('app ok')
else:
    print('app no')

with open('docs/assets/site.js','w',encoding='latin-1') as f:
    f.write(s)
