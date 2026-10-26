with open('docs/assets/site.js','r',encoding='latin-1') as f:
    s=f.read()
idx = s.find('onChange = () => render')
if idx >= 0:
    start = s.rfind('function ', 0, idx)
    print(s[start:idx+80])
else:
    print('no')
