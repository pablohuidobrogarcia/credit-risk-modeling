with open('docs/assets/site.js','rb') as f:
    data = f.read()
s = data.decode('latin-1')
idx = s.find('.then(render)')
if idx >= 0:
    start = max(0, idx-1500)
    sys = __import__('sys')
    sys.stdout.write(s[start:idx+100])
else:
    print('no')
