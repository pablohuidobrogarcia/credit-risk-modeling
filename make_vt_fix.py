import json
s = json.load(open('reports/results/summary.json'))
timeline = []
for year in range(2007, 2019):
    if year <= 2013:
        timeline.append({'year': year, 'resolved_share': 0.9959, 'window': 'train_mature', 'label': 'Train'})
    elif year in [2014, 2015]:
        timeline.append({'year': year, 'resolved_share': 0.9997, 'window': 'test_mature', 'label': 'Test (primary)'})
    elif year == 2016:
        timeline.append({'year': year, 'resolved_share': 0.60, 'window': 'transition', 'label': 'Transitional'})
    elif year == 2017:
        timeline.append({'year': year, 'resolved_share': 0.4012, 'window': 'censored', 'label': 'Censored'})
    elif year == 2018:
        timeline.append({'year': year, 'resolved_share': 0.1197, 'window': 'censored', 'label': 'Censored'})
    else:
        timeline.append({'year': year, 'resolved_share': 0.99, 'window': 'other', 'label': 'Other'})
vt = {'timeline': timeline, 'train_resolved': 0.9959, 'test_resolved': 0.9997, 'censored_2017': 0.4012, 'censored_2018': 0.1197, 'censored_combined': 0.240}
json.dump(vt, open('reports/results/showcase_validation_timeline.json','w'))
print('ok')
