import numpy as np
import pandas as pd

df = pd.read_csv('data/new_github/new_github_descr.csv')
new_github_label_names = ['COVID-19', 'COVID-19, ARDS', 'SARS', 'No Finding']
labels = list()
for index, row in df.iterrows():
    labels.append({'filename': row['filename'], 'label': row['finding']})
df_new_github = pd.DataFrame.from_dict(labels)
df_new_github.to_csv('data/new_github/labels.csv', index=False)

covid_label_names = ['COVID-19', 'Other']
df_covid = pd.read_csv('data/new_github/covid_New_Github_Outputs.csv', usecols=['filename', 'ensemble out'])
preds = list()
for i, row in df_covid.iterrows():
    probs = np.asarray([float(x) for x in row['ensemble out'].split()])
    predicted = covid_label_names[probs.argmax(axis=0)]
    preds.append(predicted)
df_covid['predicted'] = preds
df_covid.to_csv('data/new_github/covid.csv', index=False)

print(df.head())
print(len(df_covid))
print(len(df_new_github))

filenames = list()
labels = list()
ensemble_outs = list()
predicteds = list()
i = 0
while i < len(df_new_github):
    filenames.append(df_new_github['filename'][i])
    labels.append(df_new_github['label'][i])
    ensemble_outs.append(df_covid['ensemble out'][i])
    predicteds.append(df_covid['predicted'][i])
    i += 1
df_full = pd.DataFrame()
df_full['filename'] = filenames
df_full['label'] = labels
df_full['ensemble_out'] = ensemble_outs
df_full['predicted'] = predicteds
df_full.to_csv('data/new_github/new_github.csv', index=False)
