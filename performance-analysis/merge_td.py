import numpy as np
import pandas as pd

td_label_names = ['normal', 'abnormal']
df = pd.read_csv('data/td/td-lungs-stats-no-duplicates.csv')
labels = list()
for index, row in df.iterrows():
    labels.append({'filename': row['filename'], 'label': row['finding']})
df_td = pd.DataFrame.from_dict(labels)
df_td.to_csv('data/td/labels.csv', index=False)

covid_label_names = ['COVID-19', 'Other']
df_covid = pd.read_csv('data/td/covid_TD_Outputs.csv', usecols=['filename', 'ensemble out'])
preds = list()
for i, row in df_covid.iterrows():
    probs = np.asarray([float(x) for x in row['ensemble out'].split()])
    predicted = covid_label_names[probs.argmax(axis=0)]
    preds.append(predicted)
df_covid['predicted'] = preds
df_covid.to_csv('data/td/covid.csv', index=False)


print(df.head())
print(len(df_covid))
print(len(df_td))

filenames = list()
labels = list()
ensemble_outs = list()
predicteds = list()
predicted_chestxray14s = list()
predicted_chestxray14_labels = list()
descriptions = list()
i = 0
while i < len(df_td):
    filenames.append(df_td['filename'][i])
    labels.append(df_td['label'][i])
    ensemble_outs.append(df_covid['ensemble out'][i])
    predicteds.append(df_covid['predicted'][i])
    predicted_chestxray14s.append(df['predicted'][i])
    predicted_chestxray14_labels.append(df['predicted_labels'][i])
    description = df['description'][i]
    if str(df['Unnamed: 6'][i]) != 'nan':
        description += ' ' + df['Unnamed: 6'][i]
    elif str(df['Unnamed: 7'][i]) != 'nan':
        description += ' ' + df['Unnamed: 7'][i]
    descriptions.append(description)
    i += 1
df_full = pd.DataFrame()
df_full['filename'] = filenames
df_full['label'] = labels
df_full['ensemble_out'] = ensemble_outs
df_full['predicted'] = predicteds
df_full['predicted_chestxray14'] = predicted_chestxray14s
df_full['predicted_chestxray14_labels'] = predicted_chestxray14_labels
df_full['description'] = descriptions
df_full.to_csv('data/td/td.csv', index=False)
