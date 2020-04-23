import numpy as np
import pandas as pd


covid_label_names = ['COVID-19', 'Other']
df = pd.read_csv('data/chestxray14/covid_Ensemble_Combined_Outputs.csv', usecols=['filename', 'ensemble out'])
df_covid = pd.concat([df, pd.read_csv('data/chestxray14/covid_Ensemble_Train_Outputs.csv', usecols=['filename', 'ensemble out'])], ignore_index=True)
preds = list()
for i, row in df_covid.iterrows():
    probs = np.asarray([float(x) for x in row['ensemble out'].split()])
    predicted = covid_label_names[probs.argmax(axis=0)]
    preds.append(predicted)
df_covid['predicted'] = preds
df_covid.to_csv('data/chestxray14/covid.csv', index=False)

chestxray14_label_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                           'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
df = pd.read_csv('data/chestxray14/Combined_Data_Entry.csv', usecols=['Image Index', 'Finding Labels'])
df = pd.concat([df, pd.read_csv('data/chestxray14/Train_Data_Entry.csv', usecols=['Image Index', 'Finding Labels'])], ignore_index=True)
labels = list()
for index, row in df.iterrows():
    labels.append({'filename': row['Image Index'], 'label': row['Finding Labels']})
df_chestxray14 = pd.DataFrame.from_dict(labels)
df_chestxray14.to_csv('data/chestxray14/labels.csv', index=False)

print(df.head())
print(len(df_covid))
print(len(df_chestxray14))

filenames = list()
labels = list()
ensemble_outs = list()
predicteds = list()
i = 0
while i < len(df_chestxray14):
    filenames.append(df_chestxray14['filename'][i])
    labels.append(df_chestxray14['label'][i])
    ensemble_outs.append(df_covid['ensemble out'][i])
    predicteds.append(df_covid['predicted'][i])
    i += 1
df_full = pd.DataFrame()
df_full['filename'] = filenames
df_full['label'] = labels
df_full['ensemble_out'] = ensemble_outs
df_full['predicted'] = predicteds
df_full.to_csv('data/chestxray14/chestxray14.csv', index=False)
