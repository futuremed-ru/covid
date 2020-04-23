import pandas as pd

covid_label_names = ['COVID-19', 'Other']
new_github_label_names = ['COVID-19', 'COVID-19, ARDS', 'SARS', 'No Finding']
chestxray14_label_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                           'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding']
td_label_names = ['normal', 'abnormal']

gt_covid_count = 0
print('New github stats:')
df_new_github = pd.read_csv('data/new_github.csv')
print('Total images:', len(df_new_github))
print('Total COVID-19 predicted true:', len(df_new_github[df_new_github['predicted'] == 'COVID-19']))
for abnormality in new_github_label_names:
    df_abnormality = df_new_github[df_new_github['label'] == abnormality]
    count = len(df_abnormality[df_abnormality['predicted'] == 'COVID-19'])
    print('COVID-19 predicted {}: {} / {}'.format(abnormality, count, len(df_abnormality)))
    if 'COVID-19' in abnormality:
        gt_covid_count += len(df_abnormality)
print()

print('Chest-X-Ray-14 random subset stats:')
df_chestxray14 = pd.read_csv('data/chestxray14.csv').sample(n=gt_covid_count - (len(df_new_github) - gt_covid_count))
print('Total images:', len(df_chestxray14))
print('Total COVID-19 predicted true:', len(df_chestxray14[df_chestxray14['predicted'] == 'COVID-19']))
print()

print('Chest-X-Ray-14 stats:')
df_chestxray14 = pd.read_csv('data/chestxray14.csv')
print('Total images:', len(df_chestxray14))
print('Total COVID-19 predicted true:', len(df_chestxray14[df_chestxray14['predicted'] == 'COVID-19']))
for abnormality in chestxray14_label_names:
    df_abnormality = df_chestxray14[df_chestxray14.apply(lambda x: abnormality in str(x['label']), axis=1)]
    count = len(df_abnormality[df_abnormality['predicted'] == 'COVID-19'])
    print('COVID-19 predicted {}: {} / {}'.format(abnormality, count, len(df_abnormality)))
print()

print('TD stats:')
df_td = pd.read_csv('data/td.csv')
print('Total images:', len(df_td))
print('Total COVID-19 predicted true:', len(df_td[df_td['predicted'] == 'COVID-19']))
for abnormality in td_label_names:
    df_abnormality = df_td[df_td['label'] == abnormality]
    count = len(df_abnormality[df_abnormality['predicted'] == 'COVID-19'])
    print('COVID-19 predicted {}: {} / {}'.format(abnormality, count, len(df_abnormality)))
print()

tp = 61
fp = 5
tn = 63
fn = 7

accuracy = (tp + tn) / (tp + tn + fp + fn)
print('Accuracy:', accuracy)
precision = tp / (tp + fp)
print('Precision:', precision)
recall = tp / (tp + fn)
print('Recall:', recall)
specificity = tn / (tn + fp)
print('Specificity:', specificity)
f1 = 2 * precision * recall / (precision + recall)
print('F1 score:', f1)
print()

tp = 61
fp = 2788
tn = 88692
fn = 7

accuracy = (tp + tn) / (tp + tn + fp + fn)
print('Accuracy:', accuracy)
precision = tp / (tp + fp)
print('Precision:', precision)
recall = tp / (tp + fn)
print('Recall:', recall)
specificity = tn / (tn + fp)
print('Specificity:', specificity)
f1 = 2 * precision * recall / (precision + recall)
print('F1 score:', f1)
