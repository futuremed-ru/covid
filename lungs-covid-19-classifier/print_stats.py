import pandas as pd

df = pd.read_csv('data/all.csv')

covid = df[df.apply(lambda x: 'COVID-19' in str(x), axis=1)]
print('Unique COVID-19 images:', len(covid))
print('Unique COVID-19 patients:', len(covid['patientid'].unique()))

other = df[df.apply(lambda x: 'COVID-19' not in str(x), axis=1)]
print('Unique other images:', len(other))
print('Unique other patients:', len(other['patientid'].unique()))
