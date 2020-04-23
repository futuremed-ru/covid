import os

import numpy as np
import pandas as pd
import skimage.io
import skimage.color

used_for_training_filenames = os.listdir('data/images')
github_filenames = os.listdir('data/github/images')

new_filenames = []
for filename in github_filenames:
    if filename not in used_for_training_filenames:
        new_filenames.append(filename)

print('# of new images in folder:', len(new_filenames))

# found manually
de_facto_lateral_image_filenames = ['covid-19-infection-exclusive-gastrointestinal-symptoms-l.png', 'pneumocystis-jiroveci-pneumonia-4-L.png']
df = pd.read_csv('data/github/metadata.csv')
new_df = pd.DataFrame(columns=df.columns)
for i, row in df.iterrows():
    if row['filename'] in new_filenames and row['view'] in ['PA', 'AP'] and row['filename'] not in de_facto_lateral_image_filenames:
        new_df = new_df.append(row, ignore_index=True)
print('# of new frontal chest x-ray images with descriptions:', len(new_df))

src_dir = 'data/github/images'
dst_dir = 'data/new_github_images'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

for i, row in new_df.iterrows():
    img = skimage.io.imread(os.path.join(src_dir, row['filename']))
    img = skimage.color.rgb2gray(img)
    img = (255 * img).astype(np.uint8)
    skimage.io.imsave(os.path.join(dst_dir, row['filename']), img)
    print('Processed: {}/{}'.format(i + 1, len(new_df)), end='\r')
print('Processed: {}/{}'.format(i + 1, len(new_df)))
print('Test images are saved to:', dst_dir)

test_csv_path = 'data/new_github_decr.csv'
new_df.to_csv(test_csv_path, index=False)
print('Test metadata is saved to:', test_csv_path)
