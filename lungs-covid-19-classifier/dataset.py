import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import torch
import torch.utils.data
import torchvision


label_names = ['COVID-19', 'Other']


class CovidDataset(torch.utils.data.Dataset):

    def __init__(self, annotations_df, images_dir, transform=None):
        labels = list()
        for index, row in annotations_df.iterrows():
            image_label = np.zeros(2)
            if 'COVID-19' in row['finding']:
                image_label[0] = 0.95
                image_label[1] = 0.05
            else:  # Other
                image_label[0] = 0.05
                image_label[1] = 0.95
            labels.append({'filename': row['filename'], 'label': image_label})
        self.labels = labels
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]['label']
        img_path = os.path.join(self.images_dir,
                                self.labels[idx]['filename'])
        image = skimage.io.imread(img_path).astype('float32')
        image -= image.min()
        image /= image.max()
        image = image.reshape([*image.shape, 1])
        image = np.concatenate((image, image, image), axis=2)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomFlipHorizontal(object):
    '''Flips image randomly horizontaly. Must be used before ToTensor().'''

    def __init__(self, chance=0.5):
        self.chance = chance

    def __call__(self, sample):
        image = sample['image']
        if random.random() > self.chance:
            image = np.flip(image, 1)
        # torch can't work with negative strides, so we have to copy
        image = image.copy()
        return {'image': image, 'label': sample['label']}


class RandomRotate(object):
    '''Rotates image randomly within specified angle'''

    def __init__(self, max_angle=15):
        self.max_angle = max_angle

    def __call__(self, sample):
        image = sample['image']
        direction = random.choice([-1, 1])
        angle = random.uniform(0, self.max_angle)
        image = skimage.transform.rotate(image, direction * angle, mode='edge')
        image = image.copy()
        return {'image': image, 'label': sample['label']}


class RandomCrop(object):
    def __init__(self, size=512):
        self.size = size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        diff = h - self.size
        margin_w = random.randint(0, diff)
        margin_h = random.randint(0, diff)
        image = image[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        image = image.copy()
        return {'image': image, 'label': sample['label']}


class CenterCrop(object):
    def __init__(self, size=512):
        self.size = size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        diff = h - self.size
        margin_w = int(diff / 2)
        margin_h = int(diff / 2)
        image = image[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        image = image.copy()
        return {'image': image, 'label': sample['label']}


class Resize(object):
    ''' takes target side size as argument '''

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        image = skimage.transform.resize(image, (self.target_size, self.target_size), mode='reflect', anti_aliasing=True)
        image = image.copy()
        return {'image': image, 'label': sample['label']}


class RandomColorShift(object):
    def __call__(self, sample):
        image = sample['image']
        image = image * random.uniform(0.85, 1.15)
        image = image + random.uniform(-0.05, 0.05)
        image = np.clip(image, 0, 1)
        image = image.copy()
        return {'image': image, 'label': sample['label']}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        image[:][:] -= self.mean
        image[:][:] /= self.std
        return {'image': image, 'label': sample['label']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).to(self.dtype),
                'label': torch.from_numpy(label).to(self.dtype)}


class ToNdarray(object):
    """Convert Tensors in sample to ndarrays."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # torch image: C X H X W
        # numpy image: H x W x C
        image = image.numpy().transpose((1, 2, 0))
        return {'image': image,
                'label': label.numpy()}


def show_sample(image, label):
    """Show image with labels"""
    image -= image.min()
    image /= image.max()
    plt.imshow(image)
    plt.title('COVID-19' if label[0] > label[1] else 'Other')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


if __name__ == "__main__":
    print('Test run of lungs_dataset.py')

    mean = 0.5067078578848757
    std = 0.2500083139746181

    df = pd.read_csv('./data/all.csv', usecols=['patientid', 'finding', 'filename'])
    ds = CovidDataset(df, './data/images', transform=torchvision.transforms.Compose([
        RandomRotate(),
        RandomCrop(),
        # CenterCrop(),
        RandomColorShift(),
        Normalize(mean=mean, std=std),
        ToTensor(dtype=torch.float32),
        ToNdarray(),
    ]))
    print('Dataset size:', len(ds))

    calc = False
    if calc:
        mean = 0.
        for sample in ds:
            image, label = sample['image'], sample['label']
            mean += image.mean().astype(np.float64) / len(ds)
        print('Mean:', mean)
        std = 0.
        for sample in ds:
            image, label = sample['image'], sample['label']
            std += image.std().astype(np.float64) / len(ds)
        print('Std:', std)

    n = int(random.random() * len(ds))
    sample = ds[n]
    show_sample(sample['image'], sample['label'])
