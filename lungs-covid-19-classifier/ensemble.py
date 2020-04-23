import os

import pandas as pd
import torch
import torchvision

import dataset
import config

dtype = torch.float32


def floatlist_to_str(floatlist, format_string):
    assert type(format_string) == str
    string = ''
    for number in floatlist:
        string += format_string.format(number)
    return string


def ensemble_output(ensemble_dir, dataset, device):
    ensemble_out = torch.tensor((), dtype=torch.float32).new_zeros(len(dataset), 2).to(device=device)
    models_count = 0
    for path in os.listdir(ensemble_dir):
        if path.endswith('.pt'):
            print('Working with ' + path)
            model_wts = torch.load(os.path.join(ensemble_dir, path))
            # creating model
            model = torchvision.models.densenet121(pretrained=False)
            model.classifier = torch.nn.Linear(1024, 2)
            model.load_state_dict(model_wts)
            model = model.to(device=device)
            model.eval()
            i = 0
            for i in range(len(dataset)):
                sample = dataset[i]
                image = sample['image']
                tensor = image.reshape(1, *image.shape).to(device=device)
                with torch.no_grad():
                    output = model(tensor.to(device=device)).view(2)
                    probabilities = torch.sigmoid(output)
                ensemble_out[i] += probabilities
            models_count += 1
    ensemble_out /= models_count
    return ensemble_out


if __name__ == "__main__":
    data_dir = 'data'  # '/media/tower/Seagate Expansion Drive'
    annotation_csv = os.path.join(data_dir, 'new_github.csv')  # td-lungs-stats-no-duplicates.csv  # Train_Data_Entry.csv
    images_dir = os.path.join(data_dir, 'new_github_images')  # _img  # images/train
    ensemble_dir = 'checkpoints'
    device = torch.device("cuda:1")

    df = pd.read_csv(annotation_csv, usecols=['filename', 'finding'])
    # df.columns = ['finding', 'filename']  # for ChestXRay-14 lungs
    dataset = dataset.CovidDataset(df, images_dir, transform=torchvision.transforms.Compose([
        dataset.Resize(564),
        dataset.CenterCrop(512),
        dataset.Normalize(config.mean, config.std),
        dataset.ToTensor(dtype=config.dtype)
    ]))
    n_images = len(dataset)
    print('nubmer of samples: ', n_images)

    ensemble_out = ensemble_output(ensemble_dir, dataset, device).cpu().numpy()

    labels = dict()
    labels['filename'] = list()
    labels['ensemble out'] = list()
    i = 0
    for index, row in df.iterrows():
        labels['filename'].append(row['filename'])
        labels['ensemble out'].append(floatlist_to_str(ensemble_out[i], '{:e} '))
        i += 1
    df = pd.DataFrame.from_dict(labels)
    dst_csv_path = './data/covid_New_Github_Outputs.csv'
    df.to_csv(dst_csv_path)
    print('Ensemble outputs are saved to: ', dst_csv_path)
    print('Done')
