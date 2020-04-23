import os
import copy

import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import cross_validation as cv
import dataset
import config


def train_one_epoch(model, criterion, loader, optimizer):
    device = next(model.parameters()).device
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs, labels = data.values()
        inputs, labels = inputs.to(dtype=config.dtype, device=device), labels.to(dtype=config.dtype, device=device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # break  # for testing purposes
    return running_loss / (i + 1)


def train_one_epoch_accumulated(model, criterion, loader, optimizer, batch_size_to_update):
    device = next(model.parameters()).device
    running_loss = 0.0
    current_batch_size = 0
    optimizer.zero_grad()
    for i, data in enumerate(loader, 0):
        inputs, labels = data.values()
        inputs, labels = inputs.to(dtype=config.dtype, device=device), labels.to(dtype=config.dtype, device=device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # accumulate gradient and then do Adam step
        loss.backward()
        current_batch_size += loader.batch_size
        if current_batch_size >= batch_size_to_update:
            optimizer.step()
            current_batch_size = 0
            optimizer.zero_grad()
        running_loss += loss.item()
        # break  # for testing purposes
    return running_loss / (i + 1)


def evaluate_performance(model, criterion, loader):
    device = next(model.parameters()).device
    loss = 0.
    labels_all = []
    probs_all = []
    for i, data in enumerate(loader, 0):
        inputs, labels = data.values()
        inputs, labels = inputs.to(dtype=config.dtype, device=device), labels.to(dtype=config.dtype, device=device)
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
        loss += criterion(outputs, labels).item()
        labels_all.append(labels)
        probs_all.append(probs)
    labels_all = torch.cat(labels_all, dim=0).cpu().numpy()
    probs_all = torch.cat(probs_all, dim=0).cpu().numpy()
    # binarize labels
    labels_all = labels_all.argmax(axis=1)
    labels_all = np.stack([1 - labels_all, labels_all], axis=1)
    auc_multiclass = metrics.roc_auc_score(labels_all, probs_all, average=None)
    # binarize probs
    probs_all = probs_all.argmax(axis=1)
    probs_all = np.stack([1 - probs_all, probs_all], axis=1)
    true = (probs_all * labels_all).sum(axis=0)
    accuracy_multiclass = true / labels_all.sum(axis=0)
    return loss / (i + 1), auc_multiclass, accuracy_multiclass


def save_ckpt(model, ckpt_path):
    torch.save(model, ckpt_path)
    print('Checkpoint saved: ' + ckpt_path)


def get_df_for_patients(df_full, patient_ids_indexes):
    patient_ids = df_full['patientid'].unique()
    picked = pd.DataFrame()
    for idx in patient_ids[patient_ids_indexes]:
        picked = pd.concat([picked, df_full[df_full['patientid'] == idx]], ignore_index=True)
    return picked


if __name__ == '__main__':
    images_dir = './data/images'
    checkpoints_dir = './checkpoints'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    data_transforms = {
        'train': torchvision.transforms.Compose([
            dataset.RandomRotate(15),
            dataset.RandomCrop(512),
            dataset.RandomColorShift(),
            dataset.Normalize(config.mean, config.std),
            dataset.ToTensor(dtype=config.dtype)
        ]),
        'eval': torchvision.transforms.Compose([
            dataset.CenterCrop(512),
            dataset.Normalize(config.mean, config.std),
            dataset.ToTensor(dtype=config.dtype),
        ]),
    }

    df = pd.read_csv('./data/all.csv', usecols=['patientid', 'finding', 'filename'])
    covid_df = df[df.apply(lambda x: 'COVID-19' in str(x), axis=1)]
    other_df = df[df.apply(lambda x: 'COVID-19' not in str(x), axis=1)]
    print('Unique COVID-19 images:', len(covid_df))
    print('Unique COVID-19 patients:', len(covid_df['patientid'].unique()))
    print('Unique other images:', len(other_df))
    print('Unique other patients:', len(other_df['patientid'].unique()))

    n_covid_unique_patients = len(covid_df['patientid'].unique())
    n_covid_val_patients = n_covid_unique_patients // config.num_folds
    n_covid_train_patients = n_covid_unique_patients - n_covid_val_patients

    n_other_unique_patients = len(other_df['patientid'].unique())
    n_other_val_patients = n_other_unique_patients // config.num_folds
    n_other_train_patients = n_other_unique_patients - n_other_val_patients

    print('nubmer of COVID-19 validation patients: ', n_covid_val_patients)
    print('nubmer of COVID-19 train patients: ', n_covid_train_patients)
    print('nubmer of Other validation patients: ', n_other_val_patients)
    print('nubmer of Other train patients: ', n_other_train_patients)

    writer = SummaryWriter()

    checkpoints_best_list = list()
    current_fold = 0
    cv_covid_gen = cv.get_folds(n_covid_unique_patients, config.num_folds)
    cv_other_gen = cv.get_folds(n_other_unique_patients, config.num_folds)
    for current_fold in range(config.num_folds):
        patient_indices_covid_train, patient_indices_covid_val = next(cv_covid_gen)
        patient_indices_other_train, patient_indices_other_val = next(cv_other_gen)
        covid_train_df = get_df_for_patients(covid_df, patient_indices_covid_train)
        covid_val_df = get_df_for_patients(covid_df, patient_indices_covid_val)
        other_train_df = get_df_for_patients(other_df, patient_indices_other_train)
        other_val_df = get_df_for_patients(other_df, patient_indices_other_val)

        total = len(covid_train_df) + len(other_train_df)
        weights = torch.tensor([len(other_train_df) / total, len(covid_train_df) / total], dtype=config.dtype, device=config.device)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=weights)

        train_df = pd.concat([covid_train_df, other_train_df], ignore_index=True)
        val_df = pd.concat([covid_val_df, other_val_df], ignore_index=True)

        dataset_train = dataset.CovidDataset(train_df, images_dir, transform=data_transforms['train'])
        dataset_val = dataset.CovidDataset(val_df, images_dir, transform=data_transforms['eval'])
        loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.dataloader_workers, pin_memory=True)
        loader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=config.dataloader_workers, pin_memory=True)

        model = torchvision.models.densenet121(pretrained=True)
        if config.use_pretrained_lungs:
            weights = torch.load(config.pretrained_path, map_location=config.device)
            model.classifier = torch.nn.Linear(1024, 14)
            model.load_state_dict(weights)
        model.classifier = torch.nn.Linear(1024, 2)

        model = model.to(device=config.device)
        model.eval()
        best_model_wts = copy.deepcopy(model.state_dict())
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, amsgrad=True)
        milestones = [100]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

        best_epoch = 0
        no_improvement_epochs_passed = 0
        best_auc = 0.
        best_covid_acc = 0.
        for current_epoch in range(config.num_epochs):
            loss = train_one_epoch_accumulated(model, criterion, loader_train, optimizer, config.batch_size_to_update)
            print('FOLD {}/{} EPOCH {}/{}: '.format(current_fold, config.num_folds, current_epoch, config.num_epochs), end='')
            print('Running loss:', loss)
            if (current_epoch + 1) % 1 == 0:  # and current_epoch >= 100:
                model.eval()
                val_loss, val_auc, val_acc = evaluate_performance(model, criterion, loader_val)
                print('Validation loss = {}, mean auc = {}, mean acc = {}'.format(val_loss, val_auc.mean(), val_acc.mean()))
                writer.add_scalars('fold_' + str(current_fold) + '/losses', {'train': loss, 'val': val_loss}, current_epoch)
                for i in range(len(val_auc)):
                    writer.add_scalars('fold_' + str(current_fold) + '/aucs/roc_auc', {'val-' + dataset.label_names[i]: val_auc[i]}, current_epoch)
                    writer.add_scalars('fold_' + str(current_fold) + '/accs/accs', {'val-' + dataset.label_names[i]: val_acc[i]}, current_epoch)
                writer.add_scalars('fold_' + str(current_fold) + '/aucs/mean', {'val': val_auc.mean()}, current_epoch)
                writer.add_scalars('fold_' + str(current_fold) + '/accs/mean', {'val': val_acc.mean()}, current_epoch)
                if val_auc.mean() > best_auc:
                    best_auc = val_auc.mean()
                    best_covid_acc = val_acc[0]
                    best_epoch = current_epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    no_improvement_epochs_passed = 0
                else:
                    no_improvement_epochs_passed += 1
                if no_improvement_epochs_passed >= config.plateau_epochs_num:
                    print('Training is stopped because auc metric has reached a plateau. Stopped at epoch {}. Best auc metric at epoch {}.'.format(
                        current_epoch, current_epoch - config.plateau_epochs_num))
                    break
                model.train()
                scheduler.step()
            # break  # for testing purposes
        # saving best ckpt
        checkpoint_path = os.path.join(checkpoints_dir, 'ckpt_fold_' + str(current_fold) + '_epoch_' + str(best_epoch) + '_bestonval_auc_' + str(best_auc) + '_acc_covid_' + str(best_covid_acc) + '.pt')
        save_ckpt(best_model_wts, checkpoint_path)
        current_fold += 1
        # break  # for testing purposes
    writer.close()
