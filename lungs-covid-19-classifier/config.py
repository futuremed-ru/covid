import torch

use_pretrained_lungs = True
pretrained_path = './data/ckpt_epoch_13_val_loss_3.896396775472732_mean_val_auc_0.9039623122650647.pt'

mean = 0.5067078578848757
std = 0.2500083139746181

dataloader_workers = 4
batch_size = 8
batch_size_to_update = 32

num_folds = 10
num_epochs = 50
learning_rate = 3e-4
plateau_epochs_num = 1000

dtype = torch.float32
device = torch.device('cuda:1')
