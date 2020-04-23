import numpy as np


def partitions(n_images, k_folds):
    n_partitions = np.ones(k_folds) * (n_images // k_folds)
    n_partitions[0:(n_images % k_folds)] += 1
    return n_partitions


def get_indices(n_images, k_folds, indices):
    fold_sizes = partitions(n_images, k_folds)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])


def get_folds(n_images, k_folds):
    indices = np.arange(n_images)
    np.random.shuffle(indices)
    for valid_idx in get_indices(n_images, k_folds, indices):
        train_idx = np.setdiff1d(indices, valid_idx)
        yield train_idx, valid_idx


if __name__ == "__main__":
    print('Test run of cross_validation.py')
    k_folds = 2
    gen = get_folds(10, k_folds)
    for i in range(k_folds):
        print(next(gen))
