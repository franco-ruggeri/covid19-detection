import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path


def _split_class(dataset, label, split):
    # select samples and IDs for the class
    train_set = [sample for sample in dataset if sample[2] == label]
    test_set = []
    ids = list({sample[0] for sample in train_set})

    # sample IDs until the number of images is enough
    # note: IDs are sampled, not directly samples, since we want all the samples with an ID in the same set
    n_desired = split * len(train_set)
    n = 0
    while n < n_desired:
        id_ = ids.pop(np.random.choice(len(ids)))
        id_samples = {sample for sample in train_set if sample[0] == id_}
        train_set = [sample for sample in train_set if sample not in id_samples]   # remove used samples
        test_set += id_samples
        n += len(id_samples)
    return train_set, test_set


def stratified_sampling(dataset, labels, split):
    train_set = []
    test_set = []
    for label in labels:
        train_set_, test_set_ = _split_class(dataset, label, split)
        train_set += train_set_
        test_set += test_set_
    return train_set, test_set


def _copy_or_move_images(dataset, labels, output_path, move=False):
    # create paths
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    for label in labels:
        label_path = output_path / label
        label_path.mkdir()

    # copy or move images
    with tqdm(total=len(dataset)) as bar:
        bar.set_description('Filling ' + str(output_path))
        for sample in dataset:
            bar.update()
            label = sample[2]
            filepath = sample[1]
            new_filepath = output_path / label / filepath.name
            if move:
                shutil.move(filepath, new_filepath)
            else:
                shutil.copy(filepath, new_filepath)


def copy_images(dataset, labels, output_path):
    _copy_or_move_images(dataset, labels, output_path)


def move_images(dataset, labels, output_path):
    _copy_or_move_images(dataset, labels, output_path, move=True)
