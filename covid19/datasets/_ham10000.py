import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from covid19.datasets._utils import stratified_sampling, copy_images


def _process_dataset(dataset_path):
    dataset = []
    image_paths = [dataset_path / 'HAM10000_images_part_1', dataset_path / 'HAM10000_images_part_2']
    metadata_path = dataset_path / 'HAM10000_metadata.csv'
    csv_content = pd.read_csv(metadata_path)

    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing dataset')
        for _, row in csv_content.iterrows():
            bar.update()

            # get fields
            lesion_id = row['lesion_id']
            image_id = row['image_id']
            label = row['dx']

            # find filepath
            filepath_found = False
            for image_path in image_paths:
                filepath = image_path / image_id
                filepath = filepath.with_suffix('.jpg')
                if filepath.is_file():
                    filepath_found = True
                    break
            if not filepath_found:
                raise FileNotFoundError('Image ' + image_id + ' not found')

            # add to dataset
            dataset.append((lesion_id, filepath, label))
    return dataset


def generate_ham10000(dataset_path, output_path, test_split=.15, validation_split=.15, seed=None):
    """
    Generates HAM10000 dataset with the following structure:
    - train
        - class_1
        - class_2
        - ...
    - validation
        - class_1
        - class_2
        - ...
    - test
        - class_1
        - class_2
        - ...

    The source data must be downloaded from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000 and put in the
    directory indicated by dataset_path.

    :param dataset_path: path to the directory containing the source data
    :param output_path: path where to put HAM10000 dataset
    :param test_split: float, fraction of data to be used as test set (must be between 0 and 1)
    :param validation_split: float, fraction of training data to be used as validation set (must be between 0 and 1)
    :param seed: seed for random number generator
    """
    if seed is not None:
        np.random.seed(seed)

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    dataset = _process_dataset(dataset_path)
    train_set, test_set = stratified_sampling(dataset, test_split)
    train_set, val_set = stratified_sampling(train_set, validation_split)

    copy_images(train_set, output_path / 'train')
    copy_images(val_set, output_path / 'validation')
    copy_images(test_set, output_path / 'test')
