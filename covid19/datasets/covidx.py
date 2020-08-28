import shutil
import sys
import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
from pathlib import Path
from tqdm import tqdm

LABELS = ['covid-19', 'pneumonia', 'normal']


def _process_cohen_data(dataset_path):
    image_path = dataset_path / 'images'
    metadata_path = dataset_path / 'metadata.csv'
    csv_content = pd.read_csv(metadata_path)
    dataset = []
    urls = set()  # set of URLs needed in load_sirm_data() to avoid overlaps

    # drop images with invalid views (lateral or other non-frontal views)
    valid_views = ["PA", "AP", "AP Supine", "AP erect"]
    valid_idx = csv_content.view.isin(valid_views)
    csv_content = csv_content[valid_idx]

    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing cohen dataset')
        for _, row in csv_content.iterrows():
            bar.update()
            label = row['finding'].split(',')[0].lower()      # take the first finding
            assert row['view'] in valid_views
            if label in LABELS:
                patient_id = row['patientid']
                url = row['url']
                filepath = image_path / row['filename']
                if not filepath.is_file():
                    raise FileNotFoundError('file ' + str(filepath) + ' not found')
                dataset.append((patient_id, filepath, label))
                urls.add(url)
    return dataset, urls


def _process_figure1_data(dataset_path):
    image_path = dataset_path / 'images'
    metadata_path = dataset_path / 'metadata.csv'
    csv_content = pd.read_csv(metadata_path, encoding='ISO-8859-1')
    dataset = []

    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing figure1 dataset')
        for _, row in csv_content.iterrows():
            bar.update()
            label = row['finding']
            if str(label) != 'nan':
                label = label.lower()
            if label in LABELS:
                patient_id = row['patientid']
                filepath = image_path / patient_id
                if filepath.with_suffix('.jpg').is_file():
                    filepath = filepath.with_suffix('.jpg')
                elif filepath.with_suffix('.png').is_file():
                    filepath = filepath.with_suffix('.png')
                else:
                    raise FileNotFoundError('file ' + str(filepath) + ' not found')
                dataset.append((patient_id, filepath, label))
    return dataset


def _process_actualmed_data(dataset_path):
    image_path = dataset_path / 'images'
    metadata_path = dataset_path / 'metadata.csv'
    csv_content = pd.read_csv(metadata_path)
    dataset = []

    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing actualmed dataset')
        for _, row in csv_content.iterrows():
            bar.update()
            label = row['finding']
            if str(label) != 'nan':
                label = label.lower()
            if label in LABELS:
                patient_id = row['patientid']
                filepath = image_path / row['imagename']
                if not filepath.is_file():
                    raise FileNotFoundError('file ' + str(filepath) + ' not found')
                dataset.append((patient_id, filepath, label))
    return dataset


def _process_sirm_data(dataset_path, tmp_path, used_urls):
    image_path = dataset_path / 'COVID-19'
    metadata_path = dataset_path / 'COVID-19.metadata.xlsx'
    csv_content = pd.read_excel(metadata_path)
    bad_patient_ids = {'100', '101', '102', '103', '104', '105', '110', '111', '112', '113', '122', '123', '124', '125',
                       '126', '217'}
    dataset = []

    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing sirm dataset')
        for _, row in csv_content.iterrows():
            bar.update()

            # skip bad images and overlaps
            patient_id = row['FILE NAME'].split('(')[1].split(')')[0]
            url = row['URL']
            if patient_id in bad_patient_ids or url in used_urls:
                continue

            # get fields
            label = LABELS[0]
            suffix = '.' + row['FORMAT'].lower()
            filepath = image_path / row['FILE NAME']
            filepath = filepath.with_suffix(suffix)
            if not filepath.is_file():
                filename = row['FILE NAME'].split('(')[0] + ' (' + row['FILE NAME'].split('(')[1]
                filepath = filepath.with_name(filename).with_suffix(suffix)
                if not filepath.is_file():
                    raise FileNotFoundError('file ' + str(filepath) + ' not found')

            # convert color to gray scale
            image = cv2.imread(str(filepath))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            filepath = tmp_path / filepath.name
            cv2.imwrite(str(filepath), image)

            # add sample
            dataset.append((patient_id, filepath, label))

    return dataset


def _process_rsna_sample(row, image_path, tmp_path, label, dataset, file_paths):
    patient_id = row['patientId']
    filepath = image_path / patient_id
    filepath = filepath.with_suffix('.dcm')

    if filepath not in file_paths:
        if not filepath.is_file():
            raise FileNotFoundError('file ' + str(filepath) + ' not found')
        file_paths.add(filepath)

        # convert image from .dcm to .png
        ds = dicom.dcmread(filepath)
        pixel_array_numpy = ds.pixel_array
        filepath = tmp_path / (filepath.stem + '.png')
        cv2.imwrite(str(filepath), pixel_array_numpy)

        # add sample
        dataset.append((patient_id, filepath, label))


def _process_rsna_data(dataset_path, tmp_path):
    image_path = dataset_path / 'stage_2_train_images'
    dataset = []
    file_paths = set()

    # get normal samples
    metadata_path = dataset_path / 'stage_2_detailed_class_info.csv'
    csv_content = pd.read_csv(metadata_path)
    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing rsna dataset (normal)')
        for _, row in csv_content.iterrows():
            bar.update()
            label = row['class'].lower()
            if label in LABELS:
                _process_rsna_sample(row, image_path, tmp_path, label, dataset, file_paths)

    # get pneumonia samples
    metadata_path = dataset_path / 'stage_2_train_labels.csv'
    csv_content = pd.read_csv(metadata_path)
    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing rsna dataset (pneumonia)')
        for _, row in csv_content.iterrows():
            bar.update()
            label = row['Target']
            if label == 1:
                label = LABELS[1]
                _process_rsna_sample(row, image_path, tmp_path, label, dataset, file_paths)

    return dataset


def _split_class(dataset, label, split):
    # select samples and patients for the class
    train_set = [sample for sample in dataset if sample[2] == label]
    test_set = []
    patients = list({sample[0] for sample in train_set})

    # sample patients until the number of images is enough
    # note: patients are sampled, not directly samples, since we all the samples of a patient in one set
    n_desired = split * len(train_set)
    n = 0
    while n < n_desired:
        patient = patients.pop(np.random.choice(len(patients)))
        patient_samples = {sample for sample in train_set if sample[0] == patient}
        train_set = [sample for sample in train_set if sample not in patient_samples]   # remove used samples
        test_set += patient_samples
        n += len(patient_samples)
    return train_set, test_set


def _stratified_sampling(dataset, split):
    train_set = []
    test_set = []
    for label in LABELS:
        train_set_, test_set_ = _split_class(dataset, label, split)
        train_set += train_set_
        test_set += test_set_
    return train_set, test_set


def _copy_move_images(dataset, output_path, move):
    # create paths
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir()
    for label in LABELS:
        label_path = output_path / label
        label_path.mkdir()

    # move/copy images
    with tqdm(total=len(dataset)) as bar:
        bar.set_description('Building ' + str(output_path))
        for sample in dataset:
            bar.update()
            label = sample[2]
            filepath = sample[1]
            new_filepath = output_path / label / filepath.name
            if move:
                shutil.move(filepath, new_filepath)
            else:
                shutil.copy(sample[1], new_filepath)


def generate_data(dataset_path, output_path, test_split=.15, validation_split=.15, move=False):
    """
    Generates COVIDx dataset using the following sources:
    - https://github.com/ieee8023/covid-chestxray-dataset
    - https://github.com/agchung/Figure1-COVID-chestxray-dataset
    - https://github.com/agchung/Actualmed-COVID-chestxray-dataset
    - https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
    - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    These datasets must be downloaded and put in the directory indicated by dataset_path.

    The generated COVIDx dataset is put in the directory indicated by output_path with the following structure:
    - train
        - covid-19
        - normal
        - pneumonia
    - validation
        - covid-19
        - normal
        - pneumonia
    - test
        - covid-19
        - normal
        - pneumonia

    Adapted from: https://github.com/lindawangg/COVID-Net/blob/master/create_COVIDx.ipynb

    :param dataset_path: path to the directory containing the datasets described above
    :param output_path: path where to put COVIDx dataset
    :param test_split: fraction of data to be used as test set
    :param validation_split: fraction of training data to be used as validation set
    :param move: whether to move the images instead of copying (more efficient)
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise ValueError('Invalid dataset path')

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path / ('tmp' + str(np.random.randint(sys.maxsize)))
    tmp_path_sirm = tmp_path / 'sirm'
    tmp_path_rsna = tmp_path / 'rsna'
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir()
    tmp_path_sirm.mkdir()
    tmp_path_rsna.mkdir()

    dataset, urls = _process_cohen_data(dataset_path / 'covid-chestxray-dataset')
    dataset += _process_figure1_data(dataset_path / 'Figure1-COVID-chestxray-dataset')
    dataset += _process_actualmed_data(dataset_path / 'Actualmed-COVID-chestxray-dataset')
    dataset += _process_sirm_data(dataset_path / 'COVID-19 Radiography Database', tmp_path_sirm, urls)
    dataset += _process_rsna_data(dataset_path / 'rsna-pneumonia-detection-challenge', tmp_path_rsna)

    train_set, test_set = _stratified_sampling(dataset, test_split)
    train_set, val_set = _stratified_sampling(train_set, validation_split)

    _copy_move_images(train_set, output_path / 'train', move)
    _copy_move_images(val_set, output_path / 'validation', move)
    _copy_move_images(test_set, output_path / 'test', move)
    shutil.rmtree(tmp_path)

    print()
    print('Stats:')
    print('# images:', len(dataset))
    print('# patients:', len({sample[0] for sample in dataset}))
    for label in LABELS:
        print('# {}: {}'.format(label, len([sample for sample in dataset if sample[2] == label])))
        print('# train {}: {}'.format(label, len([sample for sample in train_set if sample[2] == label])))
        print('# val {}: {}'.format(label, len([sample for sample in val_set if sample[2] == label])))
        print('# test {}: {}'.format(label, len([sample for sample in test_set if sample[2] == label])))
