import shutil
import datetime
import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
from pathlib import Path
from tqdm import tqdm
from covid19.datasets._utils import stratified_sampling, move_images

LABELS = ['covid-19', 'normal', 'pneumonia']
PREFIX = 'COVIDx'


def _get_filename(count):
    return PREFIX + '_{:05d}'.format(count) + '.png'


def _copy_image_with_gray_scale(source, destination):
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)       # to gray-scale
    cv2.imwrite(str(destination), image)


def _process_dataset_1(dataset, dataset_path, tmp_path):
    image_path = dataset_path / 'images'
    metadata_path = dataset_path / 'metadata.csv'
    valid_views = ["PA", "AP", "AP Supine", "AP erect"]
    csv_content = pd.read_csv(metadata_path)
    count = len(dataset)
    urls = set()

    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing dataset 1')
        for _, row in csv_content.iterrows():
            bar.update()

            # skip invalid views (e.g. lateral)
            if row['view'] not in valid_views:
                continue

            # check label
            label = row['finding'].split(',')[0].lower()      # take the first finding
            if label in LABELS:

                # add URL (dataset 4 checks for overlaps)
                url = row['url']
                urls.add(url)

                # check filepath
                filepath = image_path / row['filename']
                if not filepath.is_file():
                    raise FileNotFoundError('File ' + str(filepath) + ' not found')

                # copy image in gray-scale
                new_filepath = tmp_path / (PREFIX + '_{:05d}'.format(count) + '.png')
                _copy_image_with_gray_scale(filepath, new_filepath)
                count += 1

                # add to dataset
                patient_id = row['patientid']
                dataset.append((patient_id, new_filepath, label))
    return dataset, urls


def _process_dataset_2(dataset, dataset_path, tmp_path):
    image_path = dataset_path / 'images'
    metadata_path = dataset_path / 'metadata.csv'
    csv_content = pd.read_csv(metadata_path, encoding='ISO-8859-1')
    count = len(dataset)

    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing dataset 2')
        for _, row in csv_content.iterrows():
            bar.update()

            # check label
            label = row['finding']
            if str(label) != 'nan':
                label = label.lower()
            if label in LABELS:

                # check filepath
                patient_id = row['patientid']
                filepath = image_path / patient_id
                if filepath.with_suffix('.jpg').is_file():
                    filepath = filepath.with_suffix('.jpg')
                elif filepath.with_suffix('.png').is_file():
                    filepath = filepath.with_suffix('.png')
                else:
                    raise FileNotFoundError('File ' + str(filepath) + ' not found')

                # copy image in gray-scale
                new_filepath = tmp_path / _get_filename(count)
                _copy_image_with_gray_scale(filepath, new_filepath)
                count += 1

                # add to dataset
                dataset.append((patient_id, new_filepath, label))
    return dataset


def _process_dataset_3(dataset, dataset_path, tmp_path):
    image_path = dataset_path / 'images'
    metadata_path = dataset_path / 'metadata.csv'
    csv_content = pd.read_csv(metadata_path)
    count = len(dataset)

    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing dataset 3')
        for _, row in csv_content.iterrows():
            bar.update()

            # check label
            label = row['finding']
            if str(label) != 'nan':
                label = label.lower()
            if label in LABELS:

                # check filepath
                patient_id = row['patientid']
                filepath = image_path / row['imagename']
                if not filepath.is_file():
                    raise FileNotFoundError('File ' + str(filepath) + ' not found')

                # copy image in gray-scale
                new_filepath = tmp_path / _get_filename(count)
                _copy_image_with_gray_scale(filepath, new_filepath)
                count += 1

                # add to dataset
                dataset.append((patient_id, new_filepath, label))
    return dataset


def _process_dataset_4(dataset, dataset_path, tmp_path, urls):
    image_path = dataset_path / 'COVID-19'
    metadata_path = dataset_path / 'COVID-19.metadata.xlsx'
    bad_patient_ids = {'100', '101', '102', '103', '104', '105', '110', '111', '112', '113', '122', '123', '124', '125',
                       '126', '217'}
    csv_content = pd.read_excel(metadata_path)
    count = len(dataset)

    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing dataset 4')
        for _, row in csv_content.iterrows():
            bar.update()
            label = LABELS[0]

            # skip bad images
            patient_id = row['FILE NAME'].split('(')[1].split(')')[0]
            if patient_id in bad_patient_ids:
                continue

            # check URL (overlaps)
            url = row['URL']
            if url in urls:
                continue

            # check filepath
            suffix = '.' + row['FORMAT'].lower()
            filepath = image_path / row['FILE NAME']
            filepath = filepath.with_suffix(suffix)
            if not filepath.is_file():
                filename = row['FILE NAME'].split('(')[0] + ' (' + row['FILE NAME'].split('(')[1]
                filepath = filepath.with_name(filename).with_suffix(suffix)
                if not filepath.is_file():
                    raise FileNotFoundError('File ' + str(filepath) + ' not found')

            # copy image in gray-scale
            new_filepath = tmp_path / _get_filename(count)
            _copy_image_with_gray_scale(filepath, new_filepath)
            count += 1

            # add to dataset
            dataset.append((patient_id, new_filepath, label))
    return dataset


def _process_dataset_5_sample(dataset, file_paths, tmp_path, row, image_path, label):
    patient_id = row['patientId']
    filepath = image_path / patient_id
    filepath = filepath.with_suffix('.dcm')
    count = len(dataset)

    # check filepath
    if filepath not in file_paths:  # repetition
        if not filepath.is_file():
            raise FileNotFoundError('File ' + str(filepath) + ' not found')
        file_paths.add(filepath)

        # copy image from .dcm to .png
        ds = dicom.dcmread(filepath)
        pixel_array_numpy = ds.pixel_array
        new_filepath = tmp_path / _get_filename(count)
        cv2.imwrite(str(new_filepath), pixel_array_numpy)

        # copy image in gray-scale
        _copy_image_with_gray_scale(new_filepath, new_filepath)

        # add to dataset
        dataset.append((patient_id, new_filepath, label))
    return dataset, file_paths


def _process_dataset_5(dataset, dataset_path, tmp_path):
    image_path = dataset_path / 'stage_2_train_images'
    file_paths = set()

    # get normal samples
    metadata_path = dataset_path / 'stage_2_detailed_class_info.csv'
    csv_content = pd.read_csv(metadata_path)
    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing dataset 5 (part 1)')
        for _, row in csv_content.iterrows():
            bar.update()
            label = row['class'].lower()
            if label in LABELS:
                dataset, file_paths = _process_dataset_5_sample(dataset, file_paths, tmp_path, row, image_path, label)

    # get pneumonia samples
    metadata_path = dataset_path / 'stage_2_train_labels.csv'
    csv_content = pd.read_csv(metadata_path)
    with tqdm(total=len(csv_content)) as bar:
        bar.set_description('Processing dataset 5 (part 2)')
        for _, row in csv_content.iterrows():
            bar.update()
            label = row['Target']
            if label == 1:
                label = LABELS[2]
                dataset, file_paths = _process_dataset_5_sample(dataset, file_paths, tmp_path, row, image_path, label)
    return dataset


def generate_covidx(dataset_path, output_path, test_split=.15, validation_split=.15, seed=None):
    """
    Generates COVIDx dataset using the following sources:
    - 1) https://github.com/ieee8023/covid-chestxray-dataset
    - 2) https://github.com/agchung/Figure1-COVID-chestxray-dataset
    - 3) https://github.com/agchung/Actualmed-COVID-chestxray-dataset
    - 4) https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
    - 5) https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
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
    :param test_split: float, fraction of data to be used as test set (must be between 0 and 1)
    :param validation_split: float, fraction of training data to be used as validation set (must be between 0 and 1)
    :param seed: seed for random number generator
    """
    if seed is not None:
        np.random.seed(seed)

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    tmp_path = output_path / ('tmp' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir()

    dataset = []
    dataset, urls = _process_dataset_1(dataset, dataset_path / 'covid-chestxray-dataset', tmp_path)
    dataset = _process_dataset_2(dataset, dataset_path / 'Figure1-COVID-chestxray-dataset', tmp_path)
    dataset = _process_dataset_3(dataset, dataset_path / 'Actualmed-COVID-chestxray-dataset', tmp_path)
    dataset = _process_dataset_4(dataset, dataset_path / 'COVID-19 Radiography Database', tmp_path, urls)
    dataset = _process_dataset_5(dataset, dataset_path / 'rsna-pneumonia-detection-challenge', tmp_path)

    train_set, test_set = stratified_sampling(dataset, test_split)
    train_set, val_set = stratified_sampling(train_set, validation_split)

    move_images(train_set, output_path / 'train')
    move_images(val_set, output_path / 'validation')
    move_images(test_set, output_path / 'test')
    shutil.rmtree(tmp_path)
