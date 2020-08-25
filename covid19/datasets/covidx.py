from pathlib import Path
import pandas as pd

COVID19 = 'covid-19'
PNEUMONIA = 'pneumonia'
NORMAL = 'normal'


def _load_cohen_data(dataset_path):
    image_path = dataset_path / 'images'
    metadata_path = dataset_path / 'metadata.csv'

    # load csv
    csv_content = pd.read_csv(metadata_path)

    # drop images with invalid views (lateral or other non-frontal views)
    valid_views = ["PA", "AP", "AP Supine", "AP erect"]
    valid_idx = csv_content.view.isin(valid_views)
    csv_content = csv_content[valid_idx]

    # fill dataset
    dataset = []
    urls = set()       # set of URLs needed in load_sirm_data() to avoid overlaps
    for _, row in csv_content.iterrows():
        label = row['finding'].split(',')[0].lower()      # take the first finding
        assert row['view'] in valid_views
        if label == COVID19:                              # take just COVID-19 samples from this dataset
            patient_id = row['patientid']
            filepath = image_path / row['filename']
            if not filepath.is_file():
                raise FileNotFoundError('file ' + str(filepath) + ' not found')
            url = row['url']
            dataset.append((patient_id, filepath, label))
            urls.add(url)
    return dataset, urls


def _load_figure1_data(dataset_path):
    image_path = dataset_path / 'images'
    metadata_path = dataset_path / 'metadata.csv'

    # load csv
    csv_content = pd.read_csv(metadata_path, encoding='ISO-8859-1')

    # fill dataset
    dataset = []
    for _, row in csv_content.iterrows():
        label = row['finding']
        if str(label) != 'nan':
            label = label.lower()
        if label == COVID19:
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


def _load_actualmed_data(dataset_path):
    image_path = dataset_path / 'images'
    metadata_path = dataset_path / 'metadata.csv'

    # load csv
    csv_content = pd.read_csv(metadata_path)

    # fill dataset
    dataset = []
    for _, row in csv_content.iterrows():
        label = row['finding']
        if str(label) != 'nan':
            label = label.lower()
        if label == COVID19:
            patient_id = row['patientid']
            filepath = image_path / row['imagename']
            if not filepath.is_file():
                raise FileNotFoundError('file ' + str(filepath) + ' not found')
            dataset.append((patient_id, filepath, label))
    return dataset


def _load_sirm_data(dataset_path, used_urls):
    image_path = dataset_path / 'COVID-19'
    metadata_path = dataset_path / 'COVID-19.metadata.xlsx'

    # bad images to drop
    bad_patient_ids = {'100', '101', '102', '103', '104', '105', '110', '111', '112', '113', '122', '123', '124', '125',
                       '126', '217'}

    # load csv
    csv_content = pd.read_excel(metadata_path)

    # fill dataset
    dataset = []
    for _, row in csv_content.iterrows():
        patient_id = row['FILE NAME'].split('(')[1].split(')')[0]
        url = row['URL']
        if patient_id in bad_patient_ids or url in used_urls:   # skip bad images and overlaps
            continue
        label = COVID19
        suffix = '.' + row['FORMAT'].lower()
        filepath = image_path / row['FILE NAME']
        filepath = filepath.with_suffix(suffix)
        if not filepath.is_file():
            filename = row['FILE NAME'].split('(')[0] + ' (' + row['FILE NAME'].split('(')[1]
            filepath = filepath.with_name(filename).with_suffix(suffix)
            if not filepath.is_file():
                raise FileNotFoundError('file ' + str(filepath) + ' not found')
        dataset.append((patient_id, filepath, label))
    return dataset


def _load_rsna_sample(row, image_path, label, dataset, file_paths):
    patient_id = row['patientId']
    filepath = image_path / patient_id
    filepath = filepath.with_suffix('.dcm')

    if filepath not in file_paths:
        dataset.append((patient_id, filepath, label))
        file_paths.add(filepath)


def _load_rsna_data(dataset_path):
    image_path = dataset_path / 'stage_2_train_images'
    dataset = []
    file_paths = set()

    # get normal samples
    metadata_path = dataset_path / 'stage_2_detailed_class_info.csv'
    csv_content = pd.read_csv(metadata_path)
    for _, row in csv_content.iterrows():
        label = row['class'].lower()
        if label == NORMAL:
            _load_rsna_sample(row, image_path, label, dataset, file_paths)

    # get pneumonia samples
    metadata_path = dataset_path / 'stage_2_train_labels.csv'
    csv_content = pd.read_csv(metadata_path)
    for _, row in csv_content.iterrows():
        label = row['Target']
        if label == 1:
            label = PNEUMONIA
            _load_rsna_sample(row, image_path, label, dataset, file_paths)

    return dataset


def generate_data(dataset_path, output_path, move=False):
    """
    Generate COVIDx dataset using the following sources:
    - https://github.com/ieee8023/covid-chestxray-dataset
    - https://github.com/agchung/Figure1-COVID-chestxray-dataset
    - https://github.com/agchung/Actualmed-COVID-chestxray-dataset
    - https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
    - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    These datasets must be downloaded and put in the directory indicated by dataset_path.

    The generated COVIDx dataset is put in the directory indicated by output_path with the following structure:
    - train
        - covid-19
        - pneumonia
        - normal
    - test
        - covid-19
        - pneumonia
        - normal

    Original work (adapted): https://github.com/lindawangg/COVID-Net/blob/master/create_COVIDx.ipynb

    :param dataset_path: path to the directory containing the datasets described above.
    :param output_path: path where to put COVIDx dataset
    :param move: whether to move the images instead of copying (more efficient)
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise ValueError('Invalid dataset path')

    dataset, urls = _load_cohen_data(dataset_path / 'covid-chestxray-dataset')
    dataset += _load_figure1_data(dataset_path / 'Figure1-COVID-chestxray-dataset')
    dataset += _load_actualmed_data(dataset_path / 'Actualmed-COVID-chestxray-dataset')
    dataset += _load_sirm_data(dataset_path / 'COVID-19 Radiography Database', urls)
    dataset += _load_rsna_data(dataset_path / 'rsna-pneumonia-detection-challenge')

    print('# covid-19:', len(list(filter(lambda sample: sample[2] == COVID19, dataset))))
    print('# pneumonia:', len(list(filter(lambda sample: sample[2] == PNEUMONIA, dataset))))
    print('# normal:', len(list(filter(lambda sample: sample[2] == NORMAL, dataset))))

    # TODO: move/copy images



# train_dataset = {'normal': [], 'pneumonia': [], 'COVID-19': []}
# test_dataset = {'normal': [], 'pneumonia': [], 'COVID-19': []}
#
#
# with open('data/train_COVIDx3.txt') as f:
#     trainfiles = f.readlines()
#
# with open('data/test_COVIDx3.txt') as fr:
#     testfiles = fr.readlines()
#
#
# pathogens = ['normal', 'pneumonia', 'COVID-19']
# for item in trainfiles:
#
#     for index, sub_item in enumerate(item.split()):
#
#         if sub_item in pathogens:
#
#             train_dataset[sub_item].append(item.split()[index-1])
#
# for item in testfiles:
#
#     for index, sub_item in enumerate(item.split()):
#
#         if sub_item in pathogens:
#
#             test_dataset[sub_item].append(item.split()[index-1])
#
# path = 'data'
# train_dir = os.path.join(path, 'train/')
# test_dir = os.path.join(path, 'test/')
#
# image_labels = 'normal', 'pneumonia', 'COVID-19'
#
# for label in image_labels:
#
#     try:
#         os.mkdir(os.path.join(train_dir, label))
#         print('Category directory: ', label, ' created')
#
#     except FileExistsError:
#         print('Category directory: ', label, ' already exists')
#
#     try:
#         os.mkdir(os.path.join(test_dir, label))
#         print('Category directory: ', label, ' created')
#
#     except FileExistsError:
#         print('Category directory: ', label, ' already exists')
#
# current_path = 'data'
# for folder in os.listdir(current_path):
#
#     if folder == 'train':
#         current_path = train_dir
#         current_dataset = train_dataset
#
#     elif folder == 'test':
#         current_path = test_dir
#         current_dataset = test_dataset
#
#     else:
#         continue
#
#     for file in os.listdir(current_path):
#         if file in current_dataset['normal']:
#             shutil.move(os.path.join(current_path, file), os.path.join(current_path, 'normal', file))
#
#         elif file in current_dataset['pneumonia']:
#             shutil.move(os.path.join(current_path, file), os.path.join(current_path, 'pneumonia', file))
#
#         elif file in current_dataset['COVID-19']:
#             shutil.move(os.path.join(current_path, file), os.path.join(current_path, 'COVID-19', file))
