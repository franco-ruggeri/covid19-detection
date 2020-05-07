import shutil
import os

"""

This script assumes that in the current working directory there is a folder called data, which has two subfolders:
train and test, in which the training images and test images are stored. This scripts then categorizes the data
into additional categories: normal, pneumonia and COVID-19, in order to be able to use the data with our 
implementation of the COVID-net using tensorflow.

The COVIDx data set was generated following the instructions at:
https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md

"""

train_dataset = {'normal': [], 'pneumonia': [], 'COVID-19': []}
test_dataset = {'normal': [], 'pneumonia': [], 'COVID-19': []}

with open('data/train_split_v3.txt') as f:
    trainfiles = f.readlines()

with open('data/test_split_v3.txt') as fr:
    testfiles = fr.readlines()

special_cases = ['PA', 'AP', 'Supine', 'semi', 'erect']

for item in trainfiles:

    if item.split()[-1] == 'Supine':
        train_dataset[item.split()[-3]].append(item.split()[1])

    elif item.split()[-1] == 'erect':
        train_dataset[item.split()[-4]].append(item.split()[1])

    elif item.split()[-1] == 'PA':
        train_dataset[item.split()[-2]].append(item.split()[1])

    elif item.split()[-1] == 'AP':
        train_dataset[item.split()[-2]].append(item.split()[1])

    else:
        train_dataset[item.split()[-1]].append(item.split()[1])

for item in testfiles:

    if item.split()[-1] == 'Supine':
        test_dataset[item.split()[-3]].append(item.split()[1])

    elif item.split()[-1] == 'erect':
        test_dataset[item.split()[-4]].append(item.split()[1])

    elif item.split()[-1] == 'PA':
        test_dataset[item.split()[-2]].append(item.split()[1])

    elif item.split()[-1] == 'AP':
        test_dataset[item.split()[-2]].append(item.split()[1])

    else:
        test_dataset[item.split()[-1]].append(item.split()[1])

path = 'data'
train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')

image_labels = 'normal', 'pneumonia', 'COVID-19'

for label in image_labels:

    try:
        os.mkdir(os.path.join(train_dir, label))
        print('Category directory: ', label, ' created')

    except FileExistsError:
        print('Category directory: ', label, ' already exists')

    try:
        os.mkdir(os.path.join(test_dir, label))
        print('Category directory: ', label, ' created')

    except FileExistsError:
        print('Category directory: ', label, ' already exists')

current_path = 'data'
for folder in os.listdir(current_path):

    if folder == 'train':
        current_path = train_dir
        current_dataset = train_dataset

    elif folder == 'test':
        current_path = test_dir
        current_dataset = test_dataset

    else:
        continue

    for file in os.listdir(current_path):
        if file in current_dataset['normal']:
            shutil.move(os.path.join(current_path, file), os.path.join(current_path, 'normal', file))

        elif file in current_dataset['pneumonia']:
            shutil.move(os.path.join(current_path, file), os.path.join(current_path, 'pneumonia', file))

        elif file in current_dataset['COVID-19']:
            shutil.move(os.path.join(current_path, file), os.path.join(current_path, 'COVID-19', file))











