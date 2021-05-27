#!/usr/bin/env python3

import argparse
from pathlib import Path
from utils import get_model, IMAGE_SIZE
from covid19.datasets import image_dataset_from_directory
from tensorflow.keras.layers import Dense


def get_command_line_arguments():
    parser = argparse.ArgumentParser(description='Convert a model to a pretrained version for another dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_pretraining', type=str, help='path to the dataset used for pretraining')
    parser.add_argument('data_new', type=str, help='path to the new dataset')
    parser.add_argument('output', type=str, help='path where to save the pretrained model')
    parser.add_argument('model', type=str, help='path to the model to convert.')
    parser.add_argument('architecture', type=str, help='architecture of the model. Supported: resnet50, covidnet.')
    return parser.parse_args()


def main():
    # command-line arguments
    args = get_command_line_arguments()

    # prepare paths
    dataset_pretraining_path = Path(args.data_pretraining) / 'train'
    dataset_new_path = Path(args.data_new) / 'train'
    output_path = Path(args.output) / 'model_pretrained'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # load model
    _, train_ds_info = image_dataset_from_directory(dataset_pretraining_path, IMAGE_SIZE)
    model = get_model(args.architecture, None, train_ds_info, args.model)

    # replace last layer with the right number of units
    _, train_ds_info = image_dataset_from_directory(dataset_new_path, IMAGE_SIZE)
    model.classifier.pop()
    model.classifier.add(Dense(train_ds_info['n_classes']))

    # save pretrained model
    model.save_weights(output_path)


if __name__ == '__main__':
    main()
