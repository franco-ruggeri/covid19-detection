#!/usr/bin/env python3

import argparse
from pathlib import Path
from covid19.datasets import generate_covidx, generate_ham10000


def get_command_line_arguments():
    parser = argparse.ArgumentParser(description='Generate dataset and split it in train, validation and test sets.')
    parser.add_argument('name', type=str, help='name of the dataset. Supported: covidx, ham10000.')
    parser.add_argument('data', type=str, help='path to the source datasets')
    parser.add_argument('output', type=str, help='path where to store the dataset')
    return parser.parse_args()


def main():
    # command-line arguments
    args = get_command_line_arguments()

    # prepare paths
    dataset_path = Path(args.data)
    output_path = Path(args.output)
    if not dataset_path.is_dir():
        raise ValueError('Invalid dataset path')
    if output_path.exists():
        raise FileExistsError(str(output_path) + ' already exists')
    output_path.mkdir(parents=True)

    # generate dataset
    if args.name == 'covidx':
        generate_covidx(dataset_path, output_path, seed=1)
    elif args.name == 'ham10000':
        generate_ham10000(dataset_path, output_path, seed=1)
    else:
        raise ValueError('Invalid dataset name.')


if __name__ == '__main__':
    main()
