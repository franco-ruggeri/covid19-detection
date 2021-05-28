#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def get_command_line_arguments():
    parser = argparse.ArgumentParser(description='Plot TensorBoard logs.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('logs', type=str, help='path to directory containing logs')
    parser.add_argument('metric', type=str, help='metric to plot (e.g. accuracy)')
    return parser.parse_args()


def main():
    args = get_command_line_arguments()

    root_path = Path(args.logs)
    lines = [
        (root_path / 'run-train-tag-epoch_accuracy.csv', 'training'),
        (root_path / 'run-validation-tag-epoch_accuracy.csv', 'validation'),
    ]

    plt.figure()
    for line in lines:
        csv_content = pd.read_csv(line[0], engine='python', sep=',')
        x = csv_content['Step']
        y = csv_content['Value']
        plt.plot(x, y, label=line[1])
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(args.metric)
    plt.show()


if __name__ == '__main__':
    main()
