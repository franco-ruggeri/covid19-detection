import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_stats(dataset_path):
    train_path = dataset_path / 'train'
    val_path = dataset_path / 'validation'
    test_path = dataset_path / 'test'

    train_class_paths = [path for path in train_path.iterdir()]
    val_class_paths = [path for path in val_path.iterdir()]
    test_class_paths = [path for path in test_path.iterdir()]

    train_class_sizes = [len(list(path.iterdir())) for path in train_class_paths]
    val_class_sizes = [len(list(path.iterdir())) for path in val_class_paths]
    test_class_sizes = [len(list(path.iterdir())) for path in test_class_paths]

    train_total = sum(train_class_sizes)
    val_total = sum(val_class_sizes)
    test_total = sum(test_class_sizes)

    stats = {
        'training': (train_class_sizes, train_total),
        'validation': (val_class_sizes, val_total),
        'test': (test_class_sizes, test_total),
    }
    class_names = [path.name for path in train_class_paths]

    return stats, class_names


def write_stats(stats, class_names, save_path):
    with open(save_path / 'dataset.txt', 'w') as f:
        for dataset, dataset_stats in stats.items():
            f.write('{:<15}\n'.format(dataset))
            print('{:<15}'.format(dataset))
            for class_name, class_size in zip(class_names, dataset_stats[0]):
                f.write('{:<15}{:^15}\n'.format(class_name, class_size))
                print('{:<15}{:^15}'.format(class_name, class_size))
            f.write('{:<15}{:^15}\n\n'.format('total', dataset_stats[1]))
            print('{:<15}{:^15}\n'.format('total', dataset_stats[1]))


def plot_stats(stats, class_names, save_path):
    x = np.arange(len(class_names))
    plt.figure()
    plt.bar(x, stats['training'][0], width=0.25, label='training')
    plt.bar(x + 0.25, stats['validation'][0], width=0.25, tick_label=class_names, label='validation')
    plt.bar(x + 0.5, stats['test'][0], width=0.25, label='test')
    plt.ylabel('number of images')
    plt.legend(loc='upper left')
    plt.savefig(save_path / 'dataset.png')
    plt.show()


def main():
    # command-line arguments
    parser = argparse.ArgumentParser(description='Analysis of COVIDx dataset.')
    parser.add_argument('data', type=str, help='Path to COVIDx dataset')
    parser.add_argument('output', type=str, help='Path where to save the dataset stats')
    args = parser.parse_args()

    # prepare paths
    dataset_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # analyze
    stats, class_names = get_stats(dataset_path)
    write_stats(stats, class_names, output_path)
    plot_stats(stats, class_names, output_path)


if __name__ == '__main__':
    main()
