import argparse
from covid19.datasets import covidx


def main():
    # command-line arguments
    parser = argparse.ArgumentParser(description='Generate COVIDx dataset.')
    parser.add_argument('data', type=str, help='Path to the source datasets')
    parser.add_argument('output', type=str, help='Path where to store COVIDx dataset')
    args = parser.parse_args()

    # generate dataset
    covidx.generate_data(args.data, args.output, seed=1)


if __name__ == '__main__':
    main()
