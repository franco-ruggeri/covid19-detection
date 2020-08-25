import argparse
from covid19.datasets import covidx

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser(description='Generate COVIDx dataset.')
    parser.add_argument('--in_path', type=str, help='Path to the source datasets', required=True)
    parser.add_argument('--out_path', type=str, help='Path where to store COVIDx dataset', required=True)
    parser.add_argument('--move', help='Whether to move images instead of copying', action='store_true', default=False)
    args = parser.parse_args()

    # generate dataset
    covidx.generate_data(args.in_path, args.out_path, move=args.move)
