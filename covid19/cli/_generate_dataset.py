from pathlib import Path
from covid19.datasets import generate_covidx, generate_ham10000


def add_arguments_generate_dataset(parser):
    parser.add_argument('name', type=str, choices=['covidx', 'ham10000'], help='name of the dataset')
    parser.add_argument('data', type=str, help='path to the source datasets')
    parser.add_argument('output', type=str, help='path where to store the dataset')


def generate_dataset(args):
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
