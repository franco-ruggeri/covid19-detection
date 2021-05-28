import argparse
from covid19.cli import train, test, generate_dataset, examine_dataset
from covid19.cli._utils import discard_argument


def get_command():
    parser = argparse.ArgumentParser(description='Tools for COVID-19 detection.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('command', type=str, choices=['train', 'test', 'dataset'], help='command to execute.')
    return parser.parse_args()


def get_dataset_subcommand():
    parser = argparse.ArgumentParser(description='Tools for datasets.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('command', type=str, choices=['generate', 'examine'], help='command to execute.')
    return parser.parse_args()


def main():
    args = get_command()
    discard_argument()
    if args.command == 'train':
        train()
    elif args.command == 'test':
        test()
    elif args.command == 'dataset':
        args = get_dataset_subcommand()
        discard_argument()
        if args.command == 'generate':
            generate_dataset()
        elif args.command == 'examine':
            examine_dataset()
        else:
            raise ValueError
    else:
        raise ValueError


if __name__ == '__main__':
    main()
