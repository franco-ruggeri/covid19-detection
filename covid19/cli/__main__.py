import argparse
from covid19.cli import train, test, generate_dataset, examine_dataset
from covid19.cli import add_arguments_train, add_arguments_test, add_arguments_generate_dataset
from covid19.cli import add_arguments_examine_dataset


def add_arguments_dataset(parser):
    subparsers = parser.add_subparsers()
    parser_generate = subparsers.add_parser('generate')
    parser_examine = subparsers.add_parser('examine')

    add_arguments_generate_dataset(parser_generate)
    add_arguments_examine_dataset(parser_examine)


def get_arguments():
    parser = argparse.ArgumentParser(description='COVID-19 detection tool suite.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser('train')
    parser_test = subparsers.add_parser('test')
    parser_dataset = subparsers.add_parser('dataset')

    add_arguments_train(parser_train)
    add_arguments_test(parser_test)
    add_arguments_dataset(parser_dataset)

    return parser.parse_args()


def main():
    args = get_arguments()
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'dataset':
        if args.dataset_command == 'generate':
            generate_dataset(args)
        elif args.dataset_command == 'examine':
            examine_dataset(args)
        else:
            raise ValueError
    else:
        raise ValueError


if __name__ == '__main__':
    main()
