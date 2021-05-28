import sys
import argparse
from PySide6.QtWidgets import QApplication
from covid19.gui import Client


def _get_arguments():
    parser = argparse.ArgumentParser(description='COVID-19 detector.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('models', type=str, help='path to the models')
    return parser.parse_args()


def main():
    args = _get_arguments()
    app = QApplication(sys.argv)
    window = Client(
        n_classes=3,
        models_path=args.models,
        class_labels={'covid-19': 0, 'normal': 1, 'pneumonia': 2}
    )
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
