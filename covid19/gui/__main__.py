import sys
from PySide6.QtWidgets import QApplication, QFileDialog
from covid19.gui import Client


def main():
    app = QApplication(sys.argv)
    app.setOrganizationName('kth')
    app.setOrganizationDomain('kth.se')
    app.setApplicationName('covid19-detector')

    window = Client(
        n_classes=3,
        class_labels={'covid-19': 0, 'normal': 1, 'pneumonia': 2}
    )
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
