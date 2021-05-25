import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from covid19.ui import Client

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Client(
        n_classes=3,
        models_path=Path(__file__).absolute().parent.parent / 'models',
        class_labels={'covid-19': 0, 'normal': 1, 'pneumonia': 2}
    )
    window.showMaximized()
    sys.exit(app.exec())