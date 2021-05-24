import cv2
import numpy as np
import time
from pathlib import Path
from PySide6.QtCore import Slot, QStandardPaths
from PySide6.QtWidgets import QMainWindow, QFileDialog
from PySide6.QtGui import QPixmap
from covid19.ui._ui_client import Ui_Client
from covid19.models import ResNet50, COVIDNet
from covid19.explainers import GradCAM, IG


def _get_size(ui):
    size = ui.maximumSize()
    return size.height(), size.width()


def numpy_to_pixmap(image, size):
    tmp_file = '.tmp_{}.png'.format(time.time())
    image = cv2.resize(image, size)
    cv2.imwrite(tmp_file, image)
    image = QPixmap(tmp_file)
    tmp_file = Path(tmp_file)
    tmp_file.unlink()
    return image


class Client(QMainWindow):
    def __init__(self, n_classes, models_path, class_labels):
        super().__init__()
        self.n_classes = n_classes
        self.models_path = Path(models_path)
        self.class_labels = {l: c for (c, l) in class_labels.items()}
        self._model = None
        self._explainer = None
        self._input = None
        self.ui = Ui_Client()
        self.ui.setupUi(self)
        self.ui.architecture.currentTextChanged.connect(self._refresh_explainer)
        self._input_size = _get_size(self.ui.input)
        self._explanation_size = _get_size(self.ui.explanation)
        self._refresh_architecture()
        self.ui.predict.setEnabled(False)

    @Slot()
    def on_predict_clicked(self):
        # forward
        print('Predicting... ', end='', flush=True)
        input_ = cv2.resize(self._input, self._model.image_shape[0:2])
        prediction, confidence, explanation = self._explainer.explain(input_)
        explanation = (explanation * 255).astype(np.uint8)  # [0,1] -> [0,255]
        print('done')

        # update GUI
        self.ui.prediction.setText(self.class_labels[prediction])
        self.ui.confidence.setText('{:.2f}%'.format(confidence * 100))
        self.ui.explanation.setPixmap(numpy_to_pixmap(explanation, self._explanation_size))
        self.ui.predict.setEnabled(False)

    @Slot()
    def on_select_image_clicked(self):
        print('Selecting image... ', end='', flush=True)
        filename, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select chest X-ray image',
            dir=QStandardPaths.standardLocations(QStandardPaths.PicturesLocation).pop(),
            filter='Images (*.png *.jpg *.jpeg)'
        )
        self._input = cv2.imread(filename)
        self.ui.input.setPixmap(numpy_to_pixmap(self._input, self._input_size))
        self.ui.predict.setEnabled(True)
        self.ui.prediction.clear()
        self.ui.confidence.clear()
        self.ui.explanation.setPixmap(':/images/default.png')
        print(filename)

    @Slot(str)
    def on_architecture_currentTextChanged(self, architecture):
        print('Loading {}... '.format(architecture), end='', flush=True)
        if architecture == 'ResNet50':
            self._model = ResNet50(self.n_classes, weights=None)
            model_path = self.models_path / 'resnet50'
        elif architecture == 'COVID-Net':
            self._model = COVIDNet(self.n_classes, weights=None)
            model_path = self.models_path / 'covidnet'
        else:
            raise ValueError('Invalid architecture')
        self._model.load_weights(model_path)
        self.ui.predict.setEnabled(True)
        print('done')

    @Slot(str)
    def on_explainer_currentTextChanged(self, explainer):
        print('Preparing {}... '.format(explainer), end='', flush=True)
        if explainer == 'Grad-CAM':
            self._explainer = GradCAM(self._model)
        elif explainer == 'Integrated Gradients':
            self._explainer = IG(self._model)
        else:
            raise ValueError('Invalid architecture')
        self.ui.predict.setEnabled(True)
        print('done')

    @Slot()
    def _refresh_architecture(self):
        self.ui.architecture.currentTextChanged.emit(self.ui.architecture.currentText())

    @Slot()
    def _refresh_explainer(self):
        self.ui.explainer.currentTextChanged.emit(self.ui.explainer.currentText())
