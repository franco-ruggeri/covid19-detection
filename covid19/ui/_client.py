import time
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from PySide6.QtCore import Slot, QStandardPaths
from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PySide6.QtGui import QPixmap
from covid19.ui._ui_client import Ui_Client
from covid19.models import ResNet50, COVIDNet
from covid19.explainers import GradCAM, IG


def _get_size(ui):
    size = ui.maximumSize()
    return size.height(), size.width()


def numpy_to_pixmap(image, size):
    tmp_file = '/tmp/covid_{}.png'.format(time.time())
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
        self._model_loaded = False
        self._explainer = None
        self._input = None
        self._input_predicted = False

        self.ui = Ui_Client()
        self.ui.setupUi(self)
        self.ui.model.currentTextChanged.connect(self._refresh_explainer)
        self.ui.model.currentTextChanged.connect(self._refresh_predict)
        self.ui.explainer.currentTextChanged.connect(self._refresh_predict)
        self.ui.predict.clicked.connect(self._refresh_predict)
        self.ui.select_image.clicked.connect(self._refresh_predict)

        self._input_size = _get_size(self.ui.input)
        self._explanation_size = _get_size(self.ui.explanation)
        self._refresh_model()
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
        self._input_predicted = True

    @Slot()
    def on_select_image_clicked(self):
        print('Selecting image... ', end='', flush=True)
        filename, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select chest X-ray image',
            dir=QStandardPaths.standardLocations(QStandardPaths.PicturesLocation).pop(),
            filter='Images (*.png *.jpg *.jpeg)'
        )
        if filename != '':
            self._input = cv2.imread(filename)
            self.ui.input.setPixmap(numpy_to_pixmap(self._input, self._input_size))
            self.ui.prediction.clear()
            self.ui.confidence.clear()
            self.ui.explanation.setPixmap(':/images/default.png')
            self._input_predicted = False
            self.ui.input_state.setPixmap(':/images/ok.png')
            print(filename)
        else:
            print('canceled')

    @Slot(str)
    def on_model_currentTextChanged(self, model):
        print('Loading {}... '.format(model), end='', flush=True)
        if model == 'ResNet50':
            self._model = ResNet50(self.n_classes, weights=None)
            model_path = self.models_path / 'resnet50'
        elif model == 'COVID-Net':
            self._model = COVIDNet(self.n_classes, weights=None)
            model_path = self.models_path / 'covidnet'
        else:
            raise ValueError('Invalid model {}'.format(model))
        try:
            self._model.load_weights(model_path)
            self._model_loaded = True
            self._input_predicted = False
            self.ui.model_state.setPixmap(':/images/ok.png')
            print('done')
        except tf.errors.NotFoundError:
            self._model_loaded = False
            self.ui.model_state.setPixmap(':/images/error.png')
            QMessageBox.critical(
                self,
                'Model not found',
                'Failed to load model, check that the models directory contains the model.'
            )
            print('failed')

    @Slot(str)
    def on_explainer_currentTextChanged(self, explainer):
        print('Preparing {}... '.format(explainer), end='', flush=True)
        if explainer == 'Grad-CAM':
            self._explainer = GradCAM(self._model)
        elif explainer == 'Integrated Gradients':
            self._explainer = IG(self._model)
        else:
            raise ValueError('Invalid explainer {}'.format(explainer))
        self.ui.explainer_state.setPixmap(':/images/ok.png')
        self._input_predicted = False
        print('done')

    @Slot()
    def _refresh_model(self):
        self.ui.model.currentTextChanged.emit(self.ui.model.currentText())

    @Slot()
    def _refresh_explainer(self):
        self.ui.explainer.currentTextChanged.emit(self.ui.explainer.currentText())

    @Slot()
    def _refresh_predict(self):
        enabled = self._model_loaded and not self._input_predicted
        self.ui.predict.setEnabled(enabled)
