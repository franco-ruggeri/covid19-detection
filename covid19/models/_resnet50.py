from covid19.models._model import Model
from tensorflow.keras import Input
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


class ResNet50(Model):
    """
    COVID-19 classifier with ResNet50v2 as convolutional base for feature extraction.

    Inputs: batches of images with shape (None, 224, 224, 3).
    Outputs: batches of softmax activations (None, 3). The 3 classes are intended to be: covid-19, normal, pneumonia.
    """

    def __init__(self, name='resnet50'):
        super().__init__(name=name)
        self._image_shape = (224, 224, 3)

        self._preprocess = preprocess_input
        self._feature_extractor = ResNet50V2(include_top=False, input_shape=self.image_shape)
        self._classifier = [
            GlobalAveragePooling2D(),
            Dense(3, activation='softmax')
        ]

        # required for summary()
        inputs = Input(shape=self.image_shape)
        outputs = self.call(inputs)
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.build(input_shape=(None, self.image_shape[0], self.image_shape[1], self.image_shape[2]))

    def call(self, inputs, training=None, mask=None):
        x = self.preprocess(inputs)
        x = self.feature_extractor(x, training=False)    # training=False to keep BN layers in inference mode
        for layer in self.classifier:
            x = layer(x)
        return x

    @property
    def preprocess(self):
        return self._preprocess

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def classifier(self):
        return self._classifier

    @property
    def image_shape(self):
        return self._image_shape
