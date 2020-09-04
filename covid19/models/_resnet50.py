import tensorflow as tf
from covid19.models._model import Model
from covid19.layers import Rescaling
from tensorflow.keras import Input, Sequential
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


class ResNet50(Model):
    """
    COVID-19 detection model with ResNet50v2 as convolutional base for feature extraction.

    Inputs: batches of images with shape (None, 224, 224, 3).
    Outputs: batches of softmax activations (None, 3). The 3 classes are meant to be: covid-19, normal, pneumonia.
    """

    def __init__(self, name='resnet50', weights='imagenet'):
        super().__init__(name=name)
        self._image_shape = (224, 224, 3)
        self._from_scratch = weights is None

        self._feature_extractor = Sequential([
            Rescaling(1./127.5, offset=-1),
            ResNet50V2(include_top=False, weights=weights, input_shape=self.image_shape)
        ], name='feature_extractor')

        self._classifier = Sequential([
            GlobalAveragePooling2D(),
            Dense(3, activation='softmax')
        ], name='classifier')

        # required for summary()
        inputs = Input(shape=self.image_shape)
        outputs = self.call(inputs)
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.build(input_shape=(None, self.image_shape[0], self.image_shape[1], self.image_shape[2]))

    def call(self, inputs, training=None, mask=None):
        # using Rescaling layer, tf.data.Dataset and tf.keras.Model.fit() causes unknown shape... reshaping fixes
        # see https://gitmemory.com/issue/tensorflow/tensorflow/24520/511633717
        x = tf.reshape(inputs, tf.constant((-1,) + self.image_shape))
        x = self.feature_extractor(x, training=self._from_scratch)      # if pre-trained, BN layers in inference mode
        x = self.classifier(x)
        return x

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def classifier(self):
        return self._classifier

    @property
    def image_shape(self):
        return self._image_shape
