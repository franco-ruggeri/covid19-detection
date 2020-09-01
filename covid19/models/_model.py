import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras.optimizers import Adam


class Model(tf.keras.Model, ABC):
    """
    Abstract Base Class for COVID-19 detection models.

    It offers two further functions with respect to tf.keras.Model:
    - fit_classifier(): fit classifier on top, freezing the convolutional base
    - fine_tune(): fine-tuning of some layers in the convolutional base

    Subclasses must have the following properties (required by the said functions and by covid19.explainers):
    1) preprocess: preprocess tf.keras.Layer (or function, e.g. tf.keras.applications.resnet_v2.preprocess_input)
    2) feature_extractor: convolutional base (e.g. tf.keras.applications.ResNet50). Must end with a convolutional
        layer, neither pooling layers nor other layers.
    3) classifier: list of tf.keras.Layer after the convolutional base (e.g. GlobalPooling2D, Flatten, Dense)
    4) image_shape: shape of the inputs (e.g. (224,224,3))
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        pass

    @property
    @abstractmethod
    def preprocess(self):
        pass

    @property
    @abstractmethod
    def feature_extractor(self):
        pass

    @property
    @abstractmethod
    def classifier(self):
        pass

    @property
    @abstractmethod
    def image_shape(self):
        pass

    def _compile_and_fit(self, learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks,
                         class_weights):
        self.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)
        self.summary()
        return self.fit(train_ds, epochs=epochs, initial_epoch=initial_epoch, validation_data=val_ds,
                        callbacks=callbacks, class_weight=class_weights)

    def fit_classifier(self, learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks,
                       class_weights=None):
        self.feature_extractor.trainable = False
        return self._compile_and_fit(learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks,
                                     class_weights)

    def fine_tune(self, learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks, fine_tune_at,
                  class_weights=None):
        self.feature_extractor.trainable = True     # unfreeze convolutional base
        for layer in self.feature_extractor.layers[:fine_tune_at]:
            layer.trainable = False                 # freeze bottom layers
        return self._compile_and_fit(learning_rate, loss, metrics, train_ds, val_ds, epochs+initial_epoch,
                                     initial_epoch, callbacks, class_weights)

    def get_config(self):
        super().get_config()
