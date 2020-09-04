import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras.optimizers import Adam


class Model(tf.keras.Model, ABC):
    """
    Abstract Base Class for COVID-19 detection models.

    The subclassing models have to contain only two major blocks of layers:
    - feature extractor: convolutional base pre-processing inputs and extracting features. It must end with a
        convolutional layer (required by explainers in covid19.explainers).
    - classifier: classification model based on dense layers, on top of the convolutional base.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def _compile_and_fit(self, learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks,
                         class_weights):
        self.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)
        self.summary()
        return self.fit(train_ds, epochs=epochs, initial_epoch=initial_epoch, validation_data=val_ds,
                        callbacks=callbacks, class_weight=class_weights)

    def fit_classifier(self, learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks,
                       class_weights=None):
        """
        Fit classifier (i.e. layers on top of the convolutional base), freezing the convolutional base.

        :param learning_rate: float, learning rate
        :param loss: tf.keras.Loss, loss function for the optimizer
        :param metrics: list of tf.keras.Metric, metrics of interest
        :param train_ds: tf.data.Dataset, training dataset
        :param val_ds: tf.data.Dataset, validation dataset
        :param epochs: int, number of epochs (effective, not counting the initial epochs to skip)
        :param initial_epoch: int, number of initial epochs to skip
        :param callbacks: list of tf.keras.Callback, callbacks for the training loop
        :param class_weights: dictionary (label -> weight), class weights to compensate dataset imbalance.
        :return: History object
        """
        self.feature_extractor.trainable = False
        return self._compile_and_fit(learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks,
                                     class_weights)

    def fine_tune(self, learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks, fine_tune_at,
                  class_weights=None):
        """
        Fine-tune some convolutional layers (i.e. fit classifier + some convolutional layers, freezing the rest).
        The learning rate should be lower than the normal fit.

        :param learning_rate: float, learning rate
        :param loss: tf.keras.Loss, loss function for the optimizer
        :param metrics: list of tf.keras.Metric, metrics of interest
        :param train_ds: tf.data.Dataset, training dataset
        :param val_ds: tf.data.Dataset, validation dataset
        :param epochs: int, number of epochs (effective, not counting the initial epochs to skip)
        :param initial_epoch: int, number of initial epochs to skip
        :param callbacks: list of tf.keras.Callback, callbacks for the training loop
        :param fine_tune_at: int, index of layer at which to start to unfreeze
        :param class_weights: dictionary (label -> weight), class weights to compensate dataset imbalance.
        :return: History object
        """
        self.feature_extractor.trainable = True     # unfreeze convolutional base
        for layer in self.feature_extractor.layers[:fine_tune_at]:
            layer.trainable = False                 # freeze bottom layers
        return self._compile_and_fit(learning_rate, loss, metrics, train_ds, val_ds, epochs+initial_epoch,
                                     initial_epoch, callbacks, class_weights)

    def get_config(self):
        super().get_config()

    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        pass

    @property
    @abstractmethod
    def feature_extractor(self):
        """Convolutional model for feature extraction (e.g. tf.keras.applications.ResNet50). Must end with a
        convolutional layer."""
        pass

    @property
    @abstractmethod
    def classifier(self):
        """Classification model on top of the convolutional base (e.g. Flatten -> Dense)."""
        pass

    @property
    @abstractmethod
    def image_shape(self):
        """Shape of the input images (height, width, channels). For example, (224, 224, 3)."""
        pass
