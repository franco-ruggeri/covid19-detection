import tensorflow as tf
from covid19.models._model import Model
from covid19.layers import Rescaling, PEPXBlock
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, MaxPool2D, Flatten, Dense, add


class _COVIDNetBlock(Layer):
    def __init__(self, channels, n_pepx, **kwargs):
        super().__init__(kwargs)
        self.channels = channels
        self.n_pepx = n_pepx

        self._branch_conv = Sequential([
            Conv2D(channels, 1),
            BatchNormalization(),
            ReLU()
        ])

        self._branch_pepx = []
        for _ in range(n_pepx):
            self._branch_pepx.append(PEPXBlock(channels))

        self._pooling = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')

    def call(self, inputs, training=None, mask=None):
        branch_conv_output = self._branch_conv(inputs, training=training)

        x = inputs
        for i in range(len(self._branch_pepx)):
            x = self._branch_pepx[i](x, training=training)
            x = add([x, branch_conv_output])

        return self._pooling(x)

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels, 'n_pepx': self.n_pepx})
        return config


class COVIDNet(Model):
    """
    COVID-19 detection model with specific architecture proposed by Linda Wangg et al. (see docs/COVID-Net).

    Inputs: batches of images with shape (None, 224, 224, 3).
    Outputs: batches of softmax activations (None, 3). The 3 classes are meant to be: covid-19, normal, pneumonia.

    Since the paper does not provide all the details, some choices have been taken according to the state of the art:
    - Batch normalization and ReLU activation for every convolutional layer (BN before ReLU).
    - Pooling 3x3 with stride 2 at the end of each block (where the dimensionality decreases in the diagram).
    - Inputs rescaled in the range [-1, 1].
    """

    def __init__(self, n_classes, name='covidnet', weights=None):
        """
        :param n_classes: int, number of classes (units in the last layer)
        :param name: string, name of the model
        :param weights: path to the weights to load
        """
        super().__init__(name=name)
        self._image_shape = (224, 224, 3)
        self._n_classes = n_classes
        self._from_scratch = weights is None

        initial_conv = Sequential([
            Conv2D(64, 7, strides=(2, 2), padding='same'),
            BatchNormalization(),
            ReLU()
        ], name='conv_initial')

        self._feature_extractor = Sequential([
            Rescaling(1./127.5, offset=-1),
            initial_conv,
            _COVIDNetBlock(256, 3),
            _COVIDNetBlock(512, 4),
            _COVIDNetBlock(1024, 5),
            _COVIDNetBlock(2048, 3),
        ], name='feature_extractor')

        self._classifier = Sequential([
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(n_classes, activation='softmax')
        ], name='classifier')

        if weights is not None:
            self.load_weights(weights)

        # required for summary()
        inputs = Input(shape=self.image_shape)
        outputs = self.call(inputs)
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.build(input_shape=(None, self.image_shape[0], self.image_shape[1], self.image_shape[2]))

    def call(self, inputs, training=None, mask=None):
        # remarks:
        # - using Rescaling layer, tf.data.Dataset and tf.keras.Model.fit() causes unknown shape... reshaping fixes
        #   see https://gitmemory.com/issue/tensorflow/tensorflow/24520/511633717
        # - if we are using pre-trained weights (self._from_scratch=True), BN layers must be kept in inference mode
        #   see https://www.tensorflow.org/tutorials/images/transfer_learning
        x = tf.reshape(inputs, tf.constant((-1,) + self.image_shape))
        x = self.feature_extractor(x, training=False if self._from_scratch else training)
        x = self.classifier(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'n_classes': self._n_classes})
        return config

    def fit_linear_classifier(self, learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks,
                              class_weights=None):
        self.feature_extractor.trainable = False
        self.classifier.trainable = True
        for layer in self.classifier.layers[:-1]:
            layer.trainable = False
        return self.compile_and_fit(learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks,
                                    class_weights)

    def fine_tune(self, learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks, fine_tune_at,
                  class_weights=None):
        n_layers_feature_extractor = len(self.feature_extractor.layers) - 1     # -1 for rescaling layer
        n_layers_classifier = len(self.classifier.layers) - 1                   # -1 for linear classifier on top

        if fine_tune_at > n_layers_feature_extractor + n_layers_classifier:
            raise ValueError('Too big fine_tune_at, more than the number of layers')

        if fine_tune_at > n_layers_feature_extractor:
            # freeze all the convolutional base + part of the classifier
            self.feature_extractor.trainable = False
            self.classifier.trainable = True
            fine_tune_at -= n_layers_feature_extractor
            for layer in self.classifier.layers[:fine_tune_at]:
                layer.trainable = False
        else:
            # freeze part of the convolutional base
            self.feature_extractor.trainable = True
            self._classifier.trainable = True
            for layer in self.feature_extractor.layers[:fine_tune_at+1]:        # +1 for rescaling layer
                layer.trainable = False
            for layer in self.classifier.layers:
                layer.trainable = True

        return self.compile_and_fit(learning_rate, loss, metrics, train_ds, val_ds, epochs, initial_epoch, callbacks,
                                    class_weights)

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def classifier(self):
        return self._classifier

    @property
    def image_shape(self):
        return self._image_shape
