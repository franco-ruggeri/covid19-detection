from covid19.models._model import Model
from covid19.layers import Rescaling, PEPXBlock
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, MaxPool2D, Flatten, Dense, add


class _COVIDNetBlock(Layer):
    def __init__(self, channels, n_pepx, **kwargs):
        super().__init__(kwargs)
        self.channels = channels
        self.n_pepx = n_pepx

        self._branch_conv = Conv2D(channels, 1)
        self._branch_pepx = []
        for _ in range(n_pepx):
            self._branch_pepx.append(PEPXBlock(channels))

    def call(self, inputs, training=None, mask=None):
        branch_conv_output = self._branch_conv(inputs, training=training)

        x = inputs
        for i in range(len(self._branch_pepx)):
            x = self._branch_pepx[i](x, training=training)
            x = add([x, branch_conv_output])
        return x

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
    - Batch normalization, ReLU activation, and L2 regularization for the fully connected layers.
    - Max pooling 3x3 with stride 2 at the end of each block (where the spatial dimensionality decreases).
    - No activation and batch normalization for the convolutional layers.
    - Inputs rescaled to the range [-1,1].
    """

    def __init__(self, n_classes, name='covidnet', weights=None):
        """
        :param n_classes: int, number of classes (units in the last layer)
        :param name: string, name of the model
        :param weights: path to the pretrained weights to load
        """
        super().__init__(name=name)
        self._image_shape = (224, 224, 3)
        self._n_classes = n_classes
        self._transfer_learning = weights is not None

        self._feature_extractor = Sequential([
            Rescaling(1./127.5, offset=-1),
            Conv2D(64, 7, strides=(2, 2), padding='same'),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
            _COVIDNetBlock(256, 3),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
            _COVIDNetBlock(512, 4),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
            _COVIDNetBlock(1024, 5),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
            _COVIDNetBlock(2048, 3),
        ], name='feature_extractor')

        self._classifier = Sequential([
            Flatten(),
            Dense(1024, kernel_regularizer='l2'),
            BatchNormalization(),
            ReLU(),
            Dense(1024, kernel_regularizer='l2'),
            BatchNormalization(),
            ReLU(),
            Dense(n_classes, kernel_regularizer='l2', activation='softmax')
        ], name='classifier')

        if weights is not None:
            self.load_weights(weights)

        # required for summary()
        inputs = Input(shape=self.image_shape)
        outputs = self.call(inputs)
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.build(input_shape=(None, self.image_shape[0], self.image_shape[1], self.image_shape[2]))

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

    @property
    def transfer_learning(self):
        return self._transfer_learning
