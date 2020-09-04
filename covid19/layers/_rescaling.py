from tensorflow.keras.layers import Layer


class Rescaling(Layer):
    """
    Pre-processing layer to rescale inputs.

    The behavior is identical to tf.keras.layers.experimental.preprocessing.Rescaling, but the latter does not support
    the offset in tensorflow 2.2.
    """

    def __init__(self, scale, offset=0, **kwargs):
        super().__init__(kwargs)
        self.scale = scale
        self.offset = offset

    def call(self, inputs, training=None, mask=None):
        return inputs * self.scale + self.offset

    def get_config(self):
        config = super().get_config()
        config.update({'scale': self.scale, 'offset': self.offset})
        return config
