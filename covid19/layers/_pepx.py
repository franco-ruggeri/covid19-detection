from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D


class PEPXBlock(Layer):
    """
    Design pattern heavily used in COVID-Net, proposed by Linda Wangg et al. (see docs/COVID-Net), consisting of:
    - First-stage Projection: 1x1 convolutions for projecting input features to a lower dimension.
    - Expansion: 1x1 convolutions for expanding features to a higher dimension.
    - Depth-wise Representation: 3x3 depth-wise convolutions for learning efficiently spatial characteristics.
    - Second-stage Projection: 1x1 convolutions for projecting features back to a lower dimension.
    - Extension: 1x1 convolutions that finally extend channel dimensionality to a higher dimension.

    Since the paper does not provide all the details, some choices have been taken according to the state of the art:
    - No batch normalization and no activation.
    - Number of filters in projections set to 1/2 of the initial number of channels.
    - Number of filters in expansion set to 3/4 of the initial number of channels.
    """

    def __init__(self, channels, **kwargs):
        super().__init__(kwargs)
        self.channels = channels

        nf_p = channels // 2            # number of filters for projections 1 and 2
        nf_e = int(3 / 4 * channels)    # number of filters for expansion

        self._projection_1 = Conv2D(nf_p, 1, name='projection_1')
        self._expansion = Conv2D(nf_e, 1, name='expansion')
        self._depth_wise = DepthwiseConv2D(3, padding='same', name='depth_wise')
        self._projection_2 = Conv2D(nf_p, 1, name='projection_2')
        self._extension = Conv2D(channels, 1, name='extension')

    def call(self, inputs, training=None, mask=None):
        x = self._projection_1(inputs, training=training)
        x = self._expansion(x, training=training)
        x = self._depth_wise(x, training=training)
        x = self._projection_2(x, training=training)
        x = self._extension(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels})
        return config
