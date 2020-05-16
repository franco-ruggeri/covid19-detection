from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, ZeroPadding2D, DepthwiseConv2D, Flatten, Dense


class PEPX(Layer):
    """
    Projection-expansion-projection-extension, a residual design pattern used in COVID-Net.

    Original proposal: COVID-Net, A tailored deep convolutional neural network design for detection of COVID-19 cases
     from chest radiography images - Linda Wang, Zhong Qiu Lin and Alexander Wong.
    """

    def __init__(self, channels_in, channels_out, **kwargs):
        super(PEPX, self).__init__(kwargs)
        self.channels_in = channels_in
        self.channels_out = channels_out

        nf_p = channels_in // 2             # number of filters for projections 1 and 2
        nf_e = int(3 / 4 * channels_in)     # number of filters for expansion

        self.p1 = Conv2D(kernel_size=1, filters=nf_p, activation='relu')            # projection 1
        self.e = Conv2D(kernel_size=1, filters=nf_e, activation='relu')             # expansion
        self.pad = ZeroPadding2D(padding=(1, 1))                                    # zero-padding 1 before 3x3 filters
        self.dw = DepthwiseConv2D(kernel_size=3, activation='relu')                 # depth-wise convolution
        self.p2 = Conv2D(kernel_size=1, filters=nf_p, activation='relu')            # projection 2
        self.x = Conv2D(kernel_size=1, filters=channels_out, activation='relu')     # extension

    def call(self, inputs, **kwargs):
        x = self.p1(inputs)
        x = self.e(x)
        x = self.pad(x)
        x = self.dw(x)
        x = self.p2(x)
        x = self.x(x)
        return x

    def get_config(self):
        config = super(PEPX, self).get_config()
        config.update({'channels_in': self.channels_in, 'channels_out': self.channels_out})
        return config


class COVIDNetLayer(Layer):
    """
    Pattern made of PEPX layers + parallel convolutional layer used in COVID-Net.
    This structure is not explicitly mentioned in the paper but it is clearly an higher level pattern in the
    architecture.
    """

    def __init__(self, channels_in, channels_out, n_pepx_layers, **kwargs):
        super(COVIDNetLayer, self).__init__(kwargs)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.n_pepx_layers = n_pepx_layers

        self.pepx = [PEPX(channels_in, channels_out)]
        for n in range(1, n_pepx_layers):                                           # PEPX layers
            self.pepx.append(PEPX(channels_out, channels_out))
        self.conv = Conv2D(kernel_size=1, filters=channels_out, activation='relu')  # parallel convolutional layer
        self.pool = MaxPool2D(pool_size=2)                                          # final pooling layer

    def call(self, inputs, **kwargs):
        # the first PEPX layer and the parallel convolutional layer take as inputs the inputs
        x = self.conv(inputs) + self.pepx[0](inputs)

        # the following layers take also skip connections from past layers and parallel convolutional layer
        for n in range(1, self.n_pepx_layers):
            x += self.pepx[n](x)

        # the next layer will take as input all the skip connections from this structure, so we define a single output
        # that sums the outputs of each layer... this output is x
        # the output is pooled to halve the spatial dimensionality
        return self.pool(x)

    def get_config(self):
        config = super(COVIDNetLayer, self).get_config()
        config.update({'channels_in': self.channels_in, 'channels_out': self.channels_out,
                       'n_pepx_layers': self.n_pepx_layers})
        return config


class COVIDNet(Model):
    """
    Implementation of COVID-Net.

    Original proposal: COVID-Net, A tailored deep convolutional neural network design for detection of COVID-19 cases
     from chest radiography images - Linda Wang, Zhong Qiu Lin and Alexander Wong.
    Inspiration from: https://github.com/IliasPap/COVIDNet
    """

    def __init__(self, input_shape, n_classes, **kwargs):
        super(COVIDNet, self).__init__(kwargs)
        self.input_shape_ = input_shape     # do not use the name 'input_shape' because it is a read-only property
        self.n_classes = n_classes

        self.pad = ZeroPadding2D(padding=(3, 3))
        self.conv0 = Conv2D(kernel_size=7, filters=64, activation='relu')
        self.pool0 = MaxPool2D(pool_size=2)

        self.covidnet_layers = [
            COVIDNetLayer(64, 256, 3),
            COVIDNetLayer(256, 512, 4),
            COVIDNetLayer(512, 1024, 6),
            COVIDNetLayer(1024, 2048, 3)
        ]

        self.flatten = Flatten()
        self.fc1 = Dense(1024, activation='relu')
        self.fc2 = Dense(1024, activation='relu')
        self.classifier = Dense(n_classes)

        # for model.summary()
        self.inputs = Input(self.input_shape_)
        self.outputs = self.call(self.inputs)
        self._init_graph_network(inputs=self.inputs, outputs=self.outputs)

    def call(self, x, **kwargs):
        x = self.pad(x)
        x = self.conv0(x)
        x = self.pool0(x)
        for layer in self.covidnet_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        return x

    def get_config(self):
        config = super(COVIDNet, self).get_config()
        config.update({'input_shape_': self.input_shape_, 'n_classes': self.n_classes})
        return config

    @classmethod
    def from_config(cls, config, custom_object=None):
        input_shape_ = config['input_shape_']
        n_classes = config['n_classes']
        del config['input_shape_']
        del config['n_classes']
        return cls(input_shape_, n_classes, **config)
