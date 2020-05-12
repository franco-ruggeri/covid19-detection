import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, ZeroPadding2D


class PEPX(Layer):
    def __init__(self, n_input, n_output):
        super(PEPX, self).__init__()
        nf_p = n_input // 2         # number of filters for projections 1 and 2
        nf_e = int(3/4 * n_input)   # number of filters for expansion

        self.p1 = Conv2D(kernel_size=1, filters=nf_p)       # projection 1
        self.e = Conv2D(kernel_size=1, filters=nf_e)        # expansion
        self.pad = ZeroPadding2D(padding=(1, 1))            # zero-padding 1 before 3x3 filters


        # TODO: this is wrong, I should use SeparableConv2D, but check doc for how it works
        self.dw = Conv2D(kernel_size=3, filters=nf_e)       # depth-wise


        self.p2 = Conv2D(kernel_size=1, filters=nf_p)       # projection 2
        self.x = Conv2D(kernel_size=1, filters=n_output)    # extension

    def call(self, inputs):
        x = self.p1(inputs)
        x = self.e(x)
        x = self.pad(x)
        x = self.dw(x)
        x = self.p2(x)
        return self.x(x)
