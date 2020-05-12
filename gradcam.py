import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model

class GradCAM:

    """Implementation of Grad-CAM using tensorflow and keras"""

    def __init__(self, model, class_index, layer_name=None):

        """

        :param model: The tensorflow model of the network
        :param class_index: The index of the class the network is prediciting, will probably be 0: normal, 1: pneumonia
        2: COVID-19
        :param layer_name: Name of the last convolutional layer

        """

        self.model = model
        self.class_index = class_index
        self.layer_name = layer_name

        # if layer name is not known, try to find the last convolutional layer of the model:
        if layer_name is None:

            self.layer_name = self.find_layer()

    def find_layer(self):
        """

        :param model: The model of the network, assumed to be built with tensorflow.
        :return: The last convolutional layer of the network, to be used for the guided backpropagation gradient.

        """

        # iterate over the layers in reversed order, when a four dimensional output is encountered this is assumed to be
        # the convolutional layer

        for layer in reversed(self.model.layers):

            output_shape = layer.output.shape
            if len(output_shape) == 4:
                return layer.name

        raise ValueError("Could not find a matching convolutional layer, its output shape should be 4 dimensional")

    def generate_heatmap(self, image, eps=1e-8):
        """

        :param image_path: The path to the image that is to be viewed using GradCAM
        :return: A tuple containing the heatmap and output (which is the input image overlayed with the heatmap)

        """

        #image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        #image = tf.keras.preprocessing.image.img_to_array(image)

        grad_model = tf.keras.models.Model(inputs=[self.model.inputs],outputs=[self.model.get_layer(self.layer_name).output,
                                           self.model.output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convolutional_outputs, predictions) = grad_model(inputs)
            loss = predictions[:, self.class_index]

        #output = convolutional_outputs[0]
        gradients = tape.gradient(loss, convolutional_outputs)

        cast_conv_outputs = tf.cast(convolutional_outputs > 0, "float32")
        cast_gradients = tf.cast(gradients > 0, "float32")
        guided_gradients = cast_conv_outputs * cast_gradients * gradients

        convolutional_outputs = convolutional_outputs[0]
        guided_gradients = guided_gradients[0]

        weights = tf.reduce_mean(guided_gradients, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convolutional_outputs), axis=-1)

        width, height = image.shape[2], image.shape[1]
        heatmap = cv2.resize(cam.numpy(), (width, height))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5):

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        output = cv2.addWeighted(image, alpha, heatmap, 1-alpha, 0)

        return heatmap, output








