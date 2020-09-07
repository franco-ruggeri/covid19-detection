import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping).

    Supported models: sub-package covid19.models
    Adapted from: https://keras.io/examples/vision/grad_cam/
    """

    def __init__(self, model):
        self.model = model

    def _predict(self, image):
        image = tf.expand_dims(image, axis=0)  # add batch dimension

        with tf.GradientTape() as tape:
            last_conv_activations = self.model.feature_extractor(image)
            tape.watch(last_conv_activations)
            probabilities = self.model.classifier(last_conv_activations)
            prediction = tf.argmax(probabilities[0]).numpy()
            top_probability = probabilities[:, prediction]

        gradients = tape.gradient(top_probability, last_conv_activations)
        return prediction, last_conv_activations, gradients

    @staticmethod
    def _make_heatmap(last_conv_activations, gradients):
        pooled_grads = tf.reduce_mean(gradients, axis=(0, 1, 2))
        last_conv_activations = last_conv_activations.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_activations[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(last_conv_activations, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)              # range [0, 1]
        return heatmap

    def explain(self, image):
        """
        Explains the image by superimposing a heatmap.

        :param image:
        :return: (prediction, explanation), where explanation is the superimposed image
        """
        # predict
        prediction, last_conv_activations, gradients = self._predict(image)

        # compute heatmap
        heatmap = self._make_heatmap(last_conv_activations, gradients)  # range [0, 1]
        heatmap = np.uint8(255 * heatmap)                               # range [0, 255] (for colormap)

        # colorize new image according to heatmap (with colormap 'jet')
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
        jet_heatmap = img_to_array(jet_heatmap)

        # superimposed image
        superimposed_image = jet_heatmap * 0.4 + image
        superimposed_image /= np.max(superimposed_image)    # range [0, 1] (for plt.imshow())
        return prediction, superimposed_image
